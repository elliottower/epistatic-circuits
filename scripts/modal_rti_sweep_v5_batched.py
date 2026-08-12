"""Modal: coalition sweep with BATCHED COALITIONS — 8-10x faster.

Key optimizations vs v3/v4:
  1. Batch N coalitions per forward pass (amortizes Python/hook overhead)
  2. Hook on hook_z instead of hook_result (avoids use_attn_result=True
     which creates a 5D intermediate that's O(batch * seq * heads * d_head * d_model))
  3. Custom forward that skips full-vocab logits (saves 22x on unembed memory)
  4. Persistent hooks with mutable mask tensor (no re-registration)

Memory: 32 coalitions x 302 prompts x 22 tokens ≈ 4GB peak on A10G (24GB).
Speed: ~1h instead of ~9h per (circuit, ablation) pair.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_rti_sweep_v5_batched.py
"""

import modal

app = modal.App("rti-sweep-v5-batched-coalitions")

results_volume = modal.Volume.from_name("rti-sweep-v5", create_if_missing=True)
eap_volume = modal.Volume.from_name("rti-eap-attribution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "typeguard==4.3.0",
        "matplotlib==3.9.4",
    )
    .add_local_file(
        "src/rti_prompts.py",
        remote_path="/app/rti_prompts.py",
    )
)

RANDOM_HEADS = [
    (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
    (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
]

RTI_HEADS = [
    (0, 8), (0, 9), (0, 11),
    (4, 11),
    (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
    (10, 11), (11, 9), (11, 11),
]

N_COAL_BATCH = 32


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume, "/eap": eap_volume},
)
def run_sweep(circuit_name: str, ablation_type: str):
    import gc
    import json
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    import transformer_lens

    from rti_prompts import make_prompts

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    if circuit_name == "C4_random":
        circuit_heads = RANDOM_HEADS
    elif circuit_name == "EAP_rti":
        eap_path = "/eap/eap_rti_head_attribution.json"
        with open(eap_path) as f:
            eap_data = json.load(f)
        circuit_heads = [tuple(h) for h in eap_data["top_15_circuit"]]
        print(f"[{ts()}] Loaded EAP circuit: "
              f"{[f'L{l}H{h}' for l, h in circuit_heads]}")
    elif circuit_name == "RTI":
        circuit_heads = RTI_HEADS
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    n_players = len(circuit_heads)
    n_total = 2 ** n_players

    out_path = f"/results/{circuit_name}_{ablation_type}_rti_v5_coalition_values.npz"

    print(f"[{ts()}] Coalition sweep v5 (batched): {circuit_name} / {ablation_type}")
    print(f"[{ts()}] Heads: {[f'L{l}H{h}' for l, h in circuit_heads]}")
    print(f"[{ts()}] {n_players} heads, {n_total} coalitions, "
          f"batch={N_COAL_BATCH} coalitions/pass")

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    # DO NOT set use_attn_result=True — it creates a 467GB intermediate tensor
    # for large batches. Hook on hook_z instead (before W_O projection).
    model.eval()

    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    print(f"[{ts()}] Model loaded")

    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n_prompts = len(valid)
    print(f"[{ts()}] {n_prompts} valid prompts")

    all_tokens = []
    correct_ids = []
    incorrect_ids = []
    max_len = 0
    for p in valid:
        toks = model.to_tokens(p["text"], prepend_bos=True)
        all_tokens.append(toks)
        correct_ids.append(p["correct_id"])
        incorrect_ids.append(p["distractor_id"])
        max_len = max(max_len, toks.shape[1])

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    incorrect_ids_t = torch.tensor(incorrect_ids, device="cuda")
    last_positions = torch.tensor(seq_lens - 1, device="cuda")

    # Compute mean z activations for mean ablation
    mean_z = None
    if ablation_type == "mean":
        print(f"[{ts()}] Computing mean z activations...")
        z_sums = {}
        total_positions = 0
        with torch.no_grad():
            for start in range(0, n_prompts, 64):
                batch = padded_tokens[start:start + 64]
                _, cache = model.run_with_cache(
                    batch,
                    names_filter=lambda n: "attn.hook_z" in n,
                )
                for l in range(n_layers):
                    act = cache[f"blocks.{l}.attn.hook_z"]
                    if l not in z_sums:
                        z_sums[l] = torch.zeros(
                            n_model_heads, d_head,
                            device="cuda", dtype=act.dtype,
                        )
                    z_sums[l] += act.sum(dim=(0, 1))
                total_positions += batch.shape[0] * batch.shape[1]
                del cache
        mean_z = {l: z_sums[l] / total_positions for l in range(n_layers)}
        torch.cuda.empty_cache()
        gc.collect()

    involved_layers = sorted(set(l for l, h in circuit_heads))

    # Mutable mask tensor: (max_coal_batch * n_prompts, 1, n_heads, 1)
    max_batch_total = N_COAL_BATCH * n_prompts
    layer_mask_tensors = {}
    for l in involved_layers:
        layer_mask_tensors[l] = torch.ones(
            max_batch_total, 1, n_model_heads, 1, device="cuda"
        )

    if ablation_type == "zero":
        def make_hook(layer):
            def hook_fn(act, hook):
                n = act.shape[0]
                return act * layer_mask_tensors[layer][:n]
            return hook_fn
    elif ablation_type == "mean":
        mean_z_expanded = {}
        for l in involved_layers:
            mean_z_expanded[l] = mean_z[l].view(1, 1, n_model_heads, d_head)

        def make_hook(layer):
            def hook_fn(act, hook):
                n = act.shape[0]
                mask = layer_mask_tensors[layer][:n]
                return act * mask + mean_z_expanded[layer] * (1 - mask)
            return hook_fn
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    for l in involved_layers:
        model.add_perma_hook(f"blocks.{l}.attn.hook_z", make_hook(l))

    print(f"[{ts()}] Persistent hooks on hook_z at layers {involved_layers}")

    def build_masks_for_batch(coal_indices):
        n_coal = len(coal_indices)
        for l in involved_layers:
            layer_mask_tensors[l][:n_coal * n_prompts] = 1.0

        for c_pos, coal_idx in enumerate(coal_indices):
            mask_bits = np.array(
                [(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool
            )
            row_start = c_pos * n_prompts
            row_end = row_start + n_prompts
            for i, (layer, head) in enumerate(circuit_heads):
                if not mask_bits[i]:
                    layer_mask_tensors[layer][row_start:row_end, 0, head, 0] = 0.0

    def forward_last_logits_only(tokens):
        """Forward through all blocks, compute logits only at last positions.

        Avoids materializing (batch, seq, vocab_size) logits tensor.
        Permanent hooks fire normally during block forward passes.
        """
        residual = model.hook_embed(model.embed(tokens))
        pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
        residual = residual + pos_embed
        for block in model.blocks:
            residual = block(residual)
        return residual

    def evaluate_batch(coal_indices):
        n_coal = len(coal_indices)
        build_masks_for_batch(coal_indices)

        tiled_tokens = padded_tokens.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)

        with torch.no_grad():
            residual = forward_last_logits_only(tiled_tokens)

            tiled_last = last_positions.repeat(n_coal)
            seq_indices = torch.arange(n_coal * n_prompts, device="cuda")
            last_resid = residual[seq_indices, tiled_last, :]
            del residual

            last_normed = model.ln_final(last_resid)
            del last_resid

            last_logits = last_normed @ model.W_U + model.b_U
            del last_normed

        last_logits = last_logits.view(n_coal, n_prompts, -1)

        tgt_ids = correct_ids_t.unsqueeze(0).expand(n_coal, -1)
        foil_ids = incorrect_ids_t.unsqueeze(0).expand(n_coal, -1)

        target = last_logits.gather(2, tgt_ids.unsqueeze(-1)).squeeze(-1)
        foil = last_logits.gather(2, foil_ids.unsqueeze(-1)).squeeze(-1)

        return target.cpu().numpy().astype(np.float64), foil.cpu().numpy().astype(np.float64)

    # Check for existing checkpoint
    all_target_logits = np.full((n_total, n_prompts), np.nan, dtype=np.float64)
    all_foil_logits = np.full((n_total, n_prompts), np.nan, dtype=np.float64)
    existing_count = 0

    try:
        existing = np.load(out_path)
        old_target = existing["target_logits"]
        old_foil = existing["foil_logits"]
        completed_mask = ~np.isnan(old_target[:, 0])
        existing_count = int(completed_mask.sum())
        if existing_count > 0:
            all_target_logits[completed_mask] = old_target[completed_mask]
            all_foil_logits[completed_mask] = old_foil[completed_mask]
        existing.close()
        print(f"[{ts()}] Resuming: {existing_count}/{n_total} already done")
    except FileNotFoundError:
        pass

    # Baseline
    print(f"[{ts()}] Computing intact-model baseline...")
    intact_tgt, intact_foil = evaluate_batch([n_total - 1])
    baseline_ld = (intact_tgt[0] - intact_foil[0]).mean()
    print(f"[{ts()}] Intact baseline: mean logit-diff = {baseline_ld:.4f}")

    # Build todo list
    todo = [i for i in range(n_total) if np.isnan(all_target_logits[i, 0])]
    n_megabatches = (len(todo) + N_COAL_BATCH - 1) // N_COAL_BATCH
    print(f"[{ts()}] {len(todo)} coalitions remaining, {n_megabatches} mega-batches")

    checkpoint_every = 512
    sweep_start = time.time()
    done_since_start = 0

    for mb_start in tqdm(range(0, len(todo), N_COAL_BATCH),
                         desc=f"{circuit_name}/{ablation_type}",
                         total=n_megabatches):
        mb_end = min(mb_start + N_COAL_BATCH, len(todo))
        coal_indices = todo[mb_start:mb_end]

        try:
            tgt_batch, foil_batch = evaluate_batch(coal_indices)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            half = len(coal_indices) // 2
            print(f"[{ts()}] OOM at batch={len(coal_indices)}, splitting to {half}")
            for sub_start in range(0, len(coal_indices), max(half, 1)):
                sub_end = min(sub_start + max(half, 1), len(coal_indices))
                sub_indices = coal_indices[sub_start:sub_end]
                tgt_sub, foil_sub = evaluate_batch(sub_indices)
                for i, coal_idx in enumerate(sub_indices):
                    all_target_logits[coal_idx] = tgt_sub[i]
                    all_foil_logits[coal_idx] = foil_sub[i]
                done_since_start += len(sub_indices)
            continue

        for i, coal_idx in enumerate(coal_indices):
            all_target_logits[coal_idx] = tgt_batch[i]
            all_foil_logits[coal_idx] = foil_batch[i]

        done_since_start += len(coal_indices)
        total_done = existing_count + done_since_start

        if done_since_start % checkpoint_every < N_COAL_BATCH:
            elapsed = time.time() - sweep_start
            rate = done_since_start / elapsed
            remaining_coal = n_total - total_done
            if rate > 0:
                remaining_s = remaining_coal / rate
                print(f"[{ts()}] {total_done}/{n_total} done, "
                      f"{rate:.1f} coal/s, ~{remaining_s / 60:.1f}min remaining")
            np.savez(
                out_path,
                target_logits=all_target_logits,
                foil_logits=all_foil_logits,
                coalition_indices=np.arange(n_total),
                circuit_name=circuit_name,
                ablation_type=ablation_type,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                intact_target_logits=intact_tgt[0],
                intact_foil_logits=intact_foil[0],
                n_coalitions_completed=total_done,
                sweep_version=5,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    print(f"[{ts()}] Sweep complete: {n_total} coalitions in {elapsed_total:.0f}s "
          f"({elapsed_total/3600:.1f}h)")

    np.savez(
        out_path,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        coalition_indices=np.arange(n_total),
        circuit_name=circuit_name,
        ablation_type=ablation_type,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        intact_target_logits=intact_tgt[0],
        intact_foil_logits=intact_foil[0],
        n_coalitions_completed=n_total,
        elapsed_seconds=elapsed_total,
        sweep_version=5,
    )
    results_volume.commit()
    print(f"[{ts()}] Final save committed")
    return (f"{circuit_name}/{ablation_type}: {n_total} coalitions in "
            f"{elapsed_total/60:.0f}min, baseline LD={baseline_ld:.4f}")


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    circuits = ["EAP_rti", "C4_random", "RTI"]
    ablation_types = ["zero", "mean"]

    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching v5 batched coalition sweeps")
    handles = []
    for circuit in circuits:
        for abl in ablation_types:
            print(f"  Spawning: {circuit} / {abl}")
            handles.append(run_sweep.spawn(circuit, abl))

    for h in handles:
        result = h.get()
        print(f"Completed: {result}")
