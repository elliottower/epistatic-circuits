"""Modal: Coalition sweep for RTI Walsh-discovered circuits (C5, C6).

2^15 = 32768 coalitions per circuit, mean + zero ablation, logit_diff metric.
With checkpointing every 2048 coalitions for resume.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_rti_walsh_circuits_sweep.py
"""

import modal

app = modal.App("rti-walsh-circuits-sweep")

results_volume = modal.Volume.from_name("rti-walsh-circuits-sweep", create_if_missing=True)

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
    .add_local_file("src/rti_prompts.py", remote_path="/app/rti_prompts.py")
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
)

# From LASSO fit on 20K random coalitions
RTI_C5_HEADS = [
    (0, 9), (11, 2), (4, 11), (10, 6), (7, 9), (10, 7), (9, 9),
    (5, 6), (4, 0), (2, 11), (8, 7), (1, 5), (11, 10), (1, 3), (6, 11),
]
RTI_C6_HEADS = [
    (0, 10), (0, 9), (11, 2), (10, 7), (10, 0), (4, 11), (11, 10),
    (0, 1), (0, 3), (9, 9), (1, 11), (9, 6), (2, 2), (5, 6), (2, 11),
]

N_COAL_BATCH = 32
CHECKPOINT_EVERY = 2048


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_sweep(circuit_name: str, ablation_type: str):
    import gc
    import os
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")
    from rti_prompts import make_prompts

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    if circuit_name == "c5_walsh":
        circuit_heads = RTI_C5_HEADS
    elif circuit_name == "c6_epistatic":
        circuit_heads = RTI_C6_HEADS
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    n_players = len(circuit_heads)
    n_total = 2 ** n_players
    head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
    involved_layers = sorted(set(l for l, h in circuit_heads))

    out_path = f"/results/rti_{circuit_name}_{ablation_type}_coalition_values.npz"

    start_idx = 0
    existing_values = None
    if os.path.exists(out_path):
        ckpt = np.load(out_path)
        if "n_completed" in ckpt and int(ckpt["n_completed"]) < n_total:
            start_idx = int(ckpt["n_completed"])
            existing_values = ckpt["logit_diff"]
            print(f"[{ts()}] Resuming from coalition {start_idx}")
        elif "n_completed" in ckpt and int(ckpt["n_completed"]) == n_total:
            print(f"[{ts()}] Already complete, skipping")
            return

    print(f"[{ts()}] RTI coalition sweep: {circuit_name} / {ablation_type}")
    print(f"[{ts()}] Heads: {head_labels}")
    print(f"[{ts()}] {n_players} heads, {n_total} coalitions, batch={N_COAL_BATCH}")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n_prompts = len(valid)
    print(f"[{ts()}] {n_prompts} valid prompts")

    all_tokens = []
    correct_ids = []
    distractor_ids = []
    max_len = 0
    for p in valid:
        toks = model.to_tokens(p["text"], prepend_bos=True)
        all_tokens.append(toks)
        correct_ids.append(p["correct_id"])
        distractor_ids.append(p["distractor_id"])
        max_len = max(max_len, toks.shape[1])

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    distractor_ids_t = torch.tensor(distractor_ids, device="cuda")
    last_positions = torch.tensor(seq_lens - 1, device="cuda")

    print(f"[{ts()}] Computing mean z...")
    z_sums = {}
    total_pos = 0
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_tokens[start:start + 64]
            _, cache = model.run_with_cache(batch, names_filter=lambda n: "attn.hook_z" in n)
            for l in range(n_layers):
                act = cache[f"blocks.{l}.attn.hook_z"]
                if l not in z_sums:
                    z_sums[l] = torch.zeros(n_model_heads, d_head, device="cuda", dtype=act.dtype)
                z_sums[l] += act.sum(dim=(0, 1))
            total_pos += batch.shape[0] * batch.shape[1]
            del cache
    mean_z = {l: z_sums[l] / total_pos for l in range(n_layers)}
    torch.cuda.empty_cache()
    gc.collect()

    max_batch_total = N_COAL_BATCH * n_prompts
    layer_mask_tensors = {}
    mean_z_expanded = {}
    for l in involved_layers:
        layer_mask_tensors[l] = torch.ones(max_batch_total, 1, n_model_heads, 1, device="cuda")
        mean_z_expanded[l] = mean_z[l].view(1, 1, n_model_heads, d_head)

    model.reset_hooks(including_permanent=True)

    if ablation_type == "zero":
        def make_hook(layer):
            def hook_fn(act, hook):
                n = act.shape[0]
                return act * layer_mask_tensors[layer][:n]
            return hook_fn
    else:
        def make_hook(layer):
            def hook_fn(act, hook):
                n = act.shape[0]
                mask = layer_mask_tensors[layer][:n]
                return act * mask + mean_z_expanded[layer] * (1 - mask)
            return hook_fn

    for l in involved_layers:
        model.add_perma_hook(f"blocks.{l}.attn.hook_z", make_hook(l))

    def build_masks(coal_indices):
        n_coal = len(coal_indices)
        for l in involved_layers:
            layer_mask_tensors[l][:n_coal * n_prompts] = 1.0
        for c_pos, coal_idx in enumerate(coal_indices):
            mask_bits = np.array([(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)
            row_start = c_pos * n_prompts
            row_end = row_start + n_prompts
            for i, (layer, head) in enumerate(circuit_heads):
                if not mask_bits[i]:
                    layer_mask_tensors[layer][row_start:row_end, 0, head, 0] = 0.0

    def forward_last_logits(tokens, n_coal):
        residual = model.hook_embed(model.embed(tokens))
        pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
        residual = residual + pos_embed
        for block in model.blocks:
            residual = block(residual)
        tiled_last = last_positions.repeat(n_coal)
        seq_idx = torch.arange(tokens.shape[0], device="cuda")
        last_resid = residual[seq_idx, tiled_last, :]
        del residual
        last_normed = model.ln_final(last_resid)
        del last_resid
        logits = last_normed @ model.W_U + model.b_U
        del last_normed
        return logits

    def evaluate_batch(coal_indices):
        n_coal = len(coal_indices)
        build_masks(coal_indices)
        tiled_tokens = padded_tokens.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)
        with torch.no_grad():
            last_logits = forward_last_logits(tiled_tokens, n_coal)
        last_logits = last_logits.view(n_coal, n_prompts, -1)
        seq_idx = torch.arange(n_prompts, device="cuda")
        correct_logits = last_logits[:, seq_idx, correct_ids_t]
        distractor_logits = last_logits[:, seq_idx, distractor_ids_t]
        ld = (correct_logits - distractor_logits).cpu().numpy().astype(np.float64)
        del last_logits
        return ld

    ld_values = np.zeros((n_total, n_prompts), dtype=np.float64)
    if existing_values is not None:
        ld_values[:start_idx] = existing_values[:start_idx]

    t0 = time.time()
    for mb_start in tqdm(range(start_idx, n_total, N_COAL_BATCH), desc=f"{circuit_name}_{ablation_type}"):
        mb_end = min(mb_start + N_COAL_BATCH, n_total)
        coal_indices = list(range(mb_start, mb_end))
        ld_batch = evaluate_batch(coal_indices)
        for i, coal_idx in enumerate(coal_indices):
            ld_values[coal_idx] = ld_batch[i]

        if (mb_end % CHECKPOINT_EVERY == 0) or (mb_end == n_total):
            np.savez(
                out_path,
                logit_diff=ld_values[:mb_end],
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                n_completed=mb_end,
                n_total=n_total,
            )
            results_volume.commit()
            elapsed = time.time() - t0
            done = mb_end - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_total - mb_end) / rate if rate > 0 else 0
            print(f"[{ts()}] {mb_end}/{n_total}, {rate:.1f} coal/s, ~{remaining/60:.0f}min left")

    elapsed_total = time.time() - t0
    print(f"[{ts()}] Done in {elapsed_total/60:.1f}min")

    model.reset_hooks(including_permanent=True)


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=3600,
)
def analyze_results():
    import json
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/app")
    from walsh import energy_by_order_normalized, wht

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    all_analysis = {}

    for circuit_name in ["c5_walsh", "c6_epistatic"]:
        if circuit_name == "c5_walsh":
            circuit_heads = RTI_C5_HEADS
        else:
            circuit_heads = RTI_C6_HEADS
        n_players = len(circuit_heads)
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]

        for ablation_type in ["mean", "zero"]:
            path = f"/results/rti_{circuit_name}_{ablation_type}_coalition_values.npz"
            data = np.load(path)
            ld_values = data["logit_diff"]
            n_prompts = int(data["n_prompts"])

            mean_ld = ld_values.mean(axis=1)
            full_mask = (1 << n_players) - 1
            intact_ld = float(mean_ld[full_mask])
            empty_ld = float(mean_ld[0])
            faithfulness = intact_ld - empty_ld

            wht_coeffs = wht(mean_ld.astype(np.float64))
            energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)
            energy_nc = energy_spectrum[1:].copy()
            total_nc = energy_nc.sum()
            energy_nc_frac = energy_nc / total_nc if total_nc > 0 else energy_nc

            per_prompt_group = ld_values[full_mask] - ld_values[0]
            per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
            for i in range(n_players):
                complement = full_mask ^ (1 << i)
                per_prompt_loo_sum += ld_values[full_mask] - ld_values[complement]
            group_mean = float(np.mean(per_prompt_group))
            loo_sum_mean = float(np.mean(per_prompt_loo_sum))
            epistasis = 1.0 - loo_sum_mean / group_mean if abs(group_mean) > 1e-10 else 0.0

            key = f"{circuit_name}_{ablation_type}"
            print(f"\n  {key}:")
            print(f"    Faithfulness: {faithfulness:+.4f}")
            print(f"    Order-1: {float(energy_nc_frac[0])*100:.1f}%, "
                  f"Order-2: {float(energy_nc_frac[1])*100:.1f}%")
            print(f"    Epistasis: {epistasis*100:.1f}%")

            all_analysis[key] = {
                "circuit": circuit_name, "ablation": ablation_type, "task": "RTI",
                "n_players": n_players, "n_prompts": n_prompts, "heads": head_labels,
                "faithfulness": round(faithfulness, 6),
                "order1_frac": round(float(energy_nc_frac[0]), 4),
                "order2_frac": round(float(energy_nc_frac[1]), 4),
                "order3plus_frac": round(float(sum(energy_nc_frac[2:])), 4),
                "epistasis": round(epistasis, 4),
            }

    with open("/results/rti_walsh_circuits_analysis.json", "w") as f:
        json.dump(all_analysis, f, indent=2)
    results_volume.commit()
    print(f"\n[{ts()}] Analysis saved")


@app.local_entrypoint()
def main():
    for circuit_name in ["c5_walsh", "c6_epistatic"]:
        for ablation_type in ["mean", "zero"]:
            print(f"\n{'='*60}")
            print(f"Running: {circuit_name} / {ablation_type}")
            run_sweep.remote(circuit_name, ablation_type)

    print("\nAll sweeps done. Running analysis...")
    analyze_results.remote()
