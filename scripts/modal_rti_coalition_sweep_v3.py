"""Modal wrapper: RTI circuit coalition sweep with CORRECT prompts (v3).

v2 used only IOI-like name templates (8 templates, wrong prompts).
v3 uses the full 12-category RTI prompt set from the RTI paper's
run_rti_task.py (302 prompts across name repetition, common nouns,
adjectives, locations, temporal sequences, pronouns, list completion,
French names/nouns, counting, BPE fragments, BOS context).

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_rti_coalition_sweep_v3.py
"""

import modal

app = modal.App("rti-coalition-sweep-v3-correct-prompts")

results_volume = modal.Volume.from_name("rti-coalition-sweep-v3", create_if_missing=True)

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
        "src/coalition_sweep.py",
        remote_path="/app/coalition_sweep.py",
    )
    .add_local_file(
        "src/rti_prompts.py",
        remote_path="/app/rti_prompts.py",
    )
)

RTI_HEADS = [
    (0, 8), (0, 9), (0, 11),
    (4, 11),
    (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
    (10, 11), (11, 9), (11, 11),
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_rti_sweep(ablation_type: str):
    import gc
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    import transformer_lens

    from coalition_sweep import (
        compute_mean_activations,
        timestamp,
    )
    from rti_prompts import make_prompts

    circuit_heads = RTI_HEADS
    n_players = len(circuit_heads)
    n_total = 2 ** n_players

    out_path = f"/results/rti_{ablation_type}_v3_coalition_values.npz"

    print(f"[{timestamp()}] RTI v3 coalition sweep: {ablation_type} ablation")
    print(f"[{timestamp()}] Using full 12-category RTI prompts (paper-matched)")
    print(f"[{timestamp()}] Heads: {[f'L{l}H{h}' for l, h in circuit_heads]}")
    print(f"[{timestamp()}] {n_players} heads, {n_total} coalitions")

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded on {model.cfg.device}")

    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n_prompts = len(valid)

    categories = sorted(set(p["category"] for p in valid))
    print(f"[{timestamp()}] {n_prompts} valid prompts across {len(categories)} categories:")
    for cat in categories:
        n = sum(1 for p in valid if p["category"] == cat)
        print(f"  {cat}: {n}")

    all_tokens = []
    correct_ids = []
    incorrect_ids = []
    prompt_categories = []
    max_len = 0
    for p in valid:
        toks = model.to_tokens(p["text"], prepend_bos=True)
        all_tokens.append(toks)
        correct_ids.append(p["correct_id"])
        incorrect_ids.append(p["distractor_id"])
        prompt_categories.append(p["category"])
        max_len = max(max_len, toks.shape[1])

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = []
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens.append(toks.shape[1])

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    incorrect_ids_t = torch.tensor(incorrect_ids, device="cuda")
    seq_lens_arr = np.array(seq_lens)

    print(f"[{timestamp()}] Padded token shape: {padded_tokens.shape}, max_len={max_len}")

    def evaluate_varlen(coalition_mask, batch_size=32):
        """Evaluate coalition with variable-length prompts (index by seq_len)."""
        from collections import defaultdict
        heads_to_ablate_by_layer = defaultdict(list)
        for i, (layer, head) in enumerate(circuit_heads):
            if not coalition_mask[i]:
                heads_to_ablate_by_layer[layer].append(head)

        target_out = np.zeros(n_prompts, dtype=np.float64)
        foil_out = np.zeros(n_prompts, dtype=np.float64)

        for start in range(0, n_prompts, batch_size):
            end = min(start + batch_size, n_prompts)
            batch_tokens = padded_tokens[start:end]
            batch_seq_lens = seq_lens_arr[start:end]

            hooks = []
            for layer, head_indices in heads_to_ablate_by_layer.items():
                if ablation_type == "zero":
                    def make_hook(hidxs):
                        def hook_fn(act, hook):
                            for h in hidxs:
                                act[:, :, h, :] = 0.0
                            return act
                        return hook_fn
                elif ablation_type == "mean":
                    mz = mean_z[layer]
                    def make_hook(hidxs, mz_=mz):
                        def hook_fn(act, hook):
                            for h in hidxs:
                                act[:, :, h, :] = mz_[h]
                            return act
                        return hook_fn
                else:
                    raise ValueError(f"Unknown ablation type: {ablation_type}")
                hooks.append((f"blocks.{layer}.attn.hook_result", make_hook(head_indices)))

            with torch.no_grad():
                logits = model.run_with_hooks(batch_tokens, fwd_hooks=hooks)
                for i_in_batch in range(end - start):
                    last_pos = batch_seq_lens[i_in_batch] - 1
                    last_logits = logits[i_in_batch, last_pos, :]
                    global_idx = start + i_in_batch
                    target_out[global_idx] = last_logits[correct_ids_t[global_idx]].item()
                    foil_out[global_idx] = last_logits[incorrect_ids_t[global_idx]].item()

        return target_out, foil_out

    mean_z = None
    if ablation_type == "mean":
        print(f"[{timestamp()}] Computing mean activations...")
        mean_z = compute_mean_activations(model, padded_tokens, batch_size=32)
        torch.cuda.empty_cache()
        gc.collect()

    coalition_indices = np.arange(n_total)
    all_target_logits = np.full((n_total, n_prompts), np.nan, dtype=np.float64)
    all_foil_logits = np.full((n_total, n_prompts), np.nan, dtype=np.float64)

    print(f"[{timestamp()}] Computing intact-model baseline...")
    intact_mask = np.ones(n_players, dtype=bool)
    intact_target, intact_foil = evaluate_varlen(intact_mask, batch_size=32)
    baseline_ld = (intact_target - intact_foil).mean()
    print(f"[{timestamp()}] Intact baseline: mean logit-diff = {baseline_ld:.4f}")

    checkpoint_every = 512
    sweep_start = time.time()

    for idx_pos in tqdm(range(n_total), desc=f"rti-v3/{ablation_type}"):
        coal_idx = coalition_indices[idx_pos]
        mask = np.array([(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)

        tgt, foil = evaluate_varlen(mask, batch_size=32)
        all_target_logits[idx_pos] = tgt
        all_foil_logits[idx_pos] = foil

        if (idx_pos + 1) % checkpoint_every == 0:
            elapsed = time.time() - sweep_start
            rate = (idx_pos + 1) / elapsed
            remaining = (n_total - idx_pos - 1) / rate
            print(f"[{timestamp()}] {idx_pos + 1}/{n_total} done, "
                  f"{rate:.2f} coal/s, ~{remaining / 3600:.1f}h remaining")
            np.savez(
                out_path,
                target_logits=all_target_logits,
                foil_logits=all_foil_logits,
                coalition_indices=coalition_indices,
                circuit_name="rti_v3",
                ablation_type=ablation_type,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                intact_target_logits=intact_target,
                intact_foil_logits=intact_foil,
                n_coalitions_completed=idx_pos + 1,
                prompt_categories=np.array(prompt_categories),
                sweep_version=3,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    print(f"[{timestamp()}] Sweep complete: {n_total} coalitions in {elapsed_total:.0f}s")

    np.savez(
        out_path,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        coalition_indices=coalition_indices,
        circuit_name="rti_v3",
        ablation_type=ablation_type,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        intact_target_logits=intact_target,
        intact_foil_logits=intact_foil,
        n_coalitions_completed=n_total,
        elapsed_seconds=elapsed_total,
        prompt_categories=np.array(prompt_categories),
        prompt_texts=np.array([p["text"] for p in valid]),
        sweep_version=3,
    )
    results_volume.commit()
    print(f"[{timestamp()}] Final save committed")
    return f"rti-v3/{ablation_type}: {n_total} coalitions, {n_prompts} prompts, baseline LD={baseline_ld:.4f}"


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    ablation_types = ["zero", "mean"]
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching RTI v3 coalition sweeps "
          f"(correct prompts): {ablation_types}")
    handles = []
    for abl in ablation_types:
        handles.append(run_rti_sweep.spawn(abl))

    for h in handles:
        result = h.get()
        print(f"Completed: {result}")
