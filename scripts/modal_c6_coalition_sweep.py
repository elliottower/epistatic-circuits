"""Modal wrapper: C6 (max-epistasis Walsh) coalition sweep (zero + mean ablation).

C6 selects the top 15 heads by total order-2 Walsh energy — the heads that
participate in the most/strongest pairwise interactions, regardless of their
individual (order-1) importance. Same LASSO fit as C5, different selection
criterion.

Runs the full 2^15 coalition sweep under both zero and mean ablation.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_c6_coalition_sweep.py
"""

import modal

app = modal.App("c6-max-epistasis-coalition-sweep")

results_volume = modal.Volume.from_name("c6-coalition-sweep", create_if_missing=True)

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
)

C6_HEADS = [
    (5, 5), (8, 6), (11, 10), (10, 7), (6, 9),
    (0, 10), (0, 1), (10, 0), (5, 9), (8, 10),
    (11, 2), (9, 9), (0, 9), (7, 9), (4, 0),
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_c6_sweep(ablation_type: str):
    import gc
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    import coalition_sweep as coalition_sweep_v2
    import transformer_lens

    from coalition_sweep import (
        compute_mean_activations,
        evaluate_coalition_batched,
        generate_prompts,
        timestamp,
        tokenize_all,
    )

    coalition_sweep_v2.NAMES_A = ["Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate"]
    coalition_sweep_v2.NAMES_B = ["Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia"]

    circuit_heads = C6_HEADS
    n_players = len(circuit_heads)
    n_total = 2 ** n_players

    out_path = f"/results/c6_{ablation_type}_v2_coalition_values.npz"

    print(f"[{timestamp()}] C6 max-epistasis coalition sweep: {ablation_type} ablation")
    print(f"[{timestamp()}] Heads: {[f'L{l}H{h}' for l, h in circuit_heads]}")
    print(f"[{timestamp()}] {n_players} heads, {n_total} coalitions")

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded on {model.cfg.device}")

    prompts = generate_prompts("ioi", model.tokenizer)
    all_tokens, correct_ids, incorrect_ids = tokenize_all(model, prompts)
    n_prompts = len(prompts)
    print(f"[{timestamp()}] {n_prompts} prompts, token shape {all_tokens.shape}")

    mean_z = None
    if ablation_type == "mean":
        print(f"[{timestamp()}] Computing mean activations...")
        mean_z = compute_mean_activations(model, all_tokens, batch_size=64)
        torch.cuda.empty_cache()
        gc.collect()

    coalition_indices = np.arange(n_total)
    all_target_logits = np.full((n_total, n_prompts), np.nan, dtype=np.float64)
    all_foil_logits = np.full((n_total, n_prompts), np.nan, dtype=np.float64)

    print(f"[{timestamp()}] Computing intact-model baseline...")
    intact_mask = np.ones(n_players, dtype=bool)
    intact_target, intact_foil = evaluate_coalition_batched(
        model, intact_mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
        ablation_type=ablation_type, mean_z=mean_z, batch_size=64,
    )
    print(f"[{timestamp()}] Intact baseline: mean logit-diff = {(intact_target - intact_foil).mean():.4f}")

    checkpoint_every = 512
    sweep_start = time.time()

    for idx_pos in tqdm(range(n_total), desc=f"c6/{ablation_type}"):
        coal_idx = coalition_indices[idx_pos]
        mask = np.array([(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)

        tgt, foil = evaluate_coalition_batched(
            model, mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
            ablation_type=ablation_type, mean_z=mean_z, batch_size=64,
        )
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
                circuit_name="c6",
                ablation_type=ablation_type,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                intact_target_logits=intact_target,
                intact_foil_logits=intact_foil,
                n_coalitions_completed=idx_pos + 1,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    print(f"[{timestamp()}] Sweep complete: {n_total} coalitions in {elapsed_total:.0f}s")

    np.savez(
        out_path,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        coalition_indices=coalition_indices,
        circuit_name="c6",
        ablation_type=ablation_type,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        intact_target_logits=intact_target,
        intact_foil_logits=intact_foil,
        n_coalitions_completed=n_total,
        elapsed_seconds=elapsed_total,
        prompt_texts=np.array([p["text"] for p in prompts]),
        sweep_version=2,
    )
    results_volume.commit()
    print(f"[{timestamp()}] Final save committed")
    return f"c6/{ablation_type}: {n_total} coalitions complete"


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    ablation_types = ["zero", "mean"]
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching C6 coalition sweeps: {ablation_types}")
    handles = []
    for abl in ablation_types:
        handles.append(run_c6_sweep.spawn(abl))

    for h in handles:
        result = h.get()
        print(f"Completed: {result}")

    print(f"\nDownload results:")
    for abl in ablation_types:
        print(f"  modal volume get c6-coalition-sweep c6_{abl}_v2_coalition_values.npz data/")
