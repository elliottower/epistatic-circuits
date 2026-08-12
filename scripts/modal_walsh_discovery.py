"""Modal wrapper: Walsh-based circuit discovery over all 144 GPT-2 heads.

Samples random coalitions of all 144 attention heads (12 layers x 12 heads),
evaluates logit_diff under mean ablation for each coalition, and saves the
results for local LASSO-Walsh fitting. The order-1 Walsh coefficients give
per-head importance; order-2 gives pairwise interactions.

This is a circuit DISCOVERY method — the top-15 heads by order-1 Walsh
magnitude form the "Walsh-discovered circuit" (C5 in the benchmark).

Design:
  - 20K random coalition samples (uniform p=0.5 per head) + all-zeros + all-ones
  - Mean ablation only (consistent with IC discovery and EAP)
  - IOI task (512 prompts, 8 templates x 64 name pairs)
  - LASSO-Walsh fitting done locally after download (no GPU needed)

Usage:
    modal run --detach experiments_batch2/genetics/modal_walsh_discovery.py
"""

import modal

app = modal.App("walsh-circuit-discovery-144heads")

results_volume = modal.Volume.from_name("walsh-discovery", create_if_missing=True)

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
        "experiments_batch2/genetics/coalition_sweep_v2.py",
        remote_path="/app/coalition_sweep_v2.py",
    )
)

ALL_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_PLAYERS = len(ALL_HEADS)
N_SAMPLES = 20_000


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_walsh_sampling(n_samples: int = N_SAMPLES):
    import gc
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    import coalition_sweep_v2
    import transformer_lens

    from coalition_sweep_v2 import (
        compute_mean_activations,
        evaluate_coalition_batched,
        generate_prompts,
        timestamp,
        tokenize_all,
    )

    coalition_sweep_v2.NAMES_A = ["Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate"]
    coalition_sweep_v2.NAMES_B = ["Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia"]

    circuit_heads = ALL_HEADS
    n_players = N_PLAYERS

    print(f"[{timestamp()}] Walsh discovery: {n_players} heads, {n_samples} random samples")

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded on {model.cfg.device}")

    prompts = generate_prompts("ioi", model.tokenizer)
    all_tokens, correct_ids, incorrect_ids = tokenize_all(model, prompts)
    n_prompts = len(prompts)
    print(f"[{timestamp()}] {n_prompts} prompts, token shape {all_tokens.shape}")

    print(f"[{timestamp()}] Computing mean activations...")
    mean_z = compute_mean_activations(model, all_tokens, batch_size=64)
    torch.cuda.empty_cache()
    gc.collect()

    rng = np.random.default_rng(seed=2024)
    random_masks = rng.integers(0, 2, size=(n_samples, n_players)).astype(bool)
    all_zeros = np.zeros((1, n_players), dtype=bool)
    all_ones = np.ones((1, n_players), dtype=bool)
    masks = np.vstack([all_zeros, all_ones, random_masks])
    total = len(masks)
    print(f"[{timestamp()}] {total} coalitions (2 calibration + {n_samples} random)")

    all_target_logits = np.full((total, n_prompts), np.nan, dtype=np.float64)
    all_foil_logits = np.full((total, n_prompts), np.nan, dtype=np.float64)

    out_path = "/results/walsh_discovery_144heads_mean_coalitions.npz"
    checkpoint_every = 500
    sweep_start = time.time()

    for idx in tqdm(range(total), desc="walsh-discovery"):
        tgt, foil = evaluate_coalition_batched(
            model, masks[idx], circuit_heads, all_tokens, correct_ids, incorrect_ids,
            ablation_type="mean", mean_z=mean_z, batch_size=64,
        )
        all_target_logits[idx] = tgt
        all_foil_logits[idx] = foil

        if (idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - sweep_start
            rate = (idx + 1) / elapsed
            remaining = (total - idx - 1) / rate
            print(f"[{timestamp()}] {idx + 1}/{total} done, "
                  f"{rate:.2f} coal/s, ~{remaining / 3600:.1f}h remaining")
            np.savez(
                out_path,
                masks=masks[:idx + 1],
                target_logits=all_target_logits[:idx + 1],
                foil_logits=all_foil_logits[:idx + 1],
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                n_completed=idx + 1,
                n_total=total,
                seed=2024,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    print(f"[{timestamp()}] Complete: {total} coalitions in {elapsed_total / 3600:.1f}h "
          f"({total / elapsed_total:.2f} coal/s)")

    logit_diffs = all_target_logits - all_foil_logits
    mean_logit_diffs = logit_diffs.mean(axis=1)

    print(f"[{timestamp()}] All-ablated logit-diff:  {mean_logit_diffs[0]:.4f}")
    print(f"[{timestamp()}] Clean-model logit-diff:  {mean_logit_diffs[1]:.4f}")
    print(f"[{timestamp()}] Random samples mean:     {mean_logit_diffs[2:].mean():.4f} "
          f"(std {mean_logit_diffs[2:].std():.4f})")

    np.savez(
        out_path,
        masks=masks,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        mean_logit_diffs=mean_logit_diffs,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        n_completed=total,
        n_total=total,
        seed=2024,
        elapsed_seconds=elapsed_total,
        prompt_texts=np.array([p["text"] for p in prompts]),
    )
    results_volume.commit()
    print(f"[{timestamp()}] Final save committed")

    return f"Walsh discovery: {total} coalitions in {elapsed_total / 3600:.1f}h"


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching Walsh discovery sweep")
    print(f"  {N_PLAYERS} heads, {N_SAMPLES} random samples + 2 calibration")
    print(f"  Estimated runtime: ~7h on A10G")

    result = run_walsh_sampling.remote(N_SAMPLES)
    print(f"\nCompleted: {result}")
    print(f"\nDownload with:")
    print(f"  modal volume get walsh-discovery walsh_discovery_144heads_mean_coalitions.npz "
          f"experiments_batch2/genetics/")
