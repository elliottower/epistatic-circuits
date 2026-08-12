"""Modal wrapper: resume incomplete RTI coalition sweep.

The initial run timed out with zero at 8192/32768 and mean at 31232/32768.
This script loads the existing checkpoint NPZ, identifies NaN rows
(un-evaluated coalitions), and fills them in.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_rti_coalition_sweep_resume.py
"""

import modal

app = modal.App("rti-coalition-sweep-resume")

results_volume = modal.Volume.from_name("rti-coalition-sweep", create_if_missing=True)

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
def resume_rti_sweep(ablation_type: str):
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

    circuit_heads = RTI_HEADS
    n_players = len(circuit_heads)
    n_total = 2 ** n_players

    out_path = f"/results/rti_{ablation_type}_v2_coalition_values.npz"

    print(f"[{timestamp()}] RTI coalition sweep RESUME: {ablation_type} ablation")

    existing = np.load(out_path)
    all_target_logits = existing["target_logits"].copy()
    all_foil_logits = existing["foil_logits"].copy()
    n_coalitions_completed = int(existing.get("n_coalitions_completed", 0))
    n_prompts_expected = int(existing["n_prompts"])
    intact_target_existing = existing.get("intact_target_logits", None)
    intact_foil_existing = existing.get("intact_foil_logits", None)
    existing.close()

    nan_rows = np.isnan(all_target_logits[:, 0])
    n_remaining = int(nan_rows.sum())
    remaining_indices = np.where(nan_rows)[0]

    print(f"[{timestamp()}] Loaded checkpoint: {n_coalitions_completed}/{n_total} completed, {n_remaining} remaining")
    print(f"[{timestamp()}] Heads: {[f'L{l}H{h}' for l, h in circuit_heads]}")

    if n_remaining == 0:
        print(f"[{timestamp()}] All coalitions already complete!")
        return f"rti/{ablation_type}: already complete"

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded on {model.cfg.device}")

    prompts = generate_prompts("rti", model.tokenizer)
    all_tokens, correct_ids, incorrect_ids = tokenize_all(model, prompts)
    n_prompts = len(prompts)
    assert n_prompts == n_prompts_expected, f"Prompt count mismatch: {n_prompts} vs {n_prompts_expected}"
    print(f"[{timestamp()}] {n_prompts} RTI prompts")

    mean_z = None
    if ablation_type == "mean":
        print(f"[{timestamp()}] Computing mean activations...")
        mean_z = compute_mean_activations(model, all_tokens, batch_size=64)
        torch.cuda.empty_cache()
        gc.collect()

    if intact_target_existing is not None:
        intact_target = intact_target_existing
        intact_foil = intact_foil_existing
    else:
        intact_mask = np.ones(n_players, dtype=bool)
        intact_target, intact_foil = evaluate_coalition_batched(
            model, intact_mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
            ablation_type=ablation_type, mean_z=mean_z, batch_size=64,
        )
    print(f"[{timestamp()}] Intact baseline: mean logit-diff = {(intact_target - intact_foil).mean():.4f}")

    checkpoint_every = 512
    sweep_start = time.time()

    for done_count, idx_pos in enumerate(tqdm(remaining_indices, desc=f"rti-resume/{ablation_type}")):
        coal_idx = int(idx_pos)
        mask = np.array([(coal_idx >> i) & 1 for i in range(n_players)], dtype=bool)

        tgt, foil = evaluate_coalition_batched(
            model, mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
            ablation_type=ablation_type, mean_z=mean_z, batch_size=64,
        )
        all_target_logits[idx_pos] = tgt
        all_foil_logits[idx_pos] = foil

        if (done_count + 1) % checkpoint_every == 0:
            elapsed = time.time() - sweep_start
            rate = (done_count + 1) / elapsed
            remaining_time = (n_remaining - done_count - 1) / rate
            total_done = n_coalitions_completed + done_count + 1
            print(f"[{timestamp()}] {total_done}/{n_total} done ({done_count+1}/{n_remaining} this run), "
                  f"{rate:.2f} coal/s, ~{remaining_time / 3600:.1f}h remaining")
            np.savez(
                out_path,
                target_logits=all_target_logits,
                foil_logits=all_foil_logits,
                coalition_indices=np.arange(n_total),
                circuit_name="rti",
                ablation_type=ablation_type,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                intact_target_logits=intact_target,
                intact_foil_logits=intact_foil,
                n_coalitions_completed=total_done,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    print(f"[{timestamp()}] Resume complete: {n_remaining} new coalitions in {elapsed_total:.0f}s")

    np.savez(
        out_path,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        coalition_indices=np.arange(n_total),
        circuit_name="rti",
        ablation_type=ablation_type,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        intact_target_logits=intact_target,
        intact_foil_logits=intact_foil,
        n_coalitions_completed=n_total,
        elapsed_seconds=elapsed_total,
        sweep_version=2,
    )
    results_volume.commit()
    print(f"[{timestamp()}] Final save committed — {n_total}/{n_total} coalitions complete")
    return f"rti/{ablation_type}: resumed {n_remaining} coalitions, now complete"


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    ablation_types = ["zero", "mean"]
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching RTI coalition sweep RESUME: {ablation_types}")
    handles = []
    for abl in ablation_types:
        handles.append(resume_rti_sweep.spawn(abl))

    for h in handles:
        result = h.get()
        print(f"Completed: {result}")
