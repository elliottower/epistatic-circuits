"""Modal: coalition sweep for EAP + random circuits on RTI prompts (v3).

Optimized over v3 single-circuit sweep:
  1. Persistent hooks with mutable mask tensor (no re-registration per coalition)
  2. Full-batch evaluation (all ~302 prompts at once instead of batches of 32)
  3. Parallel shards across containers

Each (circuit, ablation_type) pair runs on its own container.

Usage:
    cd epistatic-circuits
    # First run EAP attribution to get the circuit:
    modal run scripts/modal_eap_rti_attribution.py
    # Then update EAP_RTI_HEADS below with the result, and run:
    modal run --detach scripts/modal_rti_multi_circuit_sweep.py
"""

import modal

app = modal.App("rti-multi-circuit-sweep-v4")

results_volume = modal.Volume.from_name("rti-multi-circuit-sweep-v4", create_if_missing=True)
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
        "src/coalition_sweep.py",
        remote_path="/app/coalition_sweep.py",
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
    from collections import defaultdict

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    import transformer_lens

    from coalition_sweep import compute_mean_activations, timestamp
    from rti_prompts import make_prompts

    if circuit_name == "C4_random":
        circuit_heads = RANDOM_HEADS
    elif circuit_name == "EAP_rti":
        eap_path = "/eap/eap_rti_head_attribution.json"
        with open(eap_path) as f:
            eap_data = json.load(f)
        circuit_heads = [tuple(h) for h in eap_data["top_15_circuit"]]
        print(f"[{timestamp()}] Loaded EAP circuit from {eap_path}: "
              f"{[f'L{l}H{h}' for l, h in circuit_heads]}")
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    n_players = len(circuit_heads)
    n_total = 2 ** n_players

    out_path = f"/results/{circuit_name}_{ablation_type}_rti_coalition_values.npz"

    print(f"[{timestamp()}] RTI coalition sweep: {circuit_name} / {ablation_type}")
    print(f"[{timestamp()}] Heads: {[f'L{l}H{h}' for l, h in circuit_heads]}")
    print(f"[{timestamp()}] {n_players} heads, {n_total} coalitions")

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded")

    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n_prompts = len(valid)
    print(f"[{timestamp()}] {n_prompts} valid prompts")

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
    batch_indices = torch.arange(n_prompts, device="cuda")

    mean_z = None
    if ablation_type == "mean":
        print(f"[{timestamp()}] Computing mean activations...")
        mean_z = compute_mean_activations(model, padded_tokens, batch_size=64)
        torch.cuda.empty_cache()
        gc.collect()

    head_to_player = {}
    for i, (layer, head) in enumerate(circuit_heads):
        head_to_player[(layer, head)] = i

    layer_masks = {}
    n_model_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    for l in range(model.cfg.n_layers):
        layer_masks[l] = torch.ones(n_model_heads, device="cuda")

    if ablation_type == "zero":
        def make_perma_hook(layer):
            def hook_fn(act, hook):
                mask = layer_masks[layer].view(1, 1, n_model_heads, 1)
                return act * mask
            return hook_fn
    elif ablation_type == "mean":
        mean_expanded = {}
        for l in range(model.cfg.n_layers):
            mean_expanded[l] = mean_z[l].unsqueeze(0).unsqueeze(0)

        def make_perma_hook(layer):
            def hook_fn(act, hook):
                mask = layer_masks[layer].view(1, 1, n_model_heads, 1)
                return act * mask + mean_expanded[layer] * (1 - mask)
            return hook_fn
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    involved_layers = set(l for l, h in circuit_heads)
    for l in involved_layers:
        model.add_perma_hook(f"blocks.{l}.attn.hook_result", make_perma_hook(l))

    print(f"[{timestamp()}] Persistent hooks registered on {len(involved_layers)} layers")

    def update_masks(coalition_mask):
        for l in involved_layers:
            layer_masks[l][:] = 1.0
        for i, (layer, head) in enumerate(circuit_heads):
            if not coalition_mask[i]:
                layer_masks[layer][head] = 0.0

    def evaluate_all_prompts():
        with torch.no_grad():
            logits = model(padded_tokens)
            last_logits = logits[batch_indices, last_positions, :]
            target = last_logits[batch_indices, correct_ids_t].cpu().numpy().astype(np.float64)
            foil = last_logits[batch_indices, incorrect_ids_t].cpu().numpy().astype(np.float64)
        return target, foil

    coalition_indices = np.arange(n_total)
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
        print(f"[{timestamp()}] Resuming: {existing_count}/{n_total} already done")
    except FileNotFoundError:
        pass

    print(f"[{timestamp()}] Computing intact-model baseline...")
    intact_mask = np.ones(n_players, dtype=bool)
    update_masks(intact_mask)
    intact_target, intact_foil = evaluate_all_prompts()
    baseline_ld = (intact_target - intact_foil).mean()
    print(f"[{timestamp()}] Intact baseline: mean logit-diff = {baseline_ld:.4f}")

    checkpoint_every = 512
    sweep_start = time.time()
    done = existing_count

    for idx_pos in tqdm(range(n_total), desc=f"{circuit_name}/{ablation_type}"):
        if not np.isnan(all_target_logits[idx_pos, 0]):
            continue

        coal_idx = coalition_indices[idx_pos]
        mask = np.array([(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)

        update_masks(mask)
        tgt, foil = evaluate_all_prompts()
        all_target_logits[idx_pos] = tgt
        all_foil_logits[idx_pos] = foil
        done += 1

        if done % checkpoint_every == 0:
            elapsed = time.time() - sweep_start
            new_done = done - existing_count
            if new_done > 0:
                rate = new_done / elapsed
                remaining = (n_total - done) / rate
                print(f"[{timestamp()}] {done}/{n_total} done, "
                      f"{rate:.2f} coal/s, ~{remaining / 3600:.1f}h remaining")
            np.savez(
                out_path,
                target_logits=all_target_logits,
                foil_logits=all_foil_logits,
                coalition_indices=coalition_indices,
                circuit_name=circuit_name,
                ablation_type=ablation_type,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                intact_target_logits=intact_target,
                intact_foil_logits=intact_foil,
                n_coalitions_completed=done,
                sweep_version=4,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    print(f"[{timestamp()}] Sweep complete: {n_total} coalitions in {elapsed_total:.0f}s "
          f"({elapsed_total/3600:.1f}h)")

    np.savez(
        out_path,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        coalition_indices=coalition_indices,
        circuit_name=circuit_name,
        ablation_type=ablation_type,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        intact_target_logits=intact_target,
        intact_foil_logits=intact_foil,
        n_coalitions_completed=n_total,
        elapsed_seconds=elapsed_total,
        sweep_version=4,
    )
    results_volume.commit()
    print(f"[{timestamp()}] Final save committed")
    return f"{circuit_name}/{ablation_type}: {n_total} coalitions, baseline LD={baseline_ld:.4f}"


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    circuits = ["EAP_rti", "C4_random"]
    ablation_types = ["zero", "mean"]

    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching RTI multi-circuit sweeps")
    handles = []
    for circuit in circuits:
        for abl in ablation_types:
            print(f"  Spawning: {circuit} / {abl}")
            handles.append(run_sweep.spawn(circuit, abl))

    for h in handles:
        result = h.get()
        print(f"Completed: {result}")
