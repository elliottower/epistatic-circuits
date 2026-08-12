"""Modal: coalition sweep + Walsh-Hadamard analysis for greater-than task.

Greater-than circuit (Hanna et al. 2023): 7 attention heads.
2^7 = 128 coalitions — runs in minutes per (circuit, ablation) pair.

Metric: prob_diff = P(years > YY) - P(years <= YY) over 100 year tokens.

Steps:
  1. Compute per-head ablation attribution (144 heads, ~2 min)
     → selects top-7 "ablation-discovered" circuit
  2. Run 2^7 coalition sweeps for known, ablation-discovered, and random circuits
     × zero and mean ablation types (6 sweeps total)
  3. Walsh-Hadamard analysis on all results

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_gt_sweep.py
"""

import modal

app = modal.App("gt-coalition-sweep-and-analysis")

results_volume = modal.Volume.from_name("gt-sweep-results", create_if_missing=True)

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
        "data/greater_than_data.csv",
        remote_path="/app/greater_than_data.csv",
    )
    .add_local_file(
        "src/walsh.py",
        remote_path="/app/walsh.py",
    )
)

# Hanna et al. 2023, Section 3.2
GT_KNOWN_CIRCUIT = [(5, 1), (5, 5), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)]

N_CIRCUIT_HEADS = len(GT_KNOWN_CIRCUIT)  # 7


def make_random_circuit(n_heads, seed=42):
    import random
    rng = random.Random(seed)
    all_heads = [(l, h) for l in range(12) for h in range(12)]
    return sorted(rng.sample(all_heads, n_heads))


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_full_pipeline():
    import csv
    import gc
    import json
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")
    from walsh import energy_by_order_normalized, wht, wht_per_head_energy

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # ---- Load model ----
    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()

    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    print(f"[{ts()}] Model loaded: {n_layers}L, {n_model_heads}H, d_head={d_head}")

    # ---- Load prompts ----
    prompts = []
    with open("/app/greater_than_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                "clean": row["clean"],
                "corrupted": row["corrupted"],
                "year_yy": int(row.get("correct_idx", row.get("label"))),
            })
    n_prompts = len(prompts)
    print(f"[{ts()}] Loaded {n_prompts} greater-than prompts")

    # ---- Get year token IDs ----
    year_token_ids = []
    for year in range(100):
        toks = model.tokenizer(f"{year:02d}").input_ids
        assert len(toks) == 1, f"Year {year:02d} has {len(toks)} tokens"
        year_token_ids.append(toks[0])
    year_token_ids = torch.tensor(year_token_ids, device="cuda")

    # ---- Tokenize prompts ----
    all_tokens = []
    year_yys = []
    max_len = 0
    for p in prompts:
        toks = model.to_tokens(p["clean"], prepend_bos=True)
        all_tokens.append(toks)
        year_yys.append(p["year_yy"])
        max_len = max(max_len, toks.shape[1])

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]
    last_positions = torch.tensor(seq_lens - 1, device="cuda")
    year_yy_tensor = torch.tensor(year_yys, device="cuda")

    def compute_prob_diff(logits_at_last):
        """Compute prob_diff per prompt from logits at last position.

        logits_at_last: (batch, vocab_size) or (n_coal, n_prompts, vocab_size)
        Returns: (batch,) or (n_coal, n_prompts) prob_diff values.
        """
        if logits_at_last.dim() == 2:
            year_logits = logits_at_last[:, year_token_ids]
            year_probs = torch.softmax(year_logits, dim=-1)
            prob_diffs = torch.zeros(logits_at_last.shape[0], device="cuda")
            for i in range(logits_at_last.shape[0]):
                yy = year_yys[i]
                prob_diffs[i] = year_probs[i, yy + 1:].sum() - year_probs[i, :yy + 1].sum()
            return prob_diffs
        else:
            n_coal = logits_at_last.shape[0]
            year_logits = logits_at_last[:, :, year_token_ids]
            year_probs = torch.softmax(year_logits, dim=-1)
            prob_diffs = torch.zeros(n_coal, n_prompts, device="cuda")
            for i in range(n_prompts):
                yy = year_yys[i]
                prob_diffs[:, i] = year_probs[:, i, yy + 1:].sum(dim=-1) - year_probs[:, i, :yy + 1].sum(dim=-1)
            return prob_diffs

    # ---- Compute mean z for mean ablation ----
    print(f"[{ts()}] Computing mean z activations...")
    z_sums = {}
    total_positions = 0
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_tokens[start:start + 64]
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: "attn.hook_z" in n,
            )
            for l in range(n_layers):
                act = cache[f"blocks.{l}.attn.hook_z"]
                if l not in z_sums:
                    z_sums[l] = torch.zeros(n_model_heads, d_head, device="cuda", dtype=act.dtype)
                z_sums[l] += act.sum(dim=(0, 1))
            total_positions += batch.shape[0] * batch.shape[1]
            del cache
    mean_z = {l: z_sums[l] / total_positions for l in range(n_layers)}
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[{ts()}] Mean z computed over {total_positions} positions")

    # ---- Custom forward (skip full-vocab unembed until last positions) ----
    def forward_last_logits(tokens, n_coal=1):
        """Forward through blocks, return full logits at last positions only."""
        residual = model.hook_embed(model.embed(tokens))
        pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
        residual = residual + pos_embed
        for block in model.blocks:
            residual = block(residual)

        if n_coal > 1:
            tiled_last = last_positions.repeat(n_coal)
        else:
            tiled_last = last_positions
        seq_idx = torch.arange(tokens.shape[0], device="cuda")
        last_resid = residual[seq_idx, tiled_last, :]
        del residual
        last_normed = model.ln_final(last_resid)
        del last_resid
        last_logits = last_normed @ model.W_U + model.b_U
        del last_normed
        return last_logits

    # ---- Baseline (intact model) ----
    print(f"[{ts()}] Computing intact-model baseline...")
    with torch.no_grad():
        baseline_logits = forward_last_logits(padded_tokens, n_coal=1)
        baseline_pd = compute_prob_diff(baseline_logits)
        baseline_mean_pd = float(baseline_pd.mean())
    print(f"[{ts()}] Intact model: mean prob_diff = {baseline_mean_pd:.4f}")
    del baseline_logits
    torch.cuda.empty_cache()

    # ---- Step 1: Head-level ablation attribution (skip if already computed) ----
    existing_attr_path = "/results/gt_head_attribution.json"
    try:
        with open(existing_attr_path) as f:
            attr_results = json.load(f)
        ablation_circuit = [tuple(h) for h in attr_results["ablation_circuit"]]
        random_circuit = [tuple(h) for h in attr_results["random_circuit"]]
        print(f"[{ts()}] Loaded existing attribution")
        print(f"  Ablation circuit: {[f'L{l}H{h}' for l, h in ablation_circuit]}")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"\n[{ts()}] === Step 1: Head-level ablation attribution ===")
        head_importance = {}

        for ablation_type in ["zero", "mean"]:
            print(f"\n[{ts()}] Attribution with {ablation_type} ablation:")
            importance = np.zeros((n_layers, n_model_heads))

            for layer in tqdm(range(n_layers), desc=f"Attr-{ablation_type}"):
                for head in range(n_model_heads):
                    def hook_fn(act, hook, l=layer, h=head):
                        if ablation_type == "zero":
                            act[:, :, h, :] = 0.0
                        else:
                            act[:, :, h, :] = mean_z[l][h]
                        return act

                    with torch.no_grad():
                        logits = model.run_with_hooks(
                            padded_tokens,
                            fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)],
                            return_type="logits",
                        )
                        seq_idx = torch.arange(n_prompts, device="cuda")
                        last_logits = logits[seq_idx, last_positions, :]
                        ablated_pd = compute_prob_diff(last_logits)
                        importance[layer, head] = float(
                            (baseline_pd - ablated_pd).mean()
                        )
                        del logits, last_logits

            head_importance[ablation_type] = importance

        mean_imp = head_importance["mean"]
        flat = mean_imp.flatten()
        top_idx = np.argsort(np.abs(flat))[::-1][:N_CIRCUIT_HEADS]
        ablation_circuit = sorted(
            [(int(idx // n_model_heads), int(idx % n_model_heads)) for idx in top_idx]
        )
        random_circuit = make_random_circuit(N_CIRCUIT_HEADS)

        print(f"\n[{ts()}] Ablation-discovered circuit: {[f'L{l}H{h}' for l, h in ablation_circuit]}")

        attr_results = {
            "zero_importance": {f"L{l}H{h}": float(head_importance["zero"][l, h])
                                for l in range(n_layers) for h in range(n_model_heads)},
            "mean_importance": {f"L{l}H{h}": float(head_importance["mean"][l, h])
                                for l in range(n_layers) for h in range(n_model_heads)},
            "ablation_circuit": [list(h) for h in ablation_circuit],
            "known_circuit": [list(h) for h in GT_KNOWN_CIRCUIT],
            "random_circuit": [list(h) for h in random_circuit],
            "baseline_mean_prob_diff": baseline_mean_pd,
        }
        with open(existing_attr_path, "w") as f:
            json.dump(attr_results, f, indent=2)
        results_volume.commit()
        print(f"[{ts()}] Attribution saved")

    # ---- Step 2: Coalition sweeps ----
    print(f"\n[{ts()}] === Step 2: Coalition sweeps ===")

    circuits = {
        "known": GT_KNOWN_CIRCUIT,
        "ablation_discovered": ablation_circuit,
        "random": random_circuit,
    }

    all_sweep_results = {}

    for circuit_name, circuit_heads in circuits.items():
        circuit_heads_arr = np.array(circuit_heads)
        n_players = len(circuit_heads)
        n_total = 2 ** n_players
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
        involved_layers = sorted(set(l for l, h in circuit_heads))

        for ablation_type in ["zero", "mean"]:
            key = f"{circuit_name}_{ablation_type}"
            print(f"\n[{ts()}] Sweep: {key} ({n_total} coalitions)")

            N_COAL_BATCH = min(8, n_total)

            # Set up mutable mask tensors
            max_batch_total = N_COAL_BATCH * n_prompts
            layer_mask_tensors = {}
            for l in involved_layers:
                layer_mask_tensors[l] = torch.ones(
                    max_batch_total, 1, n_model_heads, 1, device="cuda"
                )

            mean_z_expanded = {}
            for l in involved_layers:
                mean_z_expanded[l] = mean_z[l].view(1, 1, n_model_heads, d_head)

            # Remove any existing permanent hooks
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
                    mask_bits = np.array(
                        [(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool
                    )
                    row_start = c_pos * n_prompts
                    row_end = row_start + n_prompts
                    for i, (layer, head) in enumerate(circuit_heads):
                        if not mask_bits[i]:
                            layer_mask_tensors[layer][row_start:row_end, 0, head, 0] = 0.0

            def evaluate_batch(coal_indices):
                n_coal = len(coal_indices)
                build_masks(coal_indices)
                tiled_tokens = padded_tokens.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)
                with torch.no_grad():
                    last_logits = forward_last_logits(tiled_tokens, n_coal=n_coal)
                last_logits = last_logits.view(n_coal, n_prompts, -1)
                pd = compute_prob_diff(last_logits)
                return pd.cpu().numpy().astype(np.float64)

            prob_diff_values = np.zeros((n_total, n_prompts), dtype=np.float64)
            sweep_start = time.time()

            for mb_start in tqdm(range(0, n_total, N_COAL_BATCH), desc=key):
                mb_end = min(mb_start + N_COAL_BATCH, n_total)
                coal_indices = list(range(mb_start, mb_end))
                pd_batch = evaluate_batch(coal_indices)
                for i, coal_idx in enumerate(coal_indices):
                    prob_diff_values[coal_idx] = pd_batch[i]

            elapsed = time.time() - sweep_start
            print(f"[{ts()}] {key}: {n_total} coalitions in {elapsed:.1f}s")

            # Save sweep results
            np.savez(
                f"/results/gt_{key}_coalition_values.npz",
                prob_diff=prob_diff_values,
                circuit_heads=circuit_heads_arr,
                n_players=n_players,
                n_prompts=n_prompts,
                year_yys=np.array(year_yys),
                circuit_name=circuit_name,
                ablation_type=ablation_type,
            )
            results_volume.commit()

            all_sweep_results[key] = prob_diff_values

    model.reset_hooks(including_permanent=True)
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Step 3: Walsh-Hadamard analysis ----
    print(f"\n[{ts()}] === Step 3: Walsh-Hadamard analysis ===")

    analysis_results = {}

    for key, pd_values in all_sweep_results.items():
        circuit_name = key.rsplit("_", 1)[0]
        abl_type = key.rsplit("_", 1)[1]
        circuit_heads = circuits[circuit_name]
        n_players = len(circuit_heads)
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]

        mean_pd = pd_values.mean(axis=1)

        full_mask = (1 << n_players) - 1
        intact_pd = float(mean_pd[full_mask])
        empty_pd = float(mean_pd[0])
        faithfulness = intact_pd - empty_pd

        wht_coeffs = wht(mean_pd.astype(np.float64))
        energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)
        energy_nc = energy_spectrum[1:].copy()
        total_nc = energy_nc.sum()
        if total_nc > 0:
            energy_nc_frac = energy_nc / total_nc
        else:
            energy_nc_frac = energy_nc

        head_energy = wht_per_head_energy(mean_pd.astype(np.float64), n_players)

        print(f"\n  {key}:")
        print(f"    Intact prob_diff: {intact_pd:.4f}, Empty: {empty_pd:.4f}")
        print(f"    Faithfulness: {faithfulness:+.4f}")
        print(f"    WHT non-constant energy:")
        for order in range(1, n_players + 1):
            if energy_nc_frac[order - 1] > 0.005:
                print(f"      Order {order}: {energy_nc_frac[order-1]*100:.1f}%")

        # Epistasis: RTI-paper style
        per_prompt_group = pd_values[full_mask] - pd_values[0]
        per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
        for i in range(n_players):
            complement = full_mask ^ (1 << i)
            per_prompt_loo_sum += pd_values[full_mask] - pd_values[complement]
        group_mean = float(np.mean(per_prompt_group))
        loo_sum_mean = float(np.mean(per_prompt_loo_sum))
        if abs(group_mean) > 1e-10:
            epistasis_point = 1.0 - loo_sum_mean / group_mean
        else:
            epistasis_point = 0.0

        # Bootstrap
        rng = np.random.RandomState(42)
        boot_vals = []
        for _ in range(10_000):
            idx = rng.choice(n_prompts, size=n_prompts, replace=True)
            bg = float(np.mean(per_prompt_group[idx]))
            bl = float(np.mean(per_prompt_loo_sum[idx]))
            if abs(bg) > 1e-10:
                boot_vals.append(1.0 - bl / bg)
        boot_vals = np.array(boot_vals)
        ci_lo = float(np.percentile(boot_vals, 2.5))
        ci_hi = float(np.percentile(boot_vals, 97.5))

        print(f"    Epistasis: {epistasis_point*100:.1f}% [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

        analysis_results[key] = {
            "circuit": circuit_name,
            "ablation": abl_type,
            "task": "greater_than",
            "n_players": n_players,
            "n_prompts": n_prompts,
            "heads": head_labels,
            "intact_prob_diff": round(intact_pd, 6),
            "empty_prob_diff": round(empty_pd, 6),
            "faithfulness": round(faithfulness, 6),
            "wht_energy_spectrum": [round(float(x), 6) for x in energy_spectrum],
            "wht_energy_nc_frac": [round(float(x), 6) for x in energy_nc_frac],
            "order1_frac": round(float(energy_nc_frac[0]), 4),
            "order2_frac": round(float(energy_nc_frac[1]), 4),
            "order3plus_frac": round(float(sum(energy_nc_frac[2:])), 4),
            "per_head_energy": {
                head_labels[i]: round(float(head_energy[i]), 6)
                for i in range(n_players)
            },
            "epistasis_point": round(epistasis_point, 4),
            "epistasis_ci_lo": round(ci_lo, 4),
            "epistasis_ci_hi": round(ci_hi, 4),
            "group_effect": round(group_mean, 6),
            "loo_sum": round(loo_sum_mean, 6),
        }

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"SUMMARY — Greater-Than Task")
    print(f"{'='*70}")
    print(f"\n{'Circuit':<22} {'Abl':<6} {'Faith':>8} {'Ord-1':>7} {'Ord-2':>7} {'Ord-3+':>7} {'Epi':>8}")
    print("-" * 72)
    for key in sorted(analysis_results.keys()):
        r = analysis_results[key]
        print(f"{r['circuit']:<22} {r['ablation']:<6} {r['faithfulness']:>+8.4f} "
              f"{r['order1_frac']*100:>6.1f}% {r['order2_frac']*100:>6.1f}% "
              f"{r['order3plus_frac']*100:>6.1f}% {r['epistasis_point']*100:>7.1f}%")

    with open("/results/gt_analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    results_volume.commit()
    print(f"\n[{ts()}] All results saved to gt-sweep-results volume")

    return analysis_results


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching GT pipeline")
    results = run_full_pipeline.remote()
    print("\n\nFinal results:")
    for key, r in results.items():
        print(f"  {r['circuit']:22s} {r['ablation']:6s}  "
              f"faith={r['faithfulness']:+.4f}  "
              f"o1={r['order1_frac']*100:.1f}%  o2={r['order2_frac']*100:.1f}%  "
              f"o3+={r['order3plus_frac']*100:.1f}%  "
              f"epi={r['epistasis_point']*100:.1f}%")
