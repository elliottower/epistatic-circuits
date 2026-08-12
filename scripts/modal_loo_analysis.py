"""Modal: LOO marginals, pairwise interactions, and epistasis decomposition.

Loads coalition NPZ files from all task volumes and computes:
  1. Per-head LOO marginal contributions
  2. Top pairwise interaction strengths (from Walsh order-2 coefficients)
  3. Epistasis decomposition: additive (sum LOO) vs synergistic (remainder)
  4. Shapley values for fair attribution

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_loo_analysis.py
"""

import modal

app = modal.App("loo-pairwise-analysis")

gt_vol = modal.Volume.from_name("gt-sweep-results", create_if_missing=False)
ind_vol = modal.Volume.from_name("induction-sweep-results", create_if_missing=False)
gender_vol = modal.Volume.from_name("gender-sweep-results", create_if_missing=False)
sva_vol = modal.Volume.from_name("sva-sweep-results", create_if_missing=False)
results_vol = modal.Volume.from_name("loo-analysis-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "matplotlib==3.9.4",
    )
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
)


@app.function(
    image=image,
    timeout=86400,
    volumes={
        "/gt": gt_vol,
        "/ind": ind_vol,
        "/gender": gender_vol,
        "/sva": sva_vol,
        "/results": results_vol,
    },
)
def run_analysis():
    import json
    import os
    import time
    import sys
    sys.path.insert(0, "/app")
    from walsh import wht, energy_by_order, _popcount_array, mobius_transform

    import numpy as np

    def ts():
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print(f"[{ts()}] Starting LOO + pairwise analysis")

    TASKS = {
        "greater_than": {
            "volume": "/gt",
            "prefix": "gt",
            "metric_key": "prob_diff_scores",
            "circuits": ["known", "ablation_discovered", "random"],
        },
        "induction": {
            "volume": "/ind",
            "prefix": "induction",
            "metric_key": "log_prob_scores",
            "circuits": ["known", "ablation_discovered", "random"],
        },
        "gender_bias": {
            "volume": "/gender",
            "prefix": "gender",
            "metric_key": "logit_diff_scores",
            "circuits": ["known", "ablation_discovered", "random"],
        },
        "sva": {
            "volume": "/sva",
            "prefix": "sva",
            "metric_key": "verb_scores",
            "circuits": ["known", "ablation_discovered", "random"],
        },
    }

    all_results = {}

    for task_name, task_cfg in TASKS.items():
        vol_path = task_cfg["volume"]
        prefix = task_cfg["prefix"]
        metric_key = task_cfg["metric_key"]

        attr_path = os.path.join(vol_path, f"{prefix}_head_attribution.json")
        if not os.path.exists(attr_path):
            print(f"[{ts()}] Skipping {task_name}: no attribution file")
            continue

        with open(attr_path) as f:
            attr = json.load(f)

        print(f"\n{'='*70}")
        print(f"  {task_name.upper()}")
        print(f"{'='*70}")

        task_results = {}

        for circuit_name in task_cfg["circuits"]:
            for ablation_type in ["zero", "mean"]:
                key = f"{circuit_name}_{ablation_type}"
                npz_path = os.path.join(
                    vol_path,
                    f"{prefix}_{key}_coalition_values.npz",
                )
                if not os.path.exists(npz_path):
                    print(f"  [{ts()}] Skipping {key}: NPZ not found")
                    continue

                data = np.load(npz_path)
                heads_arr = data["circuit_heads"]
                n_players = int(data["n_players"])
                n_prompts = int(data["n_prompts"])

                score_keys = [k for k in data.files if "score" in k or "diff" in k or "prob" in k]
                if score_keys:
                    values_2d = data[score_keys[0]]
                else:
                    values_2d = data[list(data.files)[0]]

                head_labels = [f"L{l}H{h}" for l, h in heads_arr]
                mean_values = values_2d.mean(axis=1).astype(np.float64)

                n_total = 2 ** n_players
                full_idx = n_total - 1
                empty_idx = 0

                v_full = mean_values[full_idx]
                v_empty = mean_values[empty_idx]
                faithfulness = v_full - v_empty

                # ---- LOO marginals ----
                loo_marginals = {}
                for j in range(n_players):
                    without_j = full_idx & ~(1 << j)
                    loo_marginals[head_labels[j]] = float(
                        v_full - mean_values[without_j]
                    )

                loo_sum = sum(loo_marginals.values())
                synergy = faithfulness - loo_sum
                if abs(faithfulness) > 1e-10:
                    synergy_frac = synergy / faithfulness
                else:
                    synergy_frac = 0.0

                # ---- Walsh coefficients ----
                w = wht(mean_values)
                w_normalized = w / n_total

                # Order-1 Walsh coefficients (individual effects)
                order1_coeffs = {}
                for j in range(n_players):
                    idx = 1 << j
                    order1_coeffs[head_labels[j]] = float(w_normalized[idx])

                # Order-2 Walsh coefficients (pairwise interactions)
                order2_coeffs = {}
                for j in range(n_players):
                    for k in range(j + 1, n_players):
                        idx = (1 << j) | (1 << k)
                        pair_label = f"{head_labels[j]}×{head_labels[k]}"
                        order2_coeffs[pair_label] = float(w_normalized[idx])

                # Sort by absolute value
                sorted_order1 = sorted(
                    order1_coeffs.items(), key=lambda x: abs(x[1]), reverse=True
                )
                sorted_order2 = sorted(
                    order2_coeffs.items(), key=lambda x: abs(x[1]), reverse=True
                )

                # Energy fractions
                pc = _popcount_array(n_players)
                w2 = w_normalized ** 2
                total_energy = w2.sum()
                nc_energy = total_energy - w2[0]
                if nc_energy > 0:
                    order_energies = {}
                    for order in range(1, n_players + 1):
                        e = w2[pc == order].sum()
                        order_energies[order] = float(e / nc_energy)
                else:
                    order_energies = {o: 0.0 for o in range(1, n_players + 1)}

                # ---- Shapley values ----
                shapley = np.zeros(n_players)
                from math import factorial
                for j in range(n_players):
                    for S_bits in range(n_total):
                        if S_bits & (1 << j):
                            continue
                        s_size = bin(S_bits).count('1')
                        S_with_j = S_bits | (1 << j)
                        marginal = mean_values[S_with_j] - mean_values[S_bits]
                        weight = (
                            factorial(s_size)
                            * factorial(n_players - s_size - 1)
                            / factorial(n_players)
                        )
                        shapley[j] += weight * marginal

                shapley_dict = {
                    head_labels[j]: float(shapley[j]) for j in range(n_players)
                }
                sorted_shapley = sorted(
                    shapley_dict.items(), key=lambda x: abs(x[1]), reverse=True
                )

                # ---- Pairwise Shapley interaction index ----
                interaction_idx = {}
                for j in range(n_players):
                    for k in range(j + 1, n_players):
                        I_jk = 0.0
                        for S_bits in range(n_total):
                            if S_bits & (1 << j) or S_bits & (1 << k):
                                continue
                            s_size = bin(S_bits).count('1')
                            S_jk = S_bits | (1 << j) | (1 << k)
                            S_j = S_bits | (1 << j)
                            S_k = S_bits | (1 << k)
                            delta = (
                                mean_values[S_jk]
                                - mean_values[S_j]
                                - mean_values[S_k]
                                + mean_values[S_bits]
                            )
                            weight = (
                                factorial(s_size)
                                * factorial(n_players - s_size - 2)
                                / factorial(n_players - 1)
                            )
                            I_jk += weight * delta
                        pair_label = f"{head_labels[j]}×{head_labels[k]}"
                        interaction_idx[pair_label] = float(I_jk)

                sorted_interactions = sorted(
                    interaction_idx.items(), key=lambda x: abs(x[1]), reverse=True
                )

                # ---- Bootstrap epistasis ----
                n_boot = 2000
                rng = np.random.default_rng(42)
                boot_epi = np.zeros(n_boot)
                for b in range(n_boot):
                    idx_boot = rng.integers(0, n_prompts, size=n_prompts)
                    boot_vals = values_2d[:, idx_boot].mean(axis=1)
                    bv_full = boot_vals[full_idx]
                    bv_empty = boot_vals[empty_idx]
                    b_faith = bv_full - bv_empty
                    b_loo = 0.0
                    for j_idx in range(n_players):
                        without = full_idx & ~(1 << j_idx)
                        b_loo += bv_full - boot_vals[without]
                    if abs(b_faith) > 1e-10:
                        boot_epi[b] = 1.0 - b_loo / b_faith
                    else:
                        boot_epi[b] = 0.0
                epi_mean = float(np.mean(boot_epi))
                epi_lo = float(np.percentile(boot_epi, 2.5))
                epi_hi = float(np.percentile(boot_epi, 97.5))

                # ---- Print results ----
                print(f"\n  --- {key} ({n_players} heads, {n_prompts} prompts) ---")
                print(f"  Faithfulness: {faithfulness:+.4f}")
                print(f"  LOO sum: {loo_sum:+.4f} | Synergy: {synergy:+.4f} ({synergy_frac:.1%})")
                print(f"  Epistasis: {epi_mean:.3f} [{epi_lo:.3f}, {epi_hi:.3f}]")

                print(f"\n  LOO marginals (|largest| first):")
                sorted_loo = sorted(
                    loo_marginals.items(), key=lambda x: abs(x[1]), reverse=True
                )
                for head, val in sorted_loo:
                    pct = abs(val / faithfulness) * 100 if abs(faithfulness) > 1e-10 else 0
                    print(f"    {head:6s}: {val:+.4f}  ({pct:.1f}% of faith.)")

                print(f"\n  Shapley values:")
                for head, val in sorted_shapley:
                    print(f"    {head:6s}: {val:+.4f}")

                print(f"\n  Top 5 pairwise interactions (Shapley interaction index):")
                for pair, val in sorted_interactions[:5]:
                    print(f"    {pair:16s}: {val:+.6f}")

                print(f"\n  Walsh energy: ord-1={order_energies.get(1,0):.1%}  "
                      f"ord-2={order_energies.get(2,0):.1%}  "
                      f"ord-3+={sum(order_energies.get(o,0) for o in range(3, n_players+1)):.1%}")

                # Store
                task_results[key] = {
                    "circuit": circuit_name,
                    "ablation": ablation_type,
                    "n_players": n_players,
                    "n_prompts": n_prompts,
                    "heads": head_labels,
                    "faithfulness": float(faithfulness),
                    "loo_marginals": loo_marginals,
                    "loo_sum": float(loo_sum),
                    "synergy": float(synergy),
                    "synergy_fraction": float(synergy_frac),
                    "epistasis_mean": epi_mean,
                    "epistasis_ci": [epi_lo, epi_hi],
                    "shapley_values": shapley_dict,
                    "top_pairwise_interactions": {
                        p: v for p, v in sorted_interactions[:10]
                    },
                    "walsh_order1_coeffs": order1_coeffs,
                    "walsh_order2_coeffs": order2_coeffs,
                    "energy_spectrum": order_energies,
                }

        all_results[task_name] = task_results
        results_vol.commit()

    # ---- Cross-task summary table ----
    print(f"\n\n{'='*90}")
    print("CROSS-TASK SUMMARY (mean ablation only)")
    print(f"{'='*90}")
    print(f"{'Task':<16} {'Circuit':<20} {'Faith':>8} {'LOO sum':>8} {'Synergy':>8} "
          f"{'Epi':>8} {'Ord-1':>7} {'Ord-2':>7}")
    print("-" * 90)
    for task_name in ["greater_than", "induction", "gender_bias", "sva"]:
        if task_name not in all_results:
            continue
        for circuit in ["known", "ablation_discovered", "random"]:
            key = f"{circuit}_mean"
            if key not in all_results[task_name]:
                continue
            r = all_results[task_name][key]
            print(f"{task_name:<16} {circuit:<20} {r['faithfulness']:>+8.4f} "
                  f"{r['loo_sum']:>+8.4f} {r['synergy']:>+8.4f} "
                  f"{r['epistasis_mean']:>8.3f} "
                  f"{r['energy_spectrum'].get(1,0):>6.1%} "
                  f"{r['energy_spectrum'].get(2,0):>6.1%}")

    # Save
    with open("/results/loo_analysis_all_tasks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    results_vol.commit()
    print(f"\n[{ts()}] Results saved to loo-analysis-results volume")


@app.local_entrypoint()
def main():
    print("Launching LOO + pairwise analysis")
    run_analysis.remote()
