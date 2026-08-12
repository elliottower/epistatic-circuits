"""Modal analysis: simple epistasis test for RTI v5 sweep results.

Classic whole-vs-sum-of-parts epistasis measure:
  - Joint effect:  v(all) - v(empty)
  - Sum of individual effects: sum_i [v({i}) - v(empty)]
  - Sum of marginal effects:   sum_i [v(all) - v(all \ {i})]

If joint > sum_of_individuals: synergy (heads do more together)
If joint < sum_of_individuals: redundancy (heads overlap)

Also computes per-head individual vs marginal contributions.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_rti_epistasis_simple.py
"""

import modal

app = modal.App("rti-epistasis-simple-test")

v5_volume = modal.Volume.from_name("rti-sweep-v5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4")
)

V5_FILES = {
    "RTI_zero": "RTI_zero_rti_v5_coalition_values.npz",
    "RTI_mean": "RTI_mean_rti_v5_coalition_values.npz",
    "EAP_rti_zero": "EAP_rti_zero_rti_v5_coalition_values.npz",
    "EAP_rti_mean": "EAP_rti_mean_rti_v5_coalition_values.npz",
    "C4_random_zero": "C4_random_zero_rti_v5_coalition_values.npz",
    "C4_random_mean": "C4_random_mean_rti_v5_coalition_values.npz",
}


@app.function(
    image=image,
    timeout=3600,
    volumes={"/data/v5": v5_volume},
)
def analyze_epistasis():
    import json
    import time
    from pathlib import Path

    import numpy as np

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    all_results = {}

    for key, filename in V5_FILES.items():
        fpath = Path("/data/v5") / filename
        if not fpath.exists():
            continue

        data = np.load(str(fpath))
        target_logits = data["target_logits"]
        foil_logits = data["foil_logits"]
        circuit_heads = data["circuit_heads"]
        n_players = int(data["n_players"])
        data.close()

        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
        logit_diffs = target_logits - foil_logits
        mean_ld = logit_diffs.mean(axis=1)

        full_mask = (1 << n_players) - 1
        v_all = float(mean_ld[full_mask])
        v_empty = float(mean_ld[0])
        joint_effect = v_all - v_empty

        individual_effects = []
        marginal_effects = []
        for i in range(n_players):
            singleton = 1 << i
            v_single = float(mean_ld[singleton])
            individual_effects.append(v_single - v_empty)

            complement = full_mask ^ singleton
            v_complement = float(mean_ld[complement])
            marginal_effects.append(v_all - v_complement)

        sum_individual = sum(individual_effects)
        sum_marginal = sum(marginal_effects)

        epistasis_additive = joint_effect - sum_individual
        epistasis_marginal = joint_effect - sum_marginal

        if sum_individual != 0:
            ratio_individual = joint_effect / sum_individual
        else:
            ratio_individual = float('inf')

        print(f"\n{'='*70}")
        print(f"[{ts()}] {key}")
        print(f"{'='*70}")
        print(f"  v(all 15) = {v_all:.4f}")
        print(f"  v(empty)  = {v_empty:.4f}")
        print(f"  Joint effect (all - empty):       {joint_effect:+.4f}")
        print(f"  Sum of individual additions:       {sum_individual:+.4f}")
        print(f"  Sum of marginal contributions:     {sum_marginal:+.4f}")
        print(f"  Epistasis (joint - sum_indiv):     {epistasis_additive:+.4f}")
        print(f"  Epistasis (joint - sum_marginal):  {epistasis_marginal:+.4f}")
        print(f"  Ratio (joint / sum_indiv):         {ratio_individual:.3f}")
        if ratio_individual > 1:
            print(f"  => SYNERGY: circuit does {ratio_individual:.1f}x more together than parts suggest")
        elif ratio_individual < 1 and ratio_individual > 0:
            print(f"  => REDUNDANCY: circuit does {1/ratio_individual:.1f}x less than parts suggest")
        elif ratio_individual < 0:
            print(f"  => SIGN FLIP: individual effects go opposite direction from joint")

        print(f"\n  Per-head breakdown:")
        print(f"  {'Head':<8} {'Individual':>11} {'Marginal':>11} {'Ratio':>8}")
        print(f"  {'-'*42}")
        for i in range(n_players):
            if individual_effects[i] != 0:
                head_ratio = marginal_effects[i] / individual_effects[i]
                ratio_str = f"{head_ratio:.2f}"
            else:
                ratio_str = "inf"
            print(f"  {head_labels[i]:<8} {individual_effects[i]:>+11.4f} "
                  f"{marginal_effects[i]:>+11.4f} {ratio_str:>8}")

        # Also compute pairwise epistasis for top 3 heads
        sorted_by_individual = sorted(range(n_players),
                                       key=lambda i: abs(individual_effects[i]),
                                       reverse=True)
        top3 = sorted_by_individual[:3]
        print(f"\n  Pairwise epistasis (top 3 heads by individual effect):")
        print(f"  {'Pair':<14} {'Joint':>8} {'Sum':>8} {'Epistasis':>10}")
        print(f"  {'-'*44}")
        for a_idx in range(len(top3)):
            for b_idx in range(a_idx + 1, len(top3)):
                a, b = top3[a_idx], top3[b_idx]
                mask_a = 1 << a
                mask_b = 1 << b
                mask_ab = mask_a | mask_b
                v_a = float(mean_ld[mask_a]) - v_empty
                v_b = float(mean_ld[mask_b]) - v_empty
                v_ab = float(mean_ld[mask_ab]) - v_empty
                pair_epi = v_ab - (v_a + v_b)
                pair_label = f"{head_labels[a]}+{head_labels[b]}"
                print(f"  {pair_label:<14} {v_ab:>+8.4f} {v_a + v_b:>+8.4f} {pair_epi:>+10.4f}")

        all_results[key] = {
            "v_all": round(v_all, 6),
            "v_empty": round(v_empty, 6),
            "joint_effect": round(joint_effect, 6),
            "sum_individual": round(sum_individual, 6),
            "sum_marginal": round(sum_marginal, 6),
            "epistasis_additive": round(epistasis_additive, 6),
            "epistasis_marginal": round(epistasis_marginal, 6),
            "ratio_joint_over_individual": round(ratio_individual, 6),
            "per_head": {
                head_labels[i]: {
                    "individual": round(individual_effects[i], 6),
                    "marginal": round(marginal_effects[i], 6),
                }
                for i in range(n_players)
            },
        }

    print(f"\n\n{'='*70}")
    print(f"SUMMARY — Simple Epistasis Test (RTI task)")
    print(f"{'='*70}")
    print(f"\n{'Circuit':<14} {'Abl':<6} {'Joint':>8} {'Sum Indiv':>10} {'Ratio':>8} {'Type':<12}")
    print("-" * 62)
    for key in ["RTI_zero", "RTI_mean", "EAP_rti_zero", "EAP_rti_mean",
                "C4_random_zero", "C4_random_mean"]:
        if key not in all_results:
            continue
        r = all_results[key]
        ratio = r["ratio_joint_over_individual"]
        if ratio > 1.05:
            etype = "SYNERGY"
        elif ratio < 0.95 and ratio > 0:
            etype = "REDUNDANCY"
        elif ratio < 0:
            etype = "SIGN FLIP"
        else:
            etype = "ADDITIVE"
        circuit = key.rsplit("_", 1)[0]
        abl = key.rsplit("_", 1)[1]
        print(f"{circuit:<14} {abl:<6} {r['joint_effect']:>+8.4f} {r['sum_individual']:>+10.4f} "
              f"{ratio:>8.2f} {etype:<12}")

    return all_results


@app.local_entrypoint()
def main():
    results = analyze_epistasis.remote()
    print("\nDone.")
