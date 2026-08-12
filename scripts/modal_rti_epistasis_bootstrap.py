"""Modal: bootstrap CIs for RTI-paper-style epistasis scores.

Computes epistasis = 1 - (sum_LOO / group_effect) per-prompt,
then bootstraps over prompts for 95% CIs.

Uses existing v5 coalition tables — no GPU needed.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_rti_epistasis_bootstrap.py
"""

import modal

app = modal.App("rti-epistasis-bootstrap")

v5_volume = modal.Volume.from_name("rti-sweep-v5")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")

V5_FILES = {
    "RTI_zero": "RTI_zero_rti_v5_coalition_values.npz",
    "RTI_mean": "RTI_mean_rti_v5_coalition_values.npz",
    "EAP_rti_zero": "EAP_rti_zero_rti_v5_coalition_values.npz",
    "EAP_rti_mean": "EAP_rti_mean_rti_v5_coalition_values.npz",
    "C4_random_zero": "C4_random_zero_rti_v5_coalition_values.npz",
    "C4_random_mean": "C4_random_mean_rti_v5_coalition_values.npz",
}

N_BOOT = 10_000


@app.function(image=image, timeout=3600, volumes={"/data/v5": v5_volume})
def bootstrap_epistasis():
    import json
    import time
    from pathlib import Path

    import numpy as np

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    results = {}

    for key, filename in V5_FILES.items():
        fpath = Path("/data/v5") / filename
        if not fpath.exists():
            continue

        data = np.load(str(fpath))
        target_logits = data["target_logits"]
        foil_logits = data["foil_logits"]
        circuit_heads = data["circuit_heads"]
        n_players = int(data["n_players"])
        n_prompts = target_logits.shape[1]
        data.close()

        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
        logit_diffs = target_logits - foil_logits  # (2^15, n_prompts)

        full_mask = (1 << n_players) - 1

        # Per-prompt group effect: v(all) - v(empty)
        per_prompt_group = logit_diffs[full_mask] - logit_diffs[0]

        # Per-prompt sum of LOO marginals: sum_i [v(all) - v(all \ {i})]
        per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
        per_head_marginals = {}
        for i in range(n_players):
            complement = full_mask ^ (1 << i)
            marginal_i = logit_diffs[full_mask] - logit_diffs[complement]
            per_prompt_loo_sum += marginal_i
            per_head_marginals[head_labels[i]] = float(np.mean(marginal_i))

        # Point estimate
        group_mean = float(np.mean(per_prompt_group))
        loo_sum_mean = float(np.mean(per_prompt_loo_sum))
        if abs(group_mean) > 1e-10:
            epistasis_point = 1.0 - loo_sum_mean / group_mean
        else:
            epistasis_point = 0.0

        # Bootstrap over prompts (same as RTI paper's bootstrap)
        rng = np.random.RandomState(42)
        boot_epistasis = []
        for _ in range(N_BOOT):
            idx = rng.choice(n_prompts, size=n_prompts, replace=True)
            boot_group = float(np.mean(per_prompt_group[idx]))
            boot_loo = float(np.mean(per_prompt_loo_sum[idx]))
            if abs(boot_group) > 1e-10:
                boot_epistasis.append(1.0 - boot_loo / boot_group)
        boot_epistasis = np.array(boot_epistasis)
        ci_lo = float(np.percentile(boot_epistasis, 2.5))
        ci_hi = float(np.percentile(boot_epistasis, 97.5))
        boot_mean = float(np.mean(boot_epistasis))
        boot_se = float(np.std(boot_epistasis))

        print(f"\n{'='*60}")
        print(f"[{ts()}] {key}")
        print(f"  Group effect (mean): {group_mean:+.4f}")
        print(f"  Sum LOO (mean):      {loo_sum_mean:+.4f}")
        print(f"  Epistasis:           {epistasis_point:.4f} ({epistasis_point*100:.1f}%)")
        print(f"  Bootstrap 95% CI:    [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  Bootstrap SE:        {boot_se:.4f}")
        print(f"  Per-head LOO marginals:")
        sorted_heads = sorted(per_head_marginals.items(), key=lambda x: abs(x[1]), reverse=True)
        for h, m in sorted_heads:
            print(f"    {h}: {m:+.4f}")

        results[key] = {
            "group_effect": round(group_mean, 6),
            "loo_sum": round(loo_sum_mean, 6),
            "epistasis": round(epistasis_point, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "boot_se": round(boot_se, 4),
            "n_prompts": n_prompts,
            "per_head_marginals": {h: round(m, 6) for h, m in per_head_marginals.items()},
        }

    print(f"\n\n{'='*60}")
    print(f"SUMMARY — RTI Paper Epistasis Score (1 - LOO_sum / group)")
    print(f"{'='*60}")
    print(f"\n{'Circuit':<14} {'Abl':<6} {'Epistasis':>10} {'95% CI':>20} {'Group':>8} {'LOO sum':>8}")
    print("-" * 72)
    for key in ["RTI_zero", "RTI_mean", "EAP_rti_zero", "EAP_rti_mean",
                "C4_random_zero", "C4_random_mean"]:
        if key not in results:
            continue
        r = results[key]
        circuit = key.rsplit("_", 1)[0]
        abl = key.rsplit("_", 1)[1]
        print(f"{circuit:<14} {abl:<6} {r['epistasis']*100:>9.1f}% "
              f"[{r['ci_lo']*100:>6.1f}%, {r['ci_hi']*100:>6.1f}%] "
              f"{r['group_effect']:>+8.4f} {r['loo_sum']:>+8.4f}")

    return results


@app.local_entrypoint()
def main():
    results = bootstrap_epistasis.remote()
    print("\nDone.")
