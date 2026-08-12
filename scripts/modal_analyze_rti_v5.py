"""Modal analysis: Walsh-Hadamard energy spectra for RTI v5 sweep results.

Analyzes all 6 sweep files (3 circuits x 2 ablation types) from the
rti-sweep-v5 volume. Reports completion status, faithfulness, and WHT
energy spectra for each.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_analyze_rti_v5.py
"""

import modal

app = modal.App("analyze-rti-v5-spectra")

results_volume = modal.Volume.from_name("coalition-analysis-results", create_if_missing=True)
v5_volume = modal.Volume.from_name("rti-sweep-v5")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "tqdm==4.67.1",
    )
    .add_local_file(
        "src/walsh.py",
        remote_path="/app/walsh.py",
    )
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
    volumes={
        "/data/v5": v5_volume,
        "/results": results_volume,
    },
)
def analyze_all():
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "/app")
    from walsh import (
        complement_values,
        energy_by_order_normalized,
        wht,
        wht_energy_by_order,
        wht_per_head_energy,
    )

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print(f"[{ts()}] Analyzing RTI v5 sweep results (3 circuits x 2 ablation types)")

    all_results = {}

    for key, filename in V5_FILES.items():
        fpath = Path("/data/v5") / filename

        if not fpath.exists():
            print(f"\n[{ts()}] MISSING: {filename}")
            continue

        data = np.load(str(fpath))
        target_logits = data["target_logits"]
        foil_logits = data["foil_logits"]
        circuit_heads = data["circuit_heads"]
        n_players = int(data["n_players"])
        n_prompts = int(data["n_prompts"])
        circuit_name = str(data.get("circuit_name", key.rsplit("_", 1)[0]))
        ablation_type = str(data.get("ablation_type", key.rsplit("_", 1)[1]))

        completed_mask = ~np.isnan(target_logits[:, 0])
        n_completed = int(completed_mask.sum())
        n_expected = 2 ** n_players

        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]

        print(f"\n{'='*60}")
        print(f"[{ts()}] {key}: {n_completed}/{n_expected} coalitions completed")
        print(f"  Heads: {head_labels}")

        if n_completed < n_expected:
            print(f"  INCOMPLETE — {n_expected - n_completed} remaining")
            data.close()
            continue

        logit_diffs = target_logits - foil_logits
        mean_logit_diff = logit_diffs.mean(axis=1)
        data.close()

        intact_idx = n_expected - 1
        empty_idx = 0
        intact_ld = float(mean_logit_diff[intact_idx])
        empty_ld = float(mean_logit_diff[empty_idx])
        faithfulness = intact_ld - empty_ld

        print(f"  Intact logit-diff:  {intact_ld:.4f}")
        print(f"  Empty logit-diff:   {empty_ld:.4f}")
        print(f"  Faithfulness delta: {faithfulness:.4f}")

        values = mean_logit_diff.astype(np.float64)
        wht_coeffs = wht(values)
        energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)

        print(f"  WHT energy spectrum:")
        for order in range(n_players + 1):
            if energy_spectrum[order] > 0.0005:
                print(f"    Order {order:2d}: {energy_spectrum[order]*100:.2f}%")

        energy_nonconstant = energy_spectrum[1:].copy()
        total_nonconstant = energy_nonconstant.sum()
        if total_nonconstant > 0:
            energy_nonconstant_frac = energy_nonconstant / total_nonconstant
        else:
            energy_nonconstant_frac = energy_nonconstant

        print(f"  Non-constant energy fractions:")
        for order in range(1, n_players + 1):
            if energy_nonconstant_frac[order - 1] > 0.0005:
                print(f"    Order {order:2d}: {energy_nonconstant_frac[order-1]*100:.2f}%")

        head_energy = wht_per_head_energy(values, n_players)
        top_heads = np.argsort(head_energy)[::-1]
        print(f"  Per-head WHT energy (top 5):")
        for rank, idx in enumerate(top_heads[:5]):
            print(f"    {rank+1}. {head_labels[idx]}: {head_energy[idx]:.4f}")

        comp_values = complement_values(values, n_players)
        comp_energy = wht_energy_by_order(comp_values, n_players)

        all_results[key] = {
            "circuit": circuit_name,
            "ablation": ablation_type,
            "task": "RTI",
            "n_players": n_players,
            "n_prompts": n_prompts,
            "n_coalitions": n_completed,
            "heads": head_labels,
            "intact_logit_diff": round(intact_ld, 6),
            "empty_logit_diff": round(empty_ld, 6),
            "faithfulness_delta": round(faithfulness, 6),
            "wht_energy_spectrum": [round(float(x), 6) for x in energy_spectrum],
            "wht_energy_nonconstant_frac": [round(float(x), 6) for x in energy_nonconstant_frac],
            "order1_frac": round(float(energy_nonconstant_frac[0]), 4),
            "order2_frac": round(float(energy_nonconstant_frac[1]), 4),
            "order3plus_frac": round(float(sum(energy_nonconstant_frac[2:])), 4),
            "per_head_energy": {head_labels[i]: round(float(head_energy[i]), 6) for i in range(n_players)},
            "complement_energy_spectrum": [round(float(x), 6) for x in comp_energy],
        }

    print(f"\n\n{'='*60}")
    print(f"[{ts()}] SUMMARY TABLE — RTI task")
    print(f"{'='*60}")

    print(f"\n{'Circuit':<14} {'Abl':<6} {'Faith':>8} {'Ord-1':>8} {'Ord-2':>8} {'Ord-3+':>8}")
    print("-" * 56)
    for key in ["RTI_zero", "RTI_mean", "EAP_rti_zero", "EAP_rti_mean",
                "C4_random_zero", "C4_random_mean"]:
        if key not in all_results:
            print(f"{key:<14} — INCOMPLETE")
            continue
        r = all_results[key]
        print(f"{r['circuit']:<14} {r['ablation']:<6} {r['faithfulness_delta']:>8.4f} "
              f"{r['order1_frac']*100:>7.1f}% {r['order2_frac']*100:>7.1f}% {r['order3plus_frac']*100:>7.1f}%")

    existing_path = "/results/walsh_energy_all_circuits.json"
    try:
        with open(existing_path) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}

    for key, result in all_results.items():
        existing[f"RTI_v5_{key}"] = result

    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)
    results_volume.commit()

    rti_v5_path = "/results/walsh_energy_rti_v5.json"
    with open(rti_v5_path, "w") as f:
        json.dump(all_results, f, indent=2)
    results_volume.commit()

    print(f"\n[{ts()}] Results saved to {rti_v5_path}")
    return all_results


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching RTI v5 Walsh-Hadamard analysis")
    results = analyze_all.remote()

    print(f"\n\n{'='*60}")
    print("RTI v5 Results Summary")
    print(f"{'='*60}")
    for key in ["RTI_zero", "RTI_mean", "EAP_rti_zero", "EAP_rti_mean",
                "C4_random_zero", "C4_random_mean"]:
        if key not in results:
            print(f"  {key}: INCOMPLETE")
            continue
        r = results[key]
        print(f"  {r['circuit']:14s} {r['ablation']:6s}  faith={r['faithfulness_delta']:+.4f}  "
              f"o1={r['order1_frac']*100:.1f}%  o2={r['order2_frac']*100:.1f}%  o3+={r['order3plus_frac']*100:.1f}%")
