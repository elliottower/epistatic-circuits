"""Modal analysis: Walsh-Hadamard energy spectra for all circuits.

Reads coalition sweep NPZs from their respective Modal volumes, computes
WHT energy spectra (interaction order decomposition), faithfulness, and
per-head energy. Saves compact JSON results to a results volume.

No model loading — pure numpy analysis on the pre-computed coalition tables.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_analyze_coalitions.py
"""

import modal

app = modal.App("analyze-coalition-spectra")

results_volume = modal.Volume.from_name("coalition-analysis-results", create_if_missing=True)

c2_volume = modal.Volume.from_name("c2-coalition-sweep")
c5_volume = modal.Volume.from_name("c5-coalition-sweep")
c6_volume = modal.Volume.from_name("c6-coalition-sweep")
ic15_volume = modal.Volume.from_name("ic15-coalition-sweep")
rti_volume = modal.Volume.from_name("rti-coalition-sweep")
legacy_volume = modal.Volume.from_name("shapiq-coalition-results-v2")

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

CIRCUIT_SOURCES = {
    "C2_eap": {
        "volume_mount": "/data/c2",
        "zero_file": "c2_zero_v2_coalition_values.npz",
        "mean_file": "c2_mean_v2_coalition_values.npz",
    },
    "C3_canonical": {
        "volume_mount": "/data/legacy",
        "zero_file": "ioi_zero_v2_coalition_values.npz",
        "mean_file": "ioi_mean_v2_coalition_values.npz",
    },
    "C4_random": {
        "volume_mount": "/data/legacy",
        "zero_file": "random15_zero_v2_coalition_values.npz",
        "mean_file": "random15_mean_v2_coalition_values.npz",
    },
    "C5_walsh": {
        "volume_mount": "/data/c5",
        "zero_file": "c5_zero_v2_coalition_values.npz",
        "mean_file": "c5_mean_v2_coalition_values.npz",
    },
    "C6_epistatic": {
        "volume_mount": "/data/c6",
        "zero_file": "c6_zero_v2_coalition_values.npz",
        "mean_file": "c6_mean_v2_coalition_values.npz",
    },
    "IC15": {
        "volume_mount": "/data/ic15",
        "zero_file": "ic15_zero_v2_coalition_values.npz",
        "mean_file": "ic15_mean_v2_coalition_values.npz",
    },
    "RTI": {
        "volume_mount": "/data/rti",
        "zero_file": "rti_zero_v2_coalition_values.npz",
        "mean_file": "rti_mean_v2_coalition_values.npz",
    },
}


@app.function(
    image=image,
    timeout=3600,
    volumes={
        "/data/c2": c2_volume,
        "/data/c5": c5_volume,
        "/data/c6": c6_volume,
        "/data/ic15": ic15_volume,
        "/data/rti": rti_volume,
        "/data/legacy": legacy_volume,
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
        per_head_energy,
        wht,
        wht_energy_by_order,
        wht_per_head_energy,
    )

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print(f"[{ts()}] Starting Walsh-Hadamard analysis of all circuits")

    all_results = {}

    for circuit_name, source in CIRCUIT_SOURCES.items():
        print(f"\n{'='*60}")
        print(f"[{ts()}] Analyzing {circuit_name}")
        print(f"{'='*60}")

        for ablation_type in ["zero", "mean"]:
            key = f"{ablation_type}_file"
            fpath = Path(source["volume_mount"]) / source[key]

            if not fpath.exists():
                print(f"  [{ts()}] WARNING: {fpath} not found, skipping")
                continue

            data = np.load(str(fpath))
            target_logits = data["target_logits"]
            foil_logits = data["foil_logits"]
            circuit_heads = data["circuit_heads"]
            n_players = int(data["n_players"])
            n_prompts = int(data["n_prompts"])
            n_coalitions_completed = int(data.get("n_coalitions_completed", target_logits.shape[0]))

            head_labels = [f"L{l}H{h}" for l, h in circuit_heads]

            logit_diffs = target_logits - foil_logits
            mean_logit_diff = logit_diffs.mean(axis=1)

            n_expected = 2 ** n_players
            if n_coalitions_completed < n_expected:
                print(f"  [{ts()}] WARNING: only {n_coalitions_completed}/{n_expected} coalitions completed")
                continue

            intact_idx = n_expected - 1
            empty_idx = 0
            intact_ld = float(mean_logit_diff[intact_idx])
            empty_ld = float(mean_logit_diff[empty_idx])
            faithfulness = intact_ld - empty_ld

            print(f"  [{ts()}] {ablation_type} ablation: {n_players} heads, {n_prompts} prompts")
            print(f"  [{ts()}] Intact logit-diff: {intact_ld:.4f}")
            print(f"  [{ts()}] Empty logit-diff:  {empty_ld:.4f}")
            print(f"  [{ts()}] Faithfulness (delta): {faithfulness:.4f}")

            values = mean_logit_diff.astype(np.float64)

            wht_coeffs = wht(values)
            energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)

            print(f"  [{ts()}] WHT energy spectrum (fraction of total variance):")
            for order in range(n_players + 1):
                if energy_spectrum[order] > 0.001:
                    print(f"    Order {order:2d}: {energy_spectrum[order]:.4f} ({energy_spectrum[order]*100:.1f}%)")

            energy_nonconstant = energy_spectrum[1:].copy()
            total_nonconstant = energy_nonconstant.sum()
            if total_nonconstant > 0:
                energy_nonconstant_frac = energy_nonconstant / total_nonconstant
            else:
                energy_nonconstant_frac = energy_nonconstant

            print(f"  [{ts()}] Non-constant energy fractions:")
            for order in range(1, n_players + 1):
                if energy_nonconstant_frac[order - 1] > 0.001:
                    print(f"    Order {order:2d}: {energy_nonconstant_frac[order-1]*100:.1f}%")

            head_energy = wht_per_head_energy(values, n_players)
            top_heads = np.argsort(head_energy)[::-1]
            print(f"  [{ts()}] Per-head WHT energy (top 5):")
            for rank, idx in enumerate(top_heads[:5]):
                print(f"    {rank+1}. {head_labels[idx]}: {head_energy[idx]:.4f}")

            comp_values = complement_values(values, n_players)
            comp_energy = wht_energy_by_order(comp_values, n_players)
            print(f"  [{ts()}] Complement (necessity) energy spectrum:")
            for order in range(n_players + 1):
                if comp_energy[order] > 0.001:
                    print(f"    Order {order:2d}: {comp_energy[order]*100:.1f}%")

            result_key = f"{circuit_name}_{ablation_type}"
            all_results[result_key] = {
                "circuit": circuit_name,
                "ablation": ablation_type,
                "n_players": n_players,
                "n_prompts": n_prompts,
                "n_coalitions": n_coalitions_completed,
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
    print(f"[{ts()}] SUMMARY TABLE")
    print(f"{'='*60}")

    print(f"\n{'Circuit':<18} {'Abl':<6} {'Faith':>8} {'Ord-1':>8} {'Ord-2':>8} {'Ord-3+':>8}")
    print("-" * 60)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{r['circuit']:<18} {r['ablation']:<6} {r['faithfulness_delta']:>8.4f} "
              f"{r['order1_frac']*100:>7.1f}% {r['order2_frac']*100:>7.1f}% {r['order3plus_frac']*100:>7.1f}%")

    out_path = "/results/walsh_energy_all_circuits.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    results_volume.commit()
    print(f"\n[{ts()}] Results saved to {out_path}")
    print(f"[{ts()}] Download: modal volume get coalition-analysis-results walsh_energy_all_circuits.json .")

    return all_results


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching Walsh-Hadamard analysis")
    result = analyze_all.remote()

    print(f"\n\nFinal results ({len(result)} entries):")
    for key in sorted(result.keys()):
        r = result[key]
        print(f"  {r['circuit']:18s} {r['ablation']:6s}  faith={r['faithfulness_delta']:+.4f}  "
              f"o1={r['order1_frac']*100:.1f}%  o2={r['order2_frac']*100:.1f}%  o3+={r['order3plus_frac']*100:.1f}%")
