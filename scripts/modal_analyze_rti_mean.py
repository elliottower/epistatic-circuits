"""Modal analysis: Walsh-Hadamard energy spectrum for RTI circuit (mean ablation).

The mean ablation sweep is complete. Zero is still running — will analyze
separately when done.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_analyze_rti_mean.py
"""

import modal

app = modal.App("analyze-rti-mean-spectrum")

results_volume = modal.Volume.from_name("coalition-analysis-results", create_if_missing=True)
rti_volume = modal.Volume.from_name("rti-coalition-sweep")

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


@app.function(
    image=image,
    timeout=3600,
    volumes={
        "/data/rti": rti_volume,
        "/results": results_volume,
    },
)
def analyze_rti_mean():
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

    print(f"[{ts()}] Analyzing RTI circuit (mean ablation)")

    fpath = "/data/rti/rti_mean_v2_coalition_values.npz"
    data = np.load(fpath)
    target_logits = data["target_logits"]
    foil_logits = data["foil_logits"]
    circuit_heads = data["circuit_heads"]
    n_players = int(data["n_players"])
    n_prompts = int(data["n_prompts"])
    n_coalitions_completed = int(data.get("n_coalitions_completed", 0))
    data.close()

    head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
    logit_diffs = target_logits - foil_logits
    mean_logit_diff = logit_diffs.mean(axis=1)

    n_expected = 2 ** n_players
    print(f"[{ts()}] {n_coalitions_completed}/{n_expected} coalitions, {n_players} heads, {n_prompts} prompts")

    intact_idx = n_expected - 1
    empty_idx = 0
    intact_ld = float(mean_logit_diff[intact_idx])
    empty_ld = float(mean_logit_diff[empty_idx])
    faithfulness = intact_ld - empty_ld

    print(f"[{ts()}] Intact logit-diff: {intact_ld:.4f}")
    print(f"[{ts()}] Empty logit-diff:  {empty_ld:.4f}")
    print(f"[{ts()}] Faithfulness (delta): {faithfulness:.4f}")

    values = mean_logit_diff.astype(np.float64)
    wht_coeffs = wht(values)
    energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)

    print(f"\n[{ts()}] WHT energy spectrum (fraction of total variance):")
    for order in range(n_players + 1):
        if energy_spectrum[order] > 0.0005:
            print(f"  Order {order:2d}: {energy_spectrum[order]*100:.2f}%")

    energy_nonconstant = energy_spectrum[1:].copy()
    total_nonconstant = energy_nonconstant.sum()
    if total_nonconstant > 0:
        energy_nonconstant_frac = energy_nonconstant / total_nonconstant
    else:
        energy_nonconstant_frac = energy_nonconstant

    print(f"\n[{ts()}] Non-constant energy fractions:")
    for order in range(1, n_players + 1):
        if energy_nonconstant_frac[order - 1] > 0.0005:
            print(f"  Order {order:2d}: {energy_nonconstant_frac[order-1]*100:.2f}%")

    head_energy = wht_per_head_energy(values, n_players)
    top_heads = np.argsort(head_energy)[::-1]
    print(f"\n[{ts()}] Per-head WHT energy (all 15):")
    for rank, idx in enumerate(top_heads):
        print(f"  {rank+1:2d}. {head_labels[idx]}: {head_energy[idx]:.4f}")

    comp_values = complement_values(values, n_players)
    comp_energy = wht_energy_by_order(comp_values, n_players)
    print(f"\n[{ts()}] Complement (necessity) energy spectrum:")
    for order in range(n_players + 1):
        if comp_energy[order] > 0.0005:
            print(f"  Order {order:2d}: {comp_energy[order]*100:.2f}%")

    result = {
        "circuit": "RTI",
        "ablation": "mean",
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

    existing_path = "/results/walsh_energy_all_circuits.json"
    try:
        with open(existing_path) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    all_results["RTI_mean"] = result

    with open(existing_path, "w") as f:
        json.dump(all_results, f, indent=2)
    results_volume.commit()

    print(f"\n[{ts()}] Result appended to {existing_path}")
    print(f"\nSummary: RTI mean — faith={faithfulness:.4f}, "
          f"o1={energy_nonconstant_frac[0]*100:.1f}%, "
          f"o2={energy_nonconstant_frac[1]*100:.1f}%, "
          f"o3+={sum(energy_nonconstant_frac[2:])*100:.1f}%")

    return result


@app.local_entrypoint()
def main():
    result = analyze_rti_mean.remote()
    print(f"\nDone: RTI mean — faith={result['faithfulness_delta']:.4f}, "
          f"o1={result['order1_frac']*100:.1f}%, o2={result['order2_frac']*100:.1f}%, "
          f"o3+={result['order3plus_frac']*100:.1f}%")
