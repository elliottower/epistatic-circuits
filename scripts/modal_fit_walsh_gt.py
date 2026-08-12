"""Modal: Fit LASSO-Walsh on GT coalition samples and extract C5/C6 circuits.

Reads from walsh-discovery-gt volume, fits sparse Walsh coefficients,
saves results JSON back to the volume.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_fit_walsh_gt.py
"""

import modal

app = modal.App("fit-walsh-gt")

results_volume = modal.Volume.from_name("walsh-discovery-gt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "scipy==1.14.1",
    )
    .add_local_file("src/walsh_discovery.py", remote_path="/app/walsh_discovery.py")
)


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=3600,
    cpu=4,
)
def fit_walsh():
    import json
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/app")
    from walsh_discovery import (
        compute_energy_spectrum,
        extract_head_rankings,
        extract_top_interactions,
        fit_lasso_walsh,
    )

    data_path = "/results/walsh_gt_144heads_mean_coalitions.npz"
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    masks = data["masks"]
    n_players = int(data["n_players"])
    all_heads = [tuple(h) for h in data["circuit_heads"]]

    scores = data["scores"]
    values = scores.mean(axis=1)

    print(f"  {masks.shape[0]} samples, {n_players} players")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}], "
          f"mean={values.mean():.3f}, std={values.std():.3f}")

    t0 = time.time()
    coefs, labels, r2, alpha_used = fit_lasso_walsh(
        masks, values, max_order=2, n_players=n_players, alpha=None
    )
    fit_time = time.time() - t0
    print(f"Fitting took {fit_time:.1f}s")

    head_rankings = extract_head_rankings(coefs, labels, n_players, all_heads)
    top_interactions = extract_top_interactions(coefs, labels, all_heads, top_k=50)
    energy = compute_energy_spectrum(coefs, labels, max_order=2)

    top_k = 7  # GT known circuit has 7 heads
    discovered_circuit = head_rankings[:top_k]
    discovered_heads = [(h["layer"], h["head_idx"]) for h in discovered_circuit]

    print(f"\n=== C5 Walsh-Discovered Circuit (top {top_k} by order-1) ===")
    for i, h in enumerate(discovered_circuit):
        print(f"  {i+1:2d}. {h['head']}: walsh_o1 = {h['walsh_o1']:+.4f}")

    # Also show top-15 for reference
    print(f"\n=== C5 top-15 by order-1 ===")
    for i, h in enumerate(head_rankings[:15]):
        print(f"  {i+1:2d}. {h['head']}: walsh_o1 = {h['walsh_o1']:+.4f}")

    print(f"\n=== Energy Spectrum ===")
    for order, frac in sorted(energy.items()):
        print(f"  Order {order}: {frac*100:.2f}%")

    print(f"\n=== Top 10 Pairwise Interactions ===")
    for ix in top_interactions[:10]:
        print(f"  {ix['heads'][0]} x {ix['heads'][1]}: {ix['walsh_o2']:+.4f}")

    # C6: top-K heads by order-2 energy
    head_o2_energy = {}
    for idx, label in enumerate(labels):
        if label.startswith("o2:"):
            parts = label.split(":")[1].split(",")
            i, j = int(parts[0]), int(parts[1])
            e = coefs[idx] ** 2
            head_o2_energy[i] = head_o2_energy.get(i, 0) + e
            head_o2_energy[j] = head_o2_energy.get(j, 0) + e

    ranked_by_o2 = sorted(head_o2_energy.items(), key=lambda x: x[1], reverse=True)
    c6_heads = []
    for player_idx, energy_val in ranked_by_o2[:top_k]:
        layer, head = all_heads[player_idx]
        c6_heads.append({
            "head": f"L{layer}H{head}",
            "layer": layer,
            "head_idx": head,
            "o2_energy": float(energy_val),
        })

    print(f"\n=== C6 Epistatic Circuit (top {top_k} by order-2 energy) ===")
    for i, h in enumerate(c6_heads):
        print(f"  {i+1:2d}. {h['head']}: o2_energy = {h['o2_energy']:.6f}")

    # Also show top-15 C6
    c6_heads_15 = []
    for player_idx, energy_val in ranked_by_o2[:15]:
        layer, head = all_heads[player_idx]
        c6_heads_15.append({
            "head": f"L{layer}H{head}",
            "layer": layer,
            "head_idx": head,
            "o2_energy": float(energy_val),
        })
    print(f"\n=== C6 top-15 by order-2 energy ===")
    for i, h in enumerate(c6_heads_15):
        print(f"  {i+1:2d}. {h['head']}: o2_energy = {h['o2_energy']:.6f}")

    # Compare with known GT circuit (Hanna et al.)
    GT_KNOWN = {(1,0),(1,5),(4,4),(5,1),(5,5),(8,8),(8,11)}
    c5_set = set(discovered_heads)
    c6_set = {(h["layer"], h["head_idx"]) for h in c6_heads}

    overlaps = {
        "C5_vs_GT_known": len(c5_set & GT_KNOWN),
        "C6_vs_GT_known": len(c6_set & GT_KNOWN),
        "C5_vs_C6": len(c5_set & c6_set),
    }
    print(f"\n=== Overlaps (at k={top_k}) ===")
    for name, count in overlaps.items():
        print(f"  {name}: {count}/{top_k}")

    results = {
        "task": "greater_than",
        "method": "walsh_lasso_discovery",
        "max_order": 2,
        "alpha": float(alpha_used) if alpha_used else None,
        "r2": float(r2),
        "n_samples": int(masks.shape[0]),
        "n_players": int(n_players),
        "fit_time_seconds": float(fit_time),
        "c5_circuit_k7": discovered_circuit,
        "c5_heads_k7": [[int(h["layer"]), int(h["head_idx"])] for h in discovered_circuit],
        "c5_circuit_k15": head_rankings[:15],
        "c5_heads_k15": [[int(h["layer"]), int(h["head_idx"])] for h in head_rankings[:15]],
        "c6_circuit_k7": c6_heads,
        "c6_heads_k7": [[int(h["layer"]), int(h["head_idx"])] for h in c6_heads],
        "c6_circuit_k15": c6_heads_15,
        "c6_heads_k15": [[int(h["layer"]), int(h["head_idx"])] for h in c6_heads_15],
        "energy_spectrum": {str(k): float(v) for k, v in energy.items()},
        "top_interactions": top_interactions,
        "overlaps": {k: int(v) for k, v in overlaps.items()},
        "head_rankings_full": head_rankings,
    }

    out_path = "/results/walsh_gt_lasso_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else x)
    results_volume.commit()
    print(f"\nResults saved to {out_path}")


@app.local_entrypoint()
def main():
    fit_walsh.remote()
