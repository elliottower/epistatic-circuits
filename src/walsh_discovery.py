"""Fit LASSO-Walsh to random coalition samples and discover circuits.

Takes the output of modal_walsh_discovery.py (random coalition masks + logit_diffs
over all 144 GPT-2 heads) and fits sparse Walsh coefficients via LASSO.

Order-1 coefficients = per-head marginal importance (Shapley-like).
Order-2 coefficients = pairwise interaction strengths.

The top-15 heads by order-1 magnitude form the "Walsh-discovered circuit" (C5).

Usage:
    uv run python experiments_batch2/genetics/fit_walsh_discovery.py \
        --data experiments_batch2/genetics/walsh_discovery_144heads_mean_coalitions.npz \
        --max-order 2 \
        --out experiments_batch2/genetics/results_v7.1/walsh_discovery_results.json
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import comb


def build_walsh_basis(masks, max_order, n_players):
    """Build Walsh basis matrix for LASSO fitting.

    For masks in {0, 1}^n, Walsh basis function for subset S is:
        chi_S(x) = product_{i in S} (2*x_i - 1)

    Returns (n_samples, n_features) matrix and list of feature labels.
    """
    n_samples = masks.shape[0]
    x_pm = 2 * masks.astype(np.float64) - 1

    features = []
    labels = []

    labels.append("intercept")
    features.append(np.ones(n_samples))

    if max_order >= 1:
        for i in range(n_players):
            labels.append(f"o1:{i}")
            features.append(x_pm[:, i])

    if max_order >= 2:
        for i, j in combinations(range(n_players), 2):
            labels.append(f"o2:{i},{j}")
            features.append(x_pm[:, i] * x_pm[:, j])

    if max_order >= 3:
        for combo in combinations(range(n_players), 3):
            labels.append(f"o3:{','.join(map(str, combo))}")
            prod = np.ones(n_samples)
            for idx in combo:
                prod *= x_pm[:, idx]
            features.append(prod)

    X = np.column_stack(features)
    return X, labels


def fit_lasso_walsh(masks, values, max_order, n_players, alpha=None):
    """Fit sparse Walsh coefficients via LASSO."""
    from sklearn.linear_model import LassoCV, Lasso

    print(f"Building Walsh basis (max_order={max_order}, n_players={n_players})...")
    n_features_expected = sum(int(comb(n_players, k)) for k in range(max_order + 1))
    print(f"  Expected features: {n_features_expected:,}")

    X, labels = build_walsh_basis(masks, max_order, n_players)
    print(f"  Basis matrix: {X.shape[0]} samples x {X.shape[1]} features")
    assert X.shape[1] == n_features_expected, f"{X.shape[1]} != {n_features_expected}"

    y = values - values.mean()
    X_centered = X - X.mean(axis=0)

    if alpha is not None:
        print(f"Fitting LASSO (alpha={alpha})...")
        model = Lasso(alpha=alpha, max_iter=10000, tol=1e-5)
        model.fit(X_centered, y)
    else:
        print("Fitting LassoCV (5-fold, auto-alpha)...")
        model = LassoCV(cv=5, max_iter=10000, tol=1e-5, n_alphas=50, n_jobs=-1)
        model.fit(X_centered, y)
        print(f"  Best alpha: {model.alpha_:.6f}")

    coefs = model.coef_
    n_nonzero = np.sum(np.abs(coefs) > 1e-10)
    r2 = model.score(X_centered, y)
    print(f"  R² = {r2:.4f}, {n_nonzero} non-zero coefficients")

    return coefs, labels, r2, getattr(model, 'alpha_', alpha)


def extract_head_rankings(coefs, labels, n_players, all_heads):
    """Extract per-head importance from order-1 Walsh coefficients."""
    order1_coefs = {}
    for idx, label in enumerate(labels):
        if label.startswith("o1:"):
            player_idx = int(label.split(":")[1])
            order1_coefs[player_idx] = coefs[idx]

    ranked = sorted(order1_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    results = []
    for player_idx, coef in ranked:
        layer, head = all_heads[player_idx]
        results.append({
            "head": f"L{layer}H{head}",
            "layer": layer,
            "head_idx": head,
            "walsh_o1": float(coef),
            "abs_walsh_o1": float(abs(coef)),
        })
    return results


def extract_top_interactions(coefs, labels, all_heads, top_k=50):
    """Extract strongest pairwise interactions from order-2 Walsh coefficients."""
    interactions = []
    for idx, label in enumerate(labels):
        if label.startswith("o2:"):
            parts = label.split(":")[1].split(",")
            i, j = int(parts[0]), int(parts[1])
            interactions.append({
                "heads": [f"L{all_heads[i][0]}H{all_heads[i][1]}",
                          f"L{all_heads[j][0]}H{all_heads[j][1]}"],
                "player_indices": [i, j],
                "walsh_o2": float(coefs[idx]),
                "abs_walsh_o2": float(abs(coefs[idx])),
            })

    interactions.sort(key=lambda x: x["abs_walsh_o2"], reverse=True)
    return interactions[:top_k]


def compute_energy_spectrum(coefs, labels, max_order):
    """Compute fraction of variance at each interaction order."""
    order_energy = {k: 0.0 for k in range(max_order + 1)}
    for idx, label in enumerate(labels):
        if label == "intercept":
            order_energy[0] += coefs[idx] ** 2
        elif label.startswith("o1:"):
            order_energy[1] += coefs[idx] ** 2
        elif label.startswith("o2:"):
            order_energy[2] += coefs[idx] ** 2
        elif label.startswith("o3:"):
            order_energy[3] += coefs[idx] ** 2

    total = sum(order_energy.values())
    if total == 0:
        return {k: 0.0 for k in range(max_order + 1)}
    return {k: v / total for k, v in order_energy.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to coalition samples npz")
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=None, help="LASSO alpha (auto if omitted)")
    parser.add_argument("--top-k", type=int, default=15, help="Number of heads for discovered circuit")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    masks = data["masks"]
    n_players = int(data["n_players"])
    all_heads = [tuple(h) for h in data["circuit_heads"]]

    if "mean_logit_diffs" in data:
        values = data["mean_logit_diffs"]
    else:
        logit_diffs = data["target_logits"] - data["foil_logits"]
        values = logit_diffs.mean(axis=1)

    print(f"  {masks.shape[0]} samples, {n_players} players")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}], "
          f"mean={values.mean():.3f}, std={values.std():.3f}")

    t0 = time.time()
    coefs, labels, r2, alpha_used = fit_lasso_walsh(
        masks, values, args.max_order, n_players, alpha=args.alpha
    )
    fit_time = time.time() - t0
    print(f"Fitting took {fit_time:.1f}s")

    head_rankings = extract_head_rankings(coefs, labels, n_players, all_heads)
    top_interactions = extract_top_interactions(coefs, labels, all_heads, top_k=50)
    energy = compute_energy_spectrum(coefs, labels, args.max_order)

    discovered_circuit = [h for h in head_rankings[:args.top_k]]
    discovered_heads = [(h["layer"], h["head_idx"]) for h in discovered_circuit]

    print(f"\n=== Walsh-Discovered Circuit (top {args.top_k}) ===")
    for i, h in enumerate(discovered_circuit):
        print(f"  {i+1:2d}. {h['head']}: walsh_o1 = {h['walsh_o1']:+.4f}")

    print(f"\n=== Energy Spectrum ===")
    for order, frac in sorted(energy.items()):
        print(f"  Order {order}: {frac*100:.2f}%")

    print(f"\n=== Top 10 Pairwise Interactions ===")
    for ix in top_interactions[:10]:
        print(f"  {ix['heads'][0]} x {ix['heads'][1]}: {ix['walsh_o2']:+.4f}")

    C1 = {(0,1),(0,5),(0,10),(4,11),(5,1),(5,5),(5,8),(5,9),(6,1),(6,9),(7,2),(7,9),(7,10),(10,2),(10,7)}
    C2 = {(0,1),(0,8),(0,9),(0,11),(1,3),(4,7),(5,10),(6,9),(7,9),(8,3),(8,6),(8,10),(10,7),(11,0),(11,10)}
    C3 = {(0,1),(3,0),(2,2),(4,11),(5,5),(6,9),(7,3),(7,9),(8,6),(8,10),(9,9),(9,6),(10,0),(10,7),(11,10)}
    IC15 = {(10,7),(4,0),(0,1),(4,11),(9,4),(5,6),(8,9),(8,4),(0,5),(11,7),(0,9),(11,8),(10,3),(1,3),(0,3)}

    discovered_set = set(discovered_heads)
    overlaps = {
        "C1_weight": len(discovered_set & C1),
        "C2_eap": len(discovered_set & C2),
        "C3_canonical": len(discovered_set & C3),
        "IC15": len(discovered_set & IC15),
    }
    print(f"\n=== Overlap with other circuits ===")
    for name, count in overlaps.items():
        print(f"  {name}: {count}/{args.top_k}")

    def jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "method": "walsh_discovery",
        "max_order": int(args.max_order),
        "alpha": float(alpha_used) if alpha_used else None,
        "r2": float(r2),
        "n_samples": int(masks.shape[0]),
        "n_players": int(n_players),
        "fit_time_seconds": float(fit_time),
        "discovered_circuit": discovered_circuit,
        "discovered_heads": [[int(h["layer"]), int(h["head_idx"])] for h in discovered_circuit],
        "energy_spectrum": {str(k): float(v) for k, v in energy.items()},
        "top_interactions": top_interactions,
        "overlaps": {k: int(v) for k, v in overlaps.items()},
        "head_rankings_full": head_rankings,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=jsonify)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
