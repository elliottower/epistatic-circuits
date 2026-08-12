"""Trace the Pareto curve: minimum M for reliable sparse Walsh recovery.

Subsamples from the 2000 Phase 2 coalitions at fine-grained M values.
No GPU needed — pure numpy/sklearn.

Usage: cd epistatic-circuits && python3.12 scripts/pareto_sparse_walsh.py
"""
import json
import itertools
import numpy as np
from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV
from scipy.stats import spearmanr

N_HEADS = 20
N_TRIALS = 20

M_VALUES = [
    20, 30, 40, 50, 60, 70, 80, 90, 100,
    120, 140, 160, 180, 200,
    250, 300, 400, 500, 750, 1000, 1500, 2000,
]


def _popcount_lut():
    lut = np.zeros(65536, dtype=np.int8)
    for i in range(1, 65536):
        lut[i] = lut[i >> 1] + (i & 1)
    return lut


def _popcount_array(arr, lut):
    arr = np.asarray(arr, dtype=np.uint32)
    return (lut[arr & 0xFFFF].astype(np.int32) +
            lut[(arr >> 16) & 0xFFFF].astype(np.int32))


def walsh_basis_matrix_fast(coalition_ids, all_indices, lut):
    coalition_ids = np.asarray(coalition_ids, dtype=np.uint32)
    all_indices = np.asarray(all_indices, dtype=np.uint32)
    bitwise_and = coalition_ids[:, None] & all_indices[None, :]
    bits = _popcount_array(bitwise_and.ravel(), lut).reshape(bitwise_and.shape)
    return np.where(bits % 2 == 0, 1.0, -1.0)


def main():
    d = np.load("results/phase2/phase2_coalition_checkpoint.npz", allow_pickle=True)
    coalition_ids = d["coalition_indices"]
    coalition_vals = d["coalition_values"]
    n = N_HEADS

    pairs = list(itertools.combinations(range(n), 2))
    order1_indices = np.array([1 << i for i in range(n)], dtype=np.uint32)
    order2_indices = np.array([(1 << i) | (1 << j) for i, j in pairs], dtype=np.uint32)
    all_indices = np.concatenate([order1_indices, order2_indices])
    k = len(all_indices)
    print(f"n={n}, k={k} (order-1: {n}, order-2: {len(pairs)})")

    lut = _popcount_lut()

    X_full = walsh_basis_matrix_fast(coalition_ids, all_indices, lut)

    lasso_full = LassoCV(alphas=np.logspace(-6, -1, 20), cv=5, max_iter=10000, fit_intercept=True)
    lasso_full.fit(X_full, coalition_vals)
    w_reference = lasso_full.coef_
    print(f"Reference recovery from M=2000: {np.count_nonzero(w_reference)} nonzero coefficients")

    top10_ref = set(np.argsort(np.abs(w_reference))[-10:])

    results = []

    for M in M_VALUES:
        if M > len(coalition_ids):
            continue

        rs_lasso, rs_omp, rs_mc = [], [], []
        t10_lasso, t10_omp, t10_mc = [], [], []
        ranks_lasso, ranks_omp = [], []
        nnz_lasso, nnz_omp = [], []

        rng = np.random.default_rng(42)

        for trial in range(N_TRIALS):
            idx = rng.choice(len(coalition_ids), size=M, replace=False)
            c_ids = coalition_ids[idx]
            c_vals = coalition_vals[idx]

            X = walsh_basis_matrix_fast(c_ids, all_indices, lut)

            w_mc = np.mean(c_vals[:, None] * X, axis=0)

            try:
                lasso = LassoCV(alphas=np.logspace(-6, -1, 20), cv=5, max_iter=10000, fit_intercept=True)
                lasso.fit(X, c_vals)
                w_lasso = lasso.coef_
            except Exception:
                w_lasso = w_mc.copy()

            try:
                omp = OrthogonalMatchingPursuitCV(cv=5, max_iter=min(k, M // 2))
                omp.fit(X, c_vals)
                w_omp = omp.coef_
            except Exception:
                w_omp = w_mc.copy()

            for w, rs, t10s, rnks, nnzs in [
                (w_lasso, rs_lasso, t10_lasso, ranks_lasso, nnz_lasso),
                (w_omp, rs_omp, t10_omp, ranks_omp, nnz_omp),
                (w_mc, rs_mc, t10_mc, None, None),
            ]:
                r = float(np.corrcoef(w_reference, w)[0, 1])
                if np.isnan(r):
                    r = 0.0
                rs.append(r)
                top10_rec = set(np.argsort(np.abs(w))[-10:])
                t10s.append(len(top10_ref & top10_rec) / 10)
                if rnks is not None:
                    rho = spearmanr(w_reference, w).statistic
                    rnks.append(float(rho) if not np.isnan(rho) else 0.0)
                    nnzs.append(int(np.count_nonzero(w)))

        row = {
            "M": M,
            "M_over_k": round(M / k, 3),
            "lasso_r_mean": round(float(np.mean(rs_lasso)), 4),
            "lasso_r_std": round(float(np.std(rs_lasso)), 4),
            "lasso_r_min": round(float(np.min(rs_lasso)), 4),
            "lasso_top10_mean": round(float(np.mean(t10_lasso)), 3),
            "lasso_rank_mean": round(float(np.mean(ranks_lasso)), 4),
            "lasso_nnz_mean": round(float(np.mean(nnz_lasso)), 1),
            "omp_r_mean": round(float(np.mean(rs_omp)), 4),
            "omp_r_std": round(float(np.std(rs_omp)), 4),
            "omp_r_min": round(float(np.min(rs_omp)), 4),
            "omp_top10_mean": round(float(np.mean(t10_omp)), 3),
            "omp_rank_mean": round(float(np.mean(ranks_omp)), 4),
            "omp_nnz_mean": round(float(np.mean(nnz_omp)), 1),
            "mc_r_mean": round(float(np.mean(rs_mc)), 4),
            "mc_r_std": round(float(np.std(rs_mc)), 4),
            "mc_top10_mean": round(float(np.mean(t10_mc)), 3),
        }
        results.append(row)

        print(f"M={M:>5} (M/k={M/k:.2f}) | "
              f"LASSO r={np.mean(rs_lasso):.4f}+-{np.std(rs_lasso):.4f} "
              f"[min={np.min(rs_lasso):.3f}] top10={np.mean(t10_lasso):.0%} "
              f"nnz={np.mean(nnz_lasso):.0f} | "
              f"OMP r={np.mean(rs_omp):.4f}+-{np.std(rs_omp):.4f} "
              f"[min={np.min(rs_omp):.3f}] top10={np.mean(t10_omp):.0%}")

    with open("results/phase2/pareto_curve.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved {len(results)} rows to results/phase2/pareto_curve.json")


if __name__ == "__main__":
    main()
