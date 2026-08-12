"""Validate sparse Walsh recovery on existing exact coalition sweep data.

Takes exact sweeps (15-head IOI/RTI, 7-head GT) and subsamples at various
sample sizes. Recovers order-1 and order-2 Walsh coefficients via:
  1. Direct Monte Carlo estimation
  2. LASSO on the Walsh basis (order <= 2)
  3. OMP (Orthogonal Matching Pursuit)
Compares to exact WHT coefficients.

Phase 1 of PREREG_SPARSE_WALSH_RECOVERY.md.

Usage: cd epistatic-circuits && modal run scripts/modal_sparse_walsh_validation.py
"""
import modal

app = modal.App("sparse-walsh-validation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2",
        "tqdm==4.67.1",
    )
)

SWEEPS = {
    "/vol/c6": "c6-coalition-sweep",
    "/vol/ioi-resample": "ioi-resample-sweep",
    "/vol/rti-resample": "rti-resample-sweep",
    "/vol/rti-walsh": "rti-walsh-circuits-sweep",
}
out_vol = modal.Volume.from_name("sparse-walsh-validation-results", create_if_missing=True)
volumes = {p: modal.Volume.from_name(n) for p, n in SWEEPS.items()}
volumes["/out"] = out_vol

SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]
N_TRIALS = 10


def _popcount_lut():
    """Precompute popcount for 0..65535 (16-bit LUT)."""
    import numpy as np
    lut = np.zeros(65536, dtype=np.int8)
    for i in range(1, 65536):
        lut[i] = lut[i >> 1] + (i & 1)
    return lut


def _popcount_array(arr, lut):
    """Vectorized popcount for uint32 array using 16-bit LUT."""
    import numpy as np
    arr = np.asarray(arr, dtype=np.uint32)
    return (lut[arr & 0xFFFF].astype(np.int32) +
            lut[(arr >> 16) & 0xFFFF].astype(np.int32))


def walsh_basis_matrix_fast(coalition_ids, all_indices, lut):
    """Build Walsh basis matrix X[m, j] = (-1)^popcount(coalition[m] & index[j]).

    Vectorized: no Python loops over coalitions or indices.
    """
    import numpy as np
    coalition_ids = np.asarray(coalition_ids, dtype=np.uint32)
    all_indices = np.asarray(all_indices, dtype=np.uint32)
    bitwise_and = coalition_ids[:, None] & all_indices[None, :]
    bits = _popcount_array(bitwise_and.ravel(), lut).reshape(bitwise_and.shape)
    return np.where(bits % 2 == 0, 1.0, -1.0)


@app.function(image=image, volumes=volumes, timeout=3600, memory=8192)
def run():
    import json, time, itertools
    from pathlib import Path
    import numpy as np
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import OrthogonalMatchingPursuitCV
    from scipy.stats import spearmanr

    t0 = time.time()
    results = []
    lut = _popcount_lut()

    def ts():
        return f"[{time.time() - t0:.0f}s]"

    def wht(v):
        v = v.astype(np.float64).copy()
        h = 1
        while h < len(v):
            for i in range(0, len(v), h * 2):
                a = v[i:i+h].copy()
                b = v[i+h:i+2*h].copy()
                v[i:i+h] = a + b
                v[i+h:i+2*h] = a - b
            h *= 2
        return v / len(v)

    def extract_pair_walsh(walsh_coeffs, n):
        pairs = list(itertools.combinations(range(n), 2))
        return np.array([walsh_coeffs[(1 << i) | (1 << j)] for i, j in pairs]), pairs

    def extract_order1_walsh(walsh_coeffs, n):
        return np.array([walsh_coeffs[1 << i] for i in range(n)])

    npz_files = []
    for mount_path in SWEEPS:
        for p in sorted(Path(mount_path).rglob("*.npz")):
            try:
                d = np.load(str(p), allow_pickle=True)
            except Exception:
                continue
            if 'circuit_heads' not in d or 'n_players' not in d:
                continue
            n_players = int(d['n_players'])
            n_total = int(d.get('n_total', 2**n_players))
            n_completed = int(d.get('n_completed', d.get('n_coalitions_completed', n_total)))
            if n_completed < n_total:
                continue

            if 'target_logits' in d and 'foil_logits' in d:
                vals = d['target_logits'] - d['foil_logits']
            elif 'logit_diff' in d:
                vals = np.asarray(d['logit_diff'], dtype=np.float64)
            elif 'prob_diff' in d:
                vals = np.asarray(d['prob_diff'], dtype=np.float64)
            else:
                continue

            if 'coalition_indices' in d:
                idx = np.asarray(d['coalition_indices'])
                vals = vals[np.argsort(idx)]

            if vals.ndim == 2:
                vals_mean = vals.mean(axis=1)
            else:
                vals_mean = vals.astype(np.float64)

            if len(vals_mean) != 2**n_players:
                continue

            stem = p.stem.lower()
            abl = None
            if '_resample' in stem or stem.endswith('resample'):
                abl = 'resample'
            elif '_mean' in stem or stem.endswith('mean'):
                abl = 'mean'
            elif '_zero' in stem or stem.endswith('zero'):
                abl = 'zero'
            if abl is None:
                continue

            s = str(p).lower()
            if '/gt' in s and 'rti' not in stem:
                task = 'gt'
            elif '/rti' in s or stem.startswith('rti'):
                task = 'rti'
            elif '/ioi' in s or '/c6' in s or stem.startswith('ioi') or stem.startswith('c6'):
                task = 'ioi'
            else:
                continue

            npz_files.append({
                'path': str(p),
                'name': p.stem,
                'task': task,
                'abl': abl,
                'n': n_players,
                'vals_mean': vals_mean,
            })

    print(f"{ts()} Loaded {len(npz_files)} exact sweep files")

    for npz in npz_files:
        n = npz['n']
        N = 2**n
        f_exact = npz['vals_mean']

        w_exact = wht(f_exact)
        w2_exact, pairs = extract_pair_walsh(w_exact, n)
        w1_exact = extract_order1_walsh(w_exact, n)
        w_target = np.concatenate([w1_exact, w2_exact])
        n_coeffs = len(w_target)

        order1_indices = np.array([1 << i for i in range(n)], dtype=np.uint32)
        order2_indices = np.array([(1 << i) | (1 << j) for i, j in pairs], dtype=np.uint32)
        all_indices = np.concatenate([order1_indices, order2_indices])

        print(f"\n{ts()} === {npz['name']} ({npz['task']} {npz['abl']}, n={n}, N={N}) ===")
        print(f"  Exact order-2 energy: {np.sum(w2_exact**2):.6f}")
        print(f"  Exact order-1 energy: {np.sum(w1_exact**2):.6f}")
        print(f"  Total energy: {np.sum(w_exact**2):.6f}")

        for M in SAMPLE_SIZES:
            if M >= N:
                continue

            corrs_mc = []
            corrs_lasso = []
            corrs_omp = []
            rmses_mc = []
            rmses_lasso = []
            rmses_omp = []
            ranks_mc = []
            ranks_lasso = []
            ranks_omp = []
            top10_recall_mc = []
            top10_recall_lasso = []
            top10_recall_omp = []

            top10_exact = set(np.argsort(np.abs(w_target))[-10:])

            rng = np.random.default_rng(42)
            for trial in range(N_TRIALS):
                sampled = rng.choice(N, size=M, replace=False)
                f_sampled = f_exact[sampled]

                X_basis = walsh_basis_matrix_fast(sampled, all_indices, lut)

                # Method 1: Direct Monte Carlo estimation
                w_mc = np.mean(f_sampled[:, None] * X_basis, axis=0)

                # Method 2: LASSO
                try:
                    alpha_grid = np.logspace(-6, -1, 20)
                    lasso = LassoCV(alphas=alpha_grid, cv=5, max_iter=10000,
                                    fit_intercept=True)
                    lasso.fit(X_basis, f_sampled)
                    w_lasso = lasso.coef_
                except Exception:
                    w_lasso = w_mc.copy()

                # Method 3: OMP (CV-selected sparsity, no oracle k)
                try:
                    omp = OrthogonalMatchingPursuitCV(cv=5, max_iter=min(n_coeffs, M // 2))
                    omp.fit(X_basis, f_sampled)
                    w_omp = omp.coef_
                except Exception:
                    w_omp = w_mc.copy()

                def metrics(w_recovered, corrs, rmses, ranks, top10s):
                    r = float(np.corrcoef(w_target, w_recovered)[0, 1])
                    if np.isnan(r):
                        r = 0.0
                    corrs.append(r)
                    rmses.append(float(np.sqrt(np.mean((w_target - w_recovered)**2))))
                    rho = spearmanr(w_target, w_recovered).statistic
                    ranks.append(float(rho) if not np.isnan(rho) else 0.0)
                    top10_rec = set(np.argsort(np.abs(w_recovered))[-10:])
                    top10s.append(len(top10_exact & top10_rec) / 10)

                metrics(w_mc, corrs_mc, rmses_mc, ranks_mc, top10_recall_mc)
                metrics(w_lasso, corrs_lasso, rmses_lasso, ranks_lasso, top10_recall_lasso)
                metrics(w_omp, corrs_omp, rmses_omp, ranks_omp, top10_recall_omp)

            row = {
                'name': npz['name'],
                'task': npz['task'],
                'abl': npz['abl'],
                'n_heads': n,
                'n_coalitions_total': N,
                'n_coeffs': n_coeffs,
                'M': M,
                'compression': round(N / M, 1),
                'M_over_k': round(M / n_coeffs, 2),
                'mc_corr_mean': round(float(np.mean(corrs_mc)), 4),
                'mc_corr_std': round(float(np.std(corrs_mc)), 4),
                'lasso_corr_mean': round(float(np.mean(corrs_lasso)), 4),
                'lasso_corr_std': round(float(np.std(corrs_lasso)), 4),
                'omp_corr_mean': round(float(np.mean(corrs_omp)), 4),
                'omp_corr_std': round(float(np.std(corrs_omp)), 4),
                'mc_rank_mean': round(float(np.mean(ranks_mc)), 4),
                'mc_rank_std': round(float(np.std(ranks_mc)), 4),
                'lasso_rank_mean': round(float(np.mean(ranks_lasso)), 4),
                'lasso_rank_std': round(float(np.std(ranks_lasso)), 4),
                'omp_rank_mean': round(float(np.mean(ranks_omp)), 4),
                'omp_rank_std': round(float(np.std(ranks_omp)), 4),
                'mc_rmse_mean': round(float(np.mean(rmses_mc)), 6),
                'lasso_rmse_mean': round(float(np.mean(rmses_lasso)), 6),
                'omp_rmse_mean': round(float(np.mean(rmses_omp)), 6),
                'mc_top10_recall': round(float(np.mean(top10_recall_mc)), 3),
                'lasso_top10_recall': round(float(np.mean(top10_recall_lasso)), 3),
                'omp_top10_recall': round(float(np.mean(top10_recall_omp)), 3),
                'exact_rmse_scale': round(float(np.std(w_target)), 6),
            }
            results.append(row)

            print(f"  M={M:>5} (M/k={M/n_coeffs:>4.1f}) | "
                  f"MC r={np.mean(corrs_mc):.3f}+-{np.std(corrs_mc):.3f} "
                  f"LASSO r={np.mean(corrs_lasso):.3f}+-{np.std(corrs_lasso):.3f} "
                  f"OMP r={np.mean(corrs_omp):.3f}+-{np.std(corrs_omp):.3f} | "
                  f"top10: MC={np.mean(top10_recall_mc):.0%} "
                  f"LASSO={np.mean(top10_recall_lasso):.0%} "
                  f"OMP={np.mean(top10_recall_omp):.0%}")

        Path("/out/sparse_walsh_results.json").write_text(json.dumps(results, indent=1))
        out_vol.commit()

    print(f"\n{ts()} Done. {len(results)} rows saved.")
    Path("/out/sparse_walsh_results.json").write_text(json.dumps(results, indent=1))
    out_vol.commit()
    return results


@app.local_entrypoint()
def main():
    import json
    result = run.remote()
    print("\n=== SUMMARY ===")
    if result:
        for r in result:
            print(f"{r['name']:>45} M={r['M']:>5} (M/k={r['M_over_k']:>4.1f}) "
                  f"MC={r['mc_corr_mean']:.3f} LASSO={r['lasso_corr_mean']:.3f} "
                  f"OMP={r['omp_corr_mean']:.3f} "
                  f"top10: MC={r['mc_top10_recall']:.0%} "
                  f"LASSO={r['lasso_top10_recall']:.0%} "
                  f"OMP={r['omp_top10_recall']:.0%}")
