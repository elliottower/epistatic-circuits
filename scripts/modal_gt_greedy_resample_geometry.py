"""Modal: compute weight geometry CV R^2 for GT greedy_sufficiency resample arm.

One missing row: 29 → 30. Loads GPT-2 weights, loads the npz from the
gt-resample-sweep volume, computes leave-both-heads-out CV R^2, saves result.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_gt_greedy_resample_geometry.py
"""
import modal

app = modal.App("gt-greedy-resample-geometry")

sweep_volume = modal.Volume.from_name("gt-resample-sweep")
out_volume = modal.Volume.from_name("primitive-geometry-results-v4", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-learn==1.5.2",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "typeguard==4.3.0",
    )
)

GREEDY_SUFFICIENCY_HEADS = [
    (0, 10), (5, 5), (5, 8), (6, 9), (7, 10), (8, 5), (9, 1),
]


@app.function(
    image=image,
    gpu="T4",
    timeout=86400,
    volumes={"/sweep": sweep_volume, "/out": out_volume},
)
def compute_geometry():
    import itertools
    import json
    import time

    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts()}] Loading GPT-2 weights")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", fold_ln=True, center_writing_weights=True, center_unembed=True)
    model.eval()

    W_OV, W_Q, W_K = {}, {}, {}
    for l in range(model.cfg.n_layers):
        W_OV[l], W_Q[l], W_K[l] = {}, {}, {}
        for h in range(model.cfg.n_heads):
            W_OV[l][h] = (model.W_V[l, h].detach().cpu().numpy()
                          @ model.W_O[l, h].detach().cpu().numpy())
            W_Q[l][h] = model.W_Q[l, h].detach().cpu().numpy()
            W_K[l][h] = model.W_K[l, h].detach().cpu().numpy()
    del model
    print(f"[{ts()}] Weights extracted")

    def _nrm(prod, a, b):
        d = np.linalg.norm(a, 'fro') * np.linalg.norm(b, 'fro')
        return float(np.linalg.norm(prod, 'fro') / d) if d > 0 else 0.0

    def feats(hi, hj):
        li, _hi = hi
        lj, _hj = hj
        ov_i, ov_j = W_OV[li][_hi], W_OV[lj][_hj]
        ov = _nrm(ov_i.T @ ov_j, ov_i, ov_j)
        if li < lj:
            q = _nrm(ov_i @ W_Q[lj][_hj], ov_i, W_Q[lj][_hj])
            k = _nrm(ov_i @ W_K[lj][_hj], ov_i, W_K[lj][_hj])
        elif lj < li:
            q = _nrm(ov_j @ W_Q[li][_hi], ov_j, W_Q[li][_hi])
            k = _nrm(ov_j @ W_K[li][_hi], ov_j, W_K[li][_hi])
        else:
            q = k = 0.0
        return [ov, q, k, abs(li - lj)]

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

    MIN_TRAIN_PAIRS = 10
    MIN_EVALUABLE_PAIRS = 10

    def cv_r2(X, y, pairs, feature_mask=None):
        y = np.asarray(y, float)
        Xf = X[:, feature_mask] if feature_mask is not None else X
        preds = np.full(len(y), np.nan)
        for t, (i, j) in enumerate(pairs):
            tr = [k for k, (a, b) in enumerate(pairs) if a not in (i, j) and b not in (i, j)]
            if len(tr) < MIN_TRAIN_PAIRS:
                continue
            preds[t] = LinearRegression().fit(Xf[tr], y[tr]).predict(Xf[t:t+1])[0]
        ok = ~np.isnan(preds)
        if ok.sum() < MIN_EVALUABLE_PAIRS:
            return None, int(ok.sum())
        yy, pp = y[ok], preds[ok]
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        if ss_tot <= 0:
            return None, int(ok.sum())
        return 1 - float(((yy - pp) ** 2).sum()) / ss_tot, int(ok.sum())

    def jackknife_se(X, y, pairs, n_heads, feature_mask=None):
        theta_hat, _ = cv_r2(X, y, pairs, feature_mask)
        if theta_hat is None:
            return None, None, None
        pseudovalues = []
        for drop in range(n_heads):
            sel = [t for t, (a, b) in enumerate(pairs) if a != drop and b != drop]
            if len(sel) < MIN_EVALUABLE_PAIRS:
                continue
            sub_pairs = [pairs[t] for t in sel]
            Xf = X[:, feature_mask] if feature_mask is not None else X
            theta_i, n_eval = cv_r2(X[sel], y[sel], sub_pairs, feature_mask)
            if theta_i is None:
                continue
            pseudovalues.append(n_heads * theta_hat - (n_heads - 1) * theta_i)
        if len(pseudovalues) < 3:
            return theta_hat, None, None
        pv = np.array(pseudovalues)
        se = float(np.sqrt(np.var(pv, ddof=1) / len(pv)))
        return theta_hat, float(np.mean(pv) - 1.96 * se), float(np.mean(pv) + 1.96 * se)

    # Load npz
    npz_path = "/sweep/gt_greedy_sufficiency_resample_coalition_values.npz"
    print(f"[{ts()}] Loading {npz_path}")
    d = np.load(npz_path)
    prob_diff = d['prob_diff']
    n_players = int(d['n_players'])
    n_prompts = int(d['n_prompts'])
    heads = [tuple(int(x) for x in hh) for hh in d['circuit_heads'].tolist()]
    print(f"[{ts()}] {n_players} heads, {prob_diff.shape[0]} coalitions, {n_prompts} prompts")

    # Walsh on mean across prompts
    mean_pd = prob_diff.mean(axis=1)
    w_mean = wht(mean_pd)
    pairs_list = list(itertools.combinations(range(n_players), 2))
    w_pairs = np.array([w_mean[(1 << i) | (1 << j)] for i, j in pairs_list])

    # Split-half
    pd_odd = prob_diff[:, 1::2].mean(axis=1)
    pd_even = prob_diff[:, 0::2].mean(axis=1)
    w_odd = wht(pd_odd)
    w_even = wht(pd_even)
    pairs_odd = np.array([w_odd[(1 << i) | (1 << j)] for i, j in pairs_list])
    pairs_even = np.array([w_even[(1 << i) | (1 << j)] for i, j in pairs_list])
    r = float(np.corrcoef(pairs_odd, pairs_even)[0, 1])
    reliability = 2 * r / (1 + r)

    # Features
    X = np.array([feats(heads[i], heads[j]) for i, j in pairs_list])
    full_mask = np.array([True, True, True, True])
    layer_dist_mask = np.array([False, False, False, True])

    # Regressions
    ins_full = float(LinearRegression().fit(X, w_pairs).score(X, w_pairs))
    ins_ld = float(LinearRegression().fit(X[:, layer_dist_mask], w_pairs).score(
        X[:, layer_dist_mask], w_pairs))
    cv_full, n_eval = cv_r2(X, w_pairs, pairs_list, full_mask)
    cv_ld, _ = cv_r2(X, w_pairs, pairs_list, layer_dist_mask)
    _, jk_lo, jk_hi = jackknife_se(X, w_pairs, pairs_list, n_players, full_mask)
    rho_ov = float(spearmanr(X[:, 0], np.abs(w_pairs)).statistic)

    row = {
        "task": "gt",
        "circuit_heads": [list(h) for h in heads],
        "primitive": "resample",
        "n_heads": n_players,
        "n_pairs": len(pairs_list),
        "n_evaluable_pairs": n_eval,
        "value_key": "prob_diff",
        "n_prompts": n_prompts,
        "confirmatory": False,
        "selection_bias_arm": False,
        "prompt_matched": True,
        "insample_r2_full": round(ins_full, 4),
        "insample_r2_layer_dist_only": round(ins_ld, 4),
        "cv_r2_full": round(cv_full, 4) if cv_full is not None else None,
        "cv_r2_layer_dist_only": round(cv_ld, 4) if cv_ld is not None else None,
        "geometry_increment": round(cv_full - cv_ld, 4) if cv_full is not None and cv_ld is not None else None,
        "jk_ci95_lo": round(jk_lo, 4) if jk_lo is not None else None,
        "jk_ci95_hi": round(jk_hi, 4) if jk_hi is not None else None,
        "spearman_ov_abs_w": round(rho_ov, 4),
        "split_half_reliability": round(reliability, 4),
        "filename": "gt_greedy_sufficiency_resample_coalition_values.npz",
    }

    print(f"\n[{ts()}] RESULT:")
    print(f"  insample_r2_full: {row['insample_r2_full']}")
    print(f"  cv_r2_full:       {row['cv_r2_full']}")
    print(f"  cv_r2_ld_only:    {row['cv_r2_layer_dist_only']}")
    print(f"  geometry_incr:    {row['geometry_increment']}")
    print(f"  jk_ci95:          [{row['jk_ci95_lo']}, {row['jk_ci95_hi']}]")
    print(f"  spearman_ov:      {row['spearman_ov_abs_w']}")
    print(f"  reliability:      {row['split_half_reliability']}")

    with open("/out/gt_greedy_sufficiency_resample_row.json", "w") as f:
        json.dump(row, f, indent=2)
    out_volume.commit()

    # Per-pair records
    pair_records = []
    for t, (i, j) in enumerate(pairs_list):
        pair_records.append({
            "task": "gt",
            "primitive": "resample",
            "n_heads": n_players,
            "head_i": list(heads[i]),
            "head_j": list(heads[j]),
            "ov_overlap": round(float(X[t, 0]), 6),
            "q_comp": round(float(X[t, 1]), 6),
            "k_comp": round(float(X[t, 2]), 6),
            "layer_dist": int(X[t, 3]),
            "walsh_coeff": round(float(w_pairs[t]), 6),
        })

    with open("/out/gt_greedy_sufficiency_resample_pairs.json", "w") as f:
        json.dump(pair_records, f, indent=2)
    out_volume.commit()

    print(f"\n[{ts()}] Done. Saved to primitive-geometry-results-v4 volume.")
    return row


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat()}] Computing geometry for GT greedy_sufficiency resample")
    result = compute_geometry.remote()
    print(f"\nCV R^2 full: {result['cv_r2_full']}")
    print(f"CV R^2 layer-dist only: {result['cv_r2_layer_dist_only']}")
    print(f"Geometry increment: {result['geometry_increment']}")
