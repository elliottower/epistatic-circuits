"""Does weight geometry predict epistasis under some ablation primitives better
than others? — v2, all four fatal blockers resolved.

Fixes from v1:
  B1. Circularity check: C6 was discovered under MEAN ablation, confirmed in
      scripts/modal_walsh_discovery.py. Mean arm has a selection effect; zero and
      resample are clean.
  B2. Grouping: keys on (task, frozenset(circuit_heads)) instead of filename.
      Guards on >= 2 arms within same task+headset. Verifies head ordering
      matches across arms.
  B3. P1 decision rule: computed from per-circuit paired differences, not
      eyeballed from arm medians.
  B4. Jackknife: delete-one-head jackknife replaces the broken bootstrap
      (set() discarded multiplicity, evaluating ~9 heads instead of 15).

Additional:
  - Layer-distance-only baseline R^2 alongside the full model.
  - Split-half reliability per arm (Spearman-Brown on odd/even prompts).
  - Per-pair records saved for P3.
  - Documents which arm carries selection bias (mean, for c6).

Prereg: docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md (frozen, amendments A1-A3).

Usage:  cd epistatic-circuits && modal run --detach scripts/modal_primitive_specific_geometry_v2.py
"""
import modal

app = modal.App("primitive-geometry-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "numpy==1.26.4", "scipy==1.13.1", "tqdm==4.67.1",
        "transformer-lens==2.17.0", "transformers==4.51.3",
        "typeguard==4.3.0", "matplotlib==3.9.4", "scikit-learn==1.5.2",
    )
)

SWEEPS = {
    "/vol/ioi-resample": "ioi-resample-sweep",
    "/vol/rti-resample": "rti-resample-sweep",
    "/vol/gt-resample":  "gt-resample-sweep",
    "/vol/c6":           "c6-coalition-sweep",
    "/vol/gt":           "gt-sweep-results",
    "/vol/rti-walsh":    "rti-walsh-circuits-sweep",
    "/vol/rti-v5":       "rti-sweep-v5",
}
out_vol = modal.Volume.from_name("primitive-geometry-results-v2", create_if_missing=True)
volumes = {p: modal.Volume.from_name(n) for p, n in SWEEPS.items()}
volumes["/out"] = out_vol

SEED = 20260807
MIN_TRAIN_PAIRS = 10
MIN_EVALUABLE_PAIRS = 10
CONFIRMATORY_MIN_HEADS = 15
VKEYS = ("logit_diff", "prob_diff", "value", "values", "metric")
ABLS = ("zero", "mean", "resample")


@app.function(image=image, volumes=volumes, timeout=86400, memory=32768)
def run():
    import json, itertools, time
    from pathlib import Path
    from collections import defaultdict
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    import transformer_lens

    t0 = time.time()
    log_lines = []

    def note(m):
        log_lines.append(m)
        print(m, flush=True)
        Path("/out/run_log.txt").write_text("\n".join(log_lines))
        out_vol.commit()

    def ts():
        return f"[{time.time() - t0:.0f}s]"

    # ---- 1. Load GPT-2 weights ------------------------------------------------
    note(f"{ts()} Loading GPT-2 with fold_ln=True")
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", fold_ln=True, center_writing_weights=True, center_unembed=True)
    model.eval()
    nl, nh = model.cfg.n_layers, model.cfg.n_heads

    W_OV, W_Q, W_K = {}, {}, {}
    for l in range(nl):
        W_OV[l], W_Q[l], W_K[l] = {}, {}, {}
        for h in range(nh):
            W_OV[l][h] = (model.W_V[l, h].detach().cpu().numpy()
                          @ model.W_O[l, h].detach().cpu().numpy())
            W_Q[l][h] = model.W_Q[l, h].detach().cpu().numpy()
            W_K[l][h] = model.W_K[l, h].detach().cpu().numpy()
    del model
    note(f"{ts()} Weight matrices extracted for {nl}L x {nh}H")

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

    # ---- 2. Walsh-Hadamard transform ------------------------------------------
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

    # ---- 3. Scan all volumes (FIX B2: metadata first, group by heads) ---------
    note(f"{ts()} Scanning volumes for NPZ files")
    entries = []
    for mount_path in SWEEPS:
        for p in Path(mount_path).rglob("*.npz"):
            try:
                d = np.load(str(p), allow_pickle=True)
            except Exception as e:
                note(f"  SKIP {p}: load error {e}")
                continue
            if 'circuit_heads' not in d:
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
            if '/gt' in s and 'rti' not in stem and 'ioi' not in stem and 'induction' not in stem:
                task = 'gt'
            elif '/rti' in s or stem.startswith('rti'):
                task = 'rti'
            elif '/ioi' in s or '/c6' in s or stem.startswith('ioi') or stem.startswith('c6'):
                task = 'ioi'
            elif 'induction' in s:
                task = 'induction'
            else:
                continue

            heads = tuple(tuple(int(x) for x in hh) for hh in d['circuit_heads'].tolist())
            vkey = next((v for v in VKEYS if v in d), None)
            if vkey is None and 'target_logits' in d and 'foil_logits' in d:
                vkey = 'target_minus_foil'
            if vkey is None:
                continue

            n_prompts = int(d['n_prompts']) if 'n_prompts' in d else -1
            n_completed = int(d['n_completed']) if 'n_completed' in d else -1
            n_total = int(d.get('n_total', 0))
            if n_completed >= 0 and n_total > 0 and n_completed < n_total:
                note(f"  SKIP {p.name}: incomplete {n_completed}/{n_total}")
                continue

            entries.append({
                'path': str(p), 'task': task, 'abl': abl,
                'heads': heads, 'heads_set': frozenset(heads),
                'n_heads': len(heads), 'vkey': vkey, 'n_prompts': n_prompts,
                'filename': p.name,
            })

    note(f"{ts()} Found {len(entries)} valid NPZ files")

    # Group by (task, frozenset(heads))
    groups = defaultdict(dict)
    for e in entries:
        key = (e['task'], e['heads_set'])
        if e['abl'] in groups[key]:
            existing = groups[key][e['abl']]
            note(f"  DUPLICATE {e['task']}/{e['abl']}: {e['filename']} vs {existing['filename']} — keeping first")
            continue
        groups[key][e['abl']] = e

    note(f"{ts()} {len(groups)} unique (task, headset) groups")

    # Filter to comparable: >= 2 arms, uniform n_prompts
    comparable = {}
    for (task, hs), arms in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0][0])):
        if len(arms) < 2:
            continue
        prompt_counts = set(e['n_prompts'] for e in arms.values())
        if len(prompt_counts) > 1:
            note(f"  NON-COMPARABLE {task}/{len(list(hs))}h {sorted(arms.keys())}: "
                 f"prompt mismatch {sorted(prompt_counts)}")
            continue
        head_orderings = set(e['heads'] for e in arms.values())
        if len(head_orderings) > 1:
            note(f"  NON-COMPARABLE {task}/{len(list(hs))}h: head ordering differs")
            continue
        comparable[(task, hs)] = arms

    note(f"{ts()} {len(comparable)} comparable groups with >= 2 arms")
    for (task, hs), arms in comparable.items():
        n_h = len(list(hs))
        prompt_count = list(arms.values())[0]['n_prompts']
        note(f"  {task} {n_h}h {sorted(arms.keys())} @ {prompt_count} prompts")

    # ---- 4. Load data, compute features, fit regressions ----------------------

    def load_walsh_2d(entry):
        """Load NPZ and return (mean_walsh_pairs, walsh_pairs_odd, walsh_pairs_even, pairs, n)."""
        d = np.load(entry['path'], allow_pickle=True)
        vkey = entry['vkey']
        if vkey == 'target_minus_foil':
            ld = d['target_logits'] - d['foil_logits']
        else:
            ld = np.asarray(d[vkey], dtype=np.float64)
        n = int(d['n_players'])
        if 'coalition_indices' in d:
            idx = np.asarray(d['coalition_indices'])
            ld = ld[np.argsort(idx)]
        if ld.ndim == 1:
            w_mean = wht(ld)
            pairs, pr = extract_pair_walsh(w_mean, n)
            return pairs, None, None, pr, n
        ld_mean = ld.mean(axis=1)
        w_mean = wht(ld_mean)
        pairs_mean, pr = extract_pair_walsh(w_mean, n)
        ld_odd = ld[:, 1::2].mean(axis=1)
        ld_even = ld[:, 0::2].mean(axis=1)
        w_odd = wht(ld_odd)
        w_even = wht(ld_even)
        pairs_odd, _ = extract_pair_walsh(w_odd, n)
        pairs_even, _ = extract_pair_walsh(w_even, n)
        return pairs_mean, pairs_odd, pairs_even, pr, n

    def cv_r2(X, y, pairs, feature_mask=None):
        """Leave-both-heads-out CV R^2. feature_mask selects columns of X."""
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
        """Delete-one-head jackknife SE for CV R^2 (FIX B4)."""
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

    def split_half_reliability(pairs_odd, pairs_even):
        """Spearman-Brown reliability from odd/even prompt split."""
        if pairs_odd is None or pairs_even is None:
            return None
        r = float(np.corrcoef(pairs_odd, pairs_even)[0, 1])
        return 2 * r / (1 + r)

    note(f"\n{ts()} === FITTING REGRESSIONS ===")
    rows = []
    pair_records = []
    circuit_results = {}

    for (task, hs), arms in sorted(comparable.items(), key=lambda x: x[0]):
        heads = list(arms.values())[0]['heads']
        n_heads = len(heads)
        pairs_list = list(itertools.combinations(range(n_heads), 2))
        X = np.array([feats(heads[i], heads[j]) for i, j in pairs_list])
        layer_dist_mask = np.array([False, False, False, True])
        full_mask = np.array([True, True, True, True])

        circuit_key = f"{task}_{n_heads}h_{sorted(list(hs))[:2]}"
        note(f"\n{ts()} Circuit: {task} {n_heads}h ({len(pairs_list)} pairs)")

        arm_results = {}
        for abl in ['zero', 'mean', 'resample']:
            if abl not in arms:
                continue
            entry = arms[abl]
            w_pairs, w_odd, w_even, pairs, n = load_walsh_2d(entry)

            ins_full = float(LinearRegression().fit(X, w_pairs).score(X, w_pairs))
            ins_ld = float(LinearRegression().fit(X[:, layer_dist_mask], w_pairs).score(
                X[:, layer_dist_mask], w_pairs))

            cv_full, n_eval = cv_r2(X, w_pairs, pairs_list, full_mask)
            cv_ld, _ = cv_r2(X, w_pairs, pairs_list, layer_dist_mask)

            _, jk_lo, jk_hi = jackknife_se(X, w_pairs, pairs_list, n_heads, full_mask)

            rho_ov = float(spearmanr(X[:, 0], np.abs(w_pairs)).statistic)
            reliability = split_half_reliability(w_odd, w_even)

            row = {
                "task": task,
                "circuit_heads": [list(h) for h in heads],
                "primitive": abl,
                "n_heads": n_heads,
                "n_pairs": len(pairs_list),
                "n_evaluable_pairs": n_eval,
                "value_key": entry['vkey'],
                "n_prompts": entry['n_prompts'],
                "confirmatory": n_heads >= CONFIRMATORY_MIN_HEADS,
                "selection_bias_arm": abl == "mean" and task == "ioi",
                "insample_r2_full": round(ins_full, 4),
                "insample_r2_layer_dist_only": round(ins_ld, 4),
                "cv_r2_full": round(cv_full, 4) if cv_full is not None else None,
                "cv_r2_layer_dist_only": round(cv_ld, 4) if cv_ld is not None else None,
                "geometry_increment": round(cv_full - cv_ld, 4) if cv_full is not None and cv_ld is not None else None,
                "jk_ci95_lo": round(jk_lo, 4) if jk_lo is not None else None,
                "jk_ci95_hi": round(jk_hi, 4) if jk_hi is not None else None,
                "spearman_ov_abs_w": round(rho_ov, 4),
                "split_half_reliability": round(reliability, 4) if reliability is not None else None,
                "filename": entry['filename'],
            }
            rows.append(row)
            arm_results[abl] = row

            note(f"  {abl:<9} in={ins_full:.3f} cv={cv_full if cv_full is None else round(cv_full, 3)} "
                 f"ld_only={cv_ld if cv_ld is None else round(cv_ld, 3)} "
                 f"JK=[{jk_lo if jk_lo is None else round(jk_lo, 3)},{jk_hi if jk_hi is None else round(jk_hi, 3)}] "
                 f"rho_ov={rho_ov:+.3f} rel={reliability if reliability is None else round(reliability, 3)}")

            for t, (i, j) in enumerate(pairs_list):
                pair_records.append({
                    "task": task,
                    "primitive": abl,
                    "n_heads": n_heads,
                    "head_i": list(heads[i]),
                    "head_j": list(heads[j]),
                    "ov_overlap": round(float(X[t, 0]), 6),
                    "q_comp": round(float(X[t, 1]), 6),
                    "k_comp": round(float(X[t, 2]), 6),
                    "layer_dist": int(X[t, 3]),
                    "walsh_coeff": round(float(w_pairs[t]), 6),
                })

        circuit_results[(task, hs)] = arm_results

        Path("/out/primitive_geometry_rows_v2.json").write_text(json.dumps(rows, indent=1))
        out_vol.commit()

    # ---- 5. Save per-pair records for P3 --------------------------------------
    note(f"\n{ts()} Saving {len(pair_records)} per-pair records")
    Path("/out/primitive_geometry_pairs_v2.json").write_text(json.dumps(pair_records, indent=1))
    out_vol.commit()

    # ---- 6. Score P1 (FIX B3: paired per-circuit rule) -------------------------
    note(f"\n{ts()} === P1 SCORING ===")

    p1_circuits = []
    for (task, hs), arm_results in circuit_results.items():
        if len(list(hs)) < CONFIRMATORY_MIN_HEADS:
            note(f"  DESCRIPTIVE: {task} {len(list(hs))}h (below {CONFIRMATORY_MIN_HEADS}h threshold)")
            continue
        if 'zero' not in arm_results:
            continue
        zero_cv = arm_results['zero'].get('cv_r2_full')
        if zero_cv is None:
            continue
        for other_abl in ['mean', 'resample']:
            if other_abl not in arm_results:
                continue
            other_cv = arm_results[other_abl].get('cv_r2_full')
            if other_cv is None:
                continue
            diff = other_cv - zero_cv
            p1_circuits.append({
                'task': task,
                'n_heads': len(list(hs)),
                'comparison': f'{other_abl} - zero',
                'diff': round(diff, 4),
                'passes_0.10': diff >= 0.10,
                'zero_cv_r2': round(zero_cv, 4),
                f'{other_abl}_cv_r2': round(other_cv, 4),
                'selection_bias_note': 'mean arm has selection bias' if task == 'ioi' and other_abl == 'mean' else None,
            })
            note(f"  {task} {len(list(hs))}h: {other_abl} - zero = {diff:+.4f} "
                 f"({'PASS' if diff >= 0.10 else 'FAIL'} 0.10 threshold)")

    n_eligible = len([c for c in p1_circuits if c['passes_0.10']])
    n_total_comparisons = len(p1_circuits)
    p1_pass = n_eligible >= n_total_comparisons / 2 if n_total_comparisons > 0 else False

    p1_result = {
        "verdict": "PASS" if p1_pass else "FAIL",
        "n_eligible_passes": n_eligible,
        "n_total_comparisons": n_total_comparisons,
        "threshold": "at least half of eligible 15-head circuits by >= 0.10",
        "note": ("P1 as registered is dead: only 1 eligible 15-head circuit "
                 "(need half). This is descriptive.") if n_total_comparisons <= 2 else None,
        "comparisons": p1_circuits,
    }
    note(f"\n  P1 verdict: {p1_result['verdict']} ({n_eligible}/{n_total_comparisons})")
    if p1_result.get('note'):
        note(f"  NOTE: {p1_result['note']}")

    # ---- 7. Score P2, P3, P4 --------------------------------------------------
    note(f"\n{ts()} === P2/P3/P4 SCORING ===")

    by_prim_cv = defaultdict(list)
    for r in rows:
        if r['cv_r2_full'] is not None:
            by_prim_cv[r['primitive']].append(r['cv_r2_full'])

    p2_result = {}
    for prim, vals in sorted(by_prim_cv.items()):
        med = float(np.median(vals))
        p2_result[prim] = {"median_cv_r2": round(med, 4), "n": len(vals), "below_0.20": med < 0.20}
        note(f"  P2 {prim}: median CV-R^2 = {med:.4f} ({'PASS' if med < 0.20 else 'FAIL'} < 0.20)")

    all_below = all(v['below_0.20'] for v in p2_result.values())
    note(f"  P2 overall: {'PASS' if all_below else 'FAIL'} (all arms below 0.20)")

    # P3: close-layer/far-layer split
    note(f"\n  P3: close-layer vs far-layer split")
    p3_result = {}
    for prim in ['zero', 'mean', 'resample']:
        close_pairs = [(r['ov_overlap'], abs(r['walsh_coeff']))
                       for r in pair_records if r['primitive'] == prim and r['layer_dist'] <= 2]
        far_pairs = [(r['ov_overlap'], abs(r['walsh_coeff']))
                     for r in pair_records if r['primitive'] == prim and r['layer_dist'] > 3]
        if len(close_pairs) >= 5 and len(far_pairs) >= 5:
            close_ov = [p[0] for p in close_pairs]
            close_w = [p[1] for p in close_pairs]
            far_ov = [p[0] for p in far_pairs]
            far_w = [p[1] for p in far_pairs]
            rho_close = float(spearmanr(close_ov, close_w).statistic)
            rho_far = float(spearmanr(far_ov, far_w).statistic)
            p3_result[prim] = {
                "rho_close": round(rho_close, 4),
                "rho_far": round(rho_far, 4),
                "close_n": len(close_pairs),
                "far_n": len(far_pairs),
                "close_exceeds_far": rho_close > rho_far,
            }
            note(f"    {prim}: close rho={rho_close:.3f} (n={len(close_pairs)}) "
                 f"far rho={rho_far:.3f} (n={len(far_pairs)}) "
                 f"{'close > far' if rho_close > rho_far else 'far >= close'}")

    # P4: CV-R^2 below in-sample
    zero_rows = [r for r in rows if r['primitive'] == 'zero']
    p4_result = None
    if zero_rows:
        for r in zero_rows:
            if r['cv_r2_full'] is not None:
                diff = r['insample_r2_full'] - r['cv_r2_full']
                p4_result = {
                    "insample": r['insample_r2_full'],
                    "cv": r['cv_r2_full'],
                    "diff": round(diff, 4),
                    "passes_0.03": diff >= 0.03,
                    "task": r['task'],
                }
                note(f"  P4 {r['task']}: in-sample={r['insample_r2_full']:.4f} "
                     f"cv={r['cv_r2_full']:.4f} diff={diff:.4f} "
                     f"({'PASS' if diff >= 0.03 else 'FAIL'} >= 0.03)")

    # ---- 8. Summary -----------------------------------------------------------
    summary = {
        "n_comparable_groups": len(comparable),
        "n_rows": len(rows),
        "n_pair_records": len(pair_records),
        "P1": p1_result,
        "P2": {"by_primitive": p2_result, "all_below_0.20": all_below},
        "P3": p3_result,
        "P4": p4_result,
        "cv_r2_by_primitive": {k: {"median": round(float(np.median(v)), 4),
                                    "values": [round(x, 4) for x in v],
                                    "n": len(v)}
                                for k, v in sorted(by_prim_cv.items())},
        "blocker_1_resolved": "C6 discovered under MEAN ablation (modal_walsh_discovery.py). "
                              "Mean arm has selection effect. Zero and resample are clean.",
    }

    note(f"\n{ts()} === FINAL SUMMARY ===")
    note(json.dumps(summary, indent=2))

    Path("/out/primitive_geometry_summary_v2.json").write_text(json.dumps(summary, indent=2))
    Path("/out/primitive_geometry_rows_v2.json").write_text(json.dumps(rows, indent=1))
    Path("/out/primitive_geometry_pairs_v2.json").write_text(json.dumps(pair_records, indent=1))
    out_vol.commit()

    note(f"\n{ts()} Done. All results saved to /out/")
    return summary


@app.local_entrypoint()
def main():
    import json
    result = run.remote()
    print("\n=== SUMMARY ===")
    print(json.dumps(result, indent=2))
