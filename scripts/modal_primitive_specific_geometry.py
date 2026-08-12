"""Does weight geometry predict epistasis under some ablation primitives better
than others?

Prereg: docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md (frozen before any mean- or
resample-ablation regression was fitted).

Reuses the feature definitions from modal_subspace_epistasis_analysis.py verbatim
(OV overlap, Q/K composition, layer distance) so any difference between arms is
attributable to the ablation primitive rather than to the model.

Usage:  modal run --detach scripts/modal_primitive_specific_geometry.py
"""
import modal

app = modal.App("primitive-specific-geometry")

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
out_vol = modal.Volume.from_name("primitive-geometry-results", create_if_missing=True)
volumes = {p: modal.Volume.from_name(n) for p, n in SWEEPS.items()}
volumes["/out"] = out_vol

N_BOOT = 1000
SEED = 20260807
MIN_TRAIN_PAIRS = 10        # a leave-both-heads-out fit needs a usable train set
MIN_EVALUABLE_PAIRS = 10    # abort condition 2
CONFIRMATORY_MIN_HEADS = 15 # GT (7 heads) is descriptive only


@app.function(image=image, volumes=volumes, timeout=86400, memory=32768)
def run():
    import json, itertools, re
    from pathlib import Path
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    import transformer_lens

    log = []
    def note(m):
        log.append(m); print(m, flush=True)
        Path("/out/run_log.txt").write_text("\n".join(log)); out_vol.commit()

    # ---- weights ------------------------------------------------------------
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
    note(f"GPT-2 loaded, fold_ln=True, {nl}L x {nh}H")

    def _nrm(prod, a, b):
        d = np.linalg.norm(a, 'fro') * np.linalg.norm(b, 'fro')
        return float(np.linalg.norm(prod, 'fro') / d) if d > 0 else 0.0

    def feats(hi, hj):
        li, _hi = hi; lj, _hj = hj
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

    # ---- coalition data -----------------------------------------------------
    def wht(v):
        v = v.astype(np.float64).copy(); h = 1
        while h < len(v):
            for i in range(0, len(v), h * 2):
                a = v[i:i+h].copy(); b = v[i+h:i+2*h].copy()
                v[i:i+h] = a + b; v[i+h:i+2*h] = a - b
            h *= 2
        return v / len(v)

    VKEYS = ("logit_diff", "prob_diff", "value", "values", "metric")
    ABL = ("zero", "mean", "resample")

    def load(p):
        d = np.load(p, allow_pickle=True); k = list(d.keys())
        ld = next((d[v] for v in VKEYS if v in k), None)
        if ld is None and 'target_logits' in k and 'foil_logits' in k:
            ld = d['target_logits'] - d['foil_logits']
        if ld is None: return None, f"no value key ({k})"
        vkey = next((v for v in VKEYS if v in k), "target_minus_foil")
        nprompts = int(d['n_prompts']) if 'n_prompts' in k else (ld.shape[1] if ld.ndim == 2 else -1)
        ld = np.asarray(ld, float); n = int(d['n_players'])
        if 'coalition_indices' in k:
            idx = np.asarray(d['coalition_indices'])
            if len(idx) != 2 ** n: return None, f"incomplete {len(idx)}/{2**n}"
            ld = ld[np.argsort(idx)]
        if ld.ndim == 2: ld = ld.mean(1)
        if len(ld) != 2 ** n: return None, f"len {len(ld)} != 2^{n}"
        heads = [tuple(int(x) for x in hh) for hh in d['circuit_heads'].tolist()]
        w = wht(ld)
        pr = list(itertools.combinations(range(n), 2))
        return (np.array([w[(1 << i) | (1 << j)] for i, j in pr]), heads, pr, vkey, nprompts), None

    found = {}
    for mnt in SWEEPS:
        for p in Path(mnt).rglob("*.npz"):
            s = p.stem.replace("_coalition_values", "")
            abl = next((a for a in ABL if f"_{a}" in s or s.endswith(a)), None)
            if abl is None: continue
            circ = re.sub(r"^(ioi|rti|gt)_", "", s.replace(f"_{abl}", "").replace("_v2", "").strip("_"))
            task = ("gt" if "/gt" in str(p) else "rti" if "rti" in str(p).lower()
                    else "ioi" if ("/ioi" in str(p) or "/c6" in str(p)) else "other")
            if task == "other": continue
            found.setdefault((task, circ), {})[abl] = p
    note(f"{sum(len(v) for v in found.values())} sweeps in {len(found)} groups")

    # ---- fit ----------------------------------------------------------------
    def cv_r2(X, y, pairs, n_heads):
        """Leave-both-heads-out R^2.

        For each pair (i,j) the model is trained ONLY on pairs (a,b) with
        a not in {i,j} and b not in {i,j}, so no head is shared between train
        and test. GroupKFold on the first head of each pair does NOT give this:
        head 7 lands in group 3 via (3,7) and group 5 via (5,7), so it can be
        in train for one of its pairs while another is held out. That leak is
        what this replaces (amendment A1).
        """
        y = np.asarray(y, float)
        preds = np.full(len(y), np.nan)
        for t, (i, j) in enumerate(pairs):
            tr = [k for k, (a, b) in enumerate(pairs)
                  if a not in (i, j) and b not in (i, j)]
            if len(tr) < MIN_TRAIN_PAIRS: continue
            preds[t] = LinearRegression().fit(X[tr], y[tr]).predict(X[t:t+1])[0]
        ok = ~np.isnan(preds)
        if ok.sum() < MIN_EVALUABLE_PAIRS: return None, int(ok.sum())
        yy, pp = y[ok], preds[ok]
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        if ss_tot <= 0: return None, int(ok.sum())
        return 1 - float(((yy - pp) ** 2).sum()) / ss_tot, int(ok.sum())

    rows = []
    for (task, circ), byabl in sorted(found.items()):
        base = None
        for abl, path in sorted(byabl.items()):
            res, why = load(path)
            if res is None: note(f"  SKIP {task}/{circ}/{abl}: {why}"); continue
            w, heads, pr, vkey, nprompts = res
            if base is None: base = heads
            elif heads != base:                       # abort condition 1
                note(f"  SKIP {task}/{circ}/{abl}: head set differs"); continue
            X = np.array([feats(heads[i], heads[j]) for i, j in pr])
            ins = LinearRegression().fit(X, w).score(X, w)
            cv, n_eval = cv_r2(X, w, pr, len(heads))
            rho = float(spearmanr(X[:, 0], w).statistic)
            # head-level cluster bootstrap: resample HEADS, keep pairs whose
            # BOTH endpoints survive. Resampling pairs would ignore the dyadic
            # dependence this is meant to account for.
            rng = np.random.default_rng(SEED); boots = []
            uh = list(range(len(heads)))
            for _ in range(N_BOOT):
                keep = set(rng.choice(uh, len(uh), replace=True).tolist())
                sel = [t for t, (a, b) in enumerate(pr) if a in keep and b in keep]
                if len(sel) < MIN_EVALUABLE_PAIRS or len(keep) < 4: continue
                sub = [pr[t] for t in sel]
                v, _ = cv_r2(X[sel], w[sel], sub, len(keep))
                if v is not None: boots.append(v)
            lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) > 50 else (None, None)
            rows.append({"task": task, "circuit": circ, "primitive": abl,
                         "n_heads": len(heads), "n_pairs": len(pr),
                         "n_evaluable_pairs": n_eval, "value_key": vkey,
                         "n_prompts": nprompts,
                         "confirmatory": len(heads) >= CONFIRMATORY_MIN_HEADS,
                         "insample_r2": float(ins), "cv_r2": cv,
                         "cv_r2_ci95": [lo, hi], "spearman_ov": rho})
            note(f"  {task:<5}{circ[:20]:<21}{abl:<9} heads={len(heads):>3} "
                 f"insample={ins:.3f} cv={cv if cv is None else round(cv,3)} "
                 f"CI=[{lo if lo is None else round(lo,3)},{hi if hi is None else round(hi,3)}] rho_ov={rho:+.3f}")
            Path("/out/primitive_geometry_rows.json").write_text(json.dumps(rows, indent=1))
            out_vol.commit()

    conf = [r for r in rows if r["confirmatory"] and r["cv_r2"] is not None]
    by_prim = {}
    for r in conf: by_prim.setdefault(r["primitive"], []).append(r["cv_r2"])
    summary = {"n_rows": len(rows), "n_confirmatory": len(conf),
               "cv_r2_by_primitive": {k: {"median": float(np.median(v)), "n": len(v)}
                                      for k, v in sorted(by_prim.items())}}
    Path("/out/primitive_geometry_summary.json").write_text(json.dumps(summary, indent=1))
    note("\n=== SUMMARY ===\n" + json.dumps(summary, indent=1))
    out_vol.commit()
    return rows


@app.local_entrypoint()
def main():
    run.remote()
