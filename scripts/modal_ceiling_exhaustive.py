"""Exhaustive weight-only R^2 ceiling, every circuit x every ablation pair.

The weights are identical across ablation primitives. So if a head-pair
interaction W_ij were a function of the weights alone, W_ij would be identical
under zero, mean and resample ablation. However much it moves is variance no
weight-based predictor can explain, which upper-bounds achievable R^2.

This is the control PREREG_SUBSPACE_EPISTASIS.md names and does not compute.

Usage:  modal run --detach scripts/modal_ceiling_exhaustive.py
"""
import modal

app = modal.App("epistasis-ceiling-exhaustive")

VOLS = {
    "/vol/ioi_resample":  "ioi-resample-sweep",
    "/vol/rti_resample":  "rti-resample-sweep",
    "/vol/gt_resample":   "gt-resample-sweep",
    "/vol/acdc_resample": "acdc-resample-sweep",
    "/vol/c6":            "c6-coalition-sweep",
    "/vol/gt":            "gt-sweep-results",
    "/vol/rti_walsh":     "rti-walsh-circuits-sweep",
    "/vol/induction":     "induction-sweep-results",
    "/vol/rti_v5":        "rti-sweep-v5",
    "/vol/rti_multi":     "rti-multi-circuit-sweep-v4",
    "/vol/sva":           "sva-sweep-results",
    "/vol/gender":        "gender-sweep-results",
}
out_vol = modal.Volume.from_name("epistasis-ceiling-results", create_if_missing=True)
volumes = {p: modal.Volume.from_name(n) for p, n in VOLS.items()}
volumes["/out"] = out_vol

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2.0", "scipy>=1.10.0", "pandas>=2.1.0", "tqdm>=4.60.0",
                 "matplotlib>=3.7.0")
)


@app.function(image=image, volumes=volumes, timeout=86400, memory=32768)
def run():
    import json, re, itertools
    from pathlib import Path
    import numpy as np
    from scipy.stats import spearmanr, pearsonr

    def wht(v):
        v = v.astype(np.float64).copy(); h = 1
        while h < len(v):
            for i in range(0, len(v), h * 2):
                a = v[i:i+h].copy(); b = v[i+h:i+2*h].copy()
                v[i:i+h] = a + b; v[i+h:i+2*h] = a - b
            h *= 2
        return v / len(v)

    VALUE_KEYS = ("logit_diff", "prob_diff", "value", "values", "metric")

    def load_order2(path):
        """Returns (order2 coeffs, head set, n) or (None, reason)."""
        d = np.load(path, allow_pickle=True); k = list(d.keys())
        ld = None
        for vk in VALUE_KEYS:
            if vk in k: ld = d[vk]; break
        if ld is None and 'target_logits' in k and 'foil_logits' in k:
            ld = d['target_logits'] - d['foil_logits']
        if ld is None:
            return None, f"no value key; has {k}"
        ld = np.asarray(ld, dtype=np.float64)
        n = int(d['n_players'])
        if 'coalition_indices' in k:                 # explicit ordering present
            idx = np.asarray(d['coalition_indices'])
            if len(idx) != 2 ** n:
                return None, f"incomplete: {len(idx)}/{2**n} coalitions"
            ld = ld[np.argsort(idx)]
        if ld.ndim == 2: ld = ld.mean(1)             # average over prompts
        if len(ld) != 2 ** n:
            return None, f"len {len(ld)} != 2^{n}"
        heads = tuple(tuple(h) for h in d['circuit_heads'].tolist())
        w = wht(ld)
        pairs = list(itertools.combinations(range(n), 2))
        return (np.array([w[(1 << i) | (1 << j)] for i, j in pairs]), heads, n), None

    ABL = ("zero", "mean", "resample")

    def parse(p: Path):
        s = p.stem.replace("_coalition_values", "")
        abl = next((a for a in ABL if f"_{a}" in s or s.endswith(a)), None)
        if abl is None: return None
        circuit = s.replace(f"_{abl}", "").replace("_v2", "").strip("_")
        task = ("ioi" if "/ioi" in str(p) or "/c6" in str(p) or p.stem.startswith("ioi") or circuit.startswith("c6")
                else "rti" if "rti" in str(p).lower()
                else "gt" if "gt" in str(p).lower()
                else "induction" if "induction" in str(p).lower()
                else "sva" if "sva" in str(p).lower()
                else "gender" if "gender" in str(p).lower() else "unknown")
        circuit = re.sub(r"^(ioi|rti|gt|induction|sva|gender)_", "", circuit)
        return task, circuit, abl

    log = []
    def note(m):
        log.append(m); print(m, flush=True)
        Path("/out/ceiling_log.txt").write_text("\n".join(log)); out_vol.commit()

    found = {}
    for mnt in VOLS:
        for p in Path(mnt).rglob("*.npz"):
            info = parse(p)
            if info: found.setdefault(info[:2], {})[info[2]] = p
    note(f"discovered {sum(len(v) for v in found.values())} sweeps "
         f"across {len(found)} (task, circuit) groups")
    for mnt in VOLS:
        npzs = list(Path(mnt).rglob("*.npz"))
        note(f"  {mnt}: {len(npzs)} npz; unparsed: "
             f"{[q.name for q in npzs if parse(q) is None][:4]}")
    for k, v in sorted(found.items()):
        note(f"  group {k[0]}/{k[1]}: {sorted(v)}")

    TASKS = {"ioi", "rti", "gt"}
    rows, cache = [], {}
    for (task, circuit), byabl in sorted(found.items()):
        if task not in TASKS: continue
        if len(byabl) < 2:
            note(f"  SKIP {task}/{circuit}: only {list(byabl)}"); continue
        for a, b in itertools.combinations(sorted(byabl), 2):
            try:
                for x in (a, b):
                    if byabl[x] not in cache: cache[byabl[x]] = load_order2(byabl[x])
                (A, ra), (B, rb) = cache[byabl[a]], cache[byabl[b]]
                if A is None or B is None:
                    note(f"  SKIP {task}/{circuit} {a}~{b}: {ra or ''} {rb or ''}"); continue
                if A[1] != B[1]:
                    note(f"  SKIP {task}/{circuit} {a}~{b}: head sets differ ({len(A[1])} vs {len(B[1])})"); continue
                wa, wb = A[0], B[0]
                r = float(pearsonr(wa, wb)[0]); rho = float(spearmanr(wa, wb).statistic)
                sign = float(np.mean(np.sign(wa) == np.sign(wb)))
                rows.append({"task": task, "circuit": circuit, "abl_a": a, "abl_b": b,
                             "n_heads": A[2], "n_pairs": len(wa),
                             "pearson_r": r, "ceiling_linear_r2": r * r,
                             "spearman_rho": rho, "sign_agreement": sign})
                print(f"  {task:<10} {circuit:<22} {a:>8}~{b:<8} "
                      f"r={r:+.3f} ceilR2={r*r:.3f} rho={rho:+.3f} sign={sign:.1%}", flush=True)
            except Exception as e:
                note(f"  FAIL {task}/{circuit} {a}~{b}: {type(e).__name__}: {e}")
        cache.clear()
        Path("/out/ceiling_rows.json").write_text(json.dumps(rows, indent=1))
        out_vol.commit()

    if rows:
        cr = [x["ceiling_linear_r2"] for x in rows]
        sg = [x["sign_agreement"] for x in rows]
        summary = {"n_comparisons": len(rows),
                   "ceiling_linear_r2": {"median": float(np.median(cr)), "min": float(min(cr)), "max": float(max(cr))},
                   "sign_agreement": {"median": float(np.median(sg)), "min": float(min(sg)), "max": float(max(sg))},
                   "p4_threshold": 0.30,
                   "n_comparisons_with_ceiling_below_p4_threshold": int(sum(c < 0.30 for c in cr))}
        Path("/out/ceiling_summary.json").write_text(json.dumps(summary, indent=1))
        print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=1), flush=True)
    out_vol.commit()
    return rows


@app.local_entrypoint()
def main():
    run.remote()
