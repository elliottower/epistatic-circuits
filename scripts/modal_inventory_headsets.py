"""Inventory only: which (task, head-set) groups have which ablation primitives.

Reads circuit_heads and metadata from every sweep NPZ. Computes no epistasis and
touches no value data, so it cannot unblind anything.
"""
import modal
app = modal.App("inventory-headsets")
SWEEPS = {
    "/vol/ioi-resample": "ioi-resample-sweep", "/vol/rti-resample": "rti-resample-sweep",
    "/vol/gt-resample": "gt-resample-sweep",   "/vol/acdc-resample": "acdc-resample-sweep",
    "/vol/acdc-disc": "acdc-discovery-ioi-rti","/vol/c6": "c6-coalition-sweep",
    "/vol/gt": "gt-sweep-results",             "/vol/rti-walsh": "rti-walsh-circuits-sweep",
    "/vol/rti-v5": "rti-sweep-v5",             "/vol/rti-v3": "rti-coalition-sweep-v3",
    "/vol/rti-v2": "rti-coalition-sweep",      "/vol/induction": "induction-sweep-results",
}
out = modal.Volume.from_name("inventory-results", create_if_missing=True)
vols = {p: modal.Volume.from_name(n) for p, n in SWEEPS.items()}; vols["/out"] = out
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2.0")

@app.function(image=image, volumes=vols, timeout=3600, memory=8192)
def run():
    import json, re
    from pathlib import Path
    import numpy as np
    ABL = ("zero", "mean", "resample")
    recs = []
    for mnt in SWEEPS:
        for p in Path(mnt).rglob("*.npz"):
            try:
                d = np.load(p, allow_pickle=True); k = list(d.keys())
                if 'circuit_heads' not in k: continue
                heads = tuple(sorted(tuple(int(x) for x in h) for h in d['circuit_heads'].tolist()))
                s = p.stem.replace("_coalition_values", "")
                abl = next((a for a in ABL if f"_{a}" in s or s.endswith(a)), "UNKNOWN")
                vkey = next((v for v in ("logit_diff","prob_diff") if v in k), "other")
                nP = int(d['n_prompts']) if 'n_prompts' in k else -1
                n = int(d['n_players']) if 'n_players' in k else -1
                ncomp = int(d['n_coalitions_completed']) if 'n_coalitions_completed' in k else -1
                recs.append({"file": p.name, "vol": mnt, "abl": abl, "n_heads": len(heads),
                             "n_players": n, "value_key": vkey, "n_prompts": nP,
                             "n_completed": ncomp, "complete": (ncomp == 2**n) if ncomp > 0 else None,
                             "heads": [list(h) for h in heads]})
            except Exception as e:
                recs.append({"file": p.name, "vol": mnt, "error": f"{type(e).__name__}: {e}"})
    # group by identical head set
    groups = {}
    for r in recs:
        if "heads" not in r: continue
        key = json.dumps(r["heads"])
        groups.setdefault(key, {"n_heads": r["n_heads"], "arms": {}})
        groups[key]["arms"].setdefault(r["abl"], []).append(
            {"file": r["file"], "vol": r["vol"], "value_key": r["value_key"],
             "n_prompts": r["n_prompts"], "complete": r["complete"]})
    summary = []
    for key, g in groups.items():
        summary.append({"n_heads": g["n_heads"], "primitives": sorted(g["arms"]),
                        "n_primitives": len(g["arms"]),
                        "files": {a: [f["file"] for f in v] for a, v in g["arms"].items()},
                        "value_keys": sorted({f["value_key"] for v in g["arms"].values() for f in v}),
                        "n_prompts": sorted({f["n_prompts"] for v in g["arms"].values() for f in v}),
                        "all_complete": all(f["complete"] is not False for v in g["arms"].values() for f in v),
                        "heads": json.loads(key)})
    summary.sort(key=lambda x: (-x["n_primitives"], -x["n_heads"]))
    Path("/out/headset_inventory.json").write_text(json.dumps(summary, indent=1))
    out.commit()
    print(f"{len(recs)} sweeps -> {len(summary)} distinct head sets\n")
    for s in summary:
        flag = "" if s["all_complete"] else "  [INCOMPLETE]"
        vk = "/".join(s["value_keys"]); npr = ",".join(str(x) for x in s["n_prompts"])
        print(f"  {s['n_heads']:>3}h  {s['n_primitives']} arms {str(s['primitives']):<32} "
              f"key={vk:<10} prompts={npr:<10}{flag}", flush=True)
        for a, fs in sorted(s["files"].items()):
            print(f"        {a:<9} {', '.join(fs)}", flush=True)
    return summary

@app.local_entrypoint()
def main(): run.remote()
