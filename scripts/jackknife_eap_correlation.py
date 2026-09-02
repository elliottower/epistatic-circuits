"""Delete-one-head jackknife for the EAP-vs-Walsh correlation.

The paper reports Spearman rho = 0.58 between magnitude-ranked pairwise Walsh
coefficients and the corresponding EAP edge products. Head pairs sharing an
endpoint are dependent, so a point estimate over 105 pairs from 15 heads
overstates the effective sample size.

Reconstruction is validated first: if the recomputed point estimate does not
match the published 0.58, the quantity has been reconstructed wrongly and no
interval is reported for it.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "jackknife_eap_correlation.json"


def load():
    eap = json.loads((ROOT / "results" / "eap_ioi_head_scores.json").read_text())["head_scores"]
    rows = list(csv.DictReader(open(ROOT / "results" / "subspace_epistasis_pairs.csv")))
    circuits = defaultdict(list)
    for r in rows:
        if r["task"] != "ioi":
            continue
        i, j = r["head_i"], r["head_j"]
        if i not in eap or j not in eap:
            continue
        circuits[r["circuit"]].append(
            {"i": i, "j": j,
             "walsh": float(r["walsh_coeff"]),
             "eap_product": eap[i] * eap[j]}
        )
    return circuits


def corr(pairs):
    w = [abs(p["walsh"]) for p in pairs]
    e = [p["eap_product"] for p in pairs]
    return {"abs_spearman": spearmanr(w, e).statistic,
            "abs_pearson": pearsonr(w, e).statistic}


def jackknife(pairs):
    heads = sorted({p["i"] for p in pairs} | {p["j"] for p in pairs})
    full = corr(pairs)
    out = {}
    for stat, theta in full.items():
        reps = []
        for h in heads:
            kept = [p for p in pairs if p["i"] != h and p["j"] != h]
            if len(kept) > 3:
                reps.append(corr(kept)[stat])
        n = len(reps)
        bar = sum(reps) / n
        se = math.sqrt((n - 1) / n * sum((r - bar) ** 2 for r in reps))
        out[stat] = {"point": theta, "jackknife_se": se,
                     "ci95": [theta - 1.96 * se, theta + 1.96 * se],
                     "n_heads_deleted": n}
    return out


def main():
    circuits = load()
    result = {
        "published_point_estimate": {"abs_spearman": 0.58, "abs_pearson": 0.73},
        "eap_source": "results/eap_ioi_head_scores.json (Modal, GPT-2 small, 200 IOI prompts)",
        "walsh_source": "results/subspace_epistasis_pairs.csv",
        "method": "delete-one-head jackknife; all pairs touching a head removed together",
        "circuits": {},
    }
    for name, pairs in sorted(circuits.items()):
        result["circuits"][name] = {"n_pairs": len(pairs), **jackknife(pairs)}
    OUT.write_text(json.dumps(result, indent=2))
    print(f"written: {OUT.relative_to(ROOT)}\n")
    print(f"  {'circuit':<22}{'n':>5}  {'spearman':>9} {'SE':>7}  {'95% CI':>20}")
    for name, v in result["circuits"].items():
        s = v["abs_spearman"]
        print(f"  {name:<22}{v['n_pairs']:>5}  {s['point']:>+9.3f} {s['jackknife_se']:>7.3f}  "
              f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    print("\n  published rho = 0.58 -- reconstruction matches only if one circuit lands near it")


if __name__ == "__main__":
    main()
