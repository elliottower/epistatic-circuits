"""EAP-IG (MIB reference implementation) vs Walsh interaction, with jackknife CIs.

The paper reports Spearman rho = 0.58 between pairwise Walsh coefficients and
EAP edge scores, but no script in this repository computes it. This recomputes
the comparison from MIB's own EAP-IG implementation (results/eapig_ioi_vanilla_gpt2.json,
produced by factorization-circuits/experiments/batch3_causal_discovery/
modal_eapig_ioi_vanilla_gpt2.py) and puts a delete-one-head jackknife interval
on it, since head pairs sharing an endpoint are dependent.

Edge scores are aggregated over the receiver's q/k/v slots and over direction,
because the Walsh coefficient for a pair is undirected.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import pearsonr, spearmanr, t as student_t

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "jackknife_eapig_correlation.json"


def load():
    eap = json.loads((ROOT / "results" / "eapig_ioi_vanilla_gpt2.json").read_text())
    edges = eap["head_edge_scores"]

    def pair_score(a, b):
        """Undirected: sum |score| over qkv slots and both directions."""
        tot, found = 0.0, False
        for key in (f"{a}->{b}", f"{b}->{a}"):
            if key in edges:
                found = True
                tot += sum(abs(v) for v in edges[key].values())
        return tot if found else None

    rows = list(csv.DictReader(open(ROOT / "results" / "subspace_epistasis_pairs.csv")))
    circuits = defaultdict(list)
    for r in rows:
        if r["task"] != "ioi":
            continue
        s = pair_score(r["head_i"], r["head_j"])
        if s is None:
            continue
        circuits[r["circuit"]].append(
            {"i": r["head_i"], "j": r["head_j"],
             "walsh": float(r["walsh_coeff"]), "eap": s}
        )
    return circuits


def corr(pairs):
    w = [abs(p["walsh"]) for p in pairs]
    e = [p["eap"] for p in pairs]
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
        tcrit = float(student_t.ppf(0.975, n - 1))
        out[stat] = {"point": theta, "jackknife_se": se, "t_critical": tcrit,
                     "ci95": [theta - tcrit * se, theta + tcrit * se],
                     "n_heads_deleted": n}
    return out


def main():
    circuits = load()
    result = {
        "published_point_estimate": {"abs_spearman": 0.58, "abs_pearson": 0.73},
        "eap_source": "MIB EAP-IG-inputs, stock GPT-2 small, mib-bench/ioi, 200 examples, ig_steps=5",
        "walsh_source": "results/subspace_epistasis_pairs.csv",
        "aggregation": "sum of |edge score| over q/k/v and both directions",
        "method": "delete-one-head jackknife; interval uses t(n_heads-1), not the normal quantile",
        "circuits": {n: {"n_pairs": len(p), **jackknife(p)} for n, p in sorted(circuits.items())},
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"written: {OUT.relative_to(ROOT)}\n")
    print(f"  {'circuit':<22}{'n':>5} {'spearman':>10} {'SE':>7}   {'95% CI':>19}  {'pearson':>8}")
    for name, v in result["circuits"].items():
        s, pe = v["abs_spearman"], v["abs_pearson"]
        print(f"  {name:<22}{v['n_pairs']:>5} {s['point']:>+10.3f} {s['jackknife_se']:>7.3f}   "
              f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  {pe['point']:>+8.3f}")


if __name__ == "__main__":
    main()
