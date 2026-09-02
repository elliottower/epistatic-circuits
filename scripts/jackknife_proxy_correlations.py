"""Delete-one-head jackknife intervals for the proxy-vs-Walsh correlations.

Head pairs are dyadic: 190 pairs over 20 heads are not 190 independent
observations, because pairs sharing an endpoint are dependent. The weight-geometry
analysis already accounts for this with leave-both-heads-out CV; this applies the
same logic to the path-patching and EAP correlations, which were reported as point
estimates.

Reads only already-computed per-pair values. No model forward passes.
"""
import json
import math
from pathlib import Path

from scipy.stats import pearsonr, spearmanr, t as student_t

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "jackknife_proxy_correlations.json"


def correlations(pairs):
    w = [p["walsh_coeff"] for p in pairs]
    d = [p["direct_effect"] for p in pairs]
    return {
        "abs_spearman": spearmanr([abs(x) for x in w], [abs(y) for y in d]).statistic,
        "signed_pearson": pearsonr(w, d).statistic,
    }


def jackknife(pairs, heads):
    """Delete-one-head: drop every pair touching head h, recompute, then combine."""
    full = correlations(pairs)
    out = {}
    for stat, theta_hat in full.items():
        reps = []
        for h in heads:
            kept = [p for p in pairs if p["sender"] != h and p["receiver"] != h]
            if len(kept) > 3:
                reps.append(correlations(kept)[stat])
        n = len(reps)
        bar = sum(reps) / n
        se = math.sqrt((n - 1) / n * sum((r - bar) ** 2 for r in reps))
        tcrit = float(student_t.ppf(0.975, n - 1))
        out[stat] = {
            "point": theta_hat,
            "jackknife_se": se,
            "t_critical": tcrit,
            "ci95": [theta_hat - tcrit * se, theta_hat + tcrit * se],
            "n_heads_deleted": n,
            "min_over_deletions": min(reps),
            "max_over_deletions": max(reps),
        }
    return out


def main():
    src = ROOT / "results" / "phase2" / "path_patching_results.json"
    data = json.loads(src.read_text())
    pairs = [p for p in data["all_pairs"] if not p["same_layer"]]
    heads = sorted({p["sender"] for p in pairs} | {p["receiver"] for p in pairs})

    result = {
        "source": str(src.relative_to(ROOT)),
        "n_pairs_cross_layer": len(pairs),
        "n_heads": len(heads),
        "method": "delete-one-head jackknife; all pairs touching a head are removed together",
        "path_patching_vs_walsh": jackknife(pairs, heads),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"written: {OUT.relative_to(ROOT)}")
    for stat, v in result["path_patching_vs_walsh"].items():
        print(f"  {stat:<16} {v['point']:+.3f}  SE {v['jackknife_se']:.3f}  "
              f"95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
