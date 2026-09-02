"""EAP vs Walsh interaction on the 20-head IOI circuit, with jackknife intervals.

Walsh coefficients: exhaustive-sweep order-2 coefficients for the 20-head IOI
circuit under mean ablation (phase2_all_walsh_coefficients.json, 190 pairs).
EAP: MIB's reference implementation on stock GPT-2 small over mib-bench/ioi,
both EAP-IG-inputs and plain EAP (results/eapig_ioi_vanilla_gpt2.json, produced by
factorization-circuits/experiments/batch3_causal_discovery/modal_eapig_ioi_vanilla_gpt2.py).

Edge scores are aggregated to an undirected head-pair quantity by summing
|score| over the receiver's q/k/v slots and over both directions, because a
Walsh coefficient for a pair is undirected.

Head pairs sharing an endpoint are dependent, so intervals come from a
delete-one-head jackknife using t on the number of heads, not the normal quantile.
"""
import json
import math
from pathlib import Path

from scipy.stats import pearsonr, spearmanr, t as student_t

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "eap_walsh_correlation.json"


def pair_score(edges, a, b):
    """Undirected: sum |score| over q/k/v slots and both directions."""
    total, found = 0.0, False
    for key in (f"{a}->{b}", f"{b}->{a}"):
        if key in edges:
            found = True
            total += sum(abs(v) for v in edges[key].values())
    return total if found else None


def stats(pairs):
    w = [p[0] for p in pairs]
    e = [p[1] for p in pairs]
    return {"spearman": spearmanr(w, e).statistic, "pearson": pearsonr(w, e).statistic}


def jackknife(pairs):
    heads = sorted({p[2] for p in pairs} | {p[3] for p in pairs})
    full = stats(pairs)
    out = {}
    for name, theta in full.items():
        reps = []
        for h in heads:
            kept = [p for p in pairs if p[2] != h and p[3] != h]
            if len(kept) > 3:
                reps.append(stats(kept)[name])
        n = len(reps)
        bar = sum(reps) / n
        se = math.sqrt((n - 1) / n * sum((r - bar) ** 2 for r in reps))
        tcrit = float(student_t.ppf(0.975, n - 1))
        entry = {"point": float(theta), "jackknife_se": se, "t_critical": tcrit,
                 "n_heads": n,
                 "leave_one_head_out_range": [min(reps), max(reps)]}
        if name == "pearson":
            # Wald on r can leave [-1, 1]; do it on Fisher z and transform back
            zs = [math.atanh(max(-0.999999, min(0.999999, r))) for r in reps]
            zbar = sum(zs) / n
            zse = math.sqrt((n - 1) / n * sum((z - zbar) ** 2 for z in zs))
            ztheta = math.atanh(max(-0.999999, min(0.999999, theta)))
            entry["jackknife_se_fisher_z"] = zse
            entry["ci95"] = [math.tanh(ztheta - tcrit * zse),
                             math.tanh(ztheta + tcrit * zse)]
            entry["interval_scale"] = "Fisher z, back-transformed"
        else:
            entry["ci95"] = [theta - tcrit * se, theta + tcrit * se]
            entry["interval_scale"] = "direct"
        out[name] = entry
    return out


def main():
    walsh = json.loads((ROOT / "data" / "phase2_all_walsh_coefficients.json").read_text())
    order2 = {k: v["coeff"] for k, v in walsh.items() if v["order"] == 2}
    variants = json.loads((ROOT / "results" / "eapig_ioi_vanilla_gpt2.json").read_text())["variants"]

    result = {
        "circuit": "20-head IOI circuit, GPT-2 small",
        "walsh": {"source": "data/phase2_all_walsh_coefficients.json",
                  "ablation": "mean", "n_order2_coefficients": len(order2)},
        "eap": {"source": "results/eapig_ioi_vanilla_gpt2.json",
                "implementation": "MIB EAP-IG (MIB-circuit-track/EAP-IG)",
                "dataset": "mib-bench/ioi, 200 examples", "metric": "logit_diff"},
        "aggregation": "sum of |edge score| over q/k/v slots and both directions",
        "interval": "delete-one-head jackknife, t(n_heads-1)",
        "variants": {},
    }
    for label, edges in variants.items():
        pairs = []
        for key, coeff in order2.items():
            a, b = key.split("-")
            s = pair_score(edges, a, b)
            if s is not None:
                pairs.append((abs(coeff), s, a, b))
        result["variants"][label] = {
            "n_pairs": len(pairs),
            "n_pairs_available": len(order2),
            "n_excluded_no_edge": len(order2) - len(pairs),
            **jackknife(pairs),
        }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"written: {OUT.relative_to(ROOT)}\n")
    for label, v in result["variants"].items():
        print(f"  {label}  ({v['n_pairs']}/{v['n_pairs_available']} pairs, "
              f"{v['n_excluded_no_edge']} with no admissible edge)")
        for stat in ("spearman", "pearson"):
            s = v[stat]
            print(f"    {stat:<9} {s['point']:+.3f}  SE {s['jackknife_se']:.3f}  "
                  f"95% CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
        print()


def _run():
    main()
    paired_difference()


def paired_difference():
    """Delta rho = rho(|W|,|EAP|) - rho(|W|,|PP|), on the same pairs and the same
    Walsh coefficients, with a delete-one-head jackknife interval on the difference.

    Overlapping marginal intervals do not settle whether two correlations differ;
    the paired difference does, because both proxies are scored against the same target.
    """
    walsh = json.loads((ROOT / "data" / "phase2_all_walsh_coefficients.json").read_text())
    order2 = {k: v["coeff"] for k, v in walsh.items() if v["order"] == 2}
    variants = json.loads((ROOT / "results" / "eapig_ioi_vanilla_gpt2.json").read_text())["variants"]
    pp = json.loads((ROOT / "results" / "phase2" / "path_patching_results.json").read_text())

    pp_by_pair = {}
    for p in pp["all_pairs"]:
        pp_by_pair[frozenset((p["sender"], p["receiver"]))] = abs(p["direct_effect"])

    out = {}
    for label, edges in variants.items():
        rows = []
        for key, coeff in order2.items():
            a, b = key.split("-")
            e = pair_score(edges, a, b)
            d = pp_by_pair.get(frozenset((a, b)))
            if e is None or d is None:
                continue
            rows.append((abs(coeff), e, d, a, b))

        def delta(rs):
            w = [r[0] for r in rs]
            return (spearmanr(w, [r[1] for r in rs]).statistic
                    - spearmanr(w, [r[2] for r in rs]).statistic)

        theta = delta(rows)
        heads = sorted({r[3] for r in rows} | {r[4] for r in rows})
        reps = [delta([r for r in rows if r[3] != h and r[4] != h]) for h in heads]
        n = len(reps)
        bar = sum(reps) / n
        se = math.sqrt((n - 1) / n * sum((r - bar) ** 2 for r in reps))
        tcrit = float(student_t.ppf(0.975, n - 1))
        out[label] = {
            "n_pairs": len(rows),
            "rho_eap": spearmanr([r[0] for r in rows], [r[1] for r in rows]).statistic,
            "rho_pp": spearmanr([r[0] for r in rows], [r[2] for r in rows]).statistic,
            "delta": theta, "jackknife_se": se, "t_critical": tcrit,
            "ci95": [theta - tcrit * se, theta + tcrit * se],
            "excludes_zero": bool((theta - tcrit * se) * (theta + tcrit * se) > 0),
        }
    path = ROOT / "results" / "eap_vs_pp_paired_difference.json"
    path.write_text(json.dumps({
        "comparison": "both proxies scored against the same Walsh coefficients, same pairs",
        "transformation": "|Walsh| vs |EAP| and |Walsh| vs |path-patch|, magnitude throughout",
        "variants": out}, indent=2))
    print(f"written: {path.relative_to(ROOT)}\n")
    for label, v in out.items():
        print(f"  {label}  (n={v['n_pairs']})")
        print(f"    rho(EAP) {v['rho_eap']:+.3f}   rho(PP) {v['rho_pp']:+.3f}")
        print(f"    delta    {v['delta']:+.3f}  SE {v['jackknife_se']:.3f}  "
              f"95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
              f"{'EXCLUDES zero' if v['excludes_zero'] else 'includes zero'}\n")


if __name__ == "__main__":
    _run()
