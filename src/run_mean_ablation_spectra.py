"""Compute Walsh energy spectra on mean-ablation coalition tables.

Prereg v8 (SHA f62bd44) specifies mean-ablation as primary evidence.
This script computes:
  1. Logit-diff Walsh spectra with bootstrap CIs (standard pipeline)
  2. Probability-scale Walsh spectra (Fisher's metrical bias check)

Probability scale: v_p(S) = tanh(logit_diff(S) / 2), which equals the
two-token probability difference P_2(IO) - P_2(S). This is a saturating
transform that compresses large logit differences — exactly the test for
whether interaction structure is an artifact of the log-ratio scale.

Usage:
    cd experiments_batch2/genetics
    uv run python run_mean_ablation_spectra.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from walsh_test import (
    load_coalition_values,
    per_prompt_values,
    per_prompt_wht_normalized,
    energy_spectrum_ci,
    energy_spectrum_contrast,
)


DATA_DIR = Path(__file__).parent
RESULTS_DIR = DATA_DIR / "results_mean_ablation_v8"
CIRCUITS = ["ioi", "weight_ioi", "random15"]
CIRCUIT_LABELS = {
    "ioi": "C3 (canonical IOI)",
    "weight_ioi": "C1 (weight-relational IOI)",
    "random15": "C4 (random-15)",
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def prob_scale_values(target_logits, foil_logits):
    """Two-token probability difference: tanh(logit_diff / 2).

    Equivalent to sigmoid(logit_diff) - sigmoid(-logit_diff),
    i.e. P_2(IO) - P_2(S_name) under a two-token softmax.
    """
    logit_diff = target_logits - foil_logits
    return np.tanh(logit_diff / 2.0)


def run_one_circuit(name, scale="logit"):
    data_path = DATA_DIR / f"{name}_mean_v2_coalition_values.npz"
    data = load_coalition_values(str(data_path))
    n = data["n_players"]

    if scale == "logit":
        v = per_prompt_values(data["target_logits"], data["foil_logits"])
    elif scale == "prob":
        v = prob_scale_values(data["target_logits"], data["foil_logits"])
    else:
        raise ValueError(f"Unknown scale: {scale}")

    wht_pp = per_prompt_wht_normalized(v, n)
    energy = energy_spectrum_ci(wht_pp, n, seed=42)

    return {
        "circuit": name,
        "label": CIRCUIT_LABELS[name],
        "scale": scale,
        "n_players": n,
        "n_prompts": data["n_prompts"],
        "n_coalitions": v.shape[0],
        "energy_spectrum": {
            f"order_{k}": {
                "mean_fraction": float(energy["mean_fraction"][k]),
                "ci_lower": float(energy["ci_lower"][k]),
                "ci_upper": float(energy["ci_upper"][k]),
            }
            for k in range(n + 1)
        },
        "order_3plus": {
            "mean_fraction": float(energy["mean_fraction"][3:].sum()),
            "ci_lower_sum": float(energy["ci_lower"][3:].sum()),
            "ci_upper_sum": float(energy["ci_upper"][3:].sum()),
        },
        "wht_pp": wht_pp,
    }


def run_contrast(wht_a, wht_b, n, label_a, label_b):
    contrast = energy_spectrum_contrast(wht_a, wht_b, n, seed=42)
    return {
        "pair": f"{label_a}_vs_{label_b}",
        "observed_delta": contrast["observed_delta"],
        "ci_lower": contrast["ci_lower"],
        "ci_upper": contrast["ci_upper"],
        "excludes_zero": contrast["excludes_zero"],
    }


def print_spectrum(result):
    name = result["label"]
    scale = result["scale"]
    n = result["n_players"]
    print(f"\n  {name} [{scale}-scale]:")
    for k in range(n + 1):
        e = result["energy_spectrum"][f"order_{k}"]
        frac = e["mean_fraction"]
        bar = "#" * int(frac * 100) if not np.isnan(frac) else "NaN"
        print(f"    order {k:2d}: {frac:6.1%}  [{e['ci_lower']:6.1%}, {e['ci_upper']:6.1%}]  {bar}")
    o3 = result["order_3plus"]
    print(f"    order 3+: {o3['mean_fraction']:6.1%}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"[{timestamp()}] Mean-ablation Walsh spectra (prereg v8, SHA f62bd44)")
    print(f"[{timestamp()}] Results dir: {RESULTS_DIR}")

    all_results = {"logit": {}, "prob": {}}
    wht_cache = {}

    for scale in ["logit", "prob"]:
        print(f"\n{'='*70}")
        print(f"  SCALE: {scale}")
        print(f"{'='*70}")

        for name in CIRCUITS:
            t0 = time.time()
            result = run_one_circuit(name, scale=scale)
            dt = time.time() - t0
            print(f"  {name} done in {dt:.1f}s")
            print_spectrum(result)

            wht_cache[(name, scale)] = result["wht_pp"]

            result_save = {k: v for k, v in result.items() if k != "wht_pp"}
            all_results[scale][name] = result_save

            out_path = RESULTS_DIR / f"{name}_mean_{scale}_spectrum.json"
            with open(out_path, "w") as f:
                json.dump(result_save, f, indent=2, cls=NumpyEncoder)

        print(f"\n  Pairwise contrasts ({scale}):")
        contrasts = {}
        pairs = [("weight_ioi", "ioi"), ("weight_ioi", "random15"), ("ioi", "random15")]
        for a, b in pairs:
            c = run_contrast(
                wht_cache[(a, scale)], wht_cache[(b, scale)],
                n=15, label_a=a, label_b=b
            )
            key = f"{a}_vs_{b}"
            contrasts[key] = c

            o3_delta = float(c["observed_delta"][3:].sum()) if isinstance(c["observed_delta"], np.ndarray) else sum(c["observed_delta"][3:])
            print(f"    {key}: order-3+ delta = {o3_delta:+.4f}")

        all_results[scale]["contrasts"] = contrasts

    combined_path = RESULTS_DIR / "all_mean_ablation_spectra.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[{timestamp()}] Combined results: {combined_path}")

    print(f"\n{'='*70}")
    print("PREDICTION CHECK (prereg v8, SHA f62bd44)")
    print(f"{'='*70}")

    for scale in ["logit", "prob"]:
        print(f"\n  --- {scale}-scale ---")
        for name in CIRCUITS:
            r = all_results[scale][name]
            o1 = r["energy_spectrum"]["order_1"]["mean_fraction"]
            o3 = r["order_3plus"]["mean_fraction"]
            print(f"  {CIRCUIT_LABELS[name]:30s}  order-1: {o1:6.1%}  order-3+: {o3:6.1%}")

        c1_o3 = all_results[scale]["weight_ioi"]["order_3plus"]["mean_fraction"]
        c3_o3 = all_results[scale]["ioi"]["order_3plus"]["mean_fraction"]
        c4_o3 = all_results[scale]["random15"]["order_3plus"]["mean_fraction"]
        ratio = c1_o3 / c3_o3 if c3_o3 > 0 else float("inf")
        print(f"\n  C1/C3 order-3+ ratio: {ratio:.2f}x")
        print(f"  Prediction: C1 > C3 > C4 in order-3+")
        print(f"  Observed:   C1={c1_o3:.1%}  C3={c3_o3:.1%}  C4={c4_o3:.1%}")
        if c1_o3 > c3_o3 > c4_o3:
            print(f"  --> ORDERING HOLDS")
        else:
            print(f"  --> ORDERING VIOLATED")


if __name__ == "__main__":
    main()
