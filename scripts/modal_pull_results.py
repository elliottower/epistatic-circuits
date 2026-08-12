"""Pull subspace epistasis results from Modal volume + check ACDC sweep status.

Saves results locally and computes Walsh metrics for any completed ACDC sweeps.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_pull_results.py
"""

import modal
import json
import sys

app = modal.App("pull-results-and-check-acdc")

subspace_vol = modal.Volume.from_name("subspace-epistasis-results")
acdc_vol = modal.Volume.from_name("acdc-resample-sweep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4",
).add_local_file("src/walsh.py", remote_path="/app/walsh.py")


@app.function(
    image=image,
    volumes={"/subspace": subspace_vol, "/acdc": acdc_vol},
    timeout=120,
)
def pull_and_check():
    import os
    import numpy as np
    sys.path.insert(0, "/app")
    from walsh import compute_all_metrics

    results = {}

    # 1. Read subspace results
    print("=== SUBSPACE EPISTASIS RESULTS ===")
    for fname in sorted(os.listdir("/subspace")):
        fpath = os.path.join("/subspace", fname)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {fname}  ({size_mb:.1f} MB)")

    summary_path = "/subspace/subspace_epistasis_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        results["subspace_summary"] = summary
        print(f"\n  Pooled rho_OV = {summary['pooled_rho_ov']:.4f}")
        print(f"  Pooled R^2 = {summary['pooled_r_squared']:.4f}")
        print(f"  Predictions: {json.dumps(summary['predictions'], indent=2)}")

    pairs_path = "/subspace/subspace_epistasis_pairs.csv"
    if os.path.exists(pairs_path):
        with open(pairs_path) as f:
            pairs_csv = f.read()
        results["pairs_csv"] = pairs_csv

    per_circuit_path = "/subspace/subspace_epistasis_per_circuit.json"
    if os.path.exists(per_circuit_path):
        with open(per_circuit_path) as f:
            per_circuit = json.load(f)
        results["per_circuit"] = per_circuit

    # 2. Check ACDC sweep status and compute metrics if complete
    print("\n=== ACDC SWEEP STATUS ===")
    acdc_metrics = []
    for fname in sorted(os.listdir("/acdc")):
        fpath = os.path.join("/acdc", fname)
        if not fname.endswith(".npz"):
            continue
        d = np.load(fpath)
        nc = int(d["n_completed"]) if "n_completed" in d else 0
        nt = int(d.get("n_total", 0))
        print(f"  {fname}: {nc}/{nt}")

        if nc == nt and nt > 0:
            # Complete! Compute Walsh metrics
            task_prefix = fname.split("_")[0]  # ioi or rti
            task = task_prefix.upper()

            values = d["logit_diff"]  # (2^n, n_prompts)
            mean_values = values.mean(axis=1)
            n_players = int(np.log2(len(mean_values)))
            full_idx = (1 << n_players) - 1

            m = compute_all_metrics(mean_values, n_players)

            row = {
                "task": task,
                "circuit": "ACDC",
                "ablation": "resample",
                "n_heads": n_players,
                "faithfulness": round(m["faithfulness"], 2),
                "WIF": round(m["walsh_interaction_fraction"], 3),
                "order1_pct": round(m["order1_frac"] * 100, 1),
                "order2_pct": round(m["order2_frac"] * 100, 1),
                "order3plus_pct": round(m["order3plus_frac"] * 100, 1),
                "LOO_epistasis": round(m["loo_epistasis"], 3),
                "LOO_stable": m["loo_stable"],
                "TSII": round(m["total_shapley_interaction"], 2),
                "metric_source": "logit_diff",
            }
            acdc_metrics.append(row)
            print(f"    -> {task} ACDC: faith={row['faithfulness']}, WIF={row['WIF']}, "
                  f"order1={row['order1_pct']}%, TSII={row['TSII']}")

    results["acdc_metrics"] = acdc_metrics

    return results


@app.local_entrypoint()
def main():
    r = pull_and_check.remote()
    print("\n=== SAVING LOCALLY ===")

    # Save pairs CSV
    if "pairs_csv" in r:
        with open("results/subspace_epistasis_pairs.csv", "w") as f:
            f.write(r["pairs_csv"])
        print("Saved results/subspace_epistasis_pairs.csv")

    # Save per-circuit JSON
    if "per_circuit" in r:
        with open("results/subspace_epistasis_per_circuit.json", "w") as f:
            json.dump(r["per_circuit"], f, indent=2)
        print("Saved results/subspace_epistasis_per_circuit.json")

    # Save subspace summary
    if "subspace_summary" in r:
        with open("results/subspace_epistasis_summary.json", "w") as f:
            json.dump(r["subspace_summary"], f, indent=2)
        print("Saved results/subspace_epistasis_summary.json")

    # Print ACDC metrics for CSV append
    if r.get("acdc_metrics"):
        print("\n=== ACDC CSV ROWS (append to all_metrics_consolidated.csv) ===")
        for row in r["acdc_metrics"]:
            csv_line = (f"{row['task']},{row['circuit']},{row['ablation']},"
                        f"{row['n_heads']},{row['faithfulness']},{row['WIF']},"
                        f"{row['order1_pct']},{row['order2_pct']},"
                        f"{row['order3plus_pct']},{row['LOO_epistasis']},"
                        f"{row['LOO_stable']},{row['TSII']},{row['metric_source']}")
            print(csv_line)
        # Also save as JSON
        with open("results/acdc_resample_metrics.json", "w") as f:
            json.dump(r["acdc_metrics"], f, indent=2)
        print("Saved results/acdc_resample_metrics.json")
    else:
        print("\nNo completed ACDC sweeps yet.")
