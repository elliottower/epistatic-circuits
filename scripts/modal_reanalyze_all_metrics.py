"""Modal: Re-analyze all coalition sweep results with full epistasis metrics.

Computes Walsh interaction fraction, Shapley interaction indices, LOO epistasis,
and full energy spectrum for every circuit × ablation combination we have.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_reanalyze_all_metrics.py
"""

import modal

app = modal.App("reanalyze-all-metrics")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4")
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
)

output_volume = modal.Volume.from_name("all-metrics-results", create_if_missing=True)

VOLUME_MOUNTS = {
    "/vol/ioi-c2": modal.Volume.from_name("c2-coalition-sweep"),
    "/vol/ioi-c5": modal.Volume.from_name("c5-coalition-sweep"),
    "/vol/ioi-c6": modal.Volume.from_name("c6-coalition-sweep"),
    "/vol/ioi-ic15": modal.Volume.from_name("ic15-coalition-sweep"),
    "/vol/ioi-v2-mean": modal.Volume.from_name("coalition-sweep-v2-mean"),
    "/vol/ioi-resample": modal.Volume.from_name("ioi-resample-sweep"),
    "/vol/gt": modal.Volume.from_name("gt-sweep-results"),
    "/vol/rti-v5": modal.Volume.from_name("rti-sweep-v5"),
    "/vol/rti-walsh": modal.Volume.from_name("rti-walsh-circuits-sweep"),
    "/vol/induction": modal.Volume.from_name("induction-sweep-results"),
    "/results": output_volume,
}


@app.function(
    image=image,
    timeout=3600,
    volumes=VOLUME_MOUNTS,
)
def analyze():
    import json
    import os
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/app")
    from walsh import compute_all_metrics, wht

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Discover all NPZ files across volumes
    npz_files = []
    vol_dirs = [d for d in os.listdir("/vol") if os.path.isdir(f"/vol/{d}")]
    for vdir in sorted(vol_dirs):
        vol_path = f"/vol/{vdir}"
        for fname in sorted(os.listdir(vol_path)):
            if fname.endswith(".npz") and "coalition" in fname:
                npz_files.append((vol_path, fname))
                print(f"  Found: {vol_path}/{fname}")

    print(f"\n[{ts()}] {len(npz_files)} NPZ files found")

    all_results = []

    for vol_path, fname in npz_files:
        path = os.path.join(vol_path, fname)
        try:
            data = np.load(path)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue

        # Determine task and ablation from filename
        # Patterns: ioi_C3_canonical_mean_coalition_values.npz
        #           ioi_C3_canonical_resample_coalition_values.npz
        #           gt_c5_walsh_mean_coalition_values.npz
        #           rti_c5_walsh_mean_coalition_values.npz
        base = fname.replace("_coalition_values.npz", "")

        # Check completion
        if "n_completed" in data and "n_total" in data:
            if int(data["n_completed"]) < int(data["n_total"]):
                print(f"  SKIP {fname}: incomplete ({data['n_completed']}/{data['n_total']})")
                continue

        n_players = int(data["n_players"])
        n_total = 1 << n_players
        circuit_heads = data["circuit_heads"] if "circuit_heads" in data else None
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads] if circuit_heads is not None else None

        # Get the values array — different files use different key names
        if "logit_diff" in data:
            values_2d = data["logit_diff"]
            metric_name = "logit_diff"
        elif "prob_diff" in data:
            values_2d = data["prob_diff"]
            metric_name = "prob_diff"
        elif "target_logits" in data and "foil_logits" in data:
            values_2d = data["target_logits"] - data["foil_logits"]
            metric_name = "logit_diff_from_raw"
        elif "log_prob_scores" in data:
            values_2d = data["log_prob_scores"]
            metric_name = "log_prob"
        elif "scores" in data:
            values_2d = data["scores"]
            metric_name = "scores"
        else:
            print(f"  SKIP {fname}: no recognized score key (keys={data.files})")
            continue

        if values_2d.shape[0] < n_total:
            print(f"  SKIP {fname}: only {values_2d.shape[0]}/{n_total} rows")
            continue

        n_prompts = values_2d.shape[1] if values_2d.ndim == 2 else 1
        mean_values = values_2d.mean(axis=1) if values_2d.ndim == 2 else values_2d

        print(f"\n[{ts()}] {base}: {n_players} heads, {n_prompts} prompts, {metric_name}")

        # Compute all metrics
        metrics = compute_all_metrics(mean_values[:n_total], n_players, head_labels)
        metrics["file"] = fname
        metrics["base"] = base
        metrics["n_players"] = n_players
        metrics["n_prompts"] = n_prompts
        metrics["metric"] = metric_name

        # Print summary
        wif = metrics["walsh_interaction_fraction"]
        loo = metrics["loo_epistasis"]
        faith = metrics["faithfulness"]
        tsii = metrics["total_shapley_interaction"]
        o1 = metrics["order1_frac"]
        o2 = metrics["order2_frac"]
        loo_flag = "" if metrics["loo_stable"] else " (UNSTABLE)"

        print(f"  Faithfulness:      {faith:+.4f}")
        print(f"  Walsh interaction: {wif*100:.1f}%  (order-1={o1*100:.1f}%, order-2={o2*100:.1f}%)")
        print(f"  LOO epistasis:     {loo*100:.1f}%{loo_flag}")
        print(f"  Total Shapley II:  {tsii:.4f}")

        top3 = metrics["top_interactions"][:3]
        if top3:
            parts = []
            for t in top3:
                p = t["pair"]
                parts.append("{}x{}={:+.4f}".format(p[0], p[1], t["sii"]))
            print("  Top interactions:  {}".format(", ".join(parts)))

        all_results.append(metrics)

    # Save full results
    with open("/results/all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    output_volume.commit()

    # Print comparison table
    print(f"\n\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    hdr = f"{'base':<45} {'faith':>7} {'WIF':>6} {'LOO':>8} {'TSII':>7} {'o1':>6} {'o2':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(all_results, key=lambda x: x["base"]):
        loo_s = f"{m['loo_epistasis']*100:.1f}%" if m["loo_stable"] else "---"
        print(
            f"{m['base']:<45} "
            f"{m['faithfulness']:>+7.3f} "
            f"{m['walsh_interaction_fraction']*100:>5.1f}% "
            f"{loo_s:>8} "
            f"{m['total_shapley_interaction']:>7.3f} "
            f"{m['order1_frac']*100:>5.1f}% "
            f"{m['order2_frac']*100:>5.1f}%"
        )

    print(f"\n[{ts()}] Saved to /results/all_metrics.json")


@app.local_entrypoint()
def main():
    analyze.remote()
