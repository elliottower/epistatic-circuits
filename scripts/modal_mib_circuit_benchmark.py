"""MIB-style circuit faithfulness benchmark: Walsh vs PP vs activation patching.

Computes faithfulness curves for circuits discovered by different head-ranking
methods. Following MIB protocol: f(C,N;m) = (m(C) - m(empty)) / (m(N) - m(empty))

Methods compared:
  1. Activation patching (AP): standard head importance (baseline)
  2. Walsh order-1: first-order Walsh coefficients
  3. Walsh interaction: order-2 aggregated per head (total interaction load)
  4. Path patching aggregate: sum of directed effects per head
  5. Walsh + PP combined
  6. Random (negative control)

Pre-reg SHA: 2e7cf364 (PREREG_MIB_CIRCUIT_BENCHMARK.md)

Usage: cd epistatic-circuits && modal run --detach scripts/modal_mib_circuit_benchmark.py
"""
import modal

app = modal.App("mib-circuit-benchmark-walsh-pp")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.41.2",
        "transformer-lens==2.6.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "tqdm==4.67.1",
        "einops==0.8.0",
        "typeguard==4.3.0",
        "matplotlib==3.9.0",
    )
)

phase2_vol = modal.Volume.from_name("sparse-walsh-phase2-results")
pp_vol = modal.Volume.from_name("path-patching-edges-results")
out_vol = modal.Volume.from_name("mib-circuit-benchmark-results", create_if_missing=True)

N_PROMPTS = 200
# Head counts to include (from 1 to all 144)
CIRCUIT_SIZES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 144]


def generate_ioi_prompts(n_prompts, tokenizer):
    import numpy as np

    NAMES_A = [
        "Mary", "Alice", "Emily", "Grace", "Kate",
        "Rose", "Sarah", "Emma", "Anna", "Lisa",
        "Amy", "Jane", "Beth", "Jean", "Ruth",
    ]
    NAMES_B = [
        "John", "Bob", "David", "Frank", "Henry",
        "Jack", "Liam", "James", "Mark", "Paul",
        "Bill", "Dan", "Mike", "Nick", "Tom",
    ]
    PLACES = [
        "store", "park", "beach", "library", "cafe",
        "mall", "school", "church", "lake", "gym",
    ]
    OBJECTS = [
        "drink", "book", "toy", "gift", "key",
        "letter", "phone", "bag", "hat", "ball",
    ]

    valid_a = [n for n in NAMES_A if len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1]
    valid_b = [n for n in NAMES_B if len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1]
    assert len(valid_a) >= 5 and len(valid_b) >= 5

    rng = np.random.default_rng(42)
    prompts, io_tokens, s_tokens = [], [], []

    for i in range(n_prompts):
        name_a = valid_a[rng.integers(len(valid_a))]
        name_b = valid_b[rng.integers(len(valid_b))]
        place = PLACES[rng.integers(len(PLACES))]
        obj = OBJECTS[rng.integers(len(OBJECTS))]

        if i % 2 == 0:
            prompt = (f"When {name_a} and {name_b} went to the {place},"
                      f" {name_b} gave a {obj} to")
        else:
            prompt = (f"When {name_b} and {name_a} went to the {place},"
                      f" {name_b} gave a {obj} to")

        prompts.append(prompt)
        io_tokens.append(tokenizer.encode(" " + name_a, add_special_tokens=False)[0])
        s_tokens.append(tokenizer.encode(" " + name_b, add_special_tokens=False)[0])

    return prompts, io_tokens, s_tokens


@app.function(
    image=image, gpu="A10G",
    volumes={"/phase2": phase2_vol, "/pp": pp_vol, "/out": out_vol},
    timeout=86400, memory=16384,
)
def run():
    import json, time
    from pathlib import Path
    import numpy as np
    import torch
    from tqdm import tqdm
    from scipy.stats import spearmanr, kendalltau

    t0 = time.time()

    def ts():
        return f"[{time.time() - t0:.0f}s]"

    def note(msg):
        print(f"{ts()} {msg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    note(f"Loading GPT-2 small on {device}")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    n_layers = model.cfg.n_layers  # 12
    n_heads_per_layer = model.cfg.n_heads  # 12
    total_heads = n_layers * n_heads_per_layer  # 144

    # ── Load Phase 2 data ───────────────────────────────────────────
    phase2_vol.reload()
    pp_vol.reload()

    with open("/phase2/phase2_head_selection.json") as f:
        head_sel = json.load(f)
    selected_heads = set(tuple(h) for h in head_sel["selected_heads"])

    with open("/phase2/phase2_all_walsh_coefficients.json") as f:
        all_coeffs = json.load(f)

    with open("/pp/path_patching_results.json") as f:
        pp_results = json.load(f)

    note(f"Loaded Walsh ({len(all_coeffs)} coefficients) and PP ({len(pp_results['all_pairs'])} pairs)")

    # ── Generate prompts ────────────────────────────────────────────
    note("Generating IOI prompts")
    prompts, io_token_ids, s_token_ids = generate_ioi_prompts(N_PROMPTS, model.tokenizer)
    tokens = model.to_tokens(prompts, prepend_bos=True)
    seq_len = tokens.shape[1]
    last_pos = seq_len - 1

    io_ids = torch.tensor(io_token_ids, device=device)
    s_ids = torch.tensor(s_token_ids, device=device)
    batch_idx = torch.arange(N_PROMPTS, device=device)

    def logit_diff(logits):
        return (logits[:, last_pos, :][batch_idx, io_ids]
                - logits[:, last_pos, :][batch_idx, s_ids]).mean()

    # ── Clean pass + cache ──────────────────────────────────────────
    note("Clean forward pass")
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(tokens)
    clean_ld = float(logit_diff(clean_logits).item())
    note(f"Clean logit diff: {clean_ld:.4f}")

    mean_hook_z = {}
    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        mean_hook_z[layer] = z.mean(dim=0)

    del cache
    torch.cuda.empty_cache()

    # ── m(empty): all heads mean-ablated ────────────────────────────
    note("Computing m(empty)")
    hooks_empty = []
    for layer in range(n_layers):
        mz = mean_hook_z[layer]

        def make_ablate_all(m):
            def hook_fn(value, hook):
                return m.unsqueeze(0).expand_as(value)
            return hook_fn

        hooks_empty.append((
            f"blocks.{layer}.attn.hook_z",
            make_ablate_all(mz),
        ))

    with torch.no_grad():
        empty_logits = model.run_with_hooks(tokens, fwd_hooks=hooks_empty)
    m_empty = float(logit_diff(empty_logits).item())
    m_full = clean_ld
    denom = m_full - m_empty
    note(f"m(empty)={m_empty:.4f}, m(N)={m_full:.4f}, denom={denom:.4f}")

    # ── Compute head importance by activation patching ──────────────
    note("Activation patching: ablating each head individually")
    ap_scores = {}

    for layer in tqdm(range(n_layers), desc="AP layers"):
        for head in range(n_heads_per_layer):
            mz = mean_hook_z[layer][:, head, :]

            def make_ablate_one(h, m):
                def hook_fn(value, hook):
                    value = value.clone()
                    value[:, :, h, :] = m.unsqueeze(0)
                    return value
                return hook_fn

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(
                        f"blocks.{layer}.attn.hook_z",
                        make_ablate_one(head, mz),
                    )]
                )
            patched_ld = float(logit_diff(patched_logits).item())
            ap_scores[(layer, head)] = clean_ld - patched_ld

    note(f"AP done. Top 5 heads by |AP|:")
    top_ap = sorted(ap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for (l, h), v in top_ap:
        note(f"  L{l}H{h}: AP={v:.4f}")

    # ── Build head-level importance for each method ─────────────────
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)]

    # Method 1: Activation patching
    method_ap = {hd: abs(ap_scores[hd]) for hd in all_heads}

    # Method 2: Walsh order-1
    walsh_o1 = {}
    for key, val in all_coeffs.items():
        if val["order"] == 1:
            layer = int(key.split("H")[0][1:])
            head = int(key.split("H")[1])
            walsh_o1[(layer, head)] = abs(val["coeff"])
    method_walsh_o1 = {hd: walsh_o1.get(hd, 0.0) for hd in all_heads}

    # Method 3: Walsh interaction aggregate (sum of |w_ij| over all partners)
    walsh_o2_per_head = {hd: 0.0 for hd in all_heads}
    for key, val in all_coeffs.items():
        if val["order"] == 2:
            parts = key.split("-")
            h1_str, h2_str = parts[0], parts[1]
            l1, hd1 = int(h1_str.split("H")[0][1:]), int(h1_str.split("H")[1])
            l2, hd2 = int(h2_str.split("H")[0][1:]), int(h2_str.split("H")[1])
            walsh_o2_per_head[(l1, hd1)] = walsh_o2_per_head.get((l1, hd1), 0.0) + abs(val["coeff"])
            walsh_o2_per_head[(l2, hd2)] = walsh_o2_per_head.get((l2, hd2), 0.0) + abs(val["coeff"])
    # Add order-1 to make it a combined Walsh score
    method_walsh_interaction = {
        hd: walsh_o2_per_head.get(hd, 0.0) + walsh_o1.get(hd, 0.0)
        for hd in all_heads
    }

    # Method 4: Path patching aggregate (sum of |PP| as sender + receiver)
    pp_per_head = {hd: 0.0 for hd in all_heads}
    for pair_data in pp_results["all_pairs"]:
        if pair_data["same_layer"]:
            continue
        sender = pair_data["sender"]
        receiver = pair_data["receiver"]
        s_l = int(sender.split("H")[0][1:])
        s_h = int(sender.split("H")[1])
        r_l = int(receiver.split("H")[0][1:])
        r_h = int(receiver.split("H")[1])
        effect = abs(pair_data["direct_effect"])
        pp_per_head[(s_l, s_h)] = pp_per_head.get((s_l, s_h), 0.0) + effect
        pp_per_head[(r_l, r_h)] = pp_per_head.get((r_l, r_h), 0.0) + effect
    method_pp_aggregate = {hd: pp_per_head.get(hd, 0.0) for hd in all_heads}

    # Method 5: Combined Walsh + PP
    # Normalize each to [0,1] then average
    walsh_vals = np.array([method_walsh_interaction[hd] for hd in all_heads])
    pp_vals = np.array([method_pp_aggregate[hd] for hd in all_heads])
    walsh_norm = walsh_vals / (walsh_vals.max() + 1e-10)
    pp_norm = pp_vals / (pp_vals.max() + 1e-10)
    combined_vals = 0.5 * walsh_norm + 0.5 * pp_norm
    method_combined = {hd: float(combined_vals[i]) for i, hd in enumerate(all_heads)}

    # Method 6: Random
    rng = np.random.default_rng(42)
    random_vals = rng.uniform(0, 1, total_heads)
    method_random = {hd: float(random_vals[i]) for i, hd in enumerate(all_heads)}

    methods = {
        "activation_patching": method_ap,
        "walsh_order1": method_walsh_o1,
        "walsh_interaction": method_walsh_interaction,
        "path_patching": method_pp_aggregate,
        "walsh_pp_combined": method_combined,
        "random": method_random,
    }

    # ── Evaluate faithfulness curves ────────────────────────────────
    note("Evaluating faithfulness for each method at each circuit size")

    results = {}
    for method_name, scores in methods.items():
        note(f"  Method: {method_name}")
        ranked_heads = sorted(all_heads, key=lambda hd: -scores[hd])
        faithfulness_curve = []

        for n_include in tqdm(CIRCUIT_SIZES, desc=method_name):
            circuit_heads = set(ranked_heads[:n_include])

            hooks = []
            for layer in range(n_layers):
                heads_to_ablate = [h for h in range(n_heads_per_layer)
                                   if (layer, h) not in circuit_heads]
                if not heads_to_ablate:
                    continue

                mz = mean_hook_z[layer]

                def make_selective_ablate(ablate_heads, m):
                    def hook_fn(value, hook):
                        value = value.clone()
                        for h in ablate_heads:
                            value[:, :, h, :] = m[:, h, :].unsqueeze(0)
                        return value
                    return hook_fn

                hooks.append((
                    f"blocks.{layer}.attn.hook_z",
                    make_selective_ablate(heads_to_ablate, mz),
                ))

            with torch.no_grad():
                circuit_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            m_circuit = float(logit_diff(circuit_logits).item())

            k_prop = n_include / total_heads
            f_score = (m_circuit - m_empty) / denom if abs(denom) > 1e-8 else 0.0

            faithfulness_curve.append({
                "n_heads": n_include,
                "k_proportion": round(k_prop, 4),
                "m_circuit": round(m_circuit, 4),
                "faithfulness": round(f_score, 4),
                "top_heads": [f"L{l}H{h}" for l, h in ranked_heads[:n_include]]
                             if n_include <= 20 else None,
            })

        # CPR and CMD via trapezoidal rule (over k_proportion)
        ks = [p["k_proportion"] for p in faithfulness_curve]
        fs = [p["faithfulness"] for p in faithfulness_curve]
        cpr = float(np.trapz(fs, ks))
        cmd = float(np.trapz([abs(1.0 - f) for f in fs], ks))

        results[method_name] = {
            "cpr": round(cpr, 4),
            "cmd": round(cmd, 4),
            "curve": faithfulness_curve,
        }
        note(f"    CPR={cpr:.4f}, CMD={cmd:.4f}")

        # Checkpoint
        Path("/out/mib_benchmark_checkpoint.json").write_text(
            json.dumps(results, indent=1)
        )
        out_vol.commit()

    # ── Head ranking correlations ───────────────────────────────────
    note("Computing head ranking correlations")
    method_names = list(methods.keys())
    ap_vec = np.array([method_ap[hd] for hd in all_heads])
    correlations = {}
    for mname in method_names:
        if mname == "random":
            continue
        m_vec = np.array([methods[mname][hd] for hd in all_heads])
        rho = float(spearmanr(ap_vec, m_vec).statistic)
        tau = float(kendalltau(ap_vec, m_vec).statistic)
        correlations[f"ap_vs_{mname}"] = {
            "spearman": round(rho, 4),
            "kendall": round(tau, 4),
        }

    # Walsh vs PP ranking correlation
    walsh_vec = np.array([method_walsh_interaction[hd] for hd in all_heads])
    pp_vec_all = np.array([method_pp_aggregate[hd] for hd in all_heads])
    rho_wp = float(spearmanr(walsh_vec, pp_vec_all).statistic)
    correlations["walsh_vs_pp"] = {
        "spearman": round(rho_wp, 4),
        "kendall": round(float(kendalltau(walsh_vec, pp_vec_all).statistic), 4),
    }

    # ── Top-K overlap analysis ──────────────────────────────────────
    note("Top-K overlap analysis")
    overlap_analysis = {}
    for K in [5, 10, 20]:
        ap_topk = set(sorted(all_heads, key=lambda hd: -method_ap[hd])[:K])
        for mname in ["walsh_order1", "walsh_interaction", "path_patching", "walsh_pp_combined"]:
            m_topk = set(sorted(all_heads, key=lambda hd: -methods[mname][hd])[:K])
            jaccard = len(ap_topk & m_topk) / len(ap_topk | m_topk)
            recall = len(ap_topk & m_topk) / K
            overlap_analysis[f"top{K}_{mname}_vs_ap"] = {
                "jaccard": round(jaccard, 4),
                "recall_of_ap": round(recall, 4),
                "shared_heads": [f"L{l}H{h}" for l, h in sorted(ap_topk & m_topk)],
            }

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "experiment": "MIB circuit faithfulness benchmark",
        "prereg_sha": "2e7cf364",
        "model": "gpt2",
        "n_prompts": N_PROMPTS,
        "total_heads": total_heads,
        "circuit_sizes": CIRCUIT_SIZES,
        "m_empty": round(m_empty, 4),
        "m_full": round(m_full, 4),
        "denominator": round(denom, 4),
        "methods": results,
        "head_ranking_correlations": correlations,
        "topk_overlap": overlap_analysis,
        "ranking_by_cpr": sorted(
            [(m, r["cpr"]) for m, r in results.items()],
            key=lambda x: -x[1]
        ),
        "ranking_by_cmd": sorted(
            [(m, r["cmd"]) for m, r in results.items()],
            key=lambda x: x[1]
        ),
        "ap_head_scores": {f"L{l}H{h}": round(v, 4) for (l, h), v in
                          sorted(ap_scores.items(), key=lambda x: -abs(x[1]))},
    }

    note("\n=== FINAL RANKINGS ===")
    note("By CPR (higher is better):")
    for m, v in summary["ranking_by_cpr"]:
        note(f"  {m}: {v:.4f}")
    note("By CMD (lower is better):")
    for m, v in summary["ranking_by_cmd"]:
        note(f"  {m}: {v:.4f}")
    note("\nHead ranking correlations with AP:")
    for k, v in correlations.items():
        note(f"  {k}: Spearman={v['spearman']:.4f}")

    Path("/out/mib_benchmark_results.json").write_text(
        json.dumps(summary, indent=1)
    )
    out_vol.commit()
    note("All results saved. Done.")

    return summary


@app.local_entrypoint()
def main():
    import json
    result = run.remote()
    if result:
        print("\n=== MIB BENCHMARK RESULTS ===")
        print(f"m(empty)={result['m_empty']:.4f}, m(N)={result['m_full']:.4f}")
        print("\nBy CPR (higher is better):")
        for m, v in result["ranking_by_cpr"]:
            print(f"  {m}: {v:.4f}")
        print("\nBy CMD (lower is better):")
        for m, v in result["ranking_by_cmd"]:
            print(f"  {m}: {v:.4f}")
