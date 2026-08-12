"""MIB faithfulness: Walsh vs activation patching for circuit discovery.

Computes Walsh interaction coefficients over all 144 attention heads in GPT-2
small via sparse recovery (M=12,500 random coalitions, LASSO).  Derives node
importance from Walsh, activation patching, and combined rankings.  Evaluates
circuit faithfulness at multiple circuit sizes using MIB's CPR/CMD metrics.

Pre-reg: PREREG_MIB_FAITHFULNESS.md

Usage: cd epistatic-circuits && modal run --detach scripts/modal_mib_faithfulness.py
"""
import modal

app = modal.App("mib-faithfulness-walsh-144head")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.41.2",
        "transformer-lens==2.6.0",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "scipy==1.13.1",
        "tqdm==4.67.1",
        "einops==0.8.0",
        "matplotlib==3.9.0",
        "typeguard==4.3.0",
    )
)

out_vol = modal.Volume.from_name("mib-faithfulness-results", create_if_missing=True)

M_COALITIONS = 12500
N_PROMPTS = 200
CHECKPOINT_EVERY = 500
RANDOM_SEEDS = 5
CIRCUIT_SIZES = [1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 72, 100, 144]


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

    valid_a = []
    for name in NAMES_A:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            valid_a.append(name)
    valid_b = []
    for name in NAMES_B:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            valid_b.append(name)

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
        io_tokens.append(
            tokenizer.encode(" " + name_a, add_special_tokens=False)[0]
        )
        s_tokens.append(
            tokenizer.encode(" " + name_b, add_special_tokens=False)[0]
        )

    return prompts, io_tokens, s_tokens


@app.function(
    image=image, gpu="A10G",
    volumes={"/out": out_vol},
    timeout=86400, memory=32768,
)
def run():
    import json, time, itertools
    from pathlib import Path
    import numpy as np
    import torch
    from tqdm import tqdm

    t0 = time.time()

    def ts():
        return f"[{time.time() - t0:.0f}s]"

    def note(msg):
        print(f"{ts()} {msg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    note(f"Loading GPT-2 small on {device}")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", device=device
    )
    model.eval()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    n_total = n_layers * n_heads
    note(f"Model: {n_layers} layers, {n_heads} heads/layer, {n_total} total heads")

    note("Generating IOI prompts (seed 42)")
    prompts, io_ids_list, s_ids_list = generate_ioi_prompts(
        N_PROMPTS, model.tokenizer
    )
    tokens = model.to_tokens(prompts, prepend_bos=True)
    seq_len = tokens.shape[1]
    last_pos = seq_len - 1
    io_ids = torch.tensor(io_ids_list, device=device)
    s_ids = torch.tensor(s_ids_list, device=device)
    batch_idx = torch.arange(N_PROMPTS, device=device)
    note(f"Tokenized: {tokens.shape}, seq_len={seq_len}")

    # ── Clean forward pass + cache mean hook_z ──────────────────────
    note("Clean forward pass")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    clean_ld = float(
        (logits[:, last_pos][batch_idx, io_ids]
         - logits[:, last_pos][batch_idx, s_ids]).mean().item()
    )
    note(f"Clean logit diff: {clean_ld:.4f}")

    mean_z = {}
    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        for head in range(n_heads):
            mean_z[(layer, head)] = z[:, :, head, :].mean(dim=0)

    del cache
    torch.cuda.empty_cache()
    note("Mean hook_z cached for all 144 heads")

    # ── Helper: measure logit diff with specific heads ablated ──────
    def measure_ld(mask):
        """mask: (n_total,) bool array. True=active, False=mean-ablated."""
        hooks = []
        for layer in range(n_layers):
            ablate_heads = []
            for head in range(n_heads):
                if not mask[layer * n_heads + head]:
                    ablate_heads.append(head)
            if not ablate_heads:
                continue
            mz = {h: mean_z[(layer, h)] for h in ablate_heads}

            def make_hook(ah, mzd):
                def hook_fn(value, hook):
                    v = value.clone()
                    for h in ah:
                        v[:, :, h, :] = mzd[h].unsqueeze(0)
                    return v
                return hook_fn

            hooks.append((
                f"blocks.{layer}.attn.hook_z",
                make_hook(ablate_heads, mz),
            ))

        with torch.no_grad():
            out = model.run_with_hooks(tokens, fwd_hooks=hooks)
        return float(
            (out[:, last_pos][batch_idx, io_ids]
             - out[:, last_pos][batch_idx, s_ids]).mean().item()
        )

    # ═══════════════════════════════════════════════════════════════
    # PHASE A1: Walsh coalition sampling (12,500 forward passes)
    # ═══════════════════════════════════════════════════════════════
    note("Phase A1: Walsh coalition sampling")

    ckpt_path = Path("/out/walsh_coalitions_144head.npz")
    out_vol.reload()

    if ckpt_path.exists():
        ckpt = np.load(str(ckpt_path))
        coalition_masks = ckpt["masks"]
        coalition_values = ckpt["values"]
        start_m = int(ckpt["completed"])
        note(f"Resumed from checkpoint: {start_m}/{M_COALITIONS} done")
    else:
        rng = np.random.default_rng(2024)
        coalition_masks = rng.random((M_COALITIONS, n_total)) < 0.5
        coalition_values = np.zeros(M_COALITIONS, dtype=np.float64)
        start_m = 0

    for m in tqdm(range(start_m, M_COALITIONS),
                  desc="Walsh coalitions",
                  initial=start_m, total=M_COALITIONS):
        coalition_values[m] = measure_ld(coalition_masks[m])

        if (m + 1) % CHECKPOINT_EVERY == 0:
            np.savez(
                str(ckpt_path),
                masks=coalition_masks,
                values=coalition_values,
                completed=m + 1,
            )
            out_vol.commit()
            note(f"Checkpoint: {m + 1}/{M_COALITIONS}")

    np.savez(
        str(ckpt_path),
        masks=coalition_masks,
        values=coalition_values,
        completed=M_COALITIONS,
    )
    out_vol.commit()
    note(f"Phase A1 complete: {M_COALITIONS} coalitions sampled")

    # ═══════════════════════════════════════════════════════════════
    # PHASE A2: Walsh sparse recovery (LASSO)
    # ═══════════════════════════════════════════════════════════════
    note("Phase A2: Walsh sparse recovery")

    pairs = list(itertools.combinations(range(n_total), 2))
    n_pairs = len(pairs)
    k_coeffs = n_total + n_pairs
    note(f"k = {n_total} order-1 + {n_pairs} order-2 = {k_coeffs} coefficients")
    note(f"M/k = {M_COALITIONS / k_coeffs:.2f}")

    # Build Walsh basis matrix
    # X_order1[m, i] = (-1)^{coalition_active[m,i]}
    X_order1 = np.where(coalition_masks, -1.0, 1.0).astype(np.float64)

    pairs_i = np.array([p[0] for p in pairs], dtype=np.int32)
    pairs_j = np.array([p[1] for p in pairs], dtype=np.int32)
    note("Building order-2 basis matrix...")
    X_order2 = X_order1[:, pairs_i] * X_order1[:, pairs_j]

    X = np.hstack([X_order1, X_order2])
    y = coalition_values
    note(f"Walsh basis matrix: {X.shape}, {X.nbytes / 1e9:.2f} GB")

    del X_order1, X_order2
    import gc
    gc.collect()

    from sklearn.linear_model import LassoCV
    note("Fitting LASSO (this may take several minutes)...")
    lasso = LassoCV(
        alphas=np.logspace(-5, -1, 15),
        cv=3,
        max_iter=20000,
        fit_intercept=True,
        n_jobs=-1,
    )
    lasso.fit(X, y)
    w_lasso = lasso.coef_

    n_nonzero = int(np.count_nonzero(w_lasso))
    note(f"LASSO: {n_nonzero}/{k_coeffs} nonzero coefficients, alpha={lasso.alpha_:.6f}")

    # Split-half validation
    note("Split-half validation...")
    half = M_COALITIONS // 2
    lasso_a = LassoCV(
        alphas=np.logspace(-5, -1, 15), cv=3,
        max_iter=20000, fit_intercept=True, n_jobs=-1,
    )
    lasso_b = LassoCV(
        alphas=np.logspace(-5, -1, 15), cv=3,
        max_iter=20000, fit_intercept=True, n_jobs=-1,
    )
    lasso_a.fit(X[:half], y[:half])
    lasso_b.fit(X[half:], y[half:])
    split_half_r = float(np.corrcoef(lasso_a.coef_, lasso_b.coef_)[0, 1])
    note(f"Split-half correlation: r = {split_half_r:.4f}")

    del X
    gc.collect()

    # Walsh node importance: |order-1| + sum |order-2| for each head
    walsh_importance = np.zeros(n_total)
    walsh_importance += np.abs(w_lasso[:n_total])
    for idx, (i, j) in enumerate(pairs):
        mag = abs(w_lasso[n_total + idx])
        walsh_importance[i] += mag
        walsh_importance[j] += mag

    walsh_top20 = [
        {"flat_idx": int(idx), "layer": int(idx // n_heads),
         "head": int(idx % n_heads),
         "importance": round(float(walsh_importance[idx]), 6)}
        for idx in np.argsort(-walsh_importance)[:20]
    ]
    note(f"Walsh top 5: {[(h['layer'], h['head']) for h in walsh_top20[:5]]}")

    # Save full Walsh coefficients
    walsh_coefficients = {}
    for i in range(n_total):
        if w_lasso[i] != 0:
            l, h = divmod(i, n_heads)
            walsh_coefficients[f"L{l}H{h}"] = {
                "order": 1,
                "coeff": round(float(w_lasso[i]), 6),
            }
    for idx, (i, j) in enumerate(pairs):
        c = w_lasso[n_total + idx]
        if c != 0:
            li, hi = divmod(i, n_heads)
            lj, hj = divmod(j, n_heads)
            walsh_coefficients[f"L{li}H{hi}-L{lj}H{hj}"] = {
                "order": 2,
                "coeff": round(float(c), 6),
            }

    walsh_results = {
        "n_nonzero": n_nonzero,
        "alpha_best": float(lasso.alpha_),
        "split_half_r": round(split_half_r, 4),
        "node_importance": [round(float(v), 6) for v in walsh_importance],
        "top20_heads": walsh_top20,
        "coefficients": walsh_coefficients,
    }
    Path("/out/walsh_144head_results.json").write_text(
        json.dumps(walsh_results, indent=1)
    )
    out_vol.commit()
    note("Phase A2 complete")

    # ═══════════════════════════════════════════════════════════════
    # PHASE A3: Activation patching (144 individual ablations)
    # ═══════════════════════════════════════════════════════════════
    note("Phase A3: Activation patching (144 heads)")

    actp_importance = np.zeros(n_total)
    for flat_idx in tqdm(range(n_total), desc="ActP"):
        mask = np.ones(n_total, dtype=bool)
        mask[flat_idx] = False
        ablated_ld = measure_ld(mask)
        actp_importance[flat_idx] = abs(clean_ld - ablated_ld)

    actp_top20 = [
        {"flat_idx": int(idx), "layer": int(idx // n_heads),
         "head": int(idx % n_heads),
         "importance": round(float(actp_importance[idx]), 6)}
        for idx in np.argsort(-actp_importance)[:20]
    ]
    note(f"ActP top 5: {[(h['layer'], h['head']) for h in actp_top20[:5]]}")

    Path("/out/actp_144head_results.json").write_text(json.dumps({
        "node_importance": [round(float(v), 6) for v in actp_importance],
        "top20_heads": actp_top20,
    }, indent=1))
    out_vol.commit()
    note("Phase A3 complete")

    # ═══════════════════════════════════════════════════════════════
    # Derive rankings
    # ═══════════════════════════════════════════════════════════════

    def rank_array(arr):
        """Return rank (0=best) for each element by descending magnitude."""
        ranks = np.empty(len(arr), dtype=np.float64)
        ranks[np.argsort(-arr)] = np.arange(len(arr))
        return ranks

    walsh_ranks = rank_array(walsh_importance)
    actp_ranks = rank_array(actp_importance)
    combined_ranks = walsh_ranks + actp_ranks

    rankings = {
        "walsh": np.argsort(-walsh_importance),
        "actp": np.argsort(-actp_importance),
        "walsh_actp": np.argsort(combined_ranks),
    }

    # Overlap analysis: how many of top-20 agree?
    walsh_top20_set = set(rankings["walsh"][:20])
    actp_top20_set = set(rankings["actp"][:20])
    overlap_20 = len(walsh_top20_set & actp_top20_set)
    note(f"Walsh-ActP top-20 overlap: {overlap_20}/20")

    # ═══════════════════════════════════════════════════════════════
    # PHASE B: Faithfulness evaluation (MIB protocol)
    # ═══════════════════════════════════════════════════════════════
    note("Phase B: Faithfulness evaluation")

    m_N = clean_ld
    empty_mask = np.zeros(n_total, dtype=bool)
    m_empty = measure_ld(empty_mask)
    denom = m_N - m_empty
    note(f"m(N) = {m_N:.4f}, m(empty) = {m_empty:.4f}, denom = {denom:.4f}")

    # Add random baselines
    for seed in range(RANDOM_SEEDS):
        rng_eval = np.random.default_rng(100 + seed)
        rankings[f"random_{seed}"] = rng_eval.permutation(n_total)

    all_results = {}

    for method_name, ranking in rankings.items():
        note(f"  Evaluating: {method_name}")
        curve = {}

        for k_size in CIRCUIT_SIZES:
            circuit_mask = np.zeros(n_total, dtype=bool)
            circuit_mask[ranking[:k_size]] = True
            m_C = measure_ld(circuit_mask)

            f_val = (m_C - m_empty) / denom if abs(denom) > 1e-6 else 0.0
            curve[k_size] = {
                "k_heads": k_size,
                "k_fraction": round(k_size / n_total, 4),
                "logit_diff": round(float(m_C), 4),
                "faithfulness": round(float(f_val), 4),
            }

        # CPR and CMD via right-endpoint Riemann sum
        k_fracs = [curve[k]["k_fraction"] for k in CIRCUIT_SIZES]
        f_vals = [curve[k]["faithfulness"] for k in CIRCUIT_SIZES]

        prev_k = 0.0
        cpr = 0.0
        cmd = 0.0
        for kf, fv in zip(k_fracs, f_vals):
            dk = kf - prev_k
            cpr += fv * dk
            cmd += abs(1.0 - fv) * dk
            prev_k = kf

        all_results[method_name] = {
            "curve": curve,
            "CPR": round(float(cpr), 4),
            "CMD": round(float(cmd), 4),
        }
        note(f"    CPR = {cpr:.4f}, CMD = {cmd:.4f}")

    # Average random baselines
    r_cprs = [all_results[f"random_{s}"]["CPR"] for s in range(RANDOM_SEEDS)]
    r_cmds = [all_results[f"random_{s}"]["CMD"] for s in range(RANDOM_SEEDS)]
    all_results["random_avg"] = {
        "CPR": round(float(np.mean(r_cprs)), 4),
        "CPR_std": round(float(np.std(r_cprs)), 4),
        "CMD": round(float(np.mean(r_cmds)), 4),
        "CMD_std": round(float(np.std(r_cmds)), 4),
    }
    note(f"  Random avg: CPR={np.mean(r_cprs):.4f}, CMD={np.mean(r_cmds):.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Save everything
    # ═══════════════════════════════════════════════════════════════
    summary = {
        "experiment": "MIB faithfulness: Walsh vs ActP on all 144 heads",
        "prereg": "PREREG_MIB_FAITHFULNESS.md",
        "model": "gpt2",
        "n_heads_total": n_total,
        "n_prompts": N_PROMPTS,
        "m_coalitions": M_COALITIONS,
        "clean_logit_diff": round(clean_ld, 4),
        "m_empty": round(float(m_empty), 4),
        "walsh_split_half_r": round(split_half_r, 4),
        "walsh_n_nonzero": n_nonzero,
        "walsh_actp_top20_overlap": overlap_20,
        "walsh_top20": walsh_top20,
        "actp_top20": actp_top20,
        "circuit_sizes": CIRCUIT_SIZES,
        "faithfulness_results": all_results,
    }

    Path("/out/mib_faithfulness_results.json").write_text(
        json.dumps(summary, indent=1)
    )
    out_vol.commit()
    note("All results saved. Done.")

    # Print summary table
    print("\n" + "=" * 60)
    print("METHOD          CPR     CMD")
    print("-" * 60)
    for name in ["walsh", "actp", "walsh_actp", "random_avg"]:
        r = all_results[name]
        print(f"{name:<16} {r['CPR']:.4f}  {r['CMD']:.4f}")
    print("=" * 60)

    return summary


@app.local_entrypoint()
def main():
    import json
    result = run.remote()
    if result:
        print("\n=== MIB FAITHFULNESS RESULTS ===")
        for name in ["walsh", "actp", "walsh_actp", "random_avg"]:
            r = result["faithfulness_results"][name]
            print(f"  {name}: CPR={r['CPR']:.4f}, CMD={r['CMD']:.4f}")
