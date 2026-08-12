"""Phase 2: Sparse Walsh recovery on 20-head IOI circuit in GPT-2 small.

Identifies the top-20 attention heads by activation patching on IOI prompts,
samples M_max=2000 random coalitions, mean-ablates selected heads, and
recovers pairwise Walsh coefficients via LASSO/OMP. Validates via split-half
consistency and exact marginal pairwise interaction for top pairs.

Pre-reg SHA: d8d0f11b (PREREG_SPARSE_WALSH_RECOVERY.md Phase 2)

Usage: cd epistatic-circuits && modal run --detach scripts/modal_sparse_walsh_phase2.py
"""
import modal

app = modal.App("sparse-walsh-phase2-ioi-20head")

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

out_vol = modal.Volume.from_name(
    "sparse-walsh-phase2-results", create_if_missing=True
)

N_CIRCUIT_HEADS = 20
M_MAX = 2000
N_PROMPTS = 200
N_TRIALS = 10
SAMPLE_SIZES = [100, 200, 500, 1000, 2000]
CHECKPOINT_EVERY = 200
N_EXACT_PAIRS = 30


def generate_ioi_prompts(n_prompts, tokenizer):
    """Generate IOI prompts with verified single-token names."""
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

    assert len(valid_a) >= 5 and len(valid_b) >= 5, (
        f"Not enough single-token names: {len(valid_a)} A, {len(valid_b)} B"
    )

    rng = np.random.default_rng(42)
    prompts = []
    io_tokens = []
    s_tokens = []

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


def _popcount_lut():
    import numpy as np
    lut = np.zeros(65536, dtype=np.int8)
    for i in range(1, 65536):
        lut[i] = lut[i >> 1] + (i & 1)
    return lut


def _popcount_array(arr, lut):
    import numpy as np
    arr = np.asarray(arr, dtype=np.uint32)
    return (
        lut[arr & 0xFFFF].astype(np.int32)
        + lut[(arr >> 16) & 0xFFFF].astype(np.int32)
    )


def walsh_basis_matrix_fast(coalition_ids, all_indices, lut):
    import numpy as np
    coalition_ids = np.asarray(coalition_ids, dtype=np.uint32)
    all_indices = np.asarray(all_indices, dtype=np.uint32)
    bitwise_and = coalition_ids[:, None] & all_indices[None, :]
    bits = _popcount_array(bitwise_and.ravel(), lut).reshape(bitwise_and.shape)
    return np.where(bits % 2 == 0, 1.0, -1.0)


def sparse_recover(X, y, k, method="lasso"):
    """Recover sparse Walsh coefficients from coalition measurements."""
    import numpy as np
    from sklearn.linear_model import LassoCV, OrthogonalMatchingPursuitCV

    M = X.shape[0]
    if method == "mc":
        return np.mean(y[:, None] * X, axis=0)
    elif method == "lasso":
        lasso = LassoCV(
            alphas=np.logspace(-6, -1, 20), cv=5,
            max_iter=10000, fit_intercept=True,
        )
        lasso.fit(X, y)
        return lasso.coef_
    elif method == "omp":
        omp = OrthogonalMatchingPursuitCV(
            cv=5, max_iter=min(k, M // 2),
        )
        omp.fit(X, y)
        return omp.coef_
    else:
        raise ValueError(f"Unknown method: {method}")


@app.function(
    image=image, gpu="A10G",
    volumes={"/out": out_vol},
    timeout=86400, memory=16384,
)
def run():
    import json, time, itertools, traceback
    from pathlib import Path
    import numpy as np
    import torch
    from tqdm import tqdm
    from scipy.stats import spearmanr

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
    n_heads_per_layer = model.cfg.n_heads
    d_head = model.cfg.d_head
    total_heads = n_layers * n_heads_per_layer

    note(f"Model: {n_layers}L x {n_heads_per_layer}H, d_head={d_head}, "
         f"total={total_heads} heads")

    # ── Step 1: Generate IOI prompts ──────────────────────────────────
    note("Generating IOI prompts")
    prompts, io_token_ids, s_token_ids = generate_ioi_prompts(
        N_PROMPTS, model.tokenizer
    )

    tokens = model.to_tokens(prompts, prepend_bos=True)
    seq_len = tokens.shape[1]
    last_pos = seq_len - 1
    note(f"Tokenized: {tokens.shape} (last_pos={last_pos})")

    io_ids = torch.tensor(io_token_ids, device=device)
    s_ids = torch.tensor(s_token_ids, device=device)
    batch_idx = torch.arange(N_PROMPTS, device=device)

    # ── Step 2: Clean pass + cache mean head outputs ──────────────────
    note("Clean forward pass + caching mean z for all heads")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    clean_ld_per_prompt = (
        logits[:, last_pos, :][batch_idx, io_ids]
        - logits[:, last_pos, :][batch_idx, s_ids]
    ).cpu().numpy()
    baseline_ld = float(clean_ld_per_prompt.mean())
    note(f"Baseline logit diff: {baseline_ld:.4f}")

    mean_z = {}
    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        for head in range(n_heads_per_layer):
            mean_z[(layer, head)] = z[:, :, head, :].mean(dim=0)

    del cache
    torch.cuda.empty_cache()
    note(f"Cached mean z for {len(mean_z)} heads")

    # ── Step 3: Activation patching — find top-20 heads ───────────────
    note("Activation patching for all heads")

    ap_scores = {}

    for layer in tqdm(range(n_layers), desc="AP layers"):
        for head in range(n_heads_per_layer):
            mz = mean_z[(layer, head)]

            def ablation_hook(value, hook, h=head, m=mz):
                value[:, :, h, :] = m.unsqueeze(0)
                return value

            with torch.no_grad():
                abl_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (f"blocks.{layer}.attn.hook_z", ablation_hook)
                    ],
                )

            abl_ld = (
                abl_logits[:, last_pos, :][batch_idx, io_ids]
                - abl_logits[:, last_pos, :][batch_idx, s_ids]
            ).cpu().numpy().mean()

            ap_scores[(layer, head)] = baseline_ld - float(abl_ld)

    sorted_heads = sorted(
        ap_scores.items(), key=lambda x: abs(x[1]), reverse=True
    )
    selected_heads = [h for h, _ in sorted_heads[:N_CIRCUIT_HEADS]]
    selected_ap = {h: ap_scores[h] for h in selected_heads}

    note(f"Top-{N_CIRCUIT_HEADS} heads:")
    for h in selected_heads:
        note(f"  L{h[0]}H{h[1]}: AP={ap_scores[h]:.4f}")

    head_selection = {
        "selected_heads": [(int(l), int(h)) for l, h in selected_heads],
        "ap_scores_all": {
            f"L{l}H{h}": round(float(v), 6)
            for (l, h), v in ap_scores.items()
        },
        "baseline_logit_diff": baseline_ld,
        "n_prompts": N_PROMPTS,
    }
    Path("/out/phase2_head_selection.json").write_text(
        json.dumps(head_selection, indent=1)
    )
    out_vol.commit()
    note("Head selection saved")

    # ── Step 4: Coalition sweep ───────────────────────────────────────
    n_circuit = N_CIRCUIT_HEADS
    max_coalition = 2**n_circuit

    rng = np.random.default_rng(42)
    coalition_indices = rng.choice(max_coalition, size=M_MAX, replace=False)
    coalition_values = np.zeros(M_MAX, dtype=np.float64)

    ckpt_path = Path("/out/phase2_coalition_checkpoint.npz")
    start_idx = 0
    out_vol.reload()
    if ckpt_path.exists():
        ckpt = np.load(str(ckpt_path))
        if np.array_equal(ckpt["coalition_indices"], coalition_indices):
            start_idx = int(ckpt["n_completed"])
            coalition_values[:start_idx] = ckpt["coalition_values"][:start_idx]
            note(f"Resuming from checkpoint: {start_idx}/{M_MAX}")
        else:
            note("Checkpoint has different coalitions, starting fresh")

    layer_to_circuit_heads = {}
    for local_idx, (l, h) in enumerate(selected_heads):
        layer_to_circuit_heads.setdefault(l, []).append((h, local_idx))

    note(f"Coalition sweep: {M_MAX} coalitions, {n_circuit} heads, "
         f"starting from {start_idx}")

    for i in tqdm(
        range(start_idx, M_MAX), desc="Coalitions",
        initial=start_idx, total=M_MAX,
    ):
        c = int(coalition_indices[i])

        fwd_hooks = []
        for layer, head_list in layer_to_circuit_heads.items():
            ablate_pairs = [
                (h_idx, layer)
                for h_idx, local_idx in head_list
                if not (c & (1 << local_idx))
            ]
            if ablate_pairs:
                def make_hook(pairs):
                    def hook_fn(value, hook):
                        for h_idx, l_idx in pairs:
                            value[:, :, h_idx, :] = (
                                mean_z[(l_idx, h_idx)].unsqueeze(0)
                            )
                        return value
                    return hook_fn

                fwd_hooks.append(
                    (f"blocks.{layer}.attn.hook_z", make_hook(ablate_pairs))
                )

        with torch.no_grad():
            abl_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        coalition_values[i] = float(
            (
                abl_logits[:, last_pos, :][batch_idx, io_ids]
                - abl_logits[:, last_pos, :][batch_idx, s_ids]
            ).cpu().numpy().mean()
        )

        if (i + 1) % CHECKPOINT_EVERY == 0:
            np.savez(
                str(ckpt_path),
                coalition_indices=coalition_indices,
                coalition_values=coalition_values,
                n_completed=i + 1,
                selected_heads=np.array(selected_heads),
            )
            out_vol.commit()
            note(f"Checkpoint: {i + 1}/{M_MAX}")

    np.savez(
        str(ckpt_path),
        coalition_indices=coalition_indices,
        coalition_values=coalition_values,
        n_completed=M_MAX,
        selected_heads=np.array(selected_heads),
    )
    out_vol.commit()
    note(f"Coalition sweep complete: {M_MAX} evaluations")

    # ── Step 5: Sparse Walsh recovery ─────────────────────────────────
    note("Sparse Walsh recovery")

    lut = _popcount_lut()
    pairs = list(itertools.combinations(range(n_circuit), 2))
    n_pairs = len(pairs)
    k = n_circuit + n_pairs

    order1_indices = np.array(
        [1 << i for i in range(n_circuit)], dtype=np.uint32
    )
    order2_indices = np.array(
        [(1 << i) | (1 << j) for i, j in pairs], dtype=np.uint32
    )
    all_indices = np.concatenate([order1_indices, order2_indices])

    note(f"k={k} coefficients ({n_circuit} order-1 + {n_pairs} order-2)")

    # Best estimate: recover from all M_MAX coalitions
    X_full = walsh_basis_matrix_fast(coalition_indices, all_indices, lut)
    w_best_lasso = sparse_recover(X_full, coalition_values, k, "lasso")
    w_best_omp = sparse_recover(X_full, coalition_values, k, "omp")
    w_best = w_best_lasso

    note(f"Best-estimate Walsh (LASSO, M={M_MAX}): "
         f"order-1 energy={np.sum(w_best[:n_circuit]**2):.6f}, "
         f"order-2 energy={np.sum(w_best[n_circuit:]**2):.6f}")

    # Recovery at subsample sizes
    recovery_results = []

    for M in SAMPLE_SIZES:
        if M > M_MAX:
            continue

        metrics = {m: {"r": [], "top10": []} for m in ["lasso", "omp", "mc"]}
        top10_best = set(np.argsort(np.abs(w_best))[-10:])

        for trial in range(N_TRIALS):
            trial_rng = np.random.default_rng(42 + trial)
            idx = trial_rng.choice(M_MAX, size=M, replace=False)
            X = walsh_basis_matrix_fast(
                coalition_indices[idx], all_indices, lut
            )
            y = coalition_values[idx]

            for method in ["lasso", "omp", "mc"]:
                try:
                    w = sparse_recover(X, y, k, method)
                except Exception:
                    w = np.zeros(k)
                r = float(np.corrcoef(w_best, w)[0, 1])
                if np.isnan(r):
                    r = 0.0
                metrics[method]["r"].append(r)
                top10_rec = set(np.argsort(np.abs(w))[-10:])
                metrics[method]["top10"].append(
                    len(top10_best & top10_rec) / 10
                )

        row = {"M": M, "M_over_k": round(M / k, 2)}
        for method in ["lasso", "omp", "mc"]:
            row[f"{method}_r_mean"] = round(
                float(np.mean(metrics[method]["r"])), 4
            )
            row[f"{method}_r_std"] = round(
                float(np.std(metrics[method]["r"])), 4
            )
            row[f"{method}_top10"] = round(
                float(np.mean(metrics[method]["top10"])), 3
            )

        recovery_results.append(row)
        note(
            f"M={M:>5} (M/k={M / k:.1f}): "
            f"LASSO r={row['lasso_r_mean']:.3f} "
            f"OMP r={row['omp_r_mean']:.3f} "
            f"MC r={row['mc_r_mean']:.3f}"
        )

    # Split-half validation
    note("Split-half validation")
    half = M_MAX // 2
    split_rng = np.random.default_rng(99)
    perm = split_rng.permutation(M_MAX)
    idx_a, idx_b = perm[:half], perm[half:]

    X_a = walsh_basis_matrix_fast(
        coalition_indices[idx_a], all_indices, lut
    )
    X_b = walsh_basis_matrix_fast(
        coalition_indices[idx_b], all_indices, lut
    )

    w_a_lasso = sparse_recover(X_a, coalition_values[idx_a], k, "lasso")
    w_b_lasso = sparse_recover(X_b, coalition_values[idx_b], k, "lasso")
    w_a_omp = sparse_recover(X_a, coalition_values[idx_a], k, "omp")
    w_b_omp = sparse_recover(X_b, coalition_values[idx_b], k, "omp")

    split_r_lasso = float(np.corrcoef(w_a_lasso, w_b_lasso)[0, 1])
    split_r_omp = float(np.corrcoef(w_a_omp, w_b_omp)[0, 1])
    split_rho_lasso = float(
        spearmanr(w_a_lasso, w_b_lasso).statistic
    )
    split_rho_omp = float(spearmanr(w_a_omp, w_b_omp).statistic)

    note(f"Split-half: LASSO r={split_r_lasso:.4f} rho={split_rho_lasso:.4f}, "
         f"OMP r={split_r_omp:.4f} rho={split_rho_omp:.4f}")

    Path("/out/phase2_sparse_recovery.json").write_text(
        json.dumps(
            {
                "recovery_results": recovery_results,
                "split_half": {
                    "lasso_pearson": round(split_r_lasso, 4),
                    "lasso_spearman": round(split_rho_lasso, 4),
                    "omp_pearson": round(split_r_omp, 4),
                    "omp_spearman": round(split_rho_omp, 4),
                    "half_size": half,
                },
            },
            indent=1,
        )
    )
    out_vol.commit()

    # ── Step 6: Exact marginal pairwise validation ────────────────────
    note(f"Exact marginal interaction for top-{N_EXACT_PAIRS} pairs")

    w_order2 = w_best[n_circuit:]
    top_pair_idx = np.argsort(np.abs(w_order2))[-N_EXACT_PAIRS:][::-1]

    exact_results = []

    for rank, pidx in enumerate(
        tqdm(top_pair_idx, desc="Exact pairs")
    ):
        i, j = pairs[pidx]
        hi = selected_heads[i]
        hj = selected_heads[j]

        ld_configs = {}
        for i_active, j_active in [
            (True, True), (False, True), (True, False), (False, False),
        ]:
            fwd_hooks = []
            if not i_active:
                li, hhi = hi
                mz_i = mean_z[(li, hhi)]

                def hook_i(value, hook, h=hhi, m=mz_i):
                    value[:, :, h, :] = m.unsqueeze(0)
                    return value

                fwd_hooks.append(
                    (f"blocks.{li}.attn.hook_z", hook_i)
                )
            if not j_active:
                lj, hhj = hj
                mz_j = mean_z[(lj, hhj)]

                if not i_active and li == lj:
                    old_hook = fwd_hooks.pop()[1]

                    def combined_hook(value, hook, oh=old_hook, h=hhj, m=mz_j):
                        value = oh(value, hook)
                        value[:, :, h, :] = m.unsqueeze(0)
                        return value

                    fwd_hooks.append(
                        (f"blocks.{lj}.attn.hook_z", combined_hook)
                    )
                else:
                    def hook_j(value, hook, h=hhj, m=mz_j):
                        value[:, :, h, :] = m.unsqueeze(0)
                        return value

                    fwd_hooks.append(
                        (f"blocks.{lj}.attn.hook_z", hook_j)
                    )

            with torch.no_grad():
                out = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

            ld = float(
                (
                    out[:, last_pos, :][batch_idx, io_ids]
                    - out[:, last_pos, :][batch_idx, s_ids]
                ).cpu().numpy().mean()
            )
            ld_configs[(i_active, j_active)] = ld

        exact_inter = (
            ld_configs[(True, True)]
            - ld_configs[(False, True)]
            - ld_configs[(True, False)]
            + ld_configs[(False, False)]
        )

        exact_results.append(
            {
                "rank": int(rank),
                "head_i": [int(hi[0]), int(hi[1])],
                "head_j": [int(hj[0]), int(hj[1])],
                "walsh_recovered": round(float(w_order2[pidx]), 6),
                "exact_marginal": round(float(exact_inter), 6),
                "ld_both_active": round(ld_configs[(True, True)], 6),
                "ld_i_ablated": round(ld_configs[(False, True)], 6),
                "ld_j_ablated": round(ld_configs[(True, False)], 6),
                "ld_both_ablated": round(ld_configs[(False, False)], 6),
            }
        )

    walsh_vals = [e["walsh_recovered"] for e in exact_results]
    exact_vals = [e["exact_marginal"] for e in exact_results]
    walsh_exact_r = float(np.corrcoef(walsh_vals, exact_vals)[0, 1])
    walsh_exact_rho = float(spearmanr(walsh_vals, exact_vals).statistic)
    note(f"Walsh vs exact marginal: r={walsh_exact_r:.3f} rho={walsh_exact_rho:.3f}")

    Path("/out/phase2_exact_validation.json").write_text(
        json.dumps(exact_results, indent=1)
    )
    out_vol.commit()

    # ── Step 7: AP-product comparison ─────────────────────────────────
    note("AP-product vs Walsh pairwise comparison")

    ap_product = np.array([
        selected_ap[selected_heads[i]] * selected_ap[selected_heads[j]]
        for i, j in pairs
    ])

    rho_walsh_ap = float(
        spearmanr(np.abs(w_order2), np.abs(ap_product)).statistic
    )
    r_walsh_ap = float(
        np.corrcoef(np.abs(w_order2), np.abs(ap_product))[0, 1]
    )
    note(f"Walsh vs AP-product: rho={rho_walsh_ap:.3f} r={r_walsh_ap:.3f}")

    # ── Save summary ──────────────────────────────────────────────────
    w_order1 = w_best[:n_circuit]

    top20_pairs = []
    for idx in np.argsort(np.abs(w_order2))[-20:][::-1]:
        i, j = pairs[idx]
        hi = selected_heads[i]
        hj = selected_heads[j]
        top20_pairs.append(
            {
                "pair": f"L{hi[0]}H{hi[1]}-L{hj[0]}H{hj[1]}",
                "coeff": round(float(w_order2[idx]), 6),
                "ap_product": round(float(ap_product[idx]), 6),
            }
        )

    summary = {
        "experiment": "Phase 2: 20-head IOI sparse Walsh",
        "prereg_sha": "d8d0f11b",
        "n_circuit_heads": N_CIRCUIT_HEADS,
        "n_prompts": N_PROMPTS,
        "M_max": M_MAX,
        "n_pairs": n_pairs,
        "k": k,
        "baseline_logit_diff": round(baseline_ld, 4),
        "selected_heads": [
            {"head": f"L{l}H{h}", "ap": round(float(selected_ap[(l, h)]), 4)}
            for l, h in selected_heads
        ],
        "recovery": recovery_results,
        "split_half": {
            "lasso_r": round(split_r_lasso, 4),
            "omp_r": round(split_r_omp, 4),
        },
        "walsh_vs_exact_marginal": {
            "pearson": round(walsh_exact_r, 4),
            "spearman": round(walsh_exact_rho, 4),
        },
        "walsh_vs_ap_product": {
            "spearman": round(rho_walsh_ap, 4),
            "pearson": round(r_walsh_ap, 4),
        },
        "top20_interacting_pairs": top20_pairs,
        "order1_scores": {
            f"L{selected_heads[i][0]}H{selected_heads[i][1]}": round(
                float(w_order1[i]), 6
            )
            for i in range(n_circuit)
        },
    }

    Path("/out/phase2_summary.json").write_text(
        json.dumps(summary, indent=1)
    )
    out_vol.commit()
    note("All results saved. Done.")

    return summary


@app.local_entrypoint()
def main():
    import json

    result = run.remote()
    if not result:
        print("No results returned.")
        return

    print("\n=== PHASE 2 SUMMARY ===")
    print(f"Circuit: {result['n_circuit_heads']} heads, "
          f"{result['n_pairs']} pairs, k={result['k']}")
    print(f"Baseline LD: {result['baseline_logit_diff']:.4f}")

    print(f"\nTop-{result['n_circuit_heads']} heads:")
    for h in result["selected_heads"]:
        print(f"  {h['head']}: AP={h['ap']:.4f}")

    print(f"\nSparse recovery (vs M={result['M_max']} best estimate):")
    for r in result["recovery"]:
        print(
            f"  M={r['M']:>5} (M/k={r['M_over_k']:>4.1f}): "
            f"LASSO={r['lasso_r_mean']:.3f} "
            f"OMP={r['omp_r_mean']:.3f} "
            f"MC={r['mc_r_mean']:.3f}"
        )

    sh = result["split_half"]
    print(f"\nSplit-half: LASSO r={sh['lasso_r']:.4f} OMP r={sh['omp_r']:.4f}")

    we = result["walsh_vs_exact_marginal"]
    print(f"Walsh vs exact marginal: r={we['pearson']:.3f} rho={we['spearman']:.3f}")

    wa = result["walsh_vs_ap_product"]
    print(f"Walsh vs AP-product: rho={wa['spearman']:.3f}")

    print(f"\nTop-20 interacting pairs:")
    for p in result["top20_interacting_pairs"]:
        print(f"  {p['pair']}: w={p['coeff']:.4f} ap_prod={p['ap_product']:.4f}")
