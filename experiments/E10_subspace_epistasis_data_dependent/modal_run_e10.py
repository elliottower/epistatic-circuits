"""E10: Data-dependent subspace epistasis with positive control.

Pre-registered analysis (PREREG.md, SHA 4cc6d1d1...).
Tests whether data-dependent subspace features predict pairwise Walsh
coefficients where static weight geometry (Section 3.3) fails. Includes
a positive control arm to confirm the evaluation protocol can detect signal.

Usage:
    cd epistatic-circuits
    modal run --detach experiments/E10_subspace_epistasis_data_dependent/modal_run_e10.py
"""

import modal

app = modal.App("e10-subspace-epistasis-data-dependent")

output_volume = modal.Volume.from_name("e10-subspace-epistasis-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-learn==1.5.2",
        "tqdm==4.67.1",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "typeguard==4.3.0",
        "matplotlib==3.9.4",
    )
    .add_local_file(
        "results/phase2/phase2_all_walsh_coefficients.json",
        remote_path="/data/phase2_all_walsh_coefficients.json",
    )
    .add_local_file(
        "results/phase2/phase2_head_selection.json",
        remote_path="/data/phase2_head_selection.json",
    )
    .add_local_file(
        "results/phase2/phase2_summary.json",
        remote_path="/data/phase2_summary.json",
    )
    .add_local_file(
        "data/ioi_prompts_200.json",
        remote_path="/data/ioi_prompts_200.json",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=86400,
    volumes={"/results": output_volume},
)
def run_e10():
    import json
    import time
    from itertools import combinations

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from tqdm import tqdm

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts()}] E10: Data-dependent subspace epistasis analysis")

    # ── Load existing data ──────────────────────────────────────────────

    with open("/data/phase2_all_walsh_coefficients.json") as f:
        walsh_data = json.load(f)

    with open("/data/phase2_head_selection.json") as f:
        head_sel = json.load(f)

    with open("/data/phase2_summary.json") as f:
        summary = json.load(f)

    heads = [tuple(h) for h in head_sel["selected_heads"]]
    n_circuit = len(heads)
    head_to_idx = {h: i for i, h in enumerate(heads)}
    print(f"[{ts()}] Circuit: {n_circuit} heads, {n_circuit*(n_circuit-1)//2} pairs")

    order2 = {}
    order1 = {}
    for key, entry in walsh_data.items():
        if entry["order"] == 2:
            h1_str, h2_str = key.split("-")
            order2[(h1_str, h2_str)] = entry["coeff"]
        elif entry["order"] == 1:
            order1[key] = entry["coeff"]

    pairs = list(combinations(range(n_circuit), 2))
    n_pairs = len(pairs)
    print(f"[{ts()}] {n_pairs} order-2 coefficients loaded")

    def head_label(h):
        return f"L{h[0]}H{h[1]}"

    w_ij = np.zeros(n_pairs)
    pair_labels = []
    for p_idx, (i, j) in enumerate(pairs):
        hi, hj = heads[i], heads[j]
        label_i, label_j = head_label(hi), head_label(hj)
        key_fwd = (label_i, label_j)
        key_rev = (label_j, label_i)
        if key_fwd in order2:
            w_ij[p_idx] = order2[key_fwd]
        elif key_rev in order2:
            w_ij[p_idx] = order2[key_rev]
        else:
            print(f"  WARNING: missing pair {label_i}-{label_j}")
            w_ij[p_idx] = 0.0
        pair_labels.append(f"{label_i}-{label_j}")

    print(f"[{ts()}] Target w_ij: mean={w_ij.mean():.4f}, std={w_ij.std():.4f}")

    # ── AP scores for positive control ──────────────────────────────────

    ap_scores = head_sel["ap_scores_all"]

    # ── Load GPT-2 small ────────────────────────────────────────────────

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", fold_ln=True, center_writing_weights=True,
        center_unembed=True, device="cuda",
    )
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads_per_layer = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    print(f"[{ts()}] GPT-2 loaded: {n_layers}L x {n_heads_per_layer}H, d={d_model}, d_head={d_head}")

    # ── ARM 1: Weight-only features ─────────────────────────────────────

    print(f"[{ts()}] Computing Arm 1: weight-only features")

    import torch

    W_OV = {}
    W_O_all = {}
    W_Q_all = {}
    W_K_all = {}
    for l in range(n_layers):
        W_OV[l] = {}
        W_O_all[l] = {}
        W_Q_all[l] = {}
        W_K_all[l] = {}
        for h in range(n_heads_per_layer):
            wv = model.W_V[l, h].detach().cpu().numpy()
            wo = model.W_O[l, h].detach().cpu().numpy()
            W_OV[l][h] = wv @ wo
            W_O_all[l][h] = wo
            W_Q_all[l][h] = model.W_Q[l, h].detach().cpu().numpy()
            W_K_all[l][h] = model.W_K[l, h].detach().cpu().numpy()

    def ov_overlap(hi, hj):
        ov_i = W_OV[hi[0]][hi[1]]
        ov_j = W_OV[hj[0]][hj[1]]
        numer = np.linalg.norm(ov_i.T @ ov_j, "fro")
        denom = np.linalg.norm(ov_i, "fro") * np.linalg.norm(ov_j, "fro")
        return float(numer / denom) if denom > 0 else 0.0

    def q_comp(hi, hj):
        if hi[0] >= hj[0]:
            return 0.0
        ov_i = W_OV[hi[0]][hi[1]]
        wq_j = W_Q_all[hj[0]][hj[1]]
        numer = np.linalg.norm(ov_i @ wq_j, "fro")
        denom = np.linalg.norm(ov_i, "fro") * np.linalg.norm(wq_j, "fro")
        return float(numer / denom) if denom > 0 else 0.0

    def k_comp(hi, hj):
        if hi[0] >= hj[0]:
            return 0.0
        ov_i = W_OV[hi[0]][hi[1]]
        wk_j = W_K_all[hj[0]][hj[1]]
        numer = np.linalg.norm(ov_i @ wk_j, "fro")
        denom = np.linalg.norm(ov_i, "fro") * np.linalg.norm(wk_j, "fro")
        return float(numer / denom) if denom > 0 else 0.0

    X_arm1 = np.zeros((n_pairs, 4))
    for p_idx, (i, j) in enumerate(pairs):
        hi, hj = heads[i], heads[j]
        X_arm1[p_idx, 0] = ov_overlap(hi, hj)
        X_arm1[p_idx, 1] = q_comp(hi, hj) + q_comp(hj, hi)
        X_arm1[p_idx, 2] = k_comp(hi, hj) + k_comp(hj, hi)
        X_arm1[p_idx, 3] = abs(hi[0] - hj[0])

    print(f"[{ts()}] Arm 1 features computed: shape {X_arm1.shape}")

    # ── ARM 2: Data-dependent features ──────────────────────────────────

    print(f"[{ts()}] Computing Arm 2: data-dependent features")
    print(f"[{ts()}] Loading IOI prompts...")

    with open("/data/ioi_prompts_200.json") as f:
        prompts = json.load(f)
    n_prompts = len(prompts)

    print(f"[{ts()}] Running {n_prompts} forward passes to collect activations...")

    hook_z_accum = {h: [] for h in heads}
    attn_pattern_accum = {h: [] for h in heads}

    batch_size = 20
    for batch_start in tqdm(range(0, n_prompts, batch_size), desc="Forward passes"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        tokens = model.to_tokens(batch_prompts)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(
                    name == f"blocks.{h[0]}.attn.hook_z" or
                    name == f"blocks.{h[0]}.attn.hook_pattern"
                    for h in heads
                ),
            )

        for h in heads:
            z = cache[f"blocks.{h[0]}.attn.hook_z"][:, :, h[1], :]
            hook_z_accum[h].append(z.cpu().numpy())

            pattern = cache[f"blocks.{h[0]}.attn.hook_pattern"][:, h[1], :, :]
            attn_pattern_accum[h].append(pattern.cpu().numpy())

        del cache
        torch.cuda.empty_cache()

    hook_z_all = {}
    attn_mean = {}
    for h in heads:
        z = np.concatenate(hook_z_accum[h], axis=0)
        z_avg = z.mean(axis=1)
        hook_z_all[h] = z_avg

        patterns = np.concatenate(attn_pattern_accum[h], axis=0)
        attn_mean[h] = patterns.mean(axis=0)

    del hook_z_accum, attn_pattern_accum
    print(f"[{ts()}] Activations collected for {len(heads)} heads")

    subspace_k = d_head // 2
    U_heads = {}
    for h in heads:
        z = hook_z_all[h]
        U, S, Vt = np.linalg.svd(z, full_matrices=False)
        U_heads[h] = Vt[:subspace_k, :].T

    def grassmannian_distance(hi, hj):
        Ui = U_heads[hi]
        Uj = U_heads[hj]
        sigmas = np.linalg.svd(Ui.T @ Uj, compute_uv=False)
        sigmas = np.clip(sigmas, -1, 1)
        angles = np.arccos(sigmas)
        return float(np.sqrt(np.sum(angles ** 2)))

    def subspace_overlap_frob(hi, hj):
        Ui = U_heads[hi]
        Uj = U_heads[hj]
        ki = Ui.shape[1]
        kj = Uj.shape[1]
        numer = np.linalg.norm(Ui.T @ Uj, "fro")
        denom = np.sqrt(ki * kj)
        return float(numer / denom) if denom > 0 else 0.0

    def attn_pattern_similarity(hi, hj):
        pi = attn_mean[hi].flatten()
        pj = attn_mean[hj].flatten()
        min_len = min(len(pi), len(pj))
        pi = pi[:min_len]
        pj = pj[:min_len]
        rho, _ = stats.spearmanr(pi, pj)
        return float(rho) if not np.isnan(rho) else 0.0

    def data_dependent_comp(hi, hj):
        if hi[0] >= hj[0]:
            return 0.0
        z_i_mean = hook_z_all[hi].mean(axis=0)  # (d_head,)
        wo_i = W_O_all[hi[0]][hi[1]]  # (d_head, d_model)
        resid_i = z_i_mean @ wo_i  # (d_model,)
        wq_j = W_Q_all[hj[0]][hj[1]]  # (d_model, d_head)
        projected = resid_i @ wq_j  # (d_head,)
        numer = float(np.sum(projected ** 2))
        denom = float(np.sum(resid_i ** 2)) * float(np.sum(wq_j ** 2))
        return numer / denom if denom > 0 else 0.0

    X_arm2 = np.zeros((n_pairs, 4))
    for p_idx, (i, j) in enumerate(pairs):
        hi, hj = heads[i], heads[j]
        X_arm2[p_idx, 0] = grassmannian_distance(hi, hj)
        X_arm2[p_idx, 1] = subspace_overlap_frob(hi, hj)
        X_arm2[p_idx, 2] = attn_pattern_similarity(hi, hj)
        X_arm2[p_idx, 3] = data_dependent_comp(hi, hj) + data_dependent_comp(hj, hi)

    print(f"[{ts()}] Arm 2 features computed: shape {X_arm2.shape}")

    # ── ARM 3: Positive control ─────────────────────────────────────────

    print(f"[{ts()}] Computing Arm 3: positive control features")

    X_arm3 = np.zeros((n_pairs, 2))
    for p_idx, (i, j) in enumerate(pairs):
        hi, hj = heads[i], heads[j]
        li, lj = head_label(hi), head_label(hj)
        w1_i = abs(order1.get(li, 0.0))
        w1_j = abs(order1.get(lj, 0.0))
        X_arm3[p_idx, 0] = w1_i * w1_j

        ap_i = abs(ap_scores.get(li, 0.0))
        ap_j = abs(ap_scores.get(lj, 0.0))
        X_arm3[p_idx, 1] = ap_i * ap_j

    print(f"[{ts()}] Arm 3 features computed: shape {X_arm3.shape}")

    # ── ARM 4: Combined ─────────────────────────────────────────────────

    X_arm4 = np.hstack([X_arm1, X_arm2, X_arm3])
    print(f"[{ts()}] Arm 4 (combined) features: shape {X_arm4.shape}")

    # ── Leave-both-heads-out CV ─────────────────────────────────────────

    print(f"[{ts()}] Running leave-both-heads-out cross-validation...")

    def leave_both_heads_out_cv(X, y, pairs, n_circuit):
        y_pred = np.full(len(y), np.nan)

        for p_idx, (i, j) in enumerate(pairs):
            train_mask = np.array([
                (pi != i and pi != j and pj != i and pj != j)
                for pi, pj in pairs
            ])
            if train_mask.sum() < X.shape[1] + 1:
                continue

            X_train = X[train_mask]
            y_train = y[train_mask]

            mu_x = X_train.mean(axis=0)
            std_x = X_train.std(axis=0)
            std_x[std_x == 0] = 1.0
            X_train_s = (X_train - mu_x) / std_x

            reg = LinearRegression()
            reg.fit(X_train_s, y_train)

            x_test = (X[p_idx] - mu_x) / std_x
            y_pred[p_idx] = reg.predict(x_test.reshape(1, -1))[0]

        valid = ~np.isnan(y_pred)
        if valid.sum() == 0:
            return {"cv_r2": float("nan"), "in_sample_r2": float("nan")}

        ss_res = np.sum((y[valid] - y_pred[valid]) ** 2)
        ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
        cv_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        mu_x = X.mean(axis=0)
        std_x = X.std(axis=0)
        std_x[std_x == 0] = 1.0
        X_s = (X - mu_x) / std_x
        reg_full = LinearRegression().fit(X_s, y)
        y_hat = reg_full.predict(X_s)
        ss_res_in = np.sum((y - y_hat) ** 2)
        ss_tot_in = np.sum((y - y.mean()) ** 2)
        in_sample_r2 = 1 - ss_res_in / ss_tot_in if ss_tot_in > 0 else float("nan")

        return {
            "cv_r2": float(cv_r2),
            "in_sample_r2": float(in_sample_r2),
            "n_valid": int(valid.sum()),
            "n_train_per_fold": int(np.mean([
                sum(1 for pi, pj in pairs if pi != i and pi != j and pj != i and pj != j)
                for i, j in pairs
            ])),
        }

    def jackknife_se_cv_r2(X, y, pairs, n_circuit):
        n = len(y)
        full_result = leave_both_heads_out_cv(X, y, pairs, n_circuit)
        full_r2 = full_result["cv_r2"]

        r2_drop = []
        for drop_idx in range(n):
            mask = np.ones(n, dtype=bool)
            mask[drop_idx] = False
            pairs_sub = [(pi, pj) for k, (pi, pj) in enumerate(pairs) if mask[k]]
            idx_map = {}
            counter = 0
            for pi, pj in pairs_sub:
                if pi not in idx_map:
                    idx_map[pi] = counter
                    counter += 1
                if pj not in idx_map:
                    idx_map[pj] = counter
                    counter += 1
            result = leave_both_heads_out_cv(X[mask], y[mask], pairs_sub, n_circuit)
            r2_drop.append(result["cv_r2"])

        r2_drop = np.array(r2_drop)
        valid_drops = ~np.isnan(r2_drop)
        if valid_drops.sum() < 2:
            return float("nan")
        pseudo = n * full_r2 - (n - 1) * r2_drop[valid_drops]
        se = float(np.std(pseudo, ddof=1) / np.sqrt(valid_drops.sum()))
        return se

    arms = {
        "arm1_weight_only": X_arm1,
        "arm2_data_dependent": X_arm2,
        "arm3_positive_control": X_arm3,
        "arm4_combined": X_arm4,
    }
    arm_names_display = {
        "arm1_weight_only": "Arm 1 (weight-only, replication)",
        "arm2_data_dependent": "Arm 2 (data-dependent)",
        "arm3_positive_control": "Arm 3 (positive control)",
        "arm4_combined": "Arm 4 (combined)",
    }
    feature_names = {
        "arm1_weight_only": ["ov_overlap", "q_comp", "k_comp", "layer_distance"],
        "arm2_data_dependent": ["grassmannian_dist", "subspace_overlap", "attn_pattern_sim", "data_dep_comp"],
        "arm3_positive_control": ["order1_mag_product", "ap_mag_product"],
        "arm4_combined": [
            "ov_overlap", "q_comp", "k_comp", "layer_distance",
            "grassmannian_dist", "subspace_overlap", "attn_pattern_sim", "data_dep_comp",
            "order1_mag_product", "ap_mag_product",
        ],
    }

    results = {
        "experiment": "E10: Data-dependent subspace epistasis with positive control",
        "prereg_sha": "4cc6d1d1",
        "n_heads": n_circuit,
        "n_pairs": n_pairs,
        "n_prompts": n_prompts,
        "subspace_k": subspace_k,
        "target_mean": float(w_ij.mean()),
        "target_std": float(w_ij.std()),
        "arms": {},
    }

    for arm_key, X in arms.items():
        print(f"\n[{ts()}] === {arm_names_display[arm_key]} ===")
        cv_result = leave_both_heads_out_cv(X, w_ij, pairs, n_circuit)
        print(f"  CV R^2:        {cv_result['cv_r2']:.4f}")
        print(f"  In-sample R^2: {cv_result['in_sample_r2']:.4f}")
        print(f"  N valid:       {cv_result.get('n_valid', 'N/A')}")
        print(f"  N train/fold:  {cv_result.get('n_train_per_fold', 'N/A')}")

        print(f"  Computing jackknife SE...")
        se = jackknife_se_cv_r2(X, w_ij, pairs, n_circuit)
        print(f"  Jackknife SE:  {se:.4f}")

        results["arms"][arm_key] = {
            "display_name": arm_names_display[arm_key],
            "features": feature_names[arm_key],
            "n_features": X.shape[1],
            **cv_result,
            "jackknife_se": float(se),
        }

    # ── Hypothesis adjudication ─────────────────────────────────────────

    r2_arm3 = results["arms"]["arm3_positive_control"]["cv_r2"]
    r2_arm2 = results["arms"]["arm2_data_dependent"]["cv_r2"]
    r2_arm1 = results["arms"]["arm1_weight_only"]["cv_r2"]

    if r2_arm3 <= 0:
        verdict = "H4: Uninformative null. Positive control failed (CV R^2 <= 0). The evaluation protocol cannot detect signal at this sample size."
    elif r2_arm2 > 0.10:
        verdict = "H1 strongly confirmed. Data-dependent features achieve CV R^2 > 0.10."
    elif r2_arm2 > 0:
        verdict = "H1 confirmed. Data-dependent features achieve positive CV R^2."
    else:
        verdict = "H3 confirmed. Positive control passes but data-dependent features fail. Interaction is not decomposable into pairwise subspace relationships."

    if r2_arm1 > 0:
        verdict += " NOTE: Arm 1 (weight-only) returned positive CV R^2, contradicting Section 3.3."

    results["verdict"] = verdict
    print(f"\n[{ts()}] VERDICT: {verdict}")

    # ── Feature correlations (Exploratory 1) ────────────────────────────

    print(f"\n[{ts()}] Exploratory 1: Feature correlation matrix")
    all_features = np.hstack([X_arm1, X_arm2, X_arm3])
    all_names = feature_names["arm1_weight_only"] + feature_names["arm2_data_dependent"] + feature_names["arm3_positive_control"]
    corr_matrix = np.corrcoef(all_features.T)
    results["exploratory_1_feature_correlations"] = {
        "names": all_names,
        "correlation_matrix": corr_matrix.tolist(),
    }
    for i_f in range(len(all_names)):
        for j_f in range(i_f + 1, len(all_names)):
            r = corr_matrix[i_f, j_f]
            if abs(r) > 0.5:
                print(f"  {all_names[i_f]} <-> {all_names[j_f]}: r = {r:.3f}")

    # ── Save results ────────────────────────────────────────────────────

    output_path = "/results/e10_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    output_volume.commit()
    print(f"\n[{ts()}] Results saved to {output_path}")

    # ── Plot ────────────────────────────────────────────────────────────

    # Pre-declared feature per arm for scatter (not post-hoc selected)
    plot_feature_idx = {
        "arm1_weight_only": 0,       # OV overlap
        "arm2_data_dependent": 0,    # Grassmannian distance
        "arm3_positive_control": 0,  # order-1 magnitude product
        "arm4_combined": 0,          # OV overlap
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax_idx, (arm_key, X) in enumerate(arms.items()):
        ax = axes[ax_idx]
        feat_idx = plot_feature_idx[arm_key]
        ax.scatter(X[:, feat_idx], w_ij, alpha=0.4, s=15)
        ax.set_xlabel(feature_names[arm_key][feat_idx])
        ax.set_ylabel("Walsh w_ij")
        r2 = results["arms"][arm_key]["cv_r2"]
        ax.set_title(f"{arm_names_display[arm_key]}\nCV R² = {r2:.3f}")
        ax.axhline(0, color="gray", lw=0.5)

    plt.tight_layout()
    fig.savefig("/results/e10_scatter.png", dpi=150)
    output_volume.commit()
    print(f"[{ts()}] Plot saved")

    # ── Summary table ───────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"{'Arm':<35} {'CV R²':>8} {'SE':>8} {'In-sample':>10}")
    print(f"{'-'*60}")
    for arm_key in arms:
        a = results["arms"][arm_key]
        print(f"{a['display_name']:<35} {a['cv_r2']:>8.4f} {a['jackknife_se']:>8.4f} {a['in_sample_r2']:>10.4f}")
    print(f"{'='*60}")
    print(f"\nVerdict: {verdict}")
    print(f"\n[{ts()}] E10 complete.")

    return results


@app.local_entrypoint()
def main():
    results = run_e10.remote()
    import json
    local_path = "experiments/E10_subspace_epistasis_data_dependent/results/e10_results.json"
    import os
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved locally to {local_path}")
