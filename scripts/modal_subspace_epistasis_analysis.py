"""Modal: Test whether head-level epistasis is predicted by residual-stream subspace overlap.

Pre-registered analysis (PREREG_SUBSPACE_EPISTASIS.md, SHA 3ef962bc...).
Tests P1-P5: OV overlap, QK composition, layer distance vs Walsh pairwise coefficients.

Steps:
1. Load GPT-2 with fold_ln=True (LN affine params folded into weights)
2. Compute W_OV, Q-comp, K-comp for all head pairs
3. Load coalition NPZs from volumes, compute Walsh pairwise coefficients
4. Correlate subspace overlap with Walsh interaction strength
5. Run linear regression (P4)

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_subspace_epistasis_analysis.py
"""

import modal

app = modal.App("subspace-epistasis-analysis")

output_volume = modal.Volume.from_name("subspace-epistasis-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "tqdm==4.67.1",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "typeguard==4.3.0",
        "matplotlib==3.9.4",
    )
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
)

DATA_VOLUMES = {
    "/vol/ioi-resample": modal.Volume.from_name("ioi-resample-sweep"),
    "/vol/rti-resample": modal.Volume.from_name("rti-resample-sweep"),
    "/vol/gt-resample": modal.Volume.from_name("gt-resample-sweep"),
}

CIRCUITS = {
    "ioi_C3_canonical": {
        "task": "ioi", "ablation": "resample",
        "heads": [
            (0, 1), (3, 0), (2, 2), (4, 11), (5, 5), (6, 9),
            (7, 3), (7, 9), (8, 6), (8, 10), (9, 9), (9, 6), (10, 0),
            (10, 7), (11, 10),
        ],
        "npz_path": "/vol/ioi-resample/ioi_C3_canonical_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "ioi_C2_eap": {
        "task": "ioi", "ablation": "resample",
        "heads": [
            (0, 1), (0, 10), (2, 2), (4, 11), (5, 5), (5, 8), (5, 9), (6, 1),
            (6, 9), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7), (11, 10),
        ],
        "npz_path": "/vol/ioi-resample/ioi_C2_eap_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "ioi_C5_walsh": {
        "task": "ioi", "ablation": "resample",
        "heads": [
            (5, 5), (10, 7), (11, 1), (8, 6), (8, 10),
            (0, 9), (7, 9), (0, 3), (6, 9), (10, 1),
            (11, 2), (10, 10), (3, 0), (11, 10), (4, 0),
        ],
        "npz_path": "/vol/ioi-resample/ioi_C5_walsh_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "ioi_C6_epistatic": {
        "task": "ioi", "ablation": "resample",
        "heads": [
            (5, 5), (8, 6), (11, 10), (10, 7), (6, 9),
            (0, 10), (0, 1), (10, 0), (5, 9), (8, 10),
            (11, 2), (9, 9), (0, 9), (7, 9), (4, 0),
        ],
        "npz_path": "/vol/ioi-resample/ioi_C6_epistatic_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "ioi_C4_random": {
        "task": "ioi", "ablation": "resample",
        "heads": [
            (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
            (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
        ],
        "npz_path": "/vol/ioi-resample/ioi_C4_random_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "rti_known": {
        "task": "rti", "ablation": "resample",
        "heads": [
            (0, 8), (0, 9), (0, 11), (4, 11),
            (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
            (10, 11), (11, 9), (11, 11),
        ],
        "npz_path": "/vol/rti-resample/rti_known_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "rti_EAP": {
        "task": "rti", "ablation": "resample",
        "heads": [
            (0, 9), (0, 10), (5, 9), (0, 8), (0, 6), (1, 10), (2, 10),
            (0, 11), (1, 3), (0, 1), (4, 7), (2, 0), (0, 5), (0, 3), (2, 8),
        ],
        "npz_path": "/vol/rti-resample/rti_EAP_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "rti_C5_walsh": {
        "task": "rti", "ablation": "resample",
        "heads": [
            (0, 9), (11, 2), (4, 11), (10, 6), (7, 9), (10, 7), (9, 9),
            (5, 6), (4, 0), (2, 11), (8, 7), (1, 5), (11, 10), (1, 3), (6, 11),
        ],
        "npz_path": "/vol/rti-resample/rti_C5_walsh_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "rti_C6_epistatic": {
        "task": "rti", "ablation": "resample",
        "heads": [
            (0, 10), (0, 9), (11, 2), (10, 7), (10, 0), (4, 11), (11, 10),
            (0, 1), (0, 3), (9, 9), (1, 11), (9, 6), (2, 2), (5, 6), (2, 11),
        ],
        "npz_path": "/vol/rti-resample/rti_C6_epistatic_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "rti_random": {
        "task": "rti", "ablation": "resample",
        "heads": [
            (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
            (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
        ],
        "npz_path": "/vol/rti-resample/rti_random_resample_coalition_values.npz",
        "value_key": "logit_diff",
    },
    "gt_known": {
        "task": "gt", "ablation": "resample",
        "heads": [(5, 1), (5, 5), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)],
        "npz_path": "/vol/gt-resample/gt_known_resample_coalition_values.npz",
        "value_key": "prob_diff",
    },
    "gt_acdc": {
        "task": "gt", "ablation": "resample",
        "heads": [(0, 7), (3, 6), (4, 11), (5, 5), (7, 10), (8, 10), (9, 1)],
        "npz_path": "/vol/gt-resample/gt_acdc_resample_coalition_values.npz",
        "value_key": "prob_diff",
    },
    "gt_c5_walsh": {
        "task": "gt", "ablation": "resample",
        "heads": [(5, 5), (7, 10), (9, 1), (6, 9), (0, 10), (8, 5), (10, 2)],
        "npz_path": "/vol/gt-resample/gt_c5_walsh_resample_coalition_values.npz",
        "value_key": "prob_diff",
    },
    "gt_c6_epistatic": {
        "task": "gt", "ablation": "resample",
        "heads": [(5, 5), (0, 10), (7, 10), (9, 1), (4, 11), (0, 3), (6, 9)],
        "npz_path": "/vol/gt-resample/gt_c6_epistatic_resample_coalition_values.npz",
        "value_key": "prob_diff",
    },
    "gt_random": {
        "task": "gt", "ablation": "resample",
        "heads": [(0, 6), (2, 2), (2, 4), (2, 11), (4, 9), (5, 2), (5, 10)],
        "npz_path": "/vol/gt-resample/gt_random_resample_coalition_values.npz",
        "value_key": "prob_diff",
    },
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": output_volume, **DATA_VOLUMES},
)
def analyze():
    import json
    import os
    import sys
    import time

    import numpy as np
    from scipy import stats
    from tqdm import tqdm

    sys.path.insert(0, "/app")
    from walsh import wht

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts()}] Starting subspace epistasis analysis")

    # Step 1: Load GPT-2 with LN folding
    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", fold_ln=True, device="cuda",
    )
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    print(f"[{ts()}] GPT-2 loaded with fold_ln=True: {n_layers}L x {n_heads}H, d={d_model}, d_head={d_head}")

    # Step 2: Precompute W_OV for every head
    # W_OV[l][h] = W_V[l,h] @ W_O[l,h], shape (d_model, d_model)
    W_OV = {}
    for l in range(n_layers):
        W_OV[l] = {}
        for h in range(n_heads):
            wv = model.W_V[l, h].detach().cpu().numpy()  # (d_model, d_head)
            wo = model.W_O[l, h].detach().cpu().numpy()  # (d_head, d_model)
            W_OV[l][h] = wv @ wo  # (d_model, d_model)

    # Also store W_Q, W_K for QK composition
    W_Q = {}
    W_K = {}
    for l in range(n_layers):
        W_Q[l] = {}
        W_K[l] = {}
        for h in range(n_heads):
            W_Q[l][h] = model.W_Q[l, h].detach().cpu().numpy()  # (d_model, d_head)
            W_K[l][h] = model.W_K[l, h].detach().cpu().numpy()  # (d_model, d_head)

    print(f"[{ts()}] W_OV, W_Q, W_K extracted for all {n_layers * n_heads} heads")

    def ov_overlap(head_i, head_j):
        """Normalized Frobenius of W_OV_i^T @ W_OV_j (subspace affinity)."""
        li, hi = head_i
        lj, hj = head_j
        ov_i = W_OV[li][hi]
        ov_j = W_OV[lj][hj]
        product = ov_i.T @ ov_j
        numer = np.linalg.norm(product, 'fro')
        denom = np.linalg.norm(ov_i, 'fro') * np.linalg.norm(ov_j, 'fro')
        return float(numer / denom) if denom > 0 else 0.0

    def q_composition(head_i, head_j):
        """Q-composition: || W_OV_i @ W_Q_j ||_F / norms. Only for l_i < l_j."""
        li, hi = head_i
        lj, hj = head_j
        if li >= lj:
            return 0.0
        ov_i = W_OV[li][hi]  # (d_model, d_model)
        wq_j = W_Q[lj][hj]  # (d_model, d_head)
        product = ov_i @ wq_j  # (d_model, d_head)
        numer = np.linalg.norm(product, 'fro')
        denom = np.linalg.norm(ov_i, 'fro') * np.linalg.norm(wq_j, 'fro')
        return float(numer / denom) if denom > 0 else 0.0

    def k_composition(head_i, head_j):
        """K-composition: || W_OV_i @ W_K_j ||_F / norms. Only for l_i < l_j."""
        li, hi = head_i
        lj, hj = head_j
        if li >= lj:
            return 0.0
        ov_i = W_OV[li][hi]
        wk_j = W_K[lj][hj]
        product = ov_i @ wk_j
        numer = np.linalg.norm(product, 'fro')
        denom = np.linalg.norm(ov_i, 'fro') * np.linalg.norm(wk_j, 'fro')
        return float(numer / denom) if denom > 0 else 0.0

    def principal_angles(head_i, head_j, rank=8):
        """Principal angles between rank-k subspaces of W_OV_i and W_OV_j."""
        li, hi = head_i
        lj, hj = head_j
        ov_i = W_OV[li][hi]
        ov_j = W_OV[lj][hj]
        U_i, _, _ = np.linalg.svd(ov_i, full_matrices=False)
        U_j, _, _ = np.linalg.svd(ov_j, full_matrices=False)
        U_i = U_i[:, :rank]
        U_j = U_j[:, :rank]
        sigmas = np.linalg.svd(U_i.T @ U_j, compute_uv=False)
        sigmas = np.clip(sigmas, 0, 1)
        return np.arccos(sigmas)

    # Step 3: Process each circuit
    all_results = {}
    all_pairs_data = []

    for circuit_name, circuit_info in tqdm(CIRCUITS.items(), desc="Circuits"):
        heads = circuit_info["heads"]
        npz_path = circuit_info["npz_path"]
        value_key = circuit_info["value_key"]
        n_players = len(heads)

        if not os.path.exists(npz_path):
            print(f"[{ts()}] SKIP {circuit_name}: {npz_path} not found")
            continue

        data = np.load(npz_path)
        if "n_completed" in data:
            n_completed = int(data["n_completed"])
            n_total = 2 ** n_players
            if n_completed < n_total:
                print(f"[{ts()}] SKIP {circuit_name}: incomplete ({n_completed}/{n_total})")
                continue

        raw_values = data[value_key]  # (2^n, n_prompts)
        mean_values = raw_values.mean(axis=1)  # (2^n,)

        # Faithfulness
        full_idx = (1 << n_players) - 1
        faith = float(mean_values[full_idx] - mean_values[0])

        # Walsh-Hadamard transform
        w_coeffs = wht(mean_values)
        # Normalize by 2^n to get standard Fourier coefficients
        w_coeffs_norm = w_coeffs / (2 ** n_players)

        # Extract pairwise (order-2) Walsh coefficients
        pairwise_walsh = {}
        for i in range(n_players):
            for j in range(i + 1, n_players):
                idx = (1 << i) | (1 << j)
                pairwise_walsh[(i, j)] = float(w_coeffs_norm[idx])

        # Compute subspace overlap for each pair
        pair_records = []
        for i in range(n_players):
            for j in range(i + 1, n_players):
                head_i = heads[i]
                head_j = heads[j]
                li, _ = head_i
                lj, _ = head_j

                ov = ov_overlap(head_i, head_j)
                qc_ij = q_composition(head_i, head_j)
                qc_ji = q_composition(head_j, head_i)
                kc_ij = k_composition(head_i, head_j)
                kc_ji = k_composition(head_j, head_i)
                qk_comp = max(qc_ij, qc_ji, kc_ij, kc_ji)
                layer_dist = abs(li - lj)

                pa = principal_angles(head_i, head_j)
                mean_pa = float(np.mean(pa))

                walsh_coeff = pairwise_walsh[(i, j)]

                rec = {
                    "circuit": circuit_name,
                    "task": circuit_info["task"],
                    "head_i": f"L{head_i[0]}H{head_i[1]}",
                    "head_j": f"L{head_j[0]}H{head_j[1]}",
                    "layer_i": li,
                    "layer_j": lj,
                    "layer_dist": layer_dist,
                    "walsh_coeff": walsh_coeff,
                    "abs_walsh": abs(walsh_coeff),
                    "ov_overlap": ov,
                    "qk_comp": qk_comp,
                    "q_comp_ij": qc_ij,
                    "q_comp_ji": qc_ji,
                    "k_comp_ij": kc_ij,
                    "k_comp_ji": kc_ji,
                    "mean_principal_angle": mean_pa,
                    "faithfulness": faith,
                }
                pair_records.append(rec)
                all_pairs_data.append(rec)

        # Per-circuit Spearman correlations
        n_pairs = len(pair_records)
        abs_walsh_arr = np.array([r["abs_walsh"] for r in pair_records])
        ov_arr = np.array([r["ov_overlap"] for r in pair_records])
        qk_arr = np.array([r["qk_comp"] for r in pair_records])
        ld_arr = np.array([r["layer_dist"] for r in pair_records])
        pa_arr = np.array([r["mean_principal_angle"] for r in pair_records])

        rho_ov, p_ov = stats.spearmanr(abs_walsh_arr, ov_arr)
        rho_qk, p_qk = stats.spearmanr(abs_walsh_arr, qk_arr)
        rho_ld, p_ld = stats.spearmanr(abs_walsh_arr, ld_arr)
        rho_pa, p_pa = stats.spearmanr(abs_walsh_arr, pa_arr)

        # Linear regression: |W_{ij}| ~ OV_overlap + QK_comp + layer_dist
        X = np.column_stack([ov_arr, qk_arr, ld_arr])
        X_with_intercept = np.column_stack([np.ones(n_pairs), X])
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, abs_walsh_arr, rcond=None)
            y_pred = X_with_intercept @ beta
            ss_res = np.sum((abs_walsh_arr - y_pred) ** 2)
            ss_tot = np.sum((abs_walsh_arr - abs_walsh_arr.mean()) ** 2)
            r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            r_squared = 0.0
            beta = np.zeros(4)

        circuit_result = {
            "circuit": circuit_name,
            "task": circuit_info["task"],
            "n_heads": n_players,
            "n_pairs": n_pairs,
            "faithfulness": faith,
            "P1_rho_ov_overlap": float(rho_ov),
            "P1_p_ov_overlap": float(p_ov),
            "rho_qk_comp": float(rho_qk),
            "p_qk_comp": float(p_qk),
            "rho_layer_dist": float(rho_ld),
            "p_layer_dist": float(p_ld),
            "rho_principal_angle": float(rho_pa),
            "p_principal_angle": float(p_pa),
            "P4_r_squared": r_squared,
            "P4_beta_intercept": float(beta[0]),
            "P4_beta_ov": float(beta[1]),
            "P4_beta_qk": float(beta[2]),
            "P4_beta_ld": float(beta[3]),
        }
        all_results[circuit_name] = circuit_result

        print(f"[{ts()}] {circuit_name}: faith={faith:.3f}, "
              f"rho_OV={rho_ov:.3f} (p={p_ov:.4f}), "
              f"rho_QK={rho_qk:.3f} (p={p_qk:.4f}), "
              f"R^2={r_squared:.3f}")

    # Step 4: Pooled analysis across all circuits
    print(f"\n[{ts()}] === Pooled analysis across {len(all_pairs_data)} pairs ===")

    abs_walsh_all = np.array([r["abs_walsh"] for r in all_pairs_data])
    ov_all = np.array([r["ov_overlap"] for r in all_pairs_data])
    qk_all = np.array([r["qk_comp"] for r in all_pairs_data])
    ld_all = np.array([r["layer_dist"] for r in all_pairs_data])
    pa_all = np.array([r["mean_principal_angle"] for r in all_pairs_data])
    faith_all = np.array([r["faithfulness"] for r in all_pairs_data])

    rho_ov_all, p_ov_all = stats.spearmanr(abs_walsh_all, ov_all)
    rho_qk_all, p_qk_all = stats.spearmanr(abs_walsh_all, qk_all)
    rho_ld_all, p_ld_all = stats.spearmanr(abs_walsh_all, ld_all)

    print(f"  Pooled rho(|Walsh|, OV_overlap) = {rho_ov_all:.4f} (p={p_ov_all:.2e})")
    print(f"  Pooled rho(|Walsh|, QK_comp)    = {rho_qk_all:.4f} (p={p_qk_all:.2e})")
    print(f"  Pooled rho(|Walsh|, layer_dist) = {rho_ld_all:.4f} (p={p_ld_all:.2e})")

    # Pooled regression
    X_all = np.column_stack([ov_all, qk_all, ld_all])
    X_all_int = np.column_stack([np.ones(len(all_pairs_data)), X_all])
    beta_all, _, _, _ = np.linalg.lstsq(X_all_int, abs_walsh_all, rcond=None)
    y_pred_all = X_all_int @ beta_all
    ss_res_all = np.sum((abs_walsh_all - y_pred_all) ** 2)
    ss_tot_all = np.sum((abs_walsh_all - abs_walsh_all.mean()) ** 2)
    r2_all = float(1 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0.0
    print(f"  Pooled R^2 = {r2_all:.4f}")

    # P3: Split by layer distance
    close_mask = ld_all <= 2
    far_mask = ld_all > 3
    if close_mask.sum() > 5 and far_mask.sum() > 5:
        rho_ov_close, _ = stats.spearmanr(abs_walsh_all[close_mask], ov_all[close_mask])
        rho_qk_close, _ = stats.spearmanr(abs_walsh_all[close_mask], qk_all[close_mask])
        rho_ov_far, _ = stats.spearmanr(abs_walsh_all[far_mask], ov_all[far_mask])
        rho_qk_far, _ = stats.spearmanr(abs_walsh_all[far_mask], qk_all[far_mask])
        print(f"\n  P3 — Close layers (dist<=2, n={close_mask.sum()}):")
        print(f"    rho_OV={rho_ov_close:.4f}, rho_QK={rho_qk_close:.4f}")
        print(f"  P3 — Far layers (dist>3, n={far_mask.sum()}):")
        print(f"    rho_OV={rho_ov_far:.4f}, rho_QK={rho_qk_far:.4f}")
    else:
        rho_ov_close = rho_qk_close = rho_ov_far = rho_qk_far = None

    # P2: Partial correlation of QK_comp controlling for OV_overlap
    # partial_rho(X,Y|Z) via residuals
    def partial_spearman(x, y, z):
        """Partial Spearman correlation of x and y controlling for z."""
        rho_xz, _ = stats.spearmanr(x, z)
        rho_yz, _ = stats.spearmanr(y, z)
        rho_xy, _ = stats.spearmanr(x, y)
        numer = rho_xy - rho_xz * rho_yz
        denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
        if denom < 1e-10:
            return 0.0, 1.0
        partial_rho = numer / denom
        n = len(x)
        t_stat = partial_rho * np.sqrt((n - 3) / (1 - partial_rho**2 + 1e-10))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))
        return float(partial_rho), float(p_val)

    partial_qk, partial_qk_p = partial_spearman(abs_walsh_all, qk_all, ov_all)
    print(f"\n  P2 — Partial rho(|Walsh|, QK_comp | OV_overlap) = {partial_qk:.4f} (p={partial_qk_p:.2e})")

    # High-faith vs low-faith split
    faith_threshold = 0.5
    circuit_faiths = {name: r["faithfulness"] for name, r in all_results.items()}
    high_faith_circuits = [n for n, f in circuit_faiths.items() if f > faith_threshold]
    low_faith_circuits = [n for n, f in circuit_faiths.items() if f <= faith_threshold]

    high_mask = np.array([r["circuit"] in high_faith_circuits for r in all_pairs_data])
    low_mask = np.array([r["circuit"] in low_faith_circuits for r in all_pairs_data])

    print(f"\n  High-faith circuits (faith>{faith_threshold}): {high_faith_circuits}")
    print(f"  Low-faith circuits (faith<={faith_threshold}): {low_faith_circuits}")

    if high_mask.sum() > 10:
        rho_ov_hf, p_ov_hf = stats.spearmanr(abs_walsh_all[high_mask], ov_all[high_mask])
        X_hf = np.column_stack([ov_all[high_mask], qk_all[high_mask], ld_all[high_mask]])
        X_hf_int = np.column_stack([np.ones(high_mask.sum()), X_hf])
        beta_hf, _, _, _ = np.linalg.lstsq(X_hf_int, abs_walsh_all[high_mask], rcond=None)
        y_pred_hf = X_hf_int @ beta_hf
        ss_res_hf = np.sum((abs_walsh_all[high_mask] - y_pred_hf) ** 2)
        ss_tot_hf = np.sum((abs_walsh_all[high_mask] - abs_walsh_all[high_mask].mean()) ** 2)
        r2_hf = float(1 - ss_res_hf / ss_tot_hf) if ss_tot_hf > 0 else 0.0
        print(f"  High-faith: rho_OV={rho_ov_hf:.4f}, R^2={r2_hf:.4f}")
    else:
        rho_ov_hf = r2_hf = None

    # Step 5: Save everything
    summary = {
        "n_circuits_analyzed": len(all_results),
        "n_total_pairs": len(all_pairs_data),
        "pooled_rho_ov": float(rho_ov_all),
        "pooled_p_ov": float(p_ov_all),
        "pooled_rho_qk": float(rho_qk_all),
        "pooled_p_qk": float(p_qk_all),
        "pooled_rho_layer_dist": float(rho_ld_all),
        "pooled_r_squared": r2_all,
        "P2_partial_rho_qk": partial_qk,
        "P2_partial_p_qk": partial_qk_p,
        "P3_close": {"rho_ov": rho_ov_close, "rho_qk": rho_qk_close} if rho_ov_close is not None else None,
        "P3_far": {"rho_ov": rho_ov_far, "rho_qk": rho_qk_far} if rho_ov_far is not None else None,
        "high_faith_rho_ov": rho_ov_hf,
        "high_faith_r_squared": r2_hf,
        "per_circuit": all_results,
        "predictions": {
            "P1": f"rho_OV > 0.3? Pooled: {rho_ov_all:.3f} ({'PASS' if rho_ov_all > 0.3 else 'FAIL'})",
            "P2": f"Partial rho(QK|OV) > 0, p<0.05? rho={partial_qk:.3f}, p={partial_qk_p:.2e} ({'PASS' if partial_qk > 0 and partial_qk_p < 0.05 else 'FAIL'})",
            "P4": f"R^2 > 0.3? Pooled: {r2_all:.3f} ({'PASS' if r2_all > 0.3 else 'FAIL'})",
        },
    }

    with open("/results/subspace_epistasis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("/results/subspace_epistasis_pairs.json", "w") as f:
        json.dump(all_pairs_data, f, indent=2)

    with open("/results/subspace_epistasis_per_circuit.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save as CSV for easy inspection
    import csv
    with open("/results/subspace_epistasis_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_pairs_data[0].keys())
        writer.writeheader()
        writer.writerows(all_pairs_data)

    output_volume.commit()

    # Print final verdict
    print(f"\n{'='*60}")
    print(f"SUBSPACE EPISTASIS ANALYSIS — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Circuits analyzed: {len(all_results)}")
    print(f"Total head pairs: {len(all_pairs_data)}")
    print(f"")
    for pred_name, pred_result in summary["predictions"].items():
        print(f"  {pred_name}: {pred_result}")
    print(f"")
    print(f"Per-circuit results:")
    for name, res in sorted(all_results.items()):
        print(f"  {name}: faith={res['faithfulness']:.3f}, "
              f"rho_OV={res['P1_rho_ov_overlap']:.3f}, "
              f"R^2={res['P4_r_squared']:.3f}")
    print(f"{'='*60}")
    print(f"[{ts()}] Results saved to /results/")


@app.local_entrypoint()
def main():
    analyze.remote()
