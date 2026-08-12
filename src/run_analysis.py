"""End-to-end analysis pipeline for v3 pre-registration.

Follows the execution order in prereg Section 9 exactly.
Skips H3 (requires additional forward passes) and H6 (requires CoAx
computability check). H4 is skipped if Exp-06 data is not available.

Inputs: {circuit}_coalition_values.npz files from the v2 sweep.
Outputs: results saved to analysis_results.json.

Can run in two modes:
  --synthetic : generate synthetic data and run the pipeline (dry run)
  --real      : run on real sweep data (breaks the seal)

Usage:
    uv run python experiments_batch2/genetics/run_analysis.py --synthetic
    uv run python experiments_batch2/genetics/run_analysis.py --real
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from mobius_wht import (
    complement_values,
    energy_by_order,
    energy_by_order_normalized,
    mobius_transform,
    per_head_energy,
    restricted_energy_ge3,
    wht,
    wht_restricted_energy_ge3,
)

N_PLAYERS = 15
N_PROMPTS = 40
N_COALITIONS = 2 ** N_PLAYERS

COPIER_INDICES = [4, 5, 6, 7, 8, 9, 10, 11]
BACKBONE_INDICES = [0, 1, 2]
DETECTOR_INDEX = 3
READOUT_INDICES = [12, 13, 14]

DATA_DIR = Path(__file__).parent


def load_real_data(circuit_name):
    """Load per-prompt raw logits from a sweep .npz file."""
    path = DATA_DIR / f"{circuit_name}_coalition_values.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "target_logits": data["target_logits"],
        "foil_logits": data["foil_logits"],
        "coalition_indices": data["coalition_indices"],
        "intact_target_logits": data["intact_target_logits"],
        "intact_foil_logits": data["intact_foil_logits"],
        "circuit_name": str(data["circuit_name"]),
        "n_players": int(data["n_players"]),
        "n_prompts": int(data["n_prompts"]),
    }


def generate_synthetic_data():
    """Generate synthetic per-prompt data with known Möbius structure.

    Backbone heads (0-2): order-1 main effects.
    Copier heads (4-8): one order-4 interaction on {4,5,6,7}.
    All effects have per-prompt noise so v_mult and v_add differ.
    """
    from mobius_wht import inverse_mobius_transform

    rng = np.random.default_rng(2026)

    target_logits = np.zeros((N_COALITIONS, N_PROMPTS))
    foil_logits = np.zeros((N_COALITIONS, N_PROMPTS))

    backbone_strengths = [2.0, 1.5, 1.8]
    copier_interaction = 1.2
    copier_subset_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

    for prompt_idx in range(N_PROMPTS):
        noise_scale = 0.1 + 0.05 * rng.standard_normal()
        prompt_baseline = 0.5 + 0.3 * rng.standard_normal()

        coeffs = np.zeros(N_COALITIONS)
        coeffs[0] = prompt_baseline
        for i, strength in enumerate(backbone_strengths):
            coeffs[1 << i] = strength + noise_scale * rng.standard_normal()
        coeffs[copier_subset_mask] = copier_interaction + noise_scale * rng.standard_normal()

        values = inverse_mobius_transform(coeffs)
        target_logits[:, prompt_idx] = values + 0.05 * rng.standard_normal(N_COALITIONS)
        foil_logits[:, prompt_idx] = -0.5 + 0.05 * rng.standard_normal(N_COALITIONS)

    intact_target = target_logits[N_COALITIONS - 1, :]
    intact_foil = foil_logits[N_COALITIONS - 1, :]

    return {
        "target_logits": target_logits,
        "foil_logits": foil_logits,
        "coalition_indices": np.arange(N_COALITIONS),
        "intact_target_logits": intact_target,
        "intact_foil_logits": intact_foil,
        "circuit_name": "synthetic_rti",
        "n_players": N_PLAYERS,
        "n_prompts": N_PROMPTS,
    }


def derive_value_functions(data, prompt_mask=None):
    """Derive v_mult and v_add from raw logits, optionally stratified.

    v_mult(S) = mean over prompts of [target_logit(S) - foil_logit(S)]
    v_add(S)  = mean over prompts of [P_target(S) - P_foil(S)]
                where P = softmax over the two logits
    """
    tgt = data["target_logits"]
    foil = data["foil_logits"]

    if prompt_mask is not None:
        tgt = tgt[:, prompt_mask]
        foil = foil[:, prompt_mask]

    v_mult = (tgt - foil).mean(axis=1)

    logit_pairs = np.stack([tgt, foil], axis=-1)
    probs = softmax(logit_pairs, axis=-1)
    p_target = probs[:, :, 0]
    p_foil = probs[:, :, 1]
    v_add = (p_target - p_foil).mean(axis=1)

    return v_mult, v_add


def compute_strata(data):
    """Split prompts into 2 strata by median intact-model logit-diff."""
    intact_diff = data["intact_target_logits"] - data["intact_foil_logits"]
    median_val = np.median(intact_diff)
    low_mask = intact_diff <= median_val
    high_mask = intact_diff > median_val

    if low_mask.sum() == 0 or high_mask.sum() == 0:
        high_mask = ~low_mask

    return {
        "pooled": np.ones(data["n_prompts"], dtype=bool),
        "stratum_low": low_mask,
        "stratum_high": high_mask,
    }


def run_h1(mobius_coeffs, circuit_label):
    """H1: Bimodal interaction order (unsupervised clustering)."""
    energies = per_head_energy(mobius_coeffs, N_PLAYERS)

    X = energies.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    labels = gmm.fit_predict(X)

    if len(set(labels)) < 2:
        return {
            "test": "H1",
            "circuit": circuit_label,
            "status": "FAIL (GMM collapsed to single cluster)",
            "silhouette": -1.0,
            "per_head_energies": energies.tolist(),
        }

    real_sil = silhouette_score(X, labels)

    positive_energies = energies[energies > 0]
    if len(positive_energies) < 3:
        return {
            "test": "H1",
            "circuit": circuit_label,
            "status": "SKIP (fewer than 3 heads with positive energy)",
            "silhouette": float(real_sil),
        }

    log_e = np.log(positive_energies)
    mu, sigma = log_e.mean(), log_e.std()
    if sigma < 1e-10:
        sigma = 1.0

    rng = np.random.default_rng(42)
    null_silhouettes = []
    for _ in range(1000):
        sample = rng.lognormal(mu, max(sigma, 0.01), size=N_PLAYERS)
        Xs = sample.reshape(-1, 1)
        gmm_null = GaussianMixture(n_components=2, random_state=0)
        labels_null = gmm_null.fit_predict(Xs)
        if len(set(labels_null)) > 1:
            null_silhouettes.append(silhouette_score(Xs, labels_null))

    if len(null_silhouettes) < 100:
        return {
            "test": "H1",
            "circuit": circuit_label,
            "status": "SKIP (too few valid null samples)",
            "silhouette": float(real_sil),
        }

    p95 = np.percentile(null_silhouettes, 95)
    passes = real_sil > p95

    backbone_copier_match = sum(
        1 for h in range(N_PLAYERS)
        if (h in BACKBONE_INDICES and labels[h] != labels[COPIER_INDICES[0]])
        or (h in COPIER_INDICES and labels[h] == labels[COPIER_INDICES[0]])
    ) / N_PLAYERS

    return {
        "test": "H1",
        "circuit": circuit_label,
        "silhouette": float(real_sil),
        "null_p95": float(p95),
        "exceeds_null": bool(passes),
        "cluster_labels": labels.tolist(),
        "backbone_copier_label_agreement": float(backbone_copier_match),
        "per_head_energies": energies.tolist(),
    }


def run_h2(values, circuit_label):
    """H2: Copier-tier order->=3 WHT/ANOVA energy fraction.

    Uses WHT (orthogonal) coefficients restricted to indices whose
    support is within the copier set. This is a valid variance
    partition — no cross-term contamination between subsets.
    """
    wht_coeffs = wht(values)
    restricted, total, fraction = restricted_energy_ge3(
        wht_coeffs, N_PLAYERS, COPIER_INDICES
    )

    rng = np.random.default_rng(42)
    all_indices = list(range(N_PLAYERS))
    null_fractions = []
    for _ in range(1000):
        subset = sorted(rng.choice(all_indices, 8, replace=False).tolist())
        _, _, f = restricted_energy_ge3(wht_coeffs, N_PLAYERS, subset)
        null_fractions.append(f)

    p95 = np.percentile(null_fractions, 95)
    passes = fraction > p95

    return {
        "test": "H2",
        "circuit": circuit_label,
        "copier_ge3_fraction": float(fraction),
        "null_p95": float(p95),
        "exceeds_null": bool(passes),
        "copier_ge3_energy": float(restricted),
        "total_energy": float(total),
        "basis": "WHT/ANOVA (orthogonal)",
    }


def run_h4_stub(mobius_coeffs, circuit_label):
    """H4: Consistency with Exp-06 (stub — needs Exp-06 data)."""
    exp06_path = DATA_DIR.parent / "pairwise_epistasis" / "results"
    if not exp06_path.exists():
        return {
            "test": "H4",
            "circuit": circuit_label,
            "status": "SKIP (Exp-06 data not found at expected path)",
            "expected_path": str(exp06_path),
        }

    return {
        "test": "H4",
        "circuit": circuit_label,
        "status": "TODO (Exp-06 data found, comparison not yet implemented)",
    }


def run_h5(v_mult_values, circuit_label):
    """H5: Necessity/sufficiency divergence."""
    w_values = complement_values(v_mult_values, N_PLAYERS)

    mobius_v = mobius_transform(v_mult_values)
    mobius_w = mobius_transform(w_values)

    energy_v = energy_by_order(mobius_v, N_PLAYERS)
    energy_w = energy_by_order(mobius_w, N_PLAYERS)

    copier_orders = list(range(1, N_PLAYERS + 1))
    copier_energy_v = [float(energy_v[k]) for k in copier_orders]
    copier_energy_w = [float(energy_w[k]) for k in copier_orders]

    if len(set(copier_energy_v)) < 2 or len(set(copier_energy_w)) < 2:
        return {
            "test": "H5",
            "circuit": circuit_label,
            "status": "SKIP (degenerate energy spectrum)",
        }

    corr_full, _ = spearmanr(copier_energy_v, copier_energy_w)

    copier_mask_orders = []
    backbone_mask_orders = []
    for k in range(1, N_PLAYERS + 1):
        copier_e_v = 0.0
        copier_e_w = 0.0
        backbone_e_v = 0.0
        backbone_e_w = 0.0
        for idx in range(N_COALITIONS):
            if bin(idx).count('1') != k:
                continue
            bits = [b for b in range(N_PLAYERS) if idx & (1 << b)]
            is_copier = all(b in COPIER_INDICES for b in bits)
            is_backbone = all(b in BACKBONE_INDICES for b in bits)
            if is_copier:
                copier_e_v += mobius_v[idx] ** 2
                copier_e_w += mobius_w[idx] ** 2
            if is_backbone:
                backbone_e_v += mobius_v[idx] ** 2
                backbone_e_w += mobius_w[idx] ** 2
        copier_mask_orders.append((copier_e_v, copier_e_w))
        backbone_mask_orders.append((backbone_e_v, backbone_e_w))

    copier_v_spec = [x[0] for x in copier_mask_orders]
    copier_w_spec = [x[1] for x in copier_mask_orders]
    backbone_v_spec = [x[0] for x in backbone_mask_orders]
    backbone_w_spec = [x[1] for x in backbone_mask_orders]

    nonzero_copier = [(v, w) for v, w in zip(copier_v_spec, copier_w_spec) if v > 0 or w > 0]
    nonzero_backbone = [(v, w) for v, w in zip(backbone_v_spec, backbone_w_spec) if v > 0 or w > 0]

    copier_corr = float('nan')
    backbone_corr = float('nan')
    if len(nonzero_copier) >= 3:
        copier_corr, _ = spearmanr([x[0] for x in nonzero_copier], [x[1] for x in nonzero_copier])
    if len(nonzero_backbone) >= 3:
        backbone_corr, _ = spearmanr([x[0] for x in nonzero_backbone], [x[1] for x in nonzero_backbone])

    passes = (
        not np.isnan(copier_corr) and copier_corr < 0.5
        and not np.isnan(backbone_corr) and backbone_corr > 0.8
    )

    return {
        "test": "H5",
        "circuit": circuit_label,
        "copier_corr": float(copier_corr) if not np.isnan(copier_corr) else None,
        "backbone_corr": float(backbone_corr) if not np.isnan(backbone_corr) else None,
        "threshold_copier": "< 0.5",
        "threshold_backbone": "> 0.8",
        "passes": bool(passes),
        "full_spectrum_corr": float(corr_full),
    }


def run_ioi_hard_stop(mobius_coeffs):
    """IOI positive-control check: order->=3 mass should be low."""
    energy = energy_by_order_normalized(mobius_coeffs, N_PLAYERS)
    ge3_fraction = sum(energy[k] for k in range(3, N_PLAYERS + 1))
    return {
        "test": "IOI_hard_stop",
        "order_ge3_fraction": float(ge3_fraction),
    }


def run_parseval_check(values):
    """WHT Parseval consistency check."""
    wht_coeffs = wht(values)
    lhs = np.sum(wht_coeffs ** 2)
    rhs = (2 ** N_PLAYERS) * np.sum(values ** 2)
    rel_err = abs(lhs - rhs) / max(abs(rhs), 1)
    return {
        "test": "WHT_Parseval",
        "rel_error": float(rel_err),
        "passes": rel_err < 1e-8,
    }


def analyze_circuit(data, circuit_label, is_primary=True):
    """Run the full analysis pipeline on one circuit's data."""
    results = {"circuit": circuit_label, "strata": {}}

    strata = compute_strata(data)

    for stratum_name, prompt_mask in strata.items():
        v_mult, v_add = derive_value_functions(data, prompt_mask)
        mobius_mult = mobius_transform(v_mult)
        mobius_add = mobius_transform(v_add)

        stratum_results = {
            "n_prompts": int(prompt_mask.sum()),
        }

        stratum_results["parseval_vmult"] = run_parseval_check(v_mult)
        stratum_results["parseval_vadd"] = run_parseval_check(v_add)

        stratum_results["energy_spectrum_vmult"] = energy_by_order_normalized(mobius_mult, N_PLAYERS).tolist()
        stratum_results["energy_spectrum_vadd"] = energy_by_order_normalized(mobius_add, N_PLAYERS).tolist()

        if is_primary:
            stratum_results["h1_vmult"] = run_h1(mobius_mult, f"{circuit_label}/{stratum_name}/vmult")
            stratum_results["h1_vadd"] = run_h1(mobius_add, f"{circuit_label}/{stratum_name}/vadd")

            stratum_results["h2_vmult"] = run_h2(v_mult, f"{circuit_label}/{stratum_name}/vmult")
            stratum_results["h2_vadd"] = run_h2(v_add, f"{circuit_label}/{stratum_name}/vadd")

            h2_both_pass = (
                stratum_results["h2_vmult"]["exceeds_null"]
                and stratum_results["h2_vadd"]["exceeds_null"]
            )
            stratum_results["h2_survives_scale"] = h2_both_pass

            if stratum_name == "pooled":
                stratum_results["h4"] = run_h4_stub(mobius_mult, circuit_label)
                stratum_results["h5"] = run_h5(v_mult, circuit_label)

        results["strata"][stratum_name] = stratum_results

    return results


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Dry run on synthetic data")
    mode.add_argument("--real", action="store_true", help="Run on real sweep data")
    args = parser.parse_args()

    all_results = {"mode": "synthetic" if args.synthetic else "real", "circuits": {}}

    if args.synthetic:
        print("=" * 60)
        print("DRY RUN: synthetic data (no real data touched)")
        print("=" * 60)

        data = generate_synthetic_data()
        results = analyze_circuit(data, "synthetic_rti", is_primary=True)
        all_results["circuits"]["synthetic_rti"] = results

    else:
        print("=" * 60)
        print("REAL DATA ANALYSIS — seal broken")
        print("=" * 60)

        print("\n--- Step 5: Stratification ---")
        rti_data = load_real_data("rti")
        ioi_data = load_real_data("ioi")
        random15_data = load_real_data("random15")

        print("\n--- Step 7: IOI positive-control hard stop ---")
        ioi_vmult, _ = derive_value_functions(ioi_data)
        ioi_mobius = mobius_transform(ioi_vmult)
        ioi_stop = run_ioi_hard_stop(ioi_mobius)
        all_results["ioi_hard_stop"] = ioi_stop
        print(f"  IOI order->=3 fraction: {ioi_stop['order_ge3_fraction']:.6f}")

        print("\n--- Steps 8-10: RTI analysis ---")
        rti_results = analyze_circuit(rti_data, "rti", is_primary=True)
        all_results["circuits"]["rti"] = rti_results

        print("\n--- Random-15 control ---")
        random15_results = analyze_circuit(random15_data, "random15", is_primary=False)
        all_results["circuits"]["random15"] = random15_results

        print("\n--- IOI control ---")
        ioi_results = analyze_circuit(ioi_data, "ioi", is_primary=False)
        all_results["circuits"]["ioi"] = ioi_results

        pooled_rti_h1 = rti_results["strata"]["pooled"].get("h1_vmult", {})
        pooled_r15_energies = per_head_energy(
            mobius_transform(derive_value_functions(random15_data)[0]),
            N_PLAYERS,
        )
        r15_X = pooled_r15_energies.reshape(-1, 1)
        r15_gmm = GaussianMixture(n_components=2, random_state=0)
        r15_labels = r15_gmm.fit_predict(r15_X)
        if len(set(r15_labels)) > 1:
            r15_sil = silhouette_score(r15_X, r15_labels)
        else:
            r15_sil = -1.0

        all_results["h1_random15_comparison"] = {
            "rti_silhouette": pooled_rti_h1.get("silhouette"),
            "random15_silhouette": float(r15_sil),
            "rti_exceeds_random15": (
                pooled_rti_h1.get("silhouette", -1) > r15_sil
            ),
        }

        rti_h2_vmult = rti_results["strata"]["pooled"].get("h2_vmult", {})
        r15_vmult = derive_value_functions(random15_data)[0]
        r15_mobius = mobius_transform(r15_vmult)
        r15_ge3 = energy_by_order_normalized(r15_mobius, N_PLAYERS)
        r15_ge3_frac = sum(r15_ge3[k] for k in range(3, N_PLAYERS + 1))

        all_results["h2_random15_comparison"] = {
            "rti_copier_ge3": rti_h2_vmult.get("copier_ge3_fraction"),
            "random15_total_ge3": float(r15_ge3_frac),
            "rti_exceeds_random15": (
                (rti_h2_vmult.get("copier_ge3_fraction") or 0) > r15_ge3_frac
            ),
        }

    print("\n--- Saving results ---")
    suffix = "synthetic" if args.synthetic else "real"
    out_path = DATA_DIR / f"analysis_results_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    print("\n--- Summary ---")
    for circuit_name, circuit_results in all_results.get("circuits", {}).items():
        print(f"\n{circuit_name}:")
        pooled = circuit_results.get("strata", {}).get("pooled", {})
        for key in ["h1_vmult", "h1_vadd", "h2_vmult", "h2_vadd", "h4", "h5"]:
            if key in pooled:
                result = pooled[key]
                if "exceeds_null" in result:
                    status = "PASS" if result["exceeds_null"] else "FAIL"
                elif "passes" in result:
                    status = "PASS" if result["passes"] else "FAIL"
                elif "status" in result:
                    status = result["status"]
                else:
                    status = "?"
                print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
