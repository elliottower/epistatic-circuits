"""E11 Gate 0: is the vector-valued interaction more than its mean?

SCOPING CALCULATION -- NOT PREREGISTERED, NO HYPOTHESIS ATTACHED.

    delta_AB^(l) = r_clean - r_{A-abl} - r_{B-abl} + r_{AB-abl}

the order-2 Walsh/Moebius coefficient kept as a residual-stream vector instead of
collapsed to a logit difference.

Jansma (ICML 2026 wksp) publishes the vector-valued order-2 Moebius coefficient over
input *spans* and aggregates it with a MEAN. MSRS (arXiv 2508.10599) publishes
SVD-of-activation-differences as a subspace-extraction recipe. So the only question
that decides whether a subspace study is worth preregistering is whether the
collection of delta vectors carries structure the mean throws away.

Four numbers per pair, then stop:
  (1) MCR   ||mean(delta)|| / mean(||delta||)
            ~1    => one direction; the mean IS the object; the scalar Walsh
                     coefficient already had it; STOP
            small => delta points different ways per prompt; a subspace is the
                     natural object; PROCEED
  (2) D_PR  participation ratio, (sum lam)^2 / sum(lam^2)
  (3) D_PR under a matched-norm random null at IDENTICAL N and d. PR is heavily
      biased when N << d (arXiv 2509.26560); identical N makes the bias cancel in
      the comparison rather than needing to be estimated away. Structured data sits
      BELOW the random null; a sign-shuffle null would be a no-op here, since row
      sign flips leave delta^T delta and hence every singular value unchanged.
  (4) sigma_2 / sigma_1

Correctness check that costs nothing: for layers strictly before
min(layer_A, layer_B) there is no causal path from either head, so delta must be
exactly zero. This is a stronger test than reprojecting to logits because it does
not depend on LayerNorm behaving.

Run:
    modal run --detach experiments/E11_interaction_vector_scoping/modal_run_e11_gate0.py
"""

import os

import modal

app = modal.App("e11-interaction-vector-gate0")
output_volume = modal.Volume.from_name(
    "e11-interaction-vector-results", create_if_missing=True
)

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
    .add_local_file(
        "results/phase2/phase2_all_walsh_coefficients.json",
        remote_path="/data/phase2_all_walsh_coefficients.json",
    )
    .add_local_file(
        "results/phase2/phase2_head_selection.json",
        remote_path="/data/phase2_head_selection.json",
    )
    .add_local_file(
        "data/ioi_prompts_200.json", remote_path="/data/ioi_prompts_200.json"
    )
)

N_PAIRS = 190     # all of them
N_SHUFFLE = 200
ABLATION = "mean"  # must match the ablation behind the scalar Walsh coefficients


def gate0(data_dir: str, out_dir: str, n_pairs: int, n_shuffle: int,
          max_prompts: int | None = None, device: str | None = None,
          commit=None) -> dict:
    """The whole calculation. Runs inside the Modal container."""
    import json
    import time

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    def ts():
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print(f"[{ts()}] E11 Gate 0 -- scoping only, no inference licensed")

    with open(f"{data_dir}/phase2_all_walsh_coefficients.json") as f:
        walsh_data = json.load(f)
    with open(f"{data_dir}/phase2_head_selection.json") as f:
        head_sel = json.load(f)
    with open(f"{data_dir}/ioi_prompts_200.json") as f:
        prompt_data = json.load(f)

    heads = [tuple(h) for h in head_sel["selected_heads"]]
    label_to_head = {f"L{l}H{h}": (l, h) for (l, h) in heads}

    order2 = []
    for key, entry in walsh_data.items():
        if entry.get("order") != 2:
            continue
        parts = key.split("-")
        if len(parts) != 2:
            continue
        a_lbl, b_lbl = parts
        if a_lbl in label_to_head and b_lbl in label_to_head:
            order2.append((a_lbl, b_lbl, float(entry["coeff"])))
    if not order2:
        raise RuntimeError("no order-2 coefficients matched the selected heads")
    order2.sort(key=lambda t: t[2])
    half = max(1, n_pairs // 2)
    selected = order2[:half] + order2[-half:]
    print(f"[{ts()}] {len(order2)} order-2 pairs available; scoping {len(selected)}")
    for a, b, w in selected:
        print(f"    {a}-{b}  w = {w:+.4f}")

    prompts = prompt_data["prompts"] if isinstance(prompt_data, dict) else prompt_data
    prompt_texts = [p["prompt"] if isinstance(p, dict) else p for p in prompts]
    if max_prompts:
        prompt_texts = prompt_texts[:max_prompts]
    n_prompts = len(prompt_texts)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] loading gpt2-small on {device}, {n_prompts} prompts")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model

    tokens = model.to_tokens(prompt_texts)
    # True final-token index per prompt. Deriving this from pad_token_id is wrong for
    # GPT-2: its pad token and its BOS token are both <|endoftext|>, so a != pad mask
    # counts the prepended BOS as padding and lands one position early. Tokenize each
    # prompt on its own and use its real length.
    lengths = [model.to_tokens(t).shape[1] for t in prompt_texts]
    final_idx = torch.tensor([l - 1 for l in lengths], device=device)
    assert int(final_idx.max()) < tokens.shape[1], "final index past the padded width"

    resid_hooks = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n.endswith("hook_z") or n in resid_hooks,
        )
    mean_z = {
        l: clean_cache[f"blocks.{l}.attn.hook_z"].mean(dim=0, keepdim=True)
        for l in range(n_layers)
    }

    def resid_stack(cache):
        out = torch.zeros(n_layers, n_prompts, d_model, device=device)
        rows = torch.arange(n_prompts, device=device)
        for l in range(n_layers):
            out[l] = cache[f"blocks.{l}.hook_resid_post"][rows, final_idx]
        return out

    def run_ablated(ablate):
        by_layer: dict[int, list[int]] = {}
        for (l, h) in ablate:
            by_layer.setdefault(l, []).append(h)

        def make_hook(layer, head_idxs):
            def hook(z, hook):
                for h in head_idxs:
                    z[:, :, h, :] = (
                        mean_z[layer][:, :, h, :] if ABLATION == "mean" else 0.0
                    )
                return z
            return hook

        hooks = [
            (f"blocks.{l}.attn.hook_z", make_hook(l, hs)) for l, hs in by_layer.items()
        ]
        # run_with_cache does NOT take fwd_hooks -- it would be swallowed by
        # **model_kwargs and the ablation would silently never be applied, leaving a
        # script that completes and prints a verdict built on nothing. Hooks have to
        # be installed around the call.
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in resid_hooks
            )
        return resid_stack(cache)

    R_clean = resid_stack(clean_cache)
    del clean_cache

    needed = sorted(
        {label_to_head[a] for a, b, _ in selected}
        | {label_to_head[b] for a, b, _ in selected}
    )
    single = {hd: run_ablated([hd]) for hd in tqdm(needed, desc="single ablations")}

    # GATE: the ablation must actually change the residual stream.
    #
    # This is the failure this script is most exposed to. If the hooks do not fire, every
    # ablated run equals the clean run, so delta = R - R - R + R = 0 exactly. Then the
    # causal-path check reads 0 and PASSES, MCR reads 0 and lands below the 0.3
    # threshold, and the script prints PROCEED on a study whose entire input was zeros.
    # A wiring bug must not be able to look like the interesting answer.
    ablation_effects = {}
    for hd, R_abl in single.items():
        eff = float((R_abl - R_clean)[hd[0]:].abs().max())
        ablation_effects[f"L{hd[0]}H{hd[1]}"] = eff
        if eff < 1e-6:
            raise RuntimeError(
                f"ablating L{hd[0]}H{hd[1]} changed the residual stream by {eff:.3e} "
                "at or after its own layer. The hooks are not being applied, so every "
                "delta would be identically zero and every statistic below meaningless."
            )
    print(
        f"[{ts()}] ablation gate passed: smallest single-head effect "
        f"{min(ablation_effects.values()):.3e}"
    )

    rng = np.random.default_rng(0)

    def top_directions(delta, k=5):
        """Top-k right singular vectors -- the directions the interaction occupies."""
        _, _, vt = np.linalg.svd(delta, full_matrices=False)
        return vt[:k]

    def participation_ratio(delta):
        s = np.linalg.svd(delta, compute_uv=False)
        lam = s ** 2
        if lam.sum() <= 0:
            return float("nan"), s
        return float(lam.sum() ** 2 / (lam ** 2).sum()), s

    results = {
        "experiment": "E11 Gate 0: interaction vector scoping",
        "status": "scoping calculation, not preregistered",
        "ablation": ABLATION,
        "n_prompts": n_prompts,
        "n_shuffle": n_shuffle,
        "pairs": {},
    }
    clean_np = R_clean.cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    ckpt = f"{out_dir}/e11_gate0_results.json"

    for a_lbl, b_lbl, w in tqdm(selected, desc="pairs"):
        A, B = label_to_head[a_lbl], label_to_head[b_lbl]
        R_both = run_ablated([A, B])
        delta_all = (R_clean - single[A] - single[B] + R_both).cpu().numpy()

        first_causal = min(A[0], B[0])
        pre_max = (
            float(np.abs(delta_all[:first_causal]).max()) if first_causal > 0 else 0.0
        )

        raw = np.linalg.norm(delta_all, axis=(1, 2))
        cln = np.linalg.norm(clean_np, axis=(1, 2))
        norm_profile = raw / np.maximum(cln, 1e-12)
        peak_layer = int(np.argmax(norm_profile))
        delta = delta_all[peak_layer]

        # Both statistics at EVERY layer, not only the peak. Computing them only at an
        # argmax-selected layer is a forking path: each pair would be measured wherever its
        # own data put the maximum. The peak stays as one summary among many.
        mcr_profile, pr_profile = [], []
        for lay in range(n_layers):
            dl = delta_all[lay]
            nl = np.linalg.norm(dl, axis=1)
            mcr_profile.append(
                float(np.linalg.norm(dl.mean(axis=0)) / np.maximum(nl.mean(), 1e-12))
            )
            pr_l, _ = participation_ratio(dl)
            pr_profile.append(pr_l)

        mean_vec = delta.mean(axis=0)
        per_norm = np.linalg.norm(delta, axis=1)
        mcr = float(np.linalg.norm(mean_vec) / np.maximum(per_norm.mean(), 1e-12))

        d_pr, svals = participation_ratio(delta)

        # Two nulls, because one statistic each needs a different thing broken.
        #
        # MCR null -- sign shuffle. Flipping row signs changes the mean while leaving every
        # row norm alone, which is exactly the structure MCR is about.
        #
        # PR null -- matched-norm random directions. Sign shuffling is USELESS here and
        # silently so: delta^T delta = sum_i d_i d_i^T and (-d_i)(-d_i)^T = d_i d_i^T, so row
        # sign flips leave every singular value unchanged and the "null" equals the observed
        # value by construction. To say anything about effective dimensionality the null has
        # to destroy the covariance structure while holding the per-row norms and the (N, d)
        # aspect ratio fixed -- PR is heavily biased when N << d, and identical N is what
        # makes the bias cancel in the comparison instead of needing to be estimated away.
        row_norms = np.linalg.norm(delta, axis=1, keepdims=True)
        null_pr, null_mcr = [], []
        for _ in range(n_shuffle):
            signs = rng.choice([-1.0, 1.0], size=(delta.shape[0], 1))
            d_sign = delta * signs
            null_mcr.append(
                float(
                    np.linalg.norm(d_sign.mean(axis=0))
                    / np.maximum(np.linalg.norm(d_sign, axis=1).mean(), 1e-12)
                )
            )
            g = rng.standard_normal(delta.shape)
            d_rand = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12) * row_norms
            pr_n, _ = participation_ratio(d_rand)
            null_pr.append(pr_n)
        null_pr, null_mcr = np.array(null_pr), np.array(null_mcr)

        energy = np.cumsum(svals ** 2) / np.sum(svals ** 2)
        results["pairs"][f"{a_lbl}-{b_lbl}"] = {
            "walsh_w": w,
            "sign": "sub_additive" if w < 0 else "super_additive",
            "peak_layer": peak_layer,
            "layer_A": A[0],
            "layer_B": B[0],
            "pre_causal_max_abs": pre_max,
            "norm_layer_profile": norm_profile.tolist(),
            "mcr_layer_profile": mcr_profile,
            "pr_layer_profile": pr_profile,
            "mean_cancellation_ratio": mcr,
            "mcr_null_mean": float(null_mcr.mean()),
            "participation_ratio": d_pr,
            "pr_null_kind": "matched-norm random directions, same N and d",
            "pr_null_mean": float(null_pr.mean()),
            "pr_null_p975": float(np.percentile(null_pr, 97.5)),
            "pr_below_null": bool(d_pr < np.percentile(null_pr, 2.5)),
            "sigma2_over_sigma1": (
                float(svals[1] / svals[0]) if len(svals) > 1 and svals[0] > 0
                else float("nan")
            ),
            "k90": int(np.searchsorted(energy, 0.90) + 1),
            "top_singular_values": svals[:10].tolist(),
            # saved so the interaction direction can be compared against each head's OV
            # write direction, and against other pairs' directions, without a re-run
            "top_directions": top_directions(delta).tolist(),
        }
        # checkpoint inside the loop: a killed run keeps every pair already done
        with open(ckpt, "w") as f:
            json.dump(results, f, indent=2)
        if commit:
            commit()
        del R_both

    mcrs = np.array([v["mean_cancellation_ratio"] for v in results["pairs"].values()])
    pre = np.array([v["pre_causal_max_abs"] for v in results["pairs"].values()])

    hdr = f"\n{'pair':<18}{'w':>9}{'peakL':>7}{'MCR':>8}{'D_PR':>8}{'PR0':>8}{'s2/s1':>8}{'k90':>6}"
    print(hdr)
    for k, v in results["pairs"].items():
        print(
            f"{k:<18}{v['walsh_w']:>+9.4f}{v['peak_layer']:>7d}"
            f"{v['mean_cancellation_ratio']:>8.3f}{v['participation_ratio']:>8.1f}"
            f"{v['pr_null_mean']:>8.1f}{v['sigma2_over_sigma1']:>8.3f}{v['k90']:>6d}"
        )

    print(f"\nCausal-path check: max |delta| before either head = {pre.max():.3e}")
    causal_ok = bool(pre.max() <= 1e-4)
    if not causal_ok:
        print("  *** FAILED -- ablations are not doing what the design assumes.")
        print("  *** Every number above is void.")

    med = float(np.median(mcrs))
    if not causal_ok:
        verdict = "VOID. The causal-path check failed; nothing here is interpretable."
    elif med > 0.7:
        verdict = (
            "STOP. The interaction is one direction. The mean is the object, the mean "
            "is what Jansma published, and the vector version reduces to the scalar "
            "Walsh coefficient already reported."
        )
    elif med < 0.3:
        verdict = (
            "PROCEED. The mean substantially cancels: delta points different ways "
            "across prompts. The next question is whether those directions cluster by "
            "an identifiable prompt property -- a mechanism finding, not a geometry one."
        )
    else:
        verdict = (
            "AMBIGUOUS. Widen to all 190 pairs and more prompts before deciding. "
            "Do not preregister on this."
        )
    print(f"\nMedian MCR: {med:.3f}\nVerdict: {verdict}")
    results.update(
        ablation_effects=ablation_effects,
        median_mcr=med,
        causal_check_max=float(pre.max()),
        causal_check_passed=causal_ok,
        verdict=verdict,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = list(results["pairs"].keys())
    colors = [
        "tab:red" if results["pairs"][k]["walsh_w"] < 0 else "tab:blue" for k in labels
    ]
    axes[0].bar(range(len(labels)), mcrs, color=colors)
    axes[0].axhline(0.7, ls="--", c="k", lw=0.8)
    axes[0].axhline(0.3, ls="--", c="k", lw=0.8)
    axes[0].set_ylabel("mean cancellation ratio")
    axes[0].set_title("Does the mean survive?")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=6)
    for k in labels:
        axes[1].plot(results["pairs"][k]["norm_layer_profile"], alpha=0.6, lw=1)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("||delta|| / ||resid||")
    axes[1].set_title("Normalised layer profile")
    obs = [results["pairs"][k]["participation_ratio"] for k in labels]
    nul = [results["pairs"][k]["pr_null_mean"] for k in labels]
    axes[2].scatter(nul, obs, c=colors)
    lim = [0, max(max(obs), max(nul)) * 1.1]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set_xlabel("PR, matched-norm random null (same N, d)")
    axes[2].set_ylabel("PR, observed")
    axes[2].set_title("Effective dimensionality vs null\n(below the line = more structured than random)")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/e11_gate0.png", dpi=150)

    with open(ckpt, "w") as f:
        json.dump(results, f, indent=2)
    if commit:
        commit()
    print(f"[{ts()}] wrote {ckpt}")
    return results


@app.function(image=image, gpu="T4", timeout=86400, volumes={"/results": output_volume})
def run_e11_gate0():
    return gate0("/data", "/results", N_PAIRS, N_SHUFFLE, commit=output_volume.commit)


@app.local_entrypoint()
def main():
    import json

    results = run_e11_gate0.remote()
    p = "experiments/E11_interaction_vector_scoping/results/e11_gate0_results.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {p}")
