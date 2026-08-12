"""E11 follow-up: the projection identity, and an anisotropy-aware null.

SCOPING, still not preregistered.

Two questions that both need the model, so neither could be answered from the results JSON.

(A) THE PROJECTION IDENTITY.
    The scalar Walsh coefficient and the vector delta are the second-order Moebius
    coefficient of the same lattice, read at different levels of projection: the scalar is
    the vector projected onto the logit-difference direction, d_i = W_U[:, IO_i] - W_U[:, S_i].
    If that is right, projecting delta onto d and averaging over prompts should track w_AB far
    more tightly than the 0.484 that ||delta|| gives -- and the residual ||delta_perp|| / ||delta||
    says what fraction of the interaction the scalar readout never sees.

    LayerNorm is the wrinkle: logits come from ln_final(resid), which is nonlinear. The
    projection is therefore taken through the LN Jacobian at the clean point, which is the
    correct first-order object, and the raw projection is reported alongside so the size of
    the correction is visible rather than assumed.

(B) AN ANISOTROPY-AWARE NULL FOR THE SHARED DIRECTION.
    The 190 top-1 interaction directions have a shared component explaining 10.3% of their
    variance. Calling that 20x chance assumes isotropy, and residual streams are famously not
    isotropic -- outlier dimensions run an order of magnitude above the rest. The honest null
    is the same statistic computed on residual-stream DIFFERENCES that carry no interaction:
    clean residuals differenced across unrelated prompt pairs, at the same layers.

Run:
    modal run --detach experiments/E11_interaction_vector_scoping/modal_run_e11_followup.py
"""

import os

import modal

app = modal.App("e11-followup")
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
    )
    .add_local_file(
        "results/phase2/phase2_all_walsh_coefficients.json",
        remote_path="/data/walsh.json",
    )
    .add_local_file(
        "results/phase2/phase2_head_selection.json", remote_path="/data/heads.json"
    )
    .add_local_file("data/ioi_prompts_200.json", remote_path="/data/prompts.json")
)

N_NULL_DRAWS = 190   # match the number of real pairs exactly
ABLATION = "mean"


@app.function(image=image, gpu="T4", timeout=86400, volumes={"/results": output_volume})
def run_followup():
    import json
    import time

    import numpy as np
    import torch
    from scipy import stats
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    def ts():
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    walsh = json.load(open("/data/walsh.json"))
    head_sel = json.load(open("/data/heads.json"))
    prompt_texts = json.load(open("/data/prompts.json"))

    heads = [tuple(h) for h in head_sel["selected_heads"]]
    label_to_head = {f"L{l}H{h}": (l, h) for (l, h) in heads}
    order2 = []
    for k, e in walsh.items():
        if e.get("order") != 2:
            continue
        parts = k.split("-")
        if len(parts) == 2 and all(x in label_to_head for x in parts):
            order2.append((parts[0], parts[1], float(e["coeff"])))
    print(f"[{ts()}] {len(order2)} order-2 pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    n_prompts = len(prompt_texts)

    tokens = model.to_tokens(prompt_texts)
    lengths = [model.to_tokens(t).shape[1] for t in prompt_texts]
    final_idx = torch.tensor([l - 1 for l in lengths], device=device)
    rows = torch.arange(n_prompts, device=device)

    # IO and S token ids, recovered from the template rather than reconstructed by heuristic:
    # the subject is the name that appears twice, the indirect object is the one that does not.
    io_ids, s_ids = [], []
    for t in prompt_texts:
        # skip index 0: the template opens with "When", which is capitalised and would
        # otherwise be read as a name -- and being the first non-subject entry, it would
        # become the IO and silently invert every projection.
        words = t.split()
        names = [w.strip(",.") for i, w in enumerate(words)
                 if i > 0 and w.strip(",.")[:1].isupper()]
        subj = next(n for n in names if names.count(n) > 1)
        io = next(n for n in names if n != subj)
        io_ids.append(model.to_single_token(" " + io))
        s_ids.append(model.to_single_token(" " + subj))
    io_ids = torch.tensor(io_ids, device=device)
    s_ids = torch.tensor(s_ids, device=device)

    # sanity: the clean logit difference must match the recorded baseline, or IO and S are swapped
    resid_hooks = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("attn.hook_z") or n in resid_hooks
        )
    last = clean_logits[rows, final_idx]
    ld = (last.gather(1, io_ids[:, None]) - last.gather(1, s_ids[:, None])).mean().item()
    ref = head_sel.get("baseline_logit_diff")
    print(f"[{ts()}] clean logit diff {ld:.4f}  recorded baseline {ref:.4f}")
    if ref is not None and abs(ld - ref) > 0.05:
        raise RuntimeError(
            f"logit difference {ld:.4f} does not match the recorded baseline {ref:.4f}. "
            "IO and S are probably swapped, which would silently flip every projection."
        )

    mean_z = {
        l: clean_cache[f"blocks.{l}.attn.hook_z"].mean(dim=0, keepdim=True)
        for l in range(n_layers)
    }

    def resid_final(cache):
        return cache[f"blocks.{n_layers - 1}.hook_resid_post"][rows, final_idx]

    def run_ablated(ablate):
        by_layer = {}
        for (l, h) in ablate:
            by_layer.setdefault(l, []).append(h)

        def mk(layer, hs):
            def hook(z, hook):
                for h in hs:
                    z[:, :, h, :] = mean_z[layer][:, :, h, :]
                return z
            return hook

        hooks = [(f"blocks.{l}.attn.hook_z", mk(l, hs)) for l, hs in by_layer.items()]
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            _, c = model.run_with_cache(tokens, names_filter=lambda n: n in resid_hooks)
        return resid_final(c)

    R_clean = resid_final(clean_cache)

    # ---- the logit-difference direction, and the LayerNorm Jacobian at the clean point ----
    W_U = model.W_U                                     # (d_model, vocab)
    d_dir = (W_U[:, io_ids] - W_U[:, s_ids]).T           # (n_prompts, d_model)
    d_hat = d_dir / d_dir.norm(dim=1, keepdim=True)

    # ln_final is x -> (x - mean) / std. Its Jacobian at x0, applied to a perturbation v, is
    # (v_c - (x0_c . v_c) x0_c / (d * var)) / std, with x0_c and v_c the centred vectors.
    x0 = R_clean
    x0c = x0 - x0.mean(dim=1, keepdim=True)
    var = x0c.pow(2).mean(dim=1, keepdim=True)
    std = (var + model.cfg.eps).sqrt()
    # TransformerLens folds the LayerNorm scale into the subsequent weights by default, so
    # ln_final is LayerNormPre -- centre and normalise, no learnable gain. The gain already
    # lives in model.W_U, which is what the projection direction is built from, so applying
    # it here as well would double-count it.
    assert not hasattr(model.ln_final, "w"), "ln_final has a gain; fold it or apply it once"

    def ln_jvp(v):
        vc = v - v.mean(dim=1, keepdim=True)
        corr = (x0c * vc).sum(dim=1, keepdim=True) / (d_model * var)
        return (vc - corr * x0c) / std

    results = {"clean_logit_diff": ld, "pairs": {}}
    single = {}
    needed = sorted({label_to_head[a] for a, b, _ in order2} | {label_to_head[b] for a, b, _ in order2})
    for hd in tqdm(needed, desc="single"):
        single[hd] = run_ablated([hd])

    for a_lbl, b_lbl, w in tqdm(order2, desc="pairs"):
        A, B = label_to_head[a_lbl], label_to_head[b_lbl]
        R_both = run_ablated([A, B])
        delta = R_clean - single[A] - single[B] + R_both      # (n_prompts, d_model)
        dl = ln_jvp(delta)
        proj_ln = (dl * d_hat).sum(dim=1)                     # through the LN Jacobian
        proj_raw = (delta * d_hat).sum(dim=1)                 # ignoring LN
        nrm = delta.norm(dim=1)
        nrm_ln = dl.norm(dim=1)
        # Numerator and denominator must live on the same side of LayerNorm. Dividing a
        # post-LN projection by a pre-LN norm deflates the fraction by the LN scale, which
        # is ~18x here -- it turned 9% into 0.7%, i.e. below the 1/sqrt(d) = 3.6% a random
        # direction would give, which is what made the number look like a finding.
        results["pairs"][f"{a_lbl}-{b_lbl}"] = {
            "walsh_w": w,
            "proj_ln_mean": float(proj_ln.mean()),
            "proj_raw_mean": float(proj_raw.mean()),
            "delta_norm_mean": float(nrm.mean()),
            "delta_norm_ln_mean": float(nrm_ln.mean()),
            # per-prompt fractions, each self-consistent
            "frac_in_logit_dir_raw": float((proj_raw.abs() / nrm.clamp_min(1e-12)).mean()),
            # The share of the interaction the readout SEES is an energy share, cos^2,
            # averaged per prompt. (mean cos)^2 is a Jensen lower bound on it, not the
            # value, so cos^2 is accumulated directly rather than squaring the mean.
            "energy_frac_in_logit_dir_raw": float(
                ((proj_raw / nrm.clamp_min(1e-12)) ** 2).mean()
            ),
            "energy_frac_in_logit_dir_ln": float(
                ((proj_ln / nrm_ln.clamp_min(1e-12)) ** 2).mean()
            ),
            "frac_in_logit_dir_ln": float((proj_ln.abs() / nrm_ln.clamp_min(1e-12)).mean()),
            # the signed mean matters separately: sign cancellation across prompts is what
            # distinguishes "every prompt pushes the same way" from "they cancel"
            "signed_over_abs_raw": float(
                proj_raw.mean().abs() / proj_raw.abs().mean().clamp_min(1e-12)
            ),
        }
        with open("/results/e11_followup.json", "w") as f:
            json.dump(results, f, indent=2)
        output_volume.commit()
        del R_both

    ws = np.array([v["walsh_w"] for v in results["pairs"].values()])
    pl = np.array([v["proj_ln_mean"] for v in results["pairs"].values()])
    pr_ = np.array([v["proj_raw_mean"] for v in results["pairs"].values()])
    nr = np.array([v["delta_norm_mean"] for v in results["pairs"].values()])
    fr = np.array([v["frac_in_logit_dir_raw"] for v in results["pairs"].values()])
    frl = np.array([v["frac_in_logit_dir_ln"] for v in results["pairs"].values()])
    sgn = np.array([v["signed_over_abs_raw"] for v in results["pairs"].values()])
    en  = np.array([v["energy_frac_in_logit_dir_raw"] for v in results["pairs"].values()])
    enl = np.array([v["energy_frac_in_logit_dir_ln"] for v in results["pairs"].values()])

    print(f"\n=== (A) projection identity ===")
    for nm, x in (("through LN Jacobian", pl), ("raw, ignoring LN", pr_), ("||delta|| only", nr)):
        rho, p = stats.spearmanr(np.abs(ws), np.abs(x))
        r, _ = stats.pearsonr(ws, x)
        print(f"  {nm:<22} Spearman|.| {rho:+.3f} (p={p:.1e})   Pearson signed {r:+.3f}")
    import math
    print(f"  median per-prompt fraction of ||delta|| in the logit direction:")
    print(f"    pre-LayerNorm  {np.median(fr):.4f}     post-LayerNorm {np.median(frl):.4f}")
    print(f"    random-vector expectation 1/sqrt(d) = {1/math.sqrt(d_model):.4f}")
    print(f"  median ENERGY share cos^2 (the share the readout sees):")
    print(f"    pre-LayerNorm  {np.median(en):.4f}     post-LayerNorm {np.median(enl):.4f}")
    print(f"    chance 1/d = {1/d_model:.5f}   ratio {np.median(en)*d_model:.1f}x")
    print(f"  median |mean(proj)| / mean|proj| = {np.median(sgn):.3f}"
          f"  (1.0 = every prompt pushes the same way, 0 = full cancellation)")
    results.update(
        spearman_absw_projln=float(stats.spearmanr(np.abs(ws), np.abs(pl))[0]),
        pearson_w_projln=float(stats.pearsonr(ws, pl)[0]),
        spearman_absw_norm=float(stats.spearmanr(np.abs(ws), nr)[0]),
        median_frac_in_logit_dir_raw=float(np.median(fr)),
        median_frac_in_logit_dir_ln=float(np.median(frl)),
        random_expectation_1_over_sqrt_d=float(1.0 / (d_model ** 0.5)),
        median_signed_over_abs=float(np.median(sgn)),
        median_energy_frac_raw=float(np.median(en)),
        median_energy_frac_ln=float(np.median(enl)),
    )

    # ---- (B) anisotropy-aware null for the shared direction ----
    print(f"\n=== (B) anisotropy-aware null ===")
    rng = np.random.default_rng(0)
    Rc = R_clean.cpu().numpy()
    dirs = []
    for _ in range(N_NULL_DRAWS):
        i = rng.integers(0, n_prompts, n_prompts)
        j = rng.integers(0, n_prompts, n_prompts)
        D = Rc[i] - Rc[j]                       # residual differences, no interaction
        v = np.linalg.svd(D, full_matrices=False)[2][0]
        dirs.append(v / np.linalg.norm(v))
    dirs = np.array(dirs)
    s = np.linalg.svd(dirs, full_matrices=False)[1]
    share = float((s ** 2 / (s ** 2).sum())[0])
    print(f"  top shared component of {N_NULL_DRAWS} interaction-free difference directions"
          f" explains {share*100:.1f}% of their variance")
    print(f"  the real interaction directions gave 10.3%; isotropic chance is ~0.5%")
    results["anisotropy_null_top_share"] = share
    with open("/results/e11_followup.json", "w") as f:
        json.dump(results, f, indent=2)
    output_volume.commit()
    return results


@app.local_entrypoint()
def main():
    import json

    res = run_followup.remote()
    p = "experiments/E11_interaction_vector_scoping/results/e11_followup.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {p}")
