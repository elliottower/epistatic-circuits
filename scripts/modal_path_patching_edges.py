"""Directional edge decomposition of Walsh interactions via path patching.

For each of the 190 pairs in the 20-head IOI circuit, computes the direct
path-patch effect (sender→receiver through residual stream, all intermediates
frozen to clean). Compares to undirected Walsh coefficients from Phase 2.

Pre-reg SHA: 4607cd9a (PREREG_PATH_PATCHING_EDGES.md)

Usage: cd epistatic-circuits && modal run --detach scripts/modal_path_patching_edges.py
"""
import modal

app = modal.App("path-patching-walsh-edges")

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
out_vol = modal.Volume.from_name("path-patching-edges-results", create_if_missing=True)

N_PROMPTS = 200


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
    volumes={"/phase2": phase2_vol, "/out": out_vol},
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

    # ── Load Phase 2 head selection ──────────────────────────────────
    phase2_vol.reload()
    with open("/phase2/phase2_head_selection.json") as f:
        head_sel = json.load(f)
    selected_heads = [tuple(h) for h in head_sel["selected_heads"]]
    note(f"Loaded {len(selected_heads)} heads from Phase 2")

    with open("/phase2/phase2_all_walsh_coefficients.json") as f:
        all_coeffs = json.load(f)

    walsh_coeffs = {}
    for key, val in all_coeffs.items():
        if val["order"] == 2:
            walsh_coeffs[key] = val["coeff"]

    note(f"Loaded {len(walsh_coeffs)} Walsh pair coefficients")

    # ── Generate IOI prompts (same seed as Phase 2) ──────────────────
    note("Generating IOI prompts")
    prompts, io_token_ids, s_token_ids = generate_ioi_prompts(
        N_PROMPTS, model.tokenizer
    )

    tokens = model.to_tokens(prompts, prepend_bos=True)
    seq_len = tokens.shape[1]
    last_pos = seq_len - 1
    note(f"Tokenized: {tokens.shape}")

    io_ids = torch.tensor(io_token_ids, device=device)
    s_ids = torch.tensor(s_token_ids, device=device)
    batch_idx = torch.arange(N_PROMPTS, device=device)

    # ── Clean pass + full cache ──────────────────────────────────────
    note("Clean forward pass with full caching")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    clean_ld = float(
        (logits[:, last_pos, :][batch_idx, io_ids]
         - logits[:, last_pos, :][batch_idx, s_ids]).mean().item()
    )
    note(f"Clean logit diff: {clean_ld:.4f}")

    # Cache clean hook_z and hook_attn_out and hook_mlp_out
    clean_z = {}
    clean_attn_out = {}
    clean_mlp_out = {}
    mean_z = {}

    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        clean_z[layer] = z.clone()
        clean_attn_out[layer] = cache[f"blocks.{layer}.hook_attn_out"].clone()
        clean_mlp_out[layer] = cache[f"blocks.{layer}.hook_mlp_out"].clone()
        for head in range(n_heads_per_layer):
            mean_z[(layer, head)] = z[:, :, head, :].mean(dim=0)

    del cache
    torch.cuda.empty_cache()
    note("Cached clean activations and mean z")

    # ── Path patching for all 190 pairs ──────────────────────────────
    note("Starting path patching for all pairs")

    pairs = list(itertools.combinations(range(len(selected_heads)), 2))
    results = []

    for pair_idx, (i, j) in enumerate(tqdm(pairs, desc="Path patching")):
        head_i = selected_heads[i]
        head_j = selected_heads[j]
        li, hi = head_i
        lj, hj = head_j

        pair_name = f"L{li}H{hi}-L{lj}H{hj}"
        reverse_name = f"L{lj}H{hj}-L{li}H{hi}"

        # Look up Walsh coefficient (stored as "L_H_-L_H_" with lower index first)
        w_coeff = walsh_coeffs.get(pair_name, walsh_coeffs.get(reverse_name, 0.0))

        # Determine direction: earlier layer is sender
        if li < lj:
            s_layer, s_head = li, hi
            r_layer, r_head = lj, hj
            direction = "forward"
        elif lj < li:
            s_layer, s_head = lj, hj
            r_layer, r_head = li, hi
            direction = "forward"
        else:
            # Same layer — no direct path
            results.append({
                "pair": pair_name,
                "sender": f"L{li}H{hi}",
                "receiver": f"L{lj}H{hj}",
                "same_layer": True,
                "layer_distance": 0,
                "walsh_coeff": round(float(w_coeff), 6),
                "direct_effect": 0.0,
                "direction": "same_layer",
            })
            continue

        sender_name = f"L{s_layer}H{s_head}"
        receiver_name = f"L{r_layer}H{r_head}"
        layer_dist = r_layer - s_layer

        # Build hooks for path patching:
        # 1. Corrupt sender's hook_z
        # 2. Freeze all attention (hook_z) in layers between sender and receiver
        # 3. Freeze all MLPs (hook_mlp_out) from sender's layer to receiver-1
        # 4. Freeze non-receiver heads in receiver's layer
        # 5. Let receiver and downstream compute naturally
        hooks = []

        mz = mean_z[(s_layer, s_head)]

        def make_corrupt_sender(sh, m):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, :, sh, :] = m.unsqueeze(0)
                return value
            return hook_fn

        hooks.append((
            f"blocks.{s_layer}.attn.hook_z",
            make_corrupt_sender(s_head, mz),
        ))

        # Freeze intermediate attention layers
        for layer in range(s_layer + 1, r_layer):
            cz = clean_z[layer]

            def make_freeze_z(c):
                def hook_fn(value, hook):
                    return c
                return hook_fn

            hooks.append((
                f"blocks.{layer}.attn.hook_z",
                make_freeze_z(cz),
            ))

        # Freeze intermediate MLPs (sender's layer through receiver-1)
        for layer in range(s_layer, r_layer):
            cm = clean_mlp_out[layer]

            def make_freeze_mlp(c):
                def hook_fn(value, hook):
                    return c
                return hook_fn

            hooks.append((
                f"blocks.{layer}.hook_mlp_out",
                make_freeze_mlp(cm),
            ))

        # Freeze non-receiver heads in receiver's layer
        cz_r = clean_z[r_layer]

        def make_freeze_non_receiver(c, rh):
            def hook_fn(value, hook):
                result = c.clone()
                result[:, :, rh, :] = value[:, :, rh, :]
                return result
            return hook_fn

        hooks.append((
            f"blocks.{r_layer}.attn.hook_z",
            make_freeze_non_receiver(cz_r, r_head),
        ))

        with torch.no_grad():
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        patched_ld = float(
            (patched_logits[:, last_pos, :][batch_idx, io_ids]
             - patched_logits[:, last_pos, :][batch_idx, s_ids]).mean().item()
        )

        direct_effect = clean_ld - patched_ld

        results.append({
            "pair": pair_name,
            "sender": sender_name,
            "receiver": receiver_name,
            "same_layer": False,
            "layer_distance": layer_dist,
            "walsh_coeff": round(float(w_coeff), 6),
            "direct_effect": round(float(direct_effect), 6),
            "direction": direction,
        })

        if (pair_idx + 1) % 50 == 0:
            note(f"  {pair_idx + 1}/{len(pairs)} pairs done")
            Path("/out/path_patching_checkpoint.json").write_text(
                json.dumps(results, indent=1)
            )
            out_vol.commit()

    note(f"Path patching complete: {len(results)} pairs")

    # ── Analysis ─────────────────────────────────────────────────────
    cross_layer = [r for r in results if not r["same_layer"]]
    same_layer = [r for r in results if r["same_layer"]]

    walsh_abs = np.array([abs(r["walsh_coeff"]) for r in cross_layer])
    pp_abs = np.array([abs(r["direct_effect"]) for r in cross_layer])

    rho_spearman = float(spearmanr(walsh_abs, pp_abs).statistic)
    rho_pearson = float(np.corrcoef(walsh_abs, pp_abs)[0, 1])

    # Signed correlation (Walsh coeff vs direct effect)
    walsh_signed = np.array([r["walsh_coeff"] for r in cross_layer])
    pp_signed = np.array([r["direct_effect"] for r in cross_layer])
    rho_signed = float(np.corrcoef(walsh_signed, pp_signed)[0, 1])

    note(f"Correlations (cross-layer, n={len(cross_layer)}):")
    note(f"  |Walsh| vs |PP|: Spearman={rho_spearman:.4f}, Pearson={rho_pearson:.4f}")
    note(f"  Signed: Pearson={rho_signed:.4f}")

    # Same-layer sanity check
    same_walsh = [abs(r["walsh_coeff"]) for r in same_layer]
    note(f"Same-layer pairs: {len(same_layer)}, "
         f"Walsh magnitudes: {[round(w, 4) for w in same_walsh]}")

    # Mediation analysis: top-quartile Walsh with bottom-quartile PP
    walsh_q75 = np.percentile(walsh_abs, 75)
    pp_q25 = np.percentile(pp_abs, 25)
    strong_walsh_weak_pp = [
        r for r in cross_layer
        if abs(r["walsh_coeff"]) >= walsh_q75 and abs(r["direct_effect"]) <= pp_q25
    ]
    mediation_frac = len(strong_walsh_weak_pp) / max(1, sum(
        1 for r in cross_layer if abs(r["walsh_coeff"]) >= walsh_q75
    ))

    note(f"Mediation fraction (top-25% Walsh, bottom-25% PP): "
         f"{mediation_frac:.1%} ({len(strong_walsh_weak_pp)} pairs)")

    # Top pairs by direct effect
    top_pp = sorted(cross_layer, key=lambda r: abs(r["direct_effect"]), reverse=True)[:20]
    note("Top 20 pairs by |direct effect|:")
    for r in top_pp:
        note(f"  {r['sender']}→{r['receiver']}: "
             f"PP={r['direct_effect']:.4f}, Walsh={r['walsh_coeff']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    summary = {
        "experiment": "Path patching edge decomposition of Walsh interactions",
        "prereg_sha": "4607cd9a",
        "n_heads": len(selected_heads),
        "n_prompts": N_PROMPTS,
        "n_pairs_total": len(results),
        "n_cross_layer": len(cross_layer),
        "n_same_layer": len(same_layer),
        "clean_logit_diff": clean_ld,
        "correlations": {
            "abs_spearman": round(rho_spearman, 4),
            "abs_pearson": round(rho_pearson, 4),
            "signed_pearson": round(rho_signed, 4),
        },
        "mediation_fraction": round(mediation_frac, 4),
        "same_layer_walsh_magnitudes": [round(w, 6) for w in same_walsh],
        "top20_by_direct_effect": [
            {
                "sender": r["sender"],
                "receiver": r["receiver"],
                "direct_effect": r["direct_effect"],
                "walsh_coeff": r["walsh_coeff"],
                "layer_distance": r["layer_distance"],
            }
            for r in top_pp
        ],
        "all_pairs": results,
    }

    Path("/out/path_patching_results.json").write_text(
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
        c = result["correlations"]
        print(f"\n=== PATH PATCHING RESULTS ===")
        print(f"|Walsh| vs |PP|: Spearman={c['abs_spearman']:.4f}, "
              f"Pearson={c['abs_pearson']:.4f}")
        print(f"Signed Pearson: {c['signed_pearson']:.4f}")
        print(f"Mediation fraction: {result['mediation_fraction']:.1%}")
        print(f"\nTop 5 direct edges:")
        for r in result["top20_by_direct_effect"][:5]:
            print(f"  {r['sender']}→{r['receiver']}: "
                  f"PP={r['direct_effect']:.4f}, Walsh={r['walsh_coeff']:.4f}")
