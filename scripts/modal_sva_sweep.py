"""Modal: coalition sweep + Walsh-Hadamard for SVA (subject-verb agreement).

SVA circuit (Lazo et al. 2025): 12 heads.
2^12 = 4,096 coalitions, subsampled to 500 prompts.

Metric: sum P(agreeing verbs) - sum P(disagreeing verbs) over all
single-token verb pairs from Marvin & Linzen vocabulary.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_sva_sweep.py
"""

import modal

app = modal.App("sva-coalition-sweep-and-analysis")

results_volume = modal.Volume.from_name("sva-sweep-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "transformer-lens==2.17.0",
        "transformers==4.51.3",
        "typeguard==4.3.0",
        "matplotlib==3.9.4",
    )
    .add_local_file(
        "data/sva_data.csv",
        remote_path="/app/sva_data.csv",
    )
    .add_local_file(
        "data/sva_verb_list.csv",
        remote_path="/app/sva_verb_list.csv",
    )
    .add_local_file(
        "src/walsh.py",
        remote_path="/app/walsh.py",
    )
)

# Lazo et al. 2025, Section 4.1 + Figure 1
SVA_KNOWN_CIRCUIT = [
    (11, 7), (11, 6), (0, 4), (11, 4), (0, 8), (2, 6),
    (1, 0), (2, 1), (1, 1), (6, 0), (10, 0), (9, 4),
]

N_CIRCUIT_HEADS = len(SVA_KNOWN_CIRCUIT)  # 12
N_PROMPTS_SUBSAMPLE = 500
N_COAL_BATCH = 8


def make_random_circuit(n_heads, seed=42):
    import random
    rng = random.Random(seed)
    all_heads = [(l, h) for l in range(12) for h in range(12)]
    return sorted(rng.sample(all_heads, n_heads))


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_full_pipeline():
    import csv
    import gc
    import json
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")
    from walsh import energy_by_order_normalized, wht, wht_per_head_energy

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()

    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    print(f"[{ts()}] Model loaded")

    # ---- Build verb token lists ----
    print(f"[{ts()}] Building verb token lists...")
    singular_ids = []
    plural_ids = []
    with open("/app/sva_verb_list.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sing_form = " " + row["sing"]
            plur_form = " " + row["plur"]
            sing_toks = model.tokenizer.encode(sing_form)
            plur_toks = model.tokenizer.encode(plur_form)
            if len(sing_toks) == 1 and len(plur_toks) == 1:
                singular_ids.append(sing_toks[0])
                plural_ids.append(plur_toks[0])

    singular_ids = torch.tensor(singular_ids, device="cuda")
    plural_ids = torch.tensor(plural_ids, device="cuda")
    print(f"[{ts()}] {len(singular_ids)} single-token verb pairs")

    # ---- Load and subsample prompts ----
    all_prompts = []
    with open("/app/sva_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_prompts.append({
                "text": row["clean"],
                "plural": int(row["plural"]),
                "group": row["group"],
            })

    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_prompts), size=N_PROMPTS_SUBSAMPLE, replace=False)
    prompts = [all_prompts[i] for i in indices]
    n_prompts = len(prompts)
    print(f"[{ts()}] Subsampled {n_prompts} of {len(all_prompts)} prompts")

    plural_flags = np.array([p["plural"] for p in prompts])

    # ---- Tokenize ----
    all_tokens = []
    max_len = 0
    for p in prompts:
        toks = model.to_tokens(p["text"], prepend_bos=True)
        all_tokens.append(toks)
        max_len = max(max_len, toks.shape[1])

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]
    last_positions = torch.tensor(seq_lens - 1, device="cuda")

    # Sign: +1 for plural subjects (should have high P(plural verbs)),
    #        -1 for singular subjects
    signs = torch.tensor([1 if p else -1 for p in plural_flags],
                         device="cuda", dtype=torch.float32)

    def compute_verb_score(logits_at_last):
        """Score = sign * [sum P(agreeing verbs) - sum P(disagreeing verbs)].

        logits_at_last: (..., vocab_size)
        Returns per-prompt scores.
        """
        orig_shape = logits_at_last.shape[:-1]
        flat_logits = logits_at_last.reshape(-1, logits_at_last.shape[-1])
        sing_logits = flat_logits[:, singular_ids]
        plur_logits = flat_logits[:, plural_ids]
        all_verb_logits = torch.cat([sing_logits, plur_logits], dim=-1)
        all_verb_probs = torch.softmax(all_verb_logits, dim=-1)
        n_verbs = len(singular_ids)
        sing_probs = all_verb_probs[:, :n_verbs].sum(dim=-1)
        plur_probs = all_verb_probs[:, n_verbs:].sum(dim=-1)
        diff = (plur_probs - sing_probs).reshape(orig_shape)
        if diff.dim() == 1 and diff.shape[0] == n_prompts:
            return diff * signs
        elif diff.dim() == 2:
            return diff * signs.unsqueeze(0)
        return diff

    # ---- Mean z ----
    print(f"[{ts()}] Computing mean z activations...")
    z_sums = {}
    total_positions = 0
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_tokens[start:start + 64]
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: "attn.hook_z" in n,
            )
            for l in range(n_layers):
                act = cache[f"blocks.{l}.attn.hook_z"]
                if l not in z_sums:
                    z_sums[l] = torch.zeros(n_model_heads, d_head, device="cuda", dtype=act.dtype)
                z_sums[l] += act.sum(dim=(0, 1))
            total_positions += batch.shape[0] * batch.shape[1]
            del cache
    mean_z = {l: z_sums[l] / total_positions for l in range(n_layers)}
    torch.cuda.empty_cache()
    gc.collect()

    def forward_last_logits(tokens, n_coal=1):
        residual = model.hook_embed(model.embed(tokens))
        pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
        residual = residual + pos_embed
        for block in model.blocks:
            residual = block(residual)
        if n_coal > 1:
            tiled_last = last_positions.repeat(n_coal)
        else:
            tiled_last = last_positions
        seq_idx = torch.arange(tokens.shape[0], device="cuda")
        last_resid = residual[seq_idx, tiled_last, :]
        del residual
        last_normed = model.ln_final(last_resid)
        del last_resid
        return last_normed @ model.W_U + model.b_U

    # ---- Baseline ----
    with torch.no_grad():
        baseline_logits = forward_last_logits(padded_tokens, n_coal=1)
        baseline_score = compute_verb_score(baseline_logits)
        baseline_mean = float(baseline_score.mean())
        baseline_acc = float((baseline_score > 0).float().mean())
    print(f"[{ts()}] Intact model: mean verb_score = {baseline_mean:.4f}, "
          f"accuracy = {baseline_acc:.3f}")
    del baseline_logits
    torch.cuda.empty_cache()

    # ---- Head-level attribution (skip if already computed) ----
    existing_attr_path = "/results/sva_head_attribution.json"
    try:
        with open(existing_attr_path) as f:
            attr_results = json.load(f)
        ablation_circuit = [tuple(h) for h in attr_results["ablation_circuit"]]
        random_circuit = [tuple(h) for h in attr_results["random_circuit"]]
        print(f"[{ts()}] Loaded existing attribution")
        print(f"  Ablation circuit: {[f'L{l}H{h}' for l, h in ablation_circuit]}")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"\n[{ts()}] === Head-level attribution ===")
        head_importance = {}

        for ablation_type in ["zero", "mean"]:
            importance = np.zeros((n_layers, n_model_heads))
            for layer in tqdm(range(n_layers), desc=f"Attr-{ablation_type}"):
                for head in range(n_model_heads):
                    def hook_fn(act, hook, l=layer, h=head):
                        if ablation_type == "zero":
                            act[:, :, h, :] = 0.0
                        else:
                            act[:, :, h, :] = mean_z[l][h]
                        return act

                    with torch.no_grad():
                        logits = model.run_with_hooks(
                            padded_tokens,
                            fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)],
                            return_type="logits",
                        )
                        seq_idx = torch.arange(n_prompts, device="cuda")
                        last_logits = logits[seq_idx, last_positions, :]
                        ablated_score = compute_verb_score(last_logits)
                        importance[layer, head] = float(
                            (baseline_score - ablated_score).mean()
                        )
                        del logits, last_logits

            head_importance[ablation_type] = importance

        mean_imp = head_importance["mean"]
        flat = mean_imp.flatten()
        top_idx = np.argsort(np.abs(flat))[::-1][:N_CIRCUIT_HEADS]
        ablation_circuit = sorted(
            [(int(idx // n_model_heads), int(idx % n_model_heads)) for idx in top_idx]
        )
        random_circuit = make_random_circuit(N_CIRCUIT_HEADS)

        print(f"\n[{ts()}] Ablation-discovered circuit: {[f'L{l}H{h}' for l, h in ablation_circuit]}")

        with open(existing_attr_path, "w") as f:
            json.dump({
                "zero_importance": {f"L{l}H{h}": float(head_importance["zero"][l, h])
                                    for l in range(n_layers) for h in range(n_model_heads)},
                "mean_importance": {f"L{l}H{h}": float(head_importance["mean"][l, h])
                                    for l in range(n_layers) for h in range(n_model_heads)},
                "ablation_circuit": [list(h) for h in ablation_circuit],
                "known_circuit": [list(h) for h in SVA_KNOWN_CIRCUIT],
                "random_circuit": [list(h) for h in random_circuit],
                "baseline_verb_score": baseline_mean,
                "baseline_accuracy": baseline_acc,
                "n_verb_pairs": int(len(singular_ids)),
            }, f, indent=2)
        results_volume.commit()

    # ---- Coalition sweeps ----
    print(f"\n[{ts()}] === Coalition sweeps ===")

    circuits = {
        "known": SVA_KNOWN_CIRCUIT,
        "ablation_discovered": ablation_circuit,
        "random": random_circuit,
    }

    all_sweep_results = {}

    for circuit_name, circuit_heads in circuits.items():
        circuit_heads_arr = np.array(circuit_heads)
        n_players = len(circuit_heads)
        n_total = 2 ** n_players
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
        involved_layers = sorted(set(l for l, h in circuit_heads))

        for ablation_type in ["zero", "mean"]:
            key = f"{circuit_name}_{ablation_type}"
            print(f"\n[{ts()}] Sweep: {key} ({n_total} coalitions)")

            max_batch_total = N_COAL_BATCH * n_prompts
            layer_mask_tensors = {}
            for l in involved_layers:
                layer_mask_tensors[l] = torch.ones(
                    max_batch_total, 1, n_model_heads, 1, device="cuda"
                )

            mean_z_expanded = {}
            for l in involved_layers:
                mean_z_expanded[l] = mean_z[l].view(1, 1, n_model_heads, d_head)

            model.reset_hooks(including_permanent=True)

            if ablation_type == "zero":
                def make_hook(layer):
                    def hook_fn(act, hook):
                        n = act.shape[0]
                        return act * layer_mask_tensors[layer][:n]
                    return hook_fn
            else:
                def make_hook(layer):
                    def hook_fn(act, hook):
                        n = act.shape[0]
                        mask = layer_mask_tensors[layer][:n]
                        return act * mask + mean_z_expanded[layer] * (1 - mask)
                    return hook_fn

            for l in involved_layers:
                model.add_perma_hook(f"blocks.{l}.attn.hook_z", make_hook(l))

            def build_masks(coal_indices):
                n_coal = len(coal_indices)
                for l in involved_layers:
                    layer_mask_tensors[l][:n_coal * n_prompts] = 1.0
                for c_pos, coal_idx in enumerate(coal_indices):
                    mask_bits = np.array(
                        [(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool
                    )
                    row_start = c_pos * n_prompts
                    row_end = row_start + n_prompts
                    for i, (layer, head) in enumerate(circuit_heads):
                        if not mask_bits[i]:
                            layer_mask_tensors[layer][row_start:row_end, 0, head, 0] = 0.0

            def evaluate_batch(coal_indices):
                n_coal = len(coal_indices)
                build_masks(coal_indices)
                tiled_tokens = padded_tokens.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)
                with torch.no_grad():
                    last_logits = forward_last_logits(tiled_tokens, n_coal=n_coal)
                last_logits = last_logits.view(n_coal, n_prompts, -1)
                scores = compute_verb_score(last_logits)
                return scores.cpu().numpy().astype(np.float64)

            verb_scores = np.zeros((n_total, n_prompts), dtype=np.float64)
            sweep_start = time.time()

            n_batches = (n_total + N_COAL_BATCH - 1) // N_COAL_BATCH
            checkpoint_every = 256

            for mb_start in tqdm(range(0, n_total, N_COAL_BATCH),
                                 desc=key, total=n_batches):
                mb_end = min(mb_start + N_COAL_BATCH, n_total)
                coal_indices = list(range(mb_start, mb_end))
                try:
                    batch_scores = evaluate_batch(coal_indices)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    half = len(coal_indices) // 2
                    for sub_start in range(0, len(coal_indices), max(half, 1)):
                        sub_end = min(sub_start + max(half, 1), len(coal_indices))
                        sub = coal_indices[sub_start:sub_end]
                        sub_scores = evaluate_batch(sub)
                        for i, ci in enumerate(sub):
                            verb_scores[ci] = sub_scores[i]
                    continue
                for i, ci in enumerate(coal_indices):
                    verb_scores[ci] = batch_scores[i]

                if mb_start > 0 and mb_start % (checkpoint_every * N_COAL_BATCH) < N_COAL_BATCH:
                    elapsed = time.time() - sweep_start
                    done = mb_end
                    rate = done / elapsed
                    remaining = (n_total - done) / rate if rate > 0 else 0
                    print(f"[{ts()}] {done}/{n_total} done, "
                          f"{rate:.1f} coal/s, ~{remaining/60:.1f}min remaining")
                    np.savez(
                        f"/results/sva_{key}_coalition_values.npz",
                        verb_scores=verb_scores,
                        circuit_heads=circuit_heads_arr,
                        n_players=n_players,
                        n_prompts=n_prompts,
                        circuit_name=circuit_name,
                        ablation_type=ablation_type,
                        n_completed=done,
                    )
                    results_volume.commit()

            elapsed = time.time() - sweep_start
            print(f"[{ts()}] {key}: {n_total} coalitions in {elapsed:.1f}s")

            np.savez(
                f"/results/sva_{key}_coalition_values.npz",
                verb_scores=verb_scores,
                circuit_heads=circuit_heads_arr,
                n_players=n_players,
                n_prompts=n_prompts,
                circuit_name=circuit_name,
                ablation_type=ablation_type,
                n_completed=n_total,
            )
            results_volume.commit()

            all_sweep_results[key] = verb_scores

    model.reset_hooks(including_permanent=True)
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Walsh-Hadamard analysis ----
    print(f"\n[{ts()}] === Walsh-Hadamard analysis ===")

    analysis_results = {}

    for key, scores in all_sweep_results.items():
        circuit_name = key.rsplit("_", 1)[0]
        abl_type = key.rsplit("_", 1)[1]
        circuit_heads = circuits[circuit_name]
        n_players = len(circuit_heads)
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]

        mean_score = scores.mean(axis=1)
        full_mask = (1 << n_players) - 1
        intact = float(mean_score[full_mask])
        empty = float(mean_score[0])
        faithfulness = intact - empty

        wht_coeffs = wht(mean_score.astype(np.float64))
        energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)
        energy_nc = energy_spectrum[1:].copy()
        total_nc = energy_nc.sum()
        energy_nc_frac = energy_nc / total_nc if total_nc > 0 else energy_nc

        head_energy = wht_per_head_energy(mean_score.astype(np.float64), n_players)

        print(f"\n  {key}:")
        print(f"    Intact: {intact:.4f}, Empty: {empty:.4f}, Faith: {faithfulness:+.4f}")
        for order in range(1, min(n_players + 1, 8)):
            if energy_nc_frac[order - 1] > 0.005:
                print(f"    Order {order}: {energy_nc_frac[order-1]*100:.1f}%")
        high_order = sum(energy_nc_frac[2:])
        print(f"    Order 3+: {high_order*100:.1f}%")

        # Epistasis
        per_prompt_group = scores[full_mask] - scores[0]
        per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
        for i in range(n_players):
            complement = full_mask ^ (1 << i)
            per_prompt_loo_sum += scores[full_mask] - scores[complement]
        group_mean = float(np.mean(per_prompt_group))
        loo_sum_mean = float(np.mean(per_prompt_loo_sum))
        epistasis_point = 1.0 - loo_sum_mean / group_mean if abs(group_mean) > 1e-10 else 0.0

        rng_boot = np.random.RandomState(42)
        boot_vals = []
        for _ in range(10_000):
            idx = rng_boot.choice(n_prompts, size=n_prompts, replace=True)
            bg = float(np.mean(per_prompt_group[idx]))
            bl = float(np.mean(per_prompt_loo_sum[idx]))
            if abs(bg) > 1e-10:
                boot_vals.append(1.0 - bl / bg)
        boot_vals = np.array(boot_vals)
        ci_lo = float(np.percentile(boot_vals, 2.5))
        ci_hi = float(np.percentile(boot_vals, 97.5))
        print(f"    Epistasis: {epistasis_point*100:.1f}% [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

        analysis_results[key] = {
            "circuit": circuit_name,
            "ablation": abl_type,
            "task": "sva",
            "n_players": n_players,
            "n_prompts": n_prompts,
            "heads": head_labels,
            "intact_score": round(intact, 6),
            "empty_score": round(empty, 6),
            "faithfulness": round(faithfulness, 6),
            "wht_energy_spectrum": [round(float(x), 6) for x in energy_spectrum],
            "wht_energy_nc_frac": [round(float(x), 6) for x in energy_nc_frac],
            "order1_frac": round(float(energy_nc_frac[0]), 4),
            "order2_frac": round(float(energy_nc_frac[1]), 4),
            "order3plus_frac": round(float(sum(energy_nc_frac[2:])), 4),
            "per_head_energy": {head_labels[i]: round(float(head_energy[i]), 6) for i in range(n_players)},
            "epistasis_point": round(epistasis_point, 4),
            "epistasis_ci_lo": round(ci_lo, 4),
            "epistasis_ci_hi": round(ci_hi, 4),
            "group_effect": round(group_mean, 6),
            "loo_sum": round(loo_sum_mean, 6),
        }

    print(f"\n\n{'='*80}")
    print(f"SUMMARY — SVA Task")
    print(f"{'='*80}")
    print(f"\n{'Circuit':<22} {'Abl':<6} {'Faith':>8} {'Ord-1':>7} {'Ord-2':>7} {'Ord-3+':>7} {'Epi':>8}")
    print("-" * 72)
    for key in sorted(analysis_results.keys()):
        r = analysis_results[key]
        print(f"{r['circuit']:<22} {r['ablation']:<6} {r['faithfulness']:>+8.4f} "
              f"{r['order1_frac']*100:>6.1f}% {r['order2_frac']*100:>6.1f}% "
              f"{r['order3plus_frac']*100:>6.1f}% {r['epistasis_point']*100:>7.1f}%")

    with open("/results/sva_analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    results_volume.commit()
    print(f"\n[{ts()}] All results saved to sva-sweep-results volume")

    return analysis_results


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching SVA pipeline")
    results = run_full_pipeline.remote()
    print("\nFinal results:")
    for key, r in results.items():
        print(f"  {r['circuit']:22s} {r['ablation']:6s}  "
              f"faith={r['faithfulness']:+.4f}  "
              f"o1={r['order1_frac']*100:.1f}%  o2={r['order2_frac']*100:.1f}%  "
              f"o3+={r['order3plus_frac']*100:.1f}%  "
              f"epi={r['epistasis_point']*100:.1f}%")
