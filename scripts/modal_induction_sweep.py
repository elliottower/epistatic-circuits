"""Modal: coalition sweep + Walsh-Hadamard for induction task.

Induction circuit (Olsson et al. 2022): 7 heads (5 induction + 2 previous token).
2^7 = 128 coalitions — trivially fast.

Metric: logit(correct_next_token) - mean_logit(other_vocab)
where correct_next_token is the token that appeared after the previous
occurrence of the current token in the prefix.

Prompts are random-token sequences with a repeated bigram: [...A B ... A ?]
The model should predict B (induction).

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_induction_sweep.py
"""

import modal

app = modal.App("induction-coalition-sweep-and-analysis")

results_volume = modal.Volume.from_name("induction-sweep-results", create_if_missing=True)

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
        "src/walsh.py",
        remote_path="/app/walsh.py",
    )
)

# Olsson et al. 2022, Table 5 Appendix
INDUCTION_KNOWN_CIRCUIT = [
    (2, 2), (4, 11),  # previous token heads
    (5, 1), (5, 5), (6, 9), (7, 2), (7, 10),  # induction heads
]

N_CIRCUIT_HEADS = len(INDUCTION_KNOWN_CIRCUIT)  # 7
N_PROMPTS = 500
SEQ_LEN = 30  # tokens per sequence (excluding BOS)


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
    vocab_size = model.cfg.d_vocab

    print(f"[{ts()}] Model loaded")

    # ---- Generate induction prompts ----
    # Pattern: [random tokens ... A B ... more random ... A ?]
    # Model should predict B at position of ?
    # Use common single-token words to make the task natural
    print(f"[{ts()}] Generating {N_PROMPTS} induction prompts...")

    rng = np.random.RandomState(42)
    # Use tokens in range [1000, 10000] to avoid special tokens and very rare tokens
    token_pool = list(range(1000, 10000))

    all_tokens_list = []
    correct_ids_list = []

    for _ in range(N_PROMPTS):
        # Pick random tokens for the sequence
        seq_tokens = rng.choice(token_pool, size=SEQ_LEN, replace=False).tolist()

        # Pick positions for the repeated bigram
        # A appears at positions first_a and second_a; B appears at first_a+1
        first_a = rng.randint(2, SEQ_LEN // 2)
        second_a = rng.randint(SEQ_LEN // 2 + 2, SEQ_LEN - 1)

        A = seq_tokens[first_a]
        B = seq_tokens[first_a + 1]

        # Place A at second_a position
        seq_tokens[second_a] = A

        # Correct prediction at second_a position: B
        correct_ids_list.append(B)

        # Build full sequence: BOS + tokens up to and including second_a
        full_seq = [model.tokenizer.bos_token_id] + seq_tokens[:second_a + 1]
        all_tokens_list.append(full_seq)

    # Pad to same length
    max_len = max(len(s) for s in all_tokens_list)
    padded_tokens = torch.zeros(N_PROMPTS, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(N_PROMPTS, dtype=np.int64)
    for i, toks in enumerate(all_tokens_list):
        padded_tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        seq_lens[i] = len(toks)

    last_positions = torch.tensor(seq_lens - 1, device="cuda")
    correct_ids = torch.tensor(correct_ids_list, device="cuda")
    n_prompts = N_PROMPTS

    print(f"[{ts()}] Generated {n_prompts} prompts, max_len={max_len}")

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

    # Metric: logit of correct token (induction target)
    # Use log-prob of correct token as scalar metric
    def compute_induction_score(logits_at_last, n_coal=1):
        """Log probability of the correct induction token."""
        if n_coal == 1:
            log_probs = torch.log_softmax(logits_at_last, dim=-1)
            seq_idx = torch.arange(n_prompts, device="cuda")
            return log_probs[seq_idx, correct_ids]
        else:
            log_probs = torch.log_softmax(logits_at_last.view(n_coal, n_prompts, -1), dim=-1)
            tgt = correct_ids.unsqueeze(0).expand(n_coal, -1)
            return log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

    # ---- Baseline ----
    with torch.no_grad():
        baseline_logits = forward_last_logits(padded_tokens, n_coal=1)
        baseline_score = compute_induction_score(baseline_logits, n_coal=1)
        baseline_mean = float(baseline_score.mean())
        top1 = baseline_logits.argmax(dim=-1)
        accuracy = float((top1 == correct_ids).float().mean())
    print(f"[{ts()}] Intact model: mean log-prob = {baseline_mean:.4f}, "
          f"top-1 accuracy = {accuracy:.3f}")
    del baseline_logits
    torch.cuda.empty_cache()

    # ---- Head-level attribution (skip if already computed) ----
    existing_attr_path = "/results/induction_head_attribution.json"
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
                        ablated_score = compute_induction_score(last_logits, n_coal=1)
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
                "known_circuit": [list(h) for h in INDUCTION_KNOWN_CIRCUIT],
                "random_circuit": [list(h) for h in random_circuit],
                "baseline_log_prob": baseline_mean,
                "baseline_accuracy": accuracy,
            }, f, indent=2)
        results_volume.commit()

    # ---- Coalition sweeps ----
    print(f"\n[{ts()}] === Coalition sweeps ===")

    circuits = {
        "known": INDUCTION_KNOWN_CIRCUIT,
        "ablation_discovered": ablation_circuit,
        "random": random_circuit,
    }

    N_COAL_BATCH = 8
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
                last_logits_r = last_logits.view(n_coal, n_prompts, -1)
                scores = compute_induction_score(last_logits_r, n_coal=n_coal)
                return scores.cpu().numpy().astype(np.float64)

            score_values = np.zeros((n_total, n_prompts), dtype=np.float64)
            sweep_start = time.time()

            for mb_start in tqdm(range(0, n_total, N_COAL_BATCH), desc=key):
                mb_end = min(mb_start + N_COAL_BATCH, n_total)
                coal_indices = list(range(mb_start, mb_end))
                batch_scores = evaluate_batch(coal_indices)
                for i, ci in enumerate(coal_indices):
                    score_values[ci] = batch_scores[i]

            elapsed = time.time() - sweep_start
            print(f"[{ts()}] {key}: {n_total} coalitions in {elapsed:.1f}s")

            np.savez(
                f"/results/induction_{key}_coalition_values.npz",
                log_prob_scores=score_values,
                circuit_heads=circuit_heads_arr,
                n_players=n_players,
                n_prompts=n_prompts,
                circuit_name=circuit_name,
                ablation_type=ablation_type,
            )
            results_volume.commit()

            all_sweep_results[key] = score_values

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
        for order in range(1, n_players + 1):
            if energy_nc_frac[order - 1] > 0.005:
                print(f"    Order {order}: {energy_nc_frac[order-1]*100:.1f}%")

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
            "task": "induction",
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

    print(f"\n\n{'='*70}")
    print(f"SUMMARY — Induction Task")
    print(f"{'='*70}")
    print(f"\n{'Circuit':<22} {'Abl':<6} {'Faith':>8} {'Ord-1':>7} {'Ord-2':>7} {'Ord-3+':>7} {'Epi':>8}")
    print("-" * 72)
    for key in sorted(analysis_results.keys()):
        r = analysis_results[key]
        print(f"{r['circuit']:<22} {r['ablation']:<6} {r['faithfulness']:>+8.4f} "
              f"{r['order1_frac']*100:>6.1f}% {r['order2_frac']*100:>6.1f}% "
              f"{r['order3plus_frac']*100:>6.1f}% {r['epistasis_point']*100:>7.1f}%")

    with open("/results/induction_analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
    results_volume.commit()
    print(f"\n[{ts()}] All results saved")

    return analysis_results


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching induction pipeline")
    results = run_full_pipeline.remote()
    print("\nFinal results:")
    for key, r in results.items():
        print(f"  {r['circuit']:22s} {r['ablation']:6s}  "
              f"faith={r['faithfulness']:+.4f}  "
              f"o1={r['order1_frac']*100:.1f}%  o2={r['order2_frac']*100:.1f}%  "
              f"o3+={r['order3plus_frac']*100:.1f}%  "
              f"epi={r['epistasis_point']*100:.1f}%")
