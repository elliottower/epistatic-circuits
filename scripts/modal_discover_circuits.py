"""Modal: Discover circuits via greedy sufficiency + ACDC for all tasks.

For each task, discovers two new circuits:
  1. Greedy sufficiency (forward selection): start empty, add heads greedily
  2. ACDC-like (backward elimination): start with top-50 by attribution, prune greedily

Then runs coalition sweeps + Walsh analysis on both, matching the existing pipeline.

Checkpoints after every greedy step, ACDC prune step, and coalition sweep.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_discover_circuits.py
"""

import modal

app = modal.App("discover-circuits-all-tasks")

gt_vol = modal.Volume.from_name("gt-sweep-results", create_if_missing=True)
ind_vol = modal.Volume.from_name("induction-sweep-results", create_if_missing=True)
gender_vol = modal.Volume.from_name("gender-sweep-results", create_if_missing=True)
sva_vol = modal.Volume.from_name("sva-sweep-results", create_if_missing=True)

VOLUMES = {
    "/gt": gt_vol,
    "/ind": ind_vol,
    "/gender": gender_vol,
    "/sva": sva_vol,
}

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
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
    .add_local_file("data/greater_than_data.csv", remote_path="/app/greater_than_data.csv")
    .add_local_file("data/sva_data.csv", remote_path="/app/sva_data.csv")
    .add_local_file("data/sva_verb_list.csv", remote_path="/app/sva_verb_list.csv")
    .add_local_file("data/gender_bias_data.csv", remote_path="/app/gender_bias_data.csv")
)


def commit_vol(vol_path):
    VOLUMES[vol_path].commit()


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes=VOLUMES,
)
def run_discovery():
    import csv
    import gc
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")
    from walsh import wht, _popcount_array

    def ts():
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print(f"[{ts()}] Loading model...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"[{ts()}] Model loaded: {n_layers}L, {n_heads}H, d_head={d_head}")

    ALL_HEADS = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    # ========== TASK DEFINITIONS ==========

    def load_greater_than():
        prompts = []
        with open("/app/greater_than_data.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({"clean": row["clean"], "year": row.get("correct_idx", row.get("label"))})
        return prompts[:1000]

    def load_sva():
        prompts = []
        with open("/app/sva_data.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({"clean": row["clean"], "plural": int(row["plural"])})
        rng = np.random.default_rng(42)
        indices = rng.choice(len(prompts), size=min(500, len(prompts)), replace=False)
        return [prompts[i] for i in indices]

    def load_gender():
        prompts = []
        with open("/app/gender_bias_data.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({"clean": row["clean"], "label": int(row["label"])})
        return prompts

    def load_induction():
        rng = np.random.default_rng(42)
        tokenizer = model.tokenizer
        vocab_size = tokenizer.vocab_size
        safe_tokens = [i for i in range(1000, min(5000, vocab_size))
                       if len(tokenizer.decode([i]).strip()) > 0]
        prompts = []
        for _ in range(500):
            prefix_len = rng.integers(3, 8)
            prefix = rng.choice(safe_tokens, size=prefix_len, replace=False).tolist()
            bigram = rng.choice(safe_tokens, size=2, replace=False).tolist()
            seq = prefix + bigram + list(rng.choice(safe_tokens, size=rng.integers(2, 5), replace=False)) + [bigram[0]]
            prompts.append({"tokens": seq, "correct_token": bigram[1]})
        return prompts

    # ========== GENERIC DISCOVERY + SWEEP ==========

    def compute_mean_z(padded_tokens, n_prompts):
        z_sums = {}
        total_pos = 0
        with torch.no_grad():
            for start in range(0, n_prompts, 64):
                batch = padded_tokens[start:start + 64]
                _, cache = model.run_with_cache(batch, names_filter=lambda n: "attn.hook_z" in n)
                for l in range(n_layers):
                    act = cache[f"blocks.{l}.attn.hook_z"]
                    if l not in z_sums:
                        z_sums[l] = torch.zeros(n_heads, d_head, device="cuda", dtype=act.dtype)
                    z_sums[l] += act.sum(dim=(0, 1))
                total_pos += batch.shape[0] * batch.shape[1]
                del cache
        return {l: z_sums[l] / total_pos for l in range(n_layers)}

    def evaluate_circuit(padded_tokens, last_positions, n_prompts,
                         active_heads, mean_z, metric_fn, ablation_type="mean"):
        active_set = set(active_heads)

        def hook_fn(act, hook, layer):
            for h in range(n_heads):
                if (layer, h) not in active_set:
                    if ablation_type == "zero":
                        act[:, :, h, :] = 0.0
                    else:
                        act[:, :, h, :] = mean_z[layer][h]
            return act

        fwd_hooks = []
        for l in range(n_layers):
            fwd_hooks.append((f"blocks.{l}.attn.hook_z",
                              lambda act, hook, layer=l: hook_fn(act, hook, layer)))

        with torch.no_grad():
            logits = model.run_with_hooks(padded_tokens, fwd_hooks=fwd_hooks, return_type="logits")
            seq_idx = torch.arange(n_prompts, device="cuda")
            last_logits = logits[seq_idx, last_positions, :]
            scores = metric_fn(last_logits)
            del logits
        return scores

    def greedy_sufficiency(padded_tokens, last_positions, n_prompts,
                           mean_z, metric_fn, K, all_heads, checkpoint_path, vol_path):
        """Forward selection with per-step checkpointing."""
        ckpt_file = f"{checkpoint_path}_greedy_ckpt.json"

        # Resume from checkpoint
        selected = []
        start_step = 0
        if os.path.exists(ckpt_file):
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            selected = [tuple(h) for h in ckpt["selected"]]
            start_step = len(selected)
            if start_step >= K:
                print(f"  [RESUME] Greedy already complete: {[f'L{l}H{h}' for l, h in selected]}")
                return sorted(selected)
            print(f"  [RESUME] Greedy from step {start_step+1}/{K}, "
                  f"have {[f'L{l}H{h}' for l, h in selected]}")

        remaining = [h for h in all_heads if h not in selected]

        for step in range(start_step, K):
            best_head = None
            best_score = -float('inf')

            for h in tqdm(remaining, desc=f"Greedy step {step+1}/{K}"):
                candidate = selected + [h]
                scores = evaluate_circuit(padded_tokens, last_positions,
                                          n_prompts, candidate, mean_z, metric_fn)
                mean_score = float(scores.mean())
                if mean_score > best_score:
                    best_score = mean_score
                    best_head = h

            selected.append(best_head)
            remaining.remove(best_head)
            print(f"  Step {step+1}: added L{best_head[0]}H{best_head[1]}, "
                  f"score={best_score:.4f}")

            # Checkpoint after every step
            with open(ckpt_file, "w") as f:
                json.dump({"selected": [list(h) for h in selected], "step": step + 1}, f)
            commit_vol(vol_path)

        return sorted(selected)

    def acdc_backward(padded_tokens, last_positions, n_prompts,
                      mean_z, metric_fn, K, starting_heads, checkpoint_path, vol_path):
        """Backward elimination with per-step checkpointing."""
        ckpt_file = f"{checkpoint_path}_acdc_ckpt.json"

        # Resume from checkpoint
        if os.path.exists(ckpt_file):
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            current = [tuple(h) for h in ckpt["current"]]
            if len(current) <= K:
                print(f"  [RESUME] ACDC already complete: {[f'L{l}H{h}' for l, h in current]}")
                return sorted(current)
            print(f"  [RESUME] ACDC from {len(current)} heads (target {K})")
        else:
            current = list(starting_heads)

        while len(current) > K:
            best_head_to_remove = None
            best_score = -float('inf')

            for h in tqdm(current, desc=f"ACDC prune {len(current)}->{len(current)-1}"):
                candidate = [x for x in current if x != h]
                scores = evaluate_circuit(padded_tokens, last_positions,
                                          n_prompts, candidate, mean_z, metric_fn)
                mean_score = float(scores.mean())
                if mean_score > best_score:
                    best_score = mean_score
                    best_head_to_remove = h

            current.remove(best_head_to_remove)
            print(f"  Removed L{best_head_to_remove[0]}H{best_head_to_remove[1]}, "
                  f"{len(current)} remaining, score={best_score:.4f}")

            # Checkpoint after every prune step
            with open(ckpt_file, "w") as f:
                json.dump({"current": [list(h) for h in current]}, f)
            commit_vol(vol_path)

        return sorted(current)

    def run_coalition_sweep(padded_tokens, last_positions, n_prompts,
                            circuit_heads, mean_z, metric_fn, ablation_type, max_len):
        n_players = len(circuit_heads)
        n_total = 2 ** n_players
        N_COAL_BATCH = min(8, n_total)
        involved_layers = sorted(set(l for l, h in circuit_heads))

        max_batch_total = N_COAL_BATCH * n_prompts
        layer_mask_tensors = {}
        for l in involved_layers:
            layer_mask_tensors[l] = torch.ones(max_batch_total, 1, n_heads, 1, device="cuda")

        mean_z_expanded = {}
        for l in involved_layers:
            mean_z_expanded[l] = mean_z[l].view(1, 1, n_heads, d_head)

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
                mask_bits = np.array([(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)
                row_start = c_pos * n_prompts
                row_end = row_start + n_prompts
                for i, (layer, head) in enumerate(circuit_heads):
                    if not mask_bits[i]:
                        layer_mask_tensors[layer][row_start:row_end, 0, head, 0] = 0.0

        score_values = np.zeros((n_total, n_prompts), dtype=np.float64)

        def forward_last_logits_only(tokens, tiled_last_pos):
            residual = model.hook_embed(model.embed(tokens))
            pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
            residual = residual + pos_embed
            for block in model.blocks:
                residual = block(residual)
            seq_idx = torch.arange(tokens.shape[0], device="cuda")
            last_resid = residual[seq_idx, tiled_last_pos, :]
            del residual
            last_normed = model.ln_final(last_resid)
            del last_resid
            logits = last_normed @ model.W_U + model.b_U
            del last_normed
            return logits

        for mb_start in tqdm(range(0, n_total, N_COAL_BATCH), desc="sweep"):
            mb_end = min(mb_start + N_COAL_BATCH, n_total)
            coal_indices = list(range(mb_start, mb_end))
            n_coal = len(coal_indices)
            build_masks(coal_indices)
            tiled = padded_tokens.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)
            tiled_last = last_positions.repeat(n_coal)
            with torch.no_grad():
                last_logits = forward_last_logits_only(tiled, tiled_last)
                last_logits = last_logits.view(n_coal, n_prompts, -1)
                scores = metric_fn(last_logits)
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
            for i, ci in enumerate(coal_indices):
                score_values[ci] = scores[i].cpu().numpy()

        model.reset_hooks(including_permanent=True)
        return score_values

    def walsh_analysis(score_values, n_players, n_prompts, circuit_heads):
        mean_values = score_values.mean(axis=1)
        n_total = 2 ** n_players
        full_idx = n_total - 1

        w = wht(mean_values)
        w_norm = w / n_total
        pc = _popcount_array(n_players)
        w2 = w_norm ** 2
        total_e = w2.sum()
        nc_e = total_e - w2[0]

        energy = {}
        if nc_e > 0:
            for order in range(1, n_players + 1):
                energy[order] = float(w2[pc == order].sum() / nc_e)

        v_full = mean_values[full_idx]
        v_empty = mean_values[0]
        faith = v_full - v_empty

        loo_sum = 0.0
        for j in range(n_players):
            without_j = full_idx & ~(1 << j)
            loo_sum += v_full - mean_values[without_j]

        if abs(faith) > 1e-10:
            epi = 1.0 - loo_sum / faith
        else:
            epi = 0.0

        return {
            "faithfulness": float(faith),
            "order1_frac": energy.get(1, 0.0),
            "order2_frac": energy.get(2, 0.0),
            "order3plus_frac": sum(energy.get(o, 0.0) for o in range(3, n_players + 1)),
            "epistasis": float(epi),
            "loo_sum": float(loo_sum),
        }

    # ========== RUN EACH TASK ==========

    def run_task(task_name, prefix, prompts_raw, metric_fn_factory, known_circuit, K, vol_path):
        print(f"\n{'='*70}")
        print(f"  {task_name.upper()}: Discovering greedy sufficiency + ACDC circuits")
        print(f"{'='*70}")

        # Check if task already fully complete
        analysis_path = f"{vol_path}/{prefix}_discovery_analysis.json"
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                existing = json.load(f)
            needed_keys = {"greedy_sufficiency_zero", "greedy_sufficiency_mean",
                           "acdc_zero", "acdc_mean"}
            if needed_keys.issubset(set(existing.keys())):
                print(f"  [SKIP] {task_name} already complete!")
                return existing

        # Tokenize
        tokenizer = model.tokenizer
        if "tokens" in prompts_raw[0]:
            token_lists = [p["tokens"] for p in prompts_raw]
        else:
            token_lists = [tokenizer.encode(p["clean"]) for p in prompts_raw]

        n_prompts = len(token_lists)
        max_len = max(len(t) for t in token_lists)
        padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
        last_positions = torch.zeros(n_prompts, dtype=torch.long, device="cuda")
        for i, toks in enumerate(token_lists):
            padded_tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            last_positions[i] = len(toks) - 1

        metric_fn = metric_fn_factory(prompts_raw, padded_tokens, last_positions, n_prompts)

        # Mean z
        print(f"[{ts()}] Computing mean z...")
        mean_z = compute_mean_z(padded_tokens, n_prompts)
        torch.cuda.empty_cache()

        # Get top-50 heads by attribution for ACDC starting set
        attr_path = f"{vol_path}/{prefix}_head_attribution.json"
        with open(attr_path) as f:
            attr_data = json.load(f)
        print(f"[{ts()}] Loaded attribution from {attr_path}")

        mean_imp = attr_data["mean_importance"]
        sorted_heads = sorted(mean_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        top50_heads = []
        for label, _ in sorted_heads[:50]:
            l = int(label.split("H")[0][1:])
            h = int(label.split("H")[1])
            top50_heads.append((l, h))

        checkpoint_path = f"{vol_path}/{prefix}"

        # Greedy sufficiency (with checkpointing)
        print(f"\n[{ts()}] Running greedy sufficiency (K={K})...")
        greedy_circuit = greedy_sufficiency(
            padded_tokens, last_positions, n_prompts,
            mean_z, metric_fn, K, ALL_HEADS, checkpoint_path, vol_path,
        )
        print(f"[{ts()}] Greedy circuit: {[f'L{l}H{h}' for l, h in greedy_circuit]}")

        # ACDC backward elimination (with checkpointing)
        print(f"\n[{ts()}] Running ACDC backward elimination (50 -> {K})...")
        acdc_circuit = acdc_backward(
            padded_tokens, last_positions, n_prompts,
            mean_z, metric_fn, K, top50_heads, checkpoint_path, vol_path,
        )
        print(f"[{ts()}] ACDC circuit: {[f'L{l}H{h}' for l, h in acdc_circuit]}")

        # Save discovery results
        discovery = {
            "greedy_circuit": [list(h) for h in greedy_circuit],
            "acdc_circuit": [list(h) for h in acdc_circuit],
            "known_circuit": [list(h) for h in known_circuit],
            "acdc_starting_set_size": 50,
        }
        with open(f"{vol_path}/{prefix}_circuit_discovery.json", "w") as f:
            json.dump(discovery, f, indent=2)
        commit_vol(vol_path)

        # Coalition sweeps on both new circuits
        new_circuits = {
            "greedy_sufficiency": greedy_circuit,
            "acdc": acdc_circuit,
        }

        results = {}
        # Load existing partial results if any
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                results = json.load(f)

        for circuit_name, circuit_heads in new_circuits.items():
            for ablation_type in ["zero", "mean"]:
                key = f"{circuit_name}_{ablation_type}"

                # Skip if already done
                npz_name = f"{prefix}_{key}_coalition_values.npz"
                npz_path = f"{vol_path}/{npz_name}"
                if os.path.exists(npz_path) and key in results:
                    print(f"\n[{ts()}] [SKIP] {key} already complete")
                    continue

                print(f"\n[{ts()}] Sweep: {key} ({2**len(circuit_heads)} coalitions)")

                scores = run_coalition_sweep(
                    padded_tokens, last_positions, n_prompts,
                    circuit_heads, mean_z, metric_fn, ablation_type, max_len,
                )

                # Save NPZ
                np.savez(npz_path,
                         scores=scores,
                         circuit_heads=np.array(circuit_heads),
                         n_players=len(circuit_heads),
                         n_prompts=n_prompts)

                # Walsh analysis
                w = walsh_analysis(scores, len(circuit_heads), n_prompts, circuit_heads)
                results[key] = w
                print(f"  faith={w['faithfulness']:+.4f}  o1={w['order1_frac']:.1%}  "
                      f"o2={w['order2_frac']:.1%}  o3+={w['order3plus_frac']:.1%}  "
                      f"epi={w['epistasis']:.3f}")

                # Save analysis after each sweep
                with open(analysis_path, "w") as f:
                    json.dump(results, f, indent=2)
                commit_vol(vol_path)

        print(f"\n[{ts()}] {task_name} complete!")
        torch.cuda.empty_cache()
        gc.collect()
        return results

    # ========== METRIC FACTORIES ==========

    def gt_metric_factory(prompts, padded_tokens, last_positions, n_prompts):
        tokenizer = model.tokenizer
        year_ids = [tokenizer.encode(f" {y:02d}")[0] for y in range(100)]
        year_ids_t = torch.tensor(year_ids, device="cuda")
        years = []
        for p in prompts:
            y = p.get("year", p.get("correct_idx", p.get("label", "50")))
            years.append(int(y))
        years_arr = np.array(years)

        def metric_fn(logits):
            if logits.dim() == 2:
                probs = torch.softmax(logits[:, year_ids_t], dim=-1)
                result = torch.zeros(logits.shape[0], device="cuda")
                for i in range(logits.shape[0]):
                    yy = years_arr[i]
                    above = probs[i, yy+1:].sum()
                    below = probs[i, :yy+1].sum()
                    result[i] = above - below
                return result
            else:
                n_coal = logits.shape[0]
                probs = torch.softmax(logits[:, :, year_ids_t], dim=-1)
                result = torch.zeros(n_coal, n_prompts, device="cuda")
                for i in range(n_prompts):
                    yy = years_arr[i]
                    above = probs[:, i, yy+1:].sum(dim=-1)
                    below = probs[:, i, :yy+1].sum(dim=-1)
                    result[:, i] = above - below
                return result
        return metric_fn

    def induction_metric_factory(prompts, padded_tokens, last_positions, n_prompts):
        correct_ids = torch.tensor([p["correct_token"] for p in prompts], device="cuda")

        def metric_fn(logits):
            if logits.dim() == 2:
                log_probs = torch.log_softmax(logits, dim=-1)
                return log_probs[torch.arange(n_prompts, device="cuda"), correct_ids]
            else:
                log_probs = torch.log_softmax(logits, dim=-1)
                tgt = correct_ids.unsqueeze(0).expand(logits.shape[0], -1)
                return log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        return metric_fn

    def gender_metric_factory(prompts, padded_tokens, last_positions, n_prompts):
        she_id = model.tokenizer.encode(" she")[0]
        he_id = model.tokenizer.encode(" he")[0]
        labels = torch.tensor([p["label"] for p in prompts], dtype=torch.float32, device="cuda")
        signs = 2 * labels - 1

        def metric_fn(logits):
            if logits.dim() == 2:
                diff = logits[:, she_id] - logits[:, he_id]
                return diff * signs
            else:
                diff = logits[:, :, she_id] - logits[:, :, he_id]
                return diff * signs.unsqueeze(0)
        return metric_fn

    def sva_metric_factory(prompts, padded_tokens, last_positions, n_prompts):
        tokenizer = model.tokenizer
        singular_ids = []
        plural_ids = []
        with open("/app/sva_verb_list.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_toks = tokenizer.encode(" " + row["sing"])
                p_toks = tokenizer.encode(" " + row["plur"])
                if len(s_toks) == 1 and len(p_toks) == 1:
                    singular_ids.append(s_toks[0])
                    plural_ids.append(p_toks[0])
        singular_ids = torch.tensor(singular_ids, device="cuda")
        plural_ids = torch.tensor(plural_ids, device="cuda")
        signs = torch.tensor([1.0 if p["plural"] else -1.0 for p in prompts], device="cuda")

        def metric_fn(logits):
            if logits.dim() == 2:
                flat = logits
            else:
                orig_shape = logits.shape[:-1]
                flat = logits.reshape(-1, logits.shape[-1])

            s_logits = flat[:, singular_ids]
            p_logits = flat[:, plural_ids]
            all_logits = torch.cat([s_logits, p_logits], dim=-1)
            all_probs = torch.softmax(all_logits, dim=-1)
            nv = len(singular_ids)
            s_prob = all_probs[:, :nv].sum(dim=-1)
            p_prob = all_probs[:, nv:].sum(dim=-1)
            diff = (p_prob - s_prob)

            if logits.dim() == 2:
                return diff * signs
            else:
                diff = diff.reshape(orig_shape)
                if diff.dim() == 2:
                    return diff * signs.unsqueeze(0)
                return diff
        return metric_fn

    # ========== CIRCUITS ==========

    GT_KNOWN = [(5, 1), (5, 5), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)]
    IND_KNOWN = [(2, 2), (4, 11), (5, 1), (5, 5), (6, 9), (7, 2), (7, 10)]
    GENDER_KNOWN = [(0, 10), (3, 0), (5, 8), (6, 6), (8, 6)]
    SVA_KNOWN = [(0, 4), (0, 8), (1, 0), (1, 1), (2, 1), (2, 6),
                 (6, 0), (9, 4), (10, 0), (11, 4), (11, 6), (11, 7)]

    TASK_CONFIGS = [
        ("greater_than", "gt", load_greater_than, gt_metric_factory, GT_KNOWN, "/gt"),
        ("induction", "induction", load_induction, induction_metric_factory, IND_KNOWN, "/ind"),
        ("gender_bias", "gender", load_gender, gender_metric_factory, GENDER_KNOWN, "/gender"),
        ("sva", "sva", load_sva, sva_metric_factory, SVA_KNOWN, "/sva"),
    ]

    # ========== MAIN ==========

    all_task_results = {}

    for task_name, prefix, loader, metric_factory, known_circuit, vol_path in TASK_CONFIGS:
        prompts = loader()
        r = run_task(task_name, prefix, prompts, metric_factory,
                     known_circuit, len(known_circuit), vol_path)
        all_task_results[task_name] = r

    # ========== FINAL SUMMARY ==========
    print(f"\n\n{'='*90}")
    print("DISCOVERY SUMMARY (mean ablation)")
    print(f"{'='*90}")
    print(f"{'Task':<16} {'Circuit':<22} {'Faith':>8} {'Ord-1':>7} {'Ord-2':>7} {'Epi':>7}")
    print("-" * 90)
    for task_name, task_r in all_task_results.items():
        for key in ["greedy_sufficiency_mean", "acdc_mean"]:
            if key in task_r:
                r = task_r[key]
                print(f"{task_name:<16} {key:<22} {r['faithfulness']:>+8.4f} "
                      f"{r['order1_frac']:>6.1%} {r['order2_frac']:>6.1%} "
                      f"{r['epistasis']:>7.3f}")

    print(f"\n[{ts()}] All tasks complete!")


@app.local_entrypoint()
def main():
    print("Launching circuit discovery for all tasks")
    run_discovery.remote()
