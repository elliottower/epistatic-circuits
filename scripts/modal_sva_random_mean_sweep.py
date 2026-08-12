"""Modal: fill the one missing SVA sweep — random circuit, mean ablation.

SVA random circuit (12 heads, seed=42): 2^12 = 4,096 coalitions.
500 prompts. Mean ablation only (zero already exists).

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_sva_random_mean_sweep.py
"""

import modal

app = modal.App("sva-random-mean-sweep")

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
    )
    .add_local_file("data/sva_data.csv", remote_path="/app/sva_data.csv")
    .add_local_file("data/sva_verb_list.csv", remote_path="/app/sva_verb_list.csv")
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
)


def make_random_circuit(n_heads, seed=42):
    import random
    rng = random.Random(seed)
    all_heads = [(l, h) for l in range(12) for h in range(12)]
    return sorted(rng.sample(all_heads, n_heads))


N_CIRCUIT_HEADS = 12
N_PROMPTS_SUBSAMPLE = 500
N_COAL_BATCH = 8
CHECKPOINT_EVERY = 256


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_sweep():
    import csv
    import gc
    import os
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")
    from walsh import energy_by_order_normalized, wht, wht_per_head_energy

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    circuit_heads = make_random_circuit(N_CIRCUIT_HEADS)
    circuit_name = "random"
    ablation_type = "mean"
    key = f"{circuit_name}_{ablation_type}"
    n_players = len(circuit_heads)
    n_total = 2 ** n_players
    head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
    involved_layers = sorted(set(l for l, h in circuit_heads))

    out_path = f"/results/sva_{key}_coalition_values.npz"

    start_idx = 0
    existing_values = None
    if os.path.exists(out_path):
        ckpt = np.load(out_path)
        if "n_completed" in ckpt:
            n_done = int(ckpt["n_completed"])
            if n_done == n_total:
                print(f"[{ts()}] Already complete ({n_total} coalitions), skipping sweep")
                return {"status": "already_complete"}
            start_idx = n_done
            existing_values = ckpt["verb_scores"]
            print(f"[{ts()}] Resuming from coalition {start_idx}")

    print(f"[{ts()}] SVA {key}: {head_labels}")
    print(f"[{ts()}] {n_players} heads, {n_total} coalitions")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    # Verb tokens
    singular_ids, plural_ids = [], []
    with open("/app/sva_verb_list.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sing_toks = model.tokenizer.encode(" " + row["sing"])
            plur_toks = model.tokenizer.encode(" " + row["plur"])
            if len(sing_toks) == 1 and len(plur_toks) == 1:
                singular_ids.append(sing_toks[0])
                plural_ids.append(plur_toks[0])
    singular_ids = torch.tensor(singular_ids, device="cuda")
    plural_ids = torch.tensor(plural_ids, device="cuda")
    print(f"[{ts()}] {len(singular_ids)} single-token verb pairs")

    # Prompts
    all_prompts = []
    with open("/app/sva_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_prompts.append({"text": row["clean"], "plural": int(row["plural"])})
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_prompts), size=N_PROMPTS_SUBSAMPLE, replace=False)
    prompts = [all_prompts[i] for i in indices]
    n_prompts = len(prompts)
    plural_flags = np.array([p["plural"] for p in prompts])

    # Tokenize
    all_tokens = [model.to_tokens(p["text"], prepend_bos=True) for p in prompts]
    max_len = max(t.shape[1] for t in all_tokens)
    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]
    last_positions = torch.tensor(seq_lens - 1, device="cuda")
    signs = torch.tensor([1 if p else -1 for p in plural_flags],
                         device="cuda", dtype=torch.float32)

    def compute_verb_score(logits_at_last):
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

    # Mean z
    print(f"[{ts()}] Computing mean z...")
    z_sums = {}
    total_positions = 0
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_tokens[start:start + 64]
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: "attn.hook_z" in n)
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
        tiled_last = last_positions.repeat(n_coal) if n_coal > 1 else last_positions
        seq_idx = torch.arange(tokens.shape[0], device="cuda")
        last_resid = residual[seq_idx, tiled_last, :]
        del residual
        last_normed = model.ln_final(last_resid)
        del last_resid
        return last_normed @ model.W_U + model.b_U

    # Hooks for mean ablation
    max_batch_total = N_COAL_BATCH * n_prompts
    layer_mask_tensors = {}
    mean_z_expanded = {}
    for l in involved_layers:
        layer_mask_tensors[l] = torch.ones(max_batch_total, 1, n_model_heads, 1, device="cuda")
        mean_z_expanded[l] = mean_z[l].view(1, 1, n_model_heads, d_head)

    model.reset_hooks(including_permanent=True)

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
    if existing_values is not None:
        verb_scores[:start_idx] = existing_values[:start_idx]

    sweep_start = time.time()
    n_batches = (n_total - start_idx + N_COAL_BATCH - 1) // N_COAL_BATCH

    for mb_start in tqdm(range(start_idx, n_total, N_COAL_BATCH), desc=key, total=n_batches):
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

        if mb_start > 0 and mb_start % (CHECKPOINT_EVERY * N_COAL_BATCH) < N_COAL_BATCH:
            elapsed = time.time() - sweep_start
            done = mb_end - start_idx
            rate = done / elapsed
            remaining = (n_total - mb_end) / rate if rate > 0 else 0
            print(f"[{ts()}] {mb_end}/{n_total} done, {rate:.1f} coal/s, ~{remaining/60:.1f}min left")
            np.savez(out_path,
                     verb_scores=verb_scores,
                     circuit_heads=np.array(circuit_heads),
                     n_players=n_players,
                     n_prompts=n_prompts,
                     circuit_name=circuit_name,
                     ablation_type=ablation_type,
                     n_completed=mb_end)
            results_volume.commit()

    elapsed = time.time() - sweep_start
    print(f"[{ts()}] {key}: {n_total} coalitions in {elapsed:.1f}s")

    np.savez(out_path,
             verb_scores=verb_scores,
             circuit_heads=np.array(circuit_heads),
             n_players=n_players,
             n_prompts=n_prompts,
             circuit_name=circuit_name,
             ablation_type=ablation_type,
             n_completed=n_total)
    results_volume.commit()

    # Walsh analysis
    print(f"\n[{ts()}] Walsh-Hadamard analysis")
    mean_score = verb_scores.mean(axis=1)
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

    per_prompt_group = verb_scores[full_mask] - verb_scores[0]
    per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
    for i in range(n_players):
        complement = full_mask ^ (1 << i)
        per_prompt_loo_sum += verb_scores[full_mask] - verb_scores[complement]
    group_mean = float(np.mean(per_prompt_group))
    loo_sum_mean = float(np.mean(per_prompt_loo_sum))
    epistasis_point = 1.0 - loo_sum_mean / group_mean if abs(group_mean) > 1e-10 else 0.0

    import json
    result = {
        "circuit": circuit_name,
        "ablation": ablation_type,
        "task": "sva",
        "n_players": n_players,
        "n_prompts": n_prompts,
        "heads": head_labels,
        "intact_score": round(intact, 6),
        "empty_score": round(empty, 6),
        "faithfulness": round(faithfulness, 6),
        "order1_frac": round(float(energy_nc_frac[0]), 4),
        "order2_frac": round(float(energy_nc_frac[1]), 4),
        "order3plus_frac": round(float(sum(energy_nc_frac[2:])), 4),
        "per_head_energy": {head_labels[i]: round(float(head_energy[i]), 6) for i in range(n_players)},
        "epistasis_point": round(epistasis_point, 4),
        "group_effect": round(group_mean, 6),
        "loo_sum": round(loo_sum_mean, 6),
    }

    with open("/results/sva_random_mean_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    print(f"\n  {key}: faith={faithfulness:+.4f}, o1={result['order1_frac']*100:.1f}%, "
          f"o2={result['order2_frac']*100:.1f}%, o3+={result['order3plus_frac']*100:.1f}%")
    print(f"[{ts()}] Done. Results on sva-sweep-results volume.")
    return result


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching SVA random mean sweep")
    result = run_sweep.remote()
    print(f"\nResult: faith={result['faithfulness']:+.4f}, "
          f"o1={result['order1_frac']*100:.1f}%, "
          f"o2={result['order2_frac']*100:.1f}%, "
          f"o3+={result['order3plus_frac']*100:.1f}%")
