"""Modal: fill the one missing GT sweep — greedy_sufficiency, resample ablation.

GT greedy_sufficiency circuit (7 heads): 2^7 = 128 coalitions.
Resample ablation: heads outside the active coalition get corrupted
activations; heads inside keep clean.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_gt_greedy_resample_sweep.py
"""

import modal

app = modal.App("gt-greedy-resample-sweep")

results_volume = modal.Volume.from_name("gt-resample-sweep", create_if_missing=True)

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
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
    .add_local_file("data/greater_than_data.csv", remote_path="/app/greater_than_data.csv")
)

GREEDY_SUFFICIENCY_CIRCUIT = [
    (0, 10), (5, 5), (5, 8), (6, 9), (7, 10), (8, 5), (9, 1),
]

N_COAL_BATCH = 16
CHECKPOINT_EVERY = 128


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

    circuit_heads = GREEDY_SUFFICIENCY_CIRCUIT
    circuit_name = "greedy_sufficiency"
    n_players = len(circuit_heads)
    n_total = 2 ** n_players
    head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
    involved_layers = sorted(set(l for l, h in circuit_heads))

    out_path = "/results/gt_greedy_sufficiency_resample_coalition_values.npz"

    start_idx = 0
    existing_values = None
    if os.path.exists(out_path):
        ckpt = np.load(out_path)
        if "n_completed" in ckpt:
            n_done = int(ckpt["n_completed"])
            if n_done == n_total:
                print(f"[{ts()}] Already complete, skipping")
                return {"status": "already_complete"}
            start_idx = n_done
            existing_values = ckpt["prob_diff"]
            print(f"[{ts()}] Resuming from coalition {start_idx}")

    print(f"[{ts()}] GT {circuit_name} resample: {head_labels}")
    print(f"[{ts()}] {n_players} heads, {n_total} coalitions")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    # Year token IDs
    year_token_ids = []
    for year in range(100):
        toks = model.tokenizer(f"{year:02d}").input_ids
        assert len(toks) == 1
        year_token_ids.append(toks[0])
    year_token_ids_t = torch.tensor(year_token_ids, device="cuda")

    # Load paired prompts
    prompts = []
    with open("/app/greater_than_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                "clean": row["clean"],
                "corrupted": row["corrupted"],
                "year_yy": int(row.get("correct_idx", row.get("label"))),
            })
    n_prompts = len(prompts)
    print(f"[{ts()}] {n_prompts} prompt pairs")

    # Tokenize
    clean_token_lists = [model.to_tokens(p["clean"], prepend_bos=True) for p in prompts]
    corrupted_token_lists = [model.to_tokens(p["corrupted"], prepend_bos=True) for p in prompts]
    max_len = max(
        max(t.shape[1] for t in clean_token_lists),
        max(t.shape[1] for t in corrupted_token_lists),
    )
    seq_lens = np.array([t.shape[1] for t in clean_token_lists], dtype=np.int64)

    padded_clean = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    padded_corrupted = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    for i in range(n_prompts):
        cl = clean_token_lists[i].shape[1]
        padded_clean[i, :cl] = clean_token_lists[i][0]
        xl = corrupted_token_lists[i].shape[1]
        padded_corrupted[i, :xl] = corrupted_token_lists[i][0]
    del clean_token_lists, corrupted_token_lists

    last_positions = torch.tensor(seq_lens - 1, device="cuda")
    year_yys = [p["year_yy"] for p in prompts]

    def compute_prob_diff(logits_at_last):
        if logits_at_last.dim() == 2:
            year_logits = logits_at_last[:, year_token_ids_t]
            year_probs = torch.softmax(year_logits, dim=-1)
            prob_diffs = torch.zeros(logits_at_last.shape[0], device="cuda")
            for i in range(logits_at_last.shape[0]):
                yy = year_yys[i]
                prob_diffs[i] = year_probs[i, yy + 1:].sum() - year_probs[i, :yy + 1].sum()
            return prob_diffs
        else:
            n_coal = logits_at_last.shape[0]
            year_logits = logits_at_last[:, :, year_token_ids_t]
            year_probs = torch.softmax(year_logits, dim=-1)
            prob_diffs = torch.zeros(n_coal, n_prompts, device="cuda")
            for i in range(n_prompts):
                yy = year_yys[i]
                prob_diffs[:, i] = (year_probs[:, i, yy + 1:].sum(dim=-1) -
                                    year_probs[:, i, :yy + 1].sum(dim=-1))
            return prob_diffs

    # Cache corrupted hook_z
    print(f"[{ts()}] Caching corrupted hook_z for layers {involved_layers}...")
    corrupted_z = {}
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_corrupted[start:start + 64]
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: "attn.hook_z" in n)
            for l in involved_layers:
                act = cache[f"blocks.{l}.attn.hook_z"]
                if l not in corrupted_z:
                    corrupted_z[l] = torch.zeros(
                        n_prompts, max_len, n_model_heads, d_head,
                        device="cuda", dtype=act.dtype)
                corrupted_z[l][start:start + act.shape[0], :act.shape[1]] = act
            del cache
    del padded_corrupted
    torch.cuda.empty_cache()
    gc.collect()

    # Resample hooks
    max_batch_total = N_COAL_BATCH * n_prompts
    layer_mask_tensors = {}
    for l in involved_layers:
        layer_mask_tensors[l] = torch.ones(
            max_batch_total, 1, n_model_heads, 1, device="cuda")

    model.reset_hooks(including_permanent=True)

    def make_hook(layer):
        def hook_fn(act, hook):
            n = act.shape[0]
            mask = layer_mask_tensors[layer][:n]
            n_coal = n // n_prompts
            donor = corrupted_z[layer].repeat(n_coal, 1, 1, 1)[:n]
            return act * mask + donor * (1 - mask)
        return hook_fn

    for l in involved_layers:
        model.add_perma_hook(f"blocks.{l}.attn.hook_z", make_hook(l))

    def build_masks(coal_indices):
        n_coal = len(coal_indices)
        for l in involved_layers:
            layer_mask_tensors[l][:n_coal * n_prompts] = 1.0
        for c_pos, coal_idx in enumerate(coal_indices):
            mask_bits = np.array(
                [(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)
            row_start = c_pos * n_prompts
            row_end = row_start + n_prompts
            for i, (layer, head) in enumerate(circuit_heads):
                if not mask_bits[i]:
                    layer_mask_tensors[layer][row_start:row_end, 0, head, 0] = 0.0

    def forward_last_logits(tokens, n_coal):
        residual = model.hook_embed(model.embed(tokens))
        pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
        residual = residual + pos_embed
        for block in model.blocks:
            residual = block(residual)
        tiled_last = last_positions.repeat(n_coal)
        seq_idx = torch.arange(tokens.shape[0], device="cuda")
        last_resid = residual[seq_idx, tiled_last, :]
        del residual
        last_normed = model.ln_final(last_resid)
        del last_resid
        return last_normed @ model.W_U + model.b_U

    def evaluate_batch(coal_indices):
        n_coal = len(coal_indices)
        build_masks(coal_indices)
        tiled_tokens = padded_clean.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)
        with torch.no_grad():
            last_logits = forward_last_logits(tiled_tokens, n_coal)
        last_logits = last_logits.view(n_coal, n_prompts, -1)
        pd = compute_prob_diff(last_logits).cpu().numpy().astype(np.float64)
        del last_logits
        return pd

    pd_values = np.zeros((n_total, n_prompts), dtype=np.float64)
    if existing_values is not None:
        pd_values[:start_idx] = existing_values[:start_idx]

    t0 = time.time()
    for mb_start in tqdm(range(start_idx, n_total, N_COAL_BATCH), desc=circuit_name):
        mb_end = min(mb_start + N_COAL_BATCH, n_total)
        coal_indices = list(range(mb_start, mb_end))
        pd_batch = evaluate_batch(coal_indices)
        for i, coal_idx in enumerate(coal_indices):
            pd_values[coal_idx] = pd_batch[i]

        if (mb_end % CHECKPOINT_EVERY == 0) or (mb_end == n_total):
            np.savez(out_path,
                     prob_diff=pd_values[:mb_end],
                     circuit_heads=np.array(circuit_heads),
                     n_players=n_players,
                     n_prompts=n_prompts,
                     n_completed=mb_end,
                     n_total=n_total)
            results_volume.commit()
            elapsed = time.time() - t0
            done = mb_end - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_total - mb_end) / rate if rate > 0 else 0
            print(f"[{ts()}] {mb_end}/{n_total}, {rate:.1f} coal/s, ~{remaining/60:.0f}min left")

    elapsed_total = time.time() - t0
    print(f"[{ts()}] Done in {elapsed_total:.1f}s")

    model.reset_hooks(including_permanent=True)
    for l in list(corrupted_z.keys()):
        del corrupted_z[l]
    torch.cuda.empty_cache()
    gc.collect()

    # Walsh analysis
    print(f"\n[{ts()}] Walsh-Hadamard analysis")
    mean_pd = pd_values.mean(axis=1)
    full_mask = (1 << n_players) - 1
    intact = float(mean_pd[full_mask])
    empty = float(mean_pd[0])
    faithfulness = intact - empty

    wht_coeffs = wht(mean_pd.astype(np.float64))
    energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)
    energy_nc = energy_spectrum[1:].copy()
    total_nc = energy_nc.sum()
    energy_nc_frac = energy_nc / total_nc if total_nc > 0 else energy_nc

    head_energy = wht_per_head_energy(mean_pd.astype(np.float64), n_players)

    per_prompt_group = pd_values[full_mask] - pd_values[0]
    per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
    for i in range(n_players):
        complement = full_mask ^ (1 << i)
        per_prompt_loo_sum += pd_values[full_mask] - pd_values[complement]
    group_mean = float(np.mean(per_prompt_group))
    loo_sum_mean = float(np.mean(per_prompt_loo_sum))
    epistasis_point = 1.0 - loo_sum_mean / group_mean if abs(group_mean) > 1e-10 else 0.0

    import json
    result = {
        "circuit": circuit_name,
        "ablation": "resample",
        "task": "greater_than",
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

    with open("/results/gt_greedy_sufficiency_resample_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    print(f"\n  {circuit_name} resample: faith={faithfulness:+.4f}, "
          f"o1={result['order1_frac']*100:.1f}%, o2={result['order2_frac']*100:.1f}%, "
          f"o3+={result['order3plus_frac']*100:.1f}%")
    print(f"[{ts()}] Results on gt-resample-sweep volume.")
    return result


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching GT greedy_sufficiency resample sweep")
    result = run_sweep.remote()
    if isinstance(result, dict) and "status" in result:
        print(f"Status: {result['status']}")
    else:
        print(f"\nResult: faith={result['faithfulness']:+.4f}, "
              f"o1={result['order1_frac']*100:.1f}%, "
              f"o2={result['order2_frac']*100:.1f}%, "
              f"o3+={result['order3plus_frac']*100:.1f}%")
