"""Modal: Coalition sweep for GT Walsh-discovered circuits (C5, C6).

2^7 = 128 coalitions per circuit, mean ablation, prob_diff metric.
Runs both C5 and C6 in sequence, saves NPZ + Walsh analysis JSON.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_gt_walsh_circuits_sweep.py
"""

import modal

app = modal.App("gt-walsh-circuits-sweep")

results_volume = modal.Volume.from_name("gt-sweep-results", create_if_missing=True)

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
    .add_local_file("data/greater_than_data.csv", remote_path="/app/greater_than_data.csv")
    .add_local_file("src/walsh.py", remote_path="/app/walsh.py")
)

# From LASSO fit on 20K random coalitions
GT_C5_HEADS = [(5, 5), (7, 10), (9, 1), (6, 9), (0, 10), (8, 5), (10, 2)]
GT_C6_HEADS = [(5, 5), (0, 10), (7, 10), (9, 1), (4, 11), (0, 3), (6, 9)]


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_sweeps():
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

    model = __import__("transformer_lens").HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    prompts = []
    with open("/app/greater_than_data.csv") as f:
        for row in csv.DictReader(f):
            prompts.append({"clean": row["clean"], "year_yy": int(row.get("correct_idx", row.get("label")))})
    n_prompts = len(prompts)
    print(f"[{ts()}] {n_prompts} prompts loaded")

    year_token_ids = []
    for year in range(100):
        toks = model.tokenizer(f"{year:02d}").input_ids
        year_token_ids.append(toks[0])
    year_token_ids = torch.tensor(year_token_ids, device="cuda")
    year_yys = [p["year_yy"] for p in prompts]

    all_tokens = []
    max_len = 0
    for p in prompts:
        toks = model.to_tokens(p["clean"], prepend_bos=True)
        all_tokens.append(toks)
        max_len = max(max_len, toks.shape[1])
    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]
    last_positions = torch.tensor(seq_lens - 1, device="cuda")

    def compute_prob_diff(logits_at_last):
        if logits_at_last.dim() == 2:
            year_logits = logits_at_last[:, year_token_ids]
            year_probs = torch.softmax(year_logits, dim=-1)
            pd = torch.zeros(logits_at_last.shape[0], device="cuda")
            for i in range(logits_at_last.shape[0]):
                yy = year_yys[i]
                pd[i] = year_probs[i, yy + 1:].sum() - year_probs[i, :yy + 1].sum()
            return pd
        else:
            n_coal = logits_at_last.shape[0]
            year_logits = logits_at_last[:, :, year_token_ids]
            year_probs = torch.softmax(year_logits, dim=-1)
            pd = torch.zeros(n_coal, n_prompts, device="cuda")
            for i in range(n_prompts):
                yy = year_yys[i]
                pd[:, i] = year_probs[:, i, yy + 1:].sum(dim=-1) - year_probs[:, i, :yy + 1].sum(dim=-1)
            return pd

    print(f"[{ts()}] Computing mean z...")
    z_sums = {}
    total_pos = 0
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_tokens[start:start + 64]
            _, cache = model.run_with_cache(batch, names_filter=lambda n: "attn.hook_z" in n)
            for l in range(n_layers):
                act = cache[f"blocks.{l}.attn.hook_z"]
                if l not in z_sums:
                    z_sums[l] = torch.zeros(n_model_heads, d_head, device="cuda", dtype=act.dtype)
                z_sums[l] += act.sum(dim=(0, 1))
            total_pos += batch.shape[0] * batch.shape[1]
            del cache
    mean_z = {l: z_sums[l] / total_pos for l in range(n_layers)}
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
        logits = last_normed @ model.W_U + model.b_U
        del last_normed
        return logits

    circuits = {"c5_walsh": GT_C5_HEADS, "c6_epistatic": GT_C6_HEADS}
    all_analysis = {}

    for circuit_name, circuit_heads in circuits.items():
        n_players = len(circuit_heads)
        n_total = 2 ** n_players
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
        involved_layers = sorted(set(l for l, h in circuit_heads))

        for ablation_type in ["mean", "zero"]:
            key = f"{circuit_name}_{ablation_type}"
            print(f"\n[{ts()}] Sweep: {key} ({n_total} coalitions)")

            N_COAL_BATCH = min(8, n_total)
            max_batch_total = N_COAL_BATCH * n_prompts

            layer_mask_tensors = {}
            mean_z_expanded = {}
            for l in involved_layers:
                layer_mask_tensors[l] = torch.ones(max_batch_total, 1, n_model_heads, 1, device="cuda")
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
                pd = compute_prob_diff(last_logits)
                return pd.cpu().numpy().astype(np.float64)

            pd_values = np.zeros((n_total, n_prompts), dtype=np.float64)
            t0 = time.time()
            for mb_start in tqdm(range(0, n_total, N_COAL_BATCH), desc=key):
                mb_end = min(mb_start + N_COAL_BATCH, n_total)
                coal_indices = list(range(mb_start, mb_end))
                pd_batch = evaluate_batch(coal_indices)
                for i, coal_idx in enumerate(coal_indices):
                    pd_values[coal_idx] = pd_batch[i]
            elapsed = time.time() - t0
            print(f"[{ts()}] {key}: {n_total} coalitions in {elapsed:.1f}s")

            np.savez(
                f"/results/gt_{key}_coalition_values.npz",
                prob_diff=pd_values,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                year_yys=np.array(year_yys),
            )
            results_volume.commit()

            # Walsh analysis
            mean_pd = pd_values.mean(axis=1)
            full_mask = (1 << n_players) - 1
            intact_pd = float(mean_pd[full_mask])
            empty_pd = float(mean_pd[0])
            faithfulness = intact_pd - empty_pd

            wht_coeffs = wht(mean_pd.astype(np.float64))
            energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)
            energy_nc = energy_spectrum[1:].copy()
            total_nc = energy_nc.sum()
            energy_nc_frac = energy_nc / total_nc if total_nc > 0 else energy_nc

            # Epistasis
            per_prompt_group = pd_values[full_mask] - pd_values[0]
            per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
            for i in range(n_players):
                complement = full_mask ^ (1 << i)
                per_prompt_loo_sum += pd_values[full_mask] - pd_values[complement]
            group_mean = float(np.mean(per_prompt_group))
            loo_sum_mean = float(np.mean(per_prompt_loo_sum))
            epistasis = 1.0 - loo_sum_mean / group_mean if abs(group_mean) > 1e-10 else 0.0

            print(f"  Faithfulness: {faithfulness:+.4f}")
            print(f"  Order-1: {float(energy_nc_frac[0])*100:.1f}%, Order-2: {float(energy_nc_frac[1])*100:.1f}%")
            print(f"  Epistasis: {epistasis*100:.1f}%")

            all_analysis[key] = {
                "circuit": circuit_name, "ablation": ablation_type, "task": "greater_than",
                "n_players": n_players, "n_prompts": n_prompts, "heads": head_labels,
                "faithfulness": round(faithfulness, 6),
                "order1_frac": round(float(energy_nc_frac[0]), 4),
                "order2_frac": round(float(energy_nc_frac[1]), 4),
                "order3plus_frac": round(float(sum(energy_nc_frac[2:])), 4),
                "epistasis": round(epistasis, 4),
            }

    model.reset_hooks(including_permanent=True)

    with open("/results/gt_walsh_circuits_analysis.json", "w") as f:
        json.dump(all_analysis, f, indent=2)
    results_volume.commit()
    print(f"\n[{ts()}] All results saved")


@app.local_entrypoint()
def main():
    run_sweeps.remote()
