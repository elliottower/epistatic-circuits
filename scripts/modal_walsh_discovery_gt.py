"""Modal: Walsh discovery over all 144 GPT-2 heads for greater-than task.

Samples 20K random coalitions, evaluates prob_diff under mean ablation.
LASSO fitting done locally after download.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_walsh_discovery_gt.py
"""

import modal

app = modal.App("walsh-discovery-greater-than")

results_volume = modal.Volume.from_name("walsh-discovery-gt", create_if_missing=True)

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
)

ALL_HEADS = [(layer, head) for layer in range(12) for head in range(12)]
N_PLAYERS = 144
N_SAMPLES = 20_000


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_walsh_sampling():
    import csv
    import gc
    import os
    import time

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    def ts():
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    checkpoint_path = "/results/walsh_gt_144heads_mean_coalitions.npz"
    start_idx = 0
    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        if "n_completed" in ckpt:
            start_idx = int(ckpt["n_completed"])
            print(f"[{ts()}] Resuming from coalition {start_idx}")

    print(f"[{ts()}] Loading model...")
    model = HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"[{ts()}] Model loaded: {n_layers}L, {n_heads}H")

    prompts = []
    with open("/app/greater_than_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({"clean": row["clean"], "year": int(row.get("label", "50"))})
    prompts = prompts[:1000]
    n_prompts = len(prompts)
    print(f"[{ts()}] {n_prompts} prompts loaded")

    tokenizer = model.tokenizer
    token_lists = [tokenizer.encode(p["clean"]) for p in prompts]
    max_len = max(len(t) for t in token_lists)
    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    last_positions = torch.zeros(n_prompts, dtype=torch.long, device="cuda")
    for i, toks in enumerate(token_lists):
        padded_tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        last_positions[i] = len(toks) - 1

    year_ids = [tokenizer.encode(f" {y:02d}")[0] for y in range(100)]
    year_ids_t = torch.tensor(year_ids, device="cuda")
    years_arr = np.array([p["year"] for p in prompts])

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
                    z_sums[l] = torch.zeros(n_heads, d_head, device="cuda", dtype=act.dtype)
                z_sums[l] += act.sum(dim=(0, 1))
            total_pos += batch.shape[0] * batch.shape[1]
            del cache
    mean_z = {l: z_sums[l] / total_pos for l in range(n_layers)}
    torch.cuda.empty_cache()
    gc.collect()

    def forward_last_logits(tokens, tiled_last):
        residual = model.hook_embed(model.embed(tokens))
        pos_embed = model.hook_pos_embed(model.pos_embed(tokens))
        residual = residual + pos_embed
        for block in model.blocks:
            residual = block(residual)
        seq_idx = torch.arange(tokens.shape[0], device="cuda")
        last_resid = residual[seq_idx, tiled_last, :]
        del residual
        last_normed = model.ln_final(last_resid)
        del last_resid
        logits = last_normed @ model.W_U + model.b_U
        del last_normed
        return logits

    def evaluate_coalition(mask):
        active_set = set()
        for i, (l, h) in enumerate(ALL_HEADS):
            if mask[i]:
                active_set.add((l, h))

        def hook_fn(act, hook, layer):
            for h in range(n_heads):
                if (layer, h) not in active_set:
                    act[:, :, h, :] = mean_z[layer][h]
            return act

        fwd_hooks = []
        for l in range(n_layers):
            fwd_hooks.append((f"blocks.{l}.attn.hook_z",
                              lambda act, hook, layer=l: hook_fn(act, hook, layer)))

        all_scores = []
        with torch.no_grad():
            for start in range(0, n_prompts, 200):
                batch = padded_tokens[start:start + 200]
                batch_last = last_positions[start:start + 200]
                batch_n = batch.shape[0]
                batch_years = years_arr[start:start + batch_n]

                logits = model.run_with_hooks(batch, fwd_hooks=fwd_hooks, return_type="logits")
                seq_idx = torch.arange(batch_n, device="cuda")
                last_logits = logits[seq_idx, batch_last, :]
                del logits

                probs = torch.softmax(last_logits[:, year_ids_t], dim=-1)
                scores = torch.zeros(batch_n, device="cuda")
                for i in range(batch_n):
                    yy = batch_years[i]
                    scores[i] = probs[i, yy + 1:].sum() - probs[i, :yy + 1].sum()
                all_scores.append(scores.cpu().numpy())
                del last_logits

        return np.concatenate(all_scores)

    rng = np.random.default_rng(seed=2024)
    random_masks = rng.integers(0, 2, size=(N_SAMPLES, N_PLAYERS)).astype(bool)
    all_zeros = np.zeros((1, N_PLAYERS), dtype=bool)
    all_ones = np.ones((1, N_PLAYERS), dtype=bool)
    masks = np.vstack([all_zeros, all_ones, random_masks])
    total = len(masks)

    all_scores = np.full((total, n_prompts), np.nan, dtype=np.float64)

    if start_idx > 0:
        ckpt = np.load(checkpoint_path)
        all_scores[:start_idx] = ckpt["scores"][:start_idx]

    print(f"[{ts()}] Sampling {total} coalitions (from idx {start_idx})...")
    sweep_start = time.time()
    checkpoint_every = 500

    for idx in tqdm(range(start_idx, total), desc="walsh-gt"):
        scores = evaluate_coalition(masks[idx])
        all_scores[idx] = scores

        if (idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - sweep_start
            done = idx + 1 - start_idx
            rate = done / elapsed
            remaining = (total - idx - 1) / rate
            print(f"[{ts()}] {idx + 1}/{total}, {rate:.2f} coal/s, ~{remaining / 3600:.1f}h left")
            np.savez(
                checkpoint_path,
                masks=masks[:idx + 1],
                scores=all_scores[:idx + 1],
                circuit_heads=np.array(ALL_HEADS),
                n_players=N_PLAYERS,
                n_prompts=n_prompts,
                n_completed=idx + 1,
                n_total=total,
                seed=2024,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    mean_scores = all_scores.mean(axis=1)
    print(f"[{ts()}] Done in {elapsed_total / 3600:.1f}h ({total / elapsed_total:.2f} coal/s)")
    print(f"  All-ablated prob_diff: {mean_scores[0]:.4f}")
    print(f"  Clean prob_diff:       {mean_scores[1]:.4f}")
    print(f"  Random mean:           {mean_scores[2:].mean():.4f} (std {mean_scores[2:].std():.4f})")

    np.savez(
        checkpoint_path,
        masks=masks,
        scores=all_scores,
        mean_scores=mean_scores,
        circuit_heads=np.array(ALL_HEADS),
        n_players=N_PLAYERS,
        n_prompts=n_prompts,
        n_completed=total,
        n_total=total,
        seed=2024,
        elapsed_seconds=elapsed_total,
    )
    results_volume.commit()
    print(f"[{ts()}] Final save committed")


@app.local_entrypoint()
def main():
    print("Launching Walsh discovery for greater-than task")
    print(f"  {N_PLAYERS} heads, {N_SAMPLES} random samples + 2 calibration")
    run_walsh_sampling.remote()
