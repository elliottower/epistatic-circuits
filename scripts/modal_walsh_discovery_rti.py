"""Modal: Walsh discovery over all 144 GPT-2 heads for RTI task.

Samples 20K random coalitions, evaluates logit_diff (correct - distractor)
under mean ablation. LASSO fitting done locally after download.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_walsh_discovery_rti.py
"""

import modal

app = modal.App("walsh-discovery-rti")

results_volume = modal.Volume.from_name("walsh-discovery-rti", create_if_missing=True)

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
    .add_local_file("src/rti_prompts.py", remote_path="/app/rti_prompts.py")
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
    import gc
    import os
    import sys
    import time

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    sys.path.insert(0, "/app")
    from rti_prompts import make_prompts

    def ts():
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    checkpoint_path = "/results/walsh_rti_144heads_mean_coalitions.npz"
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

    tokenizer = model.tokenizer
    raw_prompts = make_prompts(tokenizer)
    valid = [p for p in raw_prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    print(f"[{ts()}] {len(valid)} valid RTI prompts (from {len(raw_prompts)} total)")

    token_lists = []
    correct_ids = []
    distractor_ids = []
    for p in valid:
        toks = tokenizer.encode(p["text"])
        token_lists.append(toks)
        correct_ids.append(p["correct_id"])
        distractor_ids.append(p["distractor_id"])

    n_prompts = len(token_lists)
    max_len = max(len(t) for t in token_lists)
    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    last_positions = torch.zeros(n_prompts, dtype=torch.long, device="cuda")
    for i, toks in enumerate(token_lists):
        padded_tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        last_positions[i] = len(toks) - 1

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    distractor_ids_t = torch.tensor(distractor_ids, device="cuda")

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

        all_target = []
        all_foil = []
        with torch.no_grad():
            for start in range(0, n_prompts, 200):
                batch = padded_tokens[start:start + 200]
                batch_last = last_positions[start:start + 200]
                batch_n = batch.shape[0]
                batch_correct = correct_ids_t[start:start + batch_n]
                batch_distractor = distractor_ids_t[start:start + batch_n]

                logits = model.run_with_hooks(batch, fwd_hooks=fwd_hooks, return_type="logits")
                seq_idx = torch.arange(batch_n, device="cuda")
                last_logits = logits[seq_idx, batch_last, :]
                del logits

                target = last_logits[seq_idx, batch_correct].cpu().numpy()
                foil = last_logits[seq_idx, batch_distractor].cpu().numpy()
                all_target.append(target)
                all_foil.append(foil)
                del last_logits

        return np.concatenate(all_target), np.concatenate(all_foil)

    rng = np.random.default_rng(seed=2024)
    random_masks = rng.integers(0, 2, size=(N_SAMPLES, N_PLAYERS)).astype(bool)
    all_zeros = np.zeros((1, N_PLAYERS), dtype=bool)
    all_ones = np.ones((1, N_PLAYERS), dtype=bool)
    masks = np.vstack([all_zeros, all_ones, random_masks])
    total = len(masks)

    all_target_logits = np.full((total, n_prompts), np.nan, dtype=np.float64)
    all_foil_logits = np.full((total, n_prompts), np.nan, dtype=np.float64)

    if start_idx > 0:
        ckpt = np.load(checkpoint_path)
        all_target_logits[:start_idx] = ckpt["target_logits"][:start_idx]
        all_foil_logits[:start_idx] = ckpt["foil_logits"][:start_idx]

    print(f"[{ts()}] Sampling {total} coalitions (from idx {start_idx})...")
    sweep_start = time.time()
    checkpoint_every = 500

    for idx in tqdm(range(start_idx, total), desc="walsh-rti"):
        target, foil = evaluate_coalition(masks[idx])
        all_target_logits[idx] = target
        all_foil_logits[idx] = foil

        if (idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - sweep_start
            done = idx + 1 - start_idx
            rate = done / elapsed
            remaining = (total - idx - 1) / rate
            print(f"[{ts()}] {idx + 1}/{total}, {rate:.2f} coal/s, ~{remaining / 3600:.1f}h left")
            np.savez(
                checkpoint_path,
                masks=masks[:idx + 1],
                target_logits=all_target_logits[:idx + 1],
                foil_logits=all_foil_logits[:idx + 1],
                circuit_heads=np.array(ALL_HEADS),
                n_players=N_PLAYERS,
                n_prompts=n_prompts,
                n_completed=idx + 1,
                n_total=total,
                seed=2024,
            )
            results_volume.commit()

    elapsed_total = time.time() - sweep_start
    logit_diffs = all_target_logits - all_foil_logits
    mean_ld = logit_diffs.mean(axis=1)
    print(f"[{ts()}] Done in {elapsed_total / 3600:.1f}h")
    print(f"  All-ablated logit_diff: {mean_ld[0]:.4f}")
    print(f"  Clean logit_diff:       {mean_ld[1]:.4f}")
    print(f"  Random mean:            {mean_ld[2:].mean():.4f} (std {mean_ld[2:].std():.4f})")

    np.savez(
        checkpoint_path,
        masks=masks,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        mean_logit_diffs=mean_ld,
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
    print("Launching Walsh discovery for RTI task")
    print(f"  {N_PLAYERS} heads, {N_SAMPLES} random samples + 2 calibration")
    run_walsh_sampling.remote()
