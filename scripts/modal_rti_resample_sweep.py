"""Modal: RTI coalition sweep with resample (activation patching) ablation.

MIB-style: for each clean prompt, run the corrupted version (D/C swapped),
cache activations. Heads OUTSIDE the active coalition get corrupted hook_z;
heads INSIDE keep clean activations.

Metric: logit_diff = logit(correct_name) - logit(incorrect_name).

Runs 5 circuits: known, EAP, C5_walsh, C6_epistatic, random.
Each has 15 heads -> 2^15 = 32768 coalitions.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_rti_resample_sweep.py
"""

import modal

app = modal.App("rti-resample-sweep")

results_volume = modal.Volume.from_name("rti-resample-sweep", create_if_missing=True)

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
)

# RTI circuits from rti-v5 and rti-walsh volumes (actual heads from NPZ files)
CIRCUITS = {
    "known": [
        (0, 8), (0, 9), (0, 11), (4, 11),
        (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
        (10, 11), (11, 9), (11, 11),
    ],
    "EAP": [
        (0, 9), (0, 10), (5, 9), (0, 8), (0, 6), (1, 10), (2, 10),
        (0, 11), (1, 3), (0, 1), (4, 7), (2, 0), (0, 5), (0, 3), (2, 8),
    ],
    "C5_walsh": [
        (0, 9), (11, 2), (4, 11), (10, 6), (7, 9), (10, 7), (9, 9),
        (5, 6), (4, 0), (2, 11), (8, 7), (1, 5), (11, 10), (1, 3), (6, 11),
    ],
    "C6_epistatic": [
        (0, 10), (0, 9), (11, 2), (10, 7), (10, 0), (4, 11), (11, 10),
        (0, 1), (0, 3), (9, 9), (1, 11), (9, 6), (2, 2), (5, 6), (2, 11),
    ],
    "random": [
        (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
        (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
    ],
}

TEMPLATES = [
    "Then {D} and {C} went to the store. {D} gave a drink to",
    "{D} told {C} a story. {D} then handed the book to",
    "{D} met {C} at the park. {D} passed the ball to",
    "When {D} and {C} arrived, {D} gave the keys to",
    "{D} helped {C} move. {D} gave the boxes to",
    "Then {D} and {C} stopped for lunch. {D} offered the food to",
    "When {D} visited {C} at home, {D} showed the photos to",
    "{D} and {C} were at the library. {D} returned the books to",
]

NAMES_A = ["Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate"]
NAMES_B = ["Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia"]

N_COAL_BATCH = 8
CHECKPOINT_EVERY = 2048


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_sweep(circuit_name: str):
    import gc
    import os
    import sys
    import time
    from itertools import product as cartesian_product

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    circuit_heads = CIRCUITS[circuit_name]
    n_players = len(circuit_heads)
    n_total = 2 ** n_players
    head_labels = [f"L{l}H{h}" for l, h in circuit_heads]
    involved_layers = sorted(set(l for l, h in circuit_heads))

    out_path = f"/results/rti_{circuit_name}_resample_coalition_values.npz"

    start_idx = 0
    existing_values = None
    if os.path.exists(out_path):
        ckpt = np.load(out_path)
        if "n_completed" in ckpt:
            n_done = int(ckpt["n_completed"])
            if n_done == n_total:
                print(f"[{ts()}] {circuit_name} already complete, skipping")
                return
            start_idx = n_done
            existing_values = ckpt["logit_diff"]
            print(f"[{ts()}] Resuming {circuit_name} from {start_idx}")

    print(f"[{ts()}] RTI resample sweep: {circuit_name}")
    print(f"[{ts()}] Heads: {head_labels}")
    print(f"[{ts()}] {n_players} heads, {n_total} coalitions, batch={N_COAL_BATCH}")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    tokenizer = model.tokenizer

    # Generate paired clean/corrupted prompts
    # Clean: D appears twice (repeated), C once. Answer = C.
    # Corrupted: swap D <-> C, so C is repeated and D is the answer.
    clean_texts = []
    corrupted_texts = []
    correct_ids = []
    incorrect_ids = []

    for tmpl in TEMPLATES:
        for name_d, name_c in cartesian_product(NAMES_A, NAMES_B):
            clean_text = tmpl.format(D=name_d, C=name_c)
            corrupted_text = tmpl.format(D=name_c, C=name_d)

            correct_toks = tokenizer.encode(" " + name_c, add_special_tokens=False)
            incorrect_toks = tokenizer.encode(" " + name_d, add_special_tokens=False)
            if len(correct_toks) != 1 or len(incorrect_toks) != 1:
                continue

            clean_texts.append(clean_text)
            corrupted_texts.append(corrupted_text)
            correct_ids.append(correct_toks[0])
            incorrect_ids.append(incorrect_toks[0])

    n_prompts = len(clean_texts)
    print(f"[{ts()}] {n_prompts} prompt pairs")

    # Tokenize with per-prompt length tracking
    clean_token_lists = [model.to_tokens(t, prepend_bos=True) for t in clean_texts]
    corrupted_token_lists = [model.to_tokens(t, prepend_bos=True) for t in corrupted_texts]
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

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    incorrect_ids_t = torch.tensor(incorrect_ids, device="cuda")
    last_positions = torch.tensor(seq_lens - 1, device="cuda")

    # Cache corrupted hook_z for involved layers
    print(f"[{ts()}] Caching corrupted hook_z for layers {involved_layers}...")
    corrupted_z = {}
    with torch.no_grad():
        for start in range(0, n_prompts, 64):
            batch = padded_corrupted[start:start + 64]
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: "attn.hook_z" in n,
            )
            for l in involved_layers:
                act = cache[f"blocks.{l}.attn.hook_z"]
                if l not in corrupted_z:
                    corrupted_z[l] = torch.zeros(
                        n_prompts, max_len, n_model_heads, d_head,
                        device="cuda", dtype=act.dtype,
                    )
                corrupted_z[l][start:start + act.shape[0], :act.shape[1]] = act
            del cache
    del padded_corrupted
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[{ts()}] Corrupted z cached for {len(involved_layers)} layers")

    # Set up hooks
    max_batch_total = N_COAL_BATCH * n_prompts
    layer_mask_tensors = {}
    for l in involved_layers:
        layer_mask_tensors[l] = torch.ones(
            max_batch_total, 1, n_model_heads, 1, device="cuda",
        )

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
                [(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool,
            )
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
        logits = last_normed @ model.W_U + model.b_U
        del last_normed
        return logits

    def evaluate_batch(coal_indices):
        n_coal = len(coal_indices)
        build_masks(coal_indices)
        tiled_tokens = padded_clean.unsqueeze(0).expand(n_coal, -1, -1).reshape(-1, max_len)
        with torch.no_grad():
            last_logits = forward_last_logits(tiled_tokens, n_coal)
        last_logits = last_logits.view(n_coal, n_prompts, -1)
        seq_idx = torch.arange(n_prompts, device="cuda")
        correct_logits = last_logits[:, seq_idx, correct_ids_t]
        incorrect_logits = last_logits[:, seq_idx, incorrect_ids_t]
        ld = (correct_logits - incorrect_logits).cpu().numpy().astype(np.float64)
        del last_logits
        return ld

    ld_values = np.zeros((n_total, n_prompts), dtype=np.float64)
    if existing_values is not None:
        ld_values[:start_idx] = existing_values[:start_idx]

    t0 = time.time()
    for mb_start in tqdm(range(start_idx, n_total, N_COAL_BATCH), desc=circuit_name):
        mb_end = min(mb_start + N_COAL_BATCH, n_total)
        coal_indices = list(range(mb_start, mb_end))
        ld_batch = evaluate_batch(coal_indices)
        for i, coal_idx in enumerate(coal_indices):
            ld_values[coal_idx] = ld_batch[i]

        if (mb_end % CHECKPOINT_EVERY == 0) or (mb_end == n_total):
            np.savez(
                out_path,
                logit_diff=ld_values[:mb_end],
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                n_completed=mb_end,
                n_total=n_total,
            )
            results_volume.commit()
            elapsed = time.time() - t0
            done = mb_end - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (n_total - mb_end) / rate if rate > 0 else 0
            print(f"[{ts()}] {mb_end}/{n_total}, {rate:.1f} coal/s, ~{remaining/60:.0f}min left")

    elapsed_total = time.time() - t0
    print(f"[{ts()}] {circuit_name} done in {elapsed_total/60:.1f}min")

    model.reset_hooks(including_permanent=True)
    for l in list(corrupted_z.keys()):
        del corrupted_z[l]
    torch.cuda.empty_cache()
    gc.collect()


@app.local_entrypoint()
def main():
    handles = []
    for name in CIRCUITS:
        handles.append(run_sweep.spawn(name))
    print(f"Spawned {len(handles)} RTI resample sweeps")
    for h in handles:
        h.get()
    print("All RTI resample sweeps complete")
