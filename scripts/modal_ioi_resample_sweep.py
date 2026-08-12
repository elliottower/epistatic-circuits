"""Modal: IOI coalition sweep with resample (activation patching) ablation.

MIB-style: for each clean prompt, run the corrupted version (names swapped),
cache activations. For heads OUTSIDE the active coalition, replace hook_z
with the corrupted prompt's hook_z. Heads INSIDE the coalition keep clean acts.

This is interchange intervention / activation patching, the standard method
from Conmy et al. (ACDC) and Wang et al. (IOI).

Runs 5 circuits: C3_canonical, C2_eap, C5_walsh, C6_epistatic, C4_random.
Each has 15 heads → 2^15 = 32768 coalitions.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_ioi_resample_sweep.py
"""

import modal

app = modal.App("ioi-resample-sweep")

results_volume = modal.Volume.from_name("ioi-resample-sweep", create_if_missing=True)

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

# From src/circuits.py
CIRCUITS = {
    "C3_canonical": [
        (0, 1), (3, 0), (2, 2), (4, 11), (5, 5), (6, 9),
        (7, 3), (7, 9), (8, 6), (8, 10), (9, 9), (9, 6), (10, 0),
        (10, 7), (11, 10),
    ],
    "C2_eap": [
        (0, 1), (0, 10), (2, 2), (4, 11), (5, 5), (5, 8), (5, 9), (6, 1),
        (6, 9), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7), (11, 10),
    ],
    "C5_walsh": [
        (5, 5), (10, 7), (11, 1), (8, 6), (8, 10),
        (0, 9), (7, 9), (0, 3), (6, 9), (10, 1),
        (11, 2), (10, 10), (3, 0), (11, 10), (4, 0),
    ],
    "C6_epistatic": [
        (5, 5), (8, 6), (11, 10), (10, 7), (6, 9),
        (0, 10), (0, 1), (10, 0), (5, 9), (8, 10),
        (11, 2), (9, 9), (0, 9), (7, 9), (4, 0),
    ],
    "C4_random": [
        (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
        (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
    ],
}

TEMPLATES = [
    "When {S} and {IO} went to the store, {S} gave a bottle to",
    "Then {S} and {IO} had a meeting. {S} passed the document to",
    "{S} and {IO} were working together. {S} handed the report to",
    "After {S} met {IO} at lunch, {S} gave a gift to",
    "While {S} and {IO} were talking, {S} offered the keys to",
    "Then {S} and {IO} went to the park. {S} tossed the ball to",
    "{S} visited {IO} at the office. {S} delivered the package to",
    "When {S} saw {IO} at the party, {S} brought the cake to",
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

    out_path = f"/results/ioi_{circuit_name}_resample_coalition_values.npz"

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

    print(f"[{ts()}] IOI resample sweep: {circuit_name}")
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
    # Clean: S appears twice, IO once. Answer = IO.
    # Corrupted: names swapped (IO <-> S), so correct answer flips.
    clean_texts = []
    corrupted_texts = []
    correct_ids = []
    incorrect_ids = []

    for tmpl in TEMPLATES:
        for name_s, name_io in cartesian_product(NAMES_A, NAMES_B):
            clean_text = tmpl.format(S=name_s, IO=name_io)
            corrupted_text = tmpl.format(S=name_io, IO=name_s)

            correct_toks = tokenizer.encode(" " + name_io, add_special_tokens=False)
            incorrect_toks = tokenizer.encode(" " + name_s, add_special_tokens=False)
            if len(correct_toks) != 1 or len(incorrect_toks) != 1:
                continue

            clean_texts.append(clean_text)
            corrupted_texts.append(corrupted_text)
            correct_ids.append(correct_toks[0])
            incorrect_ids.append(incorrect_toks[0])

    n_prompts = len(clean_texts)
    print(f"[{ts()}] {n_prompts} prompt pairs")

    # Tokenize individually to track per-prompt sequence lengths
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

    # Cache corrupted hook_z activations for involved layers only
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

    # Set up batched sweep with persistent hooks
    max_batch_total = N_COAL_BATCH * n_prompts
    layer_mask_tensors = {}
    for l in involved_layers:
        layer_mask_tensors[l] = torch.ones(
            max_batch_total, 1, n_model_heads, 1, device="cuda",
        )

    # For resample: when mask=0 for a head, replace with corrupted activation
    # act * mask + corrupted * (1 - mask)
    # corrupted_z[l] needs to be tiled for N_COAL_BATCH coalitions
    model.reset_hooks(including_permanent=True)

    def make_hook(layer):
        def hook_fn(act, hook):
            n = act.shape[0]
            mask = layer_mask_tensors[layer][:n]
            n_coal = n // n_prompts
            # Tile corrupted activations for this batch
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
    # Free corrupted z
    for l in list(corrupted_z.keys()):
        del corrupted_z[l]
    torch.cuda.empty_cache()
    gc.collect()


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=3600,
)
def analyze_all():
    import json
    import sys

    import numpy as np

    sys.path.insert(0, "/app")
    from walsh import energy_by_order_normalized, wht

    all_analysis = {}

    for circuit_name, circuit_heads in CIRCUITS.items():
        path = f"/results/ioi_{circuit_name}_resample_coalition_values.npz"
        data = np.load(path)
        ld_values = data["logit_diff"]
        n_players = int(data["n_players"])
        n_prompts = int(data["n_prompts"])
        head_labels = [f"L{l}H{h}" for l, h in circuit_heads]

        mean_ld = ld_values.mean(axis=1)
        full_mask = (1 << n_players) - 1
        intact_ld = float(mean_ld[full_mask])
        empty_ld = float(mean_ld[0])
        faithfulness = intact_ld - empty_ld

        wht_coeffs = wht(mean_ld.astype(np.float64))
        energy_spectrum = energy_by_order_normalized(wht_coeffs, n_players)
        energy_nc = energy_spectrum[1:].copy()
        total_nc = energy_nc.sum()
        energy_nc_frac = energy_nc / total_nc if total_nc > 0 else energy_nc

        per_prompt_group = ld_values[full_mask] - ld_values[0]
        per_prompt_loo_sum = np.zeros(n_prompts, dtype=np.float64)
        for i in range(n_players):
            complement = full_mask ^ (1 << i)
            per_prompt_loo_sum += ld_values[full_mask] - ld_values[complement]
        group_mean = float(np.mean(per_prompt_group))
        loo_sum_mean = float(np.mean(per_prompt_loo_sum))
        epistasis = 1.0 - loo_sum_mean / group_mean if abs(group_mean) > 1e-10 else 0.0

        print(f"\n{circuit_name} (resample):")
        print(f"  Faithfulness: {faithfulness:+.4f}")
        print(f"  Order-1: {float(energy_nc_frac[0])*100:.1f}%, "
              f"Order-2: {float(energy_nc_frac[1])*100:.1f}%, "
              f"Order-3+: {float(sum(energy_nc_frac[2:]))*100:.1f}%")
        print(f"  Epistasis: {epistasis*100:.1f}%")

        all_analysis[circuit_name] = {
            "circuit": circuit_name,
            "ablation": "resample",
            "task": "IOI",
            "n_players": n_players,
            "n_prompts": n_prompts,
            "heads": head_labels,
            "faithfulness": round(faithfulness, 6),
            "intact_logit_diff": round(intact_ld, 6),
            "empty_logit_diff": round(empty_ld, 6),
            "order1_frac": round(float(energy_nc_frac[0]), 4),
            "order2_frac": round(float(energy_nc_frac[1]), 4),
            "order3plus_frac": round(float(sum(energy_nc_frac[2:])), 4),
            "epistasis": round(epistasis, 4),
        }

    with open("/results/ioi_resample_analysis.json", "w") as f:
        json.dump(all_analysis, f, indent=2)
    results_volume.commit()
    print("\nAnalysis saved")


@app.local_entrypoint()
def main():
    # Launch all 5 circuits in parallel (independent GPU containers)
    handles = []
    for circuit_name in CIRCUITS:
        print(f"Spawning: {circuit_name}")
        handles.append(run_sweep.spawn(circuit_name))

    print(f"\n{len(handles)} sweeps launched, waiting...")
    for h in handles:
        h.get()

    print("\nAll sweeps done. Running analysis...")
    analyze_all.remote()
