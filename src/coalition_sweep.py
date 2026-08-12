"""Coalition value-function sweep v2: batched, multi-ablation, 512 prompts.

Changes from v1:
- Cartesian product of names (64 pairs/template vs 8)
- 8 IOI templates (was 5) → 512 prompts
- Batched forward passes (~10x faster)
- Supports zero, mean, and resample ablation
- Pre-caches activations for resample ablation

Usage (local test):
    uv run python experiments_batch2/genetics/coalition_sweep_v2.py \
        --circuit weight_ioi --max-coalitions 16 --device cpu --ablation zero

Usage (full sweep, called by modal launcher):
    python coalition_sweep_v2.py --circuit weight_ioi --device cuda --ablation all
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product as cartesian_product
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CIRCUITS = {
    "rti": {
        "heads": [
            (0, 8), (0, 9), (0, 11),
            (4, 11),
            (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
            (10, 11), (11, 9), (11, 11),
        ],
    },
    "ioi": {
        "heads": [
            (0, 1), (3, 0),
            (2, 2), (4, 11),
            (5, 5), (6, 9),
            (7, 3), (7, 9), (8, 6), (8, 10),
            (9, 9), (9, 6), (10, 0),
            (10, 7), (11, 10),
        ],
    },
    "random15": {
        "heads": [
            (0, 3), (1, 0), (1, 10), (3, 8), (3, 9), (3, 10), (4, 8),
            (6, 5), (6, 10), (7, 10), (8, 1), (8, 8), (8, 11), (9, 4), (10, 8),
        ],
    },
    "weight_ioi": {
        "heads": [
            (0, 1), (0, 5), (0, 10),
            (4, 11),
            (5, 1), (5, 5), (5, 8), (5, 9),
            (6, 1), (6, 9),
            (7, 2), (7, 9), (7, 10),
            (10, 2), (10, 7),
        ],
    },
}

TEMPLATES_IOI = [
    "When {S} and {IO} went to the store, {S} gave a bottle to",
    "Then {S} and {IO} had a meeting. {S} passed the document to",
    "{S} and {IO} were working together. {S} handed the report to",
    "After {S} met {IO} at lunch, {S} gave a gift to",
    "While {S} and {IO} were talking, {S} offered the keys to",
    "Then {S} and {IO} went to the park. {S} tossed the ball to",
    "{S} visited {IO} at the office. {S} delivered the package to",
    "When {S} saw {IO} at the party, {S} brought the cake to",
]

TEMPLATES_RTI = [
    "Then {D} and {C} went to the store. {D} gave a drink to",
    "{D} told {C} a story. {D} then handed the book to",
    "{D} met {C} at the park. {D} passed the ball to",
    "When {D} and {C} arrived, {D} gave the keys to",
    "{D} helped {C} move. {D} gave the boxes to",
    "Then {D} and {C} stopped for lunch. {D} offered the food to",
    "When {D} visited {C} at home, {D} showed the photos to",
    "{D} and {C} were at the library. {D} returned the books to",
]

NAMES_A = ["Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate", "Mary", "Sarah", "Lisa", "Anna"]
NAMES_B = ["Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia", "Tom", "Mark", "Paul", "Jane"]

ABLATION_TYPES = ["zero", "mean", "resample"]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def generate_prompts(circuit_name, tokenizer):
    if circuit_name in ("ioi", "weight_ioi", "random15"):
        templates = TEMPLATES_IOI
    else:
        templates = TEMPLATES_RTI

    prompts = []
    for tmpl in templates:
        for name_a, name_b in cartesian_product(NAMES_A, NAMES_B):
            if circuit_name in ("ioi", "weight_ioi", "random15"):
                text = tmpl.format(S=name_a, IO=name_b)
            else:
                text = tmpl.format(D=name_a, C=name_b)
            correct_name, incorrect_name = name_b, name_a

            correct_toks = tokenizer.encode(" " + correct_name, add_special_tokens=False)
            incorrect_toks = tokenizer.encode(" " + incorrect_name, add_special_tokens=False)
            if len(correct_toks) != 1 or len(incorrect_toks) != 1:
                continue
            prompts.append({
                "text": text,
                "correct_id": correct_toks[0],
                "incorrect_id": incorrect_toks[0],
            })
    return prompts


def tokenize_all(model, prompts):
    texts = [p["text"] for p in prompts]
    all_tokens = model.to_tokens(texts)
    correct_ids = torch.tensor([p["correct_id"] for p in prompts], device=all_tokens.device)
    incorrect_ids = torch.tensor([p["incorrect_id"] for p in prompts], device=all_tokens.device)
    return all_tokens, correct_ids, incorrect_ids


def compute_mean_activations(model, all_tokens, batch_size=64):
    n_layers = model.cfg.n_layers
    n_prompts = all_tokens.shape[0]
    sums = {}
    total_positions = 0

    with torch.no_grad():
        for start in range(0, n_prompts, batch_size):
            batch = all_tokens[start:start + batch_size]
            _, cache = model.run_with_cache(
                batch,
                names_filter=lambda n: "attn.hook_result" in n,
            )
            for L in range(n_layers):
                act = cache[f"blocks.{L}.attn.hook_result"]
                if L not in sums:
                    sums[L] = torch.zeros(
                        model.cfg.n_heads, model.cfg.d_model,
                        device=all_tokens.device, dtype=act.dtype,
                    )
                sums[L] += act.sum(dim=(0, 1))
            total_positions += batch.shape[0] * batch.shape[1]

    return {L: sums[L] / total_positions for L in range(n_layers)}


def cache_clean_activations(model, all_tokens, batch_size=64):
    n_layers = model.cfg.n_layers
    n_prompts = all_tokens.shape[0]
    seq_len = all_tokens.shape[1]
    cached = {}

    with torch.no_grad():
        for start in range(0, n_prompts, batch_size):
            batch = all_tokens[start:start + batch_size]
            _, cache = model.run_with_cache(
                batch,
                names_filter=lambda n: "attn.hook_result" in n,
            )
            for L in range(n_layers):
                act = cache[f"blocks.{L}.attn.hook_result"]
                if L not in cached:
                    cached[L] = torch.zeros(
                        n_prompts, seq_len, model.cfg.n_heads, model.cfg.d_model,
                        device="cpu", dtype=act.dtype,
                    )
                cached[L][start:start + batch.shape[0]] = act.cpu()

    return cached


def make_resample_permutation(n, rng):
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    while len(fixed) > 0:
        if len(fixed) == 1:
            swap_with = (fixed[0] + 1) % n
            perm[fixed[0]], perm[swap_with] = perm[swap_with], perm[fixed[0]]
            break
        rng.shuffle(fixed)
        for i in range(0, len(fixed) - 1, 2):
            perm[fixed[i]], perm[fixed[i + 1]] = perm[fixed[i + 1]], perm[fixed[i]]
        if len(fixed) % 2 == 1:
            last = fixed[-1]
            swap_with = (last + 1) % n
            perm[last], perm[swap_with] = perm[swap_with], perm[last]
        break
    return perm


def evaluate_coalition_batched(
    model, coalition_mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
    ablation_type, mean_z=None, cached_acts=None, resample_perm=None,
    batch_size=64,
):
    heads_to_ablate_by_layer = defaultdict(list)
    for i, (layer, head) in enumerate(circuit_heads):
        if not coalition_mask[i]:
            heads_to_ablate_by_layer[layer].append(head)

    n_prompts = all_tokens.shape[0]
    target_logits = np.zeros(n_prompts, dtype=np.float64)
    foil_logits = np.zeros(n_prompts, dtype=np.float64)

    for start in range(0, n_prompts, batch_size):
        end = min(start + batch_size, n_prompts)
        batch_tokens = all_tokens[start:end]

        hooks = []
        for layer, head_indices in heads_to_ablate_by_layer.items():
            if ablation_type == "zero":
                def make_hook(hidxs):
                    def hook_fn(act, hook):
                        for h in hidxs:
                            act[:, :, h, :] = 0.0
                        return act
                    return hook_fn
            elif ablation_type == "mean":
                mz = mean_z[layer]
                def make_hook(hidxs, mz_=mz):
                    def hook_fn(act, hook):
                        for h in hidxs:
                            act[:, :, h, :] = mz_[h]
                        return act
                    return hook_fn
            elif ablation_type == "resample":
                donor_indices = resample_perm[start:end]
                donor_acts = cached_acts[layer][donor_indices].to(all_tokens.device)
                def make_hook(hidxs, donor=donor_acts):
                    def hook_fn(act, hook):
                        for h in hidxs:
                            act[:, :, h, :] = donor[:, :, h, :]
                        return act
                    return hook_fn
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")

            hooks.append((f"blocks.{layer}.attn.hook_result", make_hook(head_indices)))

        with torch.no_grad():
            logits = model.run_with_hooks(batch_tokens, fwd_hooks=hooks)
            last_logits = logits[:, -1, :]
            batch_correct = correct_ids[start:end]
            batch_incorrect = incorrect_ids[start:end]
            target_logits[start:end] = last_logits[
                torch.arange(end - start, device=logits.device), batch_correct
            ].cpu().numpy()
            foil_logits[start:end] = last_logits[
                torch.arange(end - start, device=logits.device), batch_incorrect
            ].cpu().numpy()

    return target_logits, foil_logits


def run_sweep(
    model, circuit_name, ablation_type, save_path,
    max_coalitions=None, checkpoint_every=1024, batch_size=64,
):
    circuit_def = CIRCUITS[circuit_name]
    circuit_heads = circuit_def["heads"]
    n_players = len(circuit_heads)
    n_total = 2 ** n_players

    n_coalitions = min(max_coalitions, n_total) if max_coalitions else n_total

    print(f"[{timestamp()}] Circuit: {circuit_name}, ablation: {ablation_type}, "
          f"{n_players} heads, {n_coalitions}/{n_total} coalitions")

    tokenizer = model.tokenizer
    prompts = generate_prompts(circuit_name, tokenizer)
    n_prompts = len(prompts)
    print(f"[{timestamp()}] Generated {n_prompts} prompts")

    all_tokens, correct_ids, incorrect_ids = tokenize_all(model, prompts)
    print(f"[{timestamp()}] Tokenized: shape {all_tokens.shape}")

    print(f"[{timestamp()}] Computing mean activations...")
    mean_z = compute_mean_activations(model, all_tokens, batch_size=batch_size)

    cached_acts = None
    if ablation_type == "resample":
        print(f"[{timestamp()}] Caching clean activations for resample ablation...")
        cached_acts = cache_clean_activations(model, all_tokens, batch_size=batch_size)
        print(f"[{timestamp()}] Cached {len(cached_acts)} layers")

    resample_rng = np.random.default_rng(42)

    all_target_logits = np.full((n_coalitions, n_prompts), np.nan, dtype=np.float64)
    all_foil_logits = np.full((n_coalitions, n_prompts), np.nan, dtype=np.float64)
    save_path = Path(save_path)

    if max_coalitions is not None and max_coalitions < n_total:
        coalition_indices = np.sort(
            np.random.default_rng(42).choice(n_total, n_coalitions, replace=False)
        )
    else:
        coalition_indices = np.arange(n_coalitions)

    print(f"[{timestamp()}] Computing intact-model baseline...")
    intact_mask = np.ones(n_players, dtype=bool)
    intact_target, intact_foil = evaluate_coalition_batched(
        model, intact_mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
        ablation_type="mean", mean_z=mean_z, batch_size=batch_size,
    )
    intact_logit_diff = intact_target - intact_foil
    print(f"[{timestamp()}] Intact baseline: mean logit-diff = {intact_logit_diff.mean():.4f}")

    start_time = time.time()
    for idx_pos, coal_idx in enumerate(tqdm(coalition_indices, desc=f"{circuit_name}/{ablation_type}")):
        mask = np.array([(int(coal_idx) >> i) & 1 for i in range(n_players)], dtype=bool)

        resample_perm = None
        if ablation_type == "resample":
            resample_perm = make_resample_permutation(n_prompts, resample_rng)

        tgt, foil = evaluate_coalition_batched(
            model, mask, circuit_heads, all_tokens, correct_ids, incorrect_ids,
            ablation_type=ablation_type, mean_z=mean_z,
            cached_acts=cached_acts, resample_perm=resample_perm,
            batch_size=batch_size,
        )
        all_target_logits[idx_pos] = tgt
        all_foil_logits[idx_pos] = foil

        if (idx_pos + 1) % checkpoint_every == 0:
            elapsed = time.time() - start_time
            rate = (idx_pos + 1) / elapsed
            remaining = (n_coalitions - idx_pos - 1) / rate
            print(f"[{timestamp()}] {idx_pos + 1}/{n_coalitions} done, "
                  f"{rate:.1f} coal/s, ~{remaining / 60:.1f} min remaining")
            np.savez(
                save_path,
                target_logits=all_target_logits,
                foil_logits=all_foil_logits,
                coalition_indices=coalition_indices,
                circuit_name=circuit_name,
                ablation_type=ablation_type,
                circuit_heads=np.array(circuit_heads),
                n_players=n_players,
                n_prompts=n_prompts,
                intact_target_logits=intact_target,
                intact_foil_logits=intact_foil,
                n_coalitions_completed=idx_pos + 1,
            )

    elapsed = time.time() - start_time
    print(f"[{timestamp()}] Sweep complete: {n_coalitions} coalitions in {elapsed:.1f}s "
          f"({n_coalitions / elapsed:.1f} coal/s)")

    logit_diffs = all_target_logits - all_foil_logits
    mean_logit_diffs = logit_diffs.mean(axis=1)

    np.savez(
        save_path,
        target_logits=all_target_logits,
        foil_logits=all_foil_logits,
        coalition_indices=coalition_indices,
        circuit_name=circuit_name,
        ablation_type=ablation_type,
        circuit_heads=np.array(circuit_heads),
        n_players=n_players,
        n_prompts=n_prompts,
        intact_target_logits=intact_target,
        intact_foil_logits=intact_foil,
        n_coalitions_completed=n_coalitions,
        elapsed_seconds=elapsed,
        prompt_texts=np.array([p["text"] for p in prompts]),
        prompt_correct_ids=np.array([p["correct_id"] for p in prompts]),
        prompt_incorrect_ids=np.array([p["incorrect_id"] for p in prompts]),
        sweep_version=2,
    )
    print(f"[{timestamp()}] Saved to {save_path}")
    print(f"  Data shape: ({all_target_logits.shape[0]}, {all_target_logits.shape[1]})")
    print(f"  Storage: {save_path.stat().st_size / 1e6:.1f} MB")

    metadata = {
        "circuit": circuit_name,
        "ablation_type": ablation_type,
        "n_players": n_players,
        "n_coalitions": int(n_coalitions),
        "n_prompts": n_prompts,
        "elapsed_seconds": elapsed,
        "sweep_version": 2,
        "v_mult_all_active": float(mean_logit_diffs[coalition_indices == n_total - 1][0])
            if n_total - 1 in coalition_indices else None,
        "v_mult_all_ablated": float(mean_logit_diffs[coalition_indices == 0][0])
            if 0 in coalition_indices else None,
        "intact_mean_logit_diff": float(intact_logit_diff.mean()),
        "timestamp": timestamp(),
    }
    meta_path = save_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[{timestamp()}] Metadata saved to {meta_path}")

    return all_target_logits, all_foil_logits, coalition_indices


def main():
    parser = argparse.ArgumentParser(description="Coalition sweep v2 (batched, multi-ablation)")
    parser.add_argument("--circuit", choices=list(CIRCUITS.keys()), required=True)
    parser.add_argument("--ablation", choices=ABLATION_TYPES + ["all"], default="mean")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-coalitions", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    args = parser.parse_args()

    import transformer_lens
    print(f"[{timestamp()}] Loading GPT-2 small on {args.device}...")
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device=args.device)
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded")

    ablation_types = ABLATION_TYPES if args.ablation == "all" else [args.ablation]

    for abl in ablation_types:
        out_path = Path(args.out_dir) / f"{args.circuit}_{abl}_v2_coalition_values.npz"
        run_sweep(
            model, args.circuit, abl, out_path,
            max_coalitions=args.max_coalitions,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
