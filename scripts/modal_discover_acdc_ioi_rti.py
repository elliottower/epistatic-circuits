"""Modal: Discover ACDC circuits for IOI and RTI via backward elimination.

Steps:
1. Compute head-level ablation attribution (mean ablation, measure logit_diff drop)
2. Take top-50 heads by |attribution|
3. Backward eliminate: remove least-important head at each step
4. Stop at K=15 heads

Does NOT run coalition sweeps — those are separate scripts.

Usage:
    cd epistatic-circuits
    modal run --detach scripts/modal_discover_acdc_ioi_rti.py
"""

import modal

app = modal.App("discover-acdc-ioi-rti")

results_volume = modal.Volume.from_name("acdc-discovery-ioi-rti", create_if_missing=True)

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
)

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

NAMES_A = ["Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate"]
NAMES_B = ["Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia"]

K = 15


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def discover(task_name: str):
    import gc
    import json
    import os
    import time
    from itertools import product as cartesian_product

    import numpy as np
    import torch
    from tqdm import tqdm

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    out_path = f"/results/{task_name}_acdc_circuit.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if "acdc_circuit" in existing and len(existing["acdc_circuit"]) == K:
            print(f"[{ts()}] {task_name} ACDC already complete, skipping")
            return

    print(f"[{ts()}] Discovering ACDC circuit for {task_name}")

    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    n_layers = model.cfg.n_layers
    n_model_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    tokenizer = model.tokenizer

    # Generate prompts
    if task_name == "ioi":
        templates = TEMPLATES_IOI
        fmt_keys = ("S", "IO")
    else:
        templates = TEMPLATES_RTI
        fmt_keys = ("D", "C")

    clean_texts = []
    correct_ids = []
    incorrect_ids = []

    for tmpl in templates:
        for name_a, name_b in cartesian_product(NAMES_A, NAMES_B):
            if task_name == "ioi":
                text = tmpl.format(S=name_a, IO=name_b)
                correct_name, incorrect_name = name_b, name_a
            else:
                text = tmpl.format(D=name_a, C=name_b)
                correct_name, incorrect_name = name_b, name_a

            correct_toks = tokenizer.encode(" " + correct_name, add_special_tokens=False)
            incorrect_toks = tokenizer.encode(" " + incorrect_name, add_special_tokens=False)
            if len(correct_toks) != 1 or len(incorrect_toks) != 1:
                continue

            clean_texts.append(text)
            correct_ids.append(correct_toks[0])
            incorrect_ids.append(incorrect_toks[0])

    n_prompts = len(clean_texts)
    print(f"[{ts()}] {n_prompts} prompts")

    token_lists = [model.to_tokens(t, prepend_bos=True) for t in clean_texts]
    max_len = max(t.shape[1] for t in token_lists)
    seq_lens = np.array([t.shape[1] for t in token_lists], dtype=np.int64)

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    for i, toks in enumerate(token_lists):
        padded_tokens[i, :toks.shape[1]] = toks[0]
    del token_lists

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    incorrect_ids_t = torch.tensor(incorrect_ids, device="cuda")
    last_positions = torch.tensor(seq_lens - 1, device="cuda")

    def get_logit_diff(logits_at_last):
        correct_logits = logits_at_last[torch.arange(n_prompts, device="cuda"), correct_ids_t]
        incorrect_logits = logits_at_last[torch.arange(n_prompts, device="cuda"), incorrect_ids_t]
        return (correct_logits - incorrect_logits).mean().item()

    # Compute baseline
    with torch.no_grad():
        logits = model(padded_tokens)
        seq_idx = torch.arange(n_prompts, device="cuda")
        baseline_ld = get_logit_diff(logits[seq_idx, last_positions, :])
    print(f"[{ts()}] Baseline logit_diff: {baseline_ld:.4f}")
    del logits
    torch.cuda.empty_cache()

    # Compute mean z
    print(f"[{ts()}] Computing mean z...")
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
    del z_sums
    torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Head-level attribution (mean ablation)
    print(f"[{ts()}] Computing head attribution...")
    importance = np.zeros((n_layers, n_model_heads))

    for layer in tqdm(range(n_layers), desc="Attribution"):
        for head in range(n_model_heads):
            def hook_fn(act, hook, l=layer, h=head):
                act[:, :, h, :] = mean_z[l][h]
                return act

            with torch.no_grad():
                logits = model.run_with_hooks(
                    padded_tokens,
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)],
                    return_type="logits",
                )
                seq_idx = torch.arange(n_prompts, device="cuda")
                ablated_ld = get_logit_diff(logits[seq_idx, last_positions, :])
                importance[layer, head] = baseline_ld - ablated_ld
                del logits

    # Top-50 heads by |importance|
    flat = importance.flatten()
    top50_idx = np.argsort(np.abs(flat))[::-1][:50]
    top50_heads = [(int(idx // n_model_heads), int(idx % n_model_heads)) for idx in top50_idx]
    top50_labels = [f"L{l}H{h}" for l, h in top50_heads]
    print(f"[{ts()}] Top-50: {top50_labels[:15]}...")

    attr_results = {
        "importance": {f"L{l}H{h}": float(importance[l, h])
                       for l in range(n_layers) for h in range(n_model_heads)},
        "baseline_logit_diff": baseline_ld,
        "top50": [list(h) for h in top50_heads],
    }
    with open(f"/results/{task_name}_attribution.json", "w") as f:
        json.dump(attr_results, f, indent=2)
    results_volume.commit()

    # Step 2: Backward elimination from top-50 to K
    print(f"\n[{ts()}] Backward elimination: {len(top50_heads)} -> {K}")
    current = list(top50_heads)
    elimination_log = []

    while len(current) > K:
        n_current = len(current)
        scores = np.zeros(n_current)
        involved_layers = sorted(set(l for l, h in current))

        for j in tqdm(range(n_current), desc=f"Prune {n_current}->{n_current-1}"):
            candidate = current[:j] + current[j+1:]
            candidate_set = set((l, h) for l, h in candidate)

            def hook_fn(act, hook):
                layer = int(hook.name.split(".")[1])
                for h in range(n_model_heads):
                    if (layer, h) not in candidate_set:
                        act[:, :, h, :] = mean_z[layer][h]
                return act

            hooks = [(f"blocks.{l}.attn.hook_z", hook_fn) for l in involved_layers]
            with torch.no_grad():
                logits = model.run_with_hooks(
                    padded_tokens, fwd_hooks=hooks, return_type="logits",
                )
                seq_idx = torch.arange(n_prompts, device="cuda")
                scores[j] = get_logit_diff(logits[seq_idx, last_positions, :])
                del logits

        # Remove the head whose removal causes the least damage (highest remaining score)
        best_j = int(np.argmax(scores))
        removed = current[best_j]
        remaining_score = scores[best_j]
        current = current[:best_j] + current[best_j+1:]

        elimination_log.append({
            "step": len(top50_heads) - len(current),
            "removed": list(removed),
            "removed_label": f"L{removed[0]}H{removed[1]}",
            "remaining_score": float(remaining_score),
            "n_remaining": len(current),
        })
        print(f"  Step {elimination_log[-1]['step']}: removed {elimination_log[-1]['removed_label']}, "
              f"score={remaining_score:.4f}, {len(current)} heads left")

        if len(current) % 5 == 0:
            partial = {
                "task": task_name,
                "acdc_circuit": [list(h) for h in current],
                "acdc_labels": [f"L{l}H{h}" for l, h in current],
                "elimination_log": elimination_log,
                "baseline_logit_diff": baseline_ld,
                "status": "in_progress",
            }
            with open(out_path, "w") as f:
                json.dump(partial, f, indent=2)
            results_volume.commit()

    acdc_circuit = sorted(current)
    acdc_labels = [f"L{l}H{h}" for l, h in acdc_circuit]
    print(f"\n[{ts()}] ACDC circuit ({K} heads): {acdc_labels}")

    final = {
        "task": task_name,
        "acdc_circuit": [list(h) for h in acdc_circuit],
        "acdc_labels": acdc_labels,
        "elimination_log": elimination_log,
        "baseline_logit_diff": baseline_ld,
        "status": "complete",
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    results_volume.commit()
    print(f"[{ts()}] Saved to {out_path}")


@app.local_entrypoint()
def main():
    handles = []
    for task in ["ioi", "rti"]:
        handles.append(discover.spawn(task))
    print(f"Spawned {len(handles)} ACDC discovery jobs")
    for h in handles:
        h.get()
    print("All ACDC discoveries complete")
