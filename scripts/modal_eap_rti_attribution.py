"""Modal: EAP head attribution on RTI prompts (correct 12-category set).

Computes activation × gradient head-level scores for all 144 GPT-2 heads
on the full RTI prompt set. Top-15 heads = "EAP-discovered RTI circuit."

This is zero-ablation EAP: the baseline is zero, so the edge activation
IS the clean activation. Score = |activation * gradient| summed over
(prompts, positions, d_head).

Fast job (~5 min on A10G): one forward+backward per prompt batch.

Usage:
    cd epistatic-circuits
    modal run scripts/modal_eap_rti_attribution.py
"""

import modal

app = modal.App("eap-rti-attribution")

results_volume = modal.Volume.from_name("rti-eap-attribution", create_if_missing=True)

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
    .add_local_file(
        "src/rti_prompts.py",
        remote_path="/app/rti_prompts.py",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_eap_attribution():
    import json
    import sys
    import time
    from functools import partial

    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/app")

    import transformer_lens

    from rti_prompts import make_prompts

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    print(f"[{ts()}] Loading GPT-2...")
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print(f"[{ts()}] Generating RTI prompts (12-category set)...")
    prompts = make_prompts(model.tokenizer)
    valid = [p for p in prompts if p["correct_id"] is not None and p["distractor_id"] is not None]
    n_prompts = len(valid)

    categories = sorted(set(p["category"] for p in valid))
    print(f"[{ts()}] {n_prompts} valid prompts across {len(categories)} categories")

    all_tokens = []
    correct_ids = []
    distractor_ids = []
    max_len = 0
    for p in valid:
        toks = model.to_tokens(p["text"], prepend_bos=True)
        all_tokens.append(toks)
        correct_ids.append(p["correct_id"])
        distractor_ids.append(p["distractor_id"])
        max_len = max(max_len, toks.shape[1])

    padded_tokens = torch.zeros(n_prompts, max_len, dtype=torch.long, device="cuda")
    seq_lens = np.zeros(n_prompts, dtype=np.int64)
    for i, toks in enumerate(all_tokens):
        padded_tokens[i, :toks.shape[1]] = toks[0]
        seq_lens[i] = toks.shape[1]

    correct_ids_t = torch.tensor(correct_ids, device="cuda")
    distractor_ids_t = torch.tensor(distractor_ids, device="cuda")

    print(f"[{ts()}] Padded shape: {padded_tokens.shape}, running EAP attribution...")

    head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    batch_size = 64

    for start in tqdm(range(0, n_prompts, batch_size), desc="EAP batches"):
        end = min(start + batch_size, n_prompts)
        batch_tokens = padded_tokens[start:end]
        batch_seq_lens = seq_lens[start:end]
        batch_correct = correct_ids_t[start:end]
        batch_distractor = distractor_ids_t[start:end]
        batch_n = end - start

        model.zero_grad()
        activations = {}

        def make_capture(layer):
            def hook_fn(act, hook):
                act.retain_grad()
                activations[layer] = act
                return act
            return hook_fn

        hooks = [
            (f"blocks.{l}.attn.hook_result", make_capture(l))
            for l in range(n_layers)
        ]
        logits = model.run_with_hooks(batch_tokens, fwd_hooks=hooks)

        batch_indices = torch.arange(batch_n, device="cuda")
        last_positions = torch.tensor(batch_seq_lens - 1, device="cuda")
        last_logits = logits[batch_indices, last_positions, :]

        correct_logits = last_logits[batch_indices, batch_correct]
        distractor_logits = last_logits[batch_indices, batch_distractor]
        logit_diff = (correct_logits - distractor_logits).mean()

        logit_diff.backward()

        for l in range(n_layers):
            act = activations[l]
            grad = act.grad
            if grad is not None:
                per_head = (act * grad).abs().sum(dim=(0, 1, 3))
                head_scores[l] += per_head.detach().cpu().numpy()

        del activations, logits
        torch.cuda.empty_cache()

    n_batches = (n_prompts + batch_size - 1) // batch_size
    head_scores /= n_batches

    flat_scores = head_scores.flatten()
    top_15_flat = np.argsort(flat_scores)[::-1][:15]
    top_15_heads = [(int(idx // n_heads), int(idx % n_heads)) for idx in top_15_flat]

    print(f"\n[{ts()}] EAP head attribution results (top 30):")
    all_ranked = np.argsort(flat_scores)[::-1]
    for rank, idx in enumerate(all_ranked[:30]):
        l, h = idx // n_heads, idx % n_heads
        print(f"  {rank+1:3d}. L{l}H{h}: {flat_scores[idx]:.6f}")

    top_15_labels = [f"L{l}H{h}" for l, h in top_15_heads]
    print(f"\n[{ts()}] EAP top-15 circuit for RTI:")
    print(f"  {top_15_labels}")

    known_rti = [
        (0, 8), (0, 9), (0, 11), (4, 11),
        (4, 0), (5, 6), (5, 7), (7, 0), (8, 4), (8, 7), (9, 3), (9, 10),
        (10, 11), (11, 9), (11, 11),
    ]
    overlap = set(top_15_heads) & set(known_rti)
    print(f"  Overlap with known RTI circuit: {len(overlap)}/15")
    print(f"  Shared heads: {sorted(overlap)}")

    result = {
        "method": "EAP_zero_ablation",
        "task": "RTI",
        "n_prompts": n_prompts,
        "n_categories": len(categories),
        "head_scores": {f"L{l}H{h}": round(float(head_scores[l, h]), 8)
                        for l in range(n_layers) for h in range(n_heads)},
        "top_15_circuit": [[int(l), int(h)] for l, h in top_15_heads],
        "top_15_labels": top_15_labels,
        "known_rti_overlap": len(overlap),
        "known_rti_shared_heads": sorted([[l, h] for l, h in overlap]),
        "all_rankings": [
            {"rank": rank + 1, "layer": int(idx // n_heads), "head": int(idx % n_heads),
             "score": round(float(flat_scores[idx]), 8)}
            for rank, idx in enumerate(all_ranked)
        ],
    }

    out_path = "/results/eap_rti_head_attribution.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    np.savez(
        "/results/eap_rti_head_scores.npz",
        head_scores=head_scores,
        top_15_circuit=np.array(top_15_heads),
    )
    results_volume.commit()

    print(f"\n[{ts()}] Saved to {out_path}")
    return result


@app.local_entrypoint()
def main():
    result = run_eap_attribution.remote()
    print(f"\nEAP top-15 RTI circuit: {result['top_15_labels']}")
    print(f"Overlap with known RTI circuit: {result['known_rti_overlap']}/15")
