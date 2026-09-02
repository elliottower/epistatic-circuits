"""Modal: MIB EAP-IG edge scores for vanilla GPT-2 small on IOI.

Produces real edge-level EAP-IG attribution using the MIB implementation, for
head-to-head edges in GPT-2 small on the MIB IOI dataset. The epistatic-circuits
paper reports Spearman rho = 0.58 between pairwise Walsh coefficients and "EAP
edge products" but no script in that repo computes it, so this recomputes the
quantity with the reference implementation rather than a reconstruction.

Image pins, dataset loader and the attribute() call follow a working EAP-IG
environment; the only change is loading stock GPT-2 rather than a factorized
checkpoint. The MIB sources are mounted from a local checkout of the
MIB circuit track, whose path is set by MIB_REPO below.

Usage:
    modal run --detach scripts/modal_eapig_ioi_vanilla_gpt2.py
"""
from __future__ import annotations

import os

import modal

# Local checkout supplying MIB-circuit-track; override with MIB_REPO if it lives elsewhere.
MIB_REPO = os.environ.get(
    "MIB_REPO",
    os.path.join(os.path.expanduser("~"), "Documents/GitHub/factorization-circuits"),
)


def _skip_heavy(p):
    s = str(p)
    return any(x in s for x in [
        "__pycache__", "checkpoint_inventory", "analysis_grassmanian",
        "analysis_elliot", "factorized_spectral", "weight_space_das",
        "writeups", ".pt", ".safetensors", ".git",
        "reference/TransformerLens", "reference/SAELens",
        "reference/learn-mech-interp", "artifacts", "drafts",
    ])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "transformers==4.46.3", "transformer-lens==2.11.0",
        "einops>=0.8", "scipy", "scikit-learn", "pandas", "numpy<2",
        "tqdm", "wandb", "datasets", "pyarrow", "matplotlib",
    )
    .env({"PYTHONPATH": "/root/repo"})
    .add_local_dir(
        os.path.join(MIB_REPO, "MIB/MIB-circuit-track/MIB_circuit_track"),
        "/root/repo/MIB_circuit_track",
    )
    .add_local_dir(
        os.path.join(MIB_REPO, "MIB/MIB-circuit-track/EAP-IG/src/eap"),
        "/root/repo/eap",
    )
)

app = modal.App("eapig-ioi-vanilla-gpt2", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

NUM_EXAMPLES = 200
BATCH_SIZE = 20
IG_STEPS = 5


@app.function(gpu="A10G", timeout=3600, volumes={"/results": results_vol})
def run():
    import json
    import re
    import time
    from functools import partial

    import torch
    from transformer_lens import HookedTransformer

    from eap.graph import Graph
    from eap.attribute import attribute
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric

    ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{ts()}] loading stock gpt2")
    model = HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, center_unembed=False,
        fold_ln=False, device="cuda",
    )
    model.eval()
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    print(f"[{ts()}] loading MIB IOI dataset ({NUM_EXAMPLES} examples)")
    dataset = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train",
                           task="ioi", model_name="gpt2", num_examples=NUM_EXAMPLES)
    dataloader = dataset.to_dataloader(batch_size=BATCH_SIZE)
    metric_fn = get_metric("logit_diff", "ioi", model.tokenizer, model, model_name="gpt2")
    attribution_metric = partial(metric_fn, mean=True, loss=True)

    variants = {}
    for label, kwargs in [
        ("EAP-IG-inputs", dict(method="EAP-IG-inputs", intervention="patching", ig_steps=IG_STEPS)),
        ("EAP", dict(method="EAP", intervention="patching", intervention_dataloader=dataloader)),
    ]:
        print(f"[{ts()}] running {label}")
        g = Graph.from_model(model, neuron_level=False, node_scores=False)
        t0 = time.time()
        attribute(model, g, dataloader, attribution_metric, **kwargs)
        print(f"[{ts()}] {label} done in {time.time()-t0:.1f}s")
        variants[label] = g
    graph = variants["EAP-IG-inputs"]
    elapsed = 0.0

    # head -> head edges only, aggregated over the q/k/v slots of the receiver
    head = re.compile(r"^a(\d+)\.h(\d+)$")

    def extract(g):
        out, n = {}, 0
        for _, edge in g.edges.items():
            pm, cm = head.match(edge.parent.name), head.match(edge.child.name)
            if not (pm and cm):
                continue
            src = f"L{int(pm.group(1))}H{int(pm.group(2))}"
            dst = f"L{int(cm.group(1))}H{int(cm.group(2))}"
            out.setdefault(f"{src}->{dst}", {})[edge.qkv or "none"] = float(edge.score)
            n += 1
        return out, n

    all_variants = {label: dict(zip(("scores", "n"), extract(g)))
                    for label, g in variants.items()}
    pair_scores = {}
    kept = 0
    for name, edge in graph.edges.items():
        pm, cm = head.match(edge.parent.name), head.match(edge.child.name)
        if not (pm and cm):
            continue
        src = f"L{int(pm.group(1))}H{int(pm.group(2))}"
        dst = f"L{int(cm.group(1))}H{int(cm.group(2))}"
        key = f"{src}->{dst}"
        pair_scores.setdefault(key, {})[edge.qkv or "none"] = float(edge.score)
        kept += 1

    result = {
        "experiment": "MIB EAP-IG edge scores, stock GPT-2 small, MIB IOI",
        "method": "EAP-IG-inputs", "intervention": "patching",
        "ig_steps": IG_STEPS, "num_examples": NUM_EXAMPLES,
        "batch_size": BATCH_SIZE, "elapsed_s": round(elapsed, 1),
        "n_edges_total": len(graph.edges),
        "n_head_to_head_edges": kept,
        "head_edge_scores": pair_scores,
        "variants": {k: v["scores"] for k, v in all_variants.items()},
    }
    out = "/results/eapig_ioi_vanilla_gpt2.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"[{ts()}] wrote {out}: {len(pair_scores)} head pairs, {kept} edges")
    return result


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    r = run.remote()
    dest = Path(__file__).resolve().parent.parent / "results" / "eapig_ioi_vanilla_gpt2.json"
    dest.write_text(json.dumps(r, indent=2))
    print(f"saved: {dest}")
    print(f"  {r['n_head_to_head_edges']} head-to-head edges over {len(r['head_edge_scores'])} ordered pairs")
