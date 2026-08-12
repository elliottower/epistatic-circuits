"""Modal wrapper: IC-15 coalition sweep (zero + mean ablation).

Computes exact 2^15 coalition tables for the top-15 heads from the
IC greedy forward selection trajectory (BIC, mean ablation). These
15 heads are the first 15 added by the greedy algorithm, in order:

  L10H7, L4H0, L0H1, L4H11, L9H4, L5H6, L8H9, L8H4, L0H5,
  L11H7, L0H9, L11H8, L10H3, L1H3, L0H3

Runs both zero and mean ablation in parallel on separate A10G GPUs.
Uses the 8-name lists (512 prompts) matching all other v2 sweeps.

Usage:
    modal run --detach experiments_batch2/genetics/modal_ic15_coalition_sweep.py
"""

import modal

app = modal.App("ic15-coalition-sweep-zero-and-mean")

results_volume = modal.Volume.from_name("ic15-coalition-sweep", create_if_missing=True)

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
        "experiments_batch2/genetics/coalition_sweep_v2.py",
        remote_path="/app/coalition_sweep_v2.py",
    )
)

IC15_HEADS = [
    (10, 7), (4, 0), (0, 1), (4, 11), (9, 4),
    (5, 6), (8, 9), (8, 4), (0, 5), (11, 7),
    (0, 9), (11, 8), (10, 3), (1, 3), (0, 3),
]


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/results": results_volume},
)
def run_ic15_sweep(ablation_type: str):
    import sys
    sys.path.insert(0, "/app")

    import coalition_sweep_v2
    import transformer_lens

    from coalition_sweep_v2 import run_sweep, timestamp

    coalition_sweep_v2.NAMES_A = ["Alice", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Kate"]
    coalition_sweep_v2.NAMES_B = ["Bob", "Carol", "Eric", "Fiona", "George", "Helen", "Ivan", "Julia"]

    coalition_sweep_v2.CIRCUITS["ic15"] = {"heads": IC15_HEADS}

    # IC-15 was discovered on IOI prompts — patch generate_prompts to
    # route "ic15" through the IOI template branch (the original checks
    # for circuit_name in ("ioi", "weight_ioi", "random15") only).
    _original_generate = coalition_sweep_v2.generate_prompts

    def _generate_ic15_patched(circuit_name, tokenizer):
        if circuit_name == "ic15":
            return _original_generate("ioi", tokenizer)
        return _original_generate(circuit_name, tokenizer)

    coalition_sweep_v2.generate_prompts = _generate_ic15_patched

    print(f"[{timestamp()}] IC-15 coalition sweep: ablation={ablation_type}")
    print(f"[{timestamp()}] Heads: {[f'L{l}H{h}' for l, h in IC15_HEADS]}")
    print(f"[{timestamp()}] 15 players, 32768 coalitions, 512 prompts")

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.set_use_attn_result(True)
    model.eval()
    print(f"[{timestamp()}] Model loaded on {model.cfg.device}")

    out_path = f"/results/ic15_{ablation_type}_v2_coalition_values.npz"
    run_sweep(model, "ic15", ablation_type, out_path, checkpoint_every=2048)

    results_volume.commit()
    print(f"[{timestamp()}] Volume committed for ic15/{ablation_type}")
    return f"ic15/{ablation_type}: 32768 coalitions complete"


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    ablation_types = ["zero", "mean"]
    print(f"[{datetime.now(timezone.utc).isoformat()}] Launching IC-15 sweeps: {ablation_types}")

    handles = []
    for abl in ablation_types:
        handles.append(run_ic15_sweep.spawn(abl))

    for h in handles:
        result = h.get()
        print(f"Completed: {result}")

    print(f"\nBoth sweeps complete. Download with:")
    for abl in ablation_types:
        print(f"  modal volume get ic15-coalition-sweep ic15_{abl}_v2_coalition_values.npz experiments_batch2/genetics/")
