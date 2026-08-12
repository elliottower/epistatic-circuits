"""Diagnose RTI NPZ file structure to find why v3 crashed on RTI."""
import modal

app = modal.App("diagnose-rti-npz")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")

SWEEPS = {
    "/vol/rti-resample": "rti-resample-sweep",
    "/vol/rti-walsh": "rti-walsh-circuits-sweep",
    "/vol/rti-v5": "rti-sweep-v5",
}
volumes = {p: modal.Volume.from_name(n) for p, n in SWEEPS.items()}


@app.function(image=image, volumes=volumes, timeout=300)
def run():
    import numpy as np
    from pathlib import Path

    for mount_path in SWEEPS:
        print(f"\n=== {mount_path} ===")
        for p in sorted(Path(mount_path).rglob("*.npz")):
            try:
                d = np.load(str(p), allow_pickle=True)
            except Exception as e:
                print(f"  {p.name}: LOAD ERROR {e}")
                continue

            keys = list(d.keys())
            print(f"\n  {p.name}:")
            print(f"    keys: {keys}")

            if 'circuit_heads' in d:
                ch = d['circuit_heads']
                print(f"    circuit_heads: shape={ch.shape} dtype={ch.dtype}")
                print(f"    heads (first 3): {ch[:3].tolist()}")

            if 'n_players' in d:
                print(f"    n_players: {int(d['n_players'])}")
            if 'n_prompts' in d:
                print(f"    n_prompts: {int(d['n_prompts'])}")
            if 'n_completed' in d:
                print(f"    n_completed: {int(d['n_completed'])}")
            if 'n_total' in d:
                print(f"    n_total: {int(d['n_total'])}")

            for vk in ('logit_diff', 'prob_diff', 'value', 'values', 'metric',
                        'target_logits', 'foil_logits'):
                if vk in d:
                    arr = np.asarray(d[vk])
                    print(f"    {vk}: shape={arr.shape} dtype={arr.dtype} "
                          f"min={float(arr.min()):.4f} max={float(arr.max()):.4f}")

            if 'coalition_indices' in d:
                ci = np.asarray(d['coalition_indices'])
                print(f"    coalition_indices: shape={ci.shape} range=[{ci.min()},{ci.max()}]")


@app.local_entrypoint()
def main():
    run.remote()
