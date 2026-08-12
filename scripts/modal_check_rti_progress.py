"""Quick check: how many coalitions are complete in the RTI NPZ files."""

import modal

app = modal.App("check-rti-progress")
rti_volume = modal.Volume.from_name("rti-coalition-sweep")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")


@app.function(image=image, timeout=120, volumes={"/data": rti_volume})
def check():
    import numpy as np
    for ablation in ["zero", "mean"]:
        path = f"/data/rti_{ablation}_v2_coalition_values.npz"
        data = np.load(path)
        target = data["target_logits"]
        n_complete = int(data.get("n_coalitions_completed", 0))
        nan_count = int(np.isnan(target[:, 0]).sum())
        total = target.shape[0]
        print(f"  {ablation}: {n_complete}/{total} reported, {total - nan_count}/{total} non-NaN rows")
        data.close()


@app.local_entrypoint()
def main():
    check.remote()
