"""Check completion status of Walsh discovery volumes."""
import modal

app = modal.App("check-walsh-status")
vol_rti = modal.Volume.from_name("walsh-discovery-rti")
vol_gt = modal.Volume.from_name("walsh-discovery-gt")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")

@app.function(
    image=image,
    volumes={"/rti": vol_rti, "/gt": vol_gt},
    timeout=60,
)
def check():
    import numpy as np
    for name, path in [
        ("RTI", "/rti/walsh_rti_144heads_mean_coalitions.npz"),
        ("GT", "/gt/walsh_gt_144heads_mean_coalitions.npz"),
    ]:
        data = np.load(path)
        n_completed = int(data["n_completed"])
        n_total = int(data["n_total"])
        done = n_completed == n_total
        print(f"{name}: {n_completed}/{n_total} ({'COMPLETE' if done else 'INCOMPLETE'})")
        if done:
            print(f"  Keys: {list(data.keys())}")
            if "mean_logit_diffs" in data:
                ms = data["mean_logit_diffs"]
                print(f"  All-ablated LD: {ms[0]:.4f}, Clean LD: {ms[1]:.4f}")
            if "mean_scores" in data:
                ms = data["mean_scores"]
                print(f"  All-ablated score: {ms[0]:.4f}, Clean score: {ms[1]:.4f}")

@app.local_entrypoint()
def main():
    check.remote()
