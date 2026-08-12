"""Check n_prompts for all resample sweeps across all volumes."""
import modal

app = modal.App("check-prompt-counts")

ioi_vol = modal.Volume.from_name("ioi-resample-sweep")
rti_vol = modal.Volume.from_name("rti-resample-sweep")
gt_vol = modal.Volume.from_name("gt-resample-sweep")
acdc_vol = modal.Volume.from_name("acdc-resample-sweep")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")


@app.function(
    image=image,
    volumes={"/ioi": ioi_vol, "/rti": rti_vol, "/gt": gt_vol, "/acdc": acdc_vol},
    timeout=60,
)
def check():
    import os
    import numpy as np
    for label, root_dir in [("IOI", "/ioi"), ("RTI", "/rti"), ("GT", "/gt"), ("ACDC", "/acdc")]:
        print(f"\n=== {label} ===")
        for f in sorted(os.listdir(root_dir)):
            if not f.endswith(".npz"):
                continue
            path = os.path.join(root_dir, f)
            d = np.load(path)
            n_prompts = int(d["n_prompts"]) if "n_prompts" in d else "MISSING"
            n_players = int(d["n_players"]) if "n_players" in d else "?"
            value_key = [k for k in d.keys() if k not in ("circuit_heads", "n_players", "n_prompts", "n_completed", "n_total")][0]
            shape = d[value_key].shape
            print(f"  {f}: n_prompts={n_prompts}, n_players={n_players}, "
                  f"value_key={value_key}, shape={shape}")


@app.local_entrypoint()
def main():
    check.remote()
