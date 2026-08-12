"""Quick check: what files are on the ioi-resample-sweep and acdc-resample-sweep volumes?"""
import modal

app = modal.App("check-volumes")
ioi_vol = modal.Volume.from_name("ioi-resample-sweep")
acdc_vol = modal.Volume.from_name("acdc-resample-sweep")

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==1.26.4")

@app.function(image=image, volumes={"/ioi": ioi_vol, "/acdc": acdc_vol}, timeout=60)
def check():
    import os
    import numpy as np
    for label, root_dir in [("IOI", "/ioi"), ("ACDC", "/acdc")]:
        print(f"\n=== {label} ===")
        for root, dirs, files in os.walk(root_dir):
            for f in sorted(files):
                path = os.path.join(root, f)
                size_mb = os.path.getsize(path) / 1e6
                extra = ""
                if f.endswith(".npz"):
                    d = np.load(path)
                    if "n_completed" in d:
                        nc = int(d["n_completed"])
                        nt = int(d.get("n_total", 0))
                        extra = f"  [{nc}/{nt}]" if nt else f"  [{nc} done]"
                print(f"  {f}  ({size_mb:.1f} MB){extra}")

@app.local_entrypoint()
def main():
    check.remote()
