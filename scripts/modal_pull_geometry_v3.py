"""Pull primitive-geometry v3 results from Modal volume."""
import modal
import json

app = modal.App("pull-geometry-v3")
vol = modal.Volume.from_name("primitive-geometry-results-v3")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/out": vol}, timeout=60)
def pull():
    import os
    results = {}
    for fname in sorted(os.listdir("/out")):
        fpath = os.path.join("/out", fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}  ({size} bytes)")
        if fname.endswith(".json"):
            with open(fpath) as f:
                results[fname] = json.load(f)
        elif fname.endswith(".txt"):
            with open(fpath) as f:
                results[fname] = f.read()
    return results


@app.local_entrypoint()
def main():
    r = pull.remote()
    if not r:
        print("No results yet — run may still be in progress")
        return

    if "run_log.txt" in r:
        print("=== RUN LOG ===")
        print(r["run_log.txt"])

    if "primitive_geometry_summary_v3.json" in r:
        print("\n=== SUMMARY ===")
        print(json.dumps(r["primitive_geometry_summary_v3.json"], indent=2))
        with open("results/primitive_geometry_summary_v3.json", "w") as f:
            json.dump(r["primitive_geometry_summary_v3.json"], f, indent=2)
        print("Saved results/primitive_geometry_summary_v3.json")

    if "primitive_geometry_rows_v3.json" in r:
        with open("results/primitive_geometry_rows_v3.json", "w") as f:
            json.dump(r["primitive_geometry_rows_v3.json"], f, indent=1)
        print("Saved results/primitive_geometry_rows_v3.json")

    if "primitive_geometry_pairs_v3.json" in r:
        with open("results/primitive_geometry_pairs_v3.json", "w") as f:
            json.dump(r["primitive_geometry_pairs_v3.json"], f, indent=1)
        print("Saved results/primitive_geometry_pairs_v3.json")
