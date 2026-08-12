"""Generate Figure: MIB faithfulness curves (144-head experiment).

Reads the 144-head MIB results and plots faithfulness vs circuit size
for Walsh, activation patching, combined, and random (5-seed average).

Usage: cd epistatic-circuits && python3 scripts/plot_mib_curves.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open("results/mib_144head/mib_faithfulness_results.json") as f:
    data = json.load(f)

fr = data["faithfulness_results"]

METHOD_STYLE = {
    "walsh":      ("Walsh",                  "#2ca02c", "D", "-",  1.8),
    "actp":       ("Activation patching",    "#1f77b4", "o", "-",  1.8),
    "walsh_actp": ("Walsh + ActP",           "#9467bd", "s", "--", 1.2),
    "random":     ("Random (5 seeds)",       "gray",    "x", ":",  1.0),
}

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.axhline(1.0, color="gray", linewidth=0.5, alpha=0.4, linestyle=":")

for method_key in ["walsh", "actp", "walsh_actp", "random"]:
    if method_key == "random":
        curves = []
        for seed in range(5):
            curve = fr[f"random_{seed}"]["curve"]
            sizes = [curve[k]["k_fraction"] for k in sorted(curve, key=int)]
            faiths = [curve[k]["faithfulness"] for k in sorted(curve, key=int)]
            curves.append(faiths)
        mean_faiths = np.mean(curves, axis=0)
        std_faiths = np.std(curves, axis=0)
        label, color, marker, ls, lw = METHOD_STYLE["random"]
        ax.plot(sizes, mean_faiths, marker=marker, linestyle=ls, color=color,
                linewidth=lw, markersize=4, label=label, alpha=0.7)
        ax.fill_between(sizes, mean_faiths - std_faiths, mean_faiths + std_faiths,
                        color=color, alpha=0.15)
    else:
        curve = fr[method_key]["curve"]
        sizes = [curve[k]["k_fraction"] for k in sorted(curve, key=int)]
        faiths = [curve[k]["faithfulness"] for k in sorted(curve, key=int)]
        label, color, marker, ls, lw = METHOD_STYLE[method_key]
        ax.plot(sizes, faiths, marker=marker, linestyle=ls, color=color,
                linewidth=lw, markersize=4, label=label, alpha=0.9)

ax.set_xlabel("Circuit size ($k / 144$)")
ax.set_ylabel("Faithfulness")
ax.set_xlim(-0.02, 1.05)
ax.legend(loc="lower right", frameon=True, framealpha=0.9)

plt.savefig("paper/figures/mib_faithfulness_curves.pdf")
plt.savefig("paper/figures/mib_faithfulness_curves.png")
print("Saved paper/figures/mib_faithfulness_curves.{pdf,png}")
