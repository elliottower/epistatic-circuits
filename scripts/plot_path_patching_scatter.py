"""Generate Figure: Walsh interaction vs path-patch direct effect scatter.

Reads Phase 2 path patching results. Produces a scatter plot colored
by layer distance, with marginal histograms.

Usage: cd epistatic-circuits && python3 scripts/plot_path_patching_scatter.py
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

with open("results/phase2/path_patching_results.json") as f:
    data = json.load(f)

pairs = [p for p in data["all_pairs"] if not p["same_layer"]]

walsh = np.array([p["walsh_coeff"] for p in pairs])
pp = np.array([p["direct_effect"] for p in pairs])
dist = np.array([p["layer_distance"] for p in pairs])

fig, ax = plt.subplots(figsize=(4.5, 4))

sc = ax.scatter(walsh, pp, c=dist, cmap="viridis", s=18, alpha=0.7,
                edgecolors="white", linewidths=0.3, vmin=1, vmax=10)

ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

ax.set_xlabel("Walsh interaction coefficient $w_{ij}$")
ax.set_ylabel("Path-patch direct effect (logit diff change)")

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Layer distance", fontsize=8)

rho_sp = data["correlations"]["abs_spearman"]
r_signed = data["correlations"]["signed_pearson"]
ax.text(0.03, 0.97,
        "Spearman $\\rho = {:.2f}$\nSigned Pearson $r = {:.2f}$".format(
            rho_sp, r_signed),
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.savefig("paper/figures/path_patching_scatter.pdf")
plt.savefig("paper/figures/path_patching_scatter.png")
print("Saved paper/figures/path_patching_scatter.{pdf,png}")
