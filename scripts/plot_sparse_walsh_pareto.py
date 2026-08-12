"""Generate Figure: sparse Walsh recovery Pareto curve.

Reads Phase 1 results (15-head) and Phase 2 Pareto curve (20-head).
Produces a two-panel figure:
  Left:  recovery r vs M for 20-head circuit (LASSO, OMP, MC)
  Right: recovery r vs M/k across all circuits (Phase 1 + Phase 2)

Usage: cd epistatic-circuits && python3.12 scripts/plot_sparse_walsh_pareto.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

with open("results/phase2/pareto_curve.json") as f:
    pareto = json.load(f)

with open("results/sparse_walsh_results.json") as f:
    phase1 = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"wspace": 0.35})

# --- Panel A: 20-head Pareto curve ---
Ms = [r["M"] for r in pareto]
lasso_r = [r["lasso_r_mean"] for r in pareto]
lasso_std = [r["lasso_r_std"] for r in pareto]
omp_r = [r["omp_r_mean"] for r in pareto]
omp_std = [r["omp_r_std"] for r in pareto]
mc_r = [r["mc_r_mean"] for r in pareto]
mc_std = [r["mc_r_std"] for r in pareto]

ax1.fill_between(Ms, np.array(lasso_r) - np.array(lasso_std),
                 np.array(lasso_r) + np.array(lasso_std), alpha=0.15, color="C0")
ax1.fill_between(Ms, np.array(omp_r) - np.array(omp_std),
                 np.array(omp_r) + np.array(omp_std), alpha=0.15, color="C1")
ax1.fill_between(Ms, np.array(mc_r) - np.array(mc_std),
                 np.array(mc_r) + np.array(mc_std), alpha=0.15, color="C2")

ax1.plot(Ms, lasso_r, "o-", color="C0", markersize=3, linewidth=1.2, label="LASSO")
ax1.plot(Ms, omp_r, "s-", color="C1", markersize=3, linewidth=1.2, label="OMP")
ax1.plot(Ms, mc_r, "^-", color="C2", markersize=3, linewidth=1.2, label="Monte Carlo")

ax1.axhline(0.95, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax1.text(22, 0.955, "$r = 0.95$", fontsize=7, color="gray", alpha=0.8)

ax1.axvline(210, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax1.text(215, 0.15, "$M = k$", fontsize=7, color="gray", alpha=0.8, rotation=90)

ax1.set_xscale("log")
ax1.set_xlabel("Number of coalitions ($M$)")
ax1.set_ylabel("Pearson $r$ vs reference")
ax1.set_title("(a)  20-head IOI circuit")
ax1.set_xlim(15, 2500)
ax1.set_ylim(0, 1.05)
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xticks([20, 50, 100, 200, 500, 1000, 2000])
ax1.legend(loc="lower right", frameon=True, framealpha=0.9)

# --- Panel B: M/k universality across circuits ---
circuits_15 = {}
for r in phase1:
    if r["n_heads"] == 15:
        name = f"{r['task']}-{r['abl']}"
        if name not in circuits_15:
            circuits_15[name] = {"mk": [], "r": []}
        circuits_15[name]["mk"].append(r["M_over_k"])
        circuits_15[name]["r"].append(r["lasso_corr_mean"])

for name, data in circuits_15.items():
    ax2.plot(data["mk"], data["r"], "-", color="C3", alpha=0.25, linewidth=0.8)

mk_20 = [r["M_over_k"] for r in pareto]
r_20 = [r["lasso_r_mean"] for r in pareto]
ax2.plot(mk_20, r_20, "o-", color="C0", markersize=3, linewidth=1.5, label="20-head IOI ($k{=}210$)", zorder=5)

ax2.plot([], [], "-", color="C3", alpha=0.5, linewidth=1.2, label="15-head circuits ($k{=}120$)")

ax2.axhline(0.95, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax2.axvline(1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax2.text(1.05, 0.15, "$M/k = 1$", fontsize=7, color="gray", alpha=0.8, rotation=90)

ax2.set_xscale("log")
ax2.set_xlabel("$M / k$  (samples per coefficient)")
ax2.set_ylabel("LASSO Pearson $r$")
ax2.set_title("(b)  Recovery scales with $M/k$")
ax2.set_xlim(0.08, 50)
ax2.set_ylim(0, 1.05)
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xticks([0.1, 0.3, 1, 3, 10, 30])
ax2.legend(loc="lower right", frameon=True, framealpha=0.9)

plt.savefig("paper/figures/sparse_walsh_recovery.pdf")
plt.savefig("paper/figures/sparse_walsh_recovery.png")
print("Saved paper/figures/sparse_walsh_recovery.{pdf,png}")
