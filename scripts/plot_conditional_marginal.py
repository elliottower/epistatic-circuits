"""Generate Figure: conditional marginal contribution slope chart.

Shows each head's marginal contribution with S-inhibition retained
vs removed. The two negative name-mover heads cross zero (sign inversion);
name movers decay toward zero without inverting. This is the money figure.

Usage: cd epistatic-circuits && python3 scripts/plot_conditional_marginal.py
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
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open("results/v13_conditional_marginal.json") as f:
    data = json.load(f)

ROLE_COLORS = {
    "NegNM": "#d62728",
    "NM": "#1f77b4",
    "IND": "#2ca02c",
    "DTH": "#9467bd",
    "Backup": "#8c564b",
}

ROLE_LABELS = {
    "NegNM": "Negative name mover",
    "NM": "Name mover",
    "IND": "Induction",
    "DTH": "Duplicate token",
    "Backup": "Backup name mover",
}

fig, ax = plt.subplots(figsize=(4.5, 4))

ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

plotted_roles = set()
for row in data["rows"]:
    role, head, retained, removed, delta, flips = row
    lh = "L{}H{}".format(head[0], head[1])
    color = ROLE_COLORS.get(role, "#7f7f7f")

    label = ROLE_LABELS.get(role, role) if role not in plotted_roles else None
    plotted_roles.add(role)

    lw = 2.5 if role in ("NegNM", "NM") else 1.2
    alpha = 1.0 if role in ("NegNM", "NM") else 0.5

    ax.plot([0, 1], [retained, removed], "o-", color=color,
            linewidth=lw, alpha=alpha, markersize=5, label=label,
            zorder=10 if role in ("NegNM", "NM") else 5)

    x_off = -0.08 if retained > removed else 1.08
    ha = "right" if x_off < 0.5 else "left"
    y_val = retained if x_off < 0.5 else removed
    fontsize = 7.5 if role in ("NegNM", "NM") else 6.5
    ax.text(x_off, y_val, lh, fontsize=fontsize, color=color,
            ha=ha, va="center", alpha=max(alpha, 0.7))

ax.set_xticks([0, 1])
ax.set_xticklabels(["S-inhibition\nretained", "S-inhibition\nremoved"])
ax.set_ylabel("Marginal contribution to logit difference")
ax.set_xlim(-0.25, 1.25)

ax.legend(loc="upper left", frameon=True, framealpha=0.9)

plt.savefig("paper/figures/conditional_marginal.pdf")
plt.savefig("paper/figures/conditional_marginal.png")
print("Saved paper/figures/conditional_marginal.{pdf,png}")
