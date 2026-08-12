"""Generate Figure: energy spectrum (order-1/2/3+) across tasks.

Reads all_walsh_results.csv. Produces a grouped bar chart showing
order-1, order-2, and order-3+ energy fractions by task, excluding
random circuits.

Usage: cd epistatic-circuits && python3 scripts/plot_energy_spectrum.py
"""
import csv
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

with open("results/all_walsh_results.csv") as f:
    rows = list(csv.DictReader(f))

task_order = ["IOI", "RTI", "greater_than", "induction", "sva", "gendered_pronoun"]
task_labels = ["IOI", "RTI", "Greater-than", "Induction", "SVA", "Gendered\npronoun"]

o1_means, o2_means, o3_means = [], [], []
o2_stds = []

for task in task_order:
    real = [r for r in rows
            if r["task"] == task and "random" not in r["circuit"].lower()]
    o1 = [float(r["order1_frac"]) for r in real]
    o2 = [float(r["order2_frac"]) for r in real]
    o3 = [float(r["order3plus_frac"]) for r in real]
    o1_means.append(np.mean(o1))
    o2_means.append(np.mean(o2))
    o3_means.append(np.mean(o3))
    o2_stds.append(np.std(o2))

x = np.arange(len(task_order))
width = 0.25

fig, ax = plt.subplots(figsize=(5.5, 3))

bars1 = ax.bar(x - width, o1_means, width, label="Order 1", color="C0", alpha=0.85)
bars2 = ax.bar(x, o2_means, width, label="Order 2", color="C1", alpha=0.85,
               yerr=o2_stds, capsize=2, error_kw={"linewidth": 0.8})
bars3 = ax.bar(x + width, o3_means, width, label="Order 3+", color="C2", alpha=0.85)

ax.set_ylabel("Fraction of total energy")
ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", frameon=True, framealpha=0.9)

for i, (m, s) in enumerate(zip(o2_means, o2_stds)):
    if m >= 0.05:
        ax.text(i, m + s + 0.02, "{:.0f}%".format(m * 100),
                ha="center", fontsize=7, color="C1")

plt.savefig("paper/figures/energy_spectrum.pdf")
plt.savefig("paper/figures/energy_spectrum.png")
print("Saved paper/figures/energy_spectrum.{pdf,png}")
