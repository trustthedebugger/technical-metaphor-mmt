"""Paper figures from ../results/*/runs. Rerun after new results arrive.
Fig. 3: CoMMuTE pairs both right vs both wrong, per seed, for the main systems.
Fig. 4: gate preference for metaphor tokens (AUC), per seed."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

runs = {}
for p in glob.glob(f"{ROOT}/*/runs/*/config.json"):
    c = json.load(open(p))
    runs.setdefault(c["system"], {})[c["seed"]] = os.path.dirname(p)

MAIN = [("text", "Text-only"), ("mbart", "mBART-50"), ("full", "Original full"), ("selective", "Gate only"),
        ("sel_met", "Gate + metaphor loss"), ("rm10_mem", "+ word hiding,\ndecoder sees image")]


def pairs(d):
    m = np.loadtxt(f"{d}/commute_incorrect.txt") - np.loadtxt(f"{d}/commute_correct.txt")
    x, y = m[0::2], m[1::2]
    return int(((x > 0) & (y > 0)).sum()), int(((x < 0) & (y < 0)).sum())


# ---- Fig. 3
labels, right, wrong = [], [], []
for s, name in MAIN:
    for seed in sorted(runs.get(s, {})):
        r, w = pairs(runs[s][seed])
        labels.append(f"{name} (s{seed})"); right.append(r); wrong.append(w)
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(6.3, 0.28 * len(labels) + 0.9), dpi=300)
h = 0.38
ax.barh(y + h / 2, right, h, color=BLUE, label="Both right (image chose correctly)", edgecolor="white", linewidth=1)
ax.barh(y - h / 2, wrong, h, color=ORANGE, label="Both wrong", edgecolor="white", linewidth=1, hatch="////")
for yy, r, w in zip(y, right, wrong):
    ax.text(r + 0.2, yy + h / 2, str(r), va="center", fontsize=7, color=INK)
    ax.text(w + 0.2, yy - h / 2, str(w), va="center", fontsize=7, color=MUTED)
ax.set_yticks(y, labels, fontsize=7.5)
ax.set_xlabel("CoMMuTE sentence pairs (of 154)")
ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/Figure3_commute_pairs.png"); plt.close(fig)

# ---- Fig. 4
fig, ax = plt.subplots(figsize=(6.3, 2.6), dpi=300)
names, xs = [], []
for i, (s, name) in enumerate([m for m in MAIN if m[0] in ("full", "selective", "sel_met", "rm10_mem")]):
    for seed in sorted(runs.get(s, {})):
        d = runs[s][seed]
        g, l = np.loadtxt(f"{d}/gate_values.txt"), np.loadtxt(f"{d}/gate_labels.txt")
        auc = mannwhitneyu(g[l == 1], g[l == 0]).statistic / ((l == 1).sum() * (l == 0).sum())
        ax.scatter(i, auc, s=36, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(i + 0.08, auc, f"s{seed}", fontsize=7, color=MUTED, va="center")
    names.append(name)
ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
ax.text(len(names) - 0.55, 0.51, "no preference", fontsize=7, color=MUTED, ha="right", va="bottom")
ax.set_xticks(range(len(names)), names, fontsize=7.5)
ax.set_ylabel("Gate AUC (metaphor\nvs other words)")
ax.set_ylim(0.2, 1.0); ax.set_xlim(-0.5, len(names) - 0.5)
ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/Figure4_gate_auc.png"); plt.close(fig)
print("saved Figure3_commute_pairs.png, Figure4_gate_auc.png")
