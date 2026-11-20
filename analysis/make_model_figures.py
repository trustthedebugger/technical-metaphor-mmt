"""Figures 1 (architecture) and 2 (workflow) of the revised manuscript.
Usage: python make_model_figures.py OUT_DIR"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

out = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

TEXT, IMG, FUSE, DEC, AUX = "#dbe8f6", "#e3f1df", "#fdeccf", "#eadff3", "#eeeeee"


def box(ax, x, y, w, h, text, color, bold=False, dashed=False, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444444", lw=1.0, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, dashed=False, color="#333333", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.0, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


# ---------------- Figure 1: architecture ----------------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

box(ax, 2.2, 9.3, 3.4, 0.7, "English source sentence\n\"The bottleneck is too narrow.\"", "white", fs=8)
box(ax, 7.8, 9.3, 3.4, 0.7, "Image\n(photograph, diagram or screenshot)", "white", fs=8)
box(ax, 2.2, 8.15, 3.4, 0.55, "Word hiding: 10% of words → [MASK]\n(training only)", AUX, dashed=True, fs=7.5)
box(ax, 2.2, 7.05, 3.4, 0.75, "BERT encoder (bert-base-cased)\nH_BERT  (L × 768)", TEXT, bold=False)
box(ax, 7.8, 7.05, 3.4, 0.75, "CLIP ViT-B/32 (frozen)\nV  (50 × 768)", IMG)
box(ax, 2.2, 5.9, 2.6, 0.55, "H_t = H_BERT W_t", TEXT)
box(ax, 7.8, 5.9, 2.6, 0.55, "H_v = V W_v", IMG)
box(ax, 5.0, 4.85, 3.8, 0.6, "Cross-modal attention (4 heads)\nA = MultiHead(H_t, H_v, H_v)", FUSE)
box(ax, 5.0, 3.85, 3.8, 0.6, "Gate  G = σ(MLP([H_t ; A]))\nper token and feature", FUSE, bold=True)
box(ax, 5.0, 2.85, 3.8, 0.6, "Fusion  O = (1 − G) ⊙ H_t + G ⊙ A", FUSE, bold=True)
box(ax, 1.5, 2.85, 2.4, 0.75, "Metaphor classifier\non O, loss L_met\n(training only)", AUX, dashed=True, fs=7.5)
box(ax, 5.0, 1.85, 3.8, 0.55, "M = LayerNorm(O W_o)   (L × 1024)", DEC)
box(ax, 8.6, 1.85, 2.2, 0.75, "Image memory\nLayerNorm(H_v W_o)", DEC, fs=8)
box(ax, 5.0, 0.75, 5.6, 0.65, "mBART-50 decoder (cross-attends to [M ; image memory])\n→ \"Le goulot d'étranglement est trop étroit.\"",
    DEC, bold=False, fs=8.5)

arrow(ax, (2.2, 8.95), (2.2, 8.43))
arrow(ax, (2.2, 7.87), (2.2, 7.43))
arrow(ax, (7.8, 8.95), (7.8, 7.43))
arrow(ax, (2.2, 6.67), (2.2, 6.18))
arrow(ax, (7.8, 6.67), (7.8, 6.18))
arrow(ax, (2.6, 5.62), (4.1, 5.15))   # H_t -> attention (query)
arrow(ax, (7.4, 5.62), (5.9, 5.15))   # H_v -> attention (key/value)
arrow(ax, (5.0, 4.55), (5.0, 4.15))
arrow(ax, (1.6, 5.62), (3.15, 3.95), rad=0.25)   # H_t -> gate
arrow(ax, (1.4, 5.62), (3.15, 2.85), rad=0.3)    # H_t -> fusion
arrow(ax, (5.0, 3.55), (5.0, 3.15))
arrow(ax, (3.1, 2.85), (2.7, 2.85), dashed=True)
arrow(ax, (5.0, 2.55), (5.0, 2.12))
arrow(ax, (8.4, 5.62), (8.6, 2.23), rad=-0.15)    # H_v -> image memory
arrow(ax, (5.0, 1.58), (5.0, 1.08))
arrow(ax, (8.6, 1.47), (7.0, 1.08))

ax.text(0.15, 4.3, "text", color="#1f4e79", fontsize=8, rotation=90)
ax.text(9.75, 4.3, "image", color="#2e6b2e", fontsize=8, rotation=90)
fig.tight_layout()
fig.savefig(os.path.join(out, "Figure1_architecture.png"), dpi=300)
plt.close(fig)

# ---------------- Figure 2: workflow ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.3)
ax.axis("off")

cols = [1.25, 3.75, 6.25, 8.75]
heads = ["1. Data", "2. Text-only\nfirst stage", "3. Multimodal\ntraining", "4. Evaluation"]
for x, h in zip(cols, heads):
    ax.text(x, 9.75, h, ha="center", va="center", fontsize=9.5, fontweight="bold")

box(ax, cols[0], 8.0, 2.3, 1.5, "Multi30K\n29,000 pairs\nwith photographs", "white", fs=8)
box(ax, cols[0], 5.9, 2.3, 2.3, "KDE4 + GNOME\n80,405 pairs\n→ 12,000 train\n→ 1,000 test\n→ 1,000 validation\n→ 66,405 first stage", "white", fs=8)
box(ax, cols[0], 3.55, 2.3, 1.4, "Image retrieval (CLIP)\nfrom 5,679 Wikimedia\ntechnical images", IMG, fs=8)
box(ax, cols[0], 1.75, 2.3, 1.4, "Silver metaphor labels\n(VUA detector +\n69-term lexicon)", AUX, fs=8)

box(ax, cols[1], 6.6, 2.3, 2.6, "BERT + projection\n+ mBART-50 decoder\n(baselines: mBART-50)\n\n2 epochs on 66,405\nsoftware pairs, no image", TEXT, fs=8)
box(ax, cols[1], 3.9, 2.3, 1.4, "Checkpoint chosen on\nsoftware validation\nBLEU", TEXT, fs=8)

box(ax, cols[2], 6.6, 2.3, 2.6, "Gate, attention, image\nprojection, classifier:\nnew initialisation\n\nL_MT + 0.5 L_met\nword hiding p = 0.10", FUSE, fs=8)
box(ax, cols[2], 3.9, 2.3, 1.4, "≤ 8 epochs; checkpoint\non Multi30K validation\nBLEU (patience 2)", FUSE, fs=8)
box(ax, cols[2], 1.75, 2.3, 1.2, "Seeds 1, 2, 3\n(every system)", "white", fs=8)

box(ax, cols[3], 8.0, 2.3, 1.6, "Test2016: BLEU, chrF++\ncorrect / shuffled /\nblank image", DEC, fs=8)
box(ax, cols[3], 6.05, 2.3, 1.0, "Software test:\nBLEU, chrF++", DEC, fs=8)
box(ax, cols[3], 4.35, 2.3, 1.4, "CoMMuTE: contrastive\naccuracy, paired\nimage test", DEC, fs=8)
box(ax, cols[3], 2.35, 2.3, 1.6, "Technical test (41 + 10\nterms): contrastive acc.,\npaired test, MTA, VSS", DEC, fs=8)
box(ax, cols[3], 0.75, 2.3, 0.75, "Sign tests, paired bootstrap", "white", fs=7.5)

arrow(ax, (2.4, 5.9), (2.6, 6.6))
arrow(ax, (cols[1], 5.3), (cols[1], 4.6))
arrow(ax, (4.9, 3.9), (5.1, 5.9))
# the whole data column feeds the multimodal training: one bracket, one arrow above the first stage
ax.plot([2.48, 2.55, 2.55, 2.48], [1.1, 1.1, 8.85, 8.85], color="#333333", lw=1.0)
ax.plot([2.55, 2.55], [8.85, 8.98], color="#333333", lw=1.0)
ax.plot([2.55, 6.25], [8.98, 8.98], color="#333333", lw=1.0)
arrow(ax, (6.25, 8.98), (6.25, 7.95))
ax.text(4.4, 9.12, "training data with images and labels", ha="center", va="bottom", fontsize=7.5, color="#333333")
arrow(ax, (cols[2], 5.3), (cols[2], 4.6))
arrow(ax, (7.4, 3.9), (7.6, 4.35))
arrow(ax, (7.4, 3.9), (7.6, 6.05))
arrow(ax, (7.4, 3.9), (7.6, 8.0))
arrow(ax, (7.4, 3.9), (7.6, 2.35))
fig.tight_layout()
fig.savefig(os.path.join(out, "Figure2_workflow.png"), dpi=300)
plt.close(fig)
print("saved Figure1_architecture.png, Figure2_workflow.png")
