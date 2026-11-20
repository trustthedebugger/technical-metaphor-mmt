"""Paired CoMMuTE analysis: does the image move a run's scores towards the correct translation?

CoMMuTE lines come in adjacent pairs: the same English sentence with two images and the two French
translations swapped. A model that ignores the image gets one of each pair right, so:
  both_right / both_wrong  pairs where the image changed the choice (exact two-sided sign test)
  image_effect             mean per-pair score margin in nats per token; the text preference cancels out
  changed_%                Multi30K translations that differ under a shuffled image

Usage: python commute_pairs.py RUNS_DIR [RUNS_DIR ...]
"""
import glob
import os
import sys
from math import comb

import numpy as np


def read_lines(path):
    with open(path, encoding="utf8") as f:
        return f.read().splitlines()


def analyse(run):
    margin = np.loadtxt(f"{run}/commute_incorrect.txt") - np.loadtxt(f"{run}/commute_correct.txt")
    x, y = margin[0::2], margin[1::2]
    right, wrong = int(((x > 0) & (y > 0)).sum()), int(((x < 0) & (y < 0)).sum())
    n = right + wrong
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(right, wrong) + 1)) / 2 ** n) if n else 1.0
    effect = (x + y) / 2
    row = dict(acc=100 * (margin > 0).mean(), right=right, wrong=wrong, p=p, effect=effect.mean(),
               t=effect.mean() / (effect.std(ddof=1) / np.sqrt(len(effect))), changed=None, met=None, lit=None)
    if os.path.exists(f"{run}/test2016_shuffled.fr"):
        same = zip(read_lines(f"{run}/test2016_correct.fr"), read_lines(f"{run}/test2016_shuffled.fr"))
        row["changed"] = 100 * np.mean([a != b for a, b in same])
    if os.path.exists(f"{run}/gate_values.txt"):
        gate, label = np.loadtxt(f"{run}/gate_values.txt"), np.loadtxt(f"{run}/gate_labels.txt")
        row["met"], row["lit"] = gate[label == 1].mean(), gate[label == 0].mean()
    return row


def show(value, fmt):
    return "-" if value is None else format(value, fmt)


print("| run | accuracy % | both right | both wrong | sign-test p | image effect | t | changed % | gate metaphor | gate other |")
print("|---|---|---|---|---|---|---|---|---|---|")
for root in sys.argv[1:]:
    for run in sorted(glob.glob(os.path.join(root, "*_s*"))):
        if os.path.exists(f"{run}/commute_correct.txt"):
            r = analyse(run)
            print(f"| {os.path.basename(run)} | {r['acc']:.2f} | {r['right']} | {r['wrong']} | {r['p']:.3f} | {r['effect']:+.4f} "
                  f"| {r['t']:+.2f} | {show(r['changed'], '.0f')} | {show(r['met'], '.3f')} | {show(r['lit'], '.3f')} |")
