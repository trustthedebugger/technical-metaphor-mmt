"""Gate value on metaphor-tagged vs other source tokens: means, AUC and Mann-Whitney p.
AUC = chance a random metaphor token gets a higher gate than a random other token (0.5 = no preference).
Tokens within a sentence are not independent, so p-values are optimistic; AUC is the main number.
Usage: python gate_stats.py RESULTS_DIR"""
import glob, os, sys
import numpy as np
from scipy.stats import mannwhitneyu
print("| run | gate metaphor | gate other | AUC | Mann-Whitney p |\n|---|---|---|---|---|")
for d in sorted(glob.glob(f"{sys.argv[1]}/*/runs/*/gate_values.txt")):
    g, l = np.loadtxt(d), np.loadtxt(d.replace("gate_values", "gate_labels"))
    m, o = g[l == 1], g[l == 0]
    u = mannwhitneyu(m, o, alternative="two-sided")
    print(f"| {os.path.basename(os.path.dirname(d))} | {m.mean():.3f} | {o.mean():.3f} | {u.statistic / (len(m) * len(o)):.3f} | {u.pvalue:.2g} |")
