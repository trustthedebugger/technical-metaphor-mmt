"""Technical metaphor test: paired image test, contrastive accuracy, MTA and VSS per system.

Items come in pairs (same English sentence, technical and literal image). For each run:
  contrastive accuracy   % of items whose correct translation has the lower per-token NLL
  both right / wrong     pairs right (wrong) with both images; only the image can make these differ
  MTA                    % of generated translations containing an accepted rendering (tgt_span)
                         and no excluded phrase (tgt_not), with the correct image
  VSS                    MTA(correct images) - MTA(shuffled images)
  pairs MTA both         pairs where the translation is right for both images (needs the image)

Main result: the 41 terms without a near-duplicate pool image (rule fixed before any result; see
the paper, Section 5.4). Pooled counts use an exact two-sided sign test.

Usage: python tech_stats.py RUNS_TECH_DIR
"""
import csv
import glob
import os
import sys
from math import comb

import numpy as np

DUP_TERMS = {"001", "003", "006", "011", "018", "019", "020", "022", "030", "047"}
ORDER = ["mbart", "early", "text", "full", "concat", "fixed", "no_met", "no_con", "no_aug", "no_cur", "no_ret",
         "selective", "sel_met", "sel_con", "sel_cur", "sel_aug", "sel_both", "sel_met_nr",
         "sm_mem", "sm_mask", "sm_mm", "sm_mbe", "sm_all", "mm50", "mm50_mem", "mm75", "rm10", "rm10_mem", "mm50_rm10",
         "text_pt", "rm10_mem_pt", "mbart_pt", "early_pt"]


def sign_p(r, w):
    n = r + w
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(r, w) + 1)) / 2 ** n) if n else 1.0


def lines(path):
    with open(path, encoding="utf8") as f:
        return f.read().splitlines()


def mta_hits(items, hyps):
    out = []
    for it, h in zip(items, hyps):
        h = h.lower()
        ok = any(v.strip().lower() in h for v in it["tgt_span"].split("|") if v.strip())
        bad = any(v.strip().lower() in h for v in (it.get("tgt_not") or "").split("|") if v.strip())
        out.append(int(ok and not bad))
    return np.array(out)


def analyse(run, items, keep):
    r = {}
    c, i = np.loadtxt(f"{run}/tech_correct.txt"), np.loadtxt(f"{run}/tech_incorrect.txt")
    right_item = (c < i)[keep]
    x, y = right_item[0::2], right_item[1::2]
    r["acc"] = 100 * right_item.mean()
    r["right"], r["wrong"] = int((x & y).sum()), int((~x & ~y).sum())
    sel = [it for it, k in zip(items, keep) if k]
    hc = mta_hits(sel, [h for h, k in zip(lines(f"{run}/tech_correct.fr"), keep) if k])
    r["mta"] = 100 * hc.mean()
    r["mta_pairs"] = int((hc[0::2] & hc[1::2]).sum())
    if os.path.exists(f"{run}/tech_shuffled.fr"):
        hs = mta_hits(sel, [h for h, k in zip(lines(f"{run}/tech_shuffled.fr"), keep) if k])
        r["vss"] = r["mta"] - 100 * hs.mean()
    else:
        r["vss"] = 0.0  # no image input: VSS is 0 by construction
    return r


def main(root):
    with open(os.path.join(root, "tech.tsv"), encoding="utf8") as f:
        items = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    assert all(a["id"][:3] == b["id"][:3] for a, b in zip(items[0::2], items[1::2]))
    subsets = {"41 terms without pool duplicates (main)": [it["id"][:3] not in DUP_TERMS for it in items],
               "all 51 terms": [True] * len(items)}
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*_s[0-9]"))):
        if os.path.exists(f"{d}/tech_correct.txt"):
            name = os.path.basename(d)
            runs.setdefault(name.rsplit("_s", 1)[0], []).append(d)
    for title, keep in subsets.items():
        n_terms = sum(keep) // 2
        print(f"\n## Technical test, {title}\n")
        print(f"Chance on contrastive accuracy is 50%. Both right / both wrong: pairs of the {n_terms} terms, pooled over seeds."
              " MTA pairs: terms translated correctly with both images (impossible without the image).\n")
        print("| System | Seeds | Contrastive acc. % | Both right / both wrong | Sign-test p | MTA % | MTA pairs (pooled) | VSS |")
        print("|---|---|---|---|---|---|---|---|")
        for system in [s for s in ORDER if s in runs] + sorted(set(runs) - set(ORDER)):
            rs = [analyse(d, items, keep) for d in runs[system]]
            seeds = ",".join(d.rsplit("_s", 1)[1] for d in runs[system])
            R, W = sum(r["right"] for r in rs), sum(r["wrong"] for r in rs)

            def ms(k):
                v = [r[k] for r in rs]
                return f"{v[0]:.1f}" if len(v) == 1 else f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}"
            p = sign_p(R, W)
            print(f"| {system} | {seeds} | {ms('acc')} | {R}/{W} | {'< 0.001' if p < 0.001 else f'{p:.3f}'} | {ms('mta')} | "
                  f"{sum(r['mta_pairs'] for r in rs)} | {ms('vss')} |")


if __name__ == "__main__":
    main(sys.argv[1])
