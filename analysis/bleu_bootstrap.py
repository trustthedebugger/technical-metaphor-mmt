"""Paired bootstrap (Koehn 2004) on Multi30K test2016: system vs system, and correct vs shuffled image.
Usage: python bleu_bootstrap.py RESULTS_DIR   (finds */runs/<name>/test2016_*.fr)"""
import glob, os, random, sys
import sacrebleu
from sacrebleu.metrics import BLEU
HERE = os.path.dirname(os.path.abspath(__file__))
ref = open(os.path.join(HERE, "test2016_ref.fr"), encoding="utf8").read().splitlines()
root = sys.argv[1]
def hyp(run, cond="correct"):
    p = glob.glob(f"{root}/*/runs/{run}/test2016_{cond}.fr")
    return open(p[0], encoding="utf8").read().splitlines() if p else None
bleu = BLEU()
def stats(h):  # sufficient statistics per sentence, so resampling is fast
    return [bleu._extract_corpus_statistics([x], [[r]])[0] for x, r in zip(h, ref)]
def score(st, idx):
    tot = [sum(st[i][k] for i in idx) for k in range(len(st[0]))]
    return bleu._compute_score_from_stats(tot).score
def boot(a, b, n=2000, seed=12345):
    sa, sb = stats(a), stats(b); rng = random.Random(seed); N = len(ref); allidx = range(N)
    da = score(sa, allidx) - score(sb, allidx); worse = 0
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        worse += score(sa, idx) - score(sb, idx) <= 0
    return score(sa, allidx), score(sb, allidx), da, worse / n
pairs = [(a, b) for a, b in [("rm10_mem_s1", "text_s1"), ("sel_met_s1", "text_s1"), ("rm10_mem_s1", "sel_met_s1"),
         ("selective_s1", "text_s1"), ("mbart_s1", "rm10_mem_s1")]]
pairs += [(f"{a}_s{k}", f"{b}_s{k}") for k in (1, 2, 3) for a, b in
          [("rm10_mem_pt", "text_pt"), ("mbart_pt", "rm10_mem_pt"), ("early_pt", "rm10_mem_pt"), ("rm10_mem_pt", "rm10_mem")]]
print("| A | B | BLEU A | BLEU B | A - B | p (A not better) |\n|---|---|---|---|---|---|")
for a, b in pairs:
    ha, hb = hyp(a), hyp(b)
    if ha and hb:
        x, y, d, p = boot(ha, hb); print(f"| {a} | {b} | {x:.2f} | {y:.2f} | {d:+.2f} | {p:.3f} |")
print("\nImage effect on BLEU (correct image vs shuffled image, same run)\n")
print("| run | correct | shuffled | diff | p (correct not better) |\n|---|---|---|---|---|")
for d in sorted(glob.glob(f"{root}/*/runs/*/test2016_shuffled.fr")):
    run = os.path.basename(os.path.dirname(d))
    if run.split("_s")[0] in {"rm10_mem", "sel_met", "selective", "full", "rm10", "sm_mm", "mm50_rm10", "rm10_mem_pt", "early", "early_pt"}:
        x, y, df, p = boot(hyp(run), hyp(run, "shuffled"), n=1000)
        print(f"| {run} | {x:.2f} | {y:.2f} | {df:+.2f} | {p:.3f} |")
