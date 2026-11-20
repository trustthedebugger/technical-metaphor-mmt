"""CoMMuTE pairs a run gets right with both images (the image decided the translation).
Usage: python commute_examples.py RESULTS_DIR RUN [BASELINE]"""
import glob, os, sys
import numpy as np
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commute_text")
rd = lambda f: open(os.path.join(H, f), encoding="utf8").read().rstrip("\n").split("\n")
src, cor, inc, img = rd("src.en"), rd("correct.fr"), rd("incorrect.fr"), rd("img.order")
def ok(run):
    d = glob.glob(f"{sys.argv[1]}/*/runs/{run}")[0]
    return np.loadtxt(d + "/commute_incorrect.txt") > np.loadtxt(d + "/commute_correct.txt")
run, base = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "text_s1")
a, b = ok(run), ok(base)
print(f"# {run}: pairs right with both images (baseline {base})\n")
for k in range(0, len(src), 2):
    if a[k] and a[k + 1]:
        tag = "baseline also both right" if b[k] and b[k + 1] else f"baseline right on {int(b[k]) + int(b[k+1])}/2"
        print(f"- **{src[k]}** ({tag})\n  - image {img[k]}: {cor[k]}\n  - image {img[k+1]}: {cor[k+1]}")
print(f"\nPairs both wrong for {run}:")
for k in range(0, len(src), 2):
    if not a[k] and not a[k + 1]:
        print(f"- {src[k]} | chose: {inc[k]} / {inc[k+1]}")
