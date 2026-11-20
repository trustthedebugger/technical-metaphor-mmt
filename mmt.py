"""Gated multimodal MT with metaphor supervision: data preparation, training, evaluation.

Pipeline (each step is a subcommand):
  prep   download Multi30K en-fr text, tag metaphors (silver labels), extract frozen CLIP
         features for every image, retrieve augmentation images, prepare CoMMuTE and the
         optional technical-metaphor test set.
  train  train one system with one seed, then evaluate it (same as running `eval`).
  eval   evaluate a saved checkpoint (e.g. on the technical test set once it exists).
  add_tech  add the technical test set to an existing data folder (encodes only its images).
  build_pool  download technical images from Wikimedia Commons and add their features (CPU is enough).
  add_soft  add KDE4/GNOME software text (no image) to the training data, plus a held-out test set.
  collect  mean ± s.d. over seeds of every metric, as a Markdown table.

Systems (see SYSTEMS): mbart (plain mBART-50 fine-tuning), early (mBART-50 encoder reading CLIP patches
in front of the words: early fusion), text (BERT -> mBART decoder, no
image), concat (O = Ht + A), fixed (gate fixed at 0.5), selective (learned gate only, as in
Li et al. 2022), full (ours), and the ablations no_met, no_con, no_aug, no_cur.

Example:
  python mmt.py prep  --data data --flickr_dir /kaggle/input/flickr-image-dataset --commute_dir CoMMuTE
  python mmt.py train --data data --system full --seed 1 --out runs/full_s1
  python mmt.py collect --runs runs
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

M30K = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1"
SPLITS = {"train": "train", "val": "val", "test2016": "test_2016_flickr"}

# Defaults shared by all systems; each system overrides only what it removes.
DEFAULTS = dict(arch="bert", fusion="gate", lam1=0.5, lam2=0.1, curriculum=2, aug=1, ret=1, mask_p=0.0, mem_img=0, met_mask=0.0)
SEL_MET = dict(lam2=0, curriculum=0, aug=0)  # the model chosen after Round 2: learned gate + metaphor loss
SYSTEMS = {
    "mbart": dict(arch="mbart", fusion="none", lam1=0, lam2=0, curriculum=0, aug=0),
    # Early fusion (baseline of the first submission): the projected CLIP patches are placed in front of the
    # word embeddings and the full mBART-50 encoder reads both. No gate, no auxiliary losses.
    "early": dict(arch="mbart", fusion="early", lam1=0, lam2=0, curriculum=0, aug=0),
    "text": dict(fusion="none", lam1=0, lam2=0, curriculum=0, aug=0),
    "concat": dict(fusion="add"),
    "fixed": dict(fusion="fixed"),
    "selective": dict(lam1=0, lam2=0, curriculum=0, aug=0),
    "full": dict(),
    "no_met": dict(lam1=0),
    "no_con": dict(lam2=0),
    "no_aug": dict(aug=0),
    "no_cur": dict(curriculum=0),
    "no_ret": dict(ret=0),  # software sentences get the blank image instead of a retrieved one
    # Round 2: selective plus one component at a time, to find which ones stop the model using the image.
    "sel_cur": dict(lam1=0, lam2=0, aug=0),
    "sel_aug": dict(lam1=0, lam2=0, curriculum=0),
    "sel_met": dict(lam2=0, curriculum=0, aug=0),
    "sel_con": dict(lam1=0, curriculum=0, aug=0),
    "sel_both": dict(curriculum=0, aug=0),
    "sel_met_nr": dict(SEL_MET, ret=0),  # chosen model without retrieved images for the software text
    # Improvement batch: mask_p hides source words during training, mem_img shows the decoder the image
    # tokens directly, arch="mbe" uses mBART's own encoder in place of BERT.
    "sm_mask": dict(SEL_MET, mask_p=0.25),
    "sm_mem": dict(SEL_MET, mem_img=1),
    "sm_mm": dict(SEL_MET, mask_p=0.25, mem_img=1),
    "sm_mbe": dict(SEL_MET, arch="mbe"),
    "sm_all": dict(SEL_MET, mask_p=0.25, mem_img=1, arch="mbe"),
    # Masking batch (BERT kept): met_mask hides words tagged as metaphors, so the image is needed to
    # recover their sense; mask_p hides any word. Both apply only to training rows with a real image.
    "mm50": dict(SEL_MET, met_mask=0.5),
    "mm50_mem": dict(SEL_MET, met_mask=0.5, mem_img=1),
    "mm75": dict(SEL_MET, met_mask=0.75),
    "rm10": dict(SEL_MET, mask_p=0.10),
    "rm10_mem": dict(SEL_MET, mask_p=0.10, mem_img=1),
    "mm50_rm10": dict(SEL_MET, met_mask=0.5, mask_p=0.10),
    # Bridge pre-training: a text-only first stage on the unused software pairs (soft_extra.jsonl, chosen on
    # soft_val.jsonl), then the normal training started from it (--init). Same architectures as text,
    # rm10_mem, mbart and early; only the starting weights differ.
    "pre": dict(fusion="none", lam1=0, lam2=0, curriculum=0, aug=0),
    "pre_mb": dict(arch="mbart", fusion="none", lam1=0, lam2=0, curriculum=0, aug=0),
    "text_pt": dict(fusion="none", lam1=0, lam2=0, curriculum=0, aug=0),
    "rm10_mem_pt": dict(SEL_MET, mask_p=0.10, mem_img=1),
    "mbart_pt": dict(arch="mbart", fusion="none", lam1=0, lam2=0, curriculum=0, aug=0),
    "early_pt": dict(arch="mbart", fusion="early", lam1=0, lam2=0, curriculum=0, aug=0),
}
# Fusion modules are never taken from a text-only first stage; they start fresh in the second stage.
FRESH = ("attn.", "gate.", "met.", "Wv.", "ln_v.")

# Technical image pool: freely licensed files from these Wikimedia Commons categories (and their
# sub-categories down to POOL_DEPTH). Fixed before any experiment was run; not tuned on results.
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MetaphorMMT-research/1.0 (academic research script)"
POOL_CATEGORIES = ["Screenshots_of_Linux_software", "Computer_network_diagrams", "Network_diagrams",
                   "Data_structures", "Data_centers", "Computer_hardware", "Printed_circuit_boards",
                   "Integrated_circuits"]
POOL_DEPTH = 1
POOL_PER_CATEGORY = 750

OPUS = "https://object.pouta.csc.fi/OPUS-{0}/{1}/moses/en-fr.txt.zip"
SOFT_CORPORA = [("KDE4", "v2"), ("GNOME", "v1")]

# Technical metaphors and their technical French renderings. A word in a software sentence is labelled
# metaphorical when it matches a term AND the French side contains one of the renderings.
# First block: the terms of the technical test set; second block: other common computing metaphors.
TECH_LEXICON = {
    "bottleneck": "goulot d'étranglement",
    "bug": "bogue|bug",
    "crash": "plantage|panne",
    "freeze": "planté|figé|bloqué",
    "key": "touche",
    "chip": "puce",
    "patch": "correctif|patch",
    "shell": "terminal|shell|interpréteur",
    "trash": "corbeille",
    "firewall": "pare-feu",
    "web": "web|internet",
    "stream": "direct|diffusion|stream",
    "feed": "fil d'actualité|flux",
    "thread": "thread|fil d'exécution",
    "cache": "le cache|mémoire cache",
    "driver": "pilote",
    "hub": "concentrateur|hub",
    "backbone": "dorsale|réseau fédérateur",
    "handshake": "négociation|établissement de la connexion",
    "gateway": "passerelle",
    "bucket": "bucket|compartiment",
    "socket": "socket|support",
    "wizard": "assistant",
    "kernel": "noyau",
    "dump": "vidage|dump",
    "clipboard": "presse-papiers",
    "bookmark": "favori",
    "tab": "onglet",
    "log": "journal|log",
    "rack": "baie|rack",
    "spool": "file d'attente|spool",
    "backdoor": "porte dérobée",
    "mailbox": "boîte de réception|boîte mail",
    "stick": "clé usb",
    "wallpaper": "fond d'écran",
    "frame": "trame",
    "nut": "écrou",
    "grid": "réseau électrique|réseau",
    "breadcrumbs": "fil d'ariane",
    "garbage collector": "ramasse-miettes|garbage collector",
    "cradle": "station d'accueil|socle",
    "dock": "dock|barre des applications",
    "breadboard": "plaque d'essai|platine d'essai",
    "pin": "broche",
    "lead": "câble|cordon|fil",
    "wafer": "plaquette|wafer",
    "die": "puce|die",
    "jumper": "cavalier|jumper",
    "jam": "bourrage",
    "tray": "bac",
    "sink": "dissipateur|radiateur",
    "window": "fenêtre",
    "mouse": "souris",
    "desktop": "bureau",
    "folder": "dossier",
    "virus": "virus",
    "worm": "ver",
    "path": "chemin",
    "tree": "arborescence|arbre",
    "branch": "branche",
    "root": "racine",
    "pipe": "tube",
    "port": "port",
    "bridge": "pont",
    "stack": "pile",
    "heap": "tas",
    "tunnel": "tunnel",
    "daemon": "démon",
    "bus": "bus",
}
# Sentences of the technical test set; never allowed into the training data.
TECH_TEST_SOURCES = ['Apply the patch tonight.', 'Check the clipboard.', 'Check the dump.', 'Check the feed.', 'Check the log.', 'Clear the jam.', 'Empty the trash.', "I can't find the jumper.", 'I changed the wallpaper.', 'I found a bug.', 'I lost the bookmark.', 'I lost the stick.', 'It froze again this morning.', 'Open the shell.', 'Pull the tab.', 'The backbone is damaged.', 'The backdoor is open.', 'The bottleneck is too narrow.', 'The breadboard is too small.', 'The breadcrumbs disappeared.', 'The bucket is full.', 'The chip is very thin.', 'The cradle is broken.', 'The crash happened at noon.', 'The die is cracked.', 'The dock is empty.', 'The driver is missing.', 'The firewall was damaged.', 'The frame is damaged.', 'The garbage collector is busy.', 'The gateway is closed.', 'The grid is damaged.', 'The handshake took a long time.', 'The hub is broken.', 'The kernel is damaged.', 'The key is broken.', 'The lead is too short.', 'The mailbox is full.', 'The nut is too small.', 'The pin is bent.', 'The rack is full.', 'The sink is dirty.', 'The socket is damaged.', 'The spool is empty.', 'The thread got stuck.', 'The tray is empty.', 'The wafer is broken.', 'The web is huge.', 'The wizard will guide you.', 'They found the cache.', 'Watch the stream.']

WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def words(s):
    return WORD_RE.findall(s)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_lines(path, lines):
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")


def fetch_lines(url):
    data = urllib.request.urlopen(url).read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    return data.decode("utf8").rstrip("\n").split("\n")


# --------------------------------------------------------------------------- prep

def find_image(root, name, cache={}):
    """Locate an image file anywhere under root (Kaggle datasets nest folders differently)."""
    if root not in cache:
        cache[root] = {}
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            cache[root].setdefault(os.path.basename(p), p)
    return cache[root].get(name)


def tag_metaphors(rows, model_name, o_label, bsz, device):
    """Silver word-level metaphor labels (1 = metaphorical) from a VUAM-trained tagger."""
    if model_name == "dummy":  # smoke tests only
        rng = random.Random(0)
        for r in rows:
            r["met"] = [int(rng.random() < 0.1) for _ in r["words"]]
        return
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device).eval()
    for i in range(0, len(rows), bsz):
        chunk = rows[i:i + bsz]
        enc = tok([r["words"] for r in chunk], is_split_into_words=True, truncation=True,
                  max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = model(**{k: v.to(device) for k, v in enc.items()}).logits.argmax(-1).cpu()
        for b, r in enumerate(chunk):
            lab, seen = [0] * len(r["words"]), set()
            for t, w in enumerate(enc.word_ids(b)):
                if w is not None and w not in seen:  # first sub-token decides the word
                    seen.add(w)
                    lab[w] = int(pred[b, t].item() != o_label)
            r["met"] = lab


def read_tech(path):
    """Technical-metaphor test set: TSV with id, src, ref, incorrect, image, met_span, tgt_span (+ notes).
    Optional tgt_not: phrases that must not appear for the automatic check to count a hit."""
    rows = []
    with open(path, encoding="utf8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            if line.strip():
                r = dict(zip(head, line.rstrip("\n").split("\t")))
                r["words"] = words(r["src"])
                rows.append(r)
    print("technical test set", len(rows), "items")
    return rows


class ImageFiles(torch.utils.data.Dataset):
    def __init__(self, names, paths, proc, size):
        self.names, self.paths, self.proc, self.size = names, paths, proc, size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        from PIL import Image
        n = self.names[i]
        im = Image.new("RGB", (self.size, self.size), "white") if n == "<blank>" else Image.open(self.paths[n]).convert("RGB")
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


def encode_images(clip, proc, names, paths, feats, glob_emb, offset, bsz, workers, device):
    """Frozen CLIP patch states (N x dv) and normalised global embeddings, written from row `offset`."""
    data = ImageFiles(names, paths, proc, clip.config.vision_config.image_size)
    loader = torch.utils.data.DataLoader(data, batch_size=bsz, num_workers=workers)
    t0, i = time.time(), offset
    for k, px in enumerate(loader):
        with torch.no_grad():
            out = clip.vision_model(pixel_values=px.to(device))
            g = clip.visual_projection(out.pooler_output)
        feats[i:i + len(px)] = out.last_hidden_state.half().cpu().numpy()
        if glob_emb is not None:
            glob_emb[i:i + len(px)] = F.normalize(g, dim=-1).cpu().numpy()
        i += len(px)
        if k % 50 == 0:
            print(f"  CLIP {i - offset}/{len(names)} images, {time.time() - t0:.0f}s")


def cmd_prep(a):
    os.makedirs(a.data, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {}
    for split, name in SPLITS.items():
        src = fetch_lines(f"{M30K}/raw/{name}.en.gz")
        tgt = fetch_lines(f"{M30K}/raw/{name}.fr.gz")
        imgs = fetch_lines(f"{M30K}/image_splits/{name}.txt")
        assert len(src) == len(tgt) == len(imgs), split
        n = a.limit or len(src)
        splits[split] = [dict(src=s, tgt=t, image=i, words=words(s)) for s, t, i in zip(src, tgt, imgs)][:n]
        print(split, len(splits[split]), "pairs")

    rows = [r for s in splits.values() for r in s]
    print("tagging metaphors with", a.tagger)
    tag_metaphors(rows, a.tagger, a.tagger_o_label, 64, device)
    rate = sum(map(sum, (r["met"] for r in rows))) / sum(len(r["met"]) for r in rows)
    print(f"silver metaphor rate: {100 * rate:.1f}% of words")

    # CoMMuTE en-fr: every line has a source, correct/incorrect translations and its image.
    commute = []
    if a.commute_dir:
        d = os.path.join(a.commute_dir, "en-fr")
        cols = [open(os.path.join(d, f), encoding="utf8").read().rstrip("\n").split("\n")
                for f in ("src.en", "correct.fr", "incorrect.fr", "img.order")]
        commute = [dict(src=s, correct=c, incorrect=i, image=m, words=words(s)) for s, c, i, m in zip(*cols)]
        print("CoMMuTE en-fr", len(commute), "lines")

    tech = read_tech(a.tech_tsv) if a.tech_tsv else []  # optional technical-metaphor test set

    # Frozen CLIP features for every image: patch states (N x dv) and a global embedding.
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(a.clip).to(device).eval()
    proc = CLIPProcessor.from_pretrained(a.clip)
    paths = {}
    for r in rows:
        paths[r["image"]] = find_image(a.flickr_dir, r["image"])
    for r in commute:
        paths["commute/" + r["image"]] = find_image(a.commute_images, r["image"])
    for r in tech:
        paths["tech/" + r["image"]] = find_image(a.tech_images, r["image"])
    missing = [k for k, p in paths.items() if p is None]
    assert not missing, f"{len(missing)} images not found, e.g. {missing[:3]}"
    names = sorted(paths) + ["<blank>"]
    index = {n: i for i, n in enumerate(names)}
    vc = clip.config.vision_config
    n_tok = (vc.image_size // vc.patch_size) ** 2 + 1
    feats = np.lib.format.open_memmap(os.path.join(a.data, "feats.npy"), "w+", np.float16,
                                      (len(names), n_tok, vc.hidden_size))
    glob_emb = np.zeros((len(names), clip.config.projection_dim), np.float32)
    encode_images(clip, proc, names, paths, feats, glob_emb, 0, a.clip_bsz, a.workers, device)
    feats.flush()
    json.dump(dict(index=index, n_tok=n_tok, dim=vc.hidden_size), open(os.path.join(a.data, "feats.json"), "w"))

    # Visual augmentation: for each training sentence, the k most similar *training* images
    # (CLIP text-image cosine), excluding its own. Val/test images never enter this pool.
    train = splits["train"]
    pool = sorted({r["image"] for r in train})
    pool_emb = torch.tensor(glob_emb[[index[n] for n in pool]], device=device)
    for i in range(0, len(train), 256):
        chunk = train[i:i + 256]
        tok = proc(text=[r["src"] for r in chunk], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            pooled = clip.text_model(**{k: v.to(device) for k, v in tok.items()}).pooler_output
            t = F.normalize(clip.text_projection(pooled), dim=-1)
        top = (t @ pool_emb.T).topk(a.aug_k + 1, dim=-1).indices.cpu().tolist()
        for r, cand in zip(chunk, top):
            r["aug"] = [pool[j] for j in cand if pool[j] != r["image"]][:a.aug_k]

    for split, s in splits.items():
        write_jsonl(os.path.join(a.data, f"{split}.jsonl"), s)
    if commute:
        for r in commute:
            r["image"] = "commute/" + r["image"]
        write_jsonl(os.path.join(a.data, "commute.jsonl"), commute)
    if tech:
        for r in tech:
            r["image"] = "tech/" + r["image"]
        write_jsonl(os.path.join(a.data, "tech.jsonl"), tech)
    print("prep done")


SLOWDOWN = [0.0, 0]  # extra seconds between requests (raised after each HTTP 429), successes since then


def http_get(url, tries=12, pause=0.0):
    """GET with backoff that honours Retry-After (Wikimedia rate-limits with HTTP 429). After a 429 all
    requests slow down, then speed up again by 0.5 s per 100 successes. Network errors are retried for
    about 8 minutes, so a short outage does not end an hour-long download."""
    for k in range(tries):
        try:
            time.sleep(pause + SLOWDOWN[0])
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            body = urllib.request.urlopen(req, timeout=60).read()
            SLOWDOWN[1] += 1
            if SLOWDOWN[0] and SLOWDOWN[1] % 100 == 0:
                SLOWDOWN[0] = max(SLOWDOWN[0] - 0.5, 0.0)
            return body
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
            wait = min(float(e.headers.get("Retry-After") or 2 ** k), 600)
            if e.code == 429:
                SLOWDOWN[0], SLOWDOWN[1] = min(SLOWDOWN[0] + 0.5, 5.0), 0
                print(f"  Wikimedia asked us to slow down: waiting {wait:.0f}s, then "
                      f"{pause + SLOWDOWN[0]:.1f}s between requests", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
            time.sleep(min(2 ** k, 60))


def commons_query(pause, **q):
    import urllib.parse
    q.update(action="query", format="json")
    return json.loads(http_get(COMMONS_API + "?" + urllib.parse.urlencode(q), pause=pause))


def commons_files(categories, depth, pause):
    """File titles in the categories and their sub-categories (breadth-first, up to `depth`)."""
    files, seen, frontier = set(), set(categories), list(categories)
    for level in range(depth + 1):
        nxt = []
        for cat in frontier:
            for kind in ("file", "subcat") if level < depth else ("file",):
                cont = {}
                while True:
                    d = commons_query(pause, list="categorymembers", cmtitle="Category:" + cat, cmtype=kind,
                                      cmlimit="500", **cont)
                    for m in d["query"]["categorymembers"]:
                        if kind == "file":
                            files.add(m["title"])
                        elif m["title"].split(":", 1)[1] not in seen:
                            seen.add(m["title"].split(":", 1)[1])
                            nxt.append(m["title"].split(":", 1)[1])
                    if "continue" not in d:
                        break
                    cont = d["continue"]
        frontier = nxt
        print(f"  level {level}: {len(files)} files, {len(seen)} categories", flush=True)
    return sorted(files)


def append_features(a, new, paths, clip, proc, device, glob_out=None):
    """Write a.data/feats.npy = a.data_in features + features of the `new` images; returns updated meta."""
    meta = json.load(open(os.path.join(a.data_in, "feats.json")))
    old = np.load(os.path.join(a.data_in, "feats.npy"), mmap_mode="r")
    feats = np.lib.format.open_memmap(os.path.join(a.data, "feats.npy"), "w+", np.float16,
                                      (len(old) + len(new),) + old.shape[1:])
    for i in range(0, len(old), 4096):
        j = min(i + 4096, len(old))
        feats[i:j] = old[i:j]
    full_glob = None if glob_out is None else np.zeros((len(old) + len(new), glob_out.shape[1]), np.float32)
    encode_images(clip, proc, new, paths, feats, full_glob, len(old), a.clip_bsz, a.workers, device)
    feats.flush()
    if glob_out is not None:
        glob_out[:] = full_glob[len(old):]
    meta["index"].update({n: len(old) + i for i, n in enumerate(new)})
    json.dump(meta, open(os.path.join(a.data, "feats.json"), "w"))
    return meta


def cmd_build_pool(a):
    """Download the technical image pool from Wikimedia Commons and add its CLIP features to a copy
    of a prepared data folder. Runs fine on CPU (no GPU quota needed)."""
    import shutil
    from transformers import CLIPModel, CLIPProcessor
    os.makedirs(os.path.join(a.data, "pool_images"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for f in glob.glob(os.path.join(a.data_in, "*.jsonl")):
        shutil.copy(f, a.data)

    cats = a.categories.split(",") if a.categories else POOL_CATEGORIES
    depth = POOL_DEPTH if a.depth is None else a.depth
    quota = a.per_category
    meta_rows, k, taken, t0 = [], 0, set(), time.time()
    for ci, cat in enumerate(cats):  # an equal share per category, so no single kind of image dominates the pool
        # Each category also gets an equal share of the time budget, so a slow download still finishes in time.
        deadline = t0 + a.hours * 3600 * (ci + 1) / len(cats)
        print(f"category {cat}:", flush=True)
        titles = [t for t in commons_files([cat], depth, a.pause) if t not in taken]
        rng = random.Random(a.seed)
        rng.shuffle(titles)
        got = 0
        for i in range(0, len(titles), 50):
            if got >= quota or time.time() > deadline:
                break
            d = commons_query(a.pause, titles="|".join(titles[i:i + 50]), prop="imageinfo", iiurlwidth="330",
                              iiprop="url|mime|size|extmetadata",
                              iiextmetadatafilter="LicenseShortName|Artist|ImageDescription")
            pages = sorted(d["query"].get("pages", {}).values(), key=lambda pg: pg.get("title", ""))
            rng.shuffle(pages)  # the API returns each batch in title order
            for page in pages:
                if got >= quota or time.time() > deadline:
                    break
                im, row = fetch_pool_image(page, a.pause)
                if im is None:
                    continue
                name = f"pool/{k:05d}.jpg"
                im.save(os.path.join(a.data, "pool_images", f"{k:05d}.jpg"), quality=90)
                meta_rows.append(dict(name=name, category=cat, **row))
                taken.add(page["title"])
                k, got = k + 1, got + 1
                if k % 50 == 0:
                    print(f"  downloaded {k} images ({(time.time() - t0) / 60:.0f} min)", flush=True)
        print(f"  {got} images from {cat}" + (" (time share used up)" if got < quota and time.time() > deadline else ""),
              flush=True)
    write_jsonl(os.path.join(a.data, "pool_meta.jsonl"), meta_rows)
    print("pool images:", len(meta_rows))

    clip = CLIPModel.from_pretrained(a.clip).to(device).eval()
    proc = CLIPProcessor.from_pretrained(a.clip)
    names = [r["name"] for r in meta_rows]
    paths = {r["name"]: os.path.join(a.data, "pool_images", r["name"].split("/")[1]) for r in meta_rows}
    pool_glob = np.zeros((len(names), clip.config.projection_dim), np.float32)
    append_features(a, names, paths, clip, proc, device, glob_out=pool_glob)
    np.save(os.path.join(a.data, "pool_glob.npy"), pool_glob)
    json.dump(names, open(os.path.join(a.data, "pool_names.json"), "w"))
    print("build_pool done")


def fetch_pool_image(page, pause):
    """Thumbnail and credit line for one Commons file, or (None, None) if it is unusable."""
    import io
    from PIL import Image
    info = (page.get("imageinfo") or [{}])[0]
    if info.get("mime") not in ("image/jpeg", "image/png", "image/svg+xml") or "thumburl" not in info:
        return None, None
    if info.get("width", 0) < 200 or info.get("height", 0) < 150:
        return None, None
    try:
        im = Image.open(io.BytesIO(http_get(info["thumburl"], tries=3, pause=pause))).convert("RGB")
    except Exception as e:  # unreadable or vanished file: skip it
        print("  skipped", page["title"], type(e).__name__)
        return None, None
    ext = info.get("extmetadata", {})
    return im, dict(title=page["title"], page=info.get("descriptionurl"),
                    license=ext.get("LicenseShortName", {}).get("value", ""),
                    artist=re.sub(r"<[^>]+>", "", ext.get("Artist", {}).get("value", ""))[:200])


def retrieve_pool_images(rows, data, clip_name, k, device):
    """Give each row the most CLIP-similar pool image (and the next k-1 as augmentation images)."""
    from transformers import CLIPModel, CLIPProcessor
    names = json.load(open(os.path.join(data, "pool_names.json")))
    pool = torch.tensor(np.load(os.path.join(data, "pool_glob.npy")), device=device)
    clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_name)
    sims = []
    for i in range(0, len(rows), 256):
        chunk = rows[i:i + 256]
        tok = proc(text=[r["src"] for r in chunk], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            t = clip.text_projection(clip.text_model(**{x: v.to(device) for x, v in tok.items()}).pooler_output)
            val, idx = (F.normalize(t, dim=-1) @ pool.T).topk(k, dim=-1)
        for r, v, cand in zip(chunk, val.tolist(), idx.tolist()):
            r["image"], r["aug"], r["ret"] = names[cand[0]], [names[j] for j in cand[1:]], 1
            sims.append(v[0])
    print(f"retrieved pool images for {len(rows)} sentences, mean CLIP similarity {np.mean(sims):.3f}")


def cmd_add_tech(a):
    """Add (or replace) the technical test set in a copy of a prepared data folder.
    Only the new images are encoded, so this takes minutes instead of re-running prep."""
    import shutil
    from transformers import CLIPModel, CLIPProcessor
    os.makedirs(a.data, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for f in glob.glob(os.path.join(a.data_in, "*.jsonl")):
        if os.path.basename(f) != "tech.jsonl":
            shutil.copy(f, a.data)
    meta = json.load(open(os.path.join(a.data_in, "feats.json")))
    for f in ("pool_names.json", "pool_glob.npy", "pool_meta.jsonl"):
        if os.path.exists(os.path.join(a.data_in, f)):
            shutil.copy(os.path.join(a.data_in, f), a.data)
    tech = read_tech(a.tech_tsv)
    train_src = {r["src"].lower().rstrip(".") for r in read_jsonl(os.path.join(a.data_in, "train.jsonl"))}
    overlap = [r["src"] for r in tech if r["src"].lower().rstrip(".") in train_src]
    print(f"test sentences also in training data: {len(overlap)}", overlap[:5])
    paths = {"tech/" + r["image"]: find_image(a.tech_images, r["image"]) for r in tech}
    missing = [k for k, p in paths.items() if p is None]
    assert not missing, f"{len(missing)} images not found, e.g. {missing[:3]}"
    new = sorted(n for n in paths if n not in meta["index"])
    clip = CLIPModel.from_pretrained(a.clip).to(device).eval()
    new_glob = np.zeros((len(new), clip.config.projection_dim), np.float32)
    append_features(a, new, paths, clip, CLIPProcessor.from_pretrained(a.clip), device, glob_out=new_glob)
    if os.path.exists(os.path.join(a.data_in, "pool_glob.npy")):
        # No test image may be (a near-duplicate of) a training pool image.
        pool = np.load(os.path.join(a.data_in, "pool_glob.npy"))
        pool_names = json.load(open(os.path.join(a.data_in, "pool_names.json")))
        best = (new_glob @ pool.T).argmax(1)
        sim = (new_glob @ pool.T).max(1)
        dupes = [(n, pool_names[b], round(float(x), 3)) for n, b, x in zip(new, best, sim) if x >= a.dup_threshold]
        print(f"test images near-identical to a training pool image (cosine >= {a.dup_threshold}): {len(dupes)}")
        for d in dupes:
            print("  ", d)
        assert not dupes or a.allow_dupes, "replace these test images (or pass --allow_dupes to only report them)"
    for r in tech:
        r["image"] = "tech/" + r["image"]
    write_jsonl(os.path.join(a.data, "tech.jsonl"), tech)
    print("add_tech done:", len(tech), "items,", len(new), "new images")


def lexicon_labels(ws, fr):
    """1 for words that form a TECH_LEXICON term whose technical rendering appears in the French."""
    low, fr = [w.lower() for w in ws], fr.lower()
    lab = [0] * len(ws)
    for term, renderings in TECH_LEXICON.items():
        if not any(v in fr for v in renderings.split("|")):
            continue
        t = term.split()
        for i in range(len(low) - len(t) + 1):
            if all(low[i + k] in (t[k], t[k] + "s") for k in range(len(t))):
                for k in range(len(t)):
                    lab[i + k] = 1
    return lab


def cmd_add_soft(a):
    """Add software-localisation text (KDE4 + GNOME, en-fr, from OPUS) to a prepared data folder.
    These sentences have no image (the blank image is used); they bring technical vocabulary and
    technical metaphors into training. A disjoint sample is kept as a technical text test set.
    The cleaned pairs that are in neither are written too: the first n_val as a software validation
    set (soft_val.jsonl), the rest as text-only first-stage data (soft_extra.jsonl)."""
    import io
    import shutil
    import zipfile
    os.makedirs(a.data, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for f in ("feats.npy", "feats.json", "pool_glob.npy", "pool_names.json"):  # shared, not copied
        dst = os.path.join(a.data, f)
        if os.path.exists(os.path.join(a.data_in, f)) and not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(os.path.join(a.data_in, f)), dst)
            except OSError:
                shutil.copy(os.path.join(a.data_in, f), dst)
    for f in glob.glob(os.path.join(a.data_in, "*.jsonl")):
        shutil.copy(f, a.data)

    banned = {x.lower().rstrip(".") for x in TECH_TEST_SOURCES}
    pairs, seen = [], set()
    for corpus, version in SOFT_CORPORA:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(OPUS.format(corpus, version)).read()))
        en = z.read(f"{corpus}.en-fr.en").decode("utf8").split("\n")
        fr = z.read(f"{corpus}.en-fr.fr").decode("utf8").split("\n")
        for e, f in zip(en, fr):
            e, f = e.strip(), f.strip()
            n = len(e.split())
            if not (3 <= n <= 30) or not f or e in seen or e.lower().rstrip(".") in banned:
                continue
            if re.search(r"[<>{}%&_\\]|http", e) or not (0.5 <= len(f) / len(e) <= 2.0):
                continue
            if not re.search(r"(Name|Comment|Query)$", e):  # KDE .desktop residue appended to the French
                f = re.sub(r"\s*(GenericName|Name|Comment|Query)$", "", f)
            if re.search(r"[<>{}&#_\\]|http", f):  # HTML entities, markup
                continue
            seen.add(e)
            pairs.append(dict(src=e, tgt=f, image="<blank>", words=words(e), corpus=corpus))
    print("software sentences after cleaning:", len(pairs))

    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    test, rest = pairs[:a.n_test], pairs[a.n_test:]
    for r in rest:
        r["lex"] = lexicon_labels(r["words"], r["tgt"])
    with_term = [r for r in rest if any(r["lex"])]
    without = [r for r in rest if not any(r["lex"])]
    train = with_term[:a.n_train] + without[:max(0, a.n_train - len(with_term))]
    print(f"software train: {len(train)} sentences ({min(len(with_term), a.n_train)} with a technical metaphor), "
          f"software test: {len(test)}")
    chosen = {r["src"] for r in train}
    unused = [r for r in rest if r["src"] not in chosen]
    for r in unused:
        r.pop("lex", None)
    soft_val, soft_extra = unused[:a.n_val], unused[a.n_val:]
    print(f"unused software pairs: {len(unused)} (validation {len(soft_val)}, first-stage data {len(soft_extra)})")

    tag_metaphors(train, a.tagger, a.tagger_o_label, 64, device)
    for r in train:  # silver labels: tagger OR lexicon
        r["met"] = [max(x, y) for x, y in zip(r["met"], r.pop("lex"))]
    rate = sum(map(sum, (r["met"] for r in train))) / sum(len(r["met"]) for r in train)
    print(f"software metaphor rate: {100 * rate:.1f}% of words")
    if os.path.exists(os.path.join(a.data_in, "pool_names.json")):
        retrieve_pool_images(train + test, a.data_in, a.clip, 3, device)
        retrieve_pool_images(soft_val, a.data_in, a.clip, 3, device)  # separate call: train/test unchanged
    else:
        print("no technical image pool found: software sentences keep the blank image")

    with open(os.path.join(a.data, "train.jsonl"), "a", encoding="utf8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(os.path.join(a.data, "soft_test.jsonl"), test)
    write_jsonl(os.path.join(a.data, "soft_val.jsonl"), soft_val)
    write_jsonl(os.path.join(a.data, "soft_extra.jsonl"), soft_extra)  # blank image, no metaphor labels
    print("add_soft done")


# --------------------------------------------------------------------------- model

class Feats:
    def __init__(self, data):
        meta = json.load(open(os.path.join(data, "feats.json")))
        self.index, self.dim = meta["index"], meta["dim"]
        self.arr = np.load(os.path.join(data, "feats.npy"), mmap_mode="r")

    def get(self, names):
        return torch.from_numpy(np.stack([self.arr[self.index[n]] for n in names]).astype(np.float32))


class MMT(nn.Module):
    """BERT encoder + frozen CLIP features -> gated cross-modal fusion -> mBART-50 decoder."""

    def __init__(self, c, dv):
        super().__init__()
        from transformers import AutoModel, MBartForConditionalGeneration
        self.c = c
        self.mbart = MBartForConditionalGeneration.from_pretrained(c.mbart)
        # The 250k x 1024 embeddings (tied with the output layer) stay frozen to fit a 16 GB GPU.
        self.mbart.model.shared.weight.requires_grad_(bool(getattr(c, "train_emb", 0)))
        if c.arch == "mbart":
            if c.fusion == "early":  # CLIP patches projected to the mBART embedding width
                self.Wv = nn.Linear(dv, self.mbart.config.d_model)
                self.ln_v = nn.LayerNorm(self.mbart.config.d_model)
            return
        if c.arch == "mbe":  # mBART's own encoder gives the text features
            dt = self.mbart.config.d_model
        else:
            self.mbart.model.encoder.layers = nn.ModuleList()  # unused: BERT replaces the mBART encoder
            for n, p in self.mbart.model.encoder.named_parameters():
                if not n.startswith("embed_tokens"):
                    p.requires_grad_(False)
            self.bert = AutoModel.from_pretrained(c.bert)
            dt = self.bert.config.hidden_size
        d = c.d
        self.Wt, self.Wv = nn.Linear(dt, d), nn.Linear(dv, d)
        self.attn = nn.MultiheadAttention(d, c.heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * d, c.gate_hidden), nn.ReLU(), nn.Linear(c.gate_hidden, d))
        self.met = nn.Linear(d, 1)
        self.Wo = nn.Linear(d, self.mbart.config.d_model)
        self.ln = nn.LayerNorm(self.mbart.config.d_model)

    def new_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("mbart.", "bert.")) and p.requires_grad]

    def encode(self, b, vision):
        """Eqs. (1)-(5). Returns Ht, Hv, O, G and the decoder memory M."""
        from transformers.modeling_outputs import BaseModelOutput
        if self.c.arch == "mbart":
            enc, mask = self.mbart.model.encoder, b["mb_mask"]
            if vision and self.c.fusion == "early":
                emb = enc.embed_tokens(b["mb_ids"])
                if not hasattr(enc.embed_tokens, "embed_scale"):  # older transformers scale in the encoder
                    emb = emb * getattr(enc, "embed_scale", 1.0)
                img = self.ln_v(self.Wv(b["img"])).to(emb.dtype)
                mask = torch.cat([torch.ones(img.shape[:2], dtype=mask.dtype, device=mask.device), mask], 1)
                M = enc(inputs_embeds=torch.cat([img, emb], 1), attention_mask=mask).last_hidden_state
            else:
                M = enc(input_ids=b["mb_ids"], attention_mask=mask).last_hidden_state
            return dict(M=BaseModelOutput(last_hidden_state=M), mask=mask)
        text_enc = self.mbart.model.encoder if self.c.arch == "mbe" else self.bert
        Ht = self.Wt(text_enc(input_ids=b["ids"], attention_mask=b["mask"]).last_hidden_state)
        Hv = A = G = None
        if vision and self.c.fusion != "none":
            Hv = self.Wv(b["img"])
            A, _ = self.attn(Ht, Hv, Hv, need_weights=False)
            if self.c.fusion == "add":
                O = Ht + A
            elif self.c.fusion == "fixed":
                O = 0.5 * Ht + 0.5 * A
            else:
                G = torch.sigmoid(self.gate(torch.cat([Ht, A], -1)))
                O = (1 - G) * Ht + G * A
        else:
            O = Ht
        M, mask = self.ln(self.Wo(O)), b["mask"]
        if Hv is not None and getattr(self.c, "mem_img", 0):  # the decoder can also attend to the image tokens
            M = torch.cat([M, self.ln(self.Wo(Hv))], 1)
            mask = torch.cat([mask, torch.ones(Hv.shape[:2], dtype=mask.dtype, device=mask.device)], 1)
        return dict(Ht=Ht, Hv=Hv, O=O, G=G, M=BaseModelOutput(last_hidden_state=M), mask=mask)

    def logits(self, enc, b):
        from transformers.models.mbart.modeling_mbart import shift_tokens_right
        dec_in = shift_tokens_right(b["labels"], self.mbart.config.pad_token_id)
        return self.mbart(encoder_outputs=enc["M"], attention_mask=enc["mask"], decoder_input_ids=dec_in).logits

    def loss(self, b, vision, aux):
        enc = self.encode(b, vision)
        lg = self.logits(enc, b)
        l_mt = F.cross_entropy(lg.float().view(-1, lg.size(-1)), b["labels"].view(-1),
                               ignore_index=-100, label_smoothing=self.c.label_smoothing)
        out = dict(mt=l_mt)
        total = l_mt
        if aux and self.c.lam1 > 0:  # Eq. (7): weighted BCE on metaphor tokens
            z, keep = b["met"], b["met"] >= 0
            p = self.met(enc["O"]).squeeze(-1)
            out["met"] = F.binary_cross_entropy_with_logits(
                p[keep].float(), z[keep].float(), pos_weight=torch.tensor(self.c.w_pos, device=p.device))
            total = total + self.c.lam1 * out["met"]
        has_img = b["has_img"]  # rows with a real image (software text has a blank one)
        if aux and self.c.lam2 > 0 and enc.get("Hv") is not None and has_img.sum() > 1:  # Eq. (8): symmetric InfoNCE
            m = b["mask"][has_img].unsqueeze(-1).float()
            ht = F.normalize((enc["Ht"][has_img] * m).sum(1) / m.sum(1), dim=-1)
            hv = F.normalize(enc["Hv"][has_img].mean(1), dim=-1)
            sim = ht.float() @ hv.float().T / self.c.tau
            y = torch.arange(sim.size(0), device=sim.device)
            out["con"] = (F.cross_entropy(sim, y) + F.cross_entropy(sim.T, y)) / 2
            total = total + self.c.lam2 * out["con"]
        out["total"] = total
        return out

    @torch.no_grad()
    def translate(self, b, vision, beams, fr_id):
        enc = self.encode(b, vision)
        out = self.mbart.generate(encoder_outputs=enc["M"], attention_mask=enc["mask"], forced_bos_token_id=fr_id,
                                  decoder_start_token_id=self.mbart.config.decoder_start_token_id,
                                  num_beams=beams, max_new_tokens=128, early_stopping=True)
        return out, enc.get("G")

    @torch.no_grad()
    def nll(self, b, vision):
        """Mean per-token negative log-likelihood of b['labels'] (CoMMuTE ranking score)."""
        lg = self.logits(self.encode(b, vision), b).float()
        tok = F.cross_entropy(lg.transpose(1, 2), b["labels"], ignore_index=-100, reduction="none")
        keep = (b["labels"] != -100).float()
        return ((tok * keep).sum(1) / keep.sum(1)).tolist()


# --------------------------------------------------------------------------- batching

class Batcher:
    def __init__(self, c, feats):
        from transformers import AutoTokenizer
        self.c, self.feats = c, feats
        self.bert_tok = AutoTokenizer.from_pretrained(c.bert)
        self.mb_tok = AutoTokenizer.from_pretrained(c.mbart, src_lang="en_XX", tgt_lang="fr_XX")
        self.fr_id = self.mb_tok.convert_tokens_to_ids("fr_XX")
        self.src_tok = self.mb_tok if c.arch == "mbe" else self.bert_tok

    def __call__(self, rows, images, targets=None, device="cpu", mask_p=0.0, rng=None, met_mask=0.0):
        """rows: dicts with words/src/met; images: one feature name per row; targets: French strings.
        mask_p: chance of hiding each source word in rows with a real image (training only).
        met_mask: chance of hiding each word tagged as a metaphor in those rows (training only)."""
        b = {}
        enc = self.src_tok([r["words"] for r in rows], is_split_into_words=True, truncation=True,
                           max_length=self.c.max_len, padding=True, return_tensors="pt")
        b["ids"], b["mask"] = enc["input_ids"], enc["attention_mask"]
        if mask_p > 0 or met_mask > 0:
            for i, name in enumerate(images):
                if name != "<blank>":
                    tags = rows[i].get("met") or [0] * len(rows[i]["words"])
                    hidden = {w for w in range(len(rows[i]["words"]))
                              if (met_mask > 0 and tags[w] == 1 and rng.random() < met_mask)
                              or (mask_p > 0 and rng.random() < mask_p)}
                    for t, w in enumerate(enc.word_ids(i)):
                        if w in hidden:
                            b["ids"][i, t] = self.src_tok.mask_token_id
        met = torch.full_like(b["ids"], -100)
        for i, r in enumerate(rows):
            if "met" in r:
                for t, w in enumerate(enc.word_ids(i)):
                    if w is not None:
                        met[i, t] = r["met"][w]
        b["met"] = met
        if self.c.arch == "mbart":
            mb = self.mb_tok([r["src"] for r in rows], truncation=True, max_length=self.c.max_len,
                             padding=True, return_tensors="pt")
            b["mb_ids"], b["mb_mask"] = mb["input_ids"], mb["attention_mask"]
        b["img"] = self.feats.get(images)
        b["has_img"] = torch.tensor([n != "<blank>" for n in images])
        if targets is not None:
            lab = self.mb_tok(text_target=targets, truncation=True, max_length=self.c.max_len,
                              padding=True, return_tensors="pt")["input_ids"]
            lab[lab == self.mb_tok.pad_token_id] = -100
            b["labels"] = lab
        return {k: v.to(device) for k, v in b.items()}

    def decode(self, ids):
        return self.mb_tok.batch_decode(ids, skip_special_tokens=True)


def batches(n, bsz, shuffle, rng=None):
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    return [idx[i:i + bsz] for i in range(0, n, bsz)]


# --------------------------------------------------------------------------- evaluation

def corpus_scores(hyp, ref):
    import sacrebleu
    return dict(bleu=round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
                chrf=round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2))


def derange(names, seed):
    """Give every item another item's image (for the shuffled-image condition)."""
    perm = list(range(len(names)))
    random.Random(seed).shuffle(perm)
    out = list(names)
    for i, j in zip(perm, perm[1:] + perm[:1]):
        out[i] = names[j]
    return out


def run_translate(model, bt, rows, images, vision, c, device, beams):
    hyps, gates = [], []
    for idx in batches(len(rows), c.eval_bsz, False):
        b = bt([rows[i] for i in idx], [images[i] for i in idx], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out, G = model.translate(b, vision, beams, bt.fr_id)
        hyps += bt.decode(out)
        if G is not None:  # mean gate value of each source word (first sub-token)
            g = G.float().mean(-1).cpu()
            for k, i in enumerate(idx):
                enc_ids = bt.src_tok(rows[i]["words"], is_split_into_words=True, truncation=True, max_length=c.max_len)
                first = {}
                for t, w in enumerate(enc_ids.word_ids()):
                    if w is not None and w not in first:
                        first[w] = g[k, t].item()
                gates.append([first[w] for w in sorted(first)])
    return hyps, gates


def contrastive_accuracy(model, bt, c, rows, images, vision, device):
    """CoMMuTE protocol: is the correct translation less perplexing than the incorrect one?"""
    scores = {"correct": [], "incorrect": []}
    ref_col = "ref" if "ref" in rows[0] else "correct"
    for idx in batches(len(rows), c.eval_bsz, False):
        chunk = [rows[i] for i in idx]
        for key, col in (("correct", ref_col), ("incorrect", "incorrect")):
            b = bt(chunk, [images[i] for i in idx], [r[col] for r in chunk], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                scores[key] += model.nll(b, vision)
    acc = float(np.mean([x < y for x, y in zip(scores["correct"], scores["incorrect"])]))
    return round(100 * acc, 2), scores


def evaluate(model, bt, c, out_dir, device, sets=("test2016", "commute", "tech", "soft", "soft_val")):
    model.eval()
    vision = c.fusion != "none"
    path = os.path.join(out_dir, "results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    cond_seed = 1234  # same shuffled pairing for every system

    if "test2016" in sets:
        res.update(eval_test2016(model, bt, c, out_dir, device, vision, cond_seed))
    if "commute" in sets and os.path.exists(os.path.join(c.data, "commute.jsonl")):
        com = read_jsonl(os.path.join(c.data, "commute.jsonl"))
        res["commute_accuracy"], scores = contrastive_accuracy(model, bt, c, com, [r["image"] for r in com], vision, device)
        for key, v in scores.items():  # input files for CoMMuTE's official evaluate.py
            write_lines(os.path.join(out_dir, f"commute_{key}.txt"), [f"{x:.6f}" for x in v])
        print("CoMMuTE accuracy", res["commute_accuracy"])
    if "tech" in sets and os.path.exists(os.path.join(c.data, "tech.jsonl")):
        res.update(eval_tech(model, bt, c, out_dir, device, vision, cond_seed))
    if "soft" in sets and os.path.exists(os.path.join(c.data, "soft_test.jsonl")):
        # Held-out software text (no image): translation quality on technical language.
        soft = read_jsonl(os.path.join(c.data, "soft_test.jsonl"))
        imgs = [r["image"] if getattr(c, "ret", 1) else "<blank>" for r in soft]
        hyps, _ = run_translate(model, bt, soft, imgs, vision, c, device, c.beams)
        write_lines(os.path.join(out_dir, "soft_test.fr"), hyps)
        res["soft_test"] = corpus_scores(hyps, [r["tgt"] for r in soft])
        print("software test", res["soft_test"])
    if "soft_val" in sets and os.path.exists(os.path.join(c.data, "soft_val.jsonl")):
        # Software validation set: used only to choose the first-stage checkpoint and for the
        # pre-training decision rule, never the software test set.
        sv = read_jsonl(os.path.join(c.data, "soft_val.jsonl"))
        imgs = [r["image"] if getattr(c, "ret", 1) else "<blank>" for r in sv]
        hyps, _ = run_translate(model, bt, sv, imgs, vision, c, device, c.beams)
        write_lines(os.path.join(out_dir, "soft_val.fr"), hyps)
        res["soft_val"] = corpus_scores(hyps, [r["tgt"] for r in sv])
        print("software validation", res["soft_val"])
    json.dump(res, open(path, "w"), indent=2)
    return res


def eval_test2016(model, bt, c, out_dir, device, vision, cond_seed):
    res = {}
    test = read_jsonl(os.path.join(c.data, "test2016.jsonl"))
    refs = [r["tgt"] for r in test]
    conds = {"correct": [r["image"] for r in test]}
    if vision:
        conds["shuffled"] = derange(conds["correct"], cond_seed)
        conds["blank"] = ["<blank>"] * len(test)
    for cond, imgs in conds.items():
        hyps, gates = run_translate(model, bt, test, imgs, vision, c, device, c.beams)
        write_lines(os.path.join(out_dir, f"test2016_{cond}.fr"), hyps)
        res[f"test2016_{cond}"] = corpus_scores(hyps, refs)
        print(cond, res[f"test2016_{cond}"])
        if cond == "correct" and gates:
            vals, labs = [], []
            for r, g in zip(test, gates):
                vals += g
                labs += r["met"][:len(g)]
            write_lines(os.path.join(out_dir, "gate_values.txt"), [f"{v:.5f}" for v in vals])
            write_lines(os.path.join(out_dir, "gate_labels.txt"), [str(x) for x in labs])
    return res


def span_hit(hyp, accepted, excluded=""):
    """True if the output contains an accepted rendering ('|'-separated) and no excluded phrase."""
    h = hyp.lower()
    found = any(v.strip().lower() in h for v in accepted.split("|") if v.strip())
    return found and not any(v.strip().lower() in h for v in (excluded or "").split("|") if v.strip())


def eval_tech(model, bt, c, out_dir, device, vision, cond_seed):
    res = {}
    tech = read_jsonl(os.path.join(c.data, "tech.jsonl"))
    if "incorrect" in tech[0]:
        res["tech_contrastive_accuracy"], scores = contrastive_accuracy(
            model, bt, c, tech, [r["image"] for r in tech], vision, device)
        for key, v in scores.items():  # per-item scores, for the paired (both senses right) analysis
            write_lines(os.path.join(out_dir, f"tech_{key}.txt"), [f"{x:.6f}" for x in v])
        print("tech contrastive accuracy", res["tech_contrastive_accuracy"])
        if vision:  # same test with the blank image: how much of the accuracy comes from the image
            res["tech_contrastive_accuracy_blank"], _ = contrastive_accuracy(
                model, bt, c, tech, ["<blank>"] * len(tech), vision, device)
            print("tech contrastive accuracy, blank image", res["tech_contrastive_accuracy_blank"])
    conds = {"correct": [r["image"] for r in tech]}
    if vision:
        conds["shuffled"] = derange(conds["correct"], cond_seed)
        conds["blank"] = ["<blank>"] * len(tech)
    for cond, imgs in conds.items():
        hyps, _ = run_translate(model, bt, tech, imgs, vision, c, device, c.beams)
        write_lines(os.path.join(out_dir, f"tech_{cond}.fr"), hyps)
        res[f"tech_{cond}"] = corpus_scores(hyps, [r["ref"] for r in tech])
        # Automatic metaphor-translation check: an accepted rendering of the term appears in the output,
        # and none of the row's excluded phrases does (tgt_not, e.g. "fil d'exécution" for sewing "fil").
        hit = [int(span_hit(h, r["tgt_span"], r.get("tgt_not", ""))) for r, h in zip(tech, hyps)]
        write_lines(os.path.join(out_dir, f"tech_{cond}_auto_judgements.txt"), [str(x) for x in hit])
        res[f"tech_{cond}"]["auto_mta"] = round(100 * float(np.mean(hit)), 2)
        print("tech", cond, res[f"tech_{cond}"])
    return res


# --------------------------------------------------------------------------- training

def build_config(a):
    c = argparse.Namespace(**{**DEFAULTS, **SYSTEMS[a.system]})
    for k, v in vars(a).items():
        if not k.startswith("set_"):
            setattr(c, k, v)
    for k in DEFAULTS:  # explicit --set_* overrides win over the system preset
        if getattr(a, "set_" + k) is not None:
            setattr(c, k, getattr(a, "set_" + k))
    return c


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().half().cpu() for k, v in model.state_dict().items() if k in keep}


def load_init(model, path, arch):
    """Start from a first-stage checkpoint: every trained tensor except the fusion modules (FRESH)."""
    cfg = os.path.join(os.path.dirname(path), "config.json")
    if os.path.exists(cfg):
        got = json.load(open(cfg)).get("arch", "bert")
        assert got == arch, f"{path} was trained with arch={got}, this system uses arch={arch}"
    own = model.state_dict()
    sd = torch.load(path, map_location="cpu")
    keep = {k: v.float() for k, v in sd.items() if k in own and not k.startswith(FRESH)}
    assert keep, f"no matching weights in {path}"
    bad = [k for k, v in keep.items() if v.shape != own[k].shape]
    assert not bad, f"shape mismatch: {bad[:5]}"
    model.load_state_dict(keep, strict=False)
    print(f"initialised {len(keep)} of {len(sd)} tensors from {path}")


def cmd_train(a):
    c = build_config(a)
    seed_all(c.seed)
    os.makedirs(c.out, exist_ok=True)
    json.dump(vars(c), open(os.path.join(c.out, "config.json"), "w"), indent=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = Feats(c.data)
    bt = Batcher(c, feats)
    model = MMT(c, feats.dim).to(device)
    if c.system.endswith("_pt"):
        assert c.init, f"{c.system} starts from a first-stage checkpoint: pass --init runs/pre_sN/best.pt"
    if c.init:
        load_init(model, c.init, c.arch)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"system={c.system} seed={c.seed} trainable params={n_train / 1e6:.1f}M device={device}")

    train = read_jsonl(os.path.join(c.data, c.train_file))
    val = read_jsonl(os.path.join(c.data, c.val_file))
    if c.limit:
        train, val = train[:c.limit], val[:c.limit]
    new = set(map(id, model.new_params()))
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in new], lr=c.lr),
              dict(params=[p for p in model.parameters() if p.requires_grad and id(p) in new], lr=c.lr_new)]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / c.bsz)
    total = steps_per_epoch * c.epochs
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    rng = random.Random(c.seed)
    log = open(os.path.join(c.out, "log.jsonl"), "a")
    best, bad, step, t0 = -1.0, 0, 0, time.time()

    for epoch in range(1, c.epochs + 1):
        vision = c.fusion != "none" and epoch > c.curriculum  # curriculum stage 1: text only
        aux = epoch > c.curriculum
        model.train()
        for idx in batches(len(train), c.bsz, True, rng):
            rows = [train[i] for i in idx]
            imgs = ["<blank>" if r.get("ret") and not c.ret
                    else rng.choice(r["aug"]) if c.aug and r.get("aug") and rng.random() < c.aug_p
                    else r["image"] for r in rows]
            b = bt(rows, imgs, [r["tgt"] for r in rows], device=device, mask_p=c.mask_p if vision else 0.0, rng=rng,
                   met_mask=c.met_mask if vision else 0.0)
            if step < c.warmup:  # linear warm-up, then linear decay
                lr_scale = (step + 1) / c.warmup
            else:
                lr_scale = max(0.0, (total - step) / max(1, total - c.warmup))
            for g, lr in zip(opt.param_groups, base):
                g["lr"] = lr * lr_scale
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model.loss(b, vision, aux)
            opt.zero_grad(set_to_none=True)
            scaler.scale(out["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], c.clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % c.log_every == 0:
                msg = {k: round(v.item(), 4) for k, v in out.items()}
                msg.update(step=step, epoch=epoch, min=round((time.time() - t0) / 60, 1))
                print(msg)
                log.write(json.dumps(msg) + "\n")
                log.flush()

        # Model selection on validation BLEU (greedy decoding for speed), same rule for every system.
        model.eval()
        hyps, _ = run_translate(model, bt, val, [r["image"] for r in val], vision, c, device, beams=1)
        score = corpus_scores(hyps, [r["tgt"] for r in val])
        msg = dict(epoch=epoch, val=score, vision=vision, min=round((time.time() - t0) / 60, 1))
        print(msg)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if epoch <= c.curriculum:
            continue  # stage 1 checkpoints are never selected
        if score["bleu"] > best:
            best, bad = score["bleu"], 0
            torch.save(trainable_state(model), os.path.join(c.out, "best.pt"))
        else:
            bad += 1
            if bad >= c.patience:
                print("early stop")
                break

    model.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(c.out, "best.pt")).items()}, strict=False)
    json.dump(dict(best_val_bleu=best, train_minutes=round((time.time() - t0) / 60, 1),
                   trainable_params_M=round(n_train / 1e6, 1)), open(os.path.join(c.out, "results.json"), "w"), indent=2)
    if not c.no_eval:  # a first stage is never evaluated on the test sets
        evaluate(model, bt, c, c.out, device)
    if not c.keep_ckpt:
        os.remove(os.path.join(c.out, "best.pt"))


def cmd_eval(a):
    c = argparse.Namespace(**json.load(open(os.path.join(a.run, "config.json"))))
    c.data = a.data or c.data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = Feats(c.data)
    bt = Batcher(c, feats)
    model = MMT(c, feats.dim).to(device)
    model.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(a.run, "best.pt")).items()}, strict=False)
    out = a.out or a.run
    if out != a.run:  # e.g. the run comes from a read-only Kaggle input
        os.makedirs(out, exist_ok=True)
        for f in ("config.json", "results.json"):
            if os.path.exists(os.path.join(a.run, f)):
                with open(os.path.join(a.run, f), encoding="utf8") as src, open(os.path.join(out, f), "w", encoding="utf8") as dst:
                    dst.write(src.read())
    evaluate(model, bt, c, out, device, sets=a.sets.split(","))


def cmd_collect(a):
    """Mean ± s.d. over seeds of every metric, one row per system (Tables 2-4)."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(a.runs, "**", "results.json"), recursive=True)):
        cfg = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        res = json.load(open(path))
        flat = {}
        for k, v in res.items():
            if isinstance(v, dict):
                flat.update({f"{k}.{m}": x for m, x in v.items()})
            else:
                flat[k] = v
        groups.setdefault(cfg["system"], []).append((cfg["seed"], flat))
    cols = a.metrics.split(",")
    lines = ["| system | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for system in [s for s in SYSTEMS if s in groups]:
        runs = groups[system]
        cells = []
        for m in cols:
            vals = [f[m] for _, f in runs if m in f]
            if not vals:
                cells.append("–")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.1f}")
            else:
                cells.append(f"{np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f}")
        lines.append(f"| {system} | {','.join(str(s) for s, _ in sorted(runs, key=lambda r: r[0]))} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    if a.out:
        write_lines(a.out, lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--flickr_dir", required=True)
    p.add_argument("--commute_dir")
    p.add_argument("--commute_images")
    p.add_argument("--tech_tsv")
    p.add_argument("--tech_images")
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--tagger", default="HiTZ/deberta-base-metaphor-detection-en")
    p.add_argument("--tagger_o_label", type=int, default=2)  # HiTZ labels: 0 B-MET, 1 I-MET, 2 O
    p.add_argument("--clip_bsz", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--aug_k", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--system", required=True, choices=SYSTEMS)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--bert", default="bert-base-cased")
    p.add_argument("--mbart", default="facebook/mbart-large-50")
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--gate_hidden", type=int, default=768)
    p.add_argument("--w_pos", type=float, default=5.0)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--aug_p", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lr_new", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--bsz", type=int, default=16)
    p.add_argument("--eval_bsz", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--beams", type=int, default=5)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--keep_ckpt", action="store_true")
    p.add_argument("--train_emb", type=int, default=0)
    p.add_argument("--train_file", default="train.jsonl", help="first stage: soft_extra.jsonl")
    p.add_argument("--val_file", default="val.jsonl", help="first stage: soft_val.jsonl")
    p.add_argument("--init", help="first-stage checkpoint (best.pt) to start from")
    p.add_argument("--no_eval", action="store_true", help="skip the test sets (first stage)")
    for k in DEFAULTS:
        p.add_argument("--set_" + k, type=type(DEFAULTS[k]))

    p = sub.add_parser("eval")
    p.add_argument("--run", required=True)
    p.add_argument("--data")
    p.add_argument("--out", help="where to write outputs (default: the run folder)")
    p.add_argument("--sets", default="test2016,commute,tech,soft,soft_val")

    p = sub.add_parser("build_pool")
    p.add_argument("--data_in", required=True, help="an existing prepared data folder")
    p.add_argument("--data", required=True, help="new data folder to write")
    p.add_argument("--per_category", type=int, default=POOL_PER_CATEGORY)
    p.add_argument("--categories", help="comma-separated override of POOL_CATEGORIES (tests only)")
    p.add_argument("--depth", type=int)
    p.add_argument("--pause", type=float, default=1.0, help="seconds between Wikimedia requests")
    p.add_argument("--hours", type=float, default=8.0, help="time budget for the downloads, split over the categories")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip_bsz", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("add_soft")
    p.add_argument("--data_in", required=True, help="an existing prepared data folder")
    p.add_argument("--data", required=True, help="new data folder to write")
    p.add_argument("--n_train", type=int, default=12000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tagger", default="HiTZ/deberta-base-metaphor-detection-en")
    p.add_argument("--tagger_o_label", type=int, default=2)

    p = sub.add_parser("add_tech")
    p.add_argument("--data_in", required=True, help="an existing prepared data folder")
    p.add_argument("--data", required=True, help="new data folder to write")
    p.add_argument("--tech_tsv", required=True)
    p.add_argument("--tech_images", required=True)
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--clip_bsz", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dup_threshold", type=float, default=0.90)  # real CLIP: distinct images <= 0.86, edited copies >= 0.92
    p.add_argument("--allow_dupes", action="store_true")

    p = sub.add_parser("collect")
    p.add_argument("--runs", required=True)
    p.add_argument("--metrics", default="test2016_correct.bleu,test2016_correct.chrf,commute_accuracy,"
                                         "test2016_shuffled.bleu,test2016_blank.bleu,soft_test.bleu,soft_val.bleu,"
                                         "best_val_bleu,train_minutes")
    p.add_argument("--out")

    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval, "add_tech": cmd_add_tech, "add_soft": cmd_add_soft, "build_pool": cmd_build_pool, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
