# Technical metaphors in multimodal machine translation

Code, test set and results for the paper
**"Disambiguating Technical Metaphors via Gated Attention in Multimodal Machine Translation"**
(Rohan Kumar Jha and Chandan Vishwas, VIT-AP University).

The model translates English into French with the help of an image. A BERT encoder reads the sentence, a frozen
CLIP ViT-B/32 encoder reads the image, a learned gate decides for every word how much image information to mix in,
and the mBART-50 decoder writes the French translation. A metaphor classifier on the fused states adds a training
signal. The proposed configuration also hides 10% of the source words during training, lets the decoder attend to
the image directly, and starts from a text-only first training stage on unused software text.

## What is in this repository

| Path | Contents |
|---|---|
| `mmt.py` | The whole pipeline in one file: data preparation, training, evaluation (sub-commands) |
| `notebooks/` | The eight Kaggle notebooks used to run the experiments (each writes its own copy of `mmt.py`) |
| `test_set/` | The technical metaphor test set: `tech.tsv`, 102 images, `image_credits.tsv`, licence |
| `results/` | Outputs of every training and evaluation run (scores, translations, gate values, configs, logs); model checkpoints are not included |
| `analysis/` | Scripts that rebuild every table, figure and significance test in the paper from `results/` |
| `commute_pairs.py` | Paired CoMMuTE analysis (pairs right / wrong with both images, sign test, image effect) |
| `figures/` | Written by `analysis/run_all.sh` |

## Systems

Every system is defined in `SYSTEMS` near the top of `mmt.py`; each was trained with seeds 1, 2 and 3 unless noted.

| Name in the paper | Name in the code |
|---|---|
| mBART-50 | `mbart` |
| Early fusion | `early` |
| Text-only BERT model | `text` |
| Original full model (first submission) | `full` |
| Ablations of the original model (no gate, fixed gate, no metaphor loss, no contrastive loss, no augmentation, no curriculum, no retrieved images) | `concat`, `fixed`, `no_met`, `no_con`, `no_aug`, `no_cur`, `no_ret` |
| Gate only (selective attention) | `selective` |
| Gate + one component | `sel_met`, `sel_con`, `sel_cur`, `sel_aug`, `sel_both` (the last four with seed 1 only) |
| Gate + metaphor loss without retrieved images | `sel_met_nr` |
| Variants tried while choosing the configuration (seed 1) | `sm_*`, `mm*`, `rm10` |
| Proposed configuration, single-stage training | `rm10_mem` |
| Text-only first stage (BERT model / mBART-50) | `pre` / `pre_mb` (never evaluated on a test set) |
| With the text-only first stage | `text_pt`, `mbart_pt`, `early_pt`, **`rm10_mem_pt` (proposed model)** |

## Results folders

Each run has its own folder `results/<batch>/runs/<system>_s<seed>/` with `config.json`, `results.json`,
the training log and the translations of every test set.

| Batch | Runs |
|---|---|
| `round1` | Every original system and ablation, seed 1 |
| `round2` | Gate only (seeds 2–3), gate + one component (seed 1), more seeds of the original model and the text-only model |
| `round3` | Gate + metaphor loss (seeds 2–3), text-only and mBART-50 (seed 3 / 2) |
| `improvement` | Variants of gate + metaphor loss, the no-retrieved-images runs, mBART-50 seed 3, fixed gate seed 1 |
| `masking` | Word-hiding variants, including the proposed configuration, seed 1 |
| `sessionA` | Proposed configuration seeds 2–3; ablations seed 2 |
| `sessionB1`, `sessionB2` | Early fusion (seeds 1–3); ablations seed 3 |
| `pilot` | Text-only first stage, seed 1 (`pre_s1`, `text_pt_s1`, `rm10_mem_pt_s1`); `old_evals/` holds the software-validation scores of earlier checkpoints used for the decision rule |
| `fullA`, `fullB` | Text-only first stage, seeds 2–3 for the BERT systems and seeds 1–3 for the mBART-50 systems |
| `tech_final` | Technical metaphor test set, scored for every system with a saved checkpoint |

## Technical metaphor test set

51 English terms that have a technical and a concrete meaning (e.g. *bug, firewall, bottleneck, thread, cache*).
For each term one short, ambiguous English sentence is paired with two images, one for each sense, giving 102 items.
Both authors, who are bilingual in English and French, independently checked every item (whether the sentence is
ambiguous without the image, whether the reference translation is correct, and whether the translation of the other
sense is wrong for this image); they agreed on all 102 items, and the three translations both marked as incorrect
were corrected. Ten technical-sense images have a near-duplicate in the training image pool; the main results use
the other 41 terms (see the paper, Section 5.4).

`tech.tsv` columns:

| Column | Meaning |
|---|---|
| `id` | `001t` (technical sense) / `001l` (literal sense) |
| `term`, `sense` | The term and which sense the image shows |
| `src` | English sentence (identical for both items of a term) |
| `ref` | Reference French translation for this image |
| `incorrect` | The translation that would be right for the other image |
| `image` | Image file in `test_set/images/` |
| `met_span` | The term as it appears in `src` |
| `tgt_span` | Acceptable French renderings for this sense, separated by `\|` (used for automatic MTA) |
| `image_description` | Short description of the image |
| `tgt_not` | Phrases that must not appear for an output to count as correct (rarely used) |

## Running the experiments (Kaggle)

The experiments were run on Kaggle with two NVIDIA T4 GPUs. Each notebook explains its inputs at the top.

1. **`1_prep.ipynb`** – downloads Multi30K en–fr and CoMMuTE en–fr, tags metaphors (silver labels) and extracts
   frozen CLIP features for all images. Input: the Flickr30K images (Kaggle dataset `flickr-image-dataset`).
2. **`1b_images.ipynb`** – downloads the technical image pool from Wikimedia Commons and adds its CLIP features.
   Input: the output of notebook 1.
3. **`2_train.ipynb`** – adds the KDE4/GNOME software text from OPUS (retrieving an image for each software
   sentence), then trains the systems listed in `RUNS` (two at a time, one per GPU) and evaluates each on Multi30K,
   CoMMuTE and the software test set. Input: the output of notebook 1b.
4. **`2_train_pt.ipynb`** – the text-only first stage and the pilot (seed 1).
5. **`2_train_full_a.ipynb`, `2_train_full_b.ipynb`** – the text-only first stage for the remaining seeds and for
   the mBART-50 systems; part B also scores every checkpoint on the technical test set.
6. **`2_train_c.ipynb`** – seeds 2–3 of the gate + one component systems (prepared, not run; see the paper).
7. **`3_tech_eval.ipynb`** – scores saved checkpoints on the technical metaphor test set.

The same steps can be run from the command line, e.g.

```
python mmt.py prep  --data data --flickr_dir FLICKR30K_IMAGES --commute_dir CoMMuTE --commute_images COMMUTE_IMAGES
python mmt.py add_soft --data_in data --data data
python mmt.py train --data data --system pre --seed 1 --out runs/pre_s1 --train_file soft_extra.jsonl --val_file soft_val.jsonl --epochs 2 --no_eval --keep_ckpt
python mmt.py train --data data --system rm10_mem_pt --seed 1 --out runs/rm10_mem_pt_s1 --init runs/pre_s1/best.pt
python mmt.py eval  --run runs/rm10_mem_pt_s1 --data data_tech --sets tech
```

## Rebuilding the tables and figures

```
pip install sacrebleu scipy numpy matplotlib
sh analysis/run_all.sh
```

This downloads the Multi30K test references and the CoMMuTE en–fr text (public, not redistributed here) and writes
the tables and statistics to `analysis/*.md` and the figures to `figures/`. Every table and test in the paper is
produced by this script from the files in `results/`.

## Data and licences

- Code: MIT licence (`LICENSE`).
- Technical metaphor test set (`test_set/`): CC BY-SA 4.0 (`test_set/LICENSE.md`); the Wikimedia Commons images keep
  their own licences, listed with their source page and author in `image_credits.tsv`.
- Third-party data are not redistributed: Multi30K (Elliott et al., 2016, 2017), Flickr30K images (Young et al.,
  2014), CoMMuTE (Futeral et al., 2023), KDE4 and GNOME (OPUS; Tiedemann, 2012) and the Wikimedia Commons image
  pool are downloaded by the scripts.

## Citation

If you use the code or the test set, please cite the paper (details will be added on publication).
