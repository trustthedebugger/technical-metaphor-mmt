#!/bin/sh
# Rebuild every table, figure and significance test from results/<batch>/runs.
# Needs: pip install sacrebleu scipy numpy matplotlib
set -e
cd "$(dirname "$0")"
UA="technical-metaphor-mmt (research script)"
# Public reference data used by the analysis (downloaded once, not redistributed here).
if [ ! -f test2016_ref.fr ]; then
  curl -sSL -A "$UA" https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.fr.gz | gunzip > test2016_ref.fr
fi
if [ ! -d commute_text ]; then
  mkdir -p commute_text
  for f in src.en correct.fr incorrect.fr img.order; do
    curl -sSL -A "$UA" -o commute_text/$f https://raw.githubusercontent.com/MatthieuFP/CoMMuTE/main/en-fr/$f
  done
fi
python ../commute_pairs.py ../results/*/runs > commute_pairs_all.md
python gate_stats.py ../results > gate_stats.md
python bleu_bootstrap.py ../results > bleu_bootstrap.md
for r in rm10_mem_s1 rm10_mem_s2 rm10_mem_s3 sel_met_s1 sel_met_s2 sel_met_s3; do
  if ls -d ../results/*/runs/$r >/dev/null 2>&1; then
    PYTHONIOENCODING=utf8 python commute_examples.py ../results $r > examples_$r.md
  fi
done
PYTHONIOENCODING=utf8 python make_tables.py > tables.md
PYTHONIOENCODING=utf8 python tech_stats.py ../results/tech_final/runs_tech > tech_stats.md
mkdir -p ../figures
python make_model_figures.py ../figures
python make_figures.py
echo done
