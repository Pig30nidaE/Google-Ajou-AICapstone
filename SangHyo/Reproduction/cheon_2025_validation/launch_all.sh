#!/bin/bash
# Full runs: R and G as one process each; N as five single-seed processes; then merge N. All fits single-threaded.
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p results/launcher
date -u +"%Y-%m-%dT%H:%M:%SZ launch" > results/launcher/status.txt
$PY run.py --config configs/reproduction.yaml        > results/launcher/R.out 2>&1 &
$PY run.py --config configs/subject_independent.yaml > results/launcher/G.out 2>&1 &
for s in 42 43 44 45 46; do
  $PY run.py --config configs/nested_cv.yaml --seeds $s --output-dir results/nested_cv_seed$s > results/launcher/N_$s.out 2>&1 &
done
wait
$PY merge_runs.py --config configs/nested_cv.yaml --runs results/nested_cv_seed42 results/nested_cv_seed43 results/nested_cv_seed44 results/nested_cv_seed45 results/nested_cv_seed46 --out results/nested_cv > results/launcher/merge.out 2>&1
date -u +"%Y-%m-%dT%H:%M:%SZ ALL_DONE" >> results/launcher/status.txt
echo ALL_DONE
