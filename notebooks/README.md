# Notebooks

The original research implementation remains notebook-first.

Current notebooks at repository root:

- `sample2.ipynb`: main end-to-end experiment
- `sample.ipynb`: earlier pipeline variant
- `case experiment.ipynb`: case-family screening and comparison

Recommended migration path:

1. move reusable logic into `src/hotspot_detection/`
2. keep notebooks for narrative analysis and figure generation
3. split the large monolithic notebook into topic-specific notebooks

