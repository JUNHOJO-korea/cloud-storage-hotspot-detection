# Cloud Storage Hotspot Detection

Research codebase and open-source scaffold for multi-dimensional load imbalance detection and hotspot identification in cloud storage systems.

This repository is based on a bachelor thesis project that studies how to detect node-level and port-level hotspots from real industrial monitoring data. The current codebase combines:

- a rule-based mainline built on imbalance metrics and robust SPC
- event-level hotspot construction with culprit analysis
- a metric-learning extension for structured hotspot state discrimination
- comparison baselines such as Mahalanobis distance, EWMA SPC, and Random Forest

## Project Status

The original implementation is notebook-first. This repository now includes an open-source project scaffold so the work can evolve into a maintainable Python package.

- Research notebooks remain the current source for the full experiment flow
- `src/hotspot_detection/` contains the initial package skeleton and reusable core utilities
- `docs/` translates thesis logic into software-facing documentation
- `examples/` and `tests/` provide a starting point for reproducible development

## Research Scope

The project models hotspot detection as a structured state recognition problem rather than a single-threshold anomaly problem.

Core ideas:

1. Normalize raw monitoring data from system, node, and port levels into a unified long-table schema.
2. Construct imbalance metrics such as HHI, Gini, Theil, Top1 Share, and normalized Top1 Share.
3. Detect abnormal imbalance states using median/MAD-based SPC with persistence and load gating.
4. Merge valid abnormal points into event-level hotspot segments with tier labels and culprit entities.
5. Learn a weighted state space for hotspot discrimination using weak supervision derived from the rule-based pipeline.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── docs/
├── examples/
├── notebooks/
├── src/
├── tests/
├── scripts/
├── artifacts/
└── data/
```

Important current files:

- [sample2.ipynb](/Users/jojunho/PycharmProjects/JupyterProject3/sample2.ipynb): main end-to-end experiment notebook
- [sample.ipynb](/Users/jojunho/PycharmProjects/JupyterProject3/sample.ipynb): earlier notebook variant
- [case experiment.ipynb](/Users/jojunho/PycharmProjects/JupyterProject3/case experiment.ipynb): feature-family case screening notebook

## Installation

### Option 1: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Option 2: `conda`

```bash
conda env create -f environment.yml
conda activate hotspot-detection
pip install -e .
```

## Quick Start

Check the package wiring:

```bash
python -m hotspot_detection.cli show-config
python -m hotspot_detection.cli describe-notebooks
```

Run example scripts:

```bash
python examples/run_rule_based_pipeline.py
python examples/run_metric_learning_pipeline.py
python examples/inspect_hotspot_case.py
```

Run tests:

```bash
pytest
```

## Reproducing the Thesis Pipeline

The full research pipeline currently lives in notebooks. The recommended order is:

1. [sample2.ipynb](/Users/jojunho/PycharmProjects/JupyterProject3/sample2.ipynb)
2. [case experiment.ipynb](/Users/jojunho/PycharmProjects/JupyterProject3/case experiment.ipynb)

Expected major stages:

1. Data loading and unified long-table construction
2. Time binning and metric normalization
3. Imbalance metric construction
4. Robust SPC event detection
5. Hotspot segment generation and labeling
6. Mahalanobis baseline comparison
7. Weighted metric learning
8. Final scoring, episode extraction, and validation

## Data

The original industrial monitoring data are not included in this repository.

See [docs/data_schema.md](/Users/jojunho/PycharmProjects/JupyterProject3/docs/data_schema.md) for the expected input schema and [data/README.md](/Users/jojunho/PycharmProjects/JupyterProject3/data/README.md) for repository policy.

## Documentation

- [project_overview.md](/Users/jojunho/PycharmProjects/JupyterProject3/docs/project_overview.md)
- [methodology.md](/Users/jojunho/PycharmProjects/JupyterProject3/docs/methodology.md)
- [data_schema.md](/Users/jojunho/PycharmProjects/JupyterProject3/docs/data_schema.md)
- [evaluation.md](/Users/jojunho/PycharmProjects/JupyterProject3/docs/evaluation.md)
- [reproducibility.md](/Users/jojunho/PycharmProjects/JupyterProject3/docs/reproducibility.md)

## Thesis Reference

Local thesis file:

- [毕业设计_JUNHO JO.pdf](/Users/jojunho/Documents/4-1/졸업논문/논문&자료/最终报告/论文/毕业设计_JUNHO JO.pdf)

Suggested citation format:

```bibtex
@thesis{jo2026hotspot,
  author = {Jo, Junho},
  title = {Multi-Dimensional Load Imbalance Detection and Hotspot Identification for Cloud Storage Systems},
  school = {Shanghai Jiao Tong University},
  year = {2026}
}
```

## Current Limitations

- The full experiment pipeline is still notebook-first
- Industrial raw data are private
- Evaluation keys and validation protocols need further standardization
- The package in `src/` is an initial extraction layer, not yet the full migrated pipeline

## Roadmap

- migrate notebook logic into importable modules
- standardize configs and artifact naming
- add regression tests for labeling and episode extraction
- expose a single CLI pipeline entrypoint
- provide sanitized demo data or a synthetic benchmark

