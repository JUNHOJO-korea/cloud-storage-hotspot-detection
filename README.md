# Cloud Storage Hotspot Detection

[![CI](https://github.com/JUNHOJO-korea/cloud-storage-hotspot-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/JUNHOJO-korea/cloud-storage-hotspot-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](./pyproject.toml)

Detect persistent node-level and port-level hotspots in cloud storage systems using:

- multi-dimensional load imbalance features
- robust median/MAD-based SPC
- event-level hotspot construction
- metric-learning-based hotspot state scoring

This repository turns a thesis project into a public research engineering codebase for hotspot identification from real-world storage monitoring data.

## Why This Project

In production storage systems, serious hotspots are usually not a single spike in throughput, IOPS, or latency. They are structured states where workload remains concentrated on a small subset of nodes or ports over time.

This project models hotspot detection as:

1. distribution imbalance detection
2. robust abnormal-state detection
3. event construction with temporal persistence
4. structured state discrimination with metric learning

The result is not just an anomaly flag. The pipeline is designed to answer:

- where the hotspot occurs
- how long it lasts
- which entities dominate it
- how strongly it separates from normal states

## What It Does

The full pipeline covers:

- unified long-table modeling for system, node, and port monitoring data
- imbalance metrics: HHI, Gini, Theil, Top1 Share, normalized Top1 Share
- robust SPC using median and MAD instead of mean and standard deviation
- persistence windows and load gating
- event-level hotspot segmentation
- culprit entity extraction at peak timestamps
- weighted metric learning from weak labels derived by the rule-based pipeline
- comparison against Mahalanobis distance, EWMA SPC, and Random Forest baselines

## Current Results

Based on the current thesis pipeline and saved experiment artifacts:

- `1,076` final hotspot events from the rule-based pipeline
- `859` learned hotspot episodes in the final node-centered learned pipeline
- `4,690` learned hotspot points in the final learned scoring output

Current final learned feature weighting emphasizes structural imbalance features:

- strongest learned signals: `raw_imb_gini`, `raw_imb_hhi`, `sev_imb_hhi`, `raw_imb_theil`
- weak contextual contribution: `load_ratio`
- near-zero contribution in current run: `active_ratio`

## Method Overview

```text
Raw monitoring data
  -> unified long table
  -> imbalance metric construction
  -> robust SPC on imbalance streams
  -> persistence + load gating
  -> event-level hotspot segments
  -> point labels from hotspot windows
  -> metric learning on hotspot vs clean states
  -> final point scores and hotspot episodes
```

## Repository Structure

```text
.
├── README.md
├── docs/                     # project docs and methodology notes
├── examples/                 # small runnable examples
├── notebooks/                # notebook guidance and migration target
├── src/hotspot_detection/    # package skeleton and reusable core utilities
├── tests/                    # unit tests for extracted logic
├── scripts/                  # environment and pipeline helpers
├── sample2.ipynb             # main research notebook
├── sample.ipynb              # earlier pipeline variant
└── case experiment.ipynb     # case-family screening notebook
```

## Quick Start

### 1. Install

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate hotspot-detection
pip install -e .
```

### 2. Verify the package

```bash
python -m hotspot_detection.cli show-config
python -m hotspot_detection.cli describe-notebooks
pytest
```

### 3. Run the research pipeline

Open notebooks in this order:

1. `sample2.ipynb`
2. `case experiment.ipynb`

## Main Notebooks

### `sample2.ipynb`

Primary end-to-end notebook containing:

- data loading and normalization
- imbalance metric generation
- robust SPC
- hotspot segment construction
- Mahalanobis baseline
- weighted metric learning
- final scoring, validation, and episode extraction

### `sample.ipynb`

Earlier experimental notebook with more visibly separated node and port blocks.

### `case experiment.ipynb`

Follow-up notebook for case-family screening and alternative feature space comparison.

## Data

The original industrial monitoring data are private and are not included in this repository.

Expected local raw inputs:

- `data_system.parquet`
- `data_node.parquet`
- `data_port.parquet`

See:

- [docs/data_schema.md](docs/data_schema.md)
- [data/README.md](data/README.md)

If this project is extended for broader public use, the next recommended step is to add:

- a synthetic demo dataset
- a schema validator
- a deterministic end-to-end sample run

## Documentation

- [Project Overview](docs/project_overview.md)
- [Methodology](docs/methodology.md)
- [Data Schema](docs/data_schema.md)
- [Evaluation](docs/evaluation.md)
- [Reproducibility](docs/reproducibility.md)

## Project Status

This repository is in a transition phase:

- the full scientific pipeline is still notebook-first
- the reusable package layer in `src/` has started but is not yet complete
- CI and tests currently cover the extracted utility layer, not the full notebook pipeline

That means this repository is already useful for:

- studying the methodology
- reproducing the notebook-based workflow locally
- turning the codebase into a stronger research engineering project

It is not yet a polished plug-and-play library.

## Roadmap

- migrate core notebook logic into `src/hotspot_detection/`
- standardize artifact contracts and evaluation keys
- add regression tests for hotspot labeling and episode extraction
- provide synthetic or sanitized demo data
- expose a single CLI pipeline runner

## Citation

If you use this project or build on the thesis, cite:

```bibtex
@thesis{jo2026hotspot,
  author = {Jo, Junho},
  title = {Multi-Dimensional Load Imbalance Detection and Hotspot Identification for Cloud Storage Systems},
  school = {Shanghai Jiao Tong University},
  year = {2026}
}
```

## Thesis Reference

The thesis PDF is currently stored locally and is not committed to this repository.

Recommended follow-up:

- add a public thesis link if redistribution is allowed
- or add a short `docs/thesis_summary.md` page for repository readers

## License

Released under the [MIT License](./LICENSE).
