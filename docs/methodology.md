# Methodology

## Research Problem

Hotspots in cloud storage systems are treated as persistent and structured workload concentration states, not isolated outliers.

## Feature Engineering

Imbalance metrics:

- HHI
- Gini coefficient
- Theil index
- Top1 Share
- normalized Top1 Share

Metric-learning feature families:

- raw imbalance features
- SPC severity features
- contextual load ratio
- active entity ratio

## SPC Mainline

The rule-based detector uses:

- median as center line
- MAD-based robust sigma
- upper control limit with configurable `k`
- persistence window filtering
- load gating

## Hotspot Construction

- detect abnormal imbalance points
- merge adjacent points into segments
- identify peak timestamp
- assign hotspot tiers
- extract culprit entities from raw metrics

## Metric Learning

- positive states from strong hotspot peaks
- negative states from clean non-hot windows
- similar and dissimilar pair construction
- diagonal weighted metric learning
- final score projection and episode extraction

