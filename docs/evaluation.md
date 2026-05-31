# Evaluation

This repository currently uses several evaluation views. They should be kept separate in reports.

## Point-Level Evaluation

Typical metrics:

- Precision
- Recall
- F1
- Jaccard

## Score-Based Evaluation

When probabilistic or continuous outputs are available:

- ROC-AUC
- Average Precision
- PR curve

## Episode-Level Evaluation

Compare event outputs by:

- number of episodes
- overlap with rule-based episodes
- duration distribution
- peak score distribution

## Current Caveat

Evaluation keys are not fully standardized across all notebook blocks. Before public benchmarking, define canonical keys for:

- point-level labels
- context-level labels
- episode-level labels

