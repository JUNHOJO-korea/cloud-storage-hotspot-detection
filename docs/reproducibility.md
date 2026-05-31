# Reproducibility

## Current Constraints

- the full pipeline is notebook-first
- industrial source data are private
- some artifacts are loaded from previous notebook outputs

## Minimum Reproducibility Checklist

1. Install the pinned environment
2. Prepare input parquet files with the documented schema
3. Run the main notebook or migrate the logic into `src/`
4. Save intermediate artifacts with versioned names
5. Record split strategy and threshold parameters

## Recommended Improvements

- move all absolute paths into config files
- define a single artifact manifest
- freeze random seeds consistently
- add regression tests for hotspot counts and score distributions

