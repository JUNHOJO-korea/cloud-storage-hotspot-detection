# Contributing

Thanks for contributing to this project.

## Current State

The repository is transitioning from a notebook-first research codebase to a maintainable Python package. Contributions that improve clarity, reproducibility, and modularity are especially valuable.

## Good First Contribution Areas

- migrate notebook logic into `src/hotspot_detection/`
- improve schema validation and artifact contracts
- add regression tests for hotspot labeling and episode extraction
- document evaluation protocols more precisely
- provide synthetic demo data with the same schema

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Before Opening a Pull Request

1. Run tests:

```bash
pytest
```

2. Keep changes focused and small.
3. If you change research logic, update the relevant docs in `docs/`.
4. If you change input or output contracts, update `docs/data_schema.md`.

## Pull Request Expectations

- explain what changed
- explain why the change is needed
- note whether the change affects research outputs or only project structure
- include validation evidence when touching core pipeline logic

