# PF257-EGKEM-SPNCTR (Experiment Suite)

This repository contains the Python implementation and evaluation suite used for the **PF257-EGKEM-SPNCTR** image-encryption construction (prime-field block layer + SPN/CTR pixel layer + X25519-based EG-KEM wrapper).

## Repository layout

- `runner_pf257.py` — main entry point for running experiments.
- `pf257_egkem_spnctr_v1_x25519.py` — PF257-EGKEM-SPNCTR cipher core.
- `pf257_experiment_framework.py` / `pf257_experiment_base.py` — shared experiment infrastructure.
- `exp_00_...py` … `exp_13_...py` — the 14 experiment modules.
- `images/` — input test images (TIFF/TIF) used by the experiments.
- `RESULTS_PF257_PAPER/` — output directory used by default (also contains sample artifacts if present).

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

List experiments:
```bash
python runner_pf257.py list
```

Run the full suite:
```bash
python runner_pf257.py all
```

Run only the essential sanity check (lossless verification):
```bash
python runner_pf257.py essential
```

Run a single experiment by id:
```bash
python runner_pf257.py single 3
```

Outputs are written under `RESULTS_PF257_PAPER/` (configurable inside `runner_pf257.py`).

## Notes

- The code is written for reproducibility: deterministic mode is enabled by default in `runner_pf257.py`.
- The `images/` directory must exist and contain at least one supported image file.

## License

MIT (see `LICENSE`).
