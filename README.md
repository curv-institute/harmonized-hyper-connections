# Harmonized Hyper-Connections

Supporting code for the paper:

**Harmonized Hyper-Connections: Stabilizing Residual Transport Through Feedback on Geometry**
J. W. Miller
CURV Institute, January 2026

Paper: https://curv.institute/publications/harmonized-hyper-connections/

## Overview

This repository contains experimental code for regulating composite gain in Hyper-Connections through feedback on transport geometry. The approach maintains bounded applied gain with low variance while recovering task performance comparable to unconstrained variants.

## Modes

- **hc**: Unconstrained residual transport (Hyper-Connections baseline)
- **mhc**: Hard manifold projection via Sinkhorn normalization
- **harm**: Soft equilibrium via applied-gain feedback control (proposed method)

## Files

- `harmonizer_applied_gain.py` - Core implementation with all three routing modes
- `rift_mhc_sandbox.py` - Sandbox for exploring routing behaviors
- `make_figures.py` - Generate paper figures from experimental runs
- `experiments/` - Hyperparameter tuning scripts
- `runs/` - Experimental results and metrics
- `paper_figs/` - Generated figures

## Usage

```bash
python harmonizer_applied_gain.py --mode harm --steps 15000
```

## Requirements

- Python 3.10+
- PyTorch

## License

MIT
