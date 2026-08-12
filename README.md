# Epistatic Circuits

**Exhaustive Interaction Decomposition of Transformer Circuits**

Circuit discovery methods rank components by individual importance and treat the resulting
circuits as additive. This repository provides exact, non-human-curated ground truth for
pairwise interaction structure in a real trained circuit, from exhaustive Walsh–Hadamard
coalition sweeps over subcircuits of GPT-2 small, and benchmarks three discovery methods
against it.

## Key results

| Circuit | Method | Faithfulness (Δ) | Order-1 | Order-2 | Order-3+ |
|---------|--------|:-:|:-:|:-:|:-:|
| C2 | EAP-IG | 1.13 | 90.4% | 9.0% | 0.6% |
| C3 | Canonical (Wang et al.) | 1.39 | 78.6% | 18.4% | 3.0% |
| C4 | Random baseline | 0.02 | 97.9% | 2.1% | 0.0% |
| IC-15 | Information content | −0.00 | 94.9% | 4.6% | 0.5% |
| C5 | **Walsh discovery (ours)** | **1.52** | 80.9% | 16.8% | 2.3% |

Energy fractions are non-constant (order-0 removed). Walsh discovery (C5) achieves the highest faithfulness among all automated methods while maintaining high epistasis, making it Pareto-optimal on the faithfulness–epistasis frontier.

## Repo structure

```
src/                    Core analysis code
  circuits.py           Circuit definitions (head lists, metadata)
  coalition_sweep.py    Exhaustive 2^n coalition evaluation
  walsh.py              Walsh-Hadamard transform and energy spectrum
  walsh_discovery.py    LASSO-Walsh circuit discovery from random samples
scripts/                Modal/RunPod scripts for GPU computation
  modal_c*_coalition_sweep.py
  modal_walsh_discovery.py
data/                   Coalition tables and intermediate results (gitignored, see Data section)
paper/                  LaTeX source
results/                Analysis outputs (JSON, figures)
```

## Data

Coalition tables (2^15 = 32,768 coalitions × 512 prompts per circuit, both zero and mean ablation) are too large for git. Download from the Modal volumes or regenerate:

```bash
# Download pre-computed coalition tables
modal volume get <volume-name> <file>.npz data/

# Or regenerate (requires GPU)
modal run --detach scripts/modal_coalition_sweep.py
```

## Setup

```bash
uv sync
```

## Reproducing

```bash
# 1. Coalition sweeps (GPU, ~5h per circuit per ablation type)
modal run --detach scripts/modal_coalition_sweep.py

# 2. Walsh spectrum analysis (local, fast)
uv run python src/walsh.py

# 3. Walsh discovery from random samples (GPU for sampling, local for LASSO)
modal run --detach scripts/modal_walsh_discovery.py
uv run python src/walsh_discovery.py

```

## Citation

Paper in preparation.

## License

MIT
