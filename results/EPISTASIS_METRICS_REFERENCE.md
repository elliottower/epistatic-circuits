# Epistasis Metrics Reference

## What we have

We compute three epistasis metrics from the full 2^n coalition sweep data.
Each answers a different question about how heads interact within a circuit.

### 1. Walsh Interaction Fraction (WIF) — PRIMARY

**Formula**: `WIF = 1 - (order-1 energy) / (non-constant energy)`

**What it is**: Walsh-Hadamard transform decomposes v(S) into an orthogonal
basis indexed by interaction order. Order-0 is the constant (empty-set value).
Order-1 captures individual-head effects. Order-2+ captures interactions.
WIF = fraction of non-constant variance that lives at order >= 2.

**Equivalent to**: 1 - R^2 of the best additive (linear) model.

**Range**: [0, 1]. Zero = purely additive. One = no individual-head effects.

**Uses**: ALL 2^n coalition values (full data).

**Why primary**: Stable. Never goes infinite. Directly interpretable as
"fraction of circuit behavior that requires head combinations."

### 2. LOO Epistasis — LEGACY (from the RTI paper)

**Formula**: `epistasis = 1 - sum_i[v(N) - v(N\{i})] / [v(N) - v(empty)]`

**What it is**: Checks whether the sum of leave-one-out marginal contributions
accounts for total faithfulness. If heads are additive, LOO marginals sum to
faithfulness and epistasis = 0.

**Uses**: Only 2 of the 2^n coalition levels (the full set and each n-1 subset).
Ignores all smaller coalitions.

**Problem**: Denominator is faithfulness. When faithfulness is near zero
(random circuits, negative-faithfulness circuits under resample ablation),
LOO blows up. Observed range in our data: -3211% to +126%.

**Keep for**: Backward compatibility with RTI paper numbers. Report in
supplement/appendix. Flag as UNSTABLE when |faithfulness| < 0.1.

### 3. Total Shapley Interaction Index (TSII) — SECONDARY

**Formula**: `TSII = sum_{i<j} |SII(i,j)|` where each SII(i,j) averages
the discrete derivative delta_{ij}(S) = v(S+i+j) - v(S+i) - v(S+j) + v(S)
over all coalitions S not containing i or j, weighted by Shapley weights.

**What it is**: Total absolute pairwise interaction. Positive SII = synergy
(heads amplify each other). Negative SII = redundancy (heads substitute).

**Uses**: ALL 2^n coalition values.

**Problem**: Not normalized by circuit size. A 15-head circuit has 105 pairs;
a 7-head circuit has 21 pairs. Raw TSII is not comparable across sizes.
Could normalize by n*(n-1)/2 to get mean |SII| per pair.

**Keep for**: Identifying which specific head PAIRS interact most. The per-pair
SII values are the most actionable output for mechanistic follow-up.

## Walsh energy spectrum (not a single metric)

The full order-by-order breakdown: what fraction of variance lives at each
interaction order 0, 1, 2, 3, ..., n. Reports order-1 fraction, order-2
fraction, and order-3+ fraction. This is the raw data behind WIF.

Key pattern: order-3+ is generally small (< 5% for most circuits). Most
interaction is pairwise (order-2). This means head interactions in
transformer circuits are predominantly pairwise, not higher-order.

## What the RTI paper used

The RTI paper used **only LOO epistasis**. The "55%" for GT known and "56%"
for induction known are LOO values. The paper never decomposed into
higher-order interactions or used the Walsh spectrum.

Code locations:
- `scripts/v13_bootstrap.py:27-31` — `rti_epistasis()` function
- `scripts/modal_rti_epistasis_bootstrap.py:79` — same formula
- Every sweep script (modal_gt_sweep, modal_induction_sweep, etc.)

## What to report in this paper

**Table columns**: Faithfulness, WIF (primary), order-1%, order-2%, order-3+%
**Supplement**: LOO epistasis (for comparison with RTI paper), TSII, top-3
head pair interactions per circuit.

## Observed ranges across all data (47 circuit x ablation combos)

| Metric | Min | Max | Typical real circuit |
|--------|-----|-----|---------------------|
| WIF | 0.1% (random) | 41.9% (EAP RTI zero) | 4-13% (mean abl) |
| LOO | -3211% (IC15 mean) | +126% (C6 resample) | 20-60% (when stable) |
| TSII | 0.025 (GT random) | 9.034 (EAP RTI zero) | 0.4-4.5 |
| order-2 | 0.1% | 32.5% | 3-18% |
| order-3+ | ~0% | ~8% | 1-3% |
