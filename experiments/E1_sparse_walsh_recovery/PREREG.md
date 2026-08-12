> **Provenance.** Reformatted from `docs/PREREG_SPARSE_WALSH_RECOVERY.md`.
> Original frozen at commit `be71729`. View: `git show be71729:docs/PREREG_SPARSE_WALSH_RECOVERY.md`

# Pre-registration: Sparse Walsh Circuit Interaction Discovery

## Question

Can sparse random sampling + compressed sensing recover the pairwise
interaction structure of a neural circuit from O(k log N) forward passes
instead of the full 2^n coalition sweep — making Walsh-based interaction
analysis practical for circuits with 20-50 heads in large models?

## Motivation

Exact coalition sweeps require 2^n forward passes. For 15 heads that is
32,768 — feasible but expensive. For 20 heads it is 1,048,576 — borderline.
For 30 heads it is ~10^9 — impossible. If sparse recovery works, the same
interaction map (which pairs of heads interact and by how much) can be
obtained from far fewer random coalition evaluations, turning Walsh
interaction analysis from a small-circuit-only tool into a general circuit
discovery method.

## Theoretical bounds

Two sample-complexity regimes apply:

- **Generic RIP bound** (Gaussian/Bernoulli random matrices):
  M = O(k log(N/k)). For k=120, N=32768: 120 * log(273) ≈ 670.

- **Structured Fourier/Hadamard bound** (our actual setting — random
  coalitions through a Walsh basis): M = O(k log(k) log(N)).
  For k=120, N=32768: 120 * log(120) * log(32768) ≈ 6000.

Our measurement matrix is structured (subsampled Walsh), not generic
Gaussian, so the correct regime is the latter. Recovery should become
reliable somewhere in the M = 700-6000 range, with the structured
bound as the conservative ceiling.

## Design

### Phase 1: Validation on existing exact data

Use existing IOI, RTI, and GT exact sweeps as ground truth. Subsample M
random coalitions from the full 2^n, recover Walsh coefficients, compare
to exact. This proves the recovery works before spending GPU time.

**Data:** Exact coalition values from Modal volumes (already computed).
15-head circuits (IOI, RTI) and 7-head circuits (GT).

**Target coefficients:** n order-1 + n(n-1)/2 order-2 Walsh coefficients.
k = 120 for 15 heads, k = 28 for 7 heads.

### Phase 2: New experiment on a larger circuit

Pick a circuit where exact Walsh is infeasible (20-25 heads in GPT-2, or
15+ heads in a larger model). Run M forward passes with random coalition
ablations. Recover the interaction map via sparse Walsh. Compare the
discovered interaction structure to EAP edge scores on the same circuit.

**Candidate circuits:**
- Top-20 EAP heads for IOI in GPT-2 small (2^20 = 1M exact, ~2000 sparse)
- Top-15 heads for IOI in GPT-2 medium (same task, bigger model)
- Top-20 heads for greater-than in GPT-2 small

**Comparison:** Rank-correlate the recovered order-2 Walsh coefficients with
EAP edge importance scores. If Walsh and EAP agree on which pairs interact
most, both methods are measuring the same structure. If they disagree,
that disagreement is itself a finding (different methods, different
interaction maps — the selective pressures thesis).

### Phase 3: Method comparison

If Phase 1 validates and Phase 2 is feasible, compare sparse Walsh to EAP
directly as circuit discovery methods:
- Do they identify the same top-k interacting pairs?
- Where do they disagree, and which is more faithful?
- What is the compute cost per method?

## Methods (all implemented in one script)

1. **Monte Carlo (MC):** Direct estimation. w_S = (1/M) sum f(c) × chi_S(c).
   Unbiased, high variance at small M. Baseline.

2. **LASSO:** Sparse regression on Walsh basis (order <= 2). LassoCV for
   alpha selection. Standard compressed sensing. Sample complexity
   matches OMP at O(k ln N).

3. **OMP (Orthogonal Matching Pursuit):** Greedy sparse recovery. Tropp &
   Gilbert: sufficient measurements = 2k log(n-k), matching LASSO under
   similar SNR. Sparsity level selected via 5-fold cross-validation
   (OrthogonalMatchingPursuitCV), matching LASSO's CV-based alpha
   selection — neither method receives oracle knowledge of k.

4. **Amrollahi et al. (2019) algorithm** (Tier 2, if baseline works):
   Purpose-built for low-order sparse Fourier/Walsh set functions.
   Exploits the pairwise (d=2) structure directly: sample complexity
   O(k log n) where n = number of heads (not N = 2^n coalitions).
   For k=120, n=15: dramatically fewer samples than generic CS.
   Runtime O(kn log²k log n). This is the correct Tier 2 target for
   our exact problem structure — not SPRIGHT, which is for the
   general sparse WHT without interaction-order constraints.

## Evaluation metrics

For Phase 1 (vs exact ground truth):
- Pearson r between recovered and exact coefficient vectors
- Spearman rank correlation
- Top-10 recall (fraction of 10 largest exact coefficients in recovered top-10)
- Normalized RMSE

For Phase 2 (vs EAP):
- Spearman correlation between |Walsh order-2| and EAP edge scores
- Overlap of top-k pairs identified by each method
- Qualitative: do the disagreements make sense?

## Sample sizes (Phase 1)

M = 50, 100, 200, 500, 1000, 2000, 5000
10 random trials per (method, M, circuit) combination.

## Predictions

### Theoretical brackets

The generic RIP bound (M ≈ 700) and structured Fourier bound (M ≈ 6000)
bracket where recovery should become reliable. We predict:

- Below M ≈ 700 (below generic RIP): recovery fails (r < 0.7)
- M = 700-2000: recovery improves, reaching r > 0.9 somewhere in range
- M = 2000-6000: recovery is reliable (r > 0.95)
- Above M ≈ 6000 (above structured bound): near-exact (r > 0.99)

### Phase 1: 15-head circuits (k=120, N=32768)

| M | M/k | Regime | LASSO/OMP r | MC r | LASSO top-10 |
|---|-----|--------|-------------|------|--------------|
| 50 | 0.4 | far below RIP | < 0.3 | < 0.2 | < 30% |
| 100 | 0.8 | below RIP | 0.3-0.5 | < 0.3 | 30-50% |
| 200 | 1.7 | at generic RIP floor | 0.5-0.7 | 0.3-0.5 | 50-70% |
| 500 | 4.2 | generic-to-structured | 0.7-0.85 | 0.5-0.7 | 70-85% |
| 1000 | 8.3 | mid-range | 0.85-0.95 | 0.7-0.85 | 85-95% |
| 2000 | 16.7 | approaching structured | 0.95-0.99 | 0.85-0.95 | > 90% |
| 5000 | 41.7 | above structured | > 0.99 | > 0.95 | > 95% |

### Phase 1: 7-head circuits (k=28, N=128)

| M | M/k | LASSO/OMP r | MC r | LASSO top-10 |
|---|-----|-------------|------|--------------|
| 50 | 1.8 | 0.7-0.9 | 0.5-0.7 | 70-90% |
| 100 | 3.6 | > 0.95 | 0.8-0.9 | > 90% |

### Method ordering

LASSO ≈ OMP > MC at all sample sizes. OMP and LASSO have matching
theoretical sample complexity (2k log(n-k) sufficient). OMP is faster
and parameter-free; LASSO may have a slight edge in noisy settings.

### Phase 2: Walsh vs EAP

Prediction: moderate agreement (Spearman rho 0.3-0.6) on pair rankings.
Walsh measures actual coalition interaction; EAP measures linearized
gradient attribution. They should agree on the strongest pairs but
diverge on weaker ones where nonlinearity matters.

## Success criteria

**Phase 1 passes if** LASSO or OMP achieves r > 0.95 and top-10 recall
> 90% at M <= 2000 on 15-head circuits (16x compression). If this
requires M > 2000, the method works but is in the expensive structured-
bound regime rather than the cheap generic-RIP regime.

**Phase 1 fails if** r > 0.95 requires M > 5000 (< 7x compression on
15 heads), meaning sparse recovery offers marginal savings over exact.

**Phase 2 is worth running if** Phase 1 passes. Phase 2 succeeds if
the sparse Walsh interaction map on a 20-head circuit is computable
in < 1 GPU-hour and produces a meaningful (rho > 0.3) correlation
with EAP edge scores.

**The method is competitive with EAP as a discovery tool if** its
compute cost (M forward passes with coalition ablation) is within 2x
of EAP's cost (one forward + backward pass per prompt, times number
of prompts) for the same circuit size, AND the interaction structures
they identify have at least moderate agreement (Spearman rho > 0.3).
If sparse Walsh is cheaper but finds different structure, that is a
finding about method sensitivity, not a failure.

**The method has paper-level significance if** it works at 20+ heads
where exact Walsh is infeasible, producing interaction maps that
either agree with or meaningfully diverge from EAP.

## What this tests

- Whether compressed sensing on the Walsh basis is practical for
  circuit interaction discovery
- Whether Walsh and EAP measure the same interaction structure
- Whether sparse Walsh scales to circuits too large for exact sweeps

## What this does NOT test

- Amrollahi et al. (2019) low-order sparse Fourier algorithm — the
  correct Tier 2 follow-up if LASSO/OMP baseline works. Its O(k log n)
  bound (using n=15 heads, not N=32768) should be dramatically cheaper.
- Adaptive sequential sampling — future work
- Models beyond GPT-2 family — future work (but the method is
  model-agnostic by construction)

## Key references

- Candès & Tao (2005): RIP and generic CS recovery bounds
- Bourgain (2014) / Rudelson & Vershynin (2008): subsampled Fourier/
  Hadamard RIP upper bounds, O(k log k log N)
- Li & Nakos (2019, arXiv:1903.12135/1903.12146): improved lower bounds
  for sparse reconstruction from subsampled Hadamard matrices,
  Omega(k log k log(N/k))
- Tropp & Gilbert (2007): OMP sample complexity = 2k log(n-k)
- Amrollahi et al. (2019): "Efficiently Learning Fourier Sparse Set
  Functions" — O(k log n) for pairwise sparse Walsh, the Tier 2 target
- Poelwijk et al. (2016, 2019): Walsh decomposition of fitness landscapes
