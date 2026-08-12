> **Provenance.** Reformatted from `docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md`.
> Original frozen at commit `d071339`. View: `git show d071339:docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md`

# Pre-registration: does weight geometry predict epistasis under some ablation
# primitives better than others?

**Status: FROZEN before any mean- or resample-ablation regression was fitted.**

---

## Question

`PREREG_SUBSPACE_EPISTASIS.md` asked whether residual-stream subspace overlap
predicts head-pair epistasis, and answered with a single regression against
epistasis measured under **one** ablation primitive. P4 failed (pooled
R² = 0.068 against a 0.30 threshold).

That test treats "the interaction between heads i and j" as one quantity. Two
measurements already in hand say it is not:

- **Within a primitive, W_ij is measured precisely.** Prompt split-half on the
  c6 IOI circuit gives Spearman–Brown 0.995 (`results/V13_RESULTS.md`).
- **Across primitives, W_ij disagrees substantially.** Zero vs mean on the same
  circuit gives r = 0.302, and across 18 comparisons roughly 38% of pairs change
  the sign of their interaction (`results/LAB_NOTEBOOK_ablation_series.md`).

Precise measurement plus systematic disagreement means these are three distinct
quantities, not one quantity measured with error. So the question "do the weights
predict epistasis?" has three answers, and only one has been checked.

**This registration asks whether the weights predict one primitive's epistasis
better than another's.**

---

## Disclosure: observed vs unobserved

### Observed
- The zero-ablation result: pooled R² = 0.068, pooled Spearman ρ_OV = 0.173,
  partial ρ(QK|OV) = 0.101, and the close-layer/far-layer split (0.344 / 0.009).
  **One of the three arms of this experiment is therefore already seen**, and its
  result is reported here rather than treated as an outcome.
- Cross-primitive agreement statistics, as above.
- Within-primitive reliability, as above.
- Which circuits and primitives exist on the sweep volumes.

### Genuinely unobserved
- Every regression against mean-ablation epistasis.
- Every regression against resample-ablation epistasis.
- Every cross-validated R², under any primitive, including zero — the published
  0.068 is in-sample.
- All quantities in the predictions below.

---

## Design

**Target.** Signed order-2 Walsh coefficient W_ij. Signed rather than magnitude:
|W_ij| admits a heteroskedasticity artifact in which per-pair noise scale shared
across primitives produces correlation with no invariant signal. Magnitude is
reported as a secondary only.

**Predictors.** Unchanged from `PREREG_SUBSPACE_EPISTASIS.md` — OV subspace
overlap and QK composition from LayerNorm-folded effective weights, plus layer
distance. No new features, so any difference between arms is attributable to the
primitive rather than to the model.

**Primitives.** Zero, mean and resample, wherever a circuit has the same head set
under more than one.

**Circuits.** The 15-head circuits (IOI and RTI) carry the confirmatory analysis.
GT's 7-head circuits give 21 pairs, where a five-predictor null model averages
in-sample R² ≈ 0.25; **GT is reported descriptively and is excluded from every
decision rule below.**

**Metric.** Cross-validated R², with **heads as the grouping unit** — pairs
sharing a head never straddle a fold. Pairs are dyadic, so the effective sample
size of a 15-head circuit is roughly 43 rather than 105, and in-sample R² is
inflated. The published 0.068 is in-sample and is not comparable to the CV values
produced here; both are reported for the zero arm.

**Intervals.** Head-level cluster bootstrap, 1000 resamples, percentile CIs.
Naive intervals under-cover badly on dyadic data.

---

## Predictions

Registered before any mean or resample regression was fitted.

**P1 (primary).** Mean- and resample-ablation epistasis are better predicted by
weight geometry than zero-ablation epistasis, by at least 0.10 CV-R² in at least
half of the eligible 15-head circuits.

*Mechanism.* Zero ablation removes a head's entire contribution, which changes
the residual-stream norm and therefore rescales LayerNorm for every downstream
head — a global side effect. Mean ablation leaves the mean contribution in place,
so the norm is roughly preserved and the perturbation is closer to surgical.
Resample likewise preserves scale while altering content. If the weight-blind
component of epistasis is largely LayerNorm-mediated, zero ablation should carry
the most of it and be the hardest to predict from static weights.

**P2.** All three arms nonetheless have low CV-R² — below 0.20 pooled. Weight
geometry is a weak predictor of epistasis under every primitive.

**P3.** The close-layer/far-layer split reported for zero ablation replicates
under mean and resample, with close-layer correlations exceeding far-layer ones
in every arm.

**P4.** Cross-validated R² is materially lower than the published in-sample
value, by at least 0.03 on the zero arm.

**Prediction on the headline: P1 confirms and P2 confirms.** Geometry does better
on the less disruptive primitives and is weak everywhere. If P1 fails and all
three arms are equally predicted, the original subspace conclusion is
strengthened rather than weakened, and that is worth reporting.

---

## What each outcome means

| outcome | reading |
|---|---|
| one primitive much better predicted | epistasis is partly geometric, and which part depends on the perturbation; "does geometry predict epistasis" is under-specified without naming a primitive |
| all arms equally poor | the original conclusion holds and is now properly supported: geometry does not predict head-pair epistasis under any primitive |
| all arms well predicted | the published in-sample 0.068 was an artifact of something other than the primitive; investigate before reporting anything |

---

## Abort conditions

1. **Head-set mismatch.** If a circuit's head set differs between primitives, it
   is dropped and the drop is reported. The comparison requires identical heads.
2. **Fold degeneracy.** If grouped cross-validation leaves a fold with fewer than
   5 pairs, that circuit is reported descriptively only.
3. **Predictor failure.** If OV overlap or QK composition cannot be computed for
   a circuit — missing weights, unfoldable LayerNorm — that circuit is dropped
   rather than fitted with a reduced feature set.
4. **Negative CV-R².** Expected and not an abort. Reported as-is, never clipped
   to zero, since clipping would bias every arm upward.

---

## What this cannot establish

The three primitives are three specific choices rather than a sample from a
population of perturbations, so nothing here generalises to ablation primitives
not tested. And a difference between arms shows that the weights speak to one
perturbation more than another; it does not show which perturbation is the
*right* one. No such criterion is proposed here.

---

## Freeze

SHAs recorded in `FREEZE_PRIMITIVE_SPECIFIC_GEOMETRY.txt` alongside this file. No
mean- or resample-ablation regression is fitted before that file exists.

---

# AMENDMENTS — recorded before any result was read

The first run completed and its output was **deleted from the results volume
without being inspected**. No CV-R² from it has been seen by anyone.

## A1. Cross-validation did not deliver the guarantee it promised

The frozen text says pairs sharing a head never straddle a fold. The
implementation used `GroupKFold` with the **first head** of each pair as the
group, which does not achieve that: head 7 falls in group 3 via pair (3,7) and
group 5 via pair (5,7), so it can sit in the training fold for one of its pairs
while another is held out.

**Amended to leave-both-heads-out.** For each pair (i,j) the model is trained
only on pairs (a,b) with a ∉ {i,j} and b ∉ {i,j}, so no head is shared between
train and test. With 15 heads that leaves 78 training pairs per fit; with 7 heads
it leaves 10, which is why the 7-head circuits remain descriptive only.

**Magnitude of the defect, measured rather than assumed.** On synthetic dyadic
data where the features carry head-level information and the target has head
random effects — the realistic case — the buggy split returns R² = +0.685 against
+0.645 for the corrected one. **The optimism is about 0.04 R², and it applies to
every arm equally.**

Consequences, stated plainly:
- **P1 would have survived.** It compares arms, and a bias common to all arms
  largely cancels in the difference. Its 0.10 threshold is well above 0.04.
- **P2 and P4 would not have.** Both concern absolute CV-R² values, which the
  leak inflates directly.

The run was voided for P2 and P4. Voiding it was more caution than P1 required,
and that is recorded here rather than presented as necessity.

## A2. The bootstrap was not a cluster bootstrap

It resampled heads but then selected pairs by first-head membership, inheriting
the same defect. **Amended** to resample heads and keep only pairs whose *both*
endpoints survive.

## A3. Arm comparability is now recorded

The frozen text checks that arms share a head set and nothing else. Sweeps differ
in which value key they store (`logit_diff` versus `prob_diff`) and in prompt
count, and a difference in either would make a between-arm difference
unattributable to the ablation primitive.

**Amended:** every row records `value_key` and `n_prompts`. Any circuit whose
arms disagree on either is reported as non-comparable and excluded from P1,
regardless of what its numbers show. This is a disclosure requirement, not a
new hypothesis.
