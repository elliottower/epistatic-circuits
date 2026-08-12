# Lab notebook — ablation-primitive dependence of head-pair epistasis

Session of 2026-08-07. **Handoff document.** Nothing here has been added to the
paper. This records what was measured, what was got wrong, and what the other
session should decide about.

---

## 1. The question

Head-pair epistasis is measured by ablating subsets of heads and reading the
order-2 Walsh coefficient. "Ablation" is not one operation — zero, mean and
resample are three different primitives. The measured interaction `W_ij` may
depend on which one was used, and nobody reports that dependence.

## 2. What was measured

Same circuit, same heads, same coalitions, recomputed under each available
ablation primitive; then the agreement between primitives.

`scripts/modal_ceiling_exhaustive.py` (Modal, mounts the existing sweep volumes),
`scripts/ceiling_weightonly_r2.py` (single-circuit local version).
Results: `results/ceiling/ceiling_rows.json`, `ceiling_log.txt`.

**18 comparisons across IOI, RTI and GT.**

| statistic | value |
|---|---|
| sign agreement between primitives | **median 61.9%**, range 33.3–81.0% |
| comparisons at or below chance | 3 / 18 |
| Pearson r between primitives, 15-head circuits | 0.30, 0.42, 0.43, 0.63 (one at −0.07) |
| Pearson r, 7-head GT circuits | −0.31 to +0.82, five negative |

**The headline: roughly 38% of head pairs change the sign of their interaction
depending on which ablation primitive was used.** A pair reads as masking under
one primitive and synergistic under another — same-unit versus different-unit,
from a methodological choice nobody justifies in print.

This holds across three tasks and both circuit sizes.

## 3. What was got wrong, and it was got wrong three times

Recorded because the wrong version was stated confidently before it was checked.

**Attempt 1.** Claimed that because weights do not change across ablation
primitives, the cross-primitive correlation upper-bounds the R² of any
weight-based predictor — and therefore that P4 of the subspace pre-registration
(R² > 0.30) was scored against an unreachable threshold. Computed the bound as
`r²`.

**Error A — squaring.** The bound is `r`, not `r²`. For a signal-plus-noise
model the oracle predictor's R² equals the reliability, which the cross-primitive
correlation estimates directly. Using `r²` understates it by a factor `1/r`.

**Error B — generalising from one circuit.** The first version used only IOI c6.
Across all 18 comparisons the 15-head circuits give bounds of 0.30–0.63, which is
at or above P4's threshold. **P4 was testable and its failure is real. The
subspace conclusion stands; the objection to it does not.**

**Error C — the direction of the inequality.** An independent derivation showed
the bound requires *primitive-invariance of weight-explainable structure*: that
every primitive's weight-conditional mean is the same function of the weights up
to rescaling. Without it,

> r_ab = Corr(m_a, m_b) · sqrt(Omega_a · Omega_b) ≤ sqrt(Omega_a · Omega_b)

so `r` is a **lower** bound on the geometric mean of the true ceilings. The whole
framing runs backwards unless that assumption holds, and three primitives
identify the per-primitive reliabilities but cannot test it — that needs a fourth
and fifth indicator.

**Also:** the bound only constrains a predictor held *fixed* across primitives. A
model refit per primitive using weight features only can exceed it without
contradiction.

## 4. Two real problems with the subspace analysis — not the one claimed above

These came out of the same derivation and are worth acting on.

**GT is underpowered past the point of interpretability.** Seven heads gives 21
pairs, and a **pure-null model with five features averages R² = 0.25 in-sample**
at that size, reaching 0.49 at the 95th percentile. `gt_known`'s R² = 0.412,
recorded as the standout result, sits inside that null distribution.

**Pairs are dyadic.** 105 pairs come from 15 heads; the effective sample size is
about 43, not 105. Naive confidence intervals are too narrow (coverage 0.787 at
n=15) and the correlation estimate is biased downward. A head-level cluster
bootstrap restores nominal coverage.

**Recommendation:** cross-validate every R² with heads as the grouping unit, and
report GT as descriptive only.

## 5. Prior art — this is a solved problem in genetics

The finding is not that mechinterp is broken. It is that mechinterp has a problem
another field already quantified and wrote norms about.

- **Housden et al. 2017, *Nature Reviews Genetics*** — the citation to use.
  Different loss-of-function methods "can lead to substantially different
  outcomes"; discrepancies arise from "differences in timing, duration, strength
  and mechanism of LOF"; and critically, "the differences in phenotype between
  approaches may be informative of the underlying biology." The norm: "studies
  would benefit from the application of more than one approach to the same
  question." Obtained in full (accepted author manuscript).
- **Stainier et al. 2017, *PLOS Genetics*** — an actual community guideline. A
  knockdown "should be validated by comparison to a mutant, and if there is a
  discrepancy, by injection into embryos homozygous for a null allele", plus a
  mandatory dose-response curve.
- **Kok et al. 2015, *Developmental Cell*** — the empirical case that prompted
  the guidelines: "more than 70 percent of morphant phenotypes were not observed
  in respective mutants."
- **Sanson et al. 2018, *Nature Communications*** — the directly comparable
  number. CRISPR knockout vs CRISPRi agree at **R = 0.69** at gene level, treated
  as normal and informative. Our cross-ablation correlations are 0.30–0.63.

Calibration: our disagreement is of the same order as genetics', slightly worse
than CRISPRko/CRISPRi and considerably better than morpholino/mutant.

**Muller 1932 is available but should probably not be cited.** The primary scan
was obtained and transcribed (`reference/muller_1932_TRANSCRIPTION.md`) after two
automated attempts failed; all five allele classes are verified in his own words.
But Housden 2017 states the same point more clearly and more recently, and citing
a 1932 congress proceedings reads as antiquarian. The only thing in Muller not
covered by the modern sources is his p. 246 warning that a mutant behaving like
an absence is not thereby shown to *be* an absence — which transfers, but is not
load-bearing.

## 6. For the other session to decide

**Probably yes:**
- The sign-flip number. It is new, it is robust across 18 comparisons, and it is
  the only thing here untouched by the errors in §3.
- Housden and Sanson as the framing. They convert a methodological wobble into a
  recognised phenomenon with a published comparator.

**Probably no:**
- The ceiling argument in any form. Three attempts, three errors, and the version
  that survives is a diagnostic rather than a bound.
- Muller 1932.

**Needs a decision:**
- Whether to act on §4 — cross-validating R² by head and demoting GT would change
  numbers already written into `LAB_NOTEBOOK_subspace_epistasis.md`.
- Whether the ablation-series finding is a section of the epistasis paper or its
  own short methods note.

## 7. Files

```
scripts/ceiling_weightonly_r2.py        single-circuit, local
scripts/modal_ceiling_exhaustive.py     all circuits, Modal, mounts sweep volumes
results/ceiling/ceiling_rows.json       18 comparisons
results/ceiling/ceiling_log.txt         discovery + skip diagnostics
reference/                              genetics and neuroscience sources
reference/muller_1932_TRANSCRIPTION.md  verified quotes, read from page images
```

One implementation note worth keeping: the first exhaustive run silently dropped
every GT and induction circuit because the loader only recognised `logit_diff`
and `target_logits`/`foil_logits` as value keys, while those sweeps store
`prob_diff`. It reported them as "unreadable/incomplete." The loader now tries a
list of keys and records the specific reason when it bails.
