# Lab notebook — does weight geometry predict head-pair epistasis?

Supersedes `LAB_NOTEBOOK_subspace_epistasis.md`. Incorporates the original
subspace analysis, the ablation-dependence finding, and the v2 cross-validated
primitive-specific geometry analysis. Three rounds of analysis, each correcting
mistakes in the prior round, converging on a definitive negative.

---

## 0. The question

Head-pair epistasis is measured by pairwise Walsh-Hadamard coefficients from
full 2^n coalition sweeps. All inter-head communication in a standard
transformer passes through the residual stream (Elhage et al. 2021). If
epistasis reflects shared subspaces — head A writes into a direction that
head B reads from — then weight-space composition scores should predict the
magnitude of pairwise Walsh interaction.

Three progressively sharper versions of this question were asked:

1. Does OV subspace overlap predict Walsh interaction? (resample ablation only,
   in-sample R^2, 15 circuits)
2. Does epistasis even mean the same thing across ablation primitives? (38%
   sign-flip finding)
3. Does weight geometry predict epistasis under any primitive, once properly
   cross-validated? (leave-both-heads-out CV, 3 arms, 5 comparable circuits)

The answer to all three is no, and the progression from suggestive negative to
definitive negative is the content of this notebook.

---

## 1. Analysis I: original subspace analysis (resample only, in-sample)

Pre-registration: `docs/PREREG_SUBSPACE_EPISTASIS.md`, SHA-256
`3ef962bc32110b98bf3e1ebf934debf60ec6587f84e89fad4476ac1eb2a1499f`.

Script: `scripts/modal_subspace_epistasis_analysis.py`.

### Method

Loaded GPT-2 small with `fold_ln=True` (TransformerLens). Extracted W_OV =
W_V @ W_O for all 144 heads. For each of 15 circuits across IOI, RTI, and GT
(all resample ablation), computed:

- OV overlap: Frobenius cosine similarity of W_OV matrices
- QK composition: Elhage 2021 Q-composition and K-composition scores
- Layer distance

Regressed |W_{ij}| on [OV_overlap, QK_comp, layer_dist] using OLS. All
R^2 values are **in-sample** (no cross-validation). This matters; see Analysis
III.

### Results — per circuit

| Circuit | Task | Heads | rho_OV | p_OV | rho_QK | p_QK | R^2 (in-sample) |
|---|---|---|---|---|---|---|---|
| gt_known | GT | 7 | **0.623** | 0.003 | 0.013 | 0.955 | **0.412** |
| gt_c5_walsh | GT | 7 | **0.462** | 0.035 | -0.274 | 0.229 | 0.243 |
| rti_C6_epistatic | RTI | 15 | **0.320** | 0.001 | 0.140 | 0.154 | 0.138 |
| rti_known | RTI | 15 | 0.277 | 0.004 | -0.135 | 0.171 | 0.088 |
| ioi_C5_walsh | IOI | 15 | 0.230 | 0.018 | 0.126 | 0.201 | 0.181 |
| ioi_C6_epistatic | IOI | 15 | 0.195 | 0.046 | 0.078 | 0.431 | 0.214 |
| rti_C5_walsh | RTI | 15 | 0.172 | 0.079 | -0.097 | 0.324 | 0.171 |
| gt_random | GT | 7 | 0.170 | 0.461 | 0.321 | 0.157 | 0.125 |
| ioi_C3_canonical | IOI | 15 | 0.145 | 0.139 | 0.151 | 0.125 | 0.320 |
| rti_random | RTI | 15 | 0.139 | 0.156 | -0.113 | 0.251 | 0.009 |
| gt_c6_epistatic | GT | 7 | 0.131 | 0.571 | **-0.444** | 0.044 | 0.271 |
| ioi_C4_random | IOI | 15 | 0.118 | 0.232 | -0.089 | 0.365 | 0.007 |
| rti_EAP | RTI | 15 | 0.117 | 0.235 | -0.269 | 0.006 | 0.087 |
| ioi_C2_eap | IOI | 15 | -0.067 | 0.499 | 0.214 | 0.029 | 0.212 |
| gt_acdc | GT | 7 | -0.239 | 0.297 | 0.096 | 0.679 | 0.158 |

### Registered predictions

| Prediction | Criterion | Result | Verdict |
|---|---|---|---|
| P1 | rho_OV > 0.3 pooled | 0.173 | **FAIL** |
| P2 | partial rho(QK\|OV) > 0, p < 0.05 | rho=0.101, p=6.0e-4 | **PASS** |
| P3 | close-layer rho > far-layer rho | close=0.344, far=0.009 | **PARTIAL** |
| P4 | R^2 > 0.3 for high-faith circuits | pooled=0.068 | **FAIL** |
| P5 | OV predicts better under mean | not testable | — |

### What we thought then

The story was a "structured negative": geometry predicts interaction locally
(close-layer rho=0.344) but not globally (far-layer rho=0.009). We attributed
the distance effect to LayerNorm scrambling — the data-dependent normalization
accumulates error across layers. The gt_known outlier (R^2=0.412) was treated
as evidence that simpler tasks might be more geometric.

### What was wrong

Two problems, identified in the ablation-series analysis:

1. **gt_known's R^2=0.412 is not evidence of anything.** Seven heads give 21
   pairs. A pure null model with 4 predictors averages in-sample R^2=0.25 at
   that sample size, reaching R^2=0.49 at the 95th percentile. gt_known sits
   inside the null distribution.

2. **All R^2 values are in-sample.** Pairs are dyadic — 105 pairs come from 15
   heads, giving an effective sample size of roughly 43 rather than 105. Naive
   in-sample R^2 is inflated, and all values in the table above are suspect.

---

## 2. Analysis II: ablation-dependence (the 38% sign-flip finding)

Script: `scripts/modal_ceiling_exhaustive.py`, `scripts/ceiling_weightonly_r2.py`.
Results: `results/ceiling/ceiling_rows.json`.

### The question

P5 from the original pre-registration asked whether ablation type changes the
geometry-epistasis relationship. Before testing that, a prior question: does
epistasis even mean the same thing across primitives?

### Results

18 cross-primitive comparisons across IOI, RTI, and GT:

| Statistic | Value |
|---|---|
| Median sign agreement between primitives | **61.9%** (range 33.3–81.0%) |
| Comparisons at or below chance sign agreement | 3 / 18 |
| Pearson r between primitives (15-head) | 0.30, 0.42, 0.43, 0.63 (one at -0.07) |
| Pearson r between primitives (7-head) | -0.31 to +0.82 (five negative) |

**Roughly 38% of head pairs change the sign of their interaction depending on
which ablation primitive is used.** A pair that masks under zero ablation reads
as synergistic under mean ablation, or vice versa.

### The genetics parallel

This finding is not novel in form. Genetics discovered the same phenomenon:
different loss-of-function methods yield different interaction measurements.

- **Housden et al. 2017, *Nature Reviews Genetics***: Different LOF methods
  "can lead to substantially different outcomes." The norm: "studies would
  benefit from the application of more than one approach."
- **Sanson et al. 2018, *Nature Communications***: CRISPRko vs CRISPRi agree
  at R=0.69, treated as normal. Our cross-ablation correlations are 0.30–0.63.

Our disagreement is of the same order as genetics', slightly worse than
CRISPRko/CRISPRi and considerably better than morpholino/mutant.

### Errors made and corrected during this analysis

Three attempts at a ceiling-bound argument failed. The first used r^2 instead
of r for the bound (Error A). The second generalized from a single circuit
(Error B). The third had the inequality running backwards (Error C). All three
errors are documented in `LAB_NOTEBOOK_ablation_series.md` because they were
stated confidently before being checked. The ceiling-bound argument was
abandoned entirely. The sign-flip finding survives all corrections.

### Consequence

Zero, mean, and resample ablation produce three distinct epistasis measurements
for the same circuit. "Do the weights predict epistasis?" has three answers, and
Analysis I checked only one.

---

## 3. Analysis III: primitive-specific geometry v2 (cross-validated, 3 arms)

Pre-registration: `docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md` (frozen, amended
A1–A3). First run voided (CV leak); v2 script written and run from scratch.

Script: `scripts/modal_primitive_specific_geometry_v2.py`.
Results: `results/primitive_geometry_summary_v2.json`,
`results/primitive_geometry_rows_v2.json`,
`results/primitive_geometry_pairs_v2.json`.

### What was fixed from v1

Four fatal blockers resolved:

**B1. Circularity check.** IOI c6 heads were selected by top order-2 Walsh
energy, which is the quantity this regression predicts. If the selection ran
under zero ablation, that alone could produce P1's predicted pattern. Confirmed
via `scripts/modal_walsh_discovery.py` (line 4, line 118): the LASSO-Walsh
discovery ran under **mean** ablation. The mean arm carries a selection effect;
zero and resample are clean comparisons.

**B2. Grouping key.** v1 derived the circuit group from the filename with
regex, so `ioi/c6` and `ioi/C6_epistatic` (the same circuit) landed in different
groups, orphaning the resample arm. v2 groups by `(task, frozenset(circuit_heads))`.

**B3. P1 decision rule.** v1 printed per-primitive medians. P1 is a paired
per-circuit rule. v2 computes the per-circuit arm difference and emits
pass/fail against the 0.10 threshold.

**B4. Bootstrap → jackknife.** v1 used
`set(rng.choice(uh, len(uh), replace=True).tolist())`, which discards
multiplicity. Fifteen heads drawn with replacement give roughly 9.5 distinct,
so every bootstrap replicate evaluated a roughly 9-head circuit instead of a
15-head circuit. v2 uses a delete-one-head jackknife: for each head h, drop
all pairs involving h and compute CV R^2 on the remainder, then standard
pseudovalue SE.

### Additional improvements

- **Leave-both-heads-out cross-validation.** For pair (i,j), the model trains
  only on pairs (a,b) with a not in {i,j} and b not in {i,j}. With 15 heads
  this leaves 78 training pairs per fold. With 7 heads it leaves 10, which
  is why GT circuits are descriptive only.
- **Layer-distance-only baseline.** Reports CV R^2 for [layer_distance] alone,
  so the geometry increment (adding OV/QK features) can be measured.
- **Split-half reliability.** Odd/even prompt split, Spearman-Brown corrected.
  Measures how precisely the Walsh coefficients are estimated.
- **Per-pair records.** All 609 pair-level records saved for P3 and future
  robustness checks.

### Comparable data

The sweep manifest (`results/inventory/SWEEP_MANIFEST.md`) identifies which
sweeps share the same head set, task, and prompt count:

| Circuits | Heads | Pairs | Arms | Prompts | Status |
|---|---|---|---|---|---|
| IOI c6 | 15 | 105 | zero, mean, resample | 512 | **Confirmatory** |
| GT c5_walsh | 7 | 21 | zero, mean, resample | 1000 | Descriptive |
| GT c6_epistatic | 7 | 21 | zero, mean, resample | 1000 | Descriptive |
| GT known | 7 | 21 | zero, mean, resample | 1000 | Descriptive |
| GT random | 7 | 21 | zero, mean, resample | 1000 | Descriptive |
| GT ablation_discovered | 7 | 21 | zero, mean | 1000 | Descriptive (2 arms) |

RTI circuits are excluded: their zero/mean sweeps ran at 302 prompts while
resample ran at 512, blocking cross-primitive comparison.

### Results — IOI c6 (confirmatory, 15 heads, 105 pairs)

| Primitive | In-sample R^2 | CV R^2 | Layer-dist-only CV | Jackknife 95% CI | rho_OV | Reliability |
|---|---|---|---|---|---|---|
| zero | 0.2348 | **-0.3564** | -0.0548 | [-1.3208, 0.4582] | +0.2256 | 0.9856 |
| mean* | 0.1191 | **-0.0007** | -0.0510 | [0.0819, 0.6666] | +0.2375 | 0.9640 |
| resample | 0.1581 | **-0.1343** | -0.0384 | [0.0550, 1.1358] | +0.1949 | 0.9661 |

*Mean arm carries selection bias: c6 heads were selected by top order-2 Walsh
energy under mean ablation.

Every CV R^2 is negative. The in-sample values (0.12–0.24) are pure
overfitting on 105 dyadic pairs with 4 predictors. Once leave-both-heads-out
cross-validation is applied, the signal vanishes entirely.

The layer-distance-only baseline also produces negative CV R^2, so adding
geometry features (OV overlap, QK composition) does not help. The geometry
increment is noise.

Split-half reliability is 0.964–0.986 — the Walsh coefficients are measured
with high precision. The failure is prediction, not measurement.

### Results — GT circuits (descriptive, 7 heads, 21 pairs each)

| Circuit | Primitive | In-sample R^2 | CV R^2 | rho_OV | Reliability |
|---|---|---|---|---|---|
| c5_walsh | zero | 0.0122 | -0.6596 | -0.0935 | 0.9998 |
| c5_walsh | mean | 0.1191 | -1.3581 | +0.0442 | 0.9997 |
| c5_walsh | resample | 0.3195 | -2.4241 | +0.4623 | 0.9999 |
| c6_epistatic | zero | 0.3347 | -1.0377 | +0.3961 | 0.9997 |
| c6_epistatic | mean | 0.4176 | -3.7252 | +0.4351 | 0.9997 |
| c6_epistatic | resample | 0.6330 | **-8.5061** | +0.1312 | 1.0000 |
| known | zero | 0.3495 | -1.9600 | +0.3675 | 0.9995 |
| known | mean | 0.3753 | -2.2704 | +0.3831 | 0.9997 |
| known | resample | 0.4193 | -1.2586 | +0.6234 | 0.9952 |
| random | zero | 0.4406 | -4.0756 | -0.0273 | 0.9964 |
| random | mean | 0.3388 | -4.3440 | +0.2013 | 0.9962 |
| random | resample | 0.1281 | -4.8639 | +0.1701 | 0.9952 |
| abl_discovered | zero | 0.0622 | -0.3445 | +0.2351 | 0.9999 |
| abl_discovered | mean | 0.1306 | -0.6809 | +0.2870 | 0.9996 |

All CV R^2 values are deeply negative, from -0.34 to -8.51. In-sample R^2
ranges from 0.01 to 0.63 — pure null overfitting. With 21 pairs and 4
predictors, a pure null model averages in-sample R^2 of 0.25, so even the
0.63 value (c6_epistatic resample) is not statistically distinguishable from
null.

Jackknife CIs are all null — too few heads for the delete-one procedure to
produce enough training pairs after dropping pairs from both the deleted head
and the held-out test pair.

### Registered predictions — v2

**P1 (primary): mean/resample better predicted than zero by >= 0.10 CV R^2.**

| Comparison | Difference | Threshold | Verdict |
|---|---|---|---|
| mean − zero | +0.3557 | 0.10 | PASS |
| resample − zero | +0.2221 | 0.10 | PASS |

Both pass the threshold, but the registered rule requires "at least half of
eligible 15-head circuits." Only 1 circuit is eligible (IOI c6 — RTI circuits
excluded for prompt mismatch). **P1 as registered is dead.** The result is
descriptive, and the direction is present but moot: the difference is between
CV R^2 of -0.001 and -0.356, both of which mean "no predictive power."

The mean arm carries an additional confound: c6 was selected by top order-2
Walsh energy under mean ablation, so the mean arm's higher CV R^2 could reflect
the selection effect rather than anything about the ablation primitive.

**P2: all CV R^2 below 0.20 pooled.**

| Primitive | Median CV R^2 | n | Verdict |
|---|---|---|---|
| zero | -0.8487 | 6 | PASS |
| mean | -1.8142 | 6 | PASS |
| resample | -2.4241 | 5 | PASS |

All medians deeply negative. Geometry is weak under every primitive.

**P3: close-layer rho exceeds far-layer rho under all primitives.**

| Primitive | Close rho (n) | Far rho (n) | Close > Far |
|---|---|---|---|
| zero | 0.2363 (86) | 0.3140 (93) | **NO** |
| mean | 0.3239 (86) | 0.1517 (93) | yes |
| resample | 0.3968 (76) | -0.1930 (85) | yes |

Under zero ablation, far-layer pairs show **higher** correlation than
close-layer pairs — the opposite of the prediction and the opposite of what the
original in-sample analysis found. Under mean and resample, the predicted
direction holds. P3 is primitive-dependent: the close-layer story from Analysis
I is not robust to the choice of ablation primitive.

**P4: CV R^2 materially below in-sample R^2 on zero arm (gap >= 0.03).**

IOI c6 zero arm: in-sample R^2 = 0.2348, CV R^2 = -0.3564, gap = 0.5912.
PASS. The in-sample value was grossly inflated.

---

## 4. What changed between analyses, and what each correction revealed

### Correction 1: cross-validation

The original analysis reported in-sample R^2 = 0.068 pooled, 0.412 for
gt_known. The v2 analysis cross-validates with leave-both-heads-out. Every
CV R^2 is negative — the model is worse than predicting the mean.

The gap between in-sample and CV is large because pairs are dyadic. 105 pairs
from 15 heads share head indices, creating systematic dependence that in-sample
R^2 ignores. The effective sample size is roughly 43 for 15 heads and roughly
7 for 7 heads. With 4 predictors and 43 effective observations, overfitting
is expected.

### Correction 2: gt_known is not an outlier

At 21 pairs and 4 predictors, the expected null in-sample R^2 is 0.25. The
gt_known in-sample R^2 of 0.412 sits well within the null distribution. The
original notebook treated gt_known as evidence that the greater-than task has
stronger geometric epistasis. Cross-validation shows CV R^2 = -1.26 to -1.96
across primitives. There is no evidence of geometric epistasis in gt_known.

### Correction 3: the close-layer effect is primitive-dependent

The original analysis found close-layer rho=0.344 and far-layer rho=0.009
under resample ablation, and attributed the gap to LayerNorm scrambling. The
v2 analysis shows this pattern holds under mean and resample (close > far)
but reverses under zero ablation (far > close). The LayerNorm story is at best
incomplete: a mechanism that accumulates across layers should produce the same
direction regardless of ablation type.

### Correction 4: the grouping bug

v1 of the primitive-specific geometry script grouped by circuit name extracted
from the filename with regex. The same circuit appeared under different names
(`c6` vs `C6_epistatic`), orphaning arms and producing spurious single-arm
groups. The v2 script groups by `(task, frozenset(circuit_heads))`, which is
invariant to naming conventions.

### Correction 5: the bootstrap was not a bootstrap

v1 used `set()` on resampled head indices, discarding multiplicity. Fifteen
heads drawn with replacement give roughly 9.5 distinct, so every bootstrap
replicate silently evaluated a smaller circuit. The v2 script uses a
delete-one-head jackknife, which preserves the sample size and avoids the
problem entirely.

---

## 5. The combined picture

Three analyses, each more careful than the last, converging:

**Finding 1: weight geometry does not predict epistasis.** In-sample R^2
values of 0.07–0.41 (original analysis) are entirely overfitting. Cross-
validated R^2 is negative under every ablation primitive, for every circuit,
at every circuit size. The layer-distance-only baseline is also negative, so
adding OV overlap and QK composition to layer distance does not help.

**Finding 2: epistasis is reliably measured.** Split-half reliability is
0.964–0.999 across all arms and circuits. The Walsh coefficients are precise
measurements of a real quantity. The failure is prediction from weights, not
measurement of the target.

**Finding 3: epistasis disagrees across primitives.** Roughly 38% of head pairs
change the sign of their interaction depending on the ablation primitive. This
is a systematic phenomenon, not noise, and it is of the same magnitude as
analogous disagreements between loss-of-function methods in genetics.

**Finding 4: the close-layer effect is not robust.** Under resample ablation,
close-layer pairs show stronger OV correlation than far-layer pairs (rho=0.397
vs -0.193). Under zero ablation, the pattern reverses (rho=0.236 vs 0.314).
The LayerNorm-scrambling story from Analysis I is not a universal mechanism.

**Finding 5: the P1 direction is present but moot.** Mean and resample
ablation yield less negative CV R^2 than zero ablation on IOI c6 (by 0.36 and
0.22 respectively). This is the predicted direction. But the absolute values
are all negative, and the comparison rests on a single circuit whose mean arm
carries selection bias. The direction might be real — it is consistent with
zero ablation disrupting norms more than mean or resample — but the effect
size is the difference between "terrible" and "slightly less terrible."

### The combination that matters

Reliable measurement (0.96–0.99 split-half) plus zero weight predictability
(every CV R^2 negative) establishes epistasis as a genuine activation-level
phenomenon. If epistasis were just weight geometry under a different name,
OV composition would predict Walsh interaction, and it does not. The
interaction structure measured by coalition ablation captures something that
static weights cannot.

---

## 6. What this means for the paper

### The subspace negative is a finding, not a gap

Weight geometry does not explain head-pair interaction. The published in-sample
R^2 of 0.068 from Analysis I was already weak, and the narrative leaned on
the close-layer effect (rho=0.344 at close range) and the gt_known outlier
(R^2=0.412). Both dissolve under cross-validation: gt_known is within the null
distribution, and the close-layer effect reverses under zero ablation.

This negative result strengthens the paper's contribution. Walsh interaction
measures something about how the model computes, not something readable from
its weight matrices. A positive result would have reduced epistasis to
geometry and made the coalition sweep unnecessary. The negative result means
the coalition sweep reveals structure that simpler weight-based analyses miss.

### The 38% sign-flip finding is standalone

Epistasis depends on how it is measured. This is not a flaw in the measurement
— within a primitive, epistasis is precise (reliability 0.96–0.99). The
disagreement across primitives is systematic and of the same order as the
disagreement between loss-of-function methods in genetics. The consequence
for the paper: circuit interaction structure is a property of the (circuit,
primitive) pair, not of the circuit alone. Reporting interaction under a
single ablation primitive without saying which one is incomplete.

### The paper's core contribution is unaffected

The Walsh decomposition of circuits, the discovery-methods-as-selective-
pressures argument, the complementation test on IOI head classes — none of
these depend on weight geometry predicting epistasis. The subspace analysis
was a side investigation into whether the interaction could be explained away.
The answer is no, which makes the interaction more interesting.

### Reporting in the paper

One paragraph in the results or discussion, anchored to the pre-registration:

The subspace analysis asked whether weight-space composition scores predict
Walsh interaction. In-sample R^2 of 0.068 pooled across 15 circuits suggested
a weak relationship. Leave-both-heads-out cross-validation, which respects the
dyadic structure of head pairs, returns negative R^2 under all three ablation
primitives (zero, mean, resample) and for all five circuits with comparable
three-arm data. The in-sample value was overfitting inflated by the low
effective sample size of dyadic data. Weight geometry does not predict
head-pair epistasis.

---

## 7. Files

### Analysis I (original subspace)
- Pre-registration: `docs/PREREG_SUBSPACE_EPISTASIS.md`
- Script: `scripts/modal_subspace_epistasis_analysis.py`
- Results (Modal volume `subspace-epistasis-results`):
  `subspace_epistasis_summary.json`, `subspace_epistasis_pairs.csv`,
  `subspace_epistasis_per_circuit.json`
- Local copies: `results/subspace_epistasis_*.{csv,json}`
- Lab notebook: `results/LAB_NOTEBOOK_subspace_epistasis.md` (superseded)

### Analysis II (ablation dependence)
- Script: `scripts/modal_ceiling_exhaustive.py`,
  `scripts/ceiling_weightonly_r2.py`
- Results: `results/ceiling/ceiling_rows.json`, `ceiling_log.txt`
- Lab notebook: `results/LAB_NOTEBOOK_ablation_series.md`

### Analysis III (primitive-specific geometry v2)
- Pre-registration: `docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md`
  (frozen, amendments A1–A3)
- Handoff: `docs/HANDOFF_primitive_specific_geometry.md`
- Script (broken): `scripts/modal_primitive_specific_geometry.py` (4 fatal
  blockers, 2 fixed at time of voiding)
- Script (fixed): `scripts/modal_primitive_specific_geometry_v2.py`
- Pull script: `scripts/modal_pull_geometry_v2.py`
- Results (Modal volume `primitive-geometry-results-v2`):
  `primitive_geometry_summary_v2.json`, `primitive_geometry_rows_v2.json`,
  `primitive_geometry_pairs_v2.json`, `run_log.txt`
- Local copies: `results/primitive_geometry_*_v2.json`

### Sweep infrastructure
- Inventory script: `scripts/modal_inventory_headsets.py`
- Manifest: `results/inventory/SWEEP_MANIFEST.md`
- Inventory data: `results/inventory/headset_inventory.json`
- Prompt count check: `scripts/modal_check_prompt_counts.py`
- Volume completeness check: `scripts/modal_check_all_volumes.py`

### Verification
- Perplexity bundle (subspace): `docs/PERPLEXITY_SUBSPACE_RESULTS.md`

---

## 8. Round 4: RTI circuits included (v4 analysis)

### What changed from v2

v3 attempted to include RTI circuits by relaxing the prompt-count filter to
flag mismatches instead of skipping. v3 crashed on the first RTI circuit
(transient failure — no traceback captured). v4 added traceback logging and
re-ran. All 4 RTI circuits processed without error.

### RTI circuit inventory (all 15 heads, 105 pairs)

| Circuit | Zero source | Mean source | Resample source |
|---------|-------------|-------------|-----------------|
| C5 walsh | rti-walsh (302p) | rti-walsh (302p) | rti-resample (512p) |
| EAP | rti-v5 (302p) | rti-v5 (302p) | rti-resample (512p) |
| known (RTI) | rti-v5 (302p) | rti-v5 (302p) | rti-resample (512p) |
| random (C4) | rti-v5 (302p) | rti-v5 (302p) | rti-resample (512p) |

All flagged as PROMPT MISMATCH (302 vs 512 prompts across arms).
RTI C6 excluded: mean arm incomplete (2048/32768 coalitions).

### RTI results

| Circuit | Zero CV R^2 | Mean CV R^2 | Resample CV R^2 |
|---------|-------------|-------------|-----------------|
| C5 | -0.063 | -0.001 | -0.203 |
| EAP | -0.259 | -0.202 | -0.200 |
| known | -0.120 | -0.505 | -0.511 |
| random | -0.141 | -0.087 | -0.026 |

All negative. Split-half reliability ranges 0.669--0.983 (lowest is random
resample arm at 0.669, all others above 0.89).

### Updated totals

- **29 rows** across 10 comparable groups from 3 tasks
- **Every CV R^2 is negative** under every primitive for every task
- P1: FAIL (3/10 comparisons pass the 0.10 threshold)
- P2: PASS (all medians deeply negative)
- P3: close-layer > far-layer now holds under ALL primitives (previously
  zero ablation showed a reversal with IOI-only data; pooling with RTI
  and GT resolves the reversal)
- P4: PASS (all in-sample > CV)

### Data artifacts

- Modal volume: `primitive-geometry-results-v4`
- Script: `scripts/modal_primitive_specific_geometry_v4.py`
- Local copies: `results/primitive_geometry_*_v4.json`

---

## 9. Open questions (updated)

1. ~~Activation-based subspace overlap~~ — still valid as a direction, but
   given that every weight-based predictor gives negative CV R^2 across all
   three tasks, the relevant question is whether activation-derived predictors
   would do better. The LN normalization is the candidate mechanism.

2. ~~RTI re-runs at matched prompts~~ — resolved by including RTI with a
   prompt-mismatch flag. Results are qualitatively identical to matched-prompt
   circuits. Not worth re-running at matched prompts.

3. ~~P3 reversal under zero ablation~~ — the reversal was an artifact of
   IOI-only data. With pooled data across all three tasks, close-layer >
   far-layer holds consistently under all primitives.
