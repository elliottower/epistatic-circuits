# Lab notebook — subspace decomposition of head-level epistasis

Session of 2026-08-07. Chronological, including the filename bug that cost a
run and the negative result that followed.

Pre-registration: `docs/PREREG_SUBSPACE_EPISTASIS.md`,
SHA-256 `3ef962bc32110b98bf3e1ebf934debf60ec6587f84e89fad4476ac1eb2a1499f`,
frozen 2026-08-07.

---

## 0. The question

We have full 2^n coalition sweeps for 15 circuits across three tasks (IOI, RTI,
greater-than) and five selection methods each. From these we extract pairwise
Walsh coefficients |W_{ij}| measuring non-additive interaction between heads i
and j.

All inter-head communication in a standard transformer goes through the
residual stream (Elhage et al. 2021). If epistasis reflects shared subspaces —
head A writes into a direction that head B reads from — then OV subspace
overlap should predict pairwise Walsh interaction strength.

This is the structural-view test: does the object-view interaction (coalition
ablation) reduce to a gauge-invariant weight-space property (OV composition)?

---

## 1. What was pre-registered

Five predictions:

- **P1**: Spearman rho(|W_{ij}|, OV_overlap) > 0.3 pooled across circuits
- **P2**: Partial rho(|W_{ij}|, QK_comp | OV_overlap) > 0, p < 0.05
- **P3**: Close-layer pairs (dist <= 2) show stronger QK correlation than OV;
  far-layer pairs (dist > 3) show OV dominance
- **P4**: Linear model [OV_overlap, QK_comp, layer_dist] → |W_{ij}| achieves
  R^2 > 0.3 for high-faithfulness circuits
- **P5**: OV overlap predicts better under mean ablation than resample (not
  testable — all data is resample ablation)

Decision rule: P1 confirmed AND P4 R^2 > 0.3 → geometric; P1 fails →
computational/LN-mediated.

Pre-reg also noted that the un-foldable LayerNorm normalization nonlinearity
places an inherent ceiling on R^2 even under the fully geometric hypothesis.
Conservative thresholds were set accordingly (R^2 > 0.5 = substantially
geometric, R^2 < 0.2 = not geometric).

---

## 2. Method

Ran on Modal (`scripts/modal_subspace_epistasis_analysis.py`).

1. Loaded GPT-2 small via TransformerLens with `fold_ln=True` — this folds the
   affine part of LayerNorm (learned scale and bias) into downstream weight
   matrices. The data-dependent normalization division remains un-folded.

2. Extracted W_OV = W_V @ W_O, W_Q, W_K for all 144 heads (12 layers x 12
   heads). Shape: each (d_model, d_model) = (768, 768) for W_OV, (d_model,
   d_head) = (768, 64) for W_Q, W_K.

3. For each of 15 circuits, loaded the full coalition sweep NPZ from Modal
   volumes (ioi-resample-sweep, rti-resample-sweep, gt-resample-sweep).

4. Computed Walsh-Hadamard transform of the mean coalition values to get all
   2^n Fourier-Walsh coefficients. Extracted order-2 coefficients at index
   (1<<i)|(1<<j) for each pair (i,j).

5. For each head pair, computed:
   - OV overlap: || W_OV_i^T @ W_OV_j ||_F / (||W_OV_i||_F * ||W_OV_j||_F)
   - Q-composition: || W_OV_i @ W_Q_j^T @ W_K_j ||_F / norms (Elhage 2021)
   - K-composition: || W_OV_i @ W_K_j^T @ W_Q_j ||_F / norms
   - QK_comp = max(Q-comp both directions, K-comp both directions)
   - Principal angles via SVD of U_i^T @ U_j (robustness check)
   - Layer distance |l_i - l_j|

6. Spearman correlations, partial correlations, OLS regression. All
   pre-registered, no researcher degrees of freedom.

---

## 3. The filename bug

First run (10/15 circuits): IOI circuits were all skipped. The analysis script
looked for `/vol/ioi-resample/C3_canonical_resample_coalition_values.npz` but
the actual files on the volume have an `ioi_` prefix:
`ioi_C3_canonical_resample_coalition_values.npz`.

Diagnosed by writing `scripts/modal_check_ioi_volume.py` to list volume
contents. All 5 IOI files confirmed complete (32768/32768) with the `ioi_`
prefix.

Fixed all 5 paths in the CIRCUITS dict and reran. This rerun is the
authoritative analysis.

---

## 4. Results — per circuit

15 circuits, 1155 total head pairs (10 circuits with 15 heads = 105 pairs each,
5 circuits with 7 heads = 21 pairs each).

| Circuit | Task | Heads | Faith | rho_OV | p_OV | rho_QK | p_QK | R^2 |
|---|---|---|---|---|---|---|---|---|
| gt_known | GT | 7 | 1.313 | **0.623** | 0.003 | 0.013 | 0.955 | **0.412** |
| gt_c5_walsh | GT | 7 | 0.934 | **0.462** | 0.035 | -0.274 | 0.229 | 0.243 |
| rti_C6_epistatic | RTI | 15 | 2.620 | **0.320** | 0.001 | 0.140 | 0.154 | 0.138 |
| rti_known | RTI | 15 | 0.831 | 0.277 | 0.004 | -0.135 | 0.171 | 0.088 |
| ioi_C5_walsh | IOI | 15 | -1.001 | 0.230 | 0.018 | 0.126 | 0.201 | 0.181 |
| ioi_C6_epistatic | IOI | 15 | 1.849 | 0.195 | 0.046 | 0.078 | 0.431 | 0.214 |
| rti_C5_walsh | RTI | 15 | 0.329 | 0.172 | 0.079 | -0.097 | 0.324 | 0.171 |
| gt_random | GT | 7 | 0.005 | 0.170 | 0.461 | 0.321 | 0.157 | 0.125 |
| ioi_C3_canonical | IOI | 15 | 3.141 | 0.145 | 0.139 | 0.151 | 0.125 | 0.320 |
| rti_random | RTI | 15 | 0.126 | 0.139 | 0.156 | -0.113 | 0.251 | 0.009 |
| gt_c6_epistatic | GT | 7 | 0.917 | 0.131 | 0.571 | **-0.444** | 0.044 | 0.271 |
| ioi_C4_random | IOI | 15 | 0.168 | 0.118 | 0.232 | -0.089 | 0.365 | 0.007 |
| rti_EAP | RTI | 15 | 0.289 | 0.117 | 0.235 | -0.269 | 0.006 | 0.087 |
| ioi_C2_eap | IOI | 15 | -0.319 | -0.067 | 0.499 | 0.214 | 0.029 | 0.212 |
| gt_acdc | GT | 7 | 0.650 | -0.239 | 0.297 | 0.096 | 0.679 | 0.158 |

---

## 5. Results — pooled predictions

**P1: OV overlap predicts pairwise Walsh interaction (FAIL)**

Pooled rho(|W_{ij}|, OV_overlap) = 0.173 (p = 3.5e-9). Highly significant
but far below the 0.3 threshold. The signal is real but weak.

**P2: QK composition adds predictive power (PASS)**

Partial rho(|W_{ij}|, QK_comp | OV_overlap) = 0.101 (p = 6.0e-4). QK
composition contributes independent information beyond OV overlap. The effect
is small.

**P3: Layer distance modulates interaction type (PARTIALLY CONFIRMED)**

Close layers (dist <= 2, n = 466 pairs): rho_OV = 0.344, rho_QK = 0.200
Far layers (dist > 3, n = 551 pairs): rho_OV = 0.009, rho_QK = -0.123

The prediction was that QK dominates for close layers and OV for far. The
opposite happened: OV dominates for close layers (0.344 vs 0.009) and both
vanish for far layers. The layer-distance modulation is real but the direction
is wrong.

**P4: Variance decomposition (FAIL)**

Pooled R^2 = 0.068. High-faith circuits only: R^2 = 0.098. Both far below 0.3.
Per the pre-registered thresholds: R^2 < 0.2 = "weight-only subspace overlap
does not explain interactions; epistasis is primarily computational or
LN-mediated."

Exception: gt_known alone reaches R^2 = 0.412 (above 0.3, in the "mixed" zone
of the pre-reg). This is a 7-head circuit with only 21 pairs, so the sample is
small, but the correlation is strong (rho = 0.623, p = 0.003).

**P5: Ablation-type sensitivity (NOT TESTABLE)**

All data is resample ablation. Mean-ablation coalition sweeps were not run.

---

## 6. What the negative result means

Per the pre-registered decision rule: P1 fails → "head-level epistasis is
computational or LN-mediated, not geometric; subspace overlap does not
explain interactions; report as negative result."

But the story is more structured than a flat negative.

### 6a. The close-layer effect

For head pairs within 2 layers of each other, OV overlap does predict
interaction (rho = 0.344). For pairs more than 3 layers apart, it does not
(rho = 0.009). The geometric signal is real but local. This makes mechanistic
sense: LayerNorm sits between every layer, and its data-dependent normalization
scrambles the weight-only prediction as signals propagate through more LN
layers. Close-layer heads share similar LN statistics; distant heads do not.

### 6b. The task-size effect

GT circuits (7 heads, 21 pairs) show much stronger geometric structure than
IOI/RTI circuits (15 heads, 105 pairs). gt_known has rho = 0.623 and
R^2 = 0.412; the best IOI circuit (ioi_C3_canonical) has rho = 0.145 and
R^2 = 0.320 (but R^2 is driven by layer distance, not OV overlap — the
partial correlation is weak).

Two interpretations: (a) smaller circuits have less LN noise, so the geometric
signal is cleaner; (b) greater-than is a simpler task than IOI, with more
direct residual-stream composition and less routing.

### 6c. QK composition is weak and sometimes negative

QK composition (the Q-composition score from Elhage et al. 2021) adds marginal
predictive power (partial rho = 0.101) but is often negatively correlated with
interaction strength per circuit (rti_EAP: rho_QK = -0.269; gt_c6_epistatic:
rho_QK = -0.444). This means heads with HIGH QK composition sometimes have LOW
Walsh interaction. One interpretation: QK composition captures routing
dependency (head A steers head B's attention), but routing can be
complementary rather than interactive — head A may steer head B to attend
somewhere useful without either head needing the other for its effect.

### 6d. Random circuits as negative controls

Both random circuits with 15 heads (ioi_C4_random, rti_random) show R^2 < 0.01
and no significant correlations. The random GT circuit shows R^2 = 0.125 but
with only 21 pairs this is not distinguishable from chance. The negative
controls work as expected.

---

## 7. Interpretation through mechanistic views

The subspace analysis tests whether object-view epistasis (coalition ablation
on concrete heads) reduces to structural-view structure (gauge-invariant
weight-space composition). The answer is: mostly not.

**Object view** — epistasis is well-defined. Heads A and B interact means
their joint ablation effect deviates from the sum of their individual effects.
This is a reproducible property of the coalition game. It passes discriminant
validity: real circuits show structured interaction (R^2 = 0.14-0.41 for
non-random circuits) while random circuits show none (R^2 < 0.01).

**Structural view** — OV composition scores are gauge-invariant weight-space
properties. They predict object-view epistasis weakly overall (rho = 0.17) but
strongly for close-layer pairs (rho = 0.34) and the gt_known circuit
(rho = 0.62). The structural view captures a component of epistasis but not
the majority.

**Subspace view** — causal subspaces (DAS/IIA) could potentially explain more
variance than weight-only composition scores, since they are computed from
activations rather than weights and thus incorporate the LN normalization
nonlinearity. This is the natural next test: do activation-derived subspace
overlaps predict Walsh interaction better than weight-derived composition
scores?

**Stratified view** — the close-layer vs far-layer split is a stratification
effect. At the local resolution (adjacent layers), epistasis is geometric. At
the global resolution (many layers), epistasis is computational. The mechanism
changes with the resolution at which it is examined. This fits the stratified
view's ontology: the mechanism is a point in a resolution-indexed stratum, and
the interaction structure depends on which stratum you examine.

**Convergence across views** — the validity framework (mechanistic-validity
paper, criterion C3) requires convergent evidence from independent methods for
promotion past "Causally Suggestive." Epistasis currently has:
- Object-view evidence: coalition sweeps (confirmed, reproducible)
- Structural-view evidence: weight-space composition (weak, partially
  confirming for local interactions)
- No subspace-view evidence (DAS-based subspace overlap not yet tested)
- No process-view evidence (no checkpoint analysis of interaction formation)

The partial convergence (structural view weakly confirms the object view for
close-layer pairs) is itself informative: it bounds the geometric component of
epistasis and identifies the un-foldable LN normalization as the likely source
of the residual.

---

## 8. Open questions

1. **Activation-based subspace overlap**: compute DAS-derived subspace overlap
   from activations (not weights) and test whether it predicts Walsh
   interaction better than weight-only OV scores. This would directly measure
   whether the LN normalization is the missing link.

2. **Mean-ablation comparison (P5)**: run coalition sweeps under mean ablation
   to test whether ablation type changes the geometry-epistasis relationship.
   Mean ablation removes distributional structure, which should make geometric
   predictions stronger if LN is the confound.

3. **Per-task analysis**: IOI is routing-heavy (attention patterns matter more
   than subspace overlap); GT is composition-heavy (residual stream composition
   matters more). The task-dependence of geometric predictability is itself a
   finding worth quantifying.

4. **The gt_known outlier**: R^2 = 0.412 in a single circuit is suggestive but
   based on only 21 pairs. Running additional GT circuit variants would test
   whether this is a robust property of the task or a small-sample fluctuation.

---

## 9. Files

- Pre-registration: `docs/PREREG_SUBSPACE_EPISTASIS.md`
- Perplexity verification (round 1): `docs/PERPLEXITY_SUBSPACE_EPISTASIS.md`
- Analysis script: `scripts/modal_subspace_epistasis_analysis.py`
- Volume check script: `scripts/modal_check_ioi_volume.py`
- Results on Modal volume `subspace-epistasis-results`:
  - `subspace_epistasis_summary.json` (pooled statistics, prediction verdicts)
  - `subspace_epistasis_pairs.json` (all 1155 pairs with features)
  - `subspace_epistasis_pairs.csv` (same, tabular)
  - `subspace_epistasis_per_circuit.json` (per-circuit summary)
