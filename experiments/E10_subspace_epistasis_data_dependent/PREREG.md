# Does data-dependent subspace overlap predict head-pair epistasis?

**Status:** FROZEN at `a8ac187a4709`
**Plan sha256:** `4ced83218624a42bb5a864731e1d53c1d428d336bb1c7c890cb0b9140f6d6ab5`
**Frozen:** 2026-08-11

Sections use the [OSF Preregistration](https://osf.io/prereg/) question titles verbatim, so
this maps onto a registration without being rewritten. A question that does not apply is
answered **N/A** with the reason, never deleted.

## Research questions or hypotheses

Section 3.3 established that weight-only features (OV composition, QK composition, layer
distance) carry no cross-validated predictive power for pairwise Walsh interaction
coefficients. All 30 CV R^2 values were negative across ten circuit--primitive combinations,
despite in-sample R^2 up to 0.63 and split-half reliability exceeding 0.89 in 29 of 30
arms. The paper concludes: "Head-pair epistasis is a property of the forward pass, shaped by
data-dependent LayerNorm normalization and attention routing, that static weight matrices do
not capture."

That conclusion rests on an untested assumption: that the evaluation protocol is capable of
returning positive CV R^2 at this sample size. Section 3.3 shows that weight geometry fails
but does not show that anything succeeds. An all-negative result from an evaluation that
cannot detect real signal is uninformative.

This experiment addresses both gaps. It tests whether data-dependent activation subspace
features predict pairwise Walsh coefficients where static weight geometry failed, and it
includes a positive control — predictors already known to carry signal about interaction
(EAP edge products, order-1 magnitude products) — run through the identical evaluation
pipeline. If the control arm returns positive CV R^2 and the geometry arms do not, the null
is interpretable. If the control arm also fails, the evaluation is underpowered and no
geometric conclusion follows.

**H1.** Data-dependent subspace features achieve positive cross-validated R^2 where
weight-only features failed.

**H2.** The positive control (EAP edge products and/or order-1 magnitude products) achieves
positive CV R^2, confirming the evaluation protocol can detect signal at this sample size.

**H3 (null, informative).** H2 confirmed but H1 not: data-dependent features fail despite
the protocol being adequate. The forward-pass claim in Section 3.3 needs revision — the
failure of weight geometry is not about missing data dependence.

**H4 (null, uninformative).** H2 fails: the positive control also returns negative CV R^2.
The leave-both-heads-out protocol cannot detect signal at this sample size, and no geometric
conclusion follows from the negative result. Report as a power limitation.

## Foreknowledge of data or evidence

Substantial, and it constrains the design.

**From Section 3.3 (observed):**
- All 30 weight-only CV R^2 are negative (IOI, RTI, GT; zero/mean/resample ablation).
- In-sample R^2 reaches 0.63 for 7-head GT circuits — overfitting, not signal.
- Split-half reliability of the Walsh coefficients exceeds 0.89 in 29/30 arms.
- 32% of head pairs change the sign of their interaction between zero and mean ablation.
- Cross-primitive gamma: 0.14 (zero vs mean), 0.43 (mean vs resample).

**From Section 3.1 (observed):**
- EAP edge products correlate with Walsh coefficients at Spearman rho = 0.58, Pearson
  r = 0.73 for the 20-head IOI circuit. This is a rank correlation across all pairs, not a
  CV evaluation — it is the basis for choosing EAP edges as a positive control, not a
  prior result on the evaluation protocol.
- LASSO-recovered coefficients for the 20-head circuit have split-half r = 0.997.

**From Section 4 (observed):**
- The conditional marginal analysis shows S-inhibition heads gate the sign of negative name
  mover contributions. This is a data-dependent effect (attention routing changes under
  ablation).

**Genuinely unobserved:**
- Every data-dependent subspace feature value.
- Every regression of data-dependent features against Walsh coefficients.
- Every regression of EAP edge products or order-1 magnitude products against Walsh
  coefficients through the leave-both-heads-out CV pipeline.
- Whether the leave-both-heads-out protocol can return positive CV R^2 for any predictor.

## Explanation of foreknowledge and managing unintended influences

The weight-only null (Section 3.3) constrains the static arm: Arm 1 is a replication within
the new pipeline, not a test. The EAP correlation (Section 3.1) motivates the positive
control but was measured by a different method (rank correlation, not dyadic CV). Whether
that signal survives leave-both-heads-out validation is genuinely unknown.

## Study type

Observational — no intervention beyond the standard ablation primitives already used to
compute Walsh coefficients.

## Intention for causal interpretation

No. The question is predictive (do subspace features predict interaction?), not causal (do
subspace features cause interaction?).

## Blinding of experimental treatments

N/A — no human judgement enters the measurement.

## Additional blinding during research or analysis

The features and the target (Walsh coefficients) are computed by separate code paths. The
Walsh coefficients are already computed and frozen in existing results files.

## Study design

Four predictor sets, same target, same cross-validation.

**Circuit.** The 20-head IOI circuit from Section 3.1 (top 20 attention heads by activation
patching, 190 head pairs). LASSO-recovered Walsh coefficients with split-half r = 0.997
serve as the target. This circuit provides 190 pairs; with leave-both-heads-out CV, each
fold trains on approximately 153 pairs (removing the ~37 pairs involving either test head),
giving substantially more effective observations than the 15-head circuits in Section 3.3.

**Arm 1: weight-only features (replication of Section 3.3).**
OV subspace overlap (Frobenius cosine similarity of W_OV matrices), Q-composition score,
K-composition score, layer distance. Computed from LayerNorm-folded effective weights of
GPT-2 small. This arm replicates Section 3.3's analysis on the 20-head circuit for
apples-to-apples comparison with the other arms. The result is expected to be negative.

**Arm 2: data-dependent subspace features.**

For each head h, run GPT-2 small on the same prompts used to compute the Walsh
coefficients for the 20-head circuit and collect:

1. **Output activations** z_h: the head's hook_z output, shape (N, seq, d_head). Average
   over sequence positions to get (N, d_head).

2. **Output subspace** U_h: top-k left singular vectors of the (N, d_head) activation
   matrix, where k is chosen by the elbow of the singular value spectrum (or fixed at
   d_head/2 if no elbow).

3. **Attention pattern** A_h: the (N, seq, seq) attention weights, averaged over prompts to
   get a (seq, seq) mean pattern.

For each head pair (i, j):

- **Grassmannian distance**: the geodesic distance between U_i and U_j on the Grassmannian
  Gr(k, d_head), computed as sqrt(sum(theta_k^2)) where theta_k are the principal angles
  between the two subspaces.

- **Subspace overlap (Frobenius)**: ||U_i^T U_j||_F / sqrt(k_i * k_j), the normalized
  Frobenius inner product of the subspace bases.

- **Attention pattern similarity**: Spearman correlation between the flattened mean attention
  patterns of heads i and j.

- **Data-dependent composition**: project head i's mean output through head j's W_QK to
  get the data-dependent version of QK composition. Specifically:
  comp_data(i,j) = ||mean(z_i) @ W_Q_j||^2 / (||mean(z_i)||^2 * ||W_Q_j||^2_F),
  where W_Q_j uses LayerNorm-folded weights. Only defined for layer(i) < layer(j).

**Circularity guard:** all subspaces are computed from clean forward passes only. Features
measured under ablated conditions share the intervention with the target, making any
correlation partly definitional.

**Arm 3: positive control.**

Two predictors known to carry signal about Walsh interaction structure:

- **EAP edge product**: the EAP-IG edge score for each head pair, computed by
  backpropagating through the clean forward pass and multiplying activation differences
  by gradients (Syed et al., 2023). Section 3.1 reports Spearman rho = 0.58 between
  magnitude-ranked Walsh coefficients and EAP edge products on this circuit.

- **Order-1 magnitude product**: |w_i| * |w_j|, the product of each head's order-1 Walsh
  coefficient magnitudes. Heads that are individually important (large marginal effect)
  may interact more than heads that are individually weak.

This arm has no geometric hypothesis. It exists to confirm that the evaluation protocol can
return positive CV R^2 when the predictor carries signal.

**Arm 4: combined (exploratory).**
All weight-only, data-dependent, and positive-control features together. With ~10 features
and ~153 training pairs per fold, this arm is likely to overfit and is reported for
completeness only.

## Randomization

N/A — deterministic computation given the model and prompts.

## Data collection procedures

Model: GPT-2 small, loaded via TransformerLens with fold_ln=True, center_writing_weights=True,
center_unembed=True.

Prompts: the same prompt set used to compute the Walsh coefficients for the 20-head IOI
circuit (stored in the existing LASSO recovery results files).

Circuit: 20-head IOI circuit (top 20 heads by activation patching, as in Section 3.1).

## Data collection procedures - File upload

N/A.

## Sample size

190 head pairs from 20 heads. Leave-both-heads-out cross-validation gives approximately 153
training pairs per fold (removing the ~37 pairs involving either test head). With 4-6
predictors per arm, this provides substantially more effective observations per predictor
than the 43 in Section 3.3's 15-head analysis.

## Sample size rationale

The 20-head circuit is the largest circuit with well-measured Walsh coefficients (split-half
r = 0.997). The cross-validation scheme matches Section 3.3 for comparability.

## Starting and stopping rules

N/A — all arms run once. No sequential testing or adaptive stopping.

## Manipulated variables

N/A — observational.

## Measured variables

**Target:** signed order-2 Walsh coefficient w_ij for each head pair, LASSO-recovered from
2000 random coalitions (split-half r = 0.997). Mean ablation is the primary primitive,
matching the 20-head circuit's existing data.

**Predictors (Arm 1):** OV overlap, Q-composition, K-composition, layer distance.

**Predictors (Arm 2):** Grassmannian distance, subspace overlap (Frobenius), attention pattern
similarity, data-dependent composition.

**Predictors (Arm 3):** EAP edge product, order-1 magnitude product.

**Predictors (Arm 4):** all of the above.

## Measured variables - File upload

N/A.

## Indices

Cross-validated R^2 (leave-both-heads-out) and jackknife SE for each arm.

In-sample R^2 is reported for comparison with Section 3.3 but carries no inferential weight.

## Indices - File upload

N/A.

## Statistical models

Ordinary least squares regression of w_ij on the predictor set, with leave-both-heads-out
cross-validation. Identical to Section 3.3 except for the predictor set and circuit.

For each test pair (i,j), the model trains on all pairs (a,b) with a not in {i,j} and b not
in {i,j}. Predicted values are collected across all held-out pairs, and CV R^2 is
1 - SS_res/SS_tot.

## Statistical models - File upload

N/A.

## Transformations

Predictors are standardized (zero mean, unit variance) within each cross-validation fold's
training set. The target is not transformed.

## Inference criteria

**H2 first (gate).** If Arm 3 (positive control) returns CV R^2 <= 0, the evaluation
protocol cannot detect signal at this sample size. Report H4 (uninformative null) and stop.
No conclusion about geometry follows.

**H1 confirmed** if Arm 3 passes the gate (CV R^2 > 0) and Arm 2 (data-dependent) also
achieves CV R^2 > 0.

**H1 strongly confirmed** if Arm 2 achieves CV R^2 > 0.10.

**H3 confirmed** if Arm 3 passes the gate but Arm 2 returns CV R^2 <= 0 — data-dependent
features fail despite the protocol being adequate.

**Arm 1 (weight-only replication)** is expected to return negative CV R^2, replicating
Section 3.3. A positive result would contradict the published finding and would be reported
as such.

**Arm 4 (combined, exploratory)** is reported but has no decision rule and does not support
any claim. With ~10 features and ~153 training pairs per fold, overfitting is likely. If it
exceeds both Arms 2 and 3, the features are complementary; if it matches the better arm,
the features are redundant.

## Data inclusion and exclusion

All 190 pairs are included. No pairs are excluded.

For the data-dependent composition predictor (Arm 2), only cross-layer pairs where
layer(i) < layer(j) receive a value. Same-layer and reverse-layer pairs receive zero for
this predictor and are retained in the regression.

## Missing data

A head pair whose data-dependent features cannot be computed (e.g., singular value
decomposition fails to converge) is dropped and the drop is reported. CV R^2 is computed
over the remaining pairs.

## Other planned analysis

**Exploratory 1.** Correlation matrix between the weight-only, data-dependent, and
positive-control features. If Grassmannian distance correlates strongly (|r| > 0.7) with OV
composition, the data-dependent version adds no new information.

**Exploratory 2.** Data-dependent features measured on ablated runs rather than clean runs.
If interaction under ablation is mediated by the altered forward pass, features measured
under the same ablation should predict better. This requires running the model with each head
individually ablated and collecting the remaining heads' activations — substantially more
compute, and exploratory because the feature set is not committed in advance.

**Exploratory 3.** Separate the LayerNorm contribution. Compare features computed with and
without the normalization part of LayerNorm (i.e., fold the affine part but leave the
data-dependent normalization in place vs. also replacing it with identity). If the
normalization is the missing ingredient, removing it from the feature computation should
degrade Arm 2 toward Arm 1.

**Exploratory 4.** Run the same analysis on 15-head IOI circuits with exact (exhaustive)
Walsh coefficients, to check whether the positive control result generalizes across circuit
sizes and coefficient estimation methods.

## Context and additional information

This experiment tests two things at once: the data-dependent geometry hypothesis and the
adequacy of the evaluation protocol. The positive control arm (Arm 3) is what makes the
result interpretable either way.

If H1 holds, Section 3.3's claim sharpens: the failure was specifically about weight
geometry, and the forward pass does carry the missing information.

If H3 holds, the claim needs revision. The failure of weight geometry is not about missing
data dependence — interaction is an emergent property of the coalition that pairwise subspace
relationships cannot decompose.

If H4 holds, Section 3.3's negative result is inconclusive rather than informative, and the
paper should acknowledge this limitation. The 32% sign-flip rate across ablation primitives
and the dyadic structure of the CV may together prevent any pairwise predictor from achieving
positive CV R^2.

---

## Log

Append only. Never edit above the line.

The last column is what distinguishes an amendment from a deviation, so you do not have to
decide which word to use: `nothing run`, `no results seen`, `results not opened`, `results seen`.

```
2026-08-11  created                              nothing run
2026-08-11  revised: added positive control arm, nothing run
            switched to 20-head circuit,
            added H2/H4 gate structure
2026-08-11  frozen at a8ac187a4709              nothing run
2026-08-11  bug fix: data_dependent_comp was    no results seen
            matmulling z (d_head) directly
            against W_Q (d_model, d_head);
            added W_O projection to get
            residual-stream vector first.
            Script-only change, prereg text
            unchanged.
2026-08-11  results computed                    results seen
            H3 confirmed: Arm 3 CV R²=0.199,
            Arm 2 CV R²=-0.082, Arm 1=-0.059
```
