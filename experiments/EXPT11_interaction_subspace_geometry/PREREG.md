# Where in the residual stream does head-pair interaction live?

**Status:** DRAFT

Sections use the [OSF Preregistration](https://osf.io/prereg/) question titles verbatim, so
this maps onto a registration without being rewritten. A question that does not apply is
answered **N/A** with the reason, never deleted.

## Research questions or hypotheses

The Walsh-Hadamard transform assigns each head pair a scalar interaction coefficient. That
scalar collapses all spatial information: it says heads A and B interact non-additively but
not where in the residual stream the interaction is mediated or at which layer it
concentrates.

The same quantity computed in activation space rather than at the logit is a vector:

$$\delta_{AB}^{(\ell)} = r^{(\ell)}_{\text{clean}} - r^{(\ell)}_{A\text{-ablated}} - r^{(\ell)}_{B\text{-ablated}} + r^{(\ell)}_{AB\text{-ablated}}$$

where $r^{(\ell)}$ is the residual stream at layer $\ell$. Stacked across prompts, the SVD
of the $(N \times d_\text{model})$ matrix of $\delta_{AB}$ vectors yields a principal
interaction subspace --- the directions where two heads' contributions fail to compose
additively.

This is the activation-space analogue of using epistasis to infer contact structure in
proteins. Rollins et al. (2019) showed that residue pairs with the strongest epistasis are
overwhelmingly close in 3D, sufficient to determine the fold. The question is whether the
interaction subspace recovers computational structure --- layer localization, compositional
relationships, and mechanism-specific directions --- in a transformer.

**H1 (layer localization).** The interaction subspace has higher energy (larger top singular
values of $\delta_{AB}$) at specific layers rather than being diffuse across all layers.
For cross-layer pairs where layer(A) < layer(B), the peak should fall between the two heads'
layers (A modifies B's input) or at or after B's layer (convergence at output).

**H2 (subspace sharing by interaction sign).** Sub-additive (masking) head pairs share
interaction subspaces: principal angles between their top-k interaction subspaces are smaller
than those of super-additive (synergistic) pairs at matched |w_ij|. The comparison is
stratified by interaction magnitude to avoid confounding sign with signal strength (two
random 5-dimensional subspaces in 768 dimensions have principal angles near 90 degrees almost
surely, so comparing against near-additive pairs with |w_ij| near zero would trivially
confirm).

**H3 (composition alignment).** For pairs where the interaction subspace aligns with either
head's OV output subspace (high fraction of the top interaction singular vector's norm
retained when projected onto the column space of W_OV), the interaction is compositional.
For pairs where the projection retains little norm, the interaction is mediated by something
else (LayerNorm rescaling, attention routing). Projecting onto the full column space rather
than individual columns avoids missing interaction vectors that lie inside the d_head-
dimensional OV subspace without aligning with any single column.

**H4 (scalar consistency).** Projecting $\delta_{AB}$ onto the logit direction
($W_U[\text{IO}] - W_U[\text{S}]$) and averaging across prompts should reproduce the scalar
Walsh coefficient. Because the final LayerNorm rescales differently across the four
conditions, residual-space additivity does not map exactly onto logit-space additivity.
Confirm at r > 0.85 (not 0.95); failure below 0.85 may indicate LayerNorm-mediated
interaction rather than a bug, and is reported as such.

**H4b (pre-layer zero check).** For layers strictly before min(layer_A, layer_B),
$\delta_{AB}^{(\ell)}$ must be exactly zero --- no causal path exists. This is a stronger
correctness check than H4 and costs nothing.

**H5 (null: diffuse and high-rank).** If the interaction subspace is diffuse across layers
and the top-5 singular vectors capture less than 50% of interaction energy at the peak layer
for a majority of pairs, the subspace framing does not compress the interaction into an
interpretable low-dimensional object. Report as a negative. (The interaction matrix is
$N \times d_\text{model}$ with rank at most $N$, so the kill condition is stated as a
fraction of captured energy rather than as an absolute rank threshold.)

## Foreknowledge of data or evidence

**From the paper (observed):**
- Order-2 Walsh coefficients for 190 pairs in the 20-head IOI circuit (split-half r = 0.997).
- Top interacting pairs: L8H6-L5H5 (w = +0.200, synergistic), L10H7-L9H6 (w = -0.164,
  masking), L10H7-L8H10 (w = -0.159, masking).
- Weight geometry and data-dependent subspace features both fail to predict Walsh
  coefficients (E10, CV R^2 = -0.06 and -0.08), while a positive control passes
  (CV R^2 = 0.20). Pairwise geometric features do not decompose interaction.
- Conditional marginal analysis shows S-inhibition heads gate negative name mover sign.

**From a scoping calculation on 2026-08-11 (observed, and this is the part that matters):**

`../E11_interaction_vector_scoping/` computed $\delta_{AB}$ for 12 of the 190 pairs — the 6
most sub-additive and 6 most super-additive by Walsh coefficient — over 200 prompts under
mean ablation. It carried no hypotheses and existed to decide whether a subspace study was
worth designing. **It ran before this document was consulted and it measured quantities this
document treats as predictions.** What it showed:

- **H4b is already answered.** Max $|\delta|$ at layers strictly before min(layer_A, layer_B)
  was **exactly 0.000e+00** for all 12 pairs.
- **H1's layer profile is observed** for those 12. Sub-additive pairs peaked at layer 10 or
  11; super-additive at 7, 8 or 11.
- **H5's rank question is answered for those 12, and it splits by interaction sign.**
  Participation ratio was 9.5–13.8 for sub-additive pairs against 1.2–6.6 for super-additive,
  with 90%-energy rank 14–26 against 1–13. **All 12 sat far below a matched-norm random null
  (48–141)**, so the interaction is strongly structured in every case; the sign split is
  about how many directions that structure occupies.
- Median mean-cancellation ratio **0.402** — roughly 60% of typical $\delta$ magnitude
  cancels in the mean.

**Consequences, stated rather than managed away.** H4b is a correctness check and seeing it
early costs nothing. H1 and H5 are different: for those 12 pairs they are retrodictions now.
This document therefore states **H1 and H5 as confirmatory over the 178 pairs not scoped**,
with the 12 reported separately as the exploratory set that generated them. That choice is
recorded here, before the full run, rather than after seeing whether it helps.

**One null from that run must not be reused.** Its first version sign-shuffled rows of
$\delta$, which leaves $\delta^\top\delta$ and therefore every singular value unchanged —
the null equalled the observed value in all 12 pairs by construction. It was replaced with
matched-norm random directions at identical $N$ and $d$, which is the null the numbers above
are quoted against. Any null used here must break the covariance structure, not just the
signs.

**Genuinely unobserved:**
- Every activation-space interaction vector $\delta_{AB}$ for the 178 pairs not scoped above.
- Every interaction subspace, its dimensionality, layer profile, and alignment with OV
  directions.
- Whether interaction subspaces are shared across pairs with the same interaction sign.

## Explanation of foreknowledge and managing unintended influences

E10's null result motivates this experiment: pairwise subspace features of individual heads
do not predict interaction, so the interaction may live in its own subspace aligned with
neither head. The E10 null is compatible with this hypothesis.

## Study type

Interventional --- the interaction vectors are defined by ablation (causal intervention on
head outputs).

## Intention for causal interpretation

Partially. The $\delta_{AB}$ vector is defined by causal interventions (ablation), so the
interaction subspace is a causal quantity. Whether it corresponds to a mechanism the model
uses (rather than an artifact of the ablation) is tested by H3 and the random-subspace
control.

## Blinding of experimental treatments

N/A --- no human judgement enters the measurement.

## Additional blinding during research or analysis

The Walsh coefficients are frozen from prior computation. The interaction subspaces are
computed from forward passes that do not reference the Walsh coefficients.

## Study design

**Circuit.** The 20-head IOI circuit (190 pairs). Same circuit as E10.

**Prompts.** The same 200 IOI prompts stored in `data/ioi_prompts_200.json`.

**Ablation.** Mean ablation (primary) and zero ablation (robustness check). For each
head pair (A, B), four conditions: clean, A-ablated, B-ablated, AB-ablated.

**Computation.** For each pair and each condition, cache the residual stream at every layer.
Compute $\delta_{AB}^{(\ell)}$ for each layer $\ell$, each prompt. Stack into an
$(N \times d_\text{model})$ matrix and take the SVD.

**Efficiency.** Clean and single-head-ablated residual streams are shared across pairs: 1
clean run + 20 single-ablation runs + 190 pair-ablation runs = 211 conditions total, each
over 200 prompts. Approximately 42,200 forward passes.

## Randomization

N/A --- deterministic given model and prompts.

## Data collection procedures

Model: GPT-2 small via TransformerLens with fold_ln=True, center_writing_weights=True,
center_unembed=True.

Prompts: `data/ioi_prompts_200.json` (frozen, same as E10).

Ablation: replace head output (hook_z) with mean activation across prompts (mean ablation)
or zero (zero ablation). Compute mean activations from the clean run.

Residual stream caching: `blocks.{l}.hook_resid_post` for each layer.

## Data collection procedures - File upload

N/A.

## Sample size

190 head pairs, 200 prompts, 12 layers. Per pair: a $(200 \times 768)$ interaction matrix at
each of 12 layers.

## Sample size rationale

Same circuit and prompt set as E10, for continuity.

## Starting and stopping rules

N/A --- all pairs computed once.

## Manipulated variables

Head ablation (mean or zero replacement of hook_z output) applied in four conditions per
pair: clean, A-ablated, B-ablated, AB-ablated.

## Measured variables

**Per pair, per layer:**
- Interaction matrix: $(N \times d_\text{model})$ matrix of $\delta_{AB}$ vectors.
- Singular values: from SVD of the interaction matrix.
- Top-k interaction subspace: top $k$ right singular vectors, where $k$ is chosen by
  the elbow of the singular value spectrum (or $k = 5$ if no elbow).
- Effective rank: number of singular values needed to capture 90% of variance.

**Per pair (aggregated across layers):**
- Layer profile: Frobenius norm of $\delta_{AB}^{(\ell)}$ **normalized by the clean residual
  stream norm** at each layer (residual stream norm grows with depth, so raw Frobenius norm
  is biased toward late layers).
- Peak layer: layer with maximum normalized Frobenius norm.
- OV alignment: fraction of the top interaction singular vector's norm retained when
  projected onto the column space of W_OV for heads A and B.
- Logit projection: $\delta_{AB}$ projected onto the IO-S logit direction, averaged across
  prompts. Should approximate the scalar Walsh coefficient (H4).
- Pre-layer zero check: $\delta_{AB}^{(\ell)}$ at layers before min(layer_A, layer_B) must
  be exactly zero (H4b).

**Across pairs:**
- Principal angles between interaction subspaces of different pairs, at the peak layer.
- Correlation between principal-angle distance and Walsh coefficient similarity.

## Measured variables - File upload

N/A.

## Indices

**H1:** variance of the layer profile (high = localized, low = diffuse). The peak layer
relative to the two heads' layers (between = input mediation, at/after B = output
convergence).

**H2:** mean principal angle between top-k interaction subspaces for sub-additive pairs vs
near-additive pairs. Tested by permutation of the interaction-sign labels (10,000
permutations).

**H3:** fraction of pairs where the top interaction singular vector has cosine similarity
> 0.5 with either head's OV write direction.

**H4:** Pearson correlation between the logit-projected $\delta_{AB}$ and the scalar Walsh
coefficient w_ij.

**H5:** median effective rank across pairs at the peak layer. If median effective rank
> d_model / 4 = 192, the subspace framing is uninformative.

## Indices - File upload

N/A.

## Statistical models

No regression. The analysis is geometric: SVD, principal angles, cosine similarities,
permutation tests.

## Statistical models - File upload

N/A.

## Transformations

None. All quantities are computed directly from the residual stream vectors.

## Inference criteria

**H1 confirmed** if 75% of pairs with |w_ij| > 0.05 have a single peak layer (top layer
captures > 2x the Frobenius norm of the median layer).

**H2 confirmed** if sub-additive pairs have smaller mean principal angles than super-additive
pairs at matched |w_ij| at p < 0.05 (permutation test of sign labels within magnitude bins).

**H3:** reported descriptively. If > 50% of strongly interacting pairs retain > 50% of norm
when projected onto either head's OV column space, composition is the dominant mediation
mode. If < 20%, the interaction is mediated by other mechanisms.

**H4 confirmed** if Pearson r > 0.85 between logit-projected $\delta_{AB}$ and scalar Walsh
coefficient. Failure below 0.85 may reflect LayerNorm-mediated interaction (the final LN
rescales differently across the four conditions) rather than a bug, and is reported as such.

**H4b confirmed** if $||\delta_{AB}^{(\ell)}||$ is within numerical precision of zero at
all layers before min(layer_A, layer_B).

**H5 (null):** if the top-5 interaction subspace captures < 50% of interaction energy at the
peak layer for a majority of pairs, the subspace framing does not compress the interaction
into an interpretable low-dimensional object.

## Data inclusion and exclusion

All 190 pairs included. Pairs with |w_ij| < 0.01 are retained but excluded from H1-H3
analyses (their interaction signal may be below the noise floor). Excluded pairs are reported
separately.

## Missing data

If SVD fails to converge for any pair-layer combination, that combination is dropped and
the drop is reported.

## Other planned analysis

**Control 1 (random subspace baseline).** For each pair, compare the interaction energy
captured by the top-k interaction subspace against k random directions. The interaction
subspace must capture significantly more variance than random directions of the same
dimensionality (p < 0.01, 1000 random draws). Addresses the Makelov interpretability-illusion
concern.

**Control 2 (ablation stability).** Compute interaction subspaces under both mean and zero
ablation. Compare via principal angles at the peak layer. If the subspaces are stable
(mean principal angle < 30 degrees), the finding is robust to ablation choice. If unstable,
report as ablation-dependent.

**Exploratory 1.** For the top 10 interacting pairs, visualize the layer profile and the top
interaction singular vector projected into token space (via W_U). Does the interaction
direction correspond to specific tokens (IO name, S name)?

**Exploratory 2.** For pairs involving backup name movers (if identifiable from order-1
scores), check whether the interaction subspace aligns with the backup heads' write
directions. If so, the sub-additive interaction is mediated by self-repair, which is a
finding about the mechanism of non-additivity rather than a confound.

## Context and additional information

E10 showed that pairwise geometric features of individual heads do not predict Walsh
interaction coefficients, even with a positive control confirming the evaluation protocol
works. This experiment tests a different hypothesis: that the interaction is not a function
of how A's and B's individual subspaces relate, but lives in its own subspace that may be
aligned with neither head. A low-rank, layer-localized interaction subspace would explain
E10's null while providing a geometric handle on interaction structure.

The protein-science analogy (Rollins et al. 2019) grounds the approach: in deep mutational
scanning, pairwise epistasis magnitude recovers 3D contact structure. The question is whether
pairwise interaction subspace geometry recovers computational structure in a transformer.

---

## Log

Append only. Never edit above the line.

The last column is what distinguishes an amendment from a deviation, so you do not have to
decide which word to use: `nothing run`, `no results seen`, `results not opened`, `results seen`.

```
2026-08-11  created                              nothing run
```
