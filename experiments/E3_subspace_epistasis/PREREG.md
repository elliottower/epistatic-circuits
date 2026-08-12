> **Provenance.** Reformatted from `docs/PREREG_SUBSPACE_EPISTASIS.md`.
> Original frozen at commit `be71729`. View: `git show be71729:docs/PREREG_SUBSPACE_EPISTASIS.md`

# Pre-registration: Subspace Decomposition of Head-Level Epistasis

## Research question

Is head-level epistasis (measured by pairwise Walsh interaction coefficients)
predicted by residual-stream subspace overlap between heads?

## Background

We have full 2^n coalition sweeps for multiple circuits (IOI, RTI, GT,
induction) under multiple ablation types. From these we extract pairwise Walsh
coefficients W_{ij} measuring the interaction strength between heads i and j.

In a standard transformer, all inter-head communication goes through the
residual stream (Elhage et al. 2021). Head-level epistasis can arise from
two mechanisms:
1. OV subspace overlap: head A writes into a subspace that head B reads from
2. QK routing dependency: head A's output shifts attention patterns in head B

Both are computable from weight matrices alone. One important caveat: LayerNorm has two parts — an affine part (learned
scale and bias) and a normalization part (data-dependent division by
per-token standard deviation). The affine part folds exactly and losslessly
into downstream weight matrices (standard practice via TransformerLens
fold_ln=True). The normalization part, however, cannot be folded because
it is data-dependent: it varies per input and creates implicit cross-head
interaction that no weight-only composition score can capture. All
weight-based composition scores (OV overlap, QK composition) must use
LN-folded effective weights to remove the affine contamination, but even
with best-practice folding, some fraction of measured epistasis could be
LN-normalization-driven rather than QK/OV-driven. This bounds how large
an R^2 we should expect in P4 even under the "fully geometric" hypothesis.

Note: Walsh-Hadamard coefficients are related to but distinct from Shapley
interaction indices. Both are linear functionals of the same set function
but weight coalitions differently (Walsh = Fourier/Hadamard basis;
Shapley = average over orderings). We use Walsh coefficients throughout
because the full-spectrum decomposition into order-1/2/3+ terms is more
informative than pairwise interaction indices alone.

To our knowledge, no prior work has tested whether pairwise interactions
between attention heads (measured by any method) correlate with subspace
overlap (OV or QK composition). This is the gap we aim to fill.

## Operationalizations

### Walsh pairwise interaction (from coalition data, already computed)
For each pair (i,j) in a circuit, the order-2 Walsh coefficient magnitude
|W_{ij}| measures how much the joint effect of i and j deviates from the sum
of their individual effects.

### OV subspace overlap (from LN-folded weights, to compute)
For heads i=(l_i, h_i) and j=(l_j, h_j):
  OV_overlap(i,j) = || W_OV_i^T @ W_OV_j ||_F / (|| W_OV_i ||_F * || W_OV_j ||_F)
where W_OV = W_V @ W_O (Elhage et al. 2021), shape (d_model, d_model),
computed from LayerNorm-folded effective weights.
This is the subspace affinity measure (a named special case of the
Grassmann-distance family; see Ye & Lim 2016, Heckel & Bolcskei 2017),
equivalent to cosine similarity of the OV matrices treated as flattened
vectors. We also report principal angles between subspaces (via SVD of
U^T @ V for orthonormal bases U, V; cos(theta_i) = sigma_i) as a
robustness check.

### QK composition (from LN-folded weights, to compute)
For head j in a later layer than head i:
  QK_comp(i,j) = || W_OV_i @ W_QK_j ||_F / (|| W_OV_i ||_F * || W_QK_j ||_F)
where W_QK_j = W_Q_j^T @ W_K_j, shape (d_model, d_model), computed from
LayerNorm-folded effective weights. This is Q-composition from Elhage et al.
2021. Measures how much head i's output subspace aligns with what head j
attends to. Only defined for l_i < l_j.

### Virtual attention (QK-OV composition path)
For head j attending to positions influenced by head i (l_i < l_j):
  VA(i,j) = || W_Q_j^T @ W_OV_i @ W_K_j ||_F
This measures whether head i's output changes head j's attention pattern.

## Predictions

### P1: OV overlap predicts pairwise Walsh interaction
Spearman correlation between |W_{ij}| and OV_overlap(i,j) > 0.3 across all
head pairs in each circuit.

Rationale: if epistasis is primarily geometric (shared subspaces), OV overlap
should predict interaction strength.

### P2: QK composition adds predictive power beyond OV overlap
Partial Spearman correlation of |W_{ij}| with QK_comp(i,j), controlling for
OV_overlap(i,j), is positive and significant (p < 0.05) for circuits with
faithfulness > 0.5.

Rationale: QK routing captures computational dependency beyond geometric
overlap.

### P3: Layer distance modulates interaction type
For adjacent-layer pairs (|l_i - l_j| <= 2), QK composition contributes
more to Walsh interaction than OV overlap.
For distant-layer pairs (|l_i - l_j| > 3), OV overlap dominates.

Rationale: QK routing is a local (sequential) mechanism; OV overlap is a
global (parallel) mechanism via the residual stream.

### P4: Variance decomposition
A linear model predicting |W_{ij}| from [OV_overlap, QK_comp, layer_distance]
achieves R^2 > 0.3 for high-faithfulness circuits.

Note: the un-foldable LN normalization nonlinearity is a live source of
interaction that weight-only predictors cannot capture. This places an
inherent ceiling on attainable R^2 — even if all epistasis were geometric,
the LN remainder would depress R^2 below 1.0. We therefore use conservative
thresholds:

If R^2 > 0.5: head-level epistasis is substantially geometric.
If R^2 < 0.2: weight-only subspace overlap does not explain interactions;
  epistasis is primarily computational or LN-mediated.
If 0.2 < R^2 < 0.5: mixed, both contribute; the LN-mediated fraction is
  bounded above by (1 - R^2) but not directly measurable from weights.

### P5: Ablation-type sensitivity
OV overlap predicts Walsh interaction better under mean ablation than under
resample ablation. Resample ablation preserves more distributional structure,
so routing-dependent epistasis should be more visible under resample.

## What we already have
- Full coalition sweep data for IOI (5 circuits x 15 heads), RTI (5 circuits x 15 heads),
  GT (5 circuits x 7 heads), induction (5 circuits x 7 heads)
- Walsh decomposition with pairwise coefficients for all of the above
- GPT-2 small weight matrices (publicly available)

## What we need to compute
- OV overlap matrix for each circuit's heads (weight matrices only, no GPU)
- QK composition matrix for each circuit's heads (weight matrices only, no GPU)
- Virtual attention matrix (weight matrices only, no GPU)
- Spearman correlations, partial correlations, linear regression
- All computable locally with numpy, no coalition sweeps needed

## Analysis plan
1. Extract pairwise Walsh coefficients from existing NPZ files
2. Load GPT-2 weights via TransformerLens, fold LayerNorm into effective
   W_Q, W_K, W_V, W_O per head (TransformerLens provides this via
   model.W_Q etc. after fold_ln=True in from_pretrained)
3. Compute OV overlap and QK composition matrices from folded weights
4. Compute principal-angle overlap as robustness check
5. Test P1-P5 (Spearman correlations, partial correlations, linear model)
6. Visualize: scatter plots of |W_{ij}| vs OV_overlap, colored by layer distance

## Decision rule
- If P1 confirmed (rho > 0.3) AND P4 shows R^2 > 0.3: report subspace
  decomposition as main finding, head-level epistasis is substantially geometric
- If P1 fails (rho < 0.3): head-level epistasis is computational or
  LN-mediated, not geometric; subspace overlap does not explain interactions;
  report as negative result
- If P2 confirmed: QK routing adds explanatory power; report both components
- In all cases, note that weight-only analysis has an inherent ceiling due
  to the un-foldable LN normalization nonlinearity

## Pre-registration hash
SHA-256: 3ef962bc32110b98bf3e1ebf934debf60ec6587f84e89fad4476ac1eb2a1499f
Frozen: 2026-08-07
