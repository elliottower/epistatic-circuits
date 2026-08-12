# Verification packet: subspace epistasis results and mechanistic-views interpretation

## Context for the reviewer

We ran a pre-registered analysis testing whether head-level epistasis (measured
by pairwise Walsh-Hadamard interaction coefficients from full coalition
ablation sweeps) is predicted by residual-stream subspace overlap between
attention heads in GPT-2 small.

The result is a structured negative: pooled across 15 circuits and 1155 head
pairs, OV subspace overlap explains 7% of interaction variance (R^2 = 0.068).
But the relationship is modulated by layer distance and task complexity. We
want to verify our interpretation and check whether we are missing relevant
literature.

## Claims to verify

### 1. The un-foldable LayerNorm argument

**Claim**: LayerNorm has two parts. The affine part (learned scale gamma and
bias beta) can be folded losslessly into downstream weight matrices. The
normalization part (dividing by the per-token standard deviation) cannot be
folded because it is data-dependent — it varies per input and creates implicit
cross-head interaction that no weight-only measure can capture.

**Our use of this**: We argue this bounds the attainable R^2 of any weight-only
predictor, even if all epistasis were geometric. The close-layer vs far-layer
split (rho = 0.344 vs 0.009) is consistent with LN scrambling accumulating
over layers.

**Check**:
- Is it correct that the affine part folds exactly? TransformerLens does this
  via fold_ln=True.
- Is it correct that the normalization division is un-foldable? Has anyone
  formalized the approximation error from treating it as foldable?
- Has anyone measured how much the LN normalization affects composition scores
  in practice? E.g., comparing folded-weight composition to activation-derived
  composition.
- Is "LN scrambling accumulates over layers" a known phenomenon or our
  invention?

### 2. OV composition score formula

**Claim**: We compute OV subspace overlap as:

OV_overlap(i,j) = || W_OV_i^T @ W_OV_j ||_F / (||W_OV_i||_F * ||W_OV_j||_F)

where W_OV = W_V @ W_O, shape (d_model, d_model), computed from LN-folded
effective weights. This is Frobenius cosine similarity of the OV matrices, a
special case of Grassmannian subspace affinity.

**Check**:
- Is this the standard formula used in Elhage et al. 2021 ("A Mathematical
  Framework for Transformer Circuits")? Or do they use a different
  normalization?
- The Elhage composition score has a published erratum in which the scores in
  the original figure were normalized — is our formula consistent with the
  corrected version?
- Does anyone use principal angles or other Grassmannian metrics instead of
  Frobenius for head-head composition?

### 3. Q-composition and K-composition

**Claim**: Q-composition from head i to head j (l_i < l_j):

Q_comp(i,j) = || W_OV_i @ W_QK_j ||_F / (||W_OV_i||_F * ||W_QK_j||_F)

where W_QK_j = W_Q_j^T @ W_K_j. This measures how much head i's output
subspace aligns with what head j attends to.

**Our finding**: Q-composition adds marginal predictive power beyond OV overlap
(partial rho = 0.101, p = 6e-4) but is sometimes negatively correlated with
Walsh interaction per circuit (up to rho = -0.444).

**Check**:
- Is the negative correlation interpretable? Can high QK composition coincide
  with low pairwise interaction?
- One interpretation: complementary routing (head A steers B's attention
  usefully but neither needs the other). Is this a known pattern?
- Has anyone previously found negative correlations between composition scores
  and causal interaction measures?

### 4. Close-layer vs far-layer interaction structure

**Finding**: For head pairs within 2 layers (n = 466): rho(|Walsh|, OV_overlap)
= 0.344. For pairs more than 3 layers apart (n = 551): rho = 0.009. OV overlap
predicts interaction locally but not globally.

**Our interpretation**: LayerNorm normalization between layers scrambles the
weight-only prediction. Close-layer heads share similar LN statistics; distant
heads pass through more normalization layers that introduce data-dependent
interaction weight-based measures cannot capture.

**Check**:
- Is "close-layer heads compose more predictably from weights" a known finding?
  Elhage et al. 2021 discuss V-composition and Q-composition but do they report
  it being stronger for adjacent layers?
- Has anyone else reported that weight-based composition scores become less
  predictive with increasing layer distance?
- Is there a better explanation for the distance effect than LN scrambling?
  E.g., attention pattern routing becomes more complex with more layers between
  heads?

### 5. Walsh-Hadamard pairwise coefficients as an interaction measure

**Claim**: The Walsh-Hadamard transform of a set function f: 2^[n] -> R gives
Fourier-Walsh coefficients. The order-2 coefficient at index (1<<i)|(1<<j) is
the pairwise interaction between players i and j — the amount by which the
joint effect deviates from the sum of individual effects.

**Check**:
- Is this interpretation standard in Boolean function analysis / cooperative
  game theory?
- How does this relate to the Shapley interaction index? Are they measuring the
  same thing or different things?
- We previously verified (Perplexity round 1) that Walsh and Shapley are
  "related but distinct linear functionals." Confirm this is still the correct
  characterization.

### 6. Epistasis as a cross-view phenomenon (mechanistic views framework)

**Claim**: We interpret the results through the mechanistic-views framework
(Tower, 2026). Epistasis is measured in the object view (coalition ablation on
concrete heads) and we tested whether it reduces to the structural view
(gauge-invariant weight-space composition). The structured negative — geometric
for close layers, computational for far layers — suggests epistasis is a
stratified phenomenon: the mechanism changes with the resolution at which it is
examined.

**Specific claims**:
- Object view: epistasis is well-defined as a property of the coalition game
- Structural view: OV composition captures a component but not the majority
- Stratified view: close-layer interaction is geometric, far-layer interaction
  is computational — this is resolution-dependence
- The partial convergence (structural view weakly confirms object view for
  close-layer pairs) satisfies the spirit of criterion C3 (convergent validity)
  but with a scope restriction

**Check**:
- Is this a valid use of the mechanistic-views framework? Specifically:
  - Does epistasis fit naturally as an object-view claim?
  - Is testing OV composition a legitimate structural-view test?
  - Is the close-layer/far-layer split a legitimate stratification?
- Has anyone else applied the views framework to analyze why a mechanistic
  claim fails at one level but holds at another?
- The stratified view is described as "programmatic" in the atlas (no widely
  adopted methods yet). Is our close-layer/far-layer split a method for the
  stratified view, or are we misusing the concept?

### 7. The gt_known outlier

**Finding**: One circuit (gt_known, greater-than task, 7 heads, known circuit
from Hanna et al. 2024) shows rho_OV = 0.623 (p = 0.003), R^2 = 0.412 — far
stronger than any other circuit.

**Check**:
- The greater-than task relies heavily on direct residual-stream composition
  (magnitude comparisons propagated through layers). Is this characterization
  correct per Hanna et al. 2024?
- Would you expect the greater-than circuit to show more geometric epistasis
  than IOI, which relies more on routing (attention to the correct name)?
- With only 21 pairs, how cautious should we be about R^2 = 0.412?
  What is the expected R^2 under the null for 21 pairs and 3 predictors?

## What is OURS vs BORROWED

**Ours (novel contributions):**
- Full Walsh decomposition of 2^n coalition games on transformer circuits
- Testing whether Walsh pairwise coefficients correlate with OV/QK composition
- The pre-registered prediction set (P1-P5) and decision rule
- The structured negative result: geometric for close layers, computational
  for far layers
- The mechanistic-views interpretation: epistasis as a cross-view, stratified
  phenomenon
- Walsh Interaction Fraction (WIF), LOO epistasis, TSII as construct validity
  metrics for interaction

**Borrowed/standard (need correct attribution):**
- Walsh-Hadamard transform (signal processing / Boolean function analysis)
- OV circuit / W_OV composition (Elhage et al. 2021, "A Mathematical Framework
  for Transformer Circuits")
- Q-composition, K-composition (same paper)
- LayerNorm folding into effective weights (TransformerLens standard practice)
- Coalition game / set function framework (cooperative game theory)
- The mechanistic-views framework (Tower, 2026)
- The mechanistic-validity criterion framework (Tower, 2026)

## Format requested

For each claim (1-7), give:
- Confirmed / Partially correct / Incorrect
- Correct formulation if we got it wrong
- Key citation(s)
- Any important caveats or literature we should cite
- If you find anyone who has done something similar to claim 6 (applying
  multi-view frameworks to explain structured negatives in mechinterp),
  that would be especially valuable
