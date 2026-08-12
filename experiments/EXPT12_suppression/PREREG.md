# Does amplifying one head suppress the deficit from ablating another?

**Status:** DRAFT

Sections use the [OSF Preregistration](https://osf.io/prereg/) question titles verbatim, so
this maps onto a registration without being rewritten. A question that does not apply is
answered **N/A** with the reason, never deleted.

## Research questions or hypotheses

The Walsh interaction coefficient classifies head pairs as sub-additive (masking),
additive (independent), or super-additive (synergistic). These labels predict a behavioral
consequence that has not been tested: whether one head can compensate for the loss of
another.

If two heads mask each other (sub-additive, negative w_ij under our sign convention), they
do overlapping work. Ablating A should leave B's contribution partially redundant with what
A was doing, so amplifying B should recover some of the deficit.

If two heads are synergistic (super-additive, positive w_ij), each needs the other to
function. Ablating A should impair B's effectiveness, so amplifying B should not recover the
deficit --- B cannot do its job without A's contribution.

If two heads are additive (w_ij near zero), they contribute independently. Amplifying B
should not affect the deficit from ablating A, because B's contribution is orthogonal to A's.

This is the MI analogue of an extragenic suppressor screen in genetics: a compensating
mutation at a different locus rescues a loss-of-function allele, revealing functional
relationships between genes. (This is distinct from a rescue experiment, which restores the
same component that was disrupted. A true rescue --- ablate A, patch A's clean activation
back --- is run as a validity check on the ablation operator but is not the main experiment.)

**H1.** Walsh interaction sign predicts suppression success. Sub-additive pairs show higher
suppression rates than additive pairs, which show higher rates than super-additive pairs.

**H2.** Suppression rate correlates with Walsh coefficient magnitude for sub-additive pairs.
The more strongly two heads mask each other, the more effectively one suppresses the other's
deficit.

**H3 (null).** Suppression rate is uncorrelated with Walsh interaction structure. The Walsh
coefficient captures something about joint ablation that does not translate to single-ablation
compensation.

## Foreknowledge of data or evidence

**From the paper (observed):**
- Order-2 Walsh coefficients for 190 pairs (split-half r = 0.997).
- Top sub-additive pairs: L10H7-L9H6 (w = -0.164), L10H7-L8H10 (w = -0.159),
  L10H7-L8H6 (w = -0.129).
- Top super-additive pairs: L8H6-L5H5 (w = +0.200), L5H5-L8H10 (w = +0.089),
  L8H6-L6H9 (w = +0.088).
- Order-1 scores (marginal effects) for all 20 heads. Largest magnitude: L8H10 (-0.533),
  L8H6 (-0.462), L5H5 (-0.440).
- Conditional marginal analysis shows S-inhibition heads gate negative name mover sign:
  removing S-inhibition inverts negative name movers from harmful to helpful.

**Genuinely unobserved:**
- Every rescue measurement (ablate A + amplify B).
- Whether rescue rate tracks Walsh interaction sign or magnitude.
- Whether the conditional marginal gating translates to successful rescue.

## Explanation of foreknowledge and managing unintended influences

The Walsh coefficients define the predictions. The conditional marginal finding (Section 4)
suggests that S-inhibition removal changes negative name movers' functional role, which would
predict failed rescue for that pair type (the relationship is not compensatory but gating).
This is a specific prediction, not a post-hoc rationalization.

## Study type

Interventional --- ablation plus amplification.

## Intention for causal interpretation

Yes. The rescue intervention directly tests whether one head's contribution can substitute
for another's.

## Blinding of experimental treatments

N/A --- no human judgement enters the measurement.

## Additional blinding during research or analysis

Walsh coefficients are frozen from prior computation. Rescue measurements do not reference
them during computation.

## Study design

**Circuit.** The 20-head IOI circuit (190 pairs).

**Prompts.** The same 200 IOI prompts stored in `data/ioi_prompts_200.json`.

**Metric.** Logit difference: logit(IO) - logit(S) on the final token position.

**Baseline measurements.** For each head A:
- Clean logit difference: $L_\text{clean}$
- Ablated logit difference: $L_{A\text{-abl}}$
- Amplified logit difference: $L_{B\times\alpha}$ (amplify B with A intact)

The single-amplification condition ($L_{B\times\alpha}$) is required to distinguish "B
compensates for A's loss" from "amplifying B raises performance regardless." Without it,
every suppression measurement is contaminated by B's main effect.

Only heads with $|L_{A\text{-abl}} - L_\text{clean}| > 0.1$ (meaningful change from
ablation) are used as ablation targets.

**Suppression measurements.** For each ordered pair (A ablated, B amplified):
- Ablate A (mean ablation) AND scale B's output by factor $\alpha$
- Measure logit difference: $L_{A\text{-abl}, B \times \alpha}$

**Suppression metric.** The suppression effect is the interaction term of the
ablate/amplify factorial:

$$S_{AB} = (L_{A\text{-abl}, B\times\alpha} - L_{A\text{-abl}}) - (L_{B\times\alpha} - L_\text{clean})$$

This subtracts B's main amplification effect, isolating the component of B's amplification
that specifically compensates for A's absence. This is the same second-order structure as
the Walsh coefficient itself, now over a mixed ablate/amplify factorial rather than a
double-ablation factorial.

Recovery rate uses absolute distance to clean to handle heads with negative contributions
(negative name movers have $D_A < 0$, so dividing by signed $D_A$ inverts the meaning):

$$R_{AB} = 1 - \frac{|L_{A\text{-abl}, B\times\alpha} - L_\text{clean}|}{|L_{A\text{-abl}} - L_\text{clean}|}$$

$R_{AB} = 1$ means full recovery; $R_{AB} = 0$ means no change; $R_{AB} < 0$ means the
intervention moved performance further from clean.

**Amplification factor.** $\alpha = 2.0$ (double the head's output). This is the simplest
intervention. Doubling changes the LayerNorm denominator for everything downstream, so part
of the suppression signal may be a global rescaling artifact. The LN scale factor per
condition is reported. If results are ambiguous, a dose-response sweep over
$\alpha \in \{1.5, 2.0, 2.5, 3.0\}$ is run as exploratory.

**Amplification implementation.** Scale the head's hook_z output: `hook_z[:, :, h, :] *= alpha`.
This preserves the direction of B's output while increasing its magnitude.

**Ablation implementation.** Mean ablation: replace A's hook_z with the mean across prompts,
computed from the clean run.

**True rescue control.** For each head A, ablate A and then patch A's own clean activation
back. Recovery should be near-perfect; failure indicates the ablation operator has
collateral effects beyond removing A's contribution.

## Randomization

N/A --- deterministic given model and prompts.

## Data collection procedures

Model: GPT-2 small via TransformerLens with fold_ln=True, center_writing_weights=True,
center_unembed=True.

Prompts: `data/ioi_prompts_200.json`.

Forward passes required:
- 1 clean run (200 prompts)
- 20 single-ablation runs (to measure $L_{A\text{-abl}}$)
- 20 single-amplification runs (to measure $L_{B\times\alpha}$)
- 20 true-rescue runs (ablate A + patch A's clean activation back)
- Up to 380 suppression runs (ablate A + amplify B, for each ordered pair)
- Total: ~441 conditions x 200 prompts = ~88,200 forward passes

## Data collection procedures - File upload

N/A.

## Sample size

Up to 380 ordered rescue measurements (A ablated, B amplified; and B ablated, A amplified)
from 190 unordered pairs. Pairs where neither head has $|D_A| > 0.1$ are excluded from the
rescue analysis.

## Sample size rationale

Same circuit as E10 and EXPT11, for continuity. The 20-head circuit provides enough pairs
with diverse Walsh coefficient values (range: -0.164 to +0.200) to test the sign-rescue
correlation.

## Starting and stopping rules

N/A --- all rescue measurements computed once.

## Manipulated variables

Two interventions per rescue trial: (1) mean ablation of head A, (2) output scaling of
head B by factor $\alpha = 2.0$.

## Measured variables

**Per ordered pair (A ablated, B amplified):**
- Combined logit difference: $L_{A\text{-abl}, B \times \alpha}$
- Suppression interaction: $S_{AB} = (L_{A\text{-abl}, B\times\alpha} - L_{A\text{-abl}}) - (L_{B\times\alpha} - L_\text{clean})$
- Recovery rate: $R_{AB} = 1 - |L_{A\text{-abl}, B\times\alpha} - L_\text{clean}| / |L_{A\text{-abl}} - L_\text{clean}|$
- LayerNorm scale factor at the final LN for this condition vs clean

**Per unordered pair:**
- Symmetric suppression: $(S_{AB} + S_{BA}) / 2$
- Symmetric recovery: $(R_{AB} + R_{BA}) / 2$
- Walsh coefficient: w_ij (from existing data)

## Measured variables - File upload

N/A.

## Indices

**H1:** Spearman rank correlation between Walsh coefficient w_ij and symmetric suppression
interaction S. Sub-additive pairs (negative w_ij) should show larger positive suppression
than super-additive pairs (positive w_ij).

**H2:** Pearson correlation between |w_ij| and suppression interaction S, restricted to
sub-additive pairs (w_ij < -0.01). Should be positive (stronger masking = more effective
suppression).

## Indices - File upload

N/A.

## Statistical models

No regression. The analysis is correlational: Spearman/Pearson between Walsh coefficients
and rescue rates, plus a permutation test.

## Statistical models - File upload

N/A.

## Transformations

None.

## Inference criteria

**H1 confirmed** if Spearman correlation between w_ij and symmetric suppression interaction
is negative at p < 0.01 (permutation test, 10,000 permutations of pair labels).

**H2 confirmed** if Pearson r > 0.3 between |w_ij| and suppression interaction among
sub-additive pairs (w_ij < -0.01), p < 0.05.

**H3 (null)** if neither H1 nor H2 is confirmed: Walsh interaction structure does not
predict suppression success.

## Data inclusion and exclusion

Pairs are excluded from the rescue analysis if neither head has $|D_A| > 0.1$. The threshold
is chosen to exclude heads whose ablation produces negligible deficits (nothing to rescue).
Excluded pairs are listed in the results.

## Missing data

N/A --- all measurements are deterministic.

## Other planned analysis

**Exploratory 1 (dose-response).** Sweep $\alpha \in \{1.0, 1.5, 2.0, 2.5, 3.0, 4.0\}$
for the top 10 sub-additive and top 10 super-additive pairs. Plot suppression interaction vs
amplification factor. Sub-additive pairs should show monotonic improvement; super-additive
pairs should show flat or declining suppression.

**Exploratory 2 (directional suppression).** Instead of scaling B's output magnitude, steer
it toward A's mean output direction: $z_B' = z_B + \beta \cdot \hat{z}_A$, where
$\hat{z}_A$ is the unit vector of A's mean output. This tests whether suppression requires B
to produce output similar to A's (functional substitution) or just more of its own output
(capacity compensation).

**Exploratory 3 (gating pairs).** For pairs where the conditional marginal analysis shows a
gating relationship (S-inhibition gates negative name movers), the suppression prediction is
nuanced. Ablating S-inhibition changes what negative name movers do, so amplifying negative
name movers after S-inhibition ablation should amplify the wrong behavior (suppression
interaction negative). Test this specific prediction.

**Exploratory 4 (connection to EXPT11).** If EXPT11 identifies interaction subspaces, test
whether successful suppression pairs have interaction subspaces aligned with the suppressor
head's output direction. This would connect the geometric (where the interaction lives) and
behavioral (who can compensate) descriptions.

**Exploratory 5 (norm-preserving amplification).** To control for LayerNorm artifacts from
doubling a head's output norm, run a variant where B's hook_z is scaled but then
renormalized to preserve its original L2 norm. This changes direction without changing
magnitude, isolating the directional component of amplification from the scale component.

## Context and additional information

Extragenic suppressor screens are standard in genetics: a compensating mutation at a
different locus rescues a loss-of-function allele, establishing functional relationships
between genes. The translation to circuits is direct: amplify one component to compensate for
the ablation of another. The suppression interaction $S_{AB}$ has the same second-order
structure as the Walsh coefficient (it is the Mobius coefficient of the ablate/amplify
factorial), which makes the connection between the two quantities principled rather than
ad hoc.

The Walsh coefficient provides a quantitative prediction for suppression success: sub-additive
pairs should suppress (they do overlapping work), super-additive pairs should not (they need
each other). This prediction has not been tested. If confirmed, the Walsh interaction
spectrum is not just a description of joint ablation effects but a predictor of compensation
structure --- which components are functionally interchangeable and which are functionally
dependent.

The conditional marginal analysis from Section 4 adds a complication. S-inhibition heads
gate negative name movers: removing S-inhibition changes what negative name movers do, rather
than leaving a gap that negative name movers could fill. For these pairs, the Walsh
coefficient (sub-additive) would predict suppression, but the gating mechanism predicts
failure. The suppression experiment distinguishes these interpretations.

---

## Log

Append only. Never edit above the line.

The last column is what distinguishes an amendment from a deviation, so you do not have to
decide which word to use: `nothing run`, `no results seen`, `results not opened`, `results seen`.

```
2026-08-11  created                              nothing run
```
