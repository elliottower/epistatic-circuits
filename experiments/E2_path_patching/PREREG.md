> **Provenance.** Reformatted from `docs/PREREG_PATH_PATCHING_EDGES.md`.
> Original frozen at commit `be71729`. View: `git show be71729:docs/PREREG_PATH_PATCHING_EDGES.md`

# Pre-registration: Directional edge decomposition of Walsh interactions

## Question

Do the undirected pairwise Walsh interaction coefficients recovered via
sparse compressed sensing correspond to direct causal edges (sender→receiver
through the residual stream), or are they mediated through intermediate
components?

## Motivation

Walsh coefficients w_{ij} measure the total interaction between heads i and j
but are symmetric — they do not distinguish sender from receiver. Path patching
isolates the direct causal effect of sender S on the output routed specifically
through receiver R, by corrupting S and freezing all intermediate components to
clean values. Comparing the two tells us whether Walsh-identified interactions
correspond to direct edges or to mediated functional coupling.

## Design

Use the 20-head IOI circuit from Phase 2 (same heads, same prompts, same model).
For each of the 190 head pairs, compute the direct path-patch effect in both
directions (earlier→later and later→earlier, though the latter is 0 by
architecture). Compare to the Walsh coefficients from sparse recovery.

### Corruption method

Mean ablation (replace sender's hook_z with its mean across prompts), consistent
with the ablation primitive used to compute Walsh coefficients in Phase 2.

### Path patching implementation

For edge S→R where layer(S) < layer(R):
1. Corrupt sender S (replace hook_z with mean)
2. Freeze all attention heads in layers between S and R to clean
3. Freeze all MLPs in layers from S to R-1 to clean
4. Freeze non-receiver heads in R's layer to clean
5. Let receiver R and all downstream components compute naturally
6. Measure change in logit diff vs clean

This isolates the direct S→R→output path.

### Controls

- Same-layer pairs (7 pairs): direct effect must be ~0 (sanity check)
- Reverse-direction pairs (layer(S) > layer(R)): 0 by architecture
- Bottom-20 Walsh pairs included alongside top-20 for full correlation

## Predictions

### H1 (direct correspondence): Spearman rho > 0.5

|Walsh coefficient| correlates with |direct path-patch effect| across all
190 pairs (including same-layer zeros). The interaction structure is
predominantly carried by direct edges. The Walsh method functions as a
cheap edge discovery tool: identify strong Walsh pairs, assign direction
by layer ordering.

### H2 (mediated interactions): rho < 0.3

Walsh-strong pairs show weak direct edges. The pairwise interaction is
real but routed through intermediate heads or MLPs. Same-layer pairs
with large Walsh coefficients are the clearest signal: their direct
effect is exactly 0, so any Walsh interaction must be mediated.

If H2 fires, follow-up: for each Walsh-strong pair with weak direct edge,
identify candidate mediator heads (highest Walsh interaction with both
endpoints) and test the two-hop path S→M→R.

### H3 (mixed): 0.3 < rho < 0.5

Some pairs are direct, some are mediated. Report the joint scatter and
characterize which head types (name movers, induction, backup) tend to
have direct vs mediated interactions.

### Falsifier

If rho ~ 0 AND no candidate mediator shows evidence of the two-hop path,
Walsh and path patching measure genuinely uncorrelated quantities. Report
as a negative result — the interaction structure does not decompose into
directed edges.

## Metrics

- Spearman and Pearson correlation: |w_{ij}| vs |PP(earlier→later)|
- For each pair: ratio direct/total (direct effect vs total sender AP effect)
- Same-layer sanity check: all |PP| < epsilon
- Scatter plot: Walsh coefficient vs direct path-patch effect, colored by
  layer distance
- Mediation fraction: what % of Walsh-strong pairs (top quartile) have
  direct effects in the bottom quartile?

## What this adds to the paper

If H1: sparse Walsh recovery is a complete edge discovery method (nodes +
edges + magnitudes from ~140 random coalitions). Competitive with EAP.

If H2/H3: Walsh and path patching capture complementary structure. Walsh
finds functional coupling (including mediated); path patching finds direct
causal edges. Together they give a richer circuit map than either alone.
