> **Provenance.** Reformatted from `docs/PREREG_MIB_CIRCUIT_BENCHMARK.md`.
> Original frozen at commit `be71729`. View: `git show be71729:docs/PREREG_MIB_CIRCUIT_BENCHMARK.md`

# Pre-registration: Walsh + Path Patching circuit discovery benchmarked on MIB

Pre-reg SHA: `2e7cf364`

## Question

Do circuits constructed from Walsh interaction scores, path patching scores,
or their combination achieve competitive faithfulness (CPR/CMD) compared to
EAP-IG on the IOI task in GPT-2 small?

## Motivation

Walsh coefficients measure *functional interaction* between heads — how much
their joint ablation deviates from the sum of individual effects. Path patching
measures *directed information flow* — the direct causal effect of one head on
another through the residual stream. These are nearly orthogonal (Spearman
rho=0.21, signed Pearson=0.07), suggesting they capture complementary
structure. If either or both produce faithful circuits, that validates Walsh
interaction analysis as a practical circuit discovery method, not just a
measurement tool.

## Design

### Circuit construction from head-pair scores

Walsh and path patching produce head-level or head-pair-level scores. MIB
evaluates edge-level circuits. To bridge this gap:

**Node importance (from order-1 Walsh + activation patching):**
Each head gets an importance score from its order-1 Walsh coefficient (or its
activation patching effect). All outgoing edges from a head are weighted by
that head's importance.

**Edge importance (from order-2 Walsh + path patching):**
For each directed pair (sender, receiver), the edge weight is:
- **Walsh**: |w_{ij}| (the absolute order-2 Walsh coefficient)
- **Path patching**: |PP(sender -> receiver)| (the direct causal effect)
- **Combined**: alpha * |Walsh_ij| + (1 - alpha) * |PP_ij|, with alpha chosen
  by cross-validation on the train split

For same-layer pairs where path patching is undefined, use Walsh only.

**Converting to MIB format:**
MIB circuits are sets of (source_node, target_node) edges in the computational
graph, where nodes are attention heads and MLPs. Our edge scores map directly
to attention-head-to-attention-head edges. We include MLP nodes in the circuit
by adding all MLP edges from layers that contain at least one selected head
(since Walsh/PP don't score MLPs independently, this is the conservative
choice).

### Evaluation protocol (following MIB exactly)

1. For each method (Walsh, PP, combined, EAP, EAP-IG), rank all edges by
   importance score.
2. At each proportion k in {.001, .002, .005, .01, .02, .05, .1, .2, .5, 1},
   include the top k fraction of edges.
3. Compute faithfulness: f(C_k, N; m) = (m(C_k) - m(empty)) / (m(N) - m(empty))
   where m = logit_diff(IO) - logit_diff(S) on IOI prompts.
4. Compute CPR = area under faithfulness curve (trapezoidal rule).
5. Compute CMD = area between faithfulness curve and f=1 line.
6. Ablation type: counterfactual (following MIB default for evaluation).

### Methods compared

1. **EAP (mean ablation)** — baseline, linear approximation to activation patching
2. **EAP-IG (CF)** — SOTA from MIB benchmark
3. **Walsh order-1** — node-level circuit from individual head importance
4. **Walsh order-2** — edge-level circuit from pairwise interaction coefficients
5. **Path patching** — edge-level circuit from directed causal effects
6. **Walsh + PP combined** — weighted combination of Walsh and PP edge scores
7. **Random** — uniform random edge importance (negative control)

### Data

- Model: GPT-2 small (matching MIB IOI evaluation)
- IOI prompts: 200 prompts (same as Phase 2), split 100 train / 100 test
- Train split used for computing all scores (Walsh, PP, EAP, EAP-IG)
- Test split used for evaluating faithfulness
- Walsh and PP scores already computed from Phase 2

### Implementation

For circuit evaluation at each k:
- Ablate all edges NOT in C_k by replacing the source's hook_z output at that
  edge with its mean across the test prompts
- For the empty circuit (m(empty)): ablate all heads (replace all hook_z with
  mean)
- Edges are (attention head) -> (attention head) or (attention head) -> (MLP)
  or (MLP) -> (attention head), following the computational graph

## Predictions

### H1 (Walsh competitive): Walsh CMD < 2x EAP-IG CMD

Walsh interaction coefficients produce circuits whose CMD is within 2x of
EAP-IG. Walsh captures real functional structure that translates into faithful
circuits.

### H2 (combination wins): Combined CMD < min(Walsh CMD, PP CMD, EAP-IG CMD)

The Walsh+PP combination discovers more faithful circuits than any single
method, because the two capture complementary structure (functional interaction
vs directed information flow).

### H3 (Walsh inferior): Walsh CMD > 2x EAP-IG CMD

Walsh interaction coefficients are poor edge importance scores for building
faithful circuits. The pairwise interaction structure is real but does not
directly predict which edges to keep for circuit evaluation. Walsh remains
valuable as a measurement tool (quantifying interaction strength) but not as
a circuit discovery method.

### Falsifier

If Walsh CPR and CMD are both worse than random circuits, the Walsh
interaction spectrum does not capture circuit-relevant structure at all.

## Metrics

- CPR (area under faithfulness curve) for each method
- CMD (area between faithfulness curve and 1) for each method
- Faithfulness at each k for each method (full curves)
- Ranking of methods by CPR and CMD
- Walsh vs EAP-IG edge-score Spearman correlation

## What this adds to the paper

If H1 or H2: Walsh-based circuit discovery is competitive with or better
than gradient-based methods, using only ~140 forward passes (from sparse
recovery) instead of O(Z*L) passes for EAP-IG.

If H3: Walsh coefficients measure genuine interaction structure that is
complementary to, but different from, the edge importance required for
faithful circuit construction. This is itself a finding — it means
"interaction strength" and "circuit membership" are not the same concept.
