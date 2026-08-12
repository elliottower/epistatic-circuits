> **Provenance.** Reformatted from `docs/PREREG_MIB_FAITHFULNESS.md`.
> Original frozen at commit `be71729`. View: `git show be71729:docs/PREREG_MIB_FAITHFULNESS.md`

# Pre-registration: Walsh-based circuit discovery evaluated with MIB faithfulness metrics

## Question

Can sparse Walsh interaction recovery, applied to all 144 attention heads in
GPT-2 small, discover circuits competitive with or complementary to
gradient-based attribution methods (EAP, activation patching)?

## Motivation

Sparse Walsh recovery compresses 2^n coalition sweeps into O(k) random samples
and recovers exact (not linearized) pairwise interaction coefficients.  Path
patching gives directed causal edges.  Both yield edge/node importance scores.
Walsh and path patching are nearly orthogonal (Spearman rho = 0.21 on 20-head
IOI), suggesting they capture different structure.  If combining them improves
circuit faithfulness, Walsh is a useful complement to gradient methods.

## Design

### Phase A — compute importance scores (all 144 heads)

1. **Walsh node importance**: Sample M = 12,500 random coalitions over all 144
   attention heads.  Each coalition randomly mean-ablates a subset of heads and
   measures logit diff on 200 IOI prompts.  LASSO sparse recovery targets the
   k = 10,440 order-1 and order-2 coefficients (144 + C(144,2)).  Node
   importance for head h = sum of |Walsh coefficient| for all pairs involving h,
   plus the order-1 magnitude of h.

2. **Activation patching (ActP)**: Individual mean ablation of each of 144
   heads.  Node importance = absolute change in logit diff.

3. **EAP (mean ablation)**: Edge attribution patching with mean ablation.
   Linearized gradient × activation approximation.  Aggregate edge scores to
   node importance by summing absolute outgoing edge scores.

4. **Combined Walsh+ActP**: Rank sum — normalize Walsh rank and ActP rank to
   [0,1], sum them.  Heads ranked by combined score.

5. **Combined Walsh+EAP**: Same rank-sum approach.

6. **Random**: Uniform random importance (mean over 5 seeds).

### Phase B — evaluate faithfulness (MIB protocol)

For each method, rank all 144 heads by importance.  At circuit sizes
k_heads ∈ {1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 72, 100, 144}, include
top-k heads and mean-ablate all others.  Measure:

- m(C) = mean logit diff of the circuit (top-k heads active)
- m(N) = mean logit diff of full model (no ablation)
- m(∅) = mean logit diff with all 144 heads ablated (empty circuit)

Faithfulness:
  f(C, N; m) = (m(C) - m(∅)) / (m(N) - m(∅))

Metrics (Riemann sum approximation):
  CPR = area under faithfulness curve (vs k/144)
  CMD = area between faithfulness curve and f = 1

### Ablation

Mean ablation throughout.  Each head's hook_z output is replaced with its mean
across the 200 IOI prompts.

### Prompts

Same 200 IOI prompts from Phase 2 (seed 42).  Logit diff = logit(IO name) -
logit(S name).

## Predictions

### H1 (Walsh competitive): Walsh CPR within 0.10 of ActP CPR

Walsh interaction recovery, despite never measuring individual head importance
directly, produces circuit rankings within 0.10 CPR of activation patching.
The interaction spectrum captures enough information to identify important heads.

### H2 (combination improves): Combined CMD < min(Walsh CMD, ActP CMD) - 0.02

Combining Walsh interaction structure with activation patching scores produces
circuits with lower CMD than either alone.  The complementary information
(nonlinear interaction vs linear causal effect) improves the ranking at
intermediate circuit sizes.

### H3 (null): No method beats ActP by more than 0.02 CMD

Activation patching is already near-optimal for node-level circuit discovery on
IOI, and the additional information from Walsh coefficients does not improve the
ranking meaningfully.

## What this adds to the paper

If H1+H2: Walsh sparse recovery is a practical circuit discovery tool that
captures complementary structure to gradient methods.  The paper can claim
that Walsh + AP produces better circuits than either alone, with concrete
CPR/CMD numbers benchmarked against MIB methodology.

If H3: the interaction spectrum is informative for understanding circuit
*structure* (which heads interact) but does not improve circuit *discovery*
(which heads to include).  The paper frames Walsh as an analysis tool, not a
discovery tool.

## Pre-registration SHA

To be filled after commit.
