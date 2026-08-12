> **Provenance.** Reformatted from `preregistrations/prereg_v12_selective_pressures.md`.
> Original frozen at commit `81c628e`. View: `git show 81c628e:preregistrations/prereg_v12_selective_pressures.md`

# Pre-registration v12: Selective Pressures on Circuit Epistasis

Blind predictions for the EpistasisBench evaluation of six IOI circuits
discovered by different methods. All predictions below are based on
theoretical reasoning about each method's inductive bias, the known head
lists, and pairwise overlaps. No coalition table values, Walsh spectra,
or EpistasisBench outputs have been viewed.


## Disclosure: observed vs unobserved data

### Observed (design parameters, not results)

- Circuit head lists for all six circuits (required to set up coalition sweeps)
- Pairwise head overlaps (computed from head lists)
- Walsh discovery results from 144-head game (energy spectrum: 81% order-1,
  19% order-2; used to select C5 heads)
- Prior zero-ablation spectra for C3 and C4 from v7.1 (C3: 2.8% order-3+,
  C4: <0.01% order-3+)
- Prior mean-ablation finding: old-C1 (RTI) flipped from 8.3% to 0.2%
  order-3+ when switching from zero to mean ablation (v8/v10 calibration)

### Genuinely unobserved (confirmatory)

- All 15-head coalition tables for C1, C2, IC-15, C5 under mean ablation
- All Walsh energy spectra for the 15-head subgames
- All EpistasisBench outputs (LASSO-Walsh R², iRF interactions, shapiq estimates)
- Faithfulness deltas for all six circuits
- Any C3/C4 mean-ablation spectra (only zero-ablation was observed in v7.1)


## The six circuits

| ID | Method | Scoring | Heads |
|----|--------|---------|-------|
| C1 (weight-IOI) | QK nuclear norm + composition strength | Node/weight | (0,1),(0,5),(0,10),(4,11),(5,1),(5,5),(5,8),(5,9),(6,1),(6,9),(7,2),(7,9),(7,10),(10,2),(10,7) |
| C2 (EAP) | Per-head mean-ablation patching, top 15 | Node/activation | (3,0),(4,6),(5,5),(5,8),(5,10),(6,9),(7,9),(8,3),(8,6),(8,10),(9,6),(10,0),(10,7),(10,10),(11,10) |
| C3 (canonical) | Manual mechanistic analysis (Wang et al. 2023) | Manual | (0,1),(2,2),(3,0),(4,11),(5,5),(6,9),(7,3),(7,9),(8,6),(8,10),(9,6),(9,9),(10,0),(10,7),(11,10) |
| C4 (random) | Uniform random, seed 42 | Null | (0,3),(1,0),(1,10),(3,8),(3,9),(3,10),(4,8),(6,5),(6,10),(7,10),(8,1),(8,8),(8,11),(9,4),(10,8) |
| IC-15 (greedy) | First 15 heads from greedy forward IC selection | Sufficiency | (0,1),(0,3),(0,5),(0,9),(1,3),(4,0),(4,11),(5,6),(8,4),(8,9),(9,4),(10,3),(10,7),(11,7),(11,8) |
| C5 (Walsh) | LASSO-Walsh on 144-head game, top 15 by \|order-1\| | Spectral/node | (0,3),(0,9),(3,0),(4,0),(5,5),(6,9),(7,9),(8,6),(8,10),(10,1),(10,7),(10,10),(11,1),(11,2),(11,10) |


## Pairwise head overlaps (frozen, computed from head lists)

```
       C1    C2    C3    C4   IC15    C5
C1     --     5     6     1     4     4
C2      5    --    10     0     1     9
C3      6    10    --     0     3     8
C4      1     0     0    --     2     1
IC15    4     1     3     2    --     4
C5      4     9     8     1     4    --
```

Three structural features emerge from the overlap matrix before any
results are viewed:

1. **Convergent core**: C2, C3, C5 form a cluster (8-10 mutual overlap).
   All three share 8 heads: (3,0), (5,5), (6,9), (7,9), (8,6), (8,10),
   (10,7), (11,10). These are the heads that three independent
   methods --- mechanistic tracing, per-head causal importance, and
   Shapley-like spectral decomposition --- converge on.

2. **IC-15 is the structural outlier**: 1-4 heads overlap with every
   other circuit. Its 1/15 overlap with C2 is the lowest among all
   functional circuit pairs.

3. **C4 is isolated**: 0-2 heads overlap with functional circuits,
   as expected for a random control.

### Canonical IOI role coverage

Wang et al. identify five functional groups. Coverage per circuit:

| Circuit | Name movers (of 5) | S-inhibition (of 4) | Covered heads |
|---------|-------------------|---------------------|---------------|
| C2      | 4/5 | 3/4 | (9,6),(10,0),(10,7),(11,10) + (7,9),(8,6),(8,10) |
| C3      | 5/5 | 4/4 | Complete by definition |
| C5      | 2/5 | 3/4 | (10,7),(11,10) + (7,9),(8,6),(8,10) |
| C1      | 1/5 | 1/4 | (10,7) + (7,9) |
| IC-15   | 1/5 | 0/4 | (10,7) only |
| C4      | 0/5 | 0/4 | None |

C2 captures nearly the complete canonical mechanism. C1 and IC-15 miss
most of it. This asymmetry drives the predictions below.


## Prediction 1: Additivity vs epistasis ranking

### Theoretical reasoning

A circuit's Walsh energy spectrum reflects the interaction structure of
its coalition game v(S). The spectrum decomposes into energy at each
interaction order k, where order 0 is the constant (mean game value),
order 1 is individual/additive contributions, and order 2+ is epistatic
coordination.

The key insight: a method's inductive bias determines what kind of
circuit it finds, which in turn determines the circuit's interaction
structure. Methods that score heads individually select for independent
contributors (additive circuits). Methods that score coordination or
optimize sufficiency may select for interacting heads (epistatic
circuits).

### Predicted ranking (most additive to most epistatic)

Among the non-constant energy (orders 1+), I predict the following
ranking of order-1 energy fraction:

**C5 > C2 > C3 > C1 > IC-15**

(C4 excluded from this ranking; see Prediction 5.)

| Rank | Circuit | Predicted order-1 fraction (of orders 1+) | Reasoning |
|------|---------|------------------------------------------|-----------|
| 1 | C5 (Walsh) | > 85% | Selected by order-1 Walsh coefficient magnitude from the 144-head game. While the 15-head subgame is a different cooperative game than the 144-head source game, heads with large individual effects in the full game should retain large individual effects in the restricted game. Selection bias directly favors additivity. |
| 2 | C2 (EAP) | > 75% | Node-scored activation method: each head selected for its individual causal impact under mean ablation. Captures 4/5 name movers and 3/4 S-inhibition heads, giving near-complete coverage of the modular IOI mechanism. Modular mechanism coverage implies additive decomposition. |
| 3 | C3 (canonical) | > 70% | The IOI mechanism has clear modular structure: name movers provide large independent effects, S-inhibition facilitates via pairwise coordination with specific name movers. Prior zero-ablation data showed 2.8% order-3+; mean ablation should be comparable or lower. The pairwise S-inhibition/name-mover interactions should show up at order 2, not higher. |
| 4 | C1 (weight-IOI) | 50-70% | Mixed weight metrics capture structural capacity rather than task-specific function. Only 1/5 name movers and 1/4 S-inhibition heads means C1 includes many heads that are structurally capable but not specifically IOI-functional. These non-IOI heads may interact with each other through general-purpose attention patterns (induction, previous-token) that produce pairwise/higher-order dependencies rather than clean additive effects. C1 contains all three known induction heads (L5H1, L7H2, L7H10), which form compositional chains --- an inherently pairwise mechanism. |
| 5 | IC-15 (greedy) | 40-65% | The most uncertain prediction. IC-15 includes only 1/5 name movers and 0/4 S-inhibition heads, yet was optimized for sufficiency. Two scenarios: (a) IC-15 found a different, equally modular pathway (additive, order-1 > 60%), or (b) IC-15 found heads that must cooperate in unconventional ways to compensate for missing canonical machinery (epistatic, order-1 < 50%). I lean toward (b) because greedy forward selection conditions each addition on the existing set, which can capture complementary (interacting) heads. The sequential conditioning is a form of joint optimization that node-scored methods lack. |

### Order-2 and order-3+ predictions

| Circuit | Predicted order-2 fraction | Predicted order-3+ fraction | Reasoning |
|---------|---------------------------|----------------------------|-----------|
| C5 | 10-15% | < 5% | Residual from 144-head game's 19% order-2 energy, attenuated by selection for order-1 |
| C2 | 15-25% | < 5% | S-inhibition/name-mover pairwise coordination captured at order 2; modular structure limits higher orders |
| C3 | 15-25% | < 5% | Same reasoning as C2; complete mechanism means all coordination is captured at order 2 or below |
| C1 | 20-35% | 5-15% | Induction-head composition chains generate pairwise interactions; non-IOI heads may produce some higher-order structure from general-purpose computation |
| IC-15 | 25-40% | 10-20% | If compensatory pathway requires coordinated multi-head computation, higher-order interactions are expected |

### Falsification criteria

- **C5 NOT the most additive**: would mean order-1 coefficients in the
  144-head game do not predict order-1 dominance in the 15-head subgame.
  Falsified if C5 order-1 fraction < C2 or C3.

- **IC-15 NOT the most epistatic** (among functional circuits): would mean
  greedy sufficiency optimization does not capture interaction structure,
  and the unconventional head composition is simply noise. Falsified if
  IC-15 order-1 fraction > C3.

- **C1 order-3+ < 2%**: would mean weight-space scoring, despite
  capturing composition chains, does not translate to higher-order
  epistasis in the coalition game. Would challenge the "edge-scored
  methods find epistatic circuits" thesis from v10.

- **C2 order-3+ > C3 order-3+**: would be surprising because C2 is a
  strict subset of C3's mechanism (4/5 name movers, 3/4 S-inhibition).
  A less complete mechanism producing more epistasis would suggest
  missing heads create interaction artifacts rather than reducing them.


## Prediction 2: Faithfulness

Faithfulness delta = v(all 15 active) - v(all 15 ablated), where v is
mean logit_diff across prompts.

### Predicted ranking (most to least faithful)

**C3 > C2 > C5 > IC-15 > C1 >> C4**

| Circuit | Predicted delta | Reasoning |
|---------|----------------|-----------|
| C3 | > 0.5 (high) | Complete canonical mechanism; these 15 heads were hand-verified to implement IOI. Under mean ablation, the all-ablated baseline retains some MLP-mediated signal, so delta is slightly below full-model logit_diff (~0.61). |
| C2 | > 0.4 (high) | Near-complete mechanism coverage (4/5 name movers, 3/4 S-inhibition). Missing L9H9 and L7H3 reduces delta slightly below C3, but the four included name movers carry most of the logit_diff signal. |
| C5 | 0.3-0.5 (moderate-high) | 8/15 overlap with C3 including key S-inhibition heads (7,9), (8,6), (8,10) and name movers (10,7), (11,10). Missing 3/5 name movers means less total logit_diff output, but the two included name movers plus 3 S-inhibition heads form a functional sub-circuit. |
| IC-15 | 0.1-0.4 (uncertain) | IC was optimized for per-prompt RSS reduction, so the first 15 heads should capture meaningful signal. But IC-15 contains only 1 name mover (10,7) and 0 S-inhibition heads, meaning it cannot implement the canonical IOI mechanism. Faithfulness depends on whether backup/compensatory pathways exist. The wide confidence interval reflects genuine uncertainty. |
| C1 | 0.05-0.25 (low-moderate) | Only 1 name mover (10,7) and 1 S-inhibition head (7,9). Weight-scored heads are structurally capable but not task-specific. Several heads (induction heads L5H1, L7H2, L7H10) serve general-purpose roles that may not produce IOI-specific logit_diff. |
| C4 | < 0.05 (near zero) | Random heads have no systematic relationship to IOI. Any non-zero delta is noise from accidentally including heads that weakly contribute. |

### Falsification criteria

- **IC-15 delta > C3 delta**: would mean a radically different 15-head
  set outperforms the canonical circuit. Possible only if GPT-2 has
  substantial circuit redundancy and IC-15 found a superior pathway.

- **C1 delta > C2 delta**: would mean weight-scored heads (despite
  missing most canonical mechanism) are more faithful than
  activation-scored heads. Hard to reconcile with C2's near-complete
  mechanism coverage.

- **C4 delta > 0.1**: would indicate mean ablation creates systematic
  artifacts that make random circuits appear functional, calling the
  entire evaluation into question.


## Prediction 3: EpistasisBench performance

### 3a: Budget sensitivity (LASSO-Walsh R² on held-out coalitions)

At 1% budget (328 training samples out of 32,768):

| Circuit | Predicted R² | Reasoning |
|---------|-------------|-----------|
| C5 | > 0.90 | Nearly pure order-1 game. 16 free parameters (15 order-1 + 1 order-0) with 328 samples is 20x oversampled. LASSO-Walsh should nail this. |
| C2, C3 | > 0.85 | Additive-dominated with moderate order-2. ~121 parameters (16 order-1/0 + 105 order-2) with 328 samples is 2.7x oversampled. LASSO sparsity handles this if most order-2 terms are zero. |
| C1 | 0.70-0.85 | More order-2 energy and some order-3+. LASSO at order 2 misses the higher-order terms. 328 samples may not resolve the relevant pairwise interactions. |
| IC-15 | 0.60-0.80 | If IC-15 is epistatic with substantial order-3+ energy, 328 samples cannot capture C(15,3)=455 additional order-3 terms. LASSO truncated at order 2 has model misspecification. |
| C4 | 0.00-0.30 | Near-constant game (most energy at order 0). R² measures variance explained beyond the intercept. With almost no variance to explain, R² is dominated by noise, making it either very high (fitting the intercept well) or very low (depending on normalization). Unpredictable. |

At 5% budget (1,638 samples): all functional circuits achieve R² > 0.85.
At 10% budget: all achieve R² > 0.90. Budget sensitivity effectively
disappears above 10% for circuits with primarily order-1 and order-2
structure.

**Key prediction**: the budget-sensitivity GAP between circuits narrows
as budget increases. At 1%, the R² spread between C5 (most additive)
and IC-15 (most epistatic) should be > 0.15. At 10%, the gap should
shrink to < 0.05. This tests whether epistatic circuits are intrinsically
harder to characterize or just data-hungrier.

### 3b: Method comparison (LASSO-Walsh vs iRF vs shapiq)

**LASSO-Walsh advantages:**
- Directly models the Walsh basis, which is the ground truth decomposition
- Sparsity (L1 penalty) exploits the fact that most circuits have sparse
  spectra (few large coefficients, many near-zero)
- Should dominate on additive circuits (C5, C2, C3) at all budgets

**iRF advantages:**
- Tree-based method captures interactions via split structure without
  explicitly parameterizing them
- Iterative feature weighting can focus on interacting players
- May detect higher-order interactions (order 3+) that LASSO-Walsh
  at order 2 misses by construction

**shapiq advantages:**
- Axiom-grounded (Shapley values satisfy efficiency, symmetry, etc.)
- Model-agnostic; does not assume sparsity or basis structure

Predicted ranking by overall R² at low budget:

**LASSO-Walsh > iRF > shapiq** on additive circuits (C5, C2, C3)

**LASSO-Walsh >= iRF > shapiq** on moderate-epistasis circuits (C1)

**iRF >= LASSO-Walsh > shapiq** on high-epistasis circuits (IC-15),
but only if IC-15 has substantial order-3+ energy that LASSO-Walsh
at order 2 cannot capture.

### 3c: Pairwise interaction detection (AUROC and Spearman rho)

Pairwise interaction AUROC measures whether the method correctly
identifies which pairs of heads interact (true positive = pair with
non-zero order-2 Walsh coefficient).

- **Additive circuits (C5, C2, C3)**: few true interactions, so AUROC
  is testing the false positive rate. All methods should achieve AUROC
  > 0.8 because correctly identifying "no interaction" is easy when
  few exist.

- **Epistatic circuits (C1, IC-15)**: more true interactions, so AUROC
  is a genuine discrimination test. LASSO-Walsh should excel because
  it directly estimates order-2 coefficients. iRF's pairwise scores
  (from split co-occurrence) may be noisier. Predicted: LASSO-Walsh
  AUROC > iRF AUROC on epistatic circuits.

- **Spearman rho** (rank correlation of pairwise interaction strengths):
  meaningful only for circuits with non-degenerate interaction
  distributions. On additive circuits, all interactions are near zero
  and ranking is essentially random --- Spearman rho will be low
  regardless of method quality. On epistatic circuits, Spearman rho
  is a meaningful metric. Predicted: Spearman rho for all methods is
  higher on IC-15 than on C5.

### 3d: Spectrum NMSE

Spectrum NMSE measures how well the estimated Walsh coefficients match
the ground truth across all orders.

- **Additive circuits**: NMSE should be very low (< 0.1) at 5% budget
  because the spectrum is concentrated in 15-16 coefficients.
- **Epistatic circuits**: NMSE will be higher because more coefficients
  need estimation. At 1% budget, NMSE > 0.3 for IC-15 if it has
  substantial order-3+ energy.

### Falsification criteria

- **iRF outperforms LASSO-Walsh on additive circuits (C5, C2, C3)**:
  would mean tree-based methods are competitive with basis-aware
  methods even when the basis is known and sparse. Would challenge
  the assumption that exploiting basis structure helps.

- **LASSO-Walsh R² > 0.90 on IC-15 at 1% budget**: would mean IC-15
  is actually well-approximated by an order-2 model, contradicting
  the prediction that it is the most epistatic.

- **C4 achieves R² > 0.5 at any budget**: would indicate that even a
  random circuit has learnable structure under mean ablation. Could
  suggest systematic ablation artifacts.


## Prediction 4: Method agreement (spectral similarity)

### Predicted spectral clusters

Circuits that share many heads should have similar Walsh energy spectra.
The overlap matrix predicts three groups:

1. **Convergent-core cluster** {C2, C3, C5}: 8-10 mutual overlap.
   These three circuits should have similar energy profiles --- all
   additive-dominated with moderate order-2 energy from the
   S-inhibition/name-mover coordination captured in their shared 8
   heads.

   Predicted spectral distance: ||spectrum(Ci) - spectrum(Cj)||_2 < 0.10
   for all pairs within {C2, C3, C5}.

2. **Partial-overlap outlier** {C1}: 4-6 overlap with the core cluster.
   C1's spectrum should deviate from the core cluster due to its
   unique induction heads and non-IOI-functional heads, but not as
   dramatically as IC-15.

   Predicted: ||spectrum(C1) - spectrum(C3)||_2 between 0.10 and 0.25.

3. **Structural outlier** {IC-15}: 1-4 overlap with everything. If
   IC-15 is epistatic as predicted, its energy profile should be
   qualitatively different from the core cluster.

   Predicted: ||spectrum(IC15) - spectrum(C3)||_2 > 0.20.

### Which circuit pairs will be most similar?

**C2 and C3**: 10/15 overlap, both dominated by the canonical IOI
mechanism. Should have the most similar spectra of any pair.

**C2 and C5**: 9/15 overlap. Similar spectra, possibly with C5 showing
slightly higher order-1 fraction due to its selection bias.

### Which circuit pairs will be most different (among functional circuits)?

**IC-15 and C2**: 1/15 overlap. Radically different head compositions
should produce the most different spectra.

**C1 and IC-15**: 4/15 overlap but different functional compositions.
C1's induction-head content vs IC-15's backup-pathway content should
create distinct spectral signatures.

### Falsification criteria

- **C2 and C3 have spectral distance > 0.15**: would mean 10/15 head
  overlap does not guarantee spectral similarity. Would suggest that
  the 5 differing heads have outsized influence on the game structure.

- **IC-15 and C3 have spectral distance < 0.10**: would mean IC-15,
  despite its radically different composition, produces a functionally
  similar game. Would suggest that the IOI game is surprisingly robust
  to which specific 15 heads are selected (strong circuit redundancy).


## Prediction 5: Null circuit (C4 sanity checks)

C4 (15 random heads, seed 42) is the negative control. Its behavior
establishes the noise floor for all metrics.

### Predicted behavior

| Property | Prediction | Reasoning |
|----------|-----------|-----------|
| Faithfulness delta | < 0.05 | Random heads carry no systematic IOI signal. Any non-zero delta is coincidental. |
| Order-0 energy fraction | > 90% | The game v(S) is nearly constant: ablating non-functional heads does not change logit_diff. Nearly all variance is in the intercept. |
| Order-1 fraction (of non-constant) | No directional prediction | Among the small residual signal, the partition across orders is noise-driven. Could be any value. |
| LASSO-Walsh R² at 1% | < 0.30 | Almost no variance to explain. R² is unstable when the denominator (total variance) is near zero. |
| Pairwise interaction AUROC | ~0.50 (chance) | No real interactions exist, so any detected "interactions" are false positives. AUROC should be at chance. |
| Spectral NMSE | High (> 0.5) | Small true coefficients are hard to estimate precisely. NMSE denominators are near zero, inflating the ratio. |

### Sanity checks (these MUST hold)

1. C4 delta < min(C1, C2, C3, C5, IC-15) delta.
   If violated: a random circuit is more faithful than a principled one.
   Diagnosis: mean ablation artifact or implementation bug.

2. C4 total Walsh energy (orders 1+) < min over functional circuits.
   If violated: random heads produce as much game structure as selected
   heads. Diagnosis: mean ablation creates systematic artifacts.

3. C4 does not cluster with any functional circuit in spectrum space.
   If violated: the spectral lens cannot distinguish signal from noise.

### What C4 canNOT tell us

C4 does not establish a null for within-circuit head overlap or spectral
cohesion. A random circuit might accidentally include heads that
interact in the full model --- these interactions would be real (present
in GPT-2's computation) but not meaningfully related to IOI. The
epistasis of random heads is a property of the model's residual
computation, not the task.


## Prediction 6: Walsh discovery bias (C5 circularity analysis)

### The circularity concern

C5 was selected by ranking 144 heads by |order-1 Walsh coefficient| in
a LASSO regression of the 144-head game. The EpistasisBench evaluation
computes the Walsh spectrum of C5's 15-head subgame. If C5's order-1
dominance in evaluation is driven by its selection criterion rather than
a genuine property of its heads, the comparison is circular.

### Why it is partially (but not fully) circular

The 144-head game and the 15-head subgame are different cooperative games:

- **144-head game**: v_{144}(S) for S in {0,1}^{144}. Each head's
  order-1 coefficient reflects its marginal effect averaged over
  2^{143} contexts (all possible subsets of the other 143 heads).

- **15-head subgame**: v_{15}(T) for T in {0,1}^{15}. Each head's
  order-1 coefficient reflects its marginal effect averaged over 2^{14}
  contexts, but all 129 non-circuit heads are always ablated.

The baselines differ. In the 144-head game, a head's marginal effect
is measured against a background where roughly half the other heads are
active. In the 15-head subgame, a head's marginal effect is measured
against a background where at most 14 other heads are active and the
remaining 129 are always ablated.

If IOI computation is modular (each head contributes independently of
context), the order-1 coefficients should be consistent across the two
games, and the circularity is near-total. If head contributions are
context-dependent (a head matters only when certain other heads are also
active), the 15-head subgame's order-1 coefficients may diverge from
the 144-head game's, and the circularity is partial.

### Predicted spectral signature for C5

- **Order-1 fraction**: > 85% (highest among all circuits). This is
  partly tautological and partly a genuine prediction about context
  independence. The tautological component: C5 was selected for high
  order-1 in a related game. The genuine component: this order-1
  dominance transfers to a game with a different background (all
  non-circuit heads ablated).

- **Order-2 fraction**: 10-15%. The 144-head game had 19% order-2
  energy. Some pairwise interactions among C5 heads exist (e.g.,
  the S-inhibition heads (7,9), (8,6), (8,10) coordinate with name
  movers (10,7), (11,10)). But C5's selection for order-1 means it
  avoided heads whose primary contribution is pairwise.

- **Order-3+ fraction**: < 3%. C5 selected heads with strong
  independent effects, which are precisely the heads least likely to
  participate in higher-order coordination.

### Debiasing test

The circularity can be assessed by comparing C5's spectrum to C2's.
C2 was selected by a completely independent criterion (per-head
mean-ablation patching) and shares 9/15 heads with C5. If C5 and C2
have near-identical spectra, the Walsh-discovery bias contributes
little beyond what activation-based selection already achieves. If
C5 is substantially more additive than C2 (order-1 fraction gap > 10
percentage points), the Walsh-selection bias has a measurable effect
on the circuit's game structure.

**Prediction**: C5 is more additive than C2 by 5-15 percentage points
in order-1 fraction. The bias exists but is moderate, because 9/15
shared heads anchor both circuits to similar game structure.

### Falsification criteria

- **C5 order-1 fraction < C2**: the Walsh selection bias has no effect,
  or the context shift (144 -> 15 heads) reverses the ordering. Would
  mean that order-1 magnitude in the full game does not predict order-1
  dominance in the subgame.

- **C5 and C2 spectra are indistinguishable** (spectral distance < 0.03):
  the 6 differing heads between C5 and C2 do not matter, and
  Walsh discovery adds no information beyond activation patching for
  producing additive circuits. The circularity concern is moot because
  any activation-based method produces the same result.

- **C5 order-2 fraction > 25%**: the 144-head game's pairwise
  interactions concentrate in C5's 15 heads rather than being
  distributed across all 144. Would mean Walsh discovery inadvertently
  selects for interacting heads (high order-1 heads happen to also
  have high order-2 interactions).


## Summary prediction table

| Circuit | Delta | O1 frac | O2 frac | O3+ frac | R²@1% | Cluster |
|---------|-------|---------|---------|----------|-------|---------|
| C5 (Walsh) | 0.3-0.5 | > 85% | 10-15% | < 3% | > 0.90 | Core |
| C2 (EAP) | > 0.4 | > 75% | 15-25% | < 5% | > 0.85 | Core |
| C3 (canonical) | > 0.5 | > 70% | 15-25% | < 5% | > 0.85 | Core |
| C1 (weight) | 0.05-0.25 | 50-70% | 20-35% | 5-15% | 0.70-0.85 | Partial outlier |
| IC-15 (greedy) | 0.1-0.4 | 40-65% | 25-40% | 10-20% | 0.60-0.80 | Structural outlier |
| C4 (random) | < 0.05 | N/A | N/A | N/A | < 0.30 | Null |

**O1 frac ranking**: C5 > C2 > C3 > C1 > IC-15

**Delta ranking**: C3 > C2 > C5 > IC-15 > C1 > C4

**R²@1% ranking**: C5 > C2 ≈ C3 > C1 > IC-15 > C4


## Surprises that would challenge the theoretical framework

### S1: IC-15 is MORE additive than C3

Would mean greedy sufficiency optimization, despite conditioning each
addition on the existing set, produces a more modular circuit than
hand-traced mechanistic analysis. Would suggest that the canonical
IOI circuit's functional groups (S-inhibition, name movers) interact
more than the mechanistic narrative implies, and that IC-15's
"unconventional" heads are individually cleaner contributors.

### S2: C1 has HIGHER faithfulness than C2

Would mean weight-space metrics (without seeing activations or task
data) identify heads that are more causally important for IOI than
per-head activation patching. Hard to reconcile with C1's 1/5 name
mover coverage. Would require the 14 non-name-mover C1 heads to
collectively produce more logit_diff than C2's 4 name movers --- which
would overturn the standard view that name movers are the primary
carriers of IOI signal.

### S3: C4 (random) shows structured epistasis

If C4 has > 5% order-3+ energy among its non-constant signal, this
would mean that even random groups of GPT-2 attention heads have
coordinated interaction structure. This could reflect either (a)
mean-ablation artifacts that create spurious interactions, or (b)
genuine higher-order structure in GPT-2's general computation that is
unrelated to IOI. Distinguishing (a) from (b) requires checking
whether the interactions are prompt-template-dependent (artifact) or
consistent across templates (genuine).

### S4: C5 is NOT the most additive

Would break the central prediction of this pre-registration. If C5's
order-1 fraction is lower than C2 or C3, the Walsh discovery
criterion (select by |order-1|) does not produce the most additive
subgame. Would mean that context dependence (the 144-to-15 game
shift) overwhelms the selection bias. This would be the most
informative falsification because it separates "a head has a large
individual effect in a large game" from "a head has a large individual
effect in a small game" --- a distinction that matters for all
sparse-recovery-based circuit discovery methods.

### S5: Convergent core (C2, C3, C5) shows spectral DIVERGENCE

If two circuits with 10/15 overlap (C2 and C3) have spectral
distance > 0.15, it would mean that the 5 differing heads dominate
the game's interaction structure. Would imply that Walsh spectra are
fragile --- small changes in circuit membership produce large spectral
shifts. This would undermine the use of Walsh spectra as a stable
circuit fingerprint.

### S6: IC-15 is the MOST faithful

If IC-15 delta > C3 delta despite 3/15 overlap, it would demonstrate
that GPT-2's IOI computation has deep redundancy --- the model
implements IOI through multiple independent pathways, and IC-15 found
an alternative pathway that is more efficient (higher delta per head)
than the canonical one. This would be a major finding about the
structure of learned computation.
