> **Provenance.** Reformatted from `preregistrations/prereg_v12b_selective_pressures.md`.
> Original frozen at commit `81c628e`. View: `git show 81c628e:preregistrations/prereg_v12b_selective_pressures.md`

# Pre-Registration v12b: Selective Pressures on Walsh Spectra

**Status**: FROZEN (blind -- no Walsh spectra, faithfulness deltas, or EpistasisBench outputs have been viewed)

## Overview

This pre-registration predicts how different circuit discovery methods shape the
Walsh-Hadamard interaction spectrum of the resulting circuits. Six circuits of
15 heads each are compared on the IOI task in GPT-2 Small (144 heads total).
The cooperative game v(S) = mean logit_diff when heads in S are active and the
remaining circuit heads are mean-ablated (non-circuit heads at natural values).

The core thesis: each discovery method imposes a **selective pressure** -- an
inductive bias that determines which heads enter the circuit. That bias should
leave a detectable fingerprint in the Walsh spectrum. Methods that optimize for
individual head importance should yield additive spectra (order-1 dominated).
Methods that trace mechanistic pathways should yield epistatic spectra
(order-2+ elevated).

---

## Section 1: Role Coverage

Canonical IOI role assignments (Wang et al. 2023):
- **Previous-token (PT)**: L0H1, L1H3
- **Duplicate-token (DT)**: L0H1, L3H0, L4H11
- **S-inhibition (SI)**: L7H3, L7H9, L8H6, L8H10
- **Name-mover (NM)**: L9H6, L9H9, L10H0, L10H7, L11H10
- **Backup name-mover (BNM)**: L10H2, L10H6, L10H10, L11H2

Note: L0H1 appears in both PT and DT roles. A head can count in multiple roles.
"Unique canonical" counts distinct heads that fill at least one role.

| Circuit | PT | DT | SI | NM | BNM | Unique Canonical | Non-Canonical |
|---------|----|----|----|----|-----|-----------------|---------------|
| C1 (weight-IOI)       | 1 | 2 | 1 | 1 | 1 | 5  | 10 |
| C2 (EAP)              | 2 | 1 | 3 | 2 | 0 | 7  |  8 |
| C3 (canonical IOI)    | 1 | 3 | 4 | 5 | 0 | 12 |  3 |
| C4 (random null)      | 0 | 0 | 0 | 0 | 0 |  0 | 15 |
| IC-15 (greedy suff.)  | 2 | 2 | 0 | 1 | 0 |  4 | 11 |
| C5 (Walsh discovery)  | 0 | 1 | 3 | 2 | 2 |  8 |  7 |

### Role coverage detail (which heads fill each role)

**C1 (weight-IOI)**: (0,1), (0,5), (0,10), (4,11), (5,1), (5,5), (5,8), (5,9), (6,1), (6,9), (7,2), (7,9), (7,10), (10,2), (10,7)
- PT: (0,1)
- DT: (0,1), (4,11)
- SI: (7,9)
- NM: (10,7)
- BNM: (10,2)
- Non-canonical: (0,5), (0,10), (5,1), (5,5), (5,8), (5,9), (6,1), (6,9), (7,2), (7,10)

**C2 (EAP)**: (0,1), (0,8), (0,9), (0,11), (1,3), (4,7), (5,10), (6,9), (7,9), (8,3), (8,6), (8,10), (10,7), (11,0), (11,10)
- PT: (0,1), (1,3)
- DT: (0,1)
- SI: (7,9), (8,6), (8,10)
- NM: (10,7), (11,10)
- BNM: none
- Non-canonical: (0,8), (0,9), (0,11), (4,7), (5,10), (6,9), (8,3), (11,0)

**C3 (canonical IOI)**: (0,1), (2,2), (3,0), (4,11), (5,5), (6,9), (7,3), (7,9), (8,6), (8,10), (9,6), (9,9), (10,0), (10,7), (11,10)
- PT: (0,1)
- DT: (0,1), (3,0), (4,11)
- SI: (7,3), (7,9), (8,6), (8,10)
- NM: (9,6), (9,9), (10,0), (10,7), (11,10)
- BNM: none
- Non-canonical: (2,2), (5,5), (6,9) -- likely auxiliary/induction roles identified by Wang et al. but not assigned to the five named groups

**C4 (random null)**: (0,3), (1,0), (1,10), (3,8), (3,9), (3,10), (4,8), (6,5), (6,10), (7,10), (8,1), (8,8), (8,11), (9,4), (10,8)
- No canonical roles.

**IC-15 (greedy sufficiency)**: (10,7), (4,0), (0,1), (4,11), (9,4), (5,6), (8,9), (8,4), (0,5), (11,7), (0,9), (11,8), (10,3), (1,3), (0,3)
- PT: (0,1), (1,3)
- DT: (0,1), (4,11)
- SI: none
- NM: (10,7)
- BNM: none
- Non-canonical: (4,0), (9,4), (5,6), (8,9), (8,4), (0,5), (11,7), (0,9), (11,8), (10,3), (0,3)

**C5 (Walsh discovery)**: (5,5), (10,7), (11,1), (8,6), (8,10), (0,9), (7,9), (0,3), (6,9), (10,1), (11,2), (10,10), (3,0), (11,10), (4,0)
- PT: none
- DT: (3,0)
- SI: (7,9), (8,6), (8,10)
- NM: (10,7), (11,10)
- BNM: (11,2), (10,10)
- Non-canonical: (5,5), (11,1), (0,9), (0,3), (6,9), (10,1), (4,0)

### Key observations from role coverage

1. **C3 covers 12/15 canonical roles** -- by far the most. This is expected: it
   was identified by tracing the canonical mechanism.
2. **C5 has the second-highest canonical coverage (8)** despite being selected
   purely by Walsh order-1 coefficients. This suggests that individually
   impactful heads coincide with canonical roles. C5 is the only circuit to
   include backup name movers.
3. **IC-15 has zero S-inhibition heads**. This is striking: greedy sufficiency
   apparently builds the IOI signal through an alternative route that bypasses
   S-inhibition.
4. **C1 spreads heavily into layers 5-6** (six heads in L5-L6). Weight-space
   nuclear norm and composition strength apparently favor mid-network heads
   whose weight matrices compose well, whether or not they participate in the
   canonical IOI pipeline.
5. **C4 hits zero canonical roles**, confirming its status as a null control.

---

## Section 2: Pairwise Overlap Matrix

Number of shared heads between each pair of circuits (out of 15 each):

|        | C1 | C2 | C3 | C4 | IC-15 | C5 |
|--------|---:|---:|---:|---:|------:|---:|
| **C1** | 15 |  4 |  6 |  1 |   4   |  4 |
| **C2** |  4 | 15 |  7 |  0 |   4   |  7 |
| **C3** |  6 |  7 | 15 |  0 |   3   |  8 |
| **C4** |  1 |  0 |  0 | 15 |   2   |  1 |
| **IC-15** | 4 | 4 |  3 |  2 |  15   |  4 |
| **C5** |  4 |  7 |  8 |  1 |   4   | 15 |

### Shared heads detail

- **C1 & C2** (4): (0,1), (6,9), (7,9), (10,7)
- **C1 & C3** (6): (0,1), (4,11), (5,5), (6,9), (7,9), (10,7)
- **C1 & C4** (1): (7,10)
- **C1 & IC-15** (4): (0,1), (0,5), (4,11), (10,7)
- **C1 & C5** (4): (5,5), (6,9), (7,9), (10,7)
- **C2 & C3** (7): (0,1), (6,9), (7,9), (8,6), (8,10), (10,7), (11,10)
- **C2 & C4** (0): none
- **C2 & IC-15** (4): (0,1), (0,9), (1,3), (10,7)
- **C2 & C5** (7): (0,9), (6,9), (7,9), (8,6), (8,10), (10,7), (11,10)
- **C3 & C4** (0): none
- **C3 & IC-15** (3): (0,1), (4,11), (10,7)
- **C3 & C5** (8): (3,0), (5,5), (6,9), (7,9), (8,6), (8,10), (10,7), (11,10)
- **C4 & IC-15** (2): (0,3), (9,4)
- **C4 & C5** (1): (0,3)
- **IC-15 & C5** (4): (0,3), (0,9), (4,0), (10,7)

### Key observations from overlap

1. **C3 and C5 share the most heads (8/15)** -- the highest overlap among any
   pair. The shared heads are concentrated in S-inhibition and name-mover roles,
   suggesting these groups produce the largest individual Walsh effects.
2. **C2 and C5 also share 7 heads**, reflecting convergence between EAP
   (activation-patching) and Walsh (cooperative-game) notions of individual
   importance.
3. **C4 is nearly disjoint from all functional circuits** (0-2 overlap),
   confirming random selection does not accidentally reconstruct any method's
   circuit.
4. **IC-15 has surprisingly low overlap with C3 (only 3)**, the lowest among
   functional-circuit pairs. Greedy sufficiency builds a different circuit than
   mechanistic tracing.
5. **(10,7) appears in 5 of 6 circuits** (all except C4). Head L10H7 is a name
   mover that apparently scores highly under every non-random selection
   criterion.

---

## Section 3: Predictions

### 3a: Energy Spectrum Predictions (Mean Ablation)

All fractions are percentages of total Walsh energy (sum of all squared Walsh
coefficients). Intervals are [low, high].

**C1 (weight-IOI)**:
| Order   | Predicted Range |
|---------|----------------|
| 0       | [88, 96]%      |
| 1       | [1.5, 6]%      |
| 2       | [0.5, 3.5]%    |
| 3+      | [0.3, 2]%      |

REASONING: C1 contains only 5 canonical heads and concentrates in layers 5-6.
Ablating these heads has a moderate effect on IOI logit_diff (delta ~1-2), so
the game variance is moderate and order-0 dominates. Weight-space composition
strength selects heads that pair well, potentially elevating order-2 relative to
order-1. But the overall non-constant energy is small because C1 is not the
primary IOI mechanism.

**C2 (EAP)**:
| Order   | Predicted Range |
|---------|----------------|
| 0       | [82, 92]%      |
| 1       | [4, 11]%       |
| 2       | [1, 4]%        |
| 3+      | [0.5, 3]%      |

REASONING: EAP selects heads by individual mean-ablation impact (|delta
logit_diff|). This is approximately an order-1 criterion, so C2's non-constant
energy should be order-1 dominated. However, 7 canonical heads including 3
S-inhibition and 2 name movers preserve partial pipeline structure, creating
genuine pairwise interactions. Delta should be high (~2-3), giving moderate game
variance.

**C3 (canonical IOI)**:
| Order   | Predicted Range |
|---------|----------------|
| 0       | [78, 90]%      |
| 1       | [3, 8]%        |
| 2       | [2, 7]%        |
| 3+      | [1, 5]%        |

REASONING: C3 is the pipeline circuit -- previous-token feeds duplicate-token
feeds S-inhibition feeds name-movers. This chain creates strong pairwise
interactions (order-2) and weaker higher-order synergies (order-3+). The five
name movers individually contribute to logit_diff (boosting order-1), but
their effectiveness depends on S-inhibition (boosting order-2). With 12
canonical heads, ablating C3 devastates IOI performance, creating large game
variance and the lowest order-0 fraction among all circuits.

**C4 (random null)**:
| Order   | Predicted Range |
|---------|----------------|
| 0       | [98, 99.9]%    |
| 1       | [0.05, 1.5]%   |
| 2       | [0.01, 0.4]%   |
| 3+      | [0.005, 0.3]%  |

REASONING: Random heads have negligible individual and collective impact on IOI
logit_diff. The game is nearly constant: v(S) ~ 3.4 for all S. Order-0
captures the squared mean, which dwarfs the tiny variance. Whatever
non-constant energy exists should be primarily order-1 (small individual
effects) with even smaller pairwise terms.

**IC-15 (greedy sufficiency)**:
| Order   | Predicted Range |
|---------|----------------|
| 0       | [86, 95]%      |
| 1       | [2, 7]%        |
| 2       | [1, 4]%        |
| 3+      | [0.3, 2]%      |

REASONING: Greedy forward selection picks heads sequentially by marginal
sufficiency gain. Early selections have large individual effects (high order-1),
but later selections are chosen to complement earlier ones, creating implicit
dependencies. IC-15 has zero S-inhibition heads, so it must implement IOI
through a non-canonical pathway, which could involve different interaction
patterns. Delta is moderate (~1-2) because sufficiency does not directly
optimize necessity.

**C5 (Walsh discovery)**:
| Order   | Predicted Range |
|---------|----------------|
| 0       | [80, 90]%      |
| 1       | [6, 14]%       |
| 2       | [1, 4]%        |
| 3+      | [0.3, 2]%      |

REASONING: C5 was selected by |order-1 Walsh coefficient| from the full
144-head game. This is the most directly additive selection criterion. The 15
heads should have the largest individual marginal effects on logit_diff,
concentrating non-constant energy at order-1. However, C5 contains 8 canonical
heads including 3 S-inhibition and 2 name movers, preserving partial pipeline
structure that generates some order-2 energy. The order-0 fraction is moderate
because these high-marginal-effect heads create large game variance when
toggled.

---

### 3b: Ordinal Rankings

**Additivity ranking** (order-1 fraction of non-constant energy, most to least additive):

PREDICTION: C5 > C2 > IC-15 > C4 > C1 > C3

REASONING: C5 directly optimizes for order-1 Walsh coefficients. C2 (EAP)
optimizes for individual ablation effect, which correlates with but is not
identical to Walsh order-1. IC-15's early greedy selections have large
individual effects. C4 (random) is placed mid-range because its tiny total
signal should be predominantly additive (random heads lack systematic
interactions), but the low signal-to-noise ratio introduces uncertainty. C1's
composition-focused selection elevates pairwise relative to individual effects.
C3's pipeline structure creates the most epistasis.

FALSIFIED IF: C3 ranks above C2 or C5 in additivity, or C4 ranks first (which
would suggest all circuits are equally additive and the ranking is noise).

---

**Faithfulness ranking** (delta = v(full) - v(empty), most to least faithful):

PREDICTION: C3 > C2 > C5 > IC-15 > C1 > C4

REASONING: C3 contains 12 canonical heads spanning all pipeline stages; its
ablation removes the primary IOI mechanism. C2 and C5 each contain 7-8
canonical heads selected for individual importance, but C2's activation-based
selection is more directly calibrated to IOI logit_diff than C5's
coefficient-based selection. IC-15 and C1 have fewer canonical heads (4-5).
IC-15 optimizes sufficiency not necessity, so its heads may be partially
redundant with the non-ablated 129 heads. C4 contributes negligibly to IOI.

FALSIFIED IF: C4's delta exceeds 1.0, or IC-15's delta exceeds C3's (which
would indicate greedy sufficiency incidentally selects the most necessary
heads).

---

**Total epistasis ranking** (order-2 + order-3+ as fraction of total energy, most to least):

PREDICTION: C3 > C1 > IC-15 > C2 > C5 > C4

REASONING: C3's four-stage pipeline generates strong pairwise and higher-order
interactions. C1's weight-composition selection picks heads that pair well in
weight space, potentially elevating order-2 energy even though the heads are not
canonical pipeline members. IC-15's sequential greedy selection creates
implicit dependencies. C2's EAP-based selection and C5's Walsh-based selection
both emphasize individual effects, suppressing the relative weight of
interaction terms. C4 has negligible total non-constant energy.

FALSIFIED IF: C5 ranks above C3 in total epistasis (would mean Walsh-additive
selection paradoxically creates more epistasis than mechanistic tracing), or C4
ranks above any functional circuit.

---

### 3c: Faithfulness Delta Predictions (Mean Ablation)

Delta = v(all 15 active) - v(all 15 ablated), in logit_diff units.
Full-model IOI logit_diff is approximately 3.0-4.0.

| Circuit | Predicted Delta |
|---------|----------------|
| C1 (weight-IOI)      | [0.8, 2.0]   |
| C2 (EAP)             | [2.0, 3.5]   |
| C3 (canonical IOI)   | [2.0, 3.5]   |
| C4 (random null)     | [-0.2, 0.5]  |
| IC-15 (greedy suff.) | [0.5, 2.0]   |
| C5 (Walsh discovery) | [1.5, 3.0]   |

REASONING: Delta measures necessity -- how much logit_diff drops when the
circuit is ablated while all other heads run normally. C3 and C2 should have the
highest deltas because they contain the most canonical IOI heads (12 and 7
unique). C5 contains 8 canonical heads but was selected for marginal effects in
the full 144-head game, which may not transfer perfectly to the 15-head
necessity measure. C1 and IC-15 have fewer canonical heads (5 and 4) and include
many non-canonical heads that may contribute little to IOI specifically. C4
contains zero canonical heads.

FALSIFIED IF: Any circuit's delta falls outside its predicted range (the
intervals are generous), or C4's delta exceeds C1's.

---

### 3d: Pairwise Spectral Similarity

**Most similar spectra** (excluding C4):

PREDICTION: C2 and C5

REASONING: C2 (EAP) and C5 (Walsh discovery) share 7 heads and have analogous
selection biases -- both optimize for individual head importance, one through
activation patching and one through Walsh order-1 coefficients. They should
produce similar ratios of additive to epistatic energy. Their shared heads
include S-inhibition (7,9), (8,6), (8,10) and name movers (10,7), (11,10),
ensuring similar pipeline-interaction contributions.

FALSIFIED IF: C2 and C5 differ by more than 15 percentage points on any
single order's energy fraction, or another pair (excluding C4) matches more
closely on all four order fractions simultaneously.

**Least similar spectra** (excluding C4):

PREDICTION: C3 and IC-15

REASONING: C3 and IC-15 share only 3 heads (the lowest overlap among
functional-circuit pairs) and have maximally different selection criteria --
mechanistic pathway tracing vs. greedy forward sufficiency. C3's full pipeline
creates high epistasis while IC-15's zero S-inhibition heads suggest a
fundamentally different internal structure. Their spectra should differ in both
the total non-constant energy and the additive-vs-epistatic balance.

FALSIFIED IF: C3 and IC-15 match within 3 percentage points on all four order
fractions, or any other functional-circuit pair has a larger maximum difference
across the four order fractions.

---

### 3e: Cluster Prediction

Predicted spectral clusters (by similarity of energy distribution across orders):

1. **Additive cluster**: {C5, C2} -- Both selected for individual head
   importance. High order-1 fraction among non-constant energy (predicted
   >60%). Moderate total game variance.

2. **Epistatic cluster**: {C3} -- Singleton. The canonical pipeline circuit
   should have a unique spectral signature with elevated order-2 and order-3+
   fractions that no other circuit matches, because no other circuit contains
   the full four-stage pipeline.

3. **Mixed cluster**: {C1, IC-15} -- Both have moderate canonical coverage
   (4-5 heads), many non-canonical heads, and moderate game variance. Their
   spectra should fall between the additive and epistatic extremes. C1 leans
   slightly more epistatic (composition selection) and IC-15 slightly more
   additive (greedy picks individually important heads first), but both should
   be distinguishable from the additive and epistatic clusters.

4. **Null cluster**: {C4} -- Dominated by order-0 (>98%). The non-constant
   energy is so small that its distribution across orders is poorly determined,
   making C4 an outlier in any distance metric.

FALSIFIED IF: Hierarchical clustering (on the 4-dimensional energy-fraction
vector) with a reasonable linkage criterion places C5 and C3 in the same cluster
before either joins C2, or places C4 in a cluster with any functional circuit
before the functional circuits merge with each other.

---

## Section 4: Surprises

Outcomes that would challenge the theoretical framework underpinning these
predictions:

### Surprise 1: C4 (random) shows structured epistasis

If C4's non-constant energy is >30% at order-2+ (among non-constant), this
would suggest the transformer has pervasive non-additive interactions even among
functionally unrelated heads. This could arise from LayerNorm-induced coupling
or from the softmax attention mechanism creating universal pairwise interactions.
It would undermine the assumption that high epistasis is a signature of
functional coordination.

### Surprise 2: C3 (canonical) is predominantly additive

If C3's order-1 fraction among non-constant energy exceeds 70%, the canonical
pipeline structure does not create the expected epistatic interactions under mean
ablation. This would suggest that mean ablation preserves enough residual
information flow through each pipeline stage that the stages function semi-
independently, or that the within-group redundancy (4 S-inhibition heads, 5
name movers) drowns out the cross-group interactions in the Walsh spectrum.

### Surprise 3: IC-15 has the highest faithfulness delta

IC-15 was selected for sufficiency (keeping these heads reproduces model
behavior), not necessity (removing them damages model behavior). If IC-15
nonetheless has the highest delta, it would imply that greedy sufficiency
selection incidentally identifies the most necessary heads -- that the heads
the model can least afford to lose are the same heads that, alone, best
reproduce full-model behavior. This would be a strong sufficiency-necessity
duality result.

### Surprise 4: C5 and C3 have indistinguishable spectra despite different selection criteria

C5 and C3 share 8 heads, but C5 was selected purely by additive marginal
effects while C3 was identified by mechanistic tracing. If their spectra match
within 2 percentage points on all orders, it would suggest that the Walsh
spectrum is determined primarily by which heads are present (shared heads
dominate) rather than by the method's inductive bias. This would weaken the
"selective pressure" thesis.

### Surprise 5: Order-3+ energy dominates order-2 in any functional circuit

The pipeline structure (prev-token -> duplicate-token -> S-inhibition ->
name-movers) is a pairwise chain, so order-2 interactions should dominate
higher orders. If order-3+ exceeds order-2 for any functional circuit, it would
indicate that the IOI mechanism involves irreducible three-way (or higher)
synergies -- perhaps a head's contribution depends on the joint state of two
other heads simultaneously, not just pairwise combinations. This would call for
richer interaction decomposition methods beyond pairwise analysis.
