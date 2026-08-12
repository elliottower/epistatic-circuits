> **Provenance.** Reformatted from `preregistrations/prereg_v12c_multi_task_extension.md`.
> Original frozen at commit `21f4863`. View: `git show 21f4863:preregistrations/prereg_v12c_multi_task_extension.md`

# Pre-Registration v12c: Multi-Task Extension of Walsh Energy Analysis

**Status**: FROZEN (blind -- no Walsh spectra, faithfulness deltas, or epistasis
scores for any of the four new tasks have been viewed by the author)

**Amendment 1** (added blind to greedy sufficiency and ACDC results -- discovery
script running, no outputs viewed): Predictions 5--8 and Surprise 4 added below.

**Parent document**: `prereg_v12b_selective_pressures.md` (IOI-only, 6 circuits)

---

## Overview

This amendment extends the Walsh-Hadamard interaction analysis from IOI to four
additional GPT-2 Small tasks. Each task uses its published ground-truth circuit
("known"), a circuit discovered by mean-ablation importance ("ablation-
discovered"), and a random-head null control ("random"). All circuits are
evaluated under both zero and mean ablation.

The purpose is to test whether the selective-pressure pattern observed on IOI
generalizes: ablation-based discovery selects for additive (order-1 dominated)
interaction structure, while mechanistically-traced known circuits preserve
higher-order interactions reflecting their pipeline structure.

---

## Tasks and Circuits

### Greater-Than (Hanna et al. 2023)

Metric: prob_diff = P(years > YY) - P(years <= YY) over 100 year tokens.

**Known circuit** (7 heads): L5H1, L5H5, L6H9, L7H10, L8H8, L8H11, L9H1

Pipeline structure: heads span layers 5--9 in a sequential computation that
compares numerical magnitude. The year-comparison mechanism involves early heads
(L5) attending to the year token and later heads (L8--L9) aggregating the
comparison signal.

**Ablation-discovered**: top 7 heads by mean-ablation importance on prob_diff.

**Random**: 7 heads selected uniformly at random (seed=42).

### Subject-Verb Agreement (Linzen et al. 2016, circuit from Finlayson et al. 2021)

Metric: verb_score = sum P(agreeing verb forms) - sum P(disagreeing verb forms)
over single-token verb pairs from the Marvin & Linzen vocabulary.

**Known circuit** (12 heads): L0H4, L0H8, L1H0, L1H1, L2H1, L2H6, L6H0,
L9H4, L10H0, L11H4, L11H6, L11H7

Pipeline structure: early heads (L0--L2) encode subject number, mid-layer heads
(L6) track syntactic structure across attractors, and late heads (L9--L11) read
out the agreement signal. The 12-head circuit spans nearly the full model depth,
with a gap in layers 3--5.

**Ablation-discovered**: top 12 heads by mean-ablation importance on verb_score.

**Random**: 12 heads selected uniformly at random (seed=42).

### Gendered Pronoun (de Vassimon Manela et al. 2021)

Metric: logit_diff = logit(" she") - logit(" he"), signed by profession
stereotype.

**Known circuit** (5 heads): L0H10, L3H0, L5H8, L6H6, L8H6

Pipeline structure: shallow. Only 5 heads across layers 0--8, with no obvious
sequential pipeline. The heads likely implement a relatively direct mapping from
gendered context to pronoun prediction.

**Ablation-discovered**: top 5 heads by mean-ablation importance on logit_diff.

**Random**: 5 heads selected uniformly at random (seed=42).

### Induction (Olsson et al. 2022)

Metric: log probability of the correct induction token (the token following a
repeated bigram prefix).

**Known circuit** (7 heads): L2H2, L4H11 (previous-token heads) + L5H1, L5H5,
L6H9, L7H2, L7H10 (induction heads)

Pipeline structure: two-stage. Previous-token heads attend to the token before
the repeated prefix; induction heads compose with previous-token heads to copy
the token that followed the prefix at its first occurrence. This is a canonical
example of composition through the residual stream.

**Ablation-discovered**: top 7 heads by mean-ablation importance on log
probability.

**Random**: 7 heads selected uniformly at random (seed=42).

---

## Predictions

All predictions apply to mean ablation unless stated otherwise.

### Prediction 1: Ablation-discovered circuits are more additive than known circuits

For each of the four tasks, the ablation-discovered circuit will have a higher
order-1 energy fraction (among non-constant energy) than the known circuit under
mean ablation.

REASONING: Ablation-based discovery ranks heads by individual mean-ablation
effect, which is approximately an order-1 criterion. This selects heads whose
contributions are individually large and therefore separable, producing an
additive bias. Known circuits were identified by tracing mechanistic pathways,
which preserves multi-head dependencies that appear as order-2+ energy.

FALSIFIED IF: The known circuit has a higher order-1 fraction than the
ablation-discovered circuit on 2 or more of the 4 tasks. A single violation
could reflect a task where the known circuit happens to be shallow.

### Prediction 2: Random circuits are near-additive on all tasks

Random circuits will have >95% order-1 energy fraction under mean ablation on
all four tasks.

REASONING: Random heads lack systematic functional relationships. Their
individual effects on task metrics are small and uncorrelated, so the game
v(S) is nearly constant. Whatever non-constant energy exists should be
predominantly order-1 because pairwise interactions require functional
coordination that random heads do not have.

FALSIFIED IF: Any random circuit has <90% order-1 energy under mean ablation.
A violation between 90--95% weakens but does not falsify the prediction.

### Prediction 3: Pipeline depth predicts epistasis

Among the four known circuits under mean ablation, order-2 energy fraction
(as a fraction of non-constant energy) will rank:

PREDICTION: induction > greater-than > SVA > gender-bias

REASONING: Induction has the clearest two-stage pipeline (previous-token heads
compose with induction heads). Greater-than spans 5 layers (L5--L9) with
sequential magnitude comparison. SVA spans the full model but the 12 heads
include redundant late-layer readout heads that dilute pairwise interactions.
Gender-bias has only 5 heads with no obvious sequential pipeline, so its
interactions should be weakest.

FALSIFIED IF: Gender-bias ranks above induction, or if induction ranks last.
Partial violations (e.g., SVA and greater-than swapped) do not falsify the core
claim that pipeline depth predicts epistasis.

### Prediction 4: Faithfulness ranking is consistent across tasks

For each task under mean ablation:

PREDICTION: faithfulness(known) > faithfulness(ablation-discovered) > faithfulness(random)

where faithfulness = v(all heads active) - v(all heads ablated).

REASONING: Known circuits were identified by mechanistic analysis targeting the
task's core computation. They should contain the most necessary heads.
Ablation-discovered circuits select individually important heads, which overlaps
with but is not identical to necessity. Random circuits contain no systematically
task-relevant heads.

FALSIFIED IF: The ranking is violated on 2 or more tasks. A single violation
(e.g., ablation-discovered slightly exceeding known on one task) could reflect a
task where the published circuit omits individually impactful heads.

---

## Additional Circuit Discovery Methods

Two iterative, context-aware methods are added to each task alongside the
existing knockout top-K, known, and random circuits.

### Greedy sufficiency (forward selection)

Start with an empty circuit. At each step, evaluate every candidate head by
running the model with the current circuit plus that head active and all other
heads mean-ablated. Add the head that maximizes the task metric. Repeat until
the circuit reaches K heads (matching the known circuit size for that task).

Later selections depend on earlier ones: if head A is already in the circuit,
the algorithm favors heads that complement A rather than duplicate its effect.

### ACDC-like backward elimination

Start with the top-50 heads ranked by individual mean-ablation attribution.
At each step, try removing each remaining head and measure the metric with the
reduced set active. Remove the head whose removal causes the smallest metric
drop. Continue until K heads remain.

Starting from top-50 (rather than all 144) keeps the search tractable while
ensuring the starting pool contains all plausibly relevant heads.

---

### Prediction 5: Greedy sufficiency circuits have intermediate additivity

For each task under mean ablation, greedy sufficiency circuits will have an
order-1 energy fraction between knockout top-K (higher, more additive) and
known circuits (lower, more epistatic).

REASONING: Greedy forward selection prioritizes individually strong heads at
early steps, maintaining an additive bias. But later selections complement
earlier ones — if the first head already provides signal X, the algorithm
picks heads that provide signal Y rather than redundant X. These implicit
dependencies introduce interaction structure that pure knockout misses, but
not as much as the full mechanistic pipeline preserved in known circuits.

FALSIFIED IF: Greedy sufficiency is more additive than knockout top-K on 3 or
more tasks, or more epistatic than the known circuit on 3 or more tasks.

### Prediction 6: ACDC circuits are similar to greedy sufficiency, possibly more epistatic

ACDC circuits will have order-1 fractions within 5 percentage points of greedy
sufficiency on at least 3 of 4 tasks. Where they differ, ACDC will tend
toward lower order-1 (more epistatic).

REASONING: Backward elimination preserves composition naturally — heads that
are useful only in combination survive pruning because removing either one
hurts performance. Forward selection can miss such pairs if neither head is
individually strong. However, starting from top-50 heads (pre-filtered by
individual importance) limits how much composition structure ACDC can
discover, narrowing the gap with greedy sufficiency.

FALSIFIED IF: ACDC and greedy sufficiency differ by more than 10 percentage
points on order-1 fraction for 2 or more tasks, or ACDC is consistently more
additive than greedy sufficiency across all 4 tasks.

### Prediction 7: Additivity ranking across methods

For each task under mean ablation, the order-1 energy fraction will rank:

PREDICTION: knockout top-K > greedy sufficiency > ACDC > known

This ranking should hold on at least 3 of 4 tasks.

REASONING: Methods that select for individual importance produce additive
spectra; methods that account for context produce intermediate spectra;
ground-truth circuits traced by mechanistic analysis preserve the most
interaction structure. Knockout is context-free (strongest additive bias),
greedy sufficiency is context-aware but bottom-up, ACDC is context-aware and
top-down (slightly better at preserving composition), and known circuits
reflect the full mechanistic pipeline.

FALSIFIED IF: The ranking is violated on 3 or more tasks, or if known
circuits are more additive than knockout on any task (would mean mechanistic
tracing incidentally selects more independently-acting heads than brute-force
knockout).

### Prediction 8: Faithfulness ranking across methods

For each task under mean ablation:

PREDICTION: known >= ACDC > greedy sufficiency > knockout top-K > random

REASONING: ACDC preserves necessity — backward elimination keeps heads the
circuit cannot afford to lose. Greedy sufficiency builds sufficiency — it
finds heads that collectively reproduce the behavior, but may include
partially redundant heads whose individual necessity is lower. Knockout
selects by individual importance without accounting for redundancy or
complementarity. Known circuits, traced from the full mechanism, should
contain the most necessary heads overall.

FALSIFIED IF: Random circuit faithfulness exceeds knockout on 2 or more tasks,
or knockout exceeds ACDC on 3 or more tasks.

---

## Surprises

### Surprise 1: A known circuit is predominantly additive (>85% order-1)

This would suggest that the published circuit, despite being identified through
mechanistic analysis, does not contain functionally interacting heads. Possible
explanations: the circuit has high within-group redundancy (multiple heads doing
similar work), or the mechanistic pathway is shallow enough that each head
contributes semi-independently.

Most likely candidate: gender-bias (only 5 heads, shallow pipeline).

### Surprise 2: SVA shows the highest epistasis despite having the most heads

With 12 heads, SVA has 4096 coalitions and the richest potential interaction
structure. If its order-2+ energy exceeds induction's despite lacking a clear
two-stage pipeline, it would suggest that epistasis scales with circuit size
rather than pipeline depth. This could arise from LayerNorm-mediated coupling
among the 12 heads or from the attractor structure of SVA creating implicit
multi-head dependencies.

### Surprise 3: Random circuit faithfulness exceeds 50% of known circuit faithfulness

Random heads should contribute negligibly to any specific task. If a random
circuit captures more than half the faithfulness of the known circuit, it would
suggest that GPT-2 Small has high distributed redundancy for that task -- many
heads contribute partially, making even random selections moderately faithful.

### Surprise 4: Greedy sufficiency and ACDC discover the same circuit

If both methods converge on the same K heads for a task (Jaccard similarity
>0.8), it would mean that forward selection and backward elimination are
navigating different search directions to the same basin. This would suggest the
circuit is a robust attractor in the space of K-head subsets -- its members are
simultaneously the most sufficient and the most necessary. Tasks with shallow
pipelines (gender-bias) are the most likely candidates; deep pipelines
(induction) are least likely because forward and backward searches weight
composition differently.

---

## Analysis Plan

1. Discover greedy sufficiency and ACDC circuits for each task (K matching
   known circuit size). Record which heads each method selects.
2. Run exhaustive coalition sweeps (2^K coalitions × N prompts) for each
   (task, circuit, ablation type) combination. Circuits: known, knockout top-K,
   greedy sufficiency, ACDC, random.
3. Compute Walsh-Hadamard transform of the coalition set function for each
   sweep.
4. Report energy spectrum as percentage of non-constant energy at each
   interaction order.
5. Report faithfulness delta for each (task, circuit, ablation type).
6. Report bootstrapped epistasis scores with 95% confidence intervals (2000
   bootstrap samples).
7. Compare across tasks and methods using a summary table with columns: task,
   circuit, method type, faithfulness, order-1%, order-2%, order-3+%, epistasis.
8. Compute Jaccard similarity between circuits discovered by different methods
   for each task.
