> **Provenance.** Reformatted from `preregistrations/prereg_v13_complementation_units.md`.
> Original frozen at commit `6163ab0`. View: `git show 6163ab0:preregistrations/prereg_v13_complementation_units.md`

# Pre-registration v13: Do the field's head roles survive a complementation test?

**Status: FROZEN before any order-2 coefficient was inspected.**
Written 2026-08-06. Nothing in `data/c6_zero_v2_coalition_values.npz` beyond array
shapes and metadata keys has been read at time of writing.

---

## The question

Genetics has had a procedure since the 1950s for deciding whether two mutations belong
to one unit or two, and it works without knowing what the unit is made of. Benzer used it
to define the **cistron** — the gene *is* the complementation unit — and the resulting
fine-structure map killed the bead-on-a-string picture of the gene.

Mechanistic interpretability has named head types — name mover, S-inhibition, induction,
duplicate token, previous token — and has **never applied a functional test for whether
two heads belong to the same unit.** The names come from what a head appears to do, not
from a procedure that carves.

This registration asks whether the field's role labels survive such a test.

## Why this is not the v12 question

v12 asks which *discovery method* finds epistatic circuits, taking circuits as given and
characterising them. This asks what the *units* are, deriving them from interaction
structure and treating the names as the thing under test rather than as the input.

Same coalition sweeps, different object.

---

## Disclosure: observed vs unobserved

### Observed (design parameters, not results)
- `data/c6_zero_v2_coalition_values.npz` exists; keys are `target_logits`, `foil_logits`,
  `coalition_indices`, `circuit_name`, `ablation_type`, `circuit_heads`, `n_players`,
  `n_prompts`, `intact_target_logits`, `intact_foil_logits`.
- The 17 role labels in
  `factorization-circuits/.../v1_role_weight_analysis/part2/data/confusion_data.json`:
  IOI:{DTH, IND, NM, NegNM, PTH, S-Inh}, Induction:{IND, PTH},
  Greater-Than:{early_gt, late_gt}, SVA:{embed, encode, output, route},
  Gendered Pronoun:{early_ga, late_ga, name_bind}.
- v12's published Walsh energy fractions per circuit (order-1 / order-2 / order-3+).
- That gendered-pronoun and SVA known circuits are functionally inert
  (faithfulness −0.025 and +0.038), from `results/RESULTS_SUMMARY.md`.

### Genuinely unobserved (confirmatory)
- Every individual order-2 coefficient.
- The interaction matrix, its clustering, and any comparison to role labels.
- All quantities in Predictions 1–4 below.

---

## The test, and the sign convention it turns on

Complementation in genetics reads interaction, not magnitude. The standard mapping, which
this registration adopts and which must be stated before any coefficient is seen:

| interaction between two ablations | genetics reading | unit verdict |
|---|---|---|
| **sub-additive / masking** — ablating both costs less than the sum of singles | same pathway; the second hit has nothing left to break | **same unit** |
| **additive** — order-2 coefficient ≈ 0 | independent pathways | **different units** |
| **synergistic** — ablating both costs more than the sum | parallel redundant pathways, each covering for the other | **different units, mutually buffering** |

The claim under test is that heads the field gives the *same role label* should mask each
other, and heads with *different* labels should not.

**Note the asymmetry this creates.** Masking and synergy are both "large |order-2|." A test
that used magnitude alone would call redundant-but-distinct heads one unit. Sign is
therefore load-bearing, and any result reported without it is uninterpretable.

---

## Operationalization

**Interaction estimate.** For each task, fit the LASSO-Walsh model already used in v12 to
the coalition values, and take the order-2 coefficient `w_ij` for every head pair. Use the
same fitting code, hyperparameters and coalition budget as v12 — no re-tuning.

**Masking matrix.** `M_ij = -w_ij` under the sign convention above, so that positive `M`
means masking, negative means synergy, zero means additive. Report the raw signed matrix,
never `|w_ij|`.

**Units.** Hierarchical clustering (average linkage) on `M`, cut at the height that
maximises silhouette. Cluster count is not fixed in advance.

**Comparison.** Adjusted Rand Index between the derived clusters and the field's role
labels, per task, on the heads that carry a label.

**Ablation type.** Zero ablation, since that is what the held NPZ contains. Mean ablation
is a robustness check if the corresponding sweep is available, not a primary analysis.

---

## Prediction 1 — the taxonomy test (primary)

**Same-role head pairs mask each other more than different-role pairs.**

Statistic: mean `M_ij` for within-label pairs minus mean `M_ij` for cross-label pairs, per
task, tested by label permutation (10,000 permutations of the role labels over heads).

- **Confirms** if the within-minus-cross difference is positive at p < 0.05 in ≥3 of the
  tasks that have ≥2 labels with ≥2 heads each.
- **Disconfirms** if it is null or negative in a majority of eligible tasks.

**Prediction: disconfirmed.** The field's labels come from behavioural description, not
from any interaction criterion, and there is no reason a name assigned by what a head
appears to do should track what masks what. Registering the negative because it is the
result I expect and the one that would be worth reporting.

## Prediction 2 — do units exist at all?

**Derived clusters are stable across tasks for heads that appear in more than one task.**

Statistic: for heads appearing in ≥2 tasks, the rate at which two such heads land in the
same cluster in one task and the same cluster in another.

- **Units exist** if co-clustering agreement across tasks exceeds a within-task label
  permutation null at p < 0.05.
- **No units at this level** if agreement is at chance — the strongest form of the
  too-loose finding, and a publishable negative.

## Prediction 3 — ARI against the taxonomy

Adjusted Rand Index between derived clusters and role labels, per task.

- **ARI > 0.4** — the taxonomy substantially recovers the complementation partition.
- **0.1 < ARI ≤ 0.4** — partial; report which labels survive and which dissolve.
- **ARI ≤ 0.1** — the taxonomy and the interaction structure are unrelated.

**Prediction: ARI ≤ 0.1 on IOI**, the task with the most labels and the most published
structure.

## Prediction 4 — the inert-circuit control

Gendered pronoun and SVA known circuits are functionally inert (faithfulness −0.025 and
+0.038). Their interaction structure should therefore be noise.

- **Passes** if `M` for these tasks is statistically indistinguishable from a coalition-
  shuffled null, and the clustering is unstable across bootstrap resamples.
- **Fails, and invalidates Predictions 1–3**, if inert circuits produce structure as
  strong as functional ones — which would show the pipeline manufactures clusters from
  noise.

This is the analysis's own null model and it must be run and reported first.

---

## What would make the whole thing uninterpretable

Stated in advance so it cannot be rationalised afterwards.

1. **Prediction 4 fails.** Structure in the inert circuits means the method invents units.
   Report that and stop; no other number means anything.
2. **Order-2 estimates are unreliable.** Split the coalition set in half, refit, and
   correlate the two `M` matrices. If split-half reliability is below 0.6 for a task, that
   task is dropped — a low ARI cannot be distinguished from a noisy `M`.
3. **Degrees of freedom.** With 15 heads there are 105 pairs, fitted from 20K coalitions.
   If LASSO zeroes out more than half the order-2 terms for a task, clustering is being
   done on a mostly-empty matrix and that task is reported as under-determined rather than
   as a negative result.

---

## Recovery protocol

If any of the above forces a change to the analysis after data are seen, the change is
recorded here with its date and reason, the original text is left struck through rather
than deleted, and every affected result is reported as exploratory.

---

## Freeze

Code and data SHAs recorded at freeze time in `FREEZE.txt` alongside this file. No order-2
coefficient is read before that file exists.

---

# AMENDMENTS — recorded before any order-2 coefficient was computed

## A1. Data re-acquisition (2026-08-06)

The frozen `data/c6_zero_v2_coalition_values.npz`
(sha256 `861c1799…`) is **corrupt**: `target_logits.npy` fails CRC-32 on read. Metadata
arrays read fine, which is why the freeze recorded a valid hash for a broken file.

Re-downloaded from Modal volume `c6-coalition-sweep`, saved as
`data/c6_zero_v2_FRESH.npz`, sha256 `e6bc023e54b11e2b4a0b6c550dc839309ef233708f92a417a176ea4ce9f005e6`.
Verified intact: all 16 arrays read, `n_coalitions_completed = 32768`,
`target_logits` shape (32768, 512), 32768 unique coalition indices.

No result was computed from either file before this amendment.

## A2. ~~LASSO-Walsh fit~~ → exact Walsh–Hadamard transform

The frozen text specified "fit the LASSO-Walsh model already used in v12 … same fitting
code, hyperparameters and coalition budget as v12."

**That is the wrong method for this data.** 2^15 = 32768 = the full coalition lattice over
15 players. The sweep is **exhaustive**, not sampled, so the Walsh–Hadamard transform is
*exact*: every order-2 coefficient is determined with no regularisation, no held-out set,
and no hyperparameter.

Amended method: compute the exact WHT of the per-coalition value function and read order-2
coefficients directly.

Consequences for the frozen text, all in the direction of a stronger test:

- **Abort condition 3 is void.** "If LASSO zeroes out more than half the order-2 terms" is
  inapplicable — nothing is zeroed by a penalty. All 105 pairs are estimated.
- **Abort condition 2 changes meaning.** Split-half reliability now measures *prompt*
  sampling noise (512 prompts), not coalition sampling noise. Retained at the same 0.6
  threshold, computed by splitting prompts rather than coalitions.
- **Prediction 3's under-determination clause** no longer applies.

Registered before computation. Original text struck rather than deleted.

## A3. Scope reduction — Prediction 2 cannot run as written

Prediction 2 requires heads appearing in ≥2 tasks. The held sweep is a single circuit
(`circuit_name = c6`, the order-2-discovered IOI circuit, 15 heads, zero ablation).
Other volumes hold `c2`, `c5` (also IOI circuits) and `gt_*`, `gender_*` sweeps over
different head sets.

Prediction 2 is therefore **deferred**, not tested, until the cross-task sweeps are pulled
and their head overlap is known. It is not reported as a null. Predictions 1, 3 and 4 are
unaffected for the circuits available.
