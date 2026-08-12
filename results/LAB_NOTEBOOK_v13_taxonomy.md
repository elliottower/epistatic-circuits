# Lab notebook — v13: does the IOI head taxonomy survive a complementation test?

Session of 2026-08-06. Chronological, including the passes that failed and the two
occasions where a result had to be reinterpreted after checking a source.

Pre-registration: `preregistrations/prereg_v13_complementation_units.md`,
freeze `preregistrations/FREEZE_v13.txt` (sha `1f915bbc…`, amended `e258d592…`).

---

## 0. The question

Genetics decides whether two mutations belong to one unit or two by putting them in
*trans* and reading the phenotype of the double mutant. The test needs no knowledge of
what the unit is made of. Benzer used it to define the cistron.

Interpretability names head types — name mover, S-inhibition, induction — from what a head
appears to do. No functional test decides whether two heads belong to the same unit. A
coalition sweep over an entire circuit supplies exactly the double-ablation data that the
complementation test consumes, so the test can be run.

Registered claim under test: heads sharing a role label should mask each other, and heads
with different labels should not.

---

## 1. Data problems, resolved before any coefficient was read

**The frozen file was corrupt.** `data/c6_zero_v2_coalition_values.npz` (sha
`861c1799…`) fails CRC-32 on `target_logits.npy`. The metadata arrays read fine, which is
why the freeze recorded a valid hash for a broken file — a hash over a zip container does
not certify the members decompress.

Re-downloaded from Modal volume `c6-coalition-sweep` as `data/c6_zero_v2_FRESH.npz`
(sha `e6bc023e…`), verified: 16 arrays readable, `n_coalitions_completed = 32768`,
`target_logits` shape (32768, 512), 32768 unique coalition indices. Recorded as amendment
A1 before computation.

**The registered method was wrong for the data.** The prereg specified the LASSO-Walsh fit
used in v12. But 2^15 = 32768 is the *full* coalition lattice over 15 players — the sweep
is exhaustive, not sampled. The Walsh–Hadamard transform is therefore **exact**: every one
of the 105 order-2 coefficients is determined with no regularisation, no held-out set, no
hyperparameter.

Amendment A2 switched to the exact transform. Consequences, all in the direction of a
stronger test: abort condition 3 (LASSO zeroing terms) is void; abort condition 2 changes
meaning, since split-half reliability now measures prompt-sampling noise rather than
coalition-sampling noise.

**Prediction 2 cannot run.** It requires heads appearing in ≥2 tasks; the held sweep is a
single circuit. Deferred by amendment A3, not reported as a null.

---

## 2. Sign convention — verified rather than assumed

The whole test turns on sign. Masking and synergy are both "large |w₂|", and a test using
magnitude alone would call redundant-but-distinct heads one unit.

The coalition bitmask encodes heads **kept** (v(∅) = −1.05, v(full) = +0.61), so the
direction of the mapping from Walsh order-2 to interaction sign was checked empirically
before use. Over 40 probe pairs:

```
mean discrete interaction  E_S[v(S∪ij) − v(S∪i) − v(S∪j) + v(S)]
  = exactly 4 × the Walsh order-2 coefficient,  r = 1.0000
```

So masking corresponds to negative order-2, and the masking matrix is `M = −w₂`.

---

## 3. Gates

Both registered gates ran first and both passed.

| gate | result |
|---|---|
| Reliability (abort #2, amended to prompt-split) | split-half r = 0.989, Spearman–Brown **0.995** (threshold 0.6) |
| P4 inert-circuit control (must run first; invalidates all else) | pass, below |

### P4 — the pipeline does not manufacture structure

Bootstrap over prompts, 1000 draws, percentile CIs, matching
`scripts/modal_rti_epistasis_bootstrap.py`.

| circuit | n | RTI epistasis | 95% CI | order-2 frac | 95% CI |
|---|---|---|---|---|---|
| c6 IOI (functional) | 15 | −0.329 | [−0.432, −0.228] | **0.303** | [0.286, 0.319] |
| gender known (inert, faithfulness −0.025) | 5 | +0.213 | [+0.173, +0.248] | **0.011** | [0.006, 0.016] |
| gender random | 5 | +0.748 | [−2.68, +4.34] | 0.032 | [0.013, 0.069] |

Order-2 fractions separate by a factor of ~20 with no CI overlap. The random circuit's
epistasis CI explodes because its group effect is near zero, making the ratio undefined in
the limit — itself a signature of an inert circuit.

*Correction made during this pass.* The first version of P4 used a coalition-shuffled null
and reported a z-score (z = +641 functional, −2.7 inert). That statistic was invented for
this analysis. Redone with the house bootstrap-over-prompts method. Same conclusion; the
z-score version is superseded and retained only for transparency.

---

## 4. Registered predictions

### P1 — within-role vs cross-role masking: NOT CONFIRMED, and uninformative

Within-role mean masking minus cross-role = **+0.0028**, label-permutation p = **0.33**
(10,000 permutations, 9 heads, 6 within-pairs, 30 cross-pairs).

The registered prediction was "disconfirmed", and it is not confirmed. But six within-pairs
cannot distinguish a null effect from a real one. **Report as uninformative, not as
evidence of absence.**

### P3 — ARI against the taxonomy: REGISTERED PREDICTION WRONG

k chosen by silhouette per the prereg. Point estimate k = 3, **ARI = +0.571**.
Bootstrapped over prompts (1000 draws): mean +0.410, 95% CI **[+0.125, +0.571]**. The point
estimate sits at the top of its own bootstrap distribution. k is unstable: 3 in 334 draws,
4 in 400, 5 in 206, 6 in 60.

Registered prediction was ARI ≤ 0.1, so it is rejected — but only just, and the
"substantially recovered" reading (> 0.4) does not survive resampling. Correct statement:
the taxonomy tracks interaction structure better than predicted, by an uncertain amount.

*Procedural note.* The first run selected k by maximising ARI, which is selection on the
outcome. Corrected to silhouette selection as registered. Same k, same value — but the
uncorrected version must not be reported as confirmatory.

### The partition

| cluster | heads | roles |
|---|---|---|
| 2 | L5H5, L6H9 | IND, IND — recovered exactly |
| 3 | L9H9, L10H0 | NM, NM — recovered exactly |
| 1 | L7H9, L8H6, L8H10, L10H7, L11H10 | S-Inh ×3 + NegNM ×2 — **merged** |

S-inhibition heads and negative name movers do not separate. Unlike the global fit, this is
stable: they co-cluster in **1000/1000** bootstrap resamples over prompts.

### Layer confound ruled out

Roles in IOI are layer-stratified, so the clustering could be recovering depth.

| comparison | ARI |
|---|---|
| cluster vs **role** | **0.571** |
| cluster vs layer | 0.053 |
| cluster vs layer-band (early/mid/late) | 0.217 |
| role vs layer-band | 0.526 |

The clustering tracks role an order of magnitude better than raw layer and more than twice
as well as layer-band, despite roles themselves being strongly band-aligned.

---

## 5. Multi-condition robustness

Conditions chosen by mapping to framework criteria rather than by convenience: mean
ablation tests invariance to the ablation primitive; alternative circuits test convergence
across discovery methods; linkage variation tests robustness to an analyst choice.

`scripts/v13_conditions.py`, bootstrapped over prompts, expanded role dictionary — see caveat:

```
condition                     elig  S/N  k     ARI       ARI 95% CI   merge rate
c6 zero  [primary]              12  3/2   7   0.177  [+0.006,+0.177]      81%
c6 mean  [ablation primitive]   12  3/2   9   0.106  [+0.106,+0.327]     100%
c2 EAP-IG zero  [discovery]      5  3/2   3  -0.364  [-0.364,-0.364]     100%
c5 Walsh-o1 zero [discovery]     7  3/2   3   0.212  [+0.087,+0.212]     100%
c6 zero, complete linkage       12  3/2   6   0.177  [+0.006,+0.290]      83%
c6 zero, ward linkage           12  3/2   6   0.290  [+0.006,+0.290]      86%
```

**Caveat that must travel with these numbers.** This run used an expanded role dictionary
(18 labelled heads rather than the 10 used in P3), so the eligible set grew from 9 to 12 and
the ARIs are **not comparable** to the 0.571 above. Changing the label set after seeing
results is a forking path. What it does suggest is that the registered prediction (ARI ≤
0.1) was closer to correct than 0.571 made it look, and that the fuller the labelling, the
worse the taxonomy tracks interaction structure. The c2 ARI is negative — that clustering is
worse than chance against the taxonomy.

**The merge weakens under the expanded label set, and this should be reported.** It holds in
every condition as a point estimate, and at 100\% of bootstrap draws on the mean-ablation and
alternative-circuit conditions. But on the three c6-zero conditions it drops to 81--86\%,
against 1000/1000 in the registered 10-label analysis. Adding labelled heads gives the
clustering more ways to split the pair. Quote the registered figure for the registered
analysis and this range for the expanded one; do not quote 100\% across both.

---

## 6. Checking the source — the finding deflates

An agent read Wang et al. (2211.00593) in full, working from a locally downloaded PDF and
from rendered page images, not from a summary.

**The canonical assignment.** Figure 2 (p. 4) and the minimality table (Figure 20) agree:
26 heads, 7 classes. Previous Token 2.2, 4.11; Duplicate Token 0.1, 3.0, (0.10); Induction
5.5, 6.9, (5.8, 5.9); S-Inhibition 7.3, 7.9, 8.6, 8.10; Negative Name Mover 10.7, 11.10;
Name Mover 9.9, 9.6, 10.0; Backup Name Mover 9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9.

My from-memory transcription had **11.6** where the paper has **11.9**. It affected nothing
computed, which is luck rather than method.

**Three things that weakened the original reading.**

1. *No separability claim exists to contradict.* Classes were assigned by inspection after
   path patching, with post-hoc naming — "We therefore call these heads S-Inhibition Heads".
   Backup membership is explicitly arbitrary: "We arbitrarily chose to keep the eight heads
   that were not part of any other groups". Class is used operationally only as a convenient
   sampling unit for the completeness and minimality tests.
2. *The paper reports the same non-additivity itself.* Appendix A, line 922: "it is not
   possible to ignore the interactions between heads inside a class".
3. *Figure 2 places S-Inhibition upstream of a dashed box containing Name Movers, Negative
   Name Movers and Backup Name Movers*, with S-Inhibition's output entering as a query arrow.
   On that topology, ablating S-Inhibition already disables what the box does, so a merge is
   unsurprising.

Verdict at this point: the merge is consistent with what the paper already implies. **The
"a class boundary is fake" framing is not supported.**

---

## 7. Re-reading the result — the asymmetry the diagram cannot represent

`scripts/v13_pairwise.py` tests every class pair for co-clustering, using the sourced labels:

```
NegNM   ~ S-Inh    merged in 4/4 conditions
NM      ~ S-Inh    merged in 0/2 conditions
NM      ~ NegNM    merged in 0/2 conditions
IND     ~ (all)    merged in 0/2–0/3 conditions
```

The dashed box contains Name Movers and Negative Name Movers **equally** — same query
source, same name key/value, same output edge. Wang et al. state the only difference: the
negative heads "share all the same properties as Name Mover Heads except they (1) write in
the opposite direction of names they attend to and (2) have a large negative copy score."

The topological explanation for the merge therefore predicts that S-Inhibition masks *both*.
It masks one. Figure 2 has no way to represent that difference.

### The mechanism, and the test it implies

Negative name movers write the negation of what they attend to. With S-inhibition intact
they attend to IO and write −IO, which hurts. Remove S-inhibition and they attend to S
instead, writing −S — which *helps*, since the logit difference is IO − S. Their
contribution should **invert**, not merely shrink. Name movers, copying positively, should
only be blunted.

`scripts/v13_conditional_marginal.py` computes E[v(S∪{i}) − v(S)] exactly, restricted to
coalitions where all three S-inhibition heads are kept versus all removed.

```
role     head      marg | S-Inh KEPT   marg | S-Inh GONE    delta    flip rate
NegNM    L10H7            -0.1301            +0.1369      +0.2670   1000/1000
NegNM    L11H10           -0.0571            +0.1798      +0.2368   1000/1000
NM       L9H9             +0.1290            -0.0123      -0.1413    878/1000
NM       L10H0            +0.1494            +0.0086      -0.1408    318/1000
```

Both negative heads invert, in every bootstrap draw. The name movers decay toward zero —
L10H0 does not flip, and L9H9's flip lands at −0.012, which is zero. Sign reversal and
neutralisation are different phenomena, and they produce the interaction signs the
clustering found:

- S-Inh × NegNM: interaction **−0.267** → masking → same unit
- S-Inh × NM: interaction **+0.141** → synergy → different units

**Wang et al. observed a fragment of this and filed it as unexplained.** Appendix F: "the
Negative Name Mover Heads have a less negative effect on logit difference, and 10.7 even has
a positive effect on the logit difference after the knockout", followed by "Both the reason
and the mechanism of this compensation effect are still unclear." Their conditioning set was
name-mover knockout rather than S-inhibition, and it was one head noted in passing.

Two caveats on this section. The test was derived from the pairwise result, so it is a
follow-up the registration did not anticipate and must be reported as such. And the first
run of the script carried a leaked loop variable that mislabelled all four bootstrap rows;
head identifiers and numbers were correct, labels were not. Patched and rerun; the table
above is from the corrected run.

---

## 8. What is established, and what is not

**Established.**
- The interaction decomposition is reliable (Spearman–Brown 0.995) and does not manufacture
  structure (P4, ~20× separation from inert circuits).
- S-inhibition and negative name movers do not separate under a complementation test, in
  1000/1000 bootstrap draws under the registered labelling, and as a point estimate across
  two ablation primitives, three circuits from three discovery methods, and three linkage
  rules. Under the expanded labelling the bootstrap merge rate falls to 81--86\% on the
  c6-zero conditions and stays at 100\% on the rest.
- Name movers **do** separate from S-inhibition under the same test, and the sign of the
  interaction differs between the two cases.
- Both negative name movers invert sign conditional on S-inhibition removal, in 1000/1000
  draws. Name movers do not.
- The clustering tracks role rather than depth.

**Not established.**
- One task, one model, one circuit family. GPT-2 small, IOI.
- The merge rests on five heads; the sign-flip result on two.
- Whether the sign inversion generalises to other inhibitory-head pairings anywhere.
- P1 is uninformative rather than negative — six within-pairs.
- P2 (cross-task cluster stability) deferred, not tested.
- The §5 ARI values used a post-hoc expanded label set and are not comparable to P3's.
- §7 is a follow-up, not a registered prediction.

---

## 9. Files

```
scripts/v13_complementation.py         exact WHT, reliability gate
scripts/v13_test.py                    P1, P3 first pass (ARI-selected k — superseded)
scripts/v13_p3_fix.py                  P3 with registered silhouette selection
scripts/v13_p4_control.py              P4 control (superseded z-score method)
scripts/v13_bootstrap.py               P4 via RTI bootstrap-over-prompts   [house method]
scripts/v13_ari_boot.py                ARI CI + merge-stability bootstrap  [house method]
scripts/v13_conditions.py              multi-condition, bootstrapped
scripts/v13_cond_point.py              multi-condition point estimates
scripts/v13_pairwise.py                per-class-pair co-clustering across conditions
scripts/v13_conditional_marginal.py    conditional marginal contribution + sign-flip test

results/v13_coeffs_c6_zero.json        order-1 and order-2 coefficients
results/v13_M_c6_zero.npy              masking matrix
results/v13_p3_silhouette.json         k / silhouette / ARI
results/v13_bootstrap_p4.json          P4 epistasis + order-2 CIs
results/v13_ari_bootstrap.json         ARI CI, k distribution, merge rate
results/v13_conditions_point.json      six-condition table
results/v13_pairwise.json              pair co-clustering by condition
results/v13_conditional_marginal.json  conditional marginals + flip rates

data/c6_zero_v2_FRESH.npz              primary (sha e6bc023e…)
data/c6_mean.npz  c2_zero.npz  c5_zero.npz        robustness conditions
data/gender_known_zero.npz  gender_random_zero.npz    P4 controls
```

Coalition tables are gitignored — download from the Modal volumes or regenerate; see
the README. The Wang et al. PDF, its text extraction, and the Figure 2 page renders are
also gitignored as third-party reference material. Retrieve from arXiv 2211.00593 and
extract with `pdftotext -layout` to reproduce the role assignment in §6; the figure
claims in §6 and §7 need the rendered pages rather than the extracted text, since the
extraction loses the grouping and arrow directions the argument turns on.

---

## 10. Reading for next time

Four promising findings deflated on contact with a source this session, and in every case
the source was retrievable in minutes. The working prior should be that any "nobody has
measured this" intuition about a mature literature is wrong until a source says otherwise.
The reverse also happened once: reading the source closely turned a deflated finding into a
sharper one, because the paper's own figure ruled out the explanation that had deflated it.
