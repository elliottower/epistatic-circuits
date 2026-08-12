# v13 results — complementation test on the IOI head taxonomy

Preregistered in `preregistrations/prereg_v13_complementation_units.md`,
frozen `FREEZE_v13.txt` (prereg sha `1f915bbc…`, amended `e258d592…`).

**One-line result: S-inhibition heads and negative name movers do not separate under a
complementation test — in 1000/1000 bootstrap resamples. How well the rest of the taxonomy
survives is uncertain (ARI 95% CI [0.125, 0.571]).**

> **Statistics use the RTI-paper method** — bootstrap over prompts, percentile CIs,
> matching `scripts/modal_rti_epistasis_bootstrap.py`. An earlier version of this file
> used a coalition-shuffled null and a z-score, which was invented for this analysis and
> is not the house method. Superseded; both are reported below for transparency.

---

## Gates, both passed

| gate | result |
|---|---|
| **Reliability** (prereg abort #2, amended to prompt-split) | split-half r = 0.989, Spearman-Brown = **0.995**. Pass (threshold 0.6). |
| **P4 inert-circuit control** (prereg: must run first, invalidates all else) | see below. Pass. |

### P4 — the pipeline does not invent structure

**House method (bootstrap over prompts, 1000 draws, percentile CIs):**

| circuit | n | RTI epistasis | 95% CI | order-2 frac | 95% CI |
|---|---|---|---|---|---|
| c6 IOI (functional) | 15 | −0.329 | [−0.432, −0.228] | **0.303** | [0.286, 0.319] |
| gender **known** (inert, faithfulness −0.025) | 5 | +0.213 | [+0.173, +0.248] | **0.011** | [0.006, 0.016] |
| gender random | 5 | +0.748 | [−2.68, +4.34] | 0.032 | [0.013, 0.069] |

Order-2 CIs are separated by a factor of ~20 with no overlap. The random circuit's
epistasis CI explodes because its group effect is near zero — the ratio is undefined in
the limit, which is itself a signature of an inert circuit.

*Superseded first pass, retained for transparency:* coalition-shuffled null, z = +641
(functional) vs −2.7 / −2.4 (inert). Same conclusion, non-house method.

Functionally inert circuits produce essentially no interaction structure. Structure appears
only where there is function to structure.

---

## Method note — sign convention, verified rather than assumed

The registered convention was that sub-additive masking indicates one unit. Since the
coalition encodes heads **kept** (v(∅) = −1.05, v(full) = +0.61), the mapping from Walsh
order-2 to interaction sign was checked empirically before use:

mean discrete interaction `E_S[v(S∪ij) − v(S∪i) − v(S∪j) + v(S)]` = **exactly 4×** the
Walsh order-2 coefficient, r = 1.0000 over 40 probe pairs. So masking = negative order-2,
and the masking matrix is `M = −w₂`. Exhaustive lattice (2¹⁵ = 32,768 coalitions), so the
transform is exact — no LASSO, no regularisation (amendment A2).

---

## Prediction 1 — within-role vs cross-role masking: NOT CONFIRMED

Within-role mean masking − cross-role = **+0.0028**, label-permutation p = **0.33**
(10,000 permutations, 9 heads, 6 within-pairs, 30 cross-pairs).

Registered prediction was "disconfirmed," and it is not confirmed — but **this test has no
power.** Six within-pairs cannot distinguish a null effect from a real one. Report as
uninformative, not as evidence of absence.

## Prediction 3 — ARI against the taxonomy: REGISTERED PREDICTION WRONG

Cluster count chosen by silhouette per the prereg. Point estimate k = 3, **ARI = +0.571**.

**Bootstrapped over prompts (1000 draws): ARI mean +0.410, 95% CI [+0.125, +0.571].**
The point estimate sits at the top of the bootstrap distribution. Cluster count is not
stable either — k = 3 in 334 draws, 4 in 400, 5 in 206, 6 in 60.

Registered prediction was ARI ≤ 0.1. The CI lower bound is 0.125, so the prediction is
rejected — but only just, and the "substantially recovered" reading (> 0.4) does **not**
survive resampling. Correct statement: the taxonomy tracks interaction structure better
than predicted, by an uncertain amount.

*Procedural note:* the first run selected k by maximising ARI, which is selection on the
outcome. Corrected to silhouette selection as registered. Same k, same value — but the
uncorrected version must not be reported as confirmatory.

### The partition, and where the taxonomy fails

| cluster | heads | roles |
|---|---|---|
| 2 | L5H5, L6H9 | IND, IND — **recovered exactly** |
| 3 | L9H9, L10H0 | NM, NM — **recovered exactly** |
| 1 | L7H9, L8H6, L8H10, L10H7, L11H10 | S-Inh ×3 + NegNM ×2 — **merged** |

Induction heads and name movers are recovered as clean units. **S-inhibition heads and
negative name movers do not separate under joint ablation.** Both are inhibitory — one
suppresses attention to the subject, the other suppresses the name's logit — and they mask
each other the way two hits on one pathway do.

That is a merge candidate derived from interaction structure rather than from naming.

**And unlike the global fit, it is stable: S-Inh co-clusters with NegNM in 1000/1000
bootstrap resamples over prompts.** The overall partition wobbles; this particular
non-separation does not.

### The layer confound is ruled out

Roles in IOI are layer-stratified, so the clustering could have been recovering depth.

| comparison | ARI |
|---|---|
| cluster vs **role** | **0.571** |
| cluster vs layer | 0.053 |
| cluster vs layer-band (early/mid/late) | 0.217 |
| role vs layer-band | 0.526 |

The clustering tracks role an order of magnitude better than raw layer, and more than
twice as well as layer-band — despite roles themselves being strongly band-aligned. It is
not depth.

---

## What this does not establish

- **One circuit, one ablation type.** c6 (order-2-discovered IOI), zero ablation. The
  mean-ablation sweep exists on the same volume and has not been run.
- **Nine labelled heads, four roles.** Small enough that the S-Inh/NegNM merge rests on
  five heads.
- **P2 deferred** (amendment A3) — cross-task cluster stability needs the c2/c5/gt sweeps
  pulled and their head overlap established. Not tested, not reported as null.
- **Role labels transcribed from Wang et al.'s figure**, not sourced from a file in these
  repos. `exemplar_heads.json` holds only one head per role, which is insufficient for the
  within-vs-cross test. **Fix before writing anything.**
- **P1 is uninformative**, not negative.

## Files

```
scripts/v13_complementation.py    exact WHT, reliability gate
scripts/v13_test.py               P1, P3 (first pass, ARI-selected k — superseded)
scripts/v13_p3_fix.py             P3 with prereg silhouette selection
scripts/v13_p4_control.py         P4 inert-circuit control
results/v13_coeffs_c6_zero.json   order-1 and order-2 coefficients
results/v13_M_c6_zero.npy         masking matrix
results/v13_p3_silhouette.json    k / silhouette / ARI table
results/v13_p4_control.json       control statistics (superseded z-score method)
scripts/v13_bootstrap.py          P4 via RTI bootstrap-over-prompts  [house method]
scripts/v13_ari_boot.py           ARI + merge-stability bootstrap    [house method]
results/v13_bootstrap_p4.json     P4 epistasis + order-2 CIs
results/v13_ari_bootstrap.json    ARI CI, k distribution, merge rate
```
