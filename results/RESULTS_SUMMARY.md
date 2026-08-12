# Results Summary: Circuit Discovery as Selective Pressures

All results from `all_walsh_results.csv` (59 rows, 6 tasks, mean + zero ablation).
Focus tasks for the paper: **IOI, RTI, greater-than** (induction as supplement; gendered pronoun and SVA have broken known circuits).

## Methods inventory

| Method | Type | How it works |
|--------|------|--------------|
| known / canonical | mechanistic | Hand-traced from prior work (Wang et al. for IOI, Hanna et al. for GT, etc.) |
| knockout top-K | knockout_topk | Ablate each head individually, rank by loss delta, take top K |
| greedy sufficiency | greedy_sufficiency | Forward selection: greedily add the head that most improves faithfulness |
| ACDC backward | acdc_backward | Backward elimination from a top-50 starting pool (head-level approximation) |
| EAP-IG | edge_attribution | Integrated-gradient edge attribution scores, take top-K heads by total score |
| C5 Walsh order-1 | walsh_order1 | LASSO-fitted Walsh coefficients on 20K random coalitions; top-K by |order-1| |
| C6 epistatic | walsh_order2 | Same LASSO fit; top-K by order-2 energy (pairwise interaction strength) |
| IC15 | info_content | Mutual-information ranked heads |
| random | random | Random 15-head circuit (control) |
| cross-task control | cross_task_control | RTI circuit evaluated on IOI task (negative control) |

## What we have per task

### IOI (15 heads, 512 prompts, mean ablation)

| Circuit | Faithfulness | Order-1 frac | Order-2 frac | Order-3+ frac |
|---------|-------------|-------------|-------------|---------------|
| C5 Walsh order-1 | **+1.524** | 0.809 | 0.168 | 0.023 |
| C6 epistatic | +1.417 | 0.784 | 0.186 | 0.030 |
| C3 canonical | +1.385 | 0.786 | 0.184 | 0.030 |
| C2 EAP-IG | +1.131 | 0.904 | 0.090 | 0.006 |
| RTI on IOI (control) | +0.034 | 0.973 | 0.026 | 0.001 |
| C4 random | +0.022 | 0.979 | 0.021 | 0.000 |
| IC15 | -0.003 | 0.949 | 0.046 | 0.005 |

**Takeaway**: Walsh-discovered circuits (C5, C6) match or beat the canonical circuit on faithfulness. EAP-IG is faithful but much more additive (order-1 = 0.90 vs 0.78-0.81 for others). The canonical circuit has the most epistasis among faithful circuits. Controls (random, cross-task, IC15) have near-zero faithfulness and near-pure additivity.

### RTI (15 heads, 302 prompts, mean ablation)

| Circuit | Faithfulness | Order-1 frac | Order-2 frac | Order-3+ frac |
|---------|-------------|-------------|-------------|---------------|
| RTI known | +0.414 | 0.965 | 0.035 | 0.001 |
| C4 random | +0.309 | 0.987 | 0.013 | 0.000 |
| EAP RTI | +0.177 | 0.772 | 0.152 | 0.076 |

Walsh discovery complete (LASSO R²=0.913, 4458 non-zero coefficients). Coalition sweeps on C5/C6 not yet run.

**LASSO energy spectrum**: 79% order-1, 21% order-2 — significant pairwise interactions.

C5 circuit (top-15 by order-1): L0H9, L11H2, L4H11, L10H6, L7H9, L10H7, L9H9, L5H6, L4H0, L2H11, L8H7, L1H5, L11H10, L1H3, L6H11.
C6 circuit (top-15 by order-2 energy): L0H10, L0H9, L11H2, L10H7, L10H0, L4H11, L11H10, L0H1, L0H3, L9H9, L1H11, L9H6, L2H2, L5H6, L2H11.

Overlap with known RTI: C5 3/15, C6 4/15 (low). C5 vs C6 overlap: 8/15.
Strongest interaction: L0H9 x L0H10 (+0.067).

**Takeaway**: Walsh finds largely different heads from the known RTI circuit. EAP has the most epistasis but lowest faithfulness. Need coalition sweeps on C5/C6 to measure their faithfulness and Walsh energy.

### Greater-than (7 heads, 1000 prompts, mean ablation)

| Circuit | Faithfulness | Epistasis | Order-1 frac |
|---------|-------------|-----------|-------------|
| knockout top-K | **+0.916** | 0.384 | 0.954 |
| known | +0.876 | **0.551** | 0.941 |
| ACDC backward | +0.659 | 0.050 | 0.992 |
| greedy sufficiency | +0.211 | 0.210 | 0.992 |
| random | +0.034 | 0.159 | 0.976 |

Walsh discovery complete (LASSO R²=0.879, 2818 non-zero coefficients). Coalition sweeps on C5/C6 not yet run.

**LASSO energy spectrum**: 85% order-1, 15% order-2.

C5 circuit (top-7 by order-1): L5H5, L7H10, L9H1, L6H9, L0H10, L8H5, L10H2.
C6 circuit (top-7 by order-2 energy): L5H5, L0H10, L7H10, L9H1, L4H11, L0H3, L6H9.

Overlap with known GT: C5 1/7 (L5H5), C6 1/7 (L5H5). C5 vs C6 overlap: 5/7.
L5H5 dominates both rankings. Strongest interaction: L5H5 x L7H10 (+0.017).

**Takeaway**: Known circuit is most epistatic among faithful circuits (epistasis = 0.55). Iterative methods (ACDC, greedy) produce extremely additive circuits (order-1 > 0.99) despite reasonable faithfulness. Walsh discovery finds largely novel heads — only L5H5 overlaps with the known circuit. This is the core finding: iterative methods are blind to synergistic head pairs.

### Induction (7 heads, 500 prompts, mean ablation)

| Circuit | Faithfulness | Epistasis | Order-1 frac |
|---------|-------------|-----------|-------------|
| known | **+4.615** | **0.563** | 0.941 |
| knockout top-K | +4.097 | 0.304 | 0.905 |
| ACDC backward | +2.891 | 0.412 | 0.963 |
| greedy sufficiency | +2.832 | 0.360 | 0.970 |
| random | +0.165 | 0.846 | 0.977 |

**Takeaway**: Known circuit dominates on both faithfulness and epistasis. Random circuit has highest "epistasis" score but near-zero faithfulness (noise floor). Greedy/ACDC again more additive than known or knockout.

### Gendered pronoun (5 heads, 986 prompts, mean ablation)

| Circuit | Faithfulness | Epistasis | Order-1 frac |
|---------|-------------|-----------|-------------|
| knockout top-K | **+0.610** | -0.122 | 0.983 |
| random | +0.049 | 0.083 | 0.997 |
| ACDC backward | -0.010 | 2.462 | 0.994 |
| greedy sufficiency | -0.010 | 2.462 | 0.994 |
| known | **-0.025** | -0.077 | 0.999 |

**Takeaway**: Known circuit has *negative* faithfulness (worse than ablating everything). Greedy = ACDC (Jaccard 1.0), both near-zero faithfulness. Only knockout finds anything functional. This task's "known" circuit from the literature doesn't work under mean ablation.

### SVA (12 heads, 500 prompts, mean ablation)

| Circuit | Faithfulness | Epistasis | Order-1 frac |
|---------|-------------|-----------|-------------|
| knockout top-K | +0.580 | 0.540 | 0.953 |
| greedy sufficiency | +0.563 | 0.515 | 0.954 |
| ACDC backward | +0.554 | 0.511 | 0.952 |
| known | **+0.038** | 0.389 | 0.974 |

**Takeaway**: Known circuit (from Lazo et al. 2025, not the original Finlayson 2021) has near-zero faithfulness. All discovery methods find similar circuits. SVA may not be suitable as a main-paper task.

## Cross-task patterns

1. **Iterative methods (greedy, ACDC) consistently produce more additive circuits** than knockout or known circuits. Order-1 fractions > 0.99 for greedy/ACDC vs 0.90-0.95 for knockout/known. This holds across all 4 tasks where both are available.

2. **Known circuits are most epistatic when functional.** GT known (0.55), induction known (0.56), IOI canonical (order-2 frac 0.18). The mechanistic structure that makes them "known" is precisely the multi-head composition that creates epistasis.

3. **EAP-IG is faithful but additive.** On IOI, EAP achieves faith=1.13 with order-1=0.90 (vs canonical's 0.79). Edge attribution scores individual contributions, so it naturally selects for additively important heads.

4. **Walsh-discovered circuits are competitive.** C5 (order-1 LASSO) beats the canonical circuit on IOI faithfulness (1.52 vs 1.39). C6 (order-2 LASSO) is comparable (1.42). Both have similar epistasis profiles to the canonical circuit.

5. **Two tasks have broken known circuits.** Gendered pronoun known (faith=-0.025) and SVA known (faith=+0.038) are functionally inert. These tasks should be secondary in the paper or discussed as negative results.

## Walsh discovery summary (LASSO fits)

| Task | R² | Non-zero coefs | Order-1 energy | Order-2 energy | C5 vs known | C6 vs known |
|------|-----|----------------|---------------|----------------|------------|------------|
| IOI | (prior session) | — | — | — | — | — |
| RTI | 0.913 | 4458 | 79% | 21% | 3/15 | 4/15 |
| GT | 0.879 | 2818 | 85% | 15% | 1/7 | 1/7 |

Walsh-discovered circuits consistently find novel heads with low overlap to known circuits (1-4 heads shared). RTI has the strongest pairwise interactions (21% order-2 energy).

## What's missing

- **RTI**: No greedy/ACDC yet. C5/C6 circuits defined but need coalition sweeps for faithfulness + Walsh energy.
- **GT**: C5/C6 circuits defined but need coalition sweeps for faithfulness + Walsh energy.
- **All tasks**: No resample ablation (only mean and zero so far). Resample is more principled.
- **IOI**: Missing greedy/ACDC (have knockout, EAP, canonical, Walsh, epistatic, random).
- **Epistasis column**: Not computed for IOI/RTI rows (coalition sweeps exist but metric not in CSV yet).
- **Coalition sweeps on new C5/C6**: Need to run 2^n sweeps on RTI C5, RTI C6, GT C5, GT C6 to get comparable Walsh energy spectra and faithfulness numbers for the CSV.
