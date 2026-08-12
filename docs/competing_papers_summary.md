# Competing / Related Papers Summary

## Threat papers (none are direct competitors)

### QHA — Higher-Order Token Interactions via Quantum Attention (2025)
Uses epistasis as an *application domain* (genetic epistasis prediction).
Architecture paper for biology, not circuit analysis. No overlap with our work.

### Hadamard Attention (2025)
Replaces attention output projection with WHT for computational efficiency.
Architecture optimization, not interpretability. No overlap.

### Causal Head Gating (Chen et al., NeurIPS 2025)
Learnable gates + Shapley values to identify head interactions. Reports "low
modularity" — individual head roles depend on interactions with others.
**Convergent evidence** — they identify the problem (interactions matter for
circuit evaluation), we provide a principled measurement tool (Walsh).
Cite as motivation: `\citet{chen2025causalheadgating}`.

### Self-Attention Attribution (AAAI 2021)
Integrated gradients on attention weight matrices at token-token level within
individual heads. Different granularity from Walsh head-head interaction.
Related work only.

## MIB benchmark (Mueller et al., ICML 2025)

Key protocol details:
- f(C,N;m) = (m(C) - m(empty)) / (m(N) - m(empty))
- m = logit difference (correct answer vs counterfactual)
- CPR = area under faithfulness curve (higher = better)
- CMD = area between faithfulness curve and f=1 (lower = better)
- k values: {.001, .002, .005, .01, .02, .05, .1, .2, .5, 1}
- Best method: EAP-IG-inputs with CF ablations
- EAP (CF) on GPT-2 IOI: CMD = 0.03

## Chughtai et al. (ACL 2024) — Faithfulness metrics not robust

Key findings:
- Edge-level ablation gives substantially different results from node-level
- Different ablation types (CF, mean, resample) find different "ground truth" circuits
- Circuit performance is non-monotonic (adding heads can hurt)
- Ablation methodology defines the task

## Li et al. (BlackboxNLP 2025) — Better Edge Selection

Improves circuit construction from importance scores. Shows that greedy
search from logits outperforms top-n edges for some methods (esp. IFR).
Our Walsh method produces node-level scores; top-n is the natural
construction.

## Ensemble Circuit Localization (BlackboxNLP 2025)

Combines multiple methods for better circuits. Our Walsh+PP combination
is a form of ensembling. Their finding: simple rank-sum often matches
more complex fusion strategies.

## Our simplified MIB results (20-head Walsh data, all 144 heads)

| Method | CPR | CMD |
|---|---|---|
| Activation patching | 0.909 | 0.115 |
| Walsh interaction | 0.803 | 0.190 |
| Path patching | 0.794 | 0.199 |
| Walsh + PP combined | 0.791 | 0.203 |
| Walsh order-1 | 0.788 | 0.206 |
| Random | 0.474 | 0.520 |

Key: Walsh perfectly matches AP top-5 heads (Jaccard=1.0).
Walsh+PP complementarity vanishes at node level (head Spearman 0.997).
Full 144-head experiment running on Modal.
