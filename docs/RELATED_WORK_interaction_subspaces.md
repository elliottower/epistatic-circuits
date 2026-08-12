# Related work: interactions as vectors and as subspaces

What the literature already does with second-order interactions in neural networks, and what it
does not. Written to scope a proposed experiment that keeps a pairwise interaction as a vector in
the residual stream and reads its subspace.

Every quotation below was checked against the extracted text of the source, not against a
summary. Sources are in `reference/`; a search corpus of 88 PDFs is in
`../mechanistic-views-NEW/reference/interaction-subspaces/`.

## The construction

For components `A` and `B`, the second-order inclusion–exclusion contrast

    delta_AB = resid(clean) - resid(A ablated) - resid(B ablated) + resid(both ablated)

is the order-2 Walsh/Möbius coefficient with a **vector** codomain rather than a scalar one.
Stacking over prompts and taking an SVD would give the directions the interaction occupies.

## The near-collision, and how near

**Abel Jansma, "A Compositional Calculus for Semantic Synergy in Language Model Embeddings,"
ICML 2026 workshops on Mechanistic Interpretability and on Compositional Learning.** OpenReview
`bGMPERlw5f`. Local copy: `reference/jansma2026_compositional_calculus.pdf`.

He introduces *semantic synergy*, "a training-free measure of non-compositional representation in
language models, obtained by taking the discrete derivative of a phrase embedding over its
sub-phrases." His reproduction repository states the operation directly:

> For a phrase `w1 ... wn`, the span-Möbius residual is
> `q(w1...wn) = T(w1...wn) - T(w1...w(n-1)) - T(w2...wn) + T(w2...w(n-1))`

implemented as `"mobius": full - prefix - suffix + middle`. The value function returns an
embedding and the residual is kept as a vector.

**So the vector-valued order-2 Möbius coefficient of a language model is published.** It cannot
be claimed as new. The formalism it rests on is his own earlier theorem with Forré
(arXiv:2510.05786), which generalizes Möbius inversion to functions valued in any abelian group;
his blog records that he proved it *in order to* compute this.

### What he does not do

Checked by grepping the paper's extracted text:

| term | occurrences in the paper |
|---|---|
| `svd` | 0 |
| `singular value` | 0 |
| `eigen` | 0 |
| `subspace` | 0 |
| `principal angle` | 0 |
| `rank` | 0 |
| `basis` | 0 |

`span` appears 25 times and every use is *text* span — "span mereology", "span synergy", "spans
overlap" — never linear span. The aggregation across examples is a **mean**:
`l2norm(q_train.mean(axis=0))`, benchmarked against a difference-in-means baseline and a random
direction.

**He collapses the collection of interaction vectors to a single steering direction. The geometry
of the collection is untouched.**

### A correction to an earlier reading of this paper

An initial pass over his reproduction *notebook* concluded there was no layer axis, because the
notebook uses a sentence-transformer with one pooled output per string. **The paper has one.** It
contains 52 mentions of "layer" and a figure titled "Layer-wise emergence of span-Möbius phrase
synergy" across three models — Qwen3-Embedding-0.6B, Qwen3-0.6B LM, and all-mpnet-base-v2.

The layer profile of a vector-valued Möbius residual is therefore also published, and is not
available as a contribution. The notebook is not the paper, and reasoning from a reproduction
repository to what a paper does is unsound in exactly this way.

## The rest of the ledger

| ingredient | status | source |
|---|---|---|
| order-2 Möbius kept as a vector | **done** | Jansma, ICML 2026 workshop (embeddings) |
| vector-valued Möbius formalism | **done** | Forré & Jansma, arXiv:2510.05786, Def. 3.1 — codomain any abelian group |
| the same algebra over *component ablations* | **done, scalar** | Vaidyanathan et al., arXiv:2606.27510 — output metric is logit difference |
| order-2 residual then normed | **done** | Singhvi et al., arXiv:2403.13106 — computes the vector, then takes ‖·‖₂ |
| mixed second derivative between two components | **done, normed** | Bolshim & Kugaevskikh, arXiv:2604.11639 — `∂²L/∂f_v∂f_w ∈ R^(d_v×d_w)`, reduced to Frobenius norm and stable rank |
| layer profile of the interaction | **done** | Jansma, ICML 2026 workshop |
| **stack over prompts, SVD, read the subspace** | **open** | — |
| **players are internal components, not input spans** | **open** | — |
| **alignment with OV write directions** | **open** | machinery in Merullo et al. arXiv:2406.09519, Yamagiwa et al. arXiv:2601.10266 |

## What remains, stated narrowly

Three separations survive, and only three:

1. **The object is the subspace spanned by the collection**, not its mean. Jansma tested the mean
   and it worked for steering, so this needs an argument rather than an assertion: a spectrum with
   more than one significant singular value, ideally with the second direction doing causal work
   the first does not.
2. **The players are internal components** — heads, MLPs — rather than input text spans. His
   `T(w1 w2)` re-encodes a shorter string; it does not ablate anything inside the model.
3. **Weight-space grounding.** Whether the interaction direction aligns with either head's OV
   write direction is not askable in a setup with no heads.

## Why this is worth doing anyway

E10 in this repository is the motivation and it is already frozen and run. It asked whether
pairwise subspace *relationships* predict the scalar Walsh coefficient, and the answer was no
twice — weight-only geometry at CV R² = −0.059, data-dependent subspace features at −0.082 —
while a positive control reached +0.199 through the identical pipeline. Its recorded verdict:
"Interaction is not decomposable into pairwise subspace relationships."

That null is consistent with an interaction that lives in *its own* subspace, aligned with neither
component's geometry. Such an interaction would produce exactly E10's result and would be found by
an SVD of `delta_AB`. E10 does not discourage the subspace approach; it is the reason to try it,
and it supplies a control design already shown to detect signal, plus the Walsh scalars that
`delta_AB` must reproduce when projected onto the logit direction — a free correctness check.

## Cautions

- **Cite Jansma twice**: arXiv:2510.05786 for the theorem, the ICML workshop paper for the first
  vector-valued application. Frame any contribution as the internal-component, subspace-resolved
  extension, and do it in the related-work section rather than leaving it to a referee.
- **Pre-empt the subspace-illusion objection** (Makelov, Lange & Nanda, arXiv:2311.17030) if the
  subspace is validated by patching it.
- **Provenance.** Several 2604–2608 preprints in the search corpus are unrefereed. Load-bearing
  claims should rest on Elhage et al. 2021, Merullo et al. arXiv:2406.09519, Kumar et al. NeurIPS
  2021, Sundararajan et al. ICML 2020, Tsai et al. JMLR 2023, and Makelov et al. ICLR 2024.
- **Absence claims in this project have been wrong repeatedly.** The zero-hit table above is a
  term count over one paper's text and is corroboration, not proof; it is reported alongside the
  reading that produced it.
