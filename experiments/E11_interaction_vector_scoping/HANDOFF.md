# Handoff — E11 interaction-vector code

Read this before touching `modal_run_e11_gate0.py` or writing the EXPT11 runner. Four bugs
are already fixed here and one null was already replaced; reintroducing any of them produces
a run that completes, prints a verdict, and is wrong.

## What the script computes

For each head pair, over 200 IOI prompts, at every layer:

    delta_AB^(l) = r_clean - r_{A-abl} - r_{B-abl} + r_{AB-abl}

the order-2 Walsh coefficient kept as a residual-stream vector instead of collapsed to a
logit difference. Reports per pair: normalized layer profile and peak layer, mean-cancellation
ratio, participation ratio against a null, sigma_2/sigma_1, and 90%-energy rank.

Modal, T4, about ten minutes for 12 pairs.

## Four bugs already fixed — do not reintroduce

**1. `run_with_cache` does not accept `fwd_hooks`.** Checked against the installed
TransformerLens source; its signature is `(*model_args, names_filter, device,
remove_batch_dim, incl_bwd, reset_hooks_end, clear_contexts, pos_slice, **model_kwargs)`.
Passing `fwd_hooks` sends it into `**model_kwargs`, the ablation silently never applies, and
every delta comes out identically zero — while the script completes normally. Use
`with model.hooks(fwd_hooks=hooks):` around the call, or `run_with_hooks` if no cache is
needed.

**2. `hook_z` lives at `blocks.{l}.attn.hook_z`**, not `blocks.{l}.hook_z`. Wrong path on the
cache lookup raises `KeyError`, which is the harmless version. The same mistake in a hook
*name* installs a hook that never fires, silently.

**3. The final-token index cannot come from `pad_token_id`.** GPT-2's pad token and BOS token
are both `<|endoftext|>`, so `(tokens != pad_token_id).sum(dim=1) - 1` counts the prepended
BOS as padding and lands one position early. Tokenize each prompt individually and use its
real length.

**4. A silent hook failure must not be able to look like the interesting answer.** If hooks do
not fire, `delta = R - R - R + R = 0` exactly. The pre-causal zero check then reads 0 and
*passes*; the mean-cancellation ratio reads 0 and lands in the "proceed" band; the script
recommends proceeding on a study whose input was entirely zeros. The script now asserts every
single-head ablation moves the residual stream by more than 1e-6 at or after its own layer,
and raises otherwise. **Keep this gate in any derived script.**

## The null that was replaced, and why

The first version sign-shuffled rows of `delta` as a participation-ratio null. That is a
no-op: `delta^T delta = sum_i d_i d_i^T` and `(-d_i)(-d_i)^T = d_i d_i^T`, so row sign flips
leave every singular value unchanged. The null equalled the observed value in all 12 pairs,
exactly, and `pr_exceeds_null` was meaningless.

    d = rng.standard_normal((40, 128)); signs = rng.choice([-1., 1.], size=(40, 1))
    pr(d) == pr(d * signs)      # True, exactly

Replaced with **matched-norm random directions at identical N and d**: each row is a random
direction scaled to that row's observed norm. This destroys the covariance structure while
holding the per-row norms and the aspect ratio fixed — participation ratio is heavily biased
when N << d, and identical N is what makes the bias cancel in the comparison rather than
needing to be estimated away.

Structured data sits **below** this null, so the field to read is `pr_below_null`.

The sign-shuffle is still used for the mean-cancellation null, where it is correct: flipping
signs changes the mean while leaving row norms alone, which is exactly what that statistic is
about.

## What the run found, 12 of 190 pairs

| | PR observed | PR null | MCR | 90%-energy rank | peak layer |
|---|---|---|---|---|---|
| sub-additive (w < 0) | 9.5 – 13.8 | 48 – 124 | 0.37 – 0.49 | 14 – 26 | 10, 11 |
| super-additive (w > 0) | 1.2 – 6.6 | 93 – 141 | 0.19 – 0.55 | 1 – 13 | 7, 8, 11 |

Causal-path check exactly 0.000e+00. All 12 pairs below the random null, so the interaction
is strongly structured in every case. Median MCR 0.402 — about 60% of typical delta magnitude
cancels in the mean, so it is not a single direction.

The sign split is the interesting part and it was not predicted: masking interactions occupy
roughly ten to fourteen directions, synergistic ones are close to rank-1.

## Issues in `EXPT11_interaction_subspace_geometry/PREREG.md`

Read from the document itself, not inferred from the run:

- **H5 asks whether the subspace is "diffuse and high-rank" without naming a null.** Effective
  dimensionality has no meaning without one. Use the matched-norm random null above; do not
  use a sign shuffle.
- **Participation ratio is biased at 200 prompts against d_model 768.** The fix is a null at
  identical N, not a bias correction.
- **Rank selection needs a stated rule** if any hypothesis depends on a top-k subspace.
  Cumulative-energy thresholding is standard; an elbow heuristic is not.
- **H4b is a correctness check, not a hypothesis**, and reads better as a gate: layers before
  `min(layer_A, layer_B)` have no causal path, so delta must be exactly zero there. If it is
  not, nothing else in the run is readable.
- **H2 compares sub-additive against super-additive pairs by subspace sharing — watch the
  confound.** Pairs with `|w|` near zero have a delta that is mostly noise, and random
  subspaces in 768 dimensions are near-orthogonal almost surely, so a difference in sharing
  can be manufactured by one group simply having a weaker signal. Compare at matched `|w|`,
  or restrict to pairs above a magnitude threshold fixed in advance.
- The prereg's Foreknowledge section now records the scoping run and carves the 12 scoped
  pairs out as exploratory, stating H1 and H5 as confirmatory over the remaining 178.


## Forking path that was in the first version, now fixed

`peak_layer` is chosen per pair by argmax of the normalized layer profile, and the first
version computed every downstream statistic *at that layer only*. For a scoping run that is
harmless; in a pre-registered analysis it is a forking path, because each pair gets measured
wherever its own data put the maximum.

Fixed by computing the mean-cancellation ratio and the participation ratio at **every** layer
and storing both profiles. The peak is now one summary among many rather than the thing all
inference rests on. The null is still computed at the peak layer only, for cost — 200 draws
across 12 layers and 190 pairs is 456,000 SVDs.

EXPT11 must still say which layer its hypotheses are evaluated at, and say it in advance:
a fixed layer, all layers with a correction, or the argmax rule named explicitly as the
pre-registered choice. Any of the three is defensible; leaving it unstated is not.

## Checks already run against phase 2 — do not redo

- **Ablation matches.** Phase 2 uses `z[:, :, head, :].mean(dim=0)` broadcast over the batch;
  this script does the same. Same operation, so delta sits on the same lattice as the scalar
  Walsh coefficients.
- **Prompts are identical.** Phase 2 *generates* its 200 prompts with `default_rng(42)` rather
  than reading `data/ioi_prompts_200.json`. Regenerating them from the name, place and object
  lists in `scripts/modal_sparse_walsh_phase2.py` reproduces the saved file **exactly, all
  200**. The mean ablation is therefore over the same distribution.
- **Möbius signs**: `f(0) - f({A}) - f({B}) + f({A,B})`, confirmed against the code.
- **Bounds**: MCR stayed in [0.19, 0.55] against a Jensen bound of 1; PR stayed in
  [1.22, 13.81] against a cap of min(N, d) = 200.
- **Peak layer is not an artifact of depth.** Profiles are non-monotone after the later head
  and peak values run roughly double the final-layer values, so the localization is real
  rather than delta simply accumulating.

## Order of operations

This ran before anyone checked whether a pre-registration already covered the same
quantities. It did. Check first next time — the fix cost one paragraph because the prereg was
DRAFT rather than frozen, and would have cost the experiment otherwise.
