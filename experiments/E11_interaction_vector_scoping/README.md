# E11 Gate 0 — scoping calculation, not an experiment

**No pre-registration, deliberately.** This decides whether a subspace study is worth
designing; it carries no hypotheses and licenses no inference. Its output is foreknowledge
for `../EXPT11_interaction_subspace_geometry/`, and that document's Foreknowledge section
records exactly what was seen here.

**Order of operations was wrong.** This ran before `EXPT11_interaction_subspace_geometry/PREREG.md`
was consulted, and it measured quantities that document treats as predictions — its
pre-layer zero check (H4b), its layer profile (H1), and its rank question (H5). That prereg
was DRAFT rather than frozen, so nothing sealed was contaminated, and the consequences are
declared there rather than worked around. Check for an existing prereg before running a
scoping calculation next time.

## What it found, on 12 of 190 pairs over 200 prompts

- Causal-path check exact: max |delta| before either head's layer = 0.000e+00.
- Median mean-cancellation ratio 0.402 — about 60% of the typical delta cancels in the mean,
  so the interaction is not a single direction.
- Participation ratio splits by interaction sign: 9.5–13.8 for sub-additive (masking) pairs
  against 1.2–6.6 for super-additive (synergistic) ones. All 12 sit far below a matched-norm
  random null of 48–141, so the interaction is strongly structured in every case; the split
  is about how many directions that structure occupies.

## The null was broken once and is fixed

The first version sign-shuffled rows of delta for the participation-ratio null. That is a
no-op — row sign flips leave `delta^T delta` and hence every singular value unchanged — so
the null equalled the observed value by construction. Replaced with matched-norm random
directions at identical N and d. Structured data sits *below* that null; read
`pr_below_null`. The sign shuffle is still used for the mean-cancellation null, where it is
correct.

See `HANDOFF.md` for the four wiring bugs already fixed here and the open issues in the
EXPT11 pre-registration.
