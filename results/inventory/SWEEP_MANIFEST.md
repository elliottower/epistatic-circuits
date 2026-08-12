# Sweep manifest — what can be compared with what

Generated from `scripts/modal_inventory_headsets.py`, which reads `circuit_heads`
and metadata from every sweep NPZ and groups by identical head set. It reads no
value data, so it cannot unblind anything.

**Why this exists.** Each coalition sweep was written as a standalone script, so
prompt count and metric storage were per-script decisions. Nothing enforced
consistency, because nothing was comparing across sweeps when they were written.
A sweep costs roughly ten hours. This file makes the parameters visible before
anyone spends that time assuming two sweeps are comparable.

## Comparability rule

Two sweeps can be compared across ablation primitives only if they share: the
same head set, the same task, the same prompt count, and an equivalent metric.
Head set alone is not sufficient — random circuits generated with the same seed
have identical heads across tasks, and those are different experiments.

## Status

| task | head set | arms | prompts | metric | comparable |
|---|---|---|---|---|---|
| ioi/rti | 15h | 3 (mean,resample,zero) | [302, 512] | logit_diff/other | mixes tasks |
| ioi | 15h | 3 (mean,resample,zero) | [512] | logit_diff/other | **yes** |
| rti | 15h | 3 (mean,resample,zero) | [302, 512] | logit_diff | prompt mismatch [302, 512] |
| rti | 15h | 3 (mean,resample,zero) | [302, 512] | logit_diff/other | prompt mismatch [302, 512] |
| rti | 15h | 3 (mean,resample,zero) | [302, 512] | logit_diff/other | prompt mismatch [302, 512] |
| gt | 7h | 3 (mean,resample,zero) | [1000] | other/prob_diff | **yes** |
| gt | 7h | 3 (mean,resample,zero) | [1000] | prob_diff | **yes** |
| gt | 7h | 3 (mean,resample,zero) | [1000] | prob_diff | **yes** |
| gt | 7h | 3 (mean,resample,zero) | [1000] | prob_diff | **yes** |
| gt/induction | 7h | 3 (mean,resample,zero) | [500, 1000] | other/prob_diff | mixes tasks |
| rti | 15h | 2 (mean,resample) | [302, 512] | logit_diff | prompt mismatch [302, 512] |
| gt | 7h | 2 (mean,zero) | [1000] | prob_diff | **yes** |
| gt | 7h | 2 (mean,zero) | [1000] | other | **yes** |
| induction | 7h | 2 (mean,zero) | [500] | other | **yes** |
| induction | 7h | 2 (mean,zero) | [500] | other | **yes** |
| induction | 7h | 2 (mean,zero) | [500] | other | **yes** |
| induction | 7h | 2 (mean,zero) | [500] | other | **yes** |

## Notes

- `logit_diff` and `target_logits` minus `foil_logits` are the **same quantity**,
  stored differently. The resample sweeps compute `correct_logits -
  incorrect_logits` and save it as `logit_diff`; the c6 sweep saves the two
  logit arrays separately. A differing `value_key` is therefore not on its own a
  comparability failure.
- RTI's zero and mean sweeps ran at **302 prompts** while its resample sweeps ran
  at **512**. That blocks the RTI cross-primitive comparison and nothing else —
  those sweeps remain valid for any analysis within a single primitive.
- GT and induction store `prob_diff`; IOI and RTI store logit differences. Do not
  pool across those without a stated rescaling.

## For future sweeps

Prompt count, prompt set and metric should come from one shared config rather
than being set per script, and every new sweep should be added here.
