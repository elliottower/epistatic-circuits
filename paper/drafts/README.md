# Superseded drafts

Not built. `main.tex` is the paper; nothing here is input by it.

Kept because each was a real piece of work and one of them contains an argument the paper
does not yet make.

| file | what it is |
|---|---|
| `sec_abstract.tex` | A second abstract that diverged from the live one. Never compiled — `main.tex` writes its abstract inline, so this file was edited for some time while having no effect on the PDF. |
| `sec_auditing_proxies.tex` | A wrapper section. Its framing paragraph **has been recovered** into `main.tex` as the "Auditing circuit discovery proxies" section, rewritten to say the audit runs on the exhaustive coefficients rather than the compressed reconstructions. |
| `sec_construct_validity_metrics.tex` | Comments only, no prose. The argument it sketches is not in the paper: several operationalizations of epistasis converge when faithfulness is healthy and diverge systematically when it is not, so the convergence pattern itself validates the construct. Worth writing or worth dropping deliberately. |
| `sec_mib_benchmark_v1.tex` | Superseded by the version now in `main.tex`. |
| `sec_weight_geometry_v1.tex` | Superseded by the version now in `main.tex`. |

The lesson worth keeping: a `.tex` file in this directory that `main.tex` does not `\input`
is invisible, and it will be edited anyway. Check `grep -c 'input{' main.tex` against the
file count before assuming an edit reached the PDF.
