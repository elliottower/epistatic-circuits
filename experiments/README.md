# Experiments

One folder per experiment, named `E<N>_<short_name>`. Each folder holds:

- `PREREG.md` — predictions and analysis plan, frozen before results
- `results/` — outputs, when they exist

Pre-registrations were originally scattered across `docs/PREREG_*.md` and
`preregistrations/prereg_v*.md`. They were consolidated here for navigability.
Each PREREG.md carries a provenance header with the original path and commit hash.
The git history is the freeze record; `git show <hash>:<path>` recovers the original.

## Index

| directory | question | original file |
|---|---|---|
| `E1_sparse_walsh_recovery` | Can compressed sensing recover pairwise Walsh interactions from O(k log N) samples? | `docs/PREREG_SPARSE_WALSH_RECOVERY.md` |
| `E2_path_patching` | Are Walsh interactions direct causal edges or mediated couplings? | `docs/PREREG_PATH_PATCHING_EDGES.md` |
| `E3_subspace_epistasis` | Does residual-stream subspace overlap predict head-pair epistasis? | `docs/PREREG_SUBSPACE_EPISTASIS.md` |
| `E4_mib_faithfulness` | Can sparse Walsh recovery discover circuits competitive with EAP? | `docs/PREREG_MIB_FAITHFULNESS.md` |
| `E5_mib_circuit_benchmark` | Do Walsh/PP circuits achieve competitive CPR/CMD on MIB? | `docs/PREREG_MIB_CIRCUIT_BENCHMARK.md` |
| `E6_weight_geometry` | Does weight geometry predict epistasis under some ablation primitives better than others? | `docs/PREREG_PRIMITIVE_SPECIFIC_GEOMETRY.md` |
| `E7_selective_pressures` | Do different circuit discovery methods impose detectable fingerprints on Walsh spectra? | `preregistrations/prereg_v12_selective_pressures.md` |
| `E8_multi_task_extension` | Does the selective-pressure pattern generalize beyond IOI to four additional tasks? | `preregistrations/prereg_v12c_multi_task_extension.md` |
| `E9_complementation` | Do the field's head role labels survive a complementation test? | `preregistrations/prereg_v13_complementation_units.md` |
| `E10_subspace_epistasis_data_dependent` | Does data-dependent subspace overlap predict head-pair epistasis? | — (new) |
| `EXPT11_interaction_subspace_geometry` | Where in the residual stream does head-pair interaction live? | — |
| `EXPT12_suppression` | Does amplifying one head suppress the deficit from ablating another? | — |
