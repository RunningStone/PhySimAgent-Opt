# Experiments

The inherited physical environment is `batch1/aero2d`: deterministic mesh generation, OpenFOAM execution, post-processing and reference values. `batch1/common/code/run_comparison.sh` checks and runs the inherited wing fixed/agent configuration pair.

Use `CONFIGs/interview` for the small initial checks. Original sweep/multi-seed scripts are retained as optional wing experiments and may run many cases; they are not part of setup.

`level_spec_v1/run_levels.sh --output-dir OUTPUTs/<unique-run>/suite` runs the complete Level 0–3 interview suite from `CONFIGs/level_spec_v1/suite.yaml`. It uses the ordinary-airfoil `aero2d.interview` adapter, isolated live LLM calls and actual steady/transient OpenFOAM solves. Add `--resume` to continue the same frozen run.
