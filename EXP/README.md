# Experiments

The only physical environment is `batch1/aero2d`: deterministic mesh generation, OpenFOAM execution, post-processing and reference values. `batch1/common/code/run_comparison.sh` checks and runs the inherited wing fixed/agent configuration pair.

Use `CONFIGs/interview` for the small initial checks. Original sweep/multi-seed scripts are retained as optional wing experiments and may run many cases; they are not part of setup.
