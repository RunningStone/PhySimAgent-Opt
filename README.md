# PhySimAgent-Opt

A runnable Level 0–3 experiment suite for the VerticalLab interview: an ordinary 2D airfoil, an LLM design loop, OpenFOAM simulation, formal evaluation and recoverable experiment records. The selectively inherited ground-effect wing workflow remains available as a legacy baseline.

Source: [PhySimAgent](https://github.com/RunningStone/PhySimAgent) at `f0f5b784fa092f878e9790107582c4c4501130d6`. The import inventory and migration scope are recorded in [ENV/inheritance.json](ENV/inheritance.json). This is a new project history.

## Scope

The Level 0–3 suite uses ordinary external airfoil flow at Re=100: M0 steady laminar `simpleFoam` and M1 transient laminar `pimpleFoam`. It covers parameterized design, TopK versus cold-start transfer, explicit shape changes, and autonomous exploration followed by formal confirmation. The inherited legacy case uses a moving ground boundary and a downforce objective; its physics and results are separate. Three-dimensional cases, turbulence upgrades and native MDO dependencies are outside the current frozen path.

The inherited baseline supports proposals, fixed/agent tool selection, budgets, provenance, trust gates, experiment lineage and final validation. Explicit hypothesis verdicts and adaptive evidence selection are future interview increments; they are not claimed by this import. The inherited `holdout.robust` flag means all holdout points have valid scores, not that performance is preserved.

## Setup

Supported baseline: macOS Apple Silicon, Python 3.12, uv, Git and OpenFOAM v2512. The LLM route uses an authenticated local Claude CLI; credentials are never copied into this repository.

```sh
# Only if OpenFOAM is not already installed:
brew bundle --file ENV/Brewfile
make -C ENV env
make -C ENV test-unit
make -C ENV test-smoke
```

`make env` restores the pinned AIDE source, applies its tracked patch, and installs the locked dependencies into this project's own `ENV/.venv`. It does not build an MDO stack. Keep `SRC`, `EXP`, `ENV` and `RELATED_REPOs` adjacent so editable package references remain valid.

## Run the complete Level 0–3 suite

After setup, run all six settings with real LLM and solver calls:

```sh
bash EXP/level_spec_v1/run_levels.sh --output-dir OUTPUTs/my-levels/suite
# Continue the same configuration and source snapshot; completed work is skipped:
bash EXP/level_spec_v1/run_levels.sh --output-dir OUTPUTs/my-levels/suite --resume
```

Use a new output directory for a fresh experiment. The [suite configuration](CONFIGs/level_spec_v1/suite.yaml) and [configuration guide](CONFIGs/level_spec_v1/README.md) freeze the physical models, candidate space, permissions, budgets and initialization. Results include `summary.json`, per-stage `records.json`, SQLite checkpoints, raw solver artifacts and LLM requests. Check `summary.json` status and every stage result; a successful process exit alone is not the acceptance criterion.

The checked-in preset is one small-budget smoke replicate: six settings, nine stage cells, seven independent stages. On 2026-09-24 it completed 23 real solver calls and 18 LLM calls; independent physical and complete-matrix audits passed, and completed-run resume added no calls. This verifies execution capability, not statistical superiority. For a different budget or repetitions, copy the YAML to a new experiment definition and use `ENV/.venv/bin/python -m pipeline.agentloop.driver --suite <config> --output-dir <new-output>`; each replicate must contain all six settings. Do not resume an old directory with changed configuration or code.

Level-specific regression and physical checks:

```sh
make -C ENV test-levels
# Include the independent audit of a completed matrix (a missing matrix otherwise skips):
INTERVIEW_REAL_SUITE_DIR="$PWD/OUTPUTs/my-levels/suite" make -C ENV test-levels
```

## Legacy baseline run

```sh
# Real solver, three numerical-search candidates, no LLM:
uv --project ENV run --locked python -m pipeline.agentloop.driver CONFIGs/interview/baseline.yaml
# Small agent experiment; requires Claude CLI access:
uv --project ENV run --locked python -m pipeline.agentloop.driver CONFIGs/interview/agent.yaml
```

Each configuration uses a separate experiment directory under `OUTPUTs/interview/`. Existing run directories are protected against overwrite; choose a new `exp_name` to repeat an experiment. The baseline and agent examples are startup checks, not a controlled performance comparison.

## Layout

- `SRC/pipeline/agentloop`: existing agent, search, runtime, constraints and reporting.
- `SRC/pipeline/exp_layer/aero2d`: 2D physical contract and adapter; `common` contains shared result helpers.
- `SRC/test`: inherited wing/runtime unit, smoke and flow checks, with unrelated scenarios removed.
- `EXP/batch1/aero2d`: mesh, solver and post-processing implementation plus original wing orchestration.
- `CONFIGs/interview`: small startup experiments; `CONFIGs/batch1/aero2d`: inherited wing configurations.
- `ENV`: Python/Brew dependencies, lock file, setup commands and source inventory.
- `RELATED_REPOs`: pinned dependency metadata and AIDE patch; restored sources are ignored.

Only code, configs, environment definitions, dependency metadata and this README are tracked. Local interview documents, outputs, caches, environments and credentials remain outside Git. Runtime outputs contain the evidence for each run; historical outputs were not imported.

## Verification

```sh
make -C ENV test-unit
make -C ENV test-smoke
# Longer tests; some require actual LLM access:
make -C ENV test-flow
```

A skipped real-solver or LLM test is not execution evidence. Test artifacts are written to ignored `OUTPUTs/` directories. Known compatibility fixes and completed checks are recorded in the inheritance manifest.

Verified on 2026-09-24: 285 unit checks, 20 real-solver smoke checks, 7 scripted tool-flow checks and 1 live LLM/tool-flow check passed. The numerical startup configuration completed 3 candidate evaluations plus 2 holdout checks. These checks establish the inherited baseline; they do not establish a new optimization benefit.
