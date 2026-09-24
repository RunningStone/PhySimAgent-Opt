# PhySimAgent-Opt

A selectively inherited baseline for the VerticalLab interview: **2D inverted NACA wing in ground effect**, with an AIDE-based agent, OpenFOAM simulation, tool execution, trust checks and experiment records.

Source: [PhySimAgent](https://github.com/RunningStone/PhySimAgent) at `f0f5b784fa092f878e9790107582c4c4501130d6`. The import inventory and migration scope are recorded in [ENV/inheritance.json](ENV/inheritance.json). This is a new project history.

## Scope

Only the 2D wing scenario is included. The airfoil uses a moving ground boundary and optimizes downforce with a drag penalty; it is not a free-flight wing model. Other physics scenarios, 3D cases and native MDO dependencies are excluded.

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

## First local run

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
