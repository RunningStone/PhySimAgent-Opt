# 2D ground-effect wing adapter

The inherited physical case is an inverted NACA four-digit profile above a moving ground boundary. The solver is native OpenFOAM v2512 `simpleFoam`, with a steady RANS model. Geometry, boundary conditions, scoring and trust thresholds are unchanged by this migration.

## Public interface

- `Params(h_c, alpha_deg, camber)`: geometry/design inputs.
- `RunOptions`: declared recovery options, included in the case fingerprint.
- `run_case(params, options, *, scalar_only=False, cache=True, store=None, timeout_s=900)`: validate, mesh, solve, parse and return `CaseResult`.
- `verify(...)`: rerun without using a cached result, then compare deterministic result JSON.
- `CaseResult`: identity, status, forces, trust flags, field features, failures, provenance and runtime.

`status=ok` means the solver exited and the result parsed. Accepted scoring additionally requires the configured trust flags: convergence, residual quality, settled force, y-plus and mesh quality. Failure categories cover invalid geometry, invalid mesh, divergence, non-settled solution and timeout.

`contract.py` holds types and pure validation/geometry/feature functions. `__init__.py` orchestrates execution; the heavy solver implementation is `EXP/batch1/aero2d/code/solve.py`. `environment.py` adapts the block to `ToolRuntime`. `evaluation/` contains the inherited sweep and plotting utilities.

Outputs are content-addressed under the configured store in `OUTPUTs/`. Historical stores were not imported. The native OpenFOAM installation remains a system prerequisite; tests explicitly distinguish a missing solver from an executed test.
