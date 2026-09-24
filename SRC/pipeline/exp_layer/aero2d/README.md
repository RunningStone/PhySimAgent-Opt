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

## Ordinary-airfoil interview adapter

`interview.invoke(tool, inputs, context)` is a separate concrete adapter for the Level 0–3 suite. It accepts `geometry`, `mesh`, `solve` and `post`; inputs contain a literal candidate, frozen settings, case directory and prerequisite artifact paths. The host's `context.snapshot.settings` must match. Each result includes actual artifact paths, diagnostics, effective settings/hash, source and measured cost; `post` adds raw Cl/Cd.

M0 uses steady incompressible laminar `simpleFoam`; M1 uses transient laminar `pimpleFoam` with a finite averaging window. Both use freestream outer boundaries and an ordinary airfoil wall. No ground boundary, inverted/downforce objective, turbulence or inherited RANS acceptance thresholds are applied.

`geometry.normalize_shape`, `validate_shape` and `shape_coords` accept finite x/upper/lower arrays with fixed stations, closed endpoints, bounded coordinates and positive interior thickness. These exact piecewise-linear surfaces enter Gmsh and pressure sampling. A parameterized candidate uses the fixed NACA0012 baseline and alpha_deg. An explicit candidate supplies its own complete point arrays.

Physical and numerical settings are frozen in `CONFIGs/level_spec_v1/suite.yaml`; exploration can vary only the implemented mesh_scale whitelist. The outer ToolRuntime owns deadlines and process cleanup. Geometry, mesh, solution and post manifests check the actual prerequisite files and their hashes before proceeding.
