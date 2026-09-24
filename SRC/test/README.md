# Verification

Run from the repository root using this project's environment.

| Scope | Command | Requirements |
| --- | --- | --- |
| Unit | `make -C ENV test-unit` | Python dependencies; recorded/synthetic tool responses |
| Smoke | `make -C ENV test-smoke` | Real OpenFOAM; no live LLM needed |
| Flow | `make -C ENV test-flow` | Some checks need both real OpenFOAM and authenticated Claude CLI |

`exp/` contains explicit scripted fixtures and a deterministic test environment. These do not constitute real CFD or live-LLM evidence. The inherited compatibility matrix is limited to the nine wing configurations. Temporary files and evidence go under ignored `OUTPUTs/`.

`agentloop/test_unit_level_*.py` and `test_flow_level_suite.py` independently verify the Level 0–3 public requirements, including permissions, assessment, transactional recovery, TopK, isolation and the full matrix. `aero2d/` covers explicit geometry and actual OpenFOAM adapter smoke. Run all new checks with `make -C ENV test-levels`; its physical smoke requires OpenFOAM. The separate `EXP/level_spec_v1/run_levels.sh` command runs all six settings with a real LLM and real CFD, which synthetic tests do not replace.
