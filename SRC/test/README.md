# Verification

Run from the repository root using this project's environment.

| Scope | Command | Requirements |
| --- | --- | --- |
| Unit | `make -C ENV test-unit` | Python dependencies; recorded/synthetic tool responses |
| Smoke | `make -C ENV test-smoke` | Real OpenFOAM; no live LLM needed |
| Flow | `make -C ENV test-flow` | Some checks need both real OpenFOAM and authenticated Claude CLI |

`exp/` contains explicit scripted fixtures and a deterministic test environment. These do not constitute real CFD or live-LLM evidence. The inherited compatibility matrix is limited to the nine wing configurations. Temporary files and evidence go under ignored `OUTPUTs/`.
