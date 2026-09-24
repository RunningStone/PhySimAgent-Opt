# 2D runtime environment

Python 3.12 and uv. `make -C ENV env` restores AIDE at the recorded commit plus `RELATED_REPOs/aideml-poc.patch`, then performs `uv sync --locked` in this independent environment. Do not copy or link the previous project's virtual environment: editable imports would reference its source tree.

`Brewfile` contains only the OpenFOAM v2512 tap/cask. This baseline uses the native macOS arm64 launcher `openfoam2512`. Existing system installation can be reused after validation. LLM calls use the local Claude CLI and existing authentication.

`pyproject.toml` and `uv.lock` are managed by uv. Required packages cover the wing solver, plots, agent and tests. No battery, MPI/MDO or PETSc build is part of this environment. `inheritance.json` records the upstream revision and imported file inventory.
