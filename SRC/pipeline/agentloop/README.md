# Agent loop

Inherited AIDE-based design/simulate loop for the 2D wing.

| File | Responsibility |
| --- | --- |
| `harness.py` | Proposal parsing, parent selection, stop rules, numerical baseline, seeds and aggregation |
| `scenario.py` | Wing task, objective, parameter domain and recovery menu |
| `environment.py` | TaskSpec, literal-only design parsing, tool and result contracts |
| `workflow.py` | ToolRuntime, artifact ownership, budgets, caching, workers and final validation |
| `design_agent.py` | AIDE subclass, proposal/feedback prompts and tool selection |
| `driver.py` | Configuration loading, search execution, experiment Git lineage and output records |
| `evaluation/finding_report.py` | Summary tables and prediction/outcome exports |

Run `python -m pipeline.agentloop.driver <yaml>` inside the project environment. Inputs are YAML experiment definitions. Outputs are per-run summaries, per-case proposals/parameters/results, journal, verification and holdout records under the configured `OUTPUTs` directory.

`line=A` uses a numerical optimizer, `B` exposes scalar feedback, `C` also exposes field features. A fixed workflow invokes declared tools in order; an agent workflow asks the model to choose tools. Wing stages currently remain inside the evaluate adapter; explicit stage-level agent control is not claimed.

The current holdout `robust` field denotes valid scoring at all holdout points. Hypothesis and kill-criterion text are recorded but are not yet automatically adjudicated.
