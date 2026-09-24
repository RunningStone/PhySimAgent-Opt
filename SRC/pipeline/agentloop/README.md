# Agent loop

AIDE-based design/simulate loop, with the inherited 2D ground-effect wing mode and the separate Level 0–3 ordinary-airfoil experiment mode.

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

## Level 0–3 experiment API

`python -m pipeline.agentloop.driver --suite CONFIGs/level_spec_v1/suite.yaml --output-dir OUTPUTs/<run>/suite [--resume]` executes the complete six-setting matrix. A suite declares shared task/budget fields, M0/M1 profiles, public initialization, TopK and all six runs. Missing settings/nodes are rejected before execution. `run_suite(suite, output_dir, *, backend=None, invoke_fn=None, resume=False)` exposes the same interface for tests; injected backends are test evidence, not live runs.

`TaskSpec.experiment` enables the new protocol. `validate_proposal` accepts literal design data and authorized actions; `canonical_design` hashes physical parameters/geometry independently from numerics. Fixed arms use `evaluate_design`; Level 2-b uses `explore` and `submit_for_evaluation`. Only the host selects physical stages and formal settings.

`ToolRuntime.validate_final(candidate)` actually executes geometry → mesh → solve → post, verifies the resulting files and settings, and returns observations, assessment and measured cost. `evaluation.assessment.assess(observations, protocol)` separates completeness, numerical validity, engineering feasibility and ranking eligibility. `harness.select_topk(assessments, k)` accepts one stage/protocol and distinct eligible designs.

`records.RunStore` publishes attempts, observations, assessments, costs and checkpoints in one SQLite transaction. The store preserves failed attempts and conservative interruption charges; a changed snapshot rejects resume. Successful exploratory observations have no formal score. They remain visible to the agent, with `best_confirmed=None` until a valid formal submission exists.

A failed parent stage blocks its dependent stage without creating a child store or spending tools/LLM calls. Resume completes the parent before selecting and freezing the child's TopK initialization. A normally completed parent with no eligible design still uses the public fallback sequence. Optional `ToolRuntime(cache=True)` reuses only matching evidence within the same output namespace after verifying all artifact and underlying-file hashes; the complete live suite leaves it disabled.

Each stage exports `snapshot.json`, `records.json`, `summary.json`, proposal/tool attempts, actual case artifacts and isolated LLM request/response records. Candidate parents and evidence references preserve lineage without requiring a Git commit for each candidate. The suite summary distinguishes once-only physical execution cost from each arm's historical analysis cost.
