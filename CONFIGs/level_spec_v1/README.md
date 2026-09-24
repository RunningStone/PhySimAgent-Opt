# Level 0–3 complete smoke suite

`suite.yaml` freezes all six experimental settings and the full M0 → M1 physical path in one manifest. Run from the repository root:

```bash
bash EXP/level_spec_v1/run_levels.sh --output-dir OUTPUTs/<unique-run>/suite
bash EXP/level_spec_v1/run_levels.sh --output-dir OUTPUTs/<unique-run>/suite --resume
```

M0 is steady incompressible laminar `simpleFoam`; M1 is transient laminar `pimpleFoam`, at the same Reynolds number 100. The M1 objective uses the finite time window [4,5], not an asymptotic convergence claim. Both use an ordinary airfoil in free external flow. The physical implementation is `pipeline.exp_layer.aero2d.interview:invoke`.

There are six settings, nine recorded stage cells and seven unique physical stage runs: Level 0 M0 is shared explicitly as the public M0 stage of both Level 1 arms. Level 3 runs its own M0. Every inherited design is evaluated anew in M1. Cold-start M1 receives only public initialization and its own new evidence.

Fixed arms evaluate the two public or transferred initial designs before asking the LLM for feedback-derived proposals. Level 2-b starts with the LLM: it controls exploration and formal submission; there is no automatic baseline confirmation. Its exploratory mesh changes never change the frozen formal evaluation.

The objective is `-abs(Cl - 0.15)`; greater is better. A change of 0.01 is the pilot-calibrated smoke interpretation threshold, not an uncertainty estimate. Explicit shapes use 21 fixed x stations with editable upper/lower ordinates, actual piecewise-linear geometry and pressure sampling.

The smoke prompt asks the model to exercise actual shape changes and exploration/confirmation. M1 may stop with explicit current-stage evidence once all initialized designs have been independently evaluated and the confirmed lift error is at most 0.01. A forced additional solve after reaching that stopping criterion would distort the transfer cost comparison. The LLM deadline is 180 seconds per call, further limited by the remaining 600-second stage budget.

Budgets are the same in compared arms and count initialization, transferred-design re-evaluation, exploration and confirmation. The single replicate establishes runnable coverage; it does not establish which method is statistically superior. Failed pilot attempts and independent tests live separately from the formal suite under the same output root.
