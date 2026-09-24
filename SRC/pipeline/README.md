# Pipeline

`agentloop` plans and executes experiments through `TaskSpec` and `ToolRuntime`. `exp_layer/aero2d` provides the physical contract and adapter; it calls the solver code in `EXP/batch1/aero2d`. `exp_layer/common` supplies shared failure/provenance and atomic-file helpers.

The existing module boundaries are preserved during migration. No extra controller or service is added.
