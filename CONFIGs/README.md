# Experiment configurations

YAML files use `defaults` lists, merged from left to right relative to each file. `batch1/aero2d` preserves the wing configurations and fixed/agent comparison pair. `interview` contains small, memory-free startup runs for this project.

Specify experiment name, budget, model, seed, output root and workflow in YAML. Do not reuse a completed experiment name: the driver protects existing run directories. Inputs with memory enabled need a separately generated sensitivity file; none of the old outputs were imported.
