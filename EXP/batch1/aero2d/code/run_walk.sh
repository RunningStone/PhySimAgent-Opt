#!/usr/bin/env bash
# 用途:场景 1 walk 阶段 —— h/c × α 扫描、确定性抽查、失败复现、SENSITIVITY.md(无 agent)
# 资源:CPU 串行(OpenFOAM 原生 arm64),约 30 case × ≤ 2 min;无 GPU
# 依赖:ENV/.venv、openfoam2512、claude CLI(仅 SENSITIVITY 总结)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
mkdir -p OUTPUTs/_run_logs
uv --project ENV run python -m pipeline.exp_layer.aero2d.evaluation.walk CONFIGs/batch1/aero2d/matrix/walk.yaml "$@" 2>&1 | tee OUTPUTs/_run_logs/v1_walk.log
