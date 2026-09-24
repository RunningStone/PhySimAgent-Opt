#!/usr/bin/env bash
# 用途:场景 1 run 阶段 —— 三条线 A/B/C 依次跑(同一实验树,各自 worktree),然后 finding 报告
# 资源:CPU 串行;LLM 经 claude CLI(B/C 各约 budget 次调用)
# 依赖:OUTPUTs/batch1/aero2d/v1/SENSITIVITY.md 已存在(先跑 run_walk.sh)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
mkdir -p OUTPUTs/_run_logs
for L in ${LINES:-A B C}; do
  uv --project ENV run python -m pipeline.agentloop.driver CONFIGs/batch1/aero2d/matrix/line$L.yaml 2>&1 | tee OUTPUTs/_run_logs/v1_line$L.log
done
uv --project ENV run python -m pipeline.agentloop.evaluation.finding_report OUTPUTs/batch1/aero2d/v1 --walk OUTPUTs/batch1/aero2d/v1 2>&1 | tee OUTPUTs/_run_logs/v1_finding.log
