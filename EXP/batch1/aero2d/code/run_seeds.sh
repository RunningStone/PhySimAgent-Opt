#!/usr/bin/env bash
# 用途:v1 多种子复跑 —— A/B/C × 3 种子 + B/C 无记忆消融 × 3 种子(15 次运行),放宽停滞;然后 finding 聚合
# 资源:每个 yaml 内 3 个种子并行(parallel: 3);本脚本把两个 yaml 一组并行,最多 6 路;LLM 经 claude CLI,约 360 次调用
# 依赖:OUTPUTs/batch1/aero2d/v1/SENSITIVITY.md(walk 已完成)
set -uo pipefail
cd "$(dirname "$0")/../../../.."
mkdir -p OUTPUTs/_run_logs
run() { uv --project ENV run python -m pipeline.agentloop.driver CONFIGs/batch1/aero2d/matrix/$1.yaml > OUTPUTs/_run_logs/v1_seeds_$1.log 2>&1; echo "$1 exit $?" >> OUTPUTs/_run_logs/v1_seeds_status.log; }
run seeds_lineA
run seeds_lineB & run seeds_lineC & wait
run seeds_lineB_nomem & run seeds_lineC_nomem & wait
uv --project ENV run python -m pipeline.agentloop.evaluation.finding_report OUTPUTs/batch1/aero2d/v1_seeds --walk OUTPUTs/batch1/aero2d/v1 2>&1 | tee OUTPUTs/_run_logs/v1_seeds_finding.log
