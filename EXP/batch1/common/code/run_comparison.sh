#!/usr/bin/env bash
# Run one common-interface comparison cell. Examples:
#   ENVIRONMENT=wing MODE=fixed EXP/batch1/common/code/run_comparison.sh
#   ENVIRONMENT=wing MODE=agent EXP/batch1/common/code/run_comparison.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

environment="${ENVIRONMENT:-wing}"
mode="${MODE:-fixed}"
case "$environment" in wing) exp_dir=aero2d ;; *) echo "only the wing environment is included" >&2; exit 2 ;; esac
config="CONFIGs/batch1/${exp_dir}/comparison/${environment}_${mode}.yaml"

if [[ ! -f "$config" ]]; then
  echo "unknown comparison cell: environment=$environment mode=$mode" >&2
  exit 2
fi

ENVIRONMENT="$environment" uv --project ENV run python - <<'PY'
import copy
import os

from pipeline.agentloop.driver import load_config

name = os.environ["ENVIRONMENT"]
exp_dir = "aero2d"
fixed = load_config(f"CONFIGs/batch1/{exp_dir}/comparison/{name}_fixed.yaml")
agent = load_config(f"CONFIGs/batch1/{exp_dir}/comparison/{name}_agent.yaml")
for item in (fixed, agent):
    item.pop("exp_name", None)
    item.pop("run_id", None)
    item["workflow"] = copy.deepcopy(item["workflow"])
    item["workflow"].pop("mode", None)
if fixed != agent:
    raise SystemExit(f"unfair comparison config pair for {name}: only mode/exp_name/run_id may differ")
PY

mkdir -p OUTPUTs/_run_logs
uv --project ENV run python -m pipeline.agentloop.driver "$config" \
  2>&1 | tee "OUTPUTs/_run_logs/v3_${environment}_${mode}.log"
