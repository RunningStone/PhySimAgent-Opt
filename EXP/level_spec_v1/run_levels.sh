#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
exec "$ROOT/ENV/.venv/bin/python" -m pipeline.agentloop.driver \
  --suite "$ROOT/CONFIGs/level_spec_v1/suite.yaml" "$@"
