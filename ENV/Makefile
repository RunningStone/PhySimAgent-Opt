ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
UV := uv --project $(ROOT)/ENV
PY := $(UV) run --locked python

.PHONY: repos env test-unit test-smoke test-flow verify

repos:
	@test -d $(ROOT)/RELATED_REPOs/aideml || git clone -q https://github.com/WecoAI/aideml.git $(ROOT)/RELATED_REPOs/aideml
	cd $(ROOT)/RELATED_REPOs/aideml && git checkout -q 6568c8bc27af51fe90ac83cdc87eb59512e4f527
	@if cd $(ROOT)/RELATED_REPOs/aideml && git apply --check ../aideml-poc.patch >/dev/null 2>&1; then \
		git apply ../aideml-poc.patch; \
	elif cd $(ROOT)/RELATED_REPOs/aideml && git apply --reverse --check ../aideml-poc.patch >/dev/null 2>&1; then \
		:; \
	else \
		echo "aideml-poc.patch does not apply cleanly" >&2; exit 1; \
	fi

env: repos
	$(UV) sync --locked

test-unit:
	cd $(ROOT) && $(PY) -m pytest SRC/test/unit -q --basetemp=OUTPUTs/_pytest/unit

test-smoke:
	cd $(ROOT) && $(PY) -m pytest SRC/test/smoke -q --basetemp=OUTPUTs/_pytest/smoke

test-flow:
	cd $(ROOT) && $(PY) -m pytest SRC/test/flow -q --basetemp=OUTPUTs/_pytest/flow

verify:
	cd $(ROOT) && $(PY) -c 'from pipeline.exp_layer.aero2d import verify, Params; ok, diff = verify(Params(0.3, 2, 0.04)); print("identical" if ok else diff); raise SystemExit(0 if ok else 1)'
