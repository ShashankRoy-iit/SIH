# SAHYOG - autonomous search-and-rescue drone stack
# `make help` lists everything.  Every target is safe to run from a clean clone.

PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
PYTHON  := $(BIN)/python
PIP     := $(BIN)/pip

.DEFAULT_GOAL := help

.PHONY: help venv install install-ai install-hw doctor test lint \
        mission rescue rescue-all bench eval subpixel sitl animate assets \
        dashboard onboard-dry clean distclean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/activate:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(BIN)/activate  ## Create the virtualenv

install: venv  ## Install the project (editable) + simulation extras
	$(PIP) install -e '.[sim,dev]'
	@$(PYTHON) -c "import sar, pathlib, sys; \
	  p = pathlib.Path(sar.__file__).resolve(); \
	  sys.exit(0) if str(p).startswith(str(pathlib.Path.cwd())) else sys.exit('WRONG sar package: %s' % p)"
	@echo "OK: 'import sar' resolves to this repository."

install-ai: install  ## Add the neural detector runtime (onnxruntime)
	$(PIP) install -r requirements-ai.txt

install-hw: install  ## Add companion-computer extras (camera, serial)
	$(PIP) install -e '.[hardware]'

doctor:  ## Diagnose the environment and the `No module named 'sar'` error
	$(PYTHON) scripts/doctor.py

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint:  ## Ruff, if installed
	-$(BIN)/ruff check sar scripts tests

mission:  ## Reference flood sortie (search + detect + report)
	$(PYTHON) scripts/run_mission.py --scenario flood --duration 460

rescue:  ## Full rescue loop: identify, drop payloads, route ground teams
	$(PYTHON) scripts/run_rescue_simulation.py --scenario flood --duration 60 --speedup 2.0

rescue-all:  ## The rescue loop across every disaster preset
	$(PYTHON) scripts/run_rescue_simulation.py --scenario flood      --duration 60 --speedup 2.0
	$(PYTHON) scripts/run_rescue_simulation.py --scenario earthquake --duration 90 --speedup 2.0
	$(PYTHON) scripts/run_rescue_simulation.py --scenario wildfire   --duration 90 --speedup 2.0
	$(PYTHON) scripts/run_rescue_simulation.py --scenario landslide  --duration 60 --speedup 2.0
	$(PYTHON) scripts/run_rescue_simulation.py --scenario stress     --duration 120 --speedup 2.0

bench:  ## Multi-scenario rescue benchmark -> artifacts/rescue_benchmark_results.json
	$(PYTHON) scripts/benchmark_rescue_pipeline.py

eval:  ## Detector calibration (aimed + survey)
	$(PYTHON) scripts/eval_detector.py --mode both

subpixel:  ## Sub-pixel radiometry experiment (why confirmation must fly low)
	$(PYTHON) scripts/experiment_subpixel_radiometry.py

sitl:  ## Vehicle-only flight test (ArduPilot SITL if present, else MiniSITL)
	$(PYTHON) scripts/sitl_flight_test.py

animate:  ## Rescue mission animation (HTML + GIF) into artifacts/
	$(PYTHON) scripts/animate_rescue_mission.py

assets:  ## Regenerate the teaching GIFs/infographics in docs/assets/
	$(PYTHON) scripts/make_teaching_assets.py

dashboard:  ## Sortie with the live command centre on :8088
	$(PYTHON) scripts/run_mission.py --scenario flood --duration 460 --live

onboard-dry:  ## Companion-computer entry point, hardware-free dry run
	$(PYTHON) scripts/run_onboard.py --dry-run

clean:  ## Remove caches and generated artifacts (keeps committed JSON)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info

distclean: clean  ## Also remove the virtualenv
	rm -rf $(VENV)
