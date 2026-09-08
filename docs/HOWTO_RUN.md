# How to run this, and why each thing exists

This document is written for the exact situation you are probably in: you
cloned the repository, ran a script, and got

```
Traceback (most recent call last):
  File "/workspaces/SIH/scripts/run_rescue_simulation.py", line 24, in <module>
    from sar.core.geo import GeoPoint
ModuleNotFoundError: No module named 'sar'
```

Section 1 explains that in full and fixes it permanently. Sections 2 onwards
are the actual guide: every entry point, what it does, **why you would run it**,
what it prints, and how to read the numbers.

---

## Contents

1. [The `No module named 'sar'` error, explained properly](#1-the-no-module-named-sar-error-explained-properly)
2. [Install, once](#2-install-once)
3. [`scripts/doctor.py` — run this before anything else](#3-scriptsdoctorpy--run-this-before-anything-else)
4. [The entry points, in the order you should meet them](#4-the-entry-points-in-the-order-you-should-meet-them)
5. [Reading a sortie report](#5-reading-a-sortie-report)
6. [Choosing the detector: heuristic, neural, or both](#6-choosing-the-detector-heuristic-neural-or-both)
7. [Running against real ArduPilot SITL](#7-running-against-real-ardupilot-sitl)
8. [Running on the aircraft](#8-running-on-the-aircraft)
9. [Troubleshooting table](#9-troubleshooting-table)

---

## 1. The `No module named 'sar'` error, explained properly

### What happened

You ran:

```bash
python3 scripts/run_rescue_simulation.py --scenario flood --duration 60.0
```

When Python runs a *file*, it puts **that file's directory** at the front of
`sys.path` — here that is `/workspaces/SIH/scripts`, **not** `/workspaces/SIH`.
The package `sar/` lives at the repository root, one level up, so it is not on
the path and `import sar` fails.

Nothing was broken. The code was fine. The *working directory* was fine. Only
the module search path was wrong, and that is a property of how you launched
the script.

### Why `pip install sar` made it worse

```
pip install sar
Successfully installed sar-0.2.1
...
ModuleNotFoundError: No module named 'sar.core'
```

`sar` on PyPI is **an unrelated project** — a small synthetic-aperture-radar
helper by another author. Installing it puts a *different* `sar` package into
your environment. `import sar` now succeeds (it finds the stranger), and then
`sar.core` does not exist, so the error mutates into something more confusing
than the original. `pip install sar.core` then fails because no such package
exists anywhere; `sar.core` is a *submodule of this repository*, not a
distribution.

Undo it:

```bash
pip uninstall -y sar
```

### The three fixes, in order of preference

**(a) Install this project (recommended).** The distribution is deliberately
named `sahyog-sar` so it can never be confused with the PyPI `sar`, while the
import package stays `sar`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[sim]'
python3 scripts/run_rescue_simulation.py --scenario flood --duration 60 --speedup 2
```

**(b) Just run it.** Every script in `scripts/` now carries a five-line
bootstrap (`scripts/_bootstrap.py`) that puts the repository root at the
*front* of `sys.path` before importing `sar`, so a fresh clone works with no
install at all:

```bash
git clone https://github.com/ShashankRoy-iit/SIH && cd SIH
pip install numpy scipy            # the only hard requirements
python3 scripts/run_rescue_simulation.py --duration 30
```

The bootstrap also refuses to run against a foreign `sar` and tells you exactly
what to type if it finds one. That is why the fix stays fixed:
`tests/test_packaging.py` executes every script's `--help` from the repository
root on every test run, so a script that loses its bootstrap fails CI.

**(c) Module form.** `python3 -m scripts.run_rescue_simulation` works too,
because `-m` puts the current directory on the path. Useful to know; you do not
need it.

### Why the repository root goes *first* on the path

If a stale `sar` is installed in the active environment, appending the repo
would let the installed copy win, and you would be testing code that is not in
your working tree — the worst class of confusion, because every edit appears to
do nothing. Front, always.

---

## 2. Install, once

```bash
git clone https://github.com/ShashankRoy-iit/SIH && cd SIH
make install            # venv + editable install + sim extras + dev tools
```

or by hand:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[sim,dev]'          # numpy scipy pymavlink fastapi uvicorn matplotlib pytest
pip install -r requirements-ai.txt   # optional: onnxruntime, the neural detector runtime
```

What the extras mean:

| Extra | Installs | Needed for |
|---|---|---|
| *(base)* | numpy, scipy | everything; the simulator runs on this alone |
| `sim` | pymavlink, fastapi, uvicorn, matplotlib, pillow, pyyaml | MAVLink transports, `--live` dashboard, figures |
| `ai` | onnxruntime, opencv | neural detector backends (`sar.ai`) |
| `train` | ultralytics, torch, onnx | training and export — **workstation only, never the aircraft** |
| `hardware` | pymavlink, opencv, pyserial | companion computer: real cameras and serial links |
| `dev` | pytest, ruff | tests and lint |

Verify:

```bash
python3 -m pytest tests/ -q      # 113 tests, under a minute
python3 scripts/doctor.py
```

---

## 3. `scripts/doctor.py` — run this before anything else

```bash
python3 scripts/doctor.py
```

It answers, in the order things actually break: is the Python new enough, are
you in a virtualenv, **does `import sar` resolve to this repository**, is a
foreign `sar` installed, are the dependencies present, does every subpackage
import, are any model weights available, is an ArduPilot binary on `PATH`, is
`artifacts/` writable, is TCP 5760 free.

Every failure line comes with the command that fixes it. `--json` makes it
CI-consumable; the exit code is 0 only if a simulated sortie can actually run.

---

## 4. The entry points, in the order you should meet them

### 4.1 `scripts/run_rescue_simulation.py` — the full autonomy loop

```bash
python3 scripts/run_rescue_simulation.py --scenario flood --duration 60 --speedup 2.0
python3 scripts/run_rescue_simulation.py --scenario earthquake --duration 90
python3 scripts/run_rescue_simulation.py --scenario wildfire  --duration 90
python3 scripts/run_rescue_simulation.py --scenario landslide --duration 60
python3 scripts/run_rescue_simulation.py --scenario stress    --duration 120
```

**Why you run it:** it is the end-to-end demonstration — search, detect,
identify the human, assess their condition, drop the payload, route the ground
team. It exercises every subsystem in one process and writes
`artifacts/rescue_mission_<scenario>.json`.

**What happens inside:** a MiniSITL vehicle starts and listens on TCP 5760; the
client connects to it *over MAVLink* exactly as a ground station would; an
external-nav feeder streams ODOMETRY at 30 Hz so the EKF's pre-arm checks pass;
the mission runner plans a boustrophedon survey, flies it by streaming velocity
setpoints, renders RGB + LWIR frames from the world model, runs the perception
pipeline, and pushes confirmed survivors onto a store-and-forward radio queue.

`--speedup 2.0` runs the physics at twice real time. Raise it to shorten a run;
the sortie is deterministic in *simulated* time, so results do not change.

`--live` adds the command-centre dashboard on `http://0.0.0.0:8088`.

### 4.2 `scripts/run_mission.py` — the sortie the project is judged on

```bash
python3 scripts/run_mission.py --area 220 --duration 460 --live
python3 scripts/run_mission.py --area 220 --duration 460 --deny-gps 190 --deny-for 45
python3 scripts/run_mission.py --transport elrs_telemetry
python3 scripts/run_mission.py --scenario flood_night
```

**Why you run it:** this one is *scored*. It compares what reached the ground
station against the world's ground truth and reports recall, precision, geotag
error, coverage of the searched box, energy, and link statistics.

**The flag worth running is `--deny-gps`.** It drops the GNSS solution
mid-survey and forces the EKF onto the external-nav source set. The report then
shows what every survivor's position error actually became — the GPS-denied
claim, measured rather than asserted.

### 4.3 `scripts/eval_detector.py` — is the detector any good?

```bash
python3 scripts/eval_detector.py --mode both
python3 scripts/eval_detector.py --mode survey --scenario earthquake
```

**Why:** recall from a survey is confounded with "did the aircraft happen to fly
over them". *Aimed* mode puts every survivor at a controlled pixel offset at
each altitude, so recall is measured per survivor category and false alarms per
decoy kind. *Survey* mode measures the thing only a survey can: false alarms per
square kilometre of background clutter. Both report a **TTP physics ceiling** so
"the detector failed" can be told apart from "the target was not resolvable".

### 4.4 `scripts/experiment_subpixel_radiometry.py` — why confirmation flies low

**Why:** a radiometric pixel reports the area-weighted average of everything
inside it. Below one pixel, a survivor's apparent temperature collapses toward
the ground and the human thermal band — our strongest discriminator — stops
discriminating. This script measures that collapse, and it is the quantitative
justification for the two-pass search strategy.

### 4.5 `scripts/benchmark_rescue_pipeline.py` — across all disasters

```bash
python3 scripts/benchmark_rescue_pipeline.py
```

Identification, triage, drop precision and route planning across flood,
earthquake, wildfire, landslide and the compound `stress` preset →
`artifacts/rescue_benchmark_results.json`.

### 4.6 `scripts/sitl_flight_test.py` — the vehicle, on its own

Climb, velocity control, EKF source-set switch to external nav, restore, payload
release, RTL, land, disarm. **Why:** when a mission misbehaves, this tells you
whether the vehicle layer or the autonomy layer is at fault. Run it first when
something looks like a flight-dynamics problem.

### 4.7 `scripts/animate_rescue_mission.py` — the visual

An interactive HTML5 visualiser plus a GIF, from a real mission artifact.

### 4.8 `scripts/make_teaching_assets.py` — the figures in `teach.md`

Regenerates every animation from the live code, so a figure that disagrees with
the system is caught here.

### 4.9 AI tooling

```bash
python3 scripts/fetch_models.py --list                 # the model zoo
python3 scripts/fetch_models.py --synthetic            # tiny CI plumbing model
python3 scripts/train_detector.py --synthesize 4000    # labelled data from the simulator
python3 scripts/train_detector.py --train --data datasets/sim-thermal/data.yaml --p2
python3 scripts/export_model.py --qnn-recipe           # Qualcomm AI Hub commands
python3 scripts/export_model.py --validate --reference fp32.onnx --candidate int8.onnx
```

### 4.10 `scripts/run_onboard.py` — the flight-time loop

```bash
python3 scripts/run_onboard.py --dry-run --duration 60
```

Detailed in [section 8](#8-running-on-the-aircraft).

### 4.11 Make targets

`make help` lists all of them: `install`, `doctor`, `test`, `mission`, `rescue`,
`rescue-all`, `bench`, `eval`, `subpixel`, `sitl`, `animate`, `assets`,
`dashboard`, `onboard-dry`.

---

## 5. Reading a sortie report

```
  flight      armed=True  landed=True  crashed=False  522 s  4272 m  44.3 Wh  batt 77%
  plan        8 lanes, 30 m spacing, 1760 m total, alt 55 m AGL
  perception  482 frames, 241 cycles, 860 ms mean (2304 ms p95)
  COVERAGE
    effective coverage  11.7%   <- of the whole 900 x 900 m basin
    of the search box  100.0%   <- the sortie did what it planned
  DETECTION vs GROUND TRUTH
    correctly matched  4   recall 33%   precision 17%
    geotag error       mean 29.1 m, worst 54.2 m
  DATA LINK  (lora_900)
    delivered          655 packets / 120339 B   packet success 97.8%
```

**Read the two coverage numbers together.** 100% of the box was searched; the
box was 40,000 m² of an 810,000 m² basin. Recall against the whole world is 33%
because only 4 of 12 survivors were inside it. Quoting either number alone is
dishonest in opposite directions, so both are printed.

**Precision is the weak number** and is stated as such in the README's known
limitations. Two building rooftops generate repeated person detections. The
detector's own calibration shows `recall_of_resolvable = 1.00` at 35, 50 and
70 m, so this is a false-alarm problem, not a recall problem.

Scenario + seed fully determine the world, so two runs can be diffed
numerically:

```bash
diff <(jq -S . artifacts/mission_flood_seed7.json) <(jq -S . new_run.json)
```

---

## 6. Choosing the detector: heuristic, neural, or both

One environment variable selects the stack everywhere — simulator, dashboard,
aircraft:

```bash
SAR_DETECTOR=auto      python3 scripts/run_mission.py   # default
SAR_DETECTOR=heuristic python3 scripts/run_mission.py
SAR_DETECTOR=hybrid    python3 scripts/run_mission.py
SAR_DETECTOR=neural    python3 scripts/run_mission.py   # fails if no weights
```

| Mode | Behaviour |
|---|---|
| `auto` | Neural for each modality that has weights **and** a working runtime; audited heuristic for the rest. This is the flight setting. |
| `hybrid` | Both on the same modality, fused by the ensemble. Highest recall, slowest. Evaluation runs. |
| `heuristic` | No networks. Deterministic, ~90 fps on two CPU cores. Every committed artifact in this repo was produced this way. |
| `neural` | Networks only, and it **raises** rather than falling back — otherwise a benchmark would silently measure the wrong thing. |

`python3 scripts/doctor.py` prints which weights are present.
`docs/06_AI_MODELS_AND_DATASETS.md` covers the model choice, training and the
Qualcomm export path.

---

## 7. Running against real ArduPilot SITL

MiniSITL (`sar/sim/sitl.py`) is the default because the ArduPilot binary is a
build artifact and cannot always be present. It speaks the same MAVLink dialect
on the same port and reproduces EKF source-set switching, denial flags, pre-arm
strings and the ODOMETRY frame requirements — and it publishes only the
*estimate*, never truth, which is what makes geotag error measurable.

To use the real thing:

```bash
git clone --recursive https://github.com/ArduPilot/ardupilot && cd ardupilot
./waf configure --board sitl && ./waf copter
export PATH=$PATH:$PWD/build/sitl/bin
cd -   # back to this repo
python3 scripts/sitl_flight_test.py            # detects arducopter automatically
```

Parameters live in `configs/ardupilot_sitl.parm`; the hardware set is
`configs/ardupilot_hardware.parm`.

---

## 8. Running on the aircraft

```bash
# 1. laptop, no hardware, full flight code path
python3 scripts/run_onboard.py --dry-run --duration 60

# 2. bench: real autopilot over USB, simulated cameras, PROPS OFF
python3 scripts/run_onboard.py --mode hitl --target /dev/ttyACM0:921600

# 3. aircraft on the ground, live sensors — the sortie rehearsal
python3 scripts/run_onboard.py --mode flight --config configs/onboard.yaml

# 4. pre-flight gate only; exit code 0/1, this is what the systemd unit calls
python3 scripts/run_onboard.py --mode flight --preflight-only
```

The loop does **not** arm and fly itself. That is gated behind
`docs/FIELD_TEST_CHECKLIST.md` and work package 7.8, and the code says so
instead of offering a button nobody has flight-tested. Everything up to that
line runs: perception on live cameras, geo-tagging from live EKF telemetry,
triage, store-and-forward reporting, the safety supervisor and the payload
servo path.

Bring-up, wiring, parameters and the companion-computer install are in
`docs/HARDWARE_BRINGUP.md`; the runbook is `docs/DEPLOYMENT_RUNBOOK.md`.

---

## 9. Troubleshooting table

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sar'` | script dir on the path, not the repo root | `pip install -e '.[sim]'`, or just re-pull — every script now self-bootstraps. Section 1. |
| `ModuleNotFoundError: No module named 'sar.core'` | the unrelated PyPI `sar` is installed | `pip uninstall -y sar && pip install -e .` |
| `pip install sar.core` → *No matching distribution* | `sar.core` is a submodule here, not a package anywhere | Nothing to install. Section 1. |
| `error: externally-managed-environment` (PEP 668) | system Python refuses installs | use a venv: `python3 -m venv .venv && source .venv/bin/activate` |
| `[Errno 98] Address already in use` on 5760 | a previous MiniSITL is alive | `pkill -f run_mission; pkill -f run_rescue` or pass `--port 5770` |
| Dashboard shows nothing at `:8088` | `--live` not passed, or fastapi missing | `pip install -e '.[sim]'` and add `--live` |
| `arducopter: command not found` | no ArduPilot build | optional — MiniSITL is used automatically. Section 7. |
| Mission runs but detects nothing | detector mode `neural` with an untrained/plumbing model | `SAR_DETECTOR=heuristic`, or train real weights (`docs/06_AI_MODELS_AND_DATASETS.md`) |
| `onnxruntime` import error | AI extra not installed | `pip install -r requirements-ai.txt` |
| Perception slower than 2 Hz | landing-zone search on the control thread; known limitation | lower `--perception-hz`, or run `SAR_DETECTOR=heuristic` |
| Tests pass locally, fail in CI on figures | matplotlib needs a headless backend | already forced (`MPLBACKEND=Agg` in the bootstrap); check the CI image has fonts |
| LWIR pre-flight refuses: "not radiometric" | camera delivering 8-bit AGC video | enable TLinear, or `--allow-non-radiometric` and accept triage being disabled |

---

## Related documents

* [`teach.md`](../teach.md) — the animated explainer: the problem, and how each part of the solution works.
* [`docs/PROJECT_PLAN.md`](PROJECT_PLAN.md) — the plan, work packages, acceptance tests.
* [`docs/06_AI_MODELS_AND_DATASETS.md`](06_AI_MODELS_AND_DATASETS.md) — model selection, datasets, training, Qualcomm export.
* [`docs/HARDWARE_BRINGUP.md`](HARDWARE_BRINGUP.md) — from a box of parts to a flying aircraft.
* [`docs/FIELD_TEST_CHECKLIST.md`](FIELD_TEST_CHECKLIST.md) — what must pass before, during and after a flight.
* [`STATUS.md`](../STATUS.md) — what is done, what is not, and what is next.
