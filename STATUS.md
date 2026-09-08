# STATUS — what is done, what is not, and what would close each gap

Last updated: 2026-09-08. Companion to
[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md), which holds the work-package
tables. This file exists to answer one question honestly: **what is left?**

Legend: ✅ done and tested · 🔶 partial · ⬜ not started · 🚫 deliberately not done

---

## 1. Summary

| Layer | State | One-line assessment |
|---|---|---|
| Core, geo, config, events | ✅ | Complete, 113 tests green |
| Vehicle + MAVLink + SITL | ✅ | Real MAVLink over a socket; ArduPilot SITL supported, MiniSITL default |
| GPS-denied navigation | 🔶 | Estimator, sigma tracking and EKF source switching done; **no real VIO front end** |
| Perception (heuristic) | ✅ | Recall of resolvable targets 1.00 at 35/50/70 m |
| Perception (neural) | 🔶 | Full stack, runtime, registry, export, quantisation gate — **no trained weights** |
| Search planning | ✅ | Belief map, coverage grid, belief-ordered lanes, two-pass |
| Mission execution | ✅ | End-to-end scored sorties across five scenarios |
| Comms | ✅ | Store-and-forward, priority tiers, measured 97.8% packet success |
| Rescue (payload + routing) | ✅ | Ballistic solution with wind, A* over hazards |
| Command centre | 🔶 | Live dashboard works; replay from artifact not built |
| Hardware layer | 🔶 | Camera, safety supervisor, onboard loop, preflight gate — **never run on an aircraft** |
| Docs and deployment | ✅ | Run guide, teaching doc, bring-up, checklist, runbook, systemd/udev |
| Autonomous arm-and-fly | 🚫 | Deliberately gated behind the field-test checklist |

---

## 2. Gaps, in priority order

### 2.1 Precision ≈ 17% on the reference flood sortie 🔶

**What it means.** For every real survivor reported, roughly five false alarms
also reach the operator. Recall of *resolvable* targets is 1.00 at every tested
altitude, so this is clutter rejection, not blindness. The dominant sources are
sun-heated rooftops and vehicles that pass the thermal band and the size gate.

**What would close it.**
1. **Temporal consistency** — require a candidate to persist across ≥ 2 passes
   or ≥ 3 frames with consistent geo-position. Static clutter is *perfectly*
   consistent, so the discriminator has to be movement plus thermal-decay
   behaviour, not persistence alone.
2. **Trained thermal weights** (§2.2) — a learned detector rejects roof geometry
   far better than a shape heuristic.
3. **Hazard-model context** — the RescueNet-trained segmentation already knows
   "roof"; feeding it as a prior into the veto is designed, not wired.

**Effort:** ~1 week for (1), which is where most of the win is.

### 2.2 No trained thermal weights shipped 🔶

**What exists:** dataset table with sources, synthetic data generator, training
entry point with the 1-channel stem and P2 head, ONNX export, INT8 quantisation
with a ≥0.97-recall-retained gate, model registry with sha256 verification, and
four backends behind one interface. Verified end to end with a synthetic model.

**What does not exist:** actual `.pt`/`.onnx`/`.bin` weights trained on
HIT-UAV + AIResQ + SARD. That needs GPU time and dataset access approval; it is
not a code gap.

**Consequence:** the flight default runs the audited heuristic ensemble. That is
a supported, tested configuration — but it is not the AI performance the design
targets.

**Effort:** ~2 GPU-days plus dataset access.

### 2.3 No real visual-inertial odometry front end 🔶

`TelemetryVioSource` and `SimulatedVioSource` reproduce VIO **drift statistics**
correctly (measured: 7.2 m true error against a conservative 13.5 m reported
sigma over a 15 s denial) and the EKF source-set switching is real. But nothing
processes camera frames into odometry, so the system cannot actually navigate a
real GNSS denial.

**What would close it:** integrate OpenVINS or VINS-Fusion (or a hardware VIO
module such as the one on VOXL 2) publishing MAVLink ODOMETRY at 30 Hz. The
consumer side — `ExternalNavFeeder`, `NavQualityMonitor`, `EkfSourceManager` —
is already written and tested against exactly that message stream.

**Effort:** 2–3 weeks including field validation. This is the largest single
outstanding item.

### 2.4 Perception loop at 860 ms mean vs a 500 ms target 🔶

Landing-zone search runs on the control thread. Moving it to a worker at 0.5 Hz
(it does not need per-frame rate — landing zones do not move) is scoped and
straightforward. The dry-run onboard loop already achieves 230 ms/cycle with the
heuristic stack, so the deficit is specifically the landing-zone stage.

**Effort:** 2–3 days.

### 2.5 Endurance bounds coverage 🔶

The 900 × 900 m reference basin needs ~25 km of lane at 32 m spacing — more than
one battery. Belief-ordered lanes mitigate it; multi-sortie resume and
multi-aircraft handoff would solve it. `CoverageGrid` and `BeliefMap` are
serialisable specifically so that a second battery resumes rather than restarts,
but the resume path is not implemented.

**Effort:** ~1 week for single-aircraft multi-sortie resume.

### 2.6 The hardware layer has never touched an aircraft 🔶

`sar/hardware/` is written, tested (dry run + unit tests) and documented, and
the preflight gate refuses on eight distinct real conditions. But no line of it
has driven a real camera or a real autopilot. Every number in
[`HARDWARE_BRINGUP.md`](docs/HARDWARE_BRINGUP.md) that came from the bench is
labelled as such.

**What would close it:** Phase 7 — WP 7.1 to 7.9. Blocked on parts: an LWIR
module, a companion computer, and a LoRa modem are still to acquire.

### 2.7 Dashboard replay ⬜

Every sortie writes a complete report artifact, but the dashboard cannot replay
one. Reviewing a flight after the fact currently means reading JSON. WP 6.3.

**Effort:** 2–3 days.

### 2.8 Smaller open items ⬜

| Item | Note |
|---|---|
| Active learning from field false alarms | Designed in the AI doc, not implemented |
| Multi-aircraft coordination | Belief map is shareable; no protocol yet |
| Monte Carlo robustness sweep (WP 8.1) | `stress` preset exists; the sweep harness does not |
| Ablation study (WP 8.2) | Each pillar's contribution is argued, not measured |
| QNN `.bin` benchmarked on real QCS6490 | Recipe verified against Qualcomm's documented flow; no board in hand |
| `sar-*` console scripts | Declared in `pyproject.toml`; only exercised after `pip install -e .` |

---

## 3. Things that are deliberately not done 🚫

These are decisions, not omissions. Reversing any of them should be a
considered choice.

| Not done | Why |
|---|---|
| **Autonomous arm-and-fly** | No human has flight-tested it. The code refuses to arm and says so, instead of offering a button that has never been validated. Gated behind [`FIELD_TEST_CHECKLIST.md`](docs/FIELD_TEST_CHECKLIST.md). |
| **A detection transformer for the thermal path** | Measured: 5.0% AP on small objects vs 10.7–11.6% for YOLO on aerial thermal. Rejected on evidence. |
| **Weights committed to Git** | `models/` is ignored. Binary blobs in Git make the repository unusable and the provenance unverifiable; the registry with sha256 is the mechanism instead. |
| **A "simulation mode" branch inside the autonomy** | The autonomy talks MAVLink to a socket or to a serial port and cannot tell the difference. A simulation branch is how simulated systems stop transferring. |
| **Overriding ArduPilot's failsafes** | The supervisor sits behind them. Two independent systems both aborting is worse than one. RC loss is a caution here, not an abort. |
| **Internet-dependent map tiles in the dashboard** | The field site has no internet. |

---

## 4. What was completed in this pass

For traceability, the work that closed the gaps this document previously listed:

| Delivered | Where |
|---|---|
| `ModuleNotFoundError: No module named 'sar'` diagnosed and permanently fixed | `scripts/_bootstrap.py`, every script, `tests/test_packaging.py` |
| Environment doctor | `scripts/doctor.py` |
| Complete run guide with the *why* for every command | [`docs/HOWTO_RUN.md`](docs/HOWTO_RUN.md) |
| Visual teaching document, six animations generated from the live code | [`teach.md`](teach.md), `docs/assets/`, `scripts/make_teaching_assets.py` |
| Real AI integration: registry, runtime, four backends, calibration, export | `sar/ai/` |
| Model zoo, training, export, quantisation gate | `scripts/{fetch_models,train_detector,export_model}.py` |
| Hardware layer: cameras with skew rejection, latching safety supervisor, onboard loop with a preflight gate | `sar/hardware/` |
| Companion-computer entry point | `scripts/run_onboard.py`, `configs/onboard.yaml` |
| Deployment: systemd unit, installer, udev rules | `deploy/` |
| AI model and dataset reference | [`docs/06_AI_MODELS_AND_DATASETS.md`](docs/06_AI_MODELS_AND_DATASETS.md) |
| Search theory reference | [`docs/05_SEARCH_THEORY.md`](docs/05_SEARCH_THEORY.md) |
| Hardware bring-up, field-test checklist, deployment runbook | `docs/` |
| Test suite 41 → **113 tests** | `tests/` |

---

## 5. How to verify all of this yourself

```bash
python3 scripts/doctor.py                       # environment, honestly reported
python3 -m pytest tests/ -q                     # 113 tests
python3 scripts/run_mission.py --area 220 --duration 460     # a scored sortie
python3 scripts/eval_detector.py --mode both                 # the detector numbers
python3 scripts/run_onboard.py --dry-run --duration 60       # the flight code path
python3 scripts/fetch_models.py --list                       # the model zoo
make assets                                     # regenerate every teaching figure
```

Anything that disagrees with this document is a bug in this document. Please
open an issue.
