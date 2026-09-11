# `mobile/android/` — native APK spec (Camera2 + NNAPI + usb-serial)

The Termux bridge (`../phone_onboard.py`) flies today. This spec defines the
native app that replaces it when you want one-tap operation, lower latency,
and a proper OSD — with **byte-identical MAVLink behaviour** so the flight
stack cannot tell which one is plugged in.

## 1. What the app does (and does not do)

Does: RGB capture → on-device AI → detections + health → MAVLink to the FC;
VIO pose when GPS is denied; OSD overlay on the pilot's FPV monitor via MSP;
bench/self-test mode. Does NOT do: flight control (FC owns that), path
planning (GCS/laptop owns that), cloud anything (offline-first per PS).

## 2. Modules

| Module | Tech | Notes |
|---|---|---|
| `capture` | Camera2 API, YUV_420_888 @ 1280x720 ≤ 30 fps | Fixed exposure once airborne; torch control for night sorties |
| `ai` | TFLite + NNAPI delegate, 2-model cascade | Person/thermal-blob model, then confirm model; budgets in §4 |
| `vio` | ARCore (preferred) / optical-flow fallback | Poses only trusted when `tracking_state == TRACKING`; else coast + flag |
| `mav` | usb-serial-for-android, MAVLink2 @ 921600 | Same dialect + sysids as the bridge; heartbeat 1 Hz, STATUSTEXT on verdicts |
| `osd` | MSP DisplayPort over the same USB link | Detection boxes + AGL + mode; mirrors `phone_onboard.py` `--mode osd` layout |
| `log` | GeoTIFF-ish sidecar: CSV + PNG crops | Every verdict lands with lat/lon/alt/confidence/thumbnail |

## 3. MAVLink dialect parity (normative)

The app MUST emit exactly the subset the flight stack parses — same message
IDs, same field semantics, same rates as `sar/hardware/mobile.py` expects:

- `HEARTBEAT` (type=MAV_TYPE_ONBOARD_CONTROLLER) @ 1 Hz
- `STATUSTEXT` (severity INFO) on each verdict: `HUMAN <lat> <lon> <conf>`
- `NAMED_VALUE_FLOAT` `AI_CONF`, `AI_FPS`, `VIO_OK` @ 2 Hz
- `GLOBAL_POSITION_INT` forwarded from FC → phone only (never synthesized)

Any new message needs a version bump in `sar/hardware/mobile.py` first.

## 4. Performance budgets (Snapdragon 8 Gen 2 reference phone)

| Budget | Value | Why |
|---|---|---|
| End-to-end RGB→verdict | ≤ 250 ms p95 | Keeps up with 4 m/s survey speed at 40 m AGL |
| NNAPI inference (int8) | ≤ 60 ms / model | Hexagon NPU, sustained (not peak) clocks |
| Thermal headroom | no throttle below 43 °C skin | `test_mobile.py::test_throttle_*` encodes the derate curve |
| Battery | ≥ 45 min screen-off logging | Phone flies in the payload bay, screen off |

## 5. Permissions & hardening

Camera, foreground-service (camera + connectedDevice), USB accessory/host,
wake-lock. No INTERNET, no FINE_LOCATION (position comes from the FC, never
from the phone's GNSS — one source of truth). Airplane mode + USB tethering
off is the pre-flight checklist state.

## 6. Build & test

Gradle, minSdk 29, targetSdk 34, one `release` flavor. CI builds the APK and
runs the same contract tests as the bridge: `pytest tests/test_mobile.py`
speaks to a MAVLink loopback the app also implements (`--mode loopback` in
both), so parity is machine-checked, not eyeballed.

## 7. Migration path

1. Fly the bridge, log everything. 2. Port `capture`+`mav` first, fly with AI
still on the laptop (shadow mode). 3. Move AI on-device, A/B verdicts against
the bridge logs. 4. Bridge becomes the bench fallback — never deleted.
