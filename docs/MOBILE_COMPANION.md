# The phone is the onboard computer

This is the build doc for the central hardware idea: **an Android phone
flies the AI**. RGB camera, neural inference, visual-inertial odometry,
4G relay, field display, backup battery — six subsystems in one device the
team already owns.

---

## 1. Why a phone (the 60-second case)

| Subsystem the PS needs | Buying it separately | The phone |
|---|---|---|
| RGB camera, stabilised, good low-light | ₹4–8k USB/MIPI + tuning pain | 12–50 MP, OIS, auto-tuned by the vendor |
| NPU for INT8 detection | ₹25k–1.2L board (Hailo/RB3/Jetson) | Hexagon DSP, YOLO-nano @ 10–30 fps via NNAPI |
| VIO compute + IMU | Same board + tuning | CPU + IMU + camera already fused by ARCore/OpenVINS |
| 4G telemetry relay | ₹3–6k modem + SIM hat | Built-in modem, just a SIM |
| Field display/debug | Extra monitor | Its own screen |
| Backup power for the AI | Extra BEC + supercap | Its own 5000 mAh battery |
| **Total** | **₹35k+, weeks of integration** | **₹0, one USB cable** |

And the sponsor story stays intact: the Hexagon DSP in a Snapdragon phone
is the same architecture as the RB3 Gen 2 target — TFLite + NNAPI here,
QNN there, one model format family.

## 2. Wiring

```
 Phone (USB-C OTG) ──► USB-serial CP2102 ──► Telem1 / SERIAL4 @ 921600
 Phone camera ──► RGB 1280x720 @15 (internal, no wires)
 LWIR thermal ──► phone USB via second OTG / hub ──► /dev/video (UVC)
 TF-Luna LiDAR ──► FC SERIAL5/Telem ──► RNGFND1_TYPE=20 (or phone USB)
 ELRS RX ──► SERIAL6 (control)      LoRa ──► SERIAL7 (alerts)
 Analog cam ──► VTX ──► goggles (pilot) + MSP OSD overlay from FC
```

Power: phone flies on its own battery (airplane mode + USB host still
streams 2+ hours). Everything else on 6S through the FC's BEC. A phone
holder with vibration foam replaces a whole companion-computer tray.

## 3. Software shapes (pick one)

### Shape A — Termux bridge (flies this week)

Pure Python on the phone, zero Android build tools:

```bash
# on the phone (Termux + Termux:API)
pkg install python opencv termux-api
pip install numpy pymavlink tflite-runtime
termux-usb -l            # grant USB-OTG serial permission
python ~/sar/mobile/phone_onboard.py --config ~/sar/onboard_mobile.yaml
```

`mobile/phone_onboard.py` does: Camera2 capture via Termux:API (or an
RTSP side-app for full rate) → TFLite-INT8 inference → MAVLink ODOMETRY @
30 Hz + detections to the FC/laptop → 4G relay when a tower exists. Start
here; it validates every interface the native app will use.

### Shape B — native APK (flies fastest)

`mobile/android/` holds the build spec: one Activity, three threads
(Camera2 capture, NNAPI inference, usb-serial MAVLink), foreground service
so Android never sleeps it. Build with Android Studio; the MAVLink dialect
is identical to Shape A, so the vehicle side cannot tell them apart.

## 4. MAVLink contract (both shapes)

| Message | Rate | Content |
|---|---|---|
| `ODOMETRY` (frame 20 LOCAL_FRD) | 30 Hz | VIO position + body-frame velocity + covariance from `pos_sigma_m` |
| `VISION_POSITION_ESTIMATE` | 30 Hz (redundant) | Same pose for EKF set 2 diversity |
| `HEARTBEAT` (comp 191) | 1 Hz | Phone alive + `THERMAL`/`BATTERY` in custom `STATUSTEXT`/named-value |
| `NAMED_VALUE_FLOAT` | 2 Hz | `AI_FPS`, `PHONE_T`, `RGB_Q`, `VIO_CONF` |
| `STATUSTEXT` | on event | `AI person 0.87 N+120 E-40`, veto notes, throttle warnings |

Flow control: if the FC stops ACKing ODOMETRY (link saturated), the phone
drops to 15 Hz VIO and halves RGB relay — alert traffic always wins.

## 5. Failure behaviour (what the vehicle does when the phone misbehaves)

| Phone fault | Detected by | Vehicle response |
|---|---|---|
| USB unplugged / app killed | No ODOMETRY for 0.5 s | EKF leaves set 2 → set 1 (GNSS) or set 3 (flow hold); supervisor cautions |
| RGB stale > 1 s | Frame age in `MobileCompanion` | Fuser drops to LWIR-only; planner continues (thermal is primary) |
| AI stale > 2 s | Last-detection age | Detector falls back to heuristic-on-LWIR via the laptop/FC path; alert notes "AI degraded" |
| Overheat ≥ 48 °C | `PHONE_T` named value | Phone halves AI rate itself; supervisor shortens sortie (thermal derate) |
| Battery < 20% | Heartbeat | 4G relay off, screen dim command, mission continues (AI sips power) |

Every one of these is a unit test in `tests/test_mobile.py` and a HITL
check in the field list: unplug the phone mid-flight (props off, on the
bench) and watch the EKF step down cleanly instead of faulting.

## 6. Bench validation (props off, one evening)

1. Phone on OTG → Telem1; `mavproxy.py --master=/dev/ttyACM0` shows
   ODOMETRY @ 30 Hz from comp 191.
2. Cover the phone camera: `VIO_CONF` falls, EKF stays on set 1; uncover:
   set 2 arms-ready again.
3. Run `phone_onboard.py --mode bench`: 60 s of RGB + AI + VIO against the
   flood poster on the wall; check `artifacts/phone_bench.json`.
4. Heat test: 20 min in a warm room; confirm self-throttle at 48 °C and
   recovery — no crash, no stuck state.
5. Unplug USB mid-stream: vehicle logs `PHONE FAILED`, EKF source steps
   down, replug resumes without reboot.

## 7. From phone to RB3 (the upgrade path, not a rewrite)

When the Qualcomm board arrives: the TFLite-INT8 model recompiles to QNN
(`scripts/export_phone_model.py --qnn`), the VIO thread moves to VOXL-style
stereo or stays on the phone as a *second* VIO source, and the phone
downgrades gracefully to RGB + 4G relay + field display. No interface
changes — `MobileCompanion` just reports fewer streams.
