# Hardware bring-up: from a box of parts to a flying aircraft

The order in this document is the order you should work in. Each step ends with
a **check** that must pass before the next step, because every failure here is
cheaper on the bench than in the air.

Covers Phase 7 of [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (WP 7.1 – 7.9). The flight
loop it brings up is `scripts/run_onboard.py` / `sar/hardware/`.

> **Nothing in this document arms the aircraft.** Arming and autonomous flight
> are gated behind [`FIELD_TEST_CHECKLIST.md`](FIELD_TEST_CHECKLIST.md).

---

## 0. Bill of materials

| Subsystem | Part | Notes |
|---|---|---|
| Flight controller | **TBS Lucid H743 Wing** (`TBS_LUCID_H7_WING`) | STM32H743VIH6 @ 480 MHz, 3–12S, 13 PWM, 7 UART. **No internal compass.** |
| Frame | Carbon tube quad, 3D-printed mounts | Sensor mounts must be rigid — boresight drift is unrecoverable in software |
| Propulsion | BLDC + 4-in-1 ESC, bidirectional DShot, 6S | Target hover throttle < 55% at survey weight |
| Battery | 6S LiPo 8000 mAh | Analogue monitor: `VOLT_PIN 10`, `CURR_PIN 11`, `VOLT_MULT 11.0`, `AMP_PERVLT 40.0` |
| GNSS | u-blox module | 30+ satellites in the open; **external magnetometer required** (or EKF3 without compass) |
| RC | ELRS RX + RadioMaster Pocket | CRSF on SERIAL6, `RSSI_TYPE=3`, failsafe → RTL |
| Companion computer | **Qualcomm RB3 Gen 2 (QCS6490)** or VOXL 2 | Hexagon DSP runs the INT8 QNN models |
| Thermal | **FLIR Lepton 3.5** (160×120, radiometric) or Boson 320 | *Radiometric is not optional* — see §4 |
| RGB | USB or MIPI camera, 1280×720 @ 15 fps | On Qualcomm, prefer the ISP path via `qtiqmmfsrc` |
| Data link | LoRa 900 MHz modem on SERIAL7 | The minimum alert-capable link. ELRS is **control only**. |
| Payload | Servo latch, channel 9 | `SERVO9_FUNCTION=59`, `GRIP_*` parameters |
| FPV (optional) | Analogue camera + VTX + goggles | MSP DisplayPort OSD, `SERIALx_PROTOCOL=42`, `OSD_TYPE=5` |

---

## 1. Airframe and propulsion  (WP 7.1)

1. Build the frame. Mount the FC with soft vibration isolation, arrow forward.
2. Mount the camera plate **rigidly** and as close to the FC as the CG allows.
3. Wire the 4-in-1 ESC; bidirectional DShot to SERIAL8 for ESC telemetry.
4. Balance props. Set the CG at the geometric centre with the payload fitted.

**Check** — with props off and the battery on a bench harness: motors spin in
the correct directions, ESC telemetry appears in `Mission Planner → Status`, and
`INS` vibration levels are < 15 m/s² in a hover once flying.

**Measure and record hover power.** `configs/onboard.yaml: safety.hover_power_w`
defaults to 420 W. Do not trust the motor datasheet — the safety supervisor's
return-home energy calculation is built on this number, and an optimistic value
is the single most direct way to lose the aircraft.

---

## 2. Flight controller: flash and parameters  (WP 7.2)

```bash
# ArduPilot Copter 4.5+, target TBS_LUCID_H7_WING
# Mission Planner → Config → Full Parameter List → Load from file
#   configs/ardupilot_hardware.parm
# or:
mavproxy.py --master=/dev/ttyACM0 --load-module=param
param load configs/ardupilot_hardware.parm
```

Serial routing in that profile:

```
SERIAL0  USB        ground laptop, MAVLink 2 @ 115200
SERIAL2  GPS        u-blox
SERIAL4  Telem1  →  companion computer, MAVLink 2 @ 921600
SERIAL5  spare      MSP DisplayPort OSD
SERIAL6  RC         ELRS receiver (CRSF)
SERIAL7  Telem2  →  LoRa data modem
SERIAL8  ESC telem  4-in-1 ESC
```

**Check** — after a reboot, `param diff` shows no unexpected values, and
`ARMING_CHECK` is at its **default (all checks on)**. Disabling arming checks to
make a problem go away is how field tests turn into incidents.

**Compass.** The board has none. Either fit an external magnetometer, or fly the
supplied profile which sets the EKF3 yaw source explicitly and runs without one.
Decide before the field day; it changes the parameter set.

---

## 3. RC link and failsafes  (WP 7.3)

1. Bind the ELRS receiver to the RadioMaster Pocket. CRSF on SERIAL6.
2. `RSSI_TYPE=3`; verify RSSI and LQ appear in telemetry.
3. Set a flight-mode switch with **at least** `STABILIZE / LOITER / RTL`.
4. Set a **kill switch** on a dedicated channel (`RC*_OPTION=31`).
5. RC-loss failsafe → **RTL**. Battery failsafe → **RTL**, then Land.

**Check** — with props off: powering off the transmitter triggers the failsafe
mode change within 1 s, and the kill switch disarms instantly in every mode.

> The companion computer's supervisor treats RC loss as a **caution, not an
> abort** — ArduPilot's own failsafe already owns that decision, and two
> independent systems both aborting is worse than one. The supervisor sits
> *behind* ArduPilot's failsafes, never in front of them.

---

## 4. Cameras  (WP 7.6)

### Radiometry is a hard requirement

The Lepton must run in **TLinear** mode, emitting 16-bit counts at
`resolution_k = 0.01` K/count. Not "thermal-looking video" — actual temperature.
Everything in `sar/perception/thermal.py` operates in kelvin, and 8-bit AGC video
has already thrown that away: AGC rescales per frame, so the same person is a
different pixel value in two consecutive frames and no fixed threshold means
anything.

`run_onboard.py` **refuses to fly** on a non-radiometric stream. The waiver
exists (`--allow-non-radiometric`) for bench work, and it disables thermal
triage entirely rather than pretending.

### Enumerate and configure

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --all | head -40      # confirm 160x120, 16-bit
```

Set `cameras.lwir.device`, `width`, `height`, `fps`, `hfov_deg`,
`resolution_k` in `/etc/sar/onboard.yaml`. Lepton 3.5 is export-limited to
8.7 Hz — set `fps: 9` and do not expect more.

### Boresight

Mount RGB and LWIR on the same rigid plate, optical axes parallel, as close
together as physically possible. Cross-modal fusion associates detections by
projected ground position, so a fixed offset is tolerable and a *rotation* is
not.

Calibrate with a warm target (a hand, a kettle) visible in both frames at ~10 m
and record the pixel offset.

### Synchronisation

Two free-running cameras at 9 Hz and 15 fps do not produce simultaneous frames.
`sar/hardware/camera.py::CameraPair` timestamps each frame, pairs the closest,
**measures the skew**, and rejects any pair beyond `max_skew_s = 0.08` — falling
back to LWIR-only rather than fusing two different moments. At 8 m/s, 80 ms is
0.64 m of parallax, comfortably inside the association radius.

**Check** — the dry run reports the pairing statistics:

```bash
python3 scripts/run_onboard.py --mode hitl --duration 60
#   cameras   lwir 540 frames, rgb 900 frames, 18 paired, 23 skew-rejected
```

A high rejection rate on the bench means the two sources are free-running badly:
lock the frame rates or accept LWIR-only. (~44% rejection is normal in the
*simulated*-camera dry run because the two synthetic sources deliberately
free-run; real hardware should do much better, and the number is a genuine
health signal either way.)

---

## 5. Companion computer  (WP 7.6)

```bash
# On the board (Ubuntu / Yocto with Python 3.10+)
sudo ./deploy/install_companion.sh
```

That script: installs system packages, creates `/opt/sar` with a virtualenv,
installs the project with the `hardware` and `ai` extras, copies
`configs/onboard.yaml` to `/etc/sar/onboard.yaml` if absent, installs the udev
rules from `deploy/99-sar-devices.rules`, creates `/var/log/sar`, and installs —
but does **not** enable — the `sar-onboard.service` unit.

Wire the companion computer's UART to the FC's **Telem1 (SERIAL4)** at 921600
8N1. Common device names:

| Board | Device |
|---|---|
| Qualcomm RB3 Gen 2 | `/dev/ttyHS1` |
| Jetson Orin Nano | `/dev/ttyTHS0` |
| Raspberry Pi 5 | `/dev/ttyAMA0` |
| Anything over USB | `/dev/ttyACM0` |

The udev rules give the cameras and the autopilot **stable names**
(`/dev/sar-lwir`, `/dev/sar-rgb`, `/dev/sar-autopilot`) so a reboot cannot
renumber `/dev/video*` and silently swap your thermal camera for your RGB one.

**Check:**

```bash
python3 scripts/doctor.py
python3 scripts/run_onboard.py --mode hitl --target /dev/ttyACM0:921600 --duration 60
```

`hitl` mode talks to the **real autopilot** with **simulated cameras** — props
off. It proves the MAVLink path, the telemetry rates, the safety supervisor and
the report writer before any sensor is involved.

---

## 6. Models onto the board

```bash
python3 scripts/fetch_models.py --list
python3 scripts/fetch_models.py --verify-all        # sha256, every present file
SAR_MODELS_DIR=/opt/sar/models python3 scripts/doctor.py
```

With no weights present the system runs the audited heuristic ensemble and says
so. That is a supported configuration — it is how every artifact in this
repository was produced. See
[`06_AI_MODELS_AND_DATASETS.md`](06_AI_MODELS_AND_DATASETS.md) for the QNN export
path onto the Hexagon DSP.

---

## 7. Payload latch  (WP 7.7)

```
SERVO9_FUNCTION 59      GRIP_ENABLE 1     GRIP_TYPE 0
GRIP_GRAB 1100          GRIP_RELEASE 1900
```

`payload.servo_channel`, `hold_pwm` and `release_pwm` in `onboard.yaml` must
match. **Check** — with the aircraft disarmed and the payload on the bench,
`MAV_CMD_DO_GRIPPER` releases and re-latches, ten times without a miss. Then
hang the actual payload mass on it and repeat; a latch that holds 100 g and
drops 500 g is a failure you find in the air otherwise.

---

## 8. Data link  (WP 5.4)

LoRa modem on SERIAL7. Confirm the modem's **MTU** and match it to the transport
profile in `sar/comms/link.py` (`lora_900` assumes a 222 B MTU). A survivor
alert is ~200 B by design so that it fits the worst link in the model — if your
modem's MTU is smaller, the profile must change, or alerts will silently never
send.

**Check** — walk the modem to the edge of the intended operating radius with the
aircraft powered on the ground and confirm the queue drains when you walk back.
That is the store-and-forward behaviour in
[`teach.md`](../teach.md#8-telling-someone-the-radio), on your radio, at your
site.

---

## 9. Ground station

```bash
python3 scripts/run_mission.py --live          # dashboard on :8088, no internet needed
```

The dashboard makes **no external calls** — no map tiles, no CDNs. That is a
requirement, not an optimisation: the field site has no internet.

---

## 10. What must be true before the first flight

Continue to [`FIELD_TEST_CHECKLIST.md`](FIELD_TEST_CHECKLIST.md). In summary:

- [ ] Hover power measured and written into `onboard.yaml`
- [ ] `python3 scripts/run_onboard.py --mode flight --preflight-only` exits 0 on the aircraft
- [ ] Kill switch tested in every flight mode
- [ ] RC-loss and battery failsafes verified on the bench
- [ ] Geofence set (`safety.geofence_polygon` **and** ArduPilot's own `FENCE_*`)
- [ ] Payload latch cycled 10× with the real payload mass
- [ ] Cameras boresighted and skew statistics healthy
- [ ] A pilot on the sticks who can take over instantly, every flight
