# Field test checklist

Print this. Tick it with a pen. One person reads, one person acts — the same
discipline every flight-test organisation converged on independently.

This checklist is the **gate on autonomous flight**. `scripts/run_onboard.py`
deliberately does not arm the aircraft; that capability is WP 7.8/7.9 and it
opens only after every section below has passed on the specific airframe you
are about to fly.

---

## A. The day before

- [ ] `python3 -m pytest tests/ -q` passes on the companion computer itself
- [ ] `python3 scripts/doctor.py` exits 0 on the companion computer
- [ ] Batteries charged, storage-charged spares labelled, IR thermometer to hand
- [ ] `/etc/sar/onboard.yaml` reviewed line by line; **`hover_power_w` is a
      measured number from *this* airframe**, not the default
- [ ] Geofence polygon set for *this site* in `onboard.yaml` **and** in
      ArduPilot (`FENCE_ENABLE`, `FENCE_RADIUS`, `FENCE_ALT_MAX`, `FENCE_ACTION`)
- [ ] Site permission obtained; airspace checked; DGCA/local rules confirmed
- [ ] Weather forecast: wind < 8 m/s (hard limit 12 m/s), no rain, no fog
- [ ] Someone who is not the pilot has read this checklist

## B. At the site, before power

- [ ] Take-off area clear, 30 m radius, no people downwind of the payload path
- [ ] Landing/abort area identified and clear
- [ ] Fire extinguisher and LiPo bag present
- [ ] Props checked for nicks; **props fitted last**, after all bench checks
- [ ] All fasteners torqued; camera plate does not move under thumb pressure
- [ ] Payload latched with the real payload mass

## C. Power-on, props OFF

- [ ] Battery voltage and cell balance within limits
- [ ] Flight controller boots; no pre-arm messages other than "waiting for GPS"
- [ ] GNSS: **≥ 8 satellites** (the preflight gate's minimum), HDOP < 1.5;
      wait for 30+ in the open
- [ ] EKF healthy; horizon level; heading correct against a known bearing
- [ ] RC link: RSSI/LQ good; all mode switches produce the expected mode
- [ ] **Kill switch tested in every flight mode**
- [ ] Transmitter off → failsafe → RTL mode change within 1 s → transmitter on
- [ ] Companion computer up; `/dev/sar-autopilot`, `/dev/sar-lwir`,
      `/dev/sar-rgb` all present

```bash
python3 scripts/run_onboard.py --mode flight --preflight-only
echo $?        # must be 0
```

The gate refuses on any of: telemetry silence > 3 s · fewer than 8 satellites ·
EKF not ready · no camera frames within 10 s · non-radiometric LWIR · cameras
never synchronising · a missing perception pipeline · an already-latched safety
supervisor.

- [ ] LWIR frame shows plausible temperatures — point it at a person: **30–36 °C**
      at the face; check against the IR thermometer
- [ ] RGB frame present and in focus
- [ ] Skew statistics healthy (most pairs accepted)
- [ ] Data link up; the dashboard on the laptop shows aircraft health
- [ ] Payload servo: release and re-latch once, on the ground, with the payload

## D. First flight of the day — manual

Props on now.

- [ ] Arm in `STABILIZE`, hover at 2 m for 30 s, land, disarm. Listen.
- [ ] Check motor temperatures by hand — anything too hot to hold is a fault
- [ ] Review the log: vibration `VIBE` < 15 m/s², no clipping, no EKF variance
      warnings
- [ ] `LOITER` hold for 60 s: position drift < 2 m in calm air

**Do not proceed if anything above is marginal.** A marginal hover becomes an
uncontrolled descent at 45 m.

## E. Autonomy, incrementally

Each step is a separate flight. Land, review the log, then continue.

1. - [ ] **Altitude only.** Auto climb to 20 m, hold 60 s, land. Pilot on sticks
        the whole time.
2. - [ ] **Single lane.** One 100 m lane at 45 m AGL, then RTL.
3. - [ ] **Full survey, perception passive.** The planned lanes with the
        perception loop running and *reporting*, but no autonomous response.
4. - [ ] **Confirmation pass.** Allow the descent to 22 m over a detection.
        Verify the descent stops at `min_altitude_agl_m` (8 m).
5. - [ ] **Payload drop.** Over a marked, empty target area. Measure the actual
        miss distance and compare with `compute_release_solution()`'s prediction.
6. - [ ] **GNSS denial** (WP 7.8). Fly under canopy or mask the antenna.
        **Record true position from an independent source** and compare it with
        the reported sigma. The claim to validate: reported sigma ≥ true error.

## F. Every flight, in the air

Called out loud by the observer:

- [ ] Altitude and distance within the geofence
- [ ] Battery: **the supervisor's return-home energy margin**, not the percentage
- [ ] Link status; queue depth not growing without bound
- [ ] Position sigma; **anything > 10 m means reports are not actionable**
- [ ] Wind rising? Land at 10 m/s, do not wait for 12.

**Abort — take manual control and RTL — immediately on any of:**

| Trigger | Why |
|---|---|
| Any unexpected attitude change | Nothing good starts this way |
| Safety supervisor latches any state | It only latches on a real limit |
| Position sigma > 25 m | The aircraft no longer knows where it is |
| Telemetry silence > 3 s | You have lost the ability to observe it |
| Return energy margin < 1.35× | It may not make it home |
| A person enters the operating area | Always |

## G. After every flight

- [ ] Battery removed, voltage logged, allowed to cool before charging
- [ ] `/var/log/sar/onboard_report_*.json` and the ArduPilot `.bin` log copied off
- [ ] Anything unexpected written down **while you remember it**, in the field
      log, with the timestamp
- [ ] Any latched safety state investigated before the next flight — **never
      cleared to "get one more flight in"**

## H. After the day

- [ ] Reports archived with the config that produced them (the effective
      configuration is embedded in every report)
- [ ] Detections compared against ground truth; recall and precision computed
      for the day
- [ ] Any parameter changed in the field written back into
      `configs/ardupilot_hardware.parm` or `configs/onboard.yaml` **in Git**,
      with the reason

---

## Standing rules

1. **A pilot is always on the sticks and can take over instantly.** No exceptions,
   no "it's just a short one".
2. **Never fly over people.** Not at 45 m, not at 8 m, not for a demonstration.
3. **The abort criteria are not negotiable in the field.** They are written down
   here so that nobody has to argue about them while an aircraft is airborne.
4. **A latched safety state is evidence, not an obstacle.** Investigate it.
5. **If the checklist and the schedule disagree, the checklist wins.**
