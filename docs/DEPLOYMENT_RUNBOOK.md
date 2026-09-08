# Deployment runbook

The operational document: what a team does on the day, in order, from a packed
case to a debrief. Written so that **someone who did not build the system can
run it** (WP 8.3).

Companion documents: [`HARDWARE_BRINGUP.md`](HARDWARE_BRINGUP.md) is the
one-time build; [`FIELD_TEST_CHECKLIST.md`](FIELD_TEST_CHECKLIST.md) is the
safety gate; this is the repeatable procedure.

---

## Roles

| Role | Responsibility | Never also |
|---|---|---|
| **Pilot in command** | Sticks in hand, all flight decisions, calls the abort | Operating the laptop |
| **Operator** | Dashboard, tasking, reads the survivor list | Flying |
| **Observer** | Eyes on the aircraft and the airspace, reads the checklist aloud | Looking at a screen |

Three people minimum. With two, the observer role goes to the pilot and the
sortie stays inside 200 m.

---

## 1. Pack

- [ ] Aircraft, props (packed separately), 3+ batteries, charger, LiPo bag
- [ ] Transmitter (charged), ground laptop (charged), LoRa ground modem + antenna
- [ ] Spare props, hex drivers, zip ties, tape, multimeter, IR thermometer
- [ ] Printed [`FIELD_TEST_CHECKLIST.md`](FIELD_TEST_CHECKLIST.md) and a pen
- [ ] Payloads: flotation aid / water pack / radio beacon, packed and weighed
- [ ] First-aid kit, fire extinguisher, hi-vis vests

## 2. Site setup (15 min)

1. Pick take-off and landing points; mark them. 30 m clear radius.
2. Set up the ground station upwind of the take-off point.
3. Note **home position, magnetic north, and the wind direction** in the log.
4. Define the search area as a north/east box relative to home, and put its
   dimensions into the operator's tasking:

   ```bash
   python3 scripts/run_mission.py --area 220 --live
   ```

5. Bring up the dashboard and confirm it loads **with no internet** on the
   laptop. It makes no external calls by design; if it needs a network,
   something has been changed and must be changed back.

## 3. Power-on and preflight (10 min)

Work through section C of the field-test checklist. The single machine-checkable
gate is:

```bash
ssh sar@aircraft
/opt/sar/venv/bin/python /opt/sar/scripts/run_onboard.py \
    --config /etc/sar/onboard.yaml --mode flight --preflight-only
echo $?          # 0 = fit to fly
```

If it exits non-zero it prints the specific refusal. **Do not work around it.**
The refusals map to real conditions:

| Refusal | What is actually wrong |
|---|---|
| `telemetry silence` | The companion computer is not talking to the autopilot — check the UART wiring and baud |
| `only N satellites` | Wait, or move away from the buildings/trees |
| `EKF not ready` | Let it settle; if it persists, the aircraft moved during initialisation |
| `no camera frames` | Camera not enumerated — `v4l2-ctl --list-devices`, check `/dev/sar-lwir` |
| `LWIR is not radiometric` | TLinear is off, or the camera is delivering AGC video |
| `cameras never synchronised` | Frame rates fighting; fix, or accept LWIR-only |
| `supervisor already latched` | A previous flight ended on a limit — investigate before flying |

## 4. Fly the sortie

```
Operator: "Area 220 by 220, north-east of home, survey 45 metres."
Pilot:    "Armed. Taking off."     (manual, verify hover, then hand to auto)
Observer: reads section F of the checklist, out loud, every 60 s
```

Live, the operator watches four things and nothing else:

1. **Survivor list** — new entries with position, sigma and triage tier
2. **Coverage grid** — pale cells are searched badly, not searched well
3. **Link** — queue depth; growing without bound means the link is gone
4. **Energy margin** — the return-home reachability figure, not battery percent

### Standing decisions

| Situation | Action |
|---|---|
| A survivor is reported with sigma > 10 m | Task a confirmation pass before sending anyone |
| Queue depth climbing for > 60 s | Move the ground modem or bring the aircraft closer; the alerts are safe in the queue |
| Safety supervisor latches | RTL now. It latches only on a real limit and only an operator can clear it. |
| Wind reaches 10 m/s | Land. The hard limit is 12 and it is not a target. |
| Battery: energy margin approaches 1.35× | The aircraft will initiate return. Do not override it. |

## 5. Between sorties (10 min)

- [ ] Battery out, voltage logged, cooling before charge
- [ ] Copy `/var/log/sar/onboard_report_*.json` to the ground laptop
- [ ] Copy the ArduPilot `.bin` log
- [ ] Anything odd written in the field log, with the timestamp
- [ ] Clear a latched safety state **only** after understanding why it latched
- [ ] Fresh battery, redo section C of the checklist. Every time.

## 6. Payload drop procedure

1. Confirm the survivor with a **confirmation pass at 22 m**. Never drop on a
   survey-altitude detection alone.
2. Confirm the drop zone is clear of people, including the survivor —
   **the release point is deliberately offset**; the payload travels downrange
   during the fall (about 5 m from 45 m at 8 m/s).
3. The system computes the release solution from altitude, ground speed,
   heading and wind. The operator confirms; the aircraft releases.
4. Record the actual landing point. Compare with the prediction and put the
   error in the log — that number is how the drop model gets better.

## 7. Debrief, same day

- [ ] Detections vs known ground truth → recall and precision for the day
- [ ] Every false alarm characterised: what was it? (roof, vehicle, animal, rock)
- [ ] Position errors vs reported sigma — **was sigma conservative?** It must be.
- [ ] Link statistics: delivered, shed, retried
- [ ] Any parameter changed in the field committed to Git with a reason
- [ ] Anything that surprised anybody, written down before people go home

---

## Recovery procedures

**Lost link to the aircraft.** ArduPilot's own failsafe owns this: RC loss →
RTL, telemetry loss does not by itself trigger anything. The companion
computer's supervisor treats RC loss as a *caution* precisely so that two
systems do not both act. Wait for RTL. Do not chase it.

**Fly-away.** Kill switch. That is what it is for. A destroyed aircraft is an
acceptable outcome; an uncontrolled aircraft leaving the site is not.

**Landed out of sight.** The last reported position and its sigma are in the
report and on the dashboard. Search the sigma circle, not the pin.

**LiPo damage.** LiPo bag, away from anything flammable, do not put it in a
vehicle, monitor for 30 minutes.

**Injury.** Stop everything. The flight test is over for the day.

---

## Configuration management

The effective configuration is written into **every** sortie report, so any
result can be traced to the settings that produced it. That only works if the
settings live in version control:

```
configs/onboard.yaml              → /etc/sar/onboard.yaml   (companion computer)
configs/ardupilot_hardware.parm   → the flight controller
```

Change them in Git, deploy, and note the commit in the field log. A parameter
changed only on the aircraft is a parameter nobody will be able to explain in a
month.
