#!/usr/bin/env python3
"""End-to-end flight-stack test against ArduPilot SITL.

This is the "library-based simulation" the project is built around: a real
ArduPilot process, a real MAVLink 2 link, real guided-mode flight, and the same
client code that will drive the physical TBS Lucid H743 Wing.  No 3D graphics and
no game engine - the point is to test the algorithms (navigation, control
handoff, comms protocol behaviour, denial handling) against a flight stack that
has the same parameter validation, the same pre-arm checks and the same EKF as
the aircraft.

What it exercises
-----------------
 1. Launch ``arducopter`` SITL with ``configs/ardupilot_sitl.parm`` and attach.
 2. Wait for GNSS lock, EKF alignment and a set home - the three things that
    actually gate arming, as opposed to just waiting a fixed number of seconds.
 3. Start the external-nav feeder BEFORE arming.  Order matters: with
    ``VISO_TYPE=1`` configured, pre-arm requires the VisOdom backend to be
    healthy, and it only becomes healthy once samples arrive.
 4. Arm, take off, and fly a survey box by *streaming velocity setpoints* rather
    than stepping between waypoints - which is how a search pattern is actually
    flown, because every stop costs energy and every acceleration smears the
    thermal image.
 5. Drive an EKF source-set switch to ExternalNav through ``EkfSourceManager``,
    hold, then switch back, and record the navigation-quality inputs the
    geo-tagger would have consumed throughout.
 6. Fire the Drop-to-Confirm payload release (SERVO9 gripper).
 7. RTL, land, disarm, and report energy, link health and every decision the
    source manager made.

Usage
-----
    python scripts/sitl_flight_test.py                 # launch SITL and fly
    python scripts/sitl_flight_test.py --attach        # use a SITL already running
    python scripts/sitl_flight_test.py --speedup 4     # 4x faster than real time
    python scripts/sitl_flight_test.py --backend mini  # pure-Python, no ArduPilot

``--speedup`` above 1 runs the simulation faster than real time.  Leave it at 1
when measuring anything time-dependent, including control behaviour and the
geo-tagger's latency term.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sar.core.geo import GeoPoint                              # noqa: E402
from sar.mavlink.connection import (CommandError, LinkTimeout,  # noqa: E402
                                    MavConnection, PreArmError)
from sar.nav import (EkfSourceManager, ExternalNavFeeder,      # noqa: E402
                     NavQualityMonitor, NavState, SourceSetPolicy,
                     TelemetryVioSource)
from sar.sim.sitl_launch import (ArduPilotSitl, DEFAULT_HOME,  # noqa: E402
                                 SitlNotAvailable, find_ardupilot_binary)

log = logging.getLogger("sitl_flight_test")

# Metres of latitude/longitude per degree at the reference home.  Computed once
# rather than per call, and used instead of a full geodesic because a survey box
# is a few hundred metres across, where the flat approximation is accurate to
# well under a centimetre.
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(DEFAULT_HOME[0]))


# --------------------------------------------------------------------------- #
# Small reporting helpers
# --------------------------------------------------------------------------- #
class Check:
    """One named pass/fail observation, so the run produces a verdict."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
        return bool(ok)

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if ok)

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)

    def to_dict(self) -> Dict[str, Any]:
        return {"total": len(self.rows), "passed": self.passed,
                "failed": self.failed,
                "rows": [{"check": n, "ok": o, "detail": d} for n, o, d in self.rows]}


def local_to_latlon(north_m: float, east_m: float) -> Tuple[float, float]:
    lat = DEFAULT_HOME[0] + north_m / M_PER_DEG_LAT
    lon = DEFAULT_HOME[1] + east_m / M_PER_DEG_LON
    return lat, lon


def latlon_to_local(lat: float, lon: float) -> Tuple[float, float]:
    return ((lat - DEFAULT_HOME[0]) * M_PER_DEG_LAT,
            (lon - DEFAULT_HOME[1]) * M_PER_DEG_LON)


# --------------------------------------------------------------------------- #
# Survey pattern
# --------------------------------------------------------------------------- #
def survey_legs(north_m: float, east_m: float, length_m: float,
                legs: int, spacing_m: float, heading_deg: float = 0.0
                ) -> List[Tuple[float, float]]:
    """Boustrophedon waypoints in local metres, starting at ``(north, east)``.

    Generated as a list of targets rather than uploaded as a mission, because the
    sortie flies it in guided mode by streaming velocity: a mission would stop at
    each waypoint, and stopping costs energy and smears the thermal image.  The
    same list is what the coverage planner in ``sar/decision`` produces, so the
    flight test and the planner share one definition of a survey pattern.
    """
    out: List[Tuple[float, float]] = []
    h = math.radians(heading_deg)
    along = (math.cos(h), math.sin(h))
    across = (-math.sin(h), math.cos(h))
    for i in range(legs):
        sgn = 1.0 if i % 2 == 0 else -1.0
        base_n = north_m + across[0] * spacing_m * i
        base_e = east_m + across[1] * spacing_m * i
        for end in (0.0, 1.0):
            d = sgn * length_m * end
            out.append((base_n + along[0] * d, base_e + along[1] * d))
    return out


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #
def wait_flight_ready(conn: MavConnection, timeout: float = 90.0,
                      need_gps: bool = True) -> Dict[str, Any]:
    """Block until the vehicle would actually accept an arm command.

    Waiting a fixed number of seconds after boot is how "Arm: Need Alt Estimate"
    and "Arm: AHRS: waiting for home" turn up as unexplained failures.  This
    waits on the conditions themselves and reports which one was still missing if
    it times out.
    """
    report: Dict[str, Any] = {"heartbeat": False, "gps": False, "ekf": False,
                              "home": False, "altitude": False, "ok": False}
    end = time.time() + timeout
    conn.wait_heartbeat(timeout=min(timeout, 30.0))
    report["heartbeat"] = True
    while time.time() < end:
        conn.pump(0.1)
        t = conn.telemetry
        if need_gps and not report["gps"]:
            report["gps"] = (t.n_satellites >= 8 and t.gps_fix >= 3)
        elif not need_gps:
            report["gps"] = True
        report["ekf"] = bool(t.armed.ekf_ok and t.armed.position_ok
                             and t.armed.velocity_ok and t.armed.height_ok)
        report["home"] = bool(t.has_position and (t.lat or 0) != 0.0
                              and t.age("position") < 5.0)
        report["altitude"] = t.alt_rel_m is not None
        if all(report[k] for k in ("gps", "ekf", "home", "altitude")):
            report["ok"] = True
            return report
        time.sleep(0.2)
    return report


# --------------------------------------------------------------------------- #
# The flight
# --------------------------------------------------------------------------- #
def fly(conn: MavConnection, args: argparse.Namespace,
        feeder: ExternalNavFeeder, manager: EkfSourceManager,
        monitor: NavQualityMonitor) -> Dict[str, Any]:
    """Arm, survey, deny, recover, drop, land.  Returns the run record."""
    chk = Check()
    rec: Dict[str, Any] = {"home": list(DEFAULT_HOME), "speedup": args.speedup,
                           "altitude_m": args.altitude, "checks": None}
    nav_log: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []

    tel = conn.telemetry
    lat0, lon0 = tel.lat, tel.lon
    print(f"\n  home/origin: {lat0:.6f}, {lon0:.6f}  alt_rel={tel.alt_rel_m}")

    # ---- arm ------------------------------------------------------------- #
    print("\n[3] arm and take off")
    conn.set_mode("GUIDED")
    chk.add("GUIDED mode accepted", conn.mode == "GUIDED", conn.mode)
    armed = False
    arm_err = ""
    for attempt in range(args.arm_attempts):
        try:
            conn.arm(timeout=25)
            armed = True
            break
        except PreArmError as exc:
            arm_err = str(exc)
            # "Accels inconsistent" and "Need Alt Estimate" are settling
            # conditions, not configuration errors: wait and retry.  Anything
            # naming a parameter is a configuration error and will not fix
            # itself, so say so instead of burning the retries silently.
            if any(k in arm_err for k in ("Check EK3", "require ", "requires ",
                                          "below minimum", "Fence", "terrain")):
                log.error("configuration-class pre-arm failure: %s", arm_err)
                break
            log.warning("arm attempt %d refused, settling: %s", attempt + 1,
                        arm_err[-160:])
            time.sleep(4.0)
            conn.pump(0.5)
    chk.add("vehicle armed", armed, arm_err[-200:] if not armed else "")
    if not armed:
        rec["checks"] = chk.to_dict()
        return rec

    conn.guided_limits(timeout_s=args.guided_timeout_s, alt_min_m=2.0,
                       alt_max_m=args.altitude + 40.0,
                       horiz_max_m=args.fence_radius_m)
    chk.add("guided runaway limits accepted",
            conn._pending_guided_limits is None,
            "DO_GUIDED_LIMITS refused" if conn._pending_guided_limits else "")

    conn.takeoff(args.altitude)
    reached = conn.wait_altitude(args.altitude - 1.5, args.altitude + 2.0,
                                 timeout=args.takeoff_timeout_s)
    chk.add(f"climbed to {args.altitude:.0f} m AGL", reached,
            f"alt_rel={tel.alt_rel_m}")
    rec["takeoff_alt_m"] = tel.alt_rel_m

    # ---- survey by streamed velocity ------------------------------------- #
    print(f"\n[4] survey pattern: {args.legs} legs x {args.leg_length:.0f} m "
          f"at {args.speed:.1f} m/s, {args.altitude:.0f} m AGL")
    spacing = 2.0 * args.altitude * math.tan(math.radians(args.hfov_deg) / 2.0) \
        * args.overlap_factor
    wps = survey_legs(0.0, 0.0, args.leg_length, args.legs, spacing,
                      heading_deg=args.heading_deg)
    print(f"    swath {2.0*args.altitude*math.tan(math.radians(args.hfov_deg)/2.0):.1f} m, "
          f"leg spacing {spacing:.1f} m, {len(wps)} targets, "
          f"path {args.legs*args.leg_length:.0f} m")
    rec["pattern"] = {"legs": args.legs, "leg_length_m": args.leg_length,
                     "spacing_m": round(spacing, 2), "speed_ms": args.speed,
                     "altitude_m": args.altitude,
                     "path_length_m": args.legs * args.leg_length,
                     "waypoints": [[round(a, 2), round(b, 2)] for a, b in wps]}

    flown = 0.0
    t_start = time.time()
    prev: Optional[Tuple[float, float]] = None
    max_cross_track = 0.0
    leg_idx = 0
    for (tn, te) in wps:
        leg_idx += 1
        tlat, tlon = local_to_latlon(tn, te)
        target_deadline = time.time() + (args.leg_length / max(args.speed, 0.5)) \
            * args.leg_timeout_factor + 10.0
        while time.time() < target_deadline:
            conn.pump(0.02)
            n, e = latlon_to_local(tel.lat, tel.lon) if tel.has_position else (0.0, 0.0)
            # Stream velocity toward the target rather than issuing a position
            # command: the aircraft accelerates once and holds, instead of
            # stopping and starting at every waypoint.
            dn, de = tn - n, te - e
            dist = math.hypot(dn, de)
            if dist < args.waypoint_radius_m:
                break
            v = min(args.speed, max(0.6, dist / 2.0))
            conn.set_velocity_ned(vn=v * dn / dist, ve=v * de / dist,
                                  hold_alt_rel_m=args.altitude)
            # Perception-side bookkeeping: this is the loop a real sortie runs
            # the camera pipeline from.
            a = monitor.assess(tel, external_nav_hz=feeder.measured_hz,
                               external_nav_healthy=feeder.healthy,
                               external_nav_age_s=feeder.age_s,
                               flow_healthy=False, local_north=n, local_east=e)
            manager.update(a)
            nav_log.append({"t": round(time.time() - t_start, 2),
                            "state": a.state.value,
                            **monitor.nav_quality(tel, a)})
            if prev is not None:
                flown += math.hypot(n - prev[0], e - prev[1])
            prev = (n, e)
            # Cross-track error against the intended leg, which is what says
            # whether the velocity controller is actually holding a line.
            if leg_idx >= 2:
                pn, pe = wps[leg_idx - 2]
                ln, le = tn - pn, te - pe
                L = math.hypot(ln, le) or 1.0
                xtrack = abs((n - pn) * le - (e - pe) * ln) / L
                max_cross_track = max(max_cross_track, xtrack)
            trace.append({"t": round(time.time() - t_start, 2), "n": round(n, 2),
                          "e": round(e, 2), "alt": tel.alt_rel_m,
                          "v": round(tel.speed_ms, 2),
                          "nav": a.state.value})
            if len(trace) > 20000:
                break
            time.sleep(1.0 / args.control_hz)
        else:
            log.warning("leg %d did not complete within its deadline", leg_idx)

    conn.set_velocity_ned()
    conn.pump(0.3)
    dt_survey = time.time() - t_start
    print(f"    flown {flown:.0f} m in {dt_survey:.0f} s "
          f"({flown/max(dt_survey,1e-6):.2f} m/s average), "
          f"max cross-track {max_cross_track:.2f} m")
    chk.add("survey pattern completed", flown > 0.6 * args.legs * args.leg_length,
            f"{flown:.0f} m of {args.legs*args.leg_length:.0f} m planned")
    chk.add("cross-track error within a swath", max_cross_track < spacing,
            f"{max_cross_track:.2f} m vs {spacing:.1f} m spacing")
    rec["survey"] = {"flown_m": round(flown, 1), "seconds": round(dt_survey, 1),
                     "mean_speed_ms": round(flown / max(dt_survey, 1e-6), 2),
                     "max_cross_track_m": round(max_cross_track, 2)}

    # ---- GPS denial ------------------------------------------------------ #
    print("\n[5] GPS denial: switch EKF source set to ExternalNav")
    pre = monitor.assess(conn.telemetry, external_nav_hz=feeder.measured_hz,
                         external_nav_healthy=feeder.healthy,
                         external_nav_age_s=feeder.age_s)
    print(f"    before: state={pre.state.value} set={manager.current_set} "
          f"({SOURCE_NAME(manager.current_set)}) feeder={feeder.measured_hz:.0f} Hz")
    denied = False
    if args.deny:
        denied = simulate_denial(conn, monitor, manager, feeder, args, chk)
    rec["denial"] = {"attempted": bool(args.deny), "executed": denied,
                     "manager": manager.summary(),
                     "feeder": feeder.stats()}

    # ---- payload --------------------------------------------------------- #
    if args.drop_payload:
        print("\n[6] Drop-to-Confirm payload release (SERVO9 gripper)")
        try:
            conn.drop_payload(channel=args.gripper_channel, hold_open_s=0.5)
            chk.add("payload release command accepted", True,
                    f"SERVO{args.gripper_channel} 1900us then 1100us")
        except CommandError as exc:
            chk.add("payload release command accepted", False, str(exc))

    # ---- recover and land ------------------------------------------------ #
    print("\n[7] return to launch and land")
    conn.rtl()
    chk.add("RTL mode entered", conn.wait_mode("RTL", timeout=20), conn.mode)
    disarmed = conn.wait_disarmed(timeout=args.land_timeout_s)
    chk.add("landed and disarmed", disarmed,
            f"alt_rel={conn.telemetry.alt_rel_m} mode={conn.mode}")
    rec["landing"] = {"rtl_entered": conn.mode == "RTL" or disarmed,
                      "disarmed": disarmed,
                      "final_alt_rel_m": conn.telemetry.alt_rel_m}

    rec["nav_states_seen"] = sorted({r["state"] for r in nav_log})
    rec["nav_log_samples"] = len(nav_log)
    rec["nav_log"] = nav_log[::max(1, len(nav_log) // 200)]
    rec["trace"] = trace[::max(1, len(trace) // 400)]
    rec["link"] = conn.link.to_dict()
    rec["final_telemetry"] = conn.telemetry.to_dict()
    rec["checks"] = chk.to_dict()
    return rec


def SOURCE_NAME(code: int) -> str:
    from sar.mavlink.protocol import SOURCE_SET
    return SOURCE_SET.name(code)


def simulate_denial(conn: MavConnection, monitor: NavQualityMonitor,
                    manager: EkfSourceManager, feeder: ExternalNavFeeder,
                    args: argparse.Namespace, chk: Check) -> bool:
    """Force the manager to switch to ExternalNav and hold it there.

    Two ways to produce a denial, and they test different things.

    ``--deny inject`` stops the GNSS simulation inside SITL, so the vehicle's own
    EKF sees the satellites disappear.  That is the honest test: it exercises
    ArduPilot's detection, not ours.

    ``--deny policy`` leaves GNSS alone and drives the manager directly, which
    tests the switching logic - hysteresis, the refusal to switch to an unhealthy
    source, the recovery counter - without depending on the simulator's GPS
    model.  Both are run in CI; only the first needs a real EKF.
    """
    mode = args.deny
    switched = False
    if mode == "inject":
        before = conn.param_get("SIM_GPS_ENABLE", timeout=2.0)
        conn.param_set("SIM_GPS_ENABLE", 0.0)
        # Also drop the satellite count so the fix type falls away, which is what
        # the monitor keys on.
        conn.param_set("SIM_GPS_NUMSATS", 0.0)
        log.info("GPS denial injected (SIM_GPS_ENABLE was %s)", before)
    deadline = time.time() + args.deny_seconds
    states: List[str] = []
    while time.time() < deadline:
        conn.pump(0.05)
        tel = conn.telemetry
        n, e = latlon_to_local(tel.lat, tel.lon) if tel.has_position else (0.0, 0.0)
        a = monitor.assess(tel, external_nav_hz=feeder.measured_hz,
                           external_nav_healthy=feeder.healthy,
                           external_nav_age_s=feeder.age_s,
                           local_north=n, local_east=e)
        d = manager.update(a)
        states.append(a.state.value)
        if d.get("switched"):
            switched = True
            print(f"    -> switched to set {manager.current_set} "
                  f"({SOURCE_NAME(manager.current_set)}): {d['reason']}")
        # Keep the aircraft flying straight through the denial: a survey does not
        # stop because the sky got worse.
        conn.set_velocity_ned(vn=args.speed * 0.6, ve=0.0,
                              hold_alt_rel_m=args.altitude)
        time.sleep(1.0 / args.control_hz)
    conn.set_velocity_ned(vn=0.0, ve=0.0, hold_alt_rel_m=args.altitude)
    time.sleep(1.0)
    conn.pump(0.5)
    nq = monitor.nav_quality(conn.telemetry)
    print(f"    during denial: states seen={sorted(set(states))} "
          f"set={manager.current_set} ({SOURCE_NAME(manager.current_set)})")
    print(f"    nav_quality -> pos_sigma={nq['pos_sigma_m']:.2f} m "
          f"att_sigma={nq['att_sigma_deg']:.2f} deg source={nq['source']} "
          f"since_fix={nq['seconds_since_fix']:.1f}s")
    chk.add("denial detected by the monitor",
            any(s in ("external_nav", "flow_only", "lost") for s in states),
            f"states seen: {sorted(set(states))}")
    chk.add("geo-tag uncertainty widened during denial",
            nq["pos_sigma_m"] > 1.5 * 1.2,
            f"pos_sigma {nq['pos_sigma_m']:.2f} m vs ~1.2 m with a good fix")
    if mode == "inject":
        chk.add("source set switched to ExternalNav",
                manager.current_set == SourceSetPolicy().external_nav,
                f"set={manager.current_set} ({SOURCE_NAME(manager.current_set)})")

    # ---- recovery ----
    print("    restoring GNSS")
    if mode == "inject":
        conn.param_set("SIM_GPS_ENABLE", 1.0)
        conn.param_set("SIM_GPS_NUMSATS", 30.0)
    rec_deadline = time.time() + args.recover_seconds
    while time.time() < rec_deadline:
        conn.pump(0.05)
        tel = conn.telemetry
        n, e = latlon_to_local(tel.lat, tel.lon) if tel.has_position else (0.0, 0.0)
        a = monitor.assess(tel, external_nav_hz=feeder.measured_hz,
                           external_nav_healthy=feeder.healthy,
                           external_nav_age_s=feeder.age_s,
                           local_north=n, local_east=e)
        d = manager.update(a)
        if d.get("switched"):
            print(f"    -> switched back to set {manager.current_set} "
                  f"({SOURCE_NAME(manager.current_set)})")
        conn.set_velocity_ned(vn=0.0, ve=0.0, hold_alt_rel_m=args.altitude)
        time.sleep(1.0 / args.control_hz)
        if manager.current_set == SourceSetPolicy().gnss_ok:
            break
    conn.set_velocity_ned()
    chk.add("recovered to the GNSS source set",
            manager.current_set == SourceSetPolicy().gnss_ok or mode != "inject",
            f"set={manager.current_set} ({SOURCE_NAME(manager.current_set)})")
    return switched


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("sitl", "attach", "mini"), default="sitl",
                    help="sitl: launch ArduPilot. attach: use one already "
                         "running on --target. mini: pure-Python MiniSITL.")
    ap.add_argument("--target", default=None,
                    help="connection string when --backend attach "
                         "(default tcp:127.0.0.1:5760)")
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="SITL rate multiplier (default 1 = real time)")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--keep-sitl", action="store_true",
                    help="leave the SITL process and its scratch dir behind")
    ap.add_argument("--scratch", default=str(REPO / ".sitl_run"))
    ap.add_argument("-v", "--verbose", action="store_true")

    g = ap.add_argument_group("flight")
    g.add_argument("--altitude", type=float, default=35.0, help="survey AGL [m]")
    g.add_argument("--speed", type=float, default=6.0, help="survey speed [m/s]")
    g.add_argument("--legs", type=int, default=4)
    g.add_argument("--leg-length", type=float, default=120.0)
    g.add_argument("--hfov", dest="hfov_deg", type=float, default=42.0)
    g.add_argument("--overlap-factor", type=float, default=0.7,
                   help="leg spacing as a fraction of the swath (0.7 = 30%% overlap)")
    g.add_argument("--heading-deg", type=float, default=0.0)
    g.add_argument("--control-hz", type=float, default=8.0)
    g.add_argument("--waypoint-radius-m", type=float, default=3.0)
    g.add_argument("--leg-timeout-factor", type=float, default=3.0)
    g.add_argument("--takeoff-timeout-s", type=float, default=60.0)
    g.add_argument("--land-timeout-s", type=float, default=180.0)
    g.add_argument("--guided-timeout-s", type=float, default=600.0)
    g.add_argument("--fence-radius-m", type=float, default=600.0)
    g.add_argument("--arm-attempts", type=int, default=4)

    d = ap.add_argument_group("GPS denial")
    d.add_argument("--deny", choices=("none", "inject", "policy"), default="inject",
                   help="inject: stop SITL's GPS. policy: drive the manager "
                        "directly. none: skip.")
    d.add_argument("--deny-seconds", type=float, default=12.0)
    d.add_argument("--recover-seconds", type=float, default=25.0)
    d.add_argument("--extnav-hz", type=float, default=30.0)
    d.add_argument("--extnav-dropout", type=float, default=0.0)

    p = ap.add_argument_group("payload")
    p.add_argument("--drop-payload", action="store_true", default=True)
    p.add_argument("--no-drop-payload", dest="drop_payload", action="store_false")
    p.add_argument("--gripper-channel", type=int, default=9)

    o = ap.add_argument_group("output")
    o.add_argument("--json", default=str(REPO / "artifacts" / "sitl_flight_test.json"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("sar.mavlink",):
        if not args.verbose:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    print("=" * 78)
    print("SAHYOG flight-stack test")
    print("=" * 78)
    sitl: Optional[ArduPilotSitl] = None
    conn: Optional[MavConnection] = None
    feeder: Optional[ExternalNavFeeder] = None
    rc = 1
    rec: Dict[str, Any] = {"args": {k: v for k, v in vars(args).items()}}

    try:
        # ---- backend selection ------------------------------------------ #
        if args.backend == "mini":
            from sar.sim.sitl import MiniSITL
            print(f"\n[0] backend: MiniSITL (pure Python, no ArduPilot binary)")
            mini = MiniSITL(home=GeoPoint(*DEFAULT_HOME[:3]))
            mini.start()
            conn = mini.connect()
            rec["backend"] = "mini"
        else:
            binary = find_ardupilot_binary()
            if binary is None:
                print("\n  No ArduPilot SITL binary found.  Set SAR_ARDUPILOT_BIN, "
                      "or use --backend mini for the pure-Python vehicle model.")
                return 2
            print(f"\n[0] backend: ArduPilot SITL")
            print(f"    binary  {binary}")
            print(f"    params  {REPO/'configs'/'ardupilot_sitl.parm'}")
            print(f"    home    {DEFAULT_HOME[0]}, {DEFAULT_HOME[1]}, "
                  f"{DEFAULT_HOME[2]} m, hdg {DEFAULT_HOME[3]}")
            print(f"    speedup {args.speedup:g}x")
            if args.backend == "attach":
                target = args.target or f"tcp:127.0.0.1:{5760+10*args.instance}"
                conn = MavConnection(target)
                rec["backend"] = "attach"
            else:
                sitl = ArduPilotSitl(speedup=args.speedup, instance=args.instance,
                                     scratch=None if args.keep_sitl else args.scratch)
                sitl.start()
                conn = sitl.connect(timeout=45)
                rec["backend"] = "sitl"
                rec["sitl"] = {k: v for k, v in sitl.info().items()
                               if k != "prearm_blockers"}

        chk = Check()
        # ---- link -------------------------------------------------------- #
        print("\n[1] link and telemetry")
        conn.request_streams()
        time.sleep(2.0)
        conn.pump(1.0)
        chk.add("MAVLink heartbeat received", conn.telemetry.armed.mode != "UNKNOWN",
                conn.telemetry.armed.describe())
        chk.add("telemetry streams flowing", conn.link.rx_packets > 20,
                f"{conn.link.rx_packets} messages in 3 s")
        chk.add("link loss under 1%", conn.link.loss_pct < 1.0,
                f"{conn.link.loss_pct:.2f}%")

        # ---- readiness --------------------------------------------------- #
        print("\n[2] pre-arm readiness")
        ready = wait_flight_ready(conn, timeout=90.0)
        rec["readiness"] = ready
        chk.add("GNSS lock", ready["gps"],
                f"{conn.telemetry.n_satellites} sats fix={conn.telemetry.gps_fix} "
                f"hdop={conn.telemetry.hdop:.2f}")
        chk.add("EKF aligned (attitude/position/velocity/height)", ready["ekf"],
                conn.telemetry.armed.describe())
        chk.add("home set", ready["home"])
        chk.add("altitude estimate available", ready["altitude"],
                f"alt_rel={conn.telemetry.alt_rel_m}")
        if not ready["ok"]:
            print("    readiness report:", ready)
            rec["checks"] = chk.to_dict()
            return 3

        # ---- external nav feeder, BEFORE arming -------------------------- #
        print("\n[2b] external-nav feeder (must precede arming)")
        src = TelemetryVioSource(conn, pos_sigma_m=0.15, vel_sigma_ms=0.15,
                                 latency_s=0.03)
        src.set_origin_from_telemetry()
        feeder = ExternalNavFeeder(conn, src, rate_hz=args.extnav_hz,
                                   dropout=args.extnav_dropout, use_odometry=True)
        feeder.start()
        t0 = time.time()
        while feeder.measured_hz < 10.0 and time.time() - t0 < 8.0:
            time.sleep(0.2)
        chk.add("external nav publishing at rate", feeder.measured_hz >= 15.0,
                f"{feeder.measured_hz:.0f} Hz of {args.extnav_hz:.0f} Hz target")
        chk.add("VisOdom backend healthy before arming", feeder.healthy,
                str(feeder.stats()))
        rec["feeder_at_start"] = feeder.stats()

        monitor = NavQualityMonitor(
            denial_zones=[], hysteresis_s=1.5,
            external_nav_min_hz=15.0, external_nav_max_age_s=0.5)
        manager = EkfSourceManager(conn, monitor, extnav=feeder,
                                   policy=SourceSetPolicy())

        # ---- the flight -------------------------------------------------- #
        rec.update(fly(conn, args, feeder, manager, monitor))
        chk.rows.extend([(r["check"], r["ok"], r["detail"])
                         for r in rec["checks"]["rows"]])
        rec["checks"] = chk.to_dict()

        print("\n" + "=" * 78)
        print(f"RESULT  {chk.passed} passed, {chk.failed} failed, "
              f"{len(chk.rows)} checks")
        for name, ok, detail in chk.rows:
            if not ok:
                print(f"  FAILED: {name} - {detail}")
        print(f"nav states seen: {rec.get('nav_states_seen')}")
        print(f"source manager: {json.dumps(manager.summary(), default=str)[:400]}")
        print("=" * 78)
        rc = 0 if chk.failed == 0 else 1

    except (SitlNotAvailable, LinkTimeout) as exc:
        print(f"\nFATAL: {exc}")
        rc = 4
    except KeyboardInterrupt:
        print("\ninterrupted")
        rc = 130
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nFATAL: {type(exc).__name__}: {exc}")
        rc = 5
    finally:
        if feeder is not None:
            feeder.stop()
        if conn is not None:
            try:
                if conn.telemetry.armed.armed:
                    conn.land()
                    conn.wait_disarmed(timeout=20)
            except Exception:
                pass
            conn.close()
        if sitl is not None:
            if args.keep_sitl:
                sitl.conn = None
                print(f"\nSITL left running (pid {sitl.proc.pid if sitl.proc else '?'})")
            else:
                sitl.stop()
        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            print(f"wrote {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
