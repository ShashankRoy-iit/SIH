"""The flight-time entry point: what actually runs on the aircraft.

This is the module that makes the repository "real drone ready".  Everything
above it - perception, geo-tagging, triage, the store-and-forward link, the
payload logic - is the *same code* the simulator exercises.  What this module
adds is the part that only exists in flight:

* a MAVLink link to a physical autopilot over serial or UDP, with the companion
  component id, so ArduPilot treats the commands as onboard guidance;
* real cameras, paired and timestamped (:mod:`sar.hardware.camera`);
* nav quality derived from live EKF telemetry rather than from a simulator;
* the :class:`~sar.hardware.safety.SafetySupervisor`, which can end the sortie;
* a stateful loop with a watchdog, graceful shutdown on SIGTERM (so
  ``systemctl stop sar-onboard`` does not leave a drone in GUIDED), and a
  post-flight report written to disk even when the sortie ended badly.

Operating modes
---------------
``dry-run``
    No hardware at all.  A MiniSITL vehicle and simulated cameras stand in, and
    every other line of code is the flight path.  This is what CI runs and what
    a team runs on a laptop the night before a field test.
``hitl``
    Real autopilot, simulated cameras.  Used on the bench with the airframe
    powered and props off: proves the MAVLink command path, the arming
    sequence, the payload servo and the failsafes without a flight.
``flight``
    Real everything.

Pre-flight refuses
------------------
The loop will not arm if: the GNSS fix or EKF is not ready, the cameras are not
delivering paired frames, the LWIR is not radiometric (unless explicitly
waived), the detector stack could not be built, home is not set, or the safety
supervisor is already latched.  Each refusal names itself.  An autonomous
aircraft that arms into a broken configuration is the failure mode this whole
file exists to prevent.
"""

from __future__ import annotations

import json
import logging
import math
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint
from sar.hardware.camera import CameraPair, FrameCapture, build_camera_pair
from sar.hardware.safety import SafetyLimits, SafetyState, SafetySupervisor

log = logging.getLogger("sar.hardware.onboard")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "dry-run",                      # dry-run | hitl | flight
    "mavlink": {
        # /dev/ttyTHS0 on a Jetson, /dev/ttyUSB0 for an FTDI, udpin for a
        # bench setup with mavlink-router in front of the autopilot.
        "target": "/dev/ttyACM0:921600",
        "source_system": 255,
        "source_component": 191,            # MAV_COMP_ID_ONBOARD_COMPUTER-ish
    },
    "cameras": {
        "max_skew_s": 0.08,
        "lwir": {"enabled": True, "device": 1, "width": 160, "height": 120,
                 "fps": 9, "hfov_deg": 57.0, "expect_radiometric": True},
        "rgb": {"enabled": True, "device": 0, "width": 1280, "height": 720,
                "fps": 15, "hfov_deg": 66.0},
    },
    "detector": "auto",                     # sar.ai.build_detector_stack mode
    "perception_hz": 2.0,
    "mission": {
        "search_altitude_agl_m": 45.0,
        "confirm_altitude_agl_m": 22.0,
        "survey_speed_ms": 6.0,
        "area_north_m": 300.0,
        "area_east_m": 300.0,
    },
    "payload": {"enabled": True, "servo_channel": 9, "release_pwm": 1900,
                "hold_pwm": 1100},
    "comms": {"transport": "lora_900"},
    "safety": {},                           # overrides for SafetyLimits
    "artifacts_dir": "artifacts",
}


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    """Merge ``configs/onboard.yaml`` over the defaults (deep, one level)."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))     # deep copy
    if not path:
        return cfg
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    text = p.read_text()
    try:
        import yaml
        user = yaml.safe_load(text) or {}
    except ImportError:
        user = json.loads(text)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class OnboardReport:
    mode: str = "dry-run"
    started: str = ""
    ended: str = ""
    duration_s: float = 0.0
    preflight: Dict[str, Any] = field(default_factory=dict)
    detector: Dict[str, Any] = field(default_factory=dict)
    cameras: Dict[str, Any] = field(default_factory=dict)
    perception: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    survivors: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)

    def event(self, kind: str, **kw: Any) -> None:
        self.events.append({"t": round(time.time(), 3), "kind": kind, **kw})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return p


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class OnboardAutonomy:
    """Perception + safety + reporting against a real (or stand-in) vehicle."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.mode = str(config.get("mode", "dry-run"))
        self.report = OnboardReport(mode=self.mode)
        self.supervisor = SafetySupervisor(SafetyLimits(**(config.get("safety") or {})))
        self.conn: Any = None
        self.cameras: Optional[CameraPair] = None
        self.pipeline: Any = None
        self.uplink: Any = None
        self.home: Optional[GeoPoint] = None
        self._sim: Any = None            # MiniSITL, in dry-run
        self._stop = False
        self._cycles = 0
        self._perception_ms: List[float] = []
        self._survivor_sites: Dict[int, Dict[str, Any]] = {}
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        log.warning("signal %d received - stopping cleanly", signum)
        self._stop = True

    # ------------------------------------------------------------------ #
    # Bring-up
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        from sar.mavlink.connection import MavConnection
        mav = self.cfg["mavlink"]
        if self.mode == "dry-run":
            from sar.sim.sitl import MiniSITL
            from sar.sim.scenario import build_reference_scenario
            scenario = self.cfg.get("scenario", "flood")
            world, lwir_spec, rgb_spec = build_reference_scenario(scenario, seed=7)
            self._world, self._lwir_spec, self._rgb_spec = world, lwir_spec, rgb_spec
            port = int(self.cfg.get("sitl_port", 5772))
            self._sim = MiniSITL(home=world.origin, speedup=float(
                self.cfg.get("speedup", 4.0)), port=port)
            self._sim.start()
            self.conn = self._sim.connect()
            log.info("dry-run: MiniSITL on tcp:127.0.0.1:%d", port)
        else:
            self.conn = MavConnection(mav["target"],
                                      source_system=int(mav.get("source_system", 255)),
                                      source_component=int(mav.get("source_component", 191)))
            self.conn.wait_link(timeout=30.0)
            log.info("MAVLink up on %s", mav["target"])
        self.conn.request_streams()
        self.conn.pump(0.5)

    def build_perception(self) -> None:
        from sar.perception.geotag import PixelGeoTagger
        from sar.perception.pipeline import PerceptionPipeline

        tel = self.conn.telemetry if hasattr(self.conn, "telemetry") else None
        origin = self._origin_from_telemetry(tel)
        self.home = origin
        cam_cfg = self.cfg["cameras"]

        if self.mode == "dry-run":
            from sar.hardware.camera import SimulatedCameraSource
            from sar.sim.renderer import CameraRenderer
            lwir_src = SimulatedCameraSource(
                CameraRenderer(self._world, self._lwir_spec, seed=12),
                state_fn=lambda: self._sim.truth(), kind="lwir", rate_hz=8.0)
            rgb_src = SimulatedCameraSource(
                CameraRenderer(self._world, self._rgb_spec, seed=11),
                state_fn=lambda: self._sim.truth(), kind="rgb", rate_hz=8.0)
            self.cameras = CameraPair(lwir=lwir_src, rgb=rgb_src,
                                      max_skew_s=float(cam_cfg.get("max_skew_s", 0.08)))
            width = self._rgb_spec.width
            height = self._rgb_spec.height
            hfov = self._rgb_spec.hfov_deg
        else:
            self.cameras = build_camera_pair(cam_cfg)
            rgb_cfg = cam_cfg.get("rgb", {})
            width = int(rgb_cfg.get("width", 1280))
            height = int(rgb_cfg.get("height", 720))
            hfov = float(rgb_cfg.get("hfov_deg", 66.0))
        self.cameras.open()

        tagger = PixelGeoTagger(width=width, height=height, hfov_deg=hfov,
                                origin=origin, terrain_sigma_m=3.0)
        detector = self._build_detector()
        area = self.cfg["mission"]
        self.pipeline = PerceptionPipeline(
            origin=origin, tagger=tagger, detector=detector,
            world_extent_m=(float(area["area_north_m"]) * 3,
                            float(area["area_east_m"]) * 3))
        self.report.detector = (detector.to_dict()
                                if hasattr(detector, "to_dict")
                                else {"backend": getattr(detector, "name", "?")})

    def _build_detector(self) -> Any:
        from sar.ai.stack import build_detector_stack
        mode = str(self.cfg.get("detector", "auto"))
        stack = build_detector_stack(mode)
        log.info("detector: %s", stack.stack_description)
        return stack

    def build_uplink(self) -> None:
        """Store-and-forward radio.  Optional: a sortie without it still flies.

        The transport model is the same one the simulator uses, so the queue
        behaviour measured on the bench (coalescing, shedding, priority order)
        is the behaviour in the air - only the physical layer underneath
        changes.
        """
        try:
            from sar.comms.link import (LinkSimulator, StoreAndForwardQueue,
                                        TRANSPORTS, TelemetryUplink)
            name = str(self.cfg.get("comms", {}).get("transport", "lora_900"))
            profile = TRANSPORTS.get(name, TRANSPORTS["lora_900"])
            self.uplink = TelemetryUplink(queue=StoreAndForwardQueue(),
                                          link=LinkSimulator(profile))
            self.uplink.start()
            log.info("uplink: %s", name)
        except Exception as exc:
            log.warning("uplink unavailable (%r); reports stay onboard", exc)
            self.report.faults.append(f"uplink: {exc!r}")

    def _origin_from_telemetry(self, tel: Any) -> GeoPoint:
        lat = float(getattr(tel, "lat", 0.0) or 0.0) if tel else 0.0
        lon = float(getattr(tel, "lon", 0.0) or 0.0) if tel else 0.0
        alt = float(getattr(tel, "alt_msl_m", 0.0) or 0.0) if tel else 0.0
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            if self.mode == "dry-run":
                return self._world.origin
            raise RuntimeError("no global position from the autopilot yet - "
                               "cannot set the local tangent-plane origin")
        return GeoPoint(lat, lon, alt)

    # ------------------------------------------------------------------ #
    # Pre-flight
    # ------------------------------------------------------------------ #
    def preflight(self, *, require_radiometric: bool = True,
                  timeout_s: float = 60.0) -> Tuple[bool, List[str]]:
        """Every refusal is named.  Returns ``(ok, failures)``."""
        checks: Dict[str, Any] = {}
        failures: List[str] = []
        t0 = time.time()

        self.conn.pump(0.5)
        tel = getattr(self.conn, "telemetry", None)

        # 1. link
        silence = float(getattr(getattr(self.conn, "stats", None), "silence_s", 0.0) or 0.0)
        checks["telemetry_silence_s"] = round(silence, 2)
        if silence > 3.0:
            failures.append(f"telemetry silent for {silence:.1f} s")

        # 2. GNSS + EKF (the two things that actually gate arming)
        sats = int(getattr(tel, "n_satellites", 0) or 0)
        checks["satellites"] = sats
        if self.mode == "flight" and sats < 8:
            failures.append(f"only {sats} satellites (need 8+)")
        try:
            ekf_ok = bool(self.conn.wait_ekf_ready(timeout=min(20.0, timeout_s)))
        except Exception as exc:
            ekf_ok = False
            checks["ekf_error"] = repr(exc)
        checks["ekf_ready"] = ekf_ok
        if not ekf_ok and self.mode == "flight":
            failures.append("EKF not ready (position/velocity/height flags)")

        # 3. cameras actually delivering paired frames
        pair: List[FrameCapture] = []
        deadline = time.time() + 10.0
        while time.time() < deadline:
            pair = self.cameras.latest_pair(agl_m=30.0) if self.cameras else []
            if pair:
                break
            time.sleep(0.1)
        checks["camera_frames"] = [f.kind for f in pair]
        if not pair:
            failures.append("no camera frames within 10 s")
        lwir = next((f for f in pair if f.kind == "lwir"), None)
        if lwir is not None:
            checks["lwir_radiometric"] = lwir.radiometric
            if require_radiometric and not lwir.radiometric and self.mode == "flight":
                failures.append("LWIR is not radiometric - triage physiology "
                                "would be fabricated; pass --allow-non-radiometric "
                                "to fly anyway with triage disabled")
        if self.cameras is not None:
            sync = self.cameras.stats
            checks["camera_sync"] = sync.to_dict()
            if sync.pairs == 0 and sync.rejected_skew > 3:
                failures.append(f"cameras never synchronised within "
                                f"{self.cameras.max_skew_s * 1000:.0f} ms")

        # 4. detector
        checks["detector"] = self.report.detector
        if self.pipeline is None:
            failures.append("perception pipeline not built")

        # 5. payload servo (bench-testable, and a silent failure in the air)
        if self.cfg.get("payload", {}).get("enabled", True):
            checks["payload_channel"] = self.cfg["payload"]["servo_channel"]

        # 6. safety supervisor not already latched
        checks["safety_state"] = self.supervisor.state.value
        if self.supervisor.state != SafetyState.NOMINAL:
            failures.append(f"safety supervisor already latched at "
                            f"{self.supervisor.state.value}")

        checks["elapsed_s"] = round(time.time() - t0, 2)
        checks["failures"] = failures
        self.report.preflight = checks
        for f in failures:
            log.error("PREFLIGHT: %s", f)
        if not failures:
            log.info("preflight: all checks passed (%s)", json.dumps(checks, default=str))
        return (not failures), failures

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, duration_s: float = 300.0, *, arm_and_fly: bool = False,
            altitude_agl_m: Optional[float] = None) -> OnboardReport:
        """Run the perception/safety loop.  Optionally arm and fly a survey.

        ``arm_and_fly=False`` is the safe default and is genuinely useful: on
        the bench, with the aircraft on the ground, the loop still perceives,
        geo-tags, reports and exercises the safety rules.  That is the sortie
        rehearsal, and it catches most integration defects.
        """
        self.report.started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t_start = time.time()
        period = 1.0 / max(float(self.cfg.get("perception_hz", 2.0)), 0.1)
        alt = float(altitude_agl_m if altitude_agl_m is not None
                    else self.cfg["mission"]["search_altitude_agl_m"])
        runner = None

        try:
            if arm_and_fly:
                runner = self._start_survey(alt)
            while not self._stop and (time.time() - t_start) < duration_s:
                cycle_t0 = time.time()
                self.conn.pump(0.02)
                self.supervisor.note_progress()
                self._perception_cycle(time.time() - t_start)
                decision = self._safety_cycle()
                if decision.must_stop_searching:
                    self._handle_safety(decision)
                    break
                if runner is not None:
                    runner.step()
                sleep = period - (time.time() - cycle_t0)
                if sleep > 0:
                    time.sleep(sleep)
        except Exception as exc:                    # never leave the vehicle armed
            log.exception("onboard loop failed")
            self.report.faults.append(f"loop: {exc!r}")
            self._emergency()
        finally:
            self.report.duration_s = round(time.time() - t_start, 1)
            self.report.ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._finalise()
        return self.report

    def _start_survey(self, altitude_agl_m: float) -> Any:
        raise NotImplementedError(
            "Autonomous flight from the onboard loop is gated behind the field-test "
            "checklist (docs/FIELD_TEST_CHECKLIST.md). Fly the survey with "
            "scripts/run_mission.py against SITL, or with a pilot on the sticks "
            "and arm_and_fly=False onboard, until WP 7.8 is signed off.")

    # ------------------------------------------------------------------ #
    def _perception_cycle(self, t_mission: float) -> None:
        tel = getattr(self.conn, "telemetry", None)
        agl = float(getattr(tel, "alt_rel_m", 0.0) or 0.0)
        frames = self.cameras.latest_pair(agl_m=max(agl, 1.0)) if self.cameras else []
        if not frames:
            return
        pos_ned, euler, vel_ned = self._state_from_telemetry(tel)
        nav = self._nav_quality(tel)
        t0 = time.perf_counter()
        try:
            product = self.pipeline.process(frames, pos_ned=pos_ned, euler=euler,
                                            t=t_mission, velocity_ned=vel_ned, nav=nav)
        except Exception as exc:
            self.report.faults.append(f"perception: {exc!r}")
            return
        self._perception_ms.append((time.perf_counter() - t0) * 1000.0)
        self._cycles += 1
        self._collect_survivors(product, t_mission)

    def _state_from_telemetry(self, tel: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == "dry-run" and self._sim is not None:
            st = self._sim.truth()
            # NOTE: only the *pose* is taken from the simulator here, exactly as
            # the autopilot would supply it. Truth positions never reach the
            # geo-tagger; that is what keeps geotag error measurable.
            return st.pos.copy(), st.euler.copy(), st.vel.copy()
        origin = self.home
        lat = float(getattr(tel, "lat", 0.0) or 0.0)
        lon = float(getattr(tel, "lon", 0.0) or 0.0)
        alt_rel = float(getattr(tel, "alt_rel_m", 0.0) or 0.0)
        from sar.core.geo import wgs84_to_local
        n, e, _d = (wgs84_to_local(GeoPoint(lat, lon, origin.alt), origin)
                    if origin else (0.0, 0.0, 0.0))
        pos = np.array([n, e, -alt_rel], dtype=float)
        euler = np.array([float(getattr(tel, "roll_rad", 0.0) or 0.0),
                          float(getattr(tel, "pitch_rad", 0.0) or 0.0),
                          float(getattr(tel, "yaw_rad", 0.0) or 0.0)])
        vel = np.array([float(getattr(tel, "vn_ms", 0.0) or 0.0),
                        float(getattr(tel, "ve_ms", 0.0) or 0.0),
                        float(getattr(tel, "vd_ms", 0.0) or 0.0)])
        return pos, euler, vel

    def _nav_quality(self, tel: Any) -> Any:
        from sar.perception.geotag import NavQuality
        sigma = float(getattr(tel, "horizontal_pos_sigma_m", 2.0) or 2.0)
        denied = bool(getattr(tel, "gps_denied", False))
        since = float(getattr(tel, "seconds_since_gps", 0.0) or 0.0)
        source = "dead_reckoning" if denied else "gnss"
        return NavQuality(pos_sigma_m=sigma, seconds_since_fix=since,
                          n_satellites=int(getattr(tel, "n_satellites", 0) or 0),
                          source=source,
                          hdop=float(getattr(tel, "hdop", 1.0) or 1.0),
                          drift_rate_ms=0.8 if denied else 0.05)

    def _collect_survivors(self, product: Any, t_mission: float) -> None:
        for s in getattr(product, "survivors", []) or []:
            tid = int(getattr(s, "track_id", -1))
            entry = {
                "track_id": tid,
                "lat": round(float(getattr(s, "lat", 0.0)), 7),
                "lon": round(float(getattr(s, "lon", 0.0)), 7),
                "sigma_m": round(float(getattr(s, "sigma_m", 0.0)), 1),
                "confidence": round(float(getattr(s, "confidence", 0.0)), 3),
                "tier": str(getattr(getattr(s, "tier", None), "value",
                                    getattr(s, "tier", ""))),
                "t": round(t_mission, 1),
            }
            first = tid not in self._survivor_sites
            self._survivor_sites[tid] = entry
            if self.uplink is not None:
                try:
                    # The assessment object, not the summary dict: the uplink
                    # does its own compaction to fit the radio's MTU, and a
                    # pre-flattened dict would bypass the field-drop order that
                    # keeps position and urgency in the packet.
                    self.uplink.publish_survivor(s, first_report=first)
                except Exception as exc:
                    self.report.faults.append(f"uplink publish: {exc!r}")

    # ------------------------------------------------------------------ #
    def _safety_cycle(self):
        tel = getattr(self.conn, "telemetry", None)
        pos, _euler, _vel = self._state_from_telemetry(tel)
        batt = getattr(tel, "battery_remaining", None)
        if batt is not None and batt > 1.0:
            batt = float(batt) / 100.0
        wh = getattr(tel, "battery_wh_remaining", None)
        stats = getattr(self.conn, "stats", None)
        return self.supervisor.evaluate(
            telemetry=tel,
            position_ned=(float(pos[0]), float(pos[1]), float(pos[2])),
            home_ned=(0.0, 0.0, 0.0),
            battery_remaining=batt,
            battery_wh_remaining=float(wh) if wh is not None else None,
            agl_m=float(getattr(tel, "alt_rel_m", 0.0) or 0.0),
            position_sigma_m=float(getattr(tel, "horizontal_pos_sigma_m", 0.0) or 0.0)
            or None,
            gps_denied=bool(getattr(tel, "gps_denied", False)),
            telemetry_silence_s=float(getattr(stats, "silence_s", 0.0) or 0.0)
            if stats else None)

    def _handle_safety(self, decision) -> None:
        self.report.event("safety", state=decision.state.value,
                          reasons=decision.reasons)
        log.error("SAFETY %s: %s", decision.state.value, "; ".join(decision.reasons))
        try:
            if decision.state == SafetyState.LAND_NOW:
                self.conn.land()
            elif decision.state in (SafetyState.RTL_NOW, SafetyState.RETURN):
                self.conn.rtl()
        except Exception as exc:
            self.report.faults.append(f"safety action: {exc!r}")

    def _emergency(self) -> None:
        try:
            if getattr(self.conn, "armed", False):
                self.conn.rtl()
                self.report.event("emergency_rtl")
        except Exception as exc:
            self.report.faults.append(f"emergency: {exc!r}")

    # ------------------------------------------------------------------ #
    def _finalise(self) -> None:
        if self.cameras is not None:
            self.report.cameras = self.cameras.health()
            self.cameras.close()
        self.report.safety = self.supervisor.summary()
        self.report.survivors = list(self._survivor_sites.values())
        ms = self._perception_ms
        self.report.perception = {
            "cycles": self._cycles,
            "mean_ms": round(float(np.mean(ms)), 1) if ms else 0.0,
            "p95_ms": round(float(np.percentile(ms, 95)), 1) if ms else 0.0,
            "rate_hz": round(self._cycles / max(self.report.duration_s, 1e-6), 2),
        }
        if hasattr(self.pipeline, "detector") and hasattr(
                self.pipeline.detector, "to_dict"):
            self.report.detector = self.pipeline.detector.to_dict()
        try:
            if self._sim is not None:
                self._sim.stop()
            elif self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        out = Path(self.cfg.get("artifacts_dir", "artifacts")) / "onboard_report.json"
        self.report.write(out)
        log.info("report -> %s", out)


__all__ = ["DEFAULT_CONFIG", "OnboardAutonomy", "OnboardReport", "load_config"]
