"""Android phone as the onboard computer: RGB camera + NPU AI + VIO + 4G relay.

The idea in one paragraph: a mid-range Android phone already contains a
better RGB camera than any USB board camera, a Qualcomm Hexagon DSP that runs
INT8 person detection at 30+ fps, an IMU + CPU that runs visual-inertial
odometry, a 4G modem for opportunistic telemetry, a screen for field
debugging, and its own battery so an autopilot brown-out never kills the AI.
This module is the *aircraft-side contract* for that phone: what messages it
sends, what images it delivers, and how the flight stack treats it when any
half fails.

Physical wiring (see docs/MOBILE_COMPANION.md for the diagram)::

    Phone USB-C OTG ──► USB-serial (CP2102/CH340) ──► Telem1 (SERIAL4) @ 921600
       │  MAVLink: ODOMETRY @30Hz, VISION msgs, heartbeat, sys status
       │  + optional NMEA-injected GPS assist from the phone GNSS
    Phone camera ──► RGB frames (Camera2, 1280x720 @15fps, fixed focus/exposure)
    Phone NPU ──► TFLite-INT8 person/hazard boxes via NNAPI delegate
    Phone 4G ──► opportunistic MQTT/HTTP relay of alerts + thumbnails

Two deployment shapes are supported:

1. **Native app** (``mobile/android/`` spec): Camera2 + TFLite + usb-serial
   in one APK. Best performance, most build work.
2. **Termux bridge** (``mobile/phone_onboard.py``): pure-Python fallback that
   runs today — OpenCamera/RTSP frame source, onnxruntime/TFLite runtime,
   ``pymavlink`` over USB-OTG serial. What field tests use first.

This file implements the *vehicle-side* view: it does not run on the phone,
it models what the phone provides so the autonomy (planner, fuser, safety
supervisor) can consume it through typed interfaces and so the simulators
can stand in for it.  No Android, USB or camera dependency is imported here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Phone link state
# --------------------------------------------------------------------------- #
class PhoneLinkState(str, Enum):
    ABSENT = "absent"        # nothing on the USB serial port
    PRESENT = "present"      # heartbeat seen, cameras not streaming yet
    STREAMING = "streaming"  # RGB + VIO + AI all flowing
    DEGRADED = "degraded"    # streaming but overtemp / throttled / dropping
    FAILED = "failed"        # was streaming, now silent past timeout


@dataclass
class PhoneStatus:
    """One heartbeat's worth of phone health, as the supervisor sees it."""

    t: float = 0.0
    state: PhoneLinkState = PhoneLinkState.ABSENT
    battery_pct: float = 100.0
    temp_c: float = 35.0
    throttled: bool = False
    rgb_fps: float = 0.0
    ai_fps: float = 0.0
    vio_hz: float = 0.0
    last_frame_age_s: float = 1e9
    last_vio_age_s: float = 1e9
    last_ai_age_s: float = 1e9
    modem_4g: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["state"] = self.state.value
        return d


# --------------------------------------------------------------------------- #
# Mobile companion interface
# --------------------------------------------------------------------------- #
@dataclass
class VioSample:
    """One visual-inertial odometry sample from the phone (NED, metres)."""

    t: float
    n: float
    e: float
    d: float
    vn: float
    ve: float
    vd: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    pos_sigma_m: float = 0.5
    vel_sigma_ms: float = 0.1
    tracking_confidence: float = 1.0   # 0..1, ARCore/OpenVINS quality
    features: int = 120                # tracked feature count


@dataclass
class PhoneDetection:
    """One NPU detection box from the phone, in RGB pixel coordinates."""

    label: str
    score: float
    bbox: Tuple[float, float, float, float]
    frame_t: float = 0.0
    model: str = "yolo11n-nano-int8"


class MobileCompanion:
    """Vehicle-side handle to the phone.  Transport-agnostic by design.

    The constructor takes three callables — ``read_frame``, ``read_vio`` and
    ``read_detections`` — so the same class fronts a USB-serial MAVLink link
    on the aircraft, an ADB/TCP bridge on the bench, and a scripted stub in
    simulation.  Nothing here blocks: every accessor returns the newest
    available sample or ``None``.
    """

    #: RGB is stale past this; the fuser falls back to LWIR-only.
    FRAME_TIMEOUT_S = 1.0
    #: VIO is stale past this; the EKF manager must leave source set 2.
    VIO_TIMEOUT_S = 0.5
    #: Phone CPU/GPU temperature where throttling is assumed.
    THROTTLE_TEMP_C = 48.0

    def __init__(self,
                 read_frame: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
                 read_vio: Optional[Callable[[], Optional[VioSample]]] = None,
                 read_detections: Optional[Callable[[], List[PhoneDetection]]] = None,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self._read_frame = read_frame or (lambda: None)
        self._read_vio = read_vio or (lambda: None)
        self._read_detections = read_detections or (lambda: [])
        self._clock = clock or time.monotonic
        self.status = PhoneStatus()
        self._last_frame_t = 0.0
        self._last_vio_t = 0.0
        self._last_ai_t = 0.0
        self._frames = 0
        self._fps_win: List[float] = []

    # -- accessors ------------------------------------------------------ #
    def frame(self) -> Optional[Dict[str, Any]]:
        f = self._read_frame()
        now = self._clock()
        if f is not None:
            self._last_frame_t = now
            self._frames += 1
            self._fps_win.append(now)
            self._fps_win = [t for t in self._fps_win if now - t < 2.0]
            self.status.rgb_fps = len(self._fps_win) / 2.0
            self.status.last_frame_age_s = 0.0
        else:
            self.status.last_frame_age_s = now - self._last_frame_t
        return f

    def vio(self) -> Optional[VioSample]:
        v = self._read_vio()
        now = self._clock()
        if v is not None:
            self._last_vio_t = now
            self.status.vio_hz = 30.0  # nominal; measured by the feeder
            self.status.last_vio_age_s = 0.0
        else:
            self.status.last_vio_age_s = now - self._last_vio_t
        return v

    def detections(self) -> List[PhoneDetection]:
        dets = self._read_detections() or []
        now = self._clock()
        if dets:
            self._last_ai_t = now
            self.status.last_ai_age_s = 0.0
        else:
            self.status.last_ai_age_s = now - self._last_ai_t
        return dets

    # -- health --------------------------------------------------------- #
    def poll_health(self, battery_pct: float = 100.0, temp_c: float = 35.0,
                    modem_4g: bool = False) -> PhoneStatus:
        """Fold one heartbeat into the link state machine."""
        now = self._clock()
        st = self.status
        st.t = now
        st.battery_pct = battery_pct
        st.temp_c = temp_c
        st.modem_4g = modem_4g
        st.throttled = temp_c >= self.THROTTLE_TEMP_C
        frame_age = now - self._last_frame_t
        vio_age = now - self._last_vio_t
        st.last_frame_age_s = frame_age
        st.last_vio_age_s = vio_age

        if self._last_frame_t <= 0 and self._last_vio_t <= 0:
            st.state = PhoneLinkState.ABSENT
        elif frame_age > 3.0 and vio_age > 3.0:
            st.state = PhoneLinkState.FAILED
        elif (frame_age > self.FRAME_TIMEOUT_S or vio_age > self.VIO_TIMEOUT_S
                or st.throttled):
            st.state = PhoneLinkState.DEGRADED
        elif self._frames > 0 or self._last_vio_t > 0:
            st.state = PhoneLinkState.STREAMING
        else:
            st.state = PhoneLinkState.PRESENT
        return st

    @property
    def rgb_healthy(self) -> bool:
        return (self._clock() - self._last_frame_t) <= self.FRAME_TIMEOUT_S

    @property
    def vio_healthy(self) -> bool:
        return (self._clock() - self._last_vio_t) <= self.VIO_TIMEOUT_S

    def describe(self) -> Dict[str, Any]:
        return {"status": self.status.to_dict(), "frames": self._frames,
                "rgb_healthy": self.rgb_healthy, "vio_healthy": self.vio_healthy}


# --------------------------------------------------------------------------- #
# Simulated phone (bench + unit tests + headless flood sim)
# --------------------------------------------------------------------------- #
@dataclass
class SimulatedPhoneConfig:
    rgb_size: Tuple[int, int] = (1280, 720)
    hfov_deg: float = 66.0
    fps: float = 15.0
    vio_hz: float = 30.0
    ai_fps: float = 10.0
    drift_pct: float = 1.5          # VIO drift as % of distance flown
    noise_sigma_m: float = 0.08
    dropout_prob: float = 0.0       # per-sample VIO dropout (link stress)
    throttle_after_s: float = 1e9   # overheat at this t (thermal tests)
    frame_fn: Optional[Callable[[float], np.ndarray]] = None


class SimulatedPhone(MobileCompanion):
    """A scripted phone: renders RGB via ``frame_fn`` and integrates VIO drift.

    The VIO model is deliberately simple and honest: position integrates the
    *true* velocity plus white noise plus a slow random-walk bias whose growth
    matches the configured drift percentage.  It reproduces drift *statistics*,
    not visual tracking — exactly like ``SimulatedVioSource`` in ``sar/nav``.
    """

    def __init__(self, config: Optional[SimulatedPhoneConfig] = None,
                 truth_fn: Optional[Callable[[float], Tuple[float, ...]]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 seed: int = 7) -> None:
        self.cfg = config or SimulatedPhoneConfig()
        self.truth_fn = truth_fn or (lambda t: (0.0, 0.0, -45.0, 0.0, 0.0, 0.0))
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._t = 0.0
        self._bias = np.zeros(3)
        self._vio_pos = np.zeros(3)
        self._init = False
        self._temp = 34.0
        super().__init__(read_frame=self._gen_frame, read_vio=self._gen_vio,
                         read_detections=self._gen_dets, clock=clock)
        if clock is not None:
            self._t = clock()

    # -- generators ----------------------------------------------------- #
    def _gen_frame(self) -> Optional[Dict[str, Any]]:
        self._t += 1.0 / self.cfg.fps
        if self.cfg.frame_fn is None:
            h, w = self.cfg.rgb_size[1], self.cfg.rgb_size[0]
            img = np.full((h, w, 3), 90, np.uint8)
        else:
            img = self.cfg.frame_fn(self._t)
        return {"image": img, "t": self._t, "kind": "rgb",
                "hfov_deg": self.cfg.hfov_deg,
                "size": (img.shape[1], img.shape[0])}

    def _gen_vio(self) -> Optional[VioSample]:
        if self._rng.random() < self.cfg.dropout_prob:
            return None
        dt = 1.0 / self.cfg.vio_hz
        self._t += 0.0  # vio shares the frame clock; no extra advance
        n, e, d, vn, ve, vd = self.truth_fn(self._t)[:6]
        if not self._init:
            self._vio_pos[:] = (n, e, d)
            self._init = True
        else:
            speed = math.sqrt(vn * vn + ve * ve)
            drift_rate = speed * (self.cfg.drift_pct / 100.0)
            self._bias += self._rng.normal(0, drift_rate * math.sqrt(dt), 3)
            noise = self._rng.normal(0, self.cfg.noise_sigma_m, 3)
            self._vio_pos[:] = (np.array([n, e, d]) + self._bias + noise)
        dist = float(np.linalg.norm(self._bias))
        sigma = max(0.3, 0.5 + dist * 0.6)
        conf = 1.0 if self._temp < self.THROTTLE_TEMP_C else 0.6
        return VioSample(t=self._t, n=float(self._vio_pos[0]),
                         e=float(self._vio_pos[1]), d=float(self._vio_pos[2]),
                         vn=vn, ve=ve, vd=vd, pos_sigma_m=sigma,
                         tracking_confidence=conf,
                         features=120 if conf > 0.8 else 45)

    def _gen_dets(self) -> List[PhoneDetection]:
        return []  # sim detections come from the flood renderer + FloodDetector

    # -- health --------------------------------------------------------- #
    def poll_health(self, battery_pct: float = 100.0, temp_c: Optional[float] = None,
                    modem_4g: bool = False) -> PhoneStatus:
        if temp_c is None:
            # Slow thermal ramp so overheat tests can fast-forward the clock.
            self._temp = 34.0 + 20.0 * min(self._t / max(self.cfg.throttle_after_s, 1e-6), 1.0)
            temp_c = self._temp
        return super().poll_health(battery_pct=battery_pct, temp_c=temp_c,
                                   modem_4g=modem_4g)


__all__ = ["PhoneLinkState", "PhoneStatus", "VioSample", "PhoneDetection",
           "MobileCompanion", "SimulatedPhoneConfig", "SimulatedPhone"]
