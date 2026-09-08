"""MAVLink connection layer.

One class, :class:`MavConnection`, that wraps a ``pymavlink`` link with the
behaviour a search-and-rescue sortie actually needs: pre-arm diagnostics,
mode changes that report *why* they were refused, guided-mode safety limits,
mission upload, and a telemetry cache that turns raw MAVLink into the
:class:`~sar.perception.geotag.NavQuality` the geo-tagger consumes.

Why not DroneKit
----------------
DroneKit-Python is unmaintained - 3DR left the drone market in 2016 and the
package survives on community life support, pinned against an old pymavlink and
never tested on modern Python.  Everything it offered that mattered was a thin
veneer over pymavlink, so this module provides that venier directly:
:mod:`sar.mavlink.dronekit_compat` re-exposes the ``connect()`` /
``simple_takeoff()`` / ``goto()`` API so existing DroneKit code and tutorials
port across unchanged, while the internals stay on a dependency that is
actually maintained.

Why one connection class for two backends
-----------------------------------------
The same object talks to (a) real ArduPilot SITL, launched as a subprocess,
(b) :class:`~sar.sim.sitl.MiniSITL`, a pure-Python vehicle that needs no
toolchain at all, and (c) eventually the physical TBS Lucid H743 over a radio
or USB serial.  That is the point of the layer: mission logic never learns
which of the three it is flying, so an algorithm proven in the pure-Python
simulator is the same code that runs on the aircraft.

MAVLink is a *datagram* protocol with no acknowledgements for telemetry, so
this layer is deliberately defensive: every waiter has a timeout, every command
checks its ``COMMAND_ACK``, and pre-arm failures are surfaced as the
``STATUSTEXT`` the vehicle actually emitted rather than as a bare "arm failed".
"""

from __future__ import annotations

import errno
import collections
import logging
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from pymavlink import mavutil, mavwp

from .protocol import (COPTER_MODES, COPTER_MODE_NAMES, GPS_FIX, MAV_CMD,
                       MAV_FRAME, MAV_MODE, MAV_RESULT, POSITION_MODES,
                       SOURCE_SET, TYPE_MASK, describe_message, latlon_to_int,
                       mavlink2, result_name, wrap_360)

log = logging.getLogger("sar.mavlink")

__all__ = [
    "MavConnection", "Telemetry", "CommandError", "PreArmError",
    "ArmState", "LinkStats", "default_connection_string",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class MavError(Exception):
    """Base class for MAVLink layer failures."""


class CommandError(MavError):
    """A command was refused, or no acknowledgement arrived in time."""

    def __init__(self, command: int, result: Optional[int] = None,
                 detail: str = "", timeout: Optional[float] = None) -> None:
        self.command = int(command)
        self.result = result
        self.detail = detail
        self.timeout = timeout
        if result is not None:
            msg = (f"command {command} refused: {result_name(result)}"
                   + (f" ({detail})" if detail else ""))
        elif timeout is not None:
            msg = f"command {command}: no COMMAND_ACK within {timeout:.1f} s"
        else:
            msg = f"command {command} failed" + (f": {detail}" if detail else "")
        super().__init__(msg)


class PreArmError(MavError):
    """Arming was refused.  Carries the reasons the vehicle gave.

    ArduPilot reports pre-arm failures as ``STATUSTEXT`` messages, not in the
    ``COMMAND_ACK``, so the acknowledgement alone says only "DENIED".  The
    text is what tells you whether it is the compass, the battery, the RC link
    or the EKF - which is the difference between a two-second fix and an hour of
    guessing.
    """

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        joined = "; ".join(self.reasons) if self.reasons else "no reason given"
        super().__init__(f"pre-arm check failed: {joined}")


class LinkTimeout(MavError):
    """The vehicle stopped talking."""


# --------------------------------------------------------------------------- #
# Telemetry snapshot
# --------------------------------------------------------------------------- #
#: Nominal 1-sigma values used to give the EKF's dimensionless variance ratios a
#: physical scale.  Open sky with the 30+ satellite receiver on this airframe
#: gives ~1.2 m horizontal; the barometer ~1.5 m vertical; and the EKF's own
#: velocity estimate ~0.3 m/s.  These are the sigma at ratio 1.0, and MiniSITL
#: publishes against the same constants so the two ends agree by construction.
NOMINAL_GNSS_SIGMA_M = 1.2
NOMINAL_ALT_SIGMA_M = 1.5
NOMINAL_VEL_SIGMA_MS = 0.3


@dataclass
class ArmState:
    """Whether the vehicle is armed, and what it says about that."""

    armed: bool = False
    base_mode: int = 0
    custom_mode: int = 0
    mode: str = "UNKNOWN"
    system_status: int = 0
    #: ``EKF_STATUS_REPORT`` flags, decoded against the ``EKF_*`` constants in
    #: ``mavlink.h``.  :meth:`ekf_flag_names` renders them for logs.
    ekf_flags: int = 0
    ekf_ok: bool = False

    # The masks below are what ``Copter::position_ok()`` and its cousins actually
    # test, so decoding them correctly is the difference between a pre-arm failure
    # that explains itself and one that does not.  Two traps, both hit once:
    #
    #  * They are not sequential by topic.  0x20 is POS_VERT_ABS, not horizontal
    #    position.  0x80 is CONST_POS_MODE - a *degraded* state where the EKF has
    #    stopped trusting its aiding and is holding position constant - and
    #    reading it as "height ok" makes a failing filter look healthy.
    #  * Do not OR in ``ekf_ok``.  That bit is EKF_ATTITUDE, which the filter
    #    holds even with no absolute position at all.  Folding it in turns all
    #    three of these into synonyms for "the IMUs are aligned", which is exactly
    #    the case where the distinction matters.  A vehicle reporting
    #    pos=vel=hgt=True while refusing to arm with "Need Position Estimate" is
    #    the symptom.
    _POS_HORIZ_ABS = 0x0010
    _POS_HORIZ_REL = 0x0008
    _PRED_POS_HORIZ_REL = 0x0100
    _PRED_POS_HORIZ_ABS = 0x0200
    _VEL_HORIZ = 0x0002
    _VEL_VERT = 0x0004
    _POS_VERT_ABS = 0x0020
    _POS_VERT_AGL = 0x0040
    _CONST_POS = 0x0080
    _UNINITIALISED = 0x0400
    _GPS_GLITCHING = 0x8000

    @property
    def position_ok(self) -> bool:
        """Absolute horizontal position - gates GUIDED, LOITER and RTL."""
        return bool(self.ekf_flags & (self._POS_HORIZ_ABS | self._PRED_POS_HORIZ_ABS))

    @property
    def position_relative_ok(self) -> bool:
        """Local position without an absolute fix; enough to fly, not to geotag."""
        return bool(self.ekf_flags & (self._POS_HORIZ_REL | self._PRED_POS_HORIZ_REL))

    @property
    def velocity_ok(self) -> bool:
        return bool(self.ekf_flags & (self._VEL_HORIZ | self._VEL_VERT))

    @property
    def height_ok(self) -> bool:
        return bool(self.ekf_flags & (self._POS_VERT_ABS | self._POS_VERT_AGL))

    @property
    def const_pos_mode(self) -> bool:
        """EKF is holding position constant because aiding was lost.

        The aircraft still flies and still reports a position, so this is the
        quietest way to produce a survey that is confidently geotagged to the
        wrong place.  The nav-quality monitor treats it as a denial.
        """
        return bool(self.ekf_flags & self._CONST_POS)

    @property
    def uninitialised(self) -> bool:
        return bool(self.ekf_flags & self._UNINITIALISED)

    @property
    def gps_glitching(self) -> bool:
        return bool(self.ekf_flags & self._GPS_GLITCHING)

    def ekf_flag_names(self) -> List[str]:
        """Names of the set bits, for logs and diagnostics."""
        names = {0x0001: "ATTITUDE", 0x0002: "VEL_HORIZ", 0x0004: "VEL_VERT",
                 0x0008: "POS_HORIZ_REL", 0x0010: "POS_HORIZ_ABS",
                 0x0020: "POS_VERT_ABS", 0x0040: "POS_VERT_AGL",
                 0x0080: "CONST_POS_MODE", 0x0100: "PRED_POS_HORIZ_REL",
                 0x0200: "PRED_POS_HORIZ_ABS", 0x0400: "UNINITIALIZED",
                 0x8000: "GPS_GLITCHING"}
        return [n for b, n in sorted(names.items()) if self.ekf_flags & b]

    def describe(self) -> str:
        return (f"{'ARMED' if self.armed else 'DISARMED'} mode={self.mode} "
                f"sys={self.system_status} ekf[{'|'.join(self.ekf_flag_names())}] "
                f"pos={self.position_ok} vel={self.velocity_ok} "
                f"hgt={self.height_ok}")


@dataclass
class LinkStats:
    """Link health.  A degraded link is diagnosed, not discovered mid-sortie."""

    tx_packets: int = 0
    rx_packets: int = 0
    rx_dropped: int = 0
    tx_errors: int = 0
    #: Round-trip latency to the vehicle's clock, seconds.  Negative means the
    #: vehicle clock is ahead; only the magnitude is meaningful.
    time_offset_us: int = 0
    rssi_dbm: Optional[float] = None
    #: Last time any message arrived, ``time.time()``.
    last_rx: float = 0.0

    @property
    def silence_s(self) -> float:
        return max(0.0, time.time() - self.last_rx) if self.last_rx else float("inf")

    @property
    def loss_pct(self) -> float:
        tot = self.rx_packets + self.rx_dropped
        return 100.0 * self.rx_dropped / tot if tot else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"tx": self.tx_packets, "rx": self.rx_packets,
                "dropped": self.rx_dropped, "loss_pct": round(self.loss_pct, 2),
                "silence_s": round(self.silence_s, 2),
                "rssi_dbm": self.rssi_dbm,
                "time_offset_us": self.time_offset_us}


@dataclass
class Telemetry:
    """Everything the mission and perception layers need, in one place.

    Assembled from several MAVLink messages that arrive independently, so each
    field carries the time it was last refreshed.  A field whose ``age`` exceeds
    the link timeout is stale and must not be trusted for navigation - which is
    exactly how a GPS-denied sortie detects that its aiding source died.
    """

    t: float = 0.0
    # --- position ---
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_msl_m: Optional[float] = None
    alt_rel_m: Optional[float] = None
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    hdg_deg: float = 0.0
    # --- attitude ---
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    # --- power ---
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    battery_remaining_pct: Optional[int] = None
    consumed_mah: Optional[int] = None
    consumed_wh: Optional[float] = None
    cell_voltage_v: Optional[float] = None
    # --- navigation quality ---
    gps_fix: int = GPS_FIX.NO_FIX
    n_satellites: int = 0
    hdop: float = 99.0
    vdop: float = 99.0
    gps_week: int = 0
    #: ``EKF_STATUS_REPORT`` variance fields, as ArduPilot fills them: they are
    #: *dimensionless ratios* scaled so that ~1.0 is the filter's own failure
    #: threshold - not m^2, despite the field names saying "variance" and not
    #: centimetres either.  Reading them as a linear unit makes the reported
    #: sigma track whatever scale factor the firmware happened to use instead of
    #: the actual error, which is worse than not reporting it.  They are used as
    #: a multiplier on a nominal sigma whose units are unambiguous.
    pos_horiz_variance_ratio: Optional[float] = None
    vel_variance_ratio: Optional[float] = None
    hgt_variance_ratio: Optional[float] = None
    #: EKF3 source set currently selected.
    ekf_source_set: int = 1
    ekf_source_name: str = "GPS"
    # --- link / state ---
    armed: ArmState = field(default_factory=ArmState)
    rssi_dbm: Optional[float] = None
    #: Age in seconds of each block, keyed by message type.
    ages: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def speed_ms(self) -> float:
        return math.hypot(self.vn, self.ve)

    @property
    def climb_ms(self) -> float:
        """Vertical speed, positive up (MAVLink reports vz positive down)."""
        return -self.vd

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def horizontal_pos_sigma_m(self) -> float:
        """1-sigma horizontal position error from the EKF's own variance.

        Preferred over deriving it from HDOP because the EKF knows what it is
        actually doing - in particular it inflates this correctly when the
        source set has been switched to an external nav aid, which is the whole
        basis of the geo-tagging uncertainty budget downstream.
        """
        # Two independent estimates, and the geo-tagger gets the pessimistic one.
        #
        #  * HDOP times a nominal pseudorange sigma has unambiguous units, but
        #    says nothing once the fix is gone.
        #  * The EKF's variance ratio has no units, but it keeps growing through
        #    a denial, which is exactly when the first estimate is unavailable.
        #
        # Combining them is what makes the uncertainty honest across the
        # transition: on GNSS the geometry dominates, and as the fix degrades the
        # filter's own divergence takes over without a discontinuity.
        parts: List[float] = []
        if self.pos_horiz_variance_ratio is not None \
                and self.pos_horiz_variance_ratio >= 0:
            parts.append(NOMINAL_GNSS_SIGMA_M
                         * math.sqrt(self.pos_horiz_variance_ratio))
        if self.gps_fix >= GPS_FIX.FIX_3D and self.hdop < 50:
            parts.append(max(0.4, self.hdop * NOMINAL_GNSS_SIGMA_M))
        if not parts:
            return 100.0                  # no useful fix and no filter report
        return float(min(max(parts), 1000.0))

    @property
    def gps_denied(self) -> bool:
        """True when GNSS is not currently a usable position source."""
        return self.gps_fix < GPS_FIX.FIX_3D or self.n_satellites < 6

    def age(self, key: str, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        t = self.ages.get(key)
        return float("inf") if t is None else max(0.0, now - t)

    def summary(self) -> str:
        pos = (f"({self.lat:.6f},{self.lon:.6f}) alt={self.alt_rel_m:.1f}m"
               if self.has_position and self.alt_rel_m is not None else "no-fix")
        # Each power field is independently optional: MAVLink's sentinel for
        # "not available" is per-field, so a vehicle can report voltage and no
        # current, and formatting them as one group raises on the missing one.
        def _f(v, fmt, unit, dash="?"):
            return f"{v:{fmt}}{unit}" if v is not None else dash
        if self.voltage_v is None and self.current_a is None:
            pwr = "no-battery"
        else:
            pwr = (_f(self.voltage_v, ".2f", "V") + " "
                   + _f(self.current_a, ".1f", "A") + " "
                   + _f(self.battery_remaining_pct, "d", "%"))
        return (f"{pos} v={self.speed_ms:.1f}m/s hdg={self.hdg_deg:.0f} "
                f"sats={self.n_satellites} fix={self.gps_fix} "
                f"src={self.ekf_source_name} sig_h={self.horizontal_pos_sigma_m:.2f}m "
                f"{pwr} {self.armed.describe()}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": round(self.t, 3), "lat": self.lat, "lon": self.lon,
            "alt_msl_m": self.alt_msl_m, "alt_rel_m": self.alt_rel_m,
            "vn": round(self.vn, 3), "ve": round(self.ve, 3), "vd": round(self.vd, 3),
            "speed_ms": round(self.speed_ms, 3), "hdg_deg": round(self.hdg_deg, 1),
            "roll_deg": round(self.roll_deg, 2), "pitch_deg": round(self.pitch_deg, 2),
            "yaw_deg": round(self.yaw_deg, 2),
            "voltage_v": self.voltage_v, "current_a": self.current_a,
            "battery_remaining_pct": self.battery_remaining_pct,
            "consumed_wh": self.consumed_wh,
            "gps_fix": self.gps_fix, "n_satellites": self.n_satellites,
            "hdop": self.hdop, "pos_sigma_m": round(self.horizontal_pos_sigma_m, 3),
            "gps_denied": self.gps_denied,
            "ekf_source_set": self.ekf_source_set,
            "ekf_source_name": self.ekf_source_name,
            "armed": self.armed.armed, "mode": self.armed.mode,
            "ekf_ok": self.armed.ekf_ok, "rssi_dbm": self.rssi_dbm,
        }


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def default_connection_string(backend: str = "sitl", instance: int = 0) -> str:
    """Connection string for the usual backends.

    ``sitl``      - ArduPilot SITL, which multicasts telemetry to UDP 14550 and
                    listens there too, so the client binds the same port.
    ``mini``      - :class:`~sar.sim.sitl.MiniSITL`, pure Python.
    ``hardware``  - a real flight controller over USB serial.  The baud rate is
                    the one ArduPilot defaults to on USB (115200); SERIAL4/7 for
                    a companion computer runs 921600 on the TBS Lucid H743.
    """
    if backend == "hardware":
        dev = os.environ.get("SAR_FC_DEVICE", "/dev/ttyACM0")
        baud = int(os.environ.get("SAR_FC_BAUD", "115200"))
        return f"{dev}:{baud}"
    port = 14550 + 10 * int(instance)
    return f"udpin:0.0.0.0:{port}"


class MavConnection:
    """A MAVLink link to one vehicle.

    Parameters
    ----------
    target : str
        Anything ``pymavlink.mavutil.mavlink_connection`` accepts:
        ``udpin:host:port``, ``udpout:host:port``, ``tcp:host:port``,
        ``/dev/ttyACM0:115200``.
    source_system, source_component : int
        MAVLink identity of this client.  Defaults to the companion-computer
        component ID, because that is what this code is: the onboard computer,
        not the ground station.  ArduPilot routes some messages differently
        depending on the source component, and using the GCS id makes the
        vehicle treat guidance commands as operator overrides.
    heartbeat_interval_s : float
        How often to emit a GCS-style heartbeat.  ArduPilot's ``FS_GCS_ENABLE``
        failsafe triggers on heartbeat loss, so a sortie that stops beating
        while it is still flying will RTL itself.  1 Hz is the MAVLink default.
    link_timeout_s : float
        How long without *any* inbound message before the link is declared dead.
    vehicle_is_copter : bool
        Selects the Copter mode table.  Plane and Rover number their modes
        differently, and sending Copter's GUIDED=4 to a Plane puts it in
        AUTO rather than GUIDED.
    """

    #: Telemetry messages the link asks the vehicle to stream.  Rates are
    #: chosen against what the perception stack consumes: 8 Hz matches the
    #: camera frame rate so every frame can be geo-tagged with a fresh pose,
    #: and the geo-tagger's latency term assumes a pose no older than ~125 ms.
    DEFAULT_STREAMS: Tuple[Tuple[str, int], ...] = (
        ("EXTENDED_STATUS", 2),
        ("EXTRA1", 4),              # attitude
        ("EXTRA2", 8),              # VFR_HUD / global position
        ("EXTRA3", 2),
        ("POSITION", 8),
        ("RAW_SENSORS", 2),
        ("RC_CHANNELS", 2),
    )

    def __init__(self, target: str,
                 source_system: int = 255,
                 source_component: int = 191,
                 heartbeat_interval_s: float = 1.0,
                 link_timeout_s: float = 3.0,
                 vehicle_is_copter: bool = True,
                 input_hook: Optional[Callable[[Any], None]] = None) -> None:
        self.target = target
        self.source_system = int(source_system)
        self.source_component = int(source_component)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.link_timeout_s = float(link_timeout_s)
        self.vehicle_is_copter = bool(vehicle_is_copter)
        self.modes = COPTER_MODES if vehicle_is_copter else {}
        self.mode_names = COPTER_MODE_NAMES if vehicle_is_copter else {}
        self.input_hook = input_hook

        self.master = mavutil.mavlink_connection(
            target, input=False,
            source_system=self.source_system,
            source_component=self.source_component,
            dialect="ardupilotmega",
        )
        self._lock = threading.RLock()
        #: Guards the socket receive path specifically.  Separate from ``_lock``
        #: because that one is held while sending, and holding a single lock
        #: across a blocking recv would serialise the heartbeat behind it.
        self._rx_lock = threading.RLock()
        #: Recent messages with their arrival time, newest last.  Shared across
        #: all reader threads on this connection; see :meth:`recv`.
        self._inbox: "collections.deque[Tuple[float, Any]]" = \
            collections.deque(maxlen=512)
        self._inbox_lock = threading.RLock()
        self._cache: Dict[str, Any] = {}
        self._cache_t: Dict[str, float] = {}
        self._statustext: List[Tuple[float, str]] = []
        self._callbacks: Dict[str, List[Callable[[Any], None]]] = {}
        self._closed = False
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        self.telemetry = Telemetry()
        self.link = LinkStats()
        self.target_system = 1
        self.target_component = 1
        self.vehicle_time_boot_ms = 0
        self._wall_of_boot: Dict[int, float] = {}
        #: Runaway guard re-applied once the vehicle is armed and in GUIDED.
        self._pending_guided_limits: Optional[Tuple[float, float, float, float]] = None

        if heartbeat_interval_s > 0:
            self.start_heartbeat()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"<MavConnection {self.target} sys={self.target_system}>"

    def __enter__(self) -> "MavConnection":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hb_stop.set()
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=2.0)
        try:
            self.master.close()
        except Exception:                                  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    # Heartbeat
    # ------------------------------------------------------------------ #
    def start_heartbeat(self) -> None:
        """Emit GCS heartbeats on a daemon thread.

        Runs on its own thread rather than being piggy-backed onto the receive
        loop so that a long ``wait_for`` does not starve it: if the heartbeat
        stops while the code is blocked waiting for the vehicle to reach an
        altitude, ArduPilot's GCS failsafe triggers and the aircraft returns to
        launch in the middle of the wait.
        """
        if self._hb_thread and self._hb_thread.is_alive():
            return
        self._hb_stop.clear()

        def _loop() -> None:
            while not self._hb_stop.is_set():
                try:
                    self.send_heartbeat()
                except Exception as exc:                   # pragma: no cover
                    log.debug("heartbeat send failed: %r", exc)
                    self.link.tx_errors += 1
                self._hb_stop.wait(self.heartbeat_interval_s)

        self._hb_thread = threading.Thread(target=_loop, name="mav-hb", daemon=True)
        self._hb_thread.start()

    def send_heartbeat(self) -> None:
        """One heartbeat, declaring ourselves an onboard computer.

        ``MAV_TYPE_ONBOARD_CONTROLLER`` and ``AUTOPILOT_INVALID`` are the right
        identity for a companion computer: it is not a GCS (so it does not
        trigger the GCS failsafe semantics intended for a human operator's
        laptop) and it is not an autopilot (so the vehicle does not treat it as
        a redundant flight controller).
        """
        with self._lock:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
            self.link.tx_packets += 1

    # ------------------------------------------------------------------ #
    # Receive
    # ------------------------------------------------------------------ #
    def on(self, message_type: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to a message type.  Called from the receive loop."""
        self._callbacks.setdefault(message_type, []).append(callback)

    def _ingest(self, msg: Any) -> None:
        """Cache a message, update telemetry, and fan out to callbacks."""
        t = msg.get_type()
        now = time.time()
        self._cache[t] = msg
        self._cache_t[t] = now
        self.link.rx_packets += 1
        self.link.last_rx = now
        if hasattr(msg, "time_boot_ms"):
            self.vehicle_time_boot_ms = int(msg.time_boot_ms)
            self._wall_of_boot[int(msg.time_boot_ms)] = now
        self._apply(t, msg, now)
        if t == "STATUSTEXT":
            self._statustext.append((now, str(msg.text)))
            if len(self._statustext) > 400:
                del self._statustext[:200]
            log.debug("STATUSTEXT: %s", msg.text)
        if t == "HEARTBEAT" and msg.get_srcSystem():
            self.target_system = msg.get_srcSystem()
            self.target_component = msg.get_srcComponent()
        for cb in self._callbacks.get(t, ()):
            try:
                cb(msg)
            except Exception as exc:                       # pragma: no cover
                log.warning("callback for %s raised %r", t, exc)
        if self.input_hook is not None:
            try:
                self.input_hook(msg)
            except Exception as exc:                       # pragma: no cover
                log.warning("input hook raised %r", exc)

    def _apply(self, t: str, m: Any, now: float) -> None:
        """Fold one message into the :class:`Telemetry` snapshot."""
        tel = self.telemetry
        tel.t = now
        if t in ("GLOBAL_POSITION_INT", "GPS_GLOBAL_ORIGIN"):
            if t == "GLOBAL_POSITION_INT":
                tel.lat = m.lat / 1e7
                tel.lon = m.lon / 1e7
                tel.alt_msl_m = m.alt / 1000.0
                tel.alt_rel_m = m.relative_alt / 1000.0
                tel.vn = m.vx / 100.0
                tel.ve = m.vy / 100.0
                tel.vd = m.vz / 100.0
                tel.hdg_deg = wrap_360(m.hdg / 100.0)
                tel.ages["position"] = now
        elif t == "ATTITUDE":
            tel.roll_deg = math.degrees(m.roll)
            tel.pitch_deg = math.degrees(m.pitch)
            tel.yaw_deg = wrap_360(math.degrees(m.yaw))
            tel.roll_rate = m.rollspeed
            tel.pitch_rate = m.pitchspeed
            tel.yaw_rate = m.yawspeed
            tel.ages["attitude"] = now
        elif t == "SYS_STATUS":
            # 65534/65535 mV is MAVLink's "voltage not available" sentinel, and
            # reading it as 65.5 V makes a 6S pack look over-charged by 3x.
            tel.voltage_v = (m.voltage_battery / 1000.0
                             if 0 <= m.voltage_battery < 65000 else None)
            tel.current_a = (m.current_battery / 100.0
                             if 0 <= m.current_battery < 65000 else None)
            tel.battery_remaining_pct = (int(m.battery_remaining)
                                         if 0 <= m.battery_remaining <= 100 else None)
            tel.ages["status"] = now
        elif t == "BATTERY_STATUS":
            if m.current_battery >= 0:
                tel.current_a = m.current_battery / 100.0
            if getattr(m, "energy_consumed", -1) > 0:
                tel.consumed_wh = m.energy_consumed / 100.0
            if getattr(m, "battery_remaining", -1) >= 0:
                tel.battery_remaining_pct = int(m.battery_remaining)
            # Same 65535 mV sentinel as SYS_STATUS, but arriving on the message
            # that is supposed to be the authoritative one - so it has to be
            # filtered here too, or it overwrites a good reading with a bad one.
            if getattr(m, "voltages", None) and 0 < m.voltages[0] < 65000:
                v = m.voltages[0] / 1000.0
                tel.voltage_v = v
                cells = int(m.voltages[1]) if len(m.voltages) > 1 else 0
                tel.cell_voltage_v = v / cells if cells else None
            tel.ages["battery"] = now
        elif t == "GPS_RAW_INT":
            tel.gps_fix = int(m.fix_type)
            tel.n_satellites = int(m.satellites_visible)
            # GPS_RAW_INT carries dilution of precision as eph/epv in
            # centimetres, not as a hdop/vdop ratio - the field names invite the
            # wrong reading, and 65535 is the "unknown" sentinel.
            tel.hdop = m.eph / 100.0 if 0 < m.eph < 65535 else 99.0
            tel.vdop = m.epv / 100.0 if 0 < m.epv < 65535 else 99.0
            if m.fix_type >= GPS_FIX.FIX_3D:
                tel.lat = m.lat / 1e7
                tel.lon = m.lon / 1e7
                tel.alt_msl_m = m.alt / 1000.0
            tel.ages["gps"] = now
        elif t == "EKF_STATUS_REPORT":
            tel.armed.ekf_flags = int(m.flags)
            # The real field names are ``pos_horiz_variance`` /
            # ``pos_vert_variance`` / ``velocity_variance``.  This used to read
            # ``pos_horiz_accuracy``, which is not a field of this message at all,
            # so the value was always absent and every consumer silently fell
            # through to the "no fix, assume the worst" branch - a geo-tagger that
            # reported a 100 m ellipse through an entire sortie while the filter
            # was perfectly happy.  Read defensively anyway: the field set does
            # differ between stacks, and a missing attribute must not kill
            # telemetry.
            for attr, dest in (("pos_horiz_variance", "pos_horiz_variance_ratio"),
                               ("pos_vert_variance", "hgt_variance_ratio"),
                               ("velocity_variance", "vel_variance_ratio")):
                v = getattr(m, attr, None)
                if v is not None and math.isfinite(v):
                    setattr(tel, dest, float(v))
            tel.armed.ekf_ok = bool(m.flags & 0x0001)
            tel.ages["ekf"] = now
        elif t == "HEARTBEAT":
            a = tel.armed
            a.armed = bool(m.base_mode & MAV_MODE.SAFETY_ARMED)
            a.base_mode = int(m.base_mode)
            a.custom_mode = int(m.custom_mode)
            a.mode = self.mode_names.get(int(m.custom_mode), f"MODE_{m.custom_mode}")
            a.system_status = int(m.system_status)
            tel.ages["heartbeat"] = now
        elif t == "RADIO" or t == "RADIO_STATUS":
            tel.rssi_dbm = float(m.rssi)
            self.link.rssi_dbm = float(m.rssi)
            tel.ages["radio"] = now
        elif t == "RC_CHANNELS":
            # ArduPilot reports RSSI 0..254; RSSI_TYPE=3 on the Lucid H743 maps
            # ELRS link quality into this field.
            if getattr(m, "rssi", 0) > 0:
                self.link.rssi_dbm = float(m.rssi)
        elif t == "AHRS2" or t == "AHRS3":
            if t == "AHRS3" and tel.lat is None:
                tel.lat = m.lat / 1e7
                tel.lon = m.lng / 1e7
        elif t == "TIMESYNC":
            if getattr(m, "tc1", 0):
                self.link.time_offset_us = int(m.tc1)

    def recv(self, timeout: float = 0.0) -> Optional[Any]:
        """Receive and ingest one message.  ``None`` on timeout.

        Non-blocking by default so it can be called from inside a control loop
        without stalling it.
        """
        if self._closed:
            return None
        # Serialised: the heartbeat thread, the external-nav feeder thread and
        # the mission loop all touch this socket, and pymavlink's receive path is
        # not reentrant.  Without the lock a message can be parsed twice or a
        # partial datagram interleaved, which shows up as intermittent BAD_DATA
        # that is impossible to reproduce.
        with self._rx_lock:
            msg = self.master.recv_match(blocking=True, timeout=max(0.0, timeout))
        if msg is None:
            return None
        if getattr(msg, "get_type", lambda: "")() == "BAD_DATA":
            self.link.rx_dropped += 1
            return None
        self._ingest(msg)
        # Every message also goes into a shared inbox.  One socket has several
        # readers - the mission loop, the heartbeat thread, the external-nav
        # feeder - and whichever reads first consumes the message for everyone
        # else.  Telemetry survives that because it is folded into a cache, but
        # a one-shot reply does not: a feeder pumping at 30 Hz would eat a
        # COMMAND_ACK before the thread that issued the command ever saw it, and
        # the symptom is "no ACK within 3 s" for a command the vehicle actually
        # accepted.  On hardware the same thing happens between a companion
        # computer's telemetry thread and the mission thread sharing one serial
        # port, so this is not a simulator artifact.
        with self._inbox_lock:
            self._inbox.append((time.time(), msg))
        return msg

    def pump(self, duration_s: float = 0.05, idle_breaks: int = 2) -> int:
        """Drain everything pending, for at most ``duration_s``.

        ``recv`` returns ``None`` both for "no message yet" and for a corrupt
        datagram, and breaking on the first ``None`` made a single BAD_DATA frame
        end the drain - which is how a link that was in fact delivering 30
        messages a second reported 4.  It now takes ``idle_breaks`` consecutive
        empties before giving up.
        """
        end = time.time() + max(0.0, duration_s)
        n = 0
        idle = 0
        while time.time() < end and idle < idle_breaks:
            if self.recv(timeout=min(0.25, max(0.0, end - time.time()))) is None:
                idle += 1
            else:
                idle = 0
                n += 1
        return n

    def get(self, message_type: str, max_age_s: float = 1e9) -> Optional[Any]:
        """Most recent message of a type, or ``None`` if absent or stale."""
        with self._lock:
            self.pump(0.0)
            msg = self._cache.get(message_type)
            if msg is None:
                return None
            if time.time() - self._cache_t.get(message_type, 0.0) > max_age_s:
                return None
            return msg

    def wait_heartbeat(self, timeout: float = 20.0) -> Any:
        """Block until the vehicle says hello.  Raises :class:`LinkTimeout`."""
        hb = self.master.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise LinkTimeout(
                f"no MAVLink heartbeat from {self.target} within {timeout:.0f} s - "
                "is the vehicle running, and is the connection string right?")
        self.target_system = self.master.target_system
        self.target_component = self.master.target_component
        self._ingest(hb)
        log.info("vehicle %d/%d online: %s", self.target_system,
                 self.target_component, self.telemetry.armed.describe())
        return hb

    def wait_message(self, message_type: str, timeout: float = 10.0,
                     predicate: Optional[Callable[[Any], bool]] = None) -> Optional[Any]:
        """Wait for a message of a type, optionally matching a predicate.

        Scans the shared inbox rather than reading the socket directly, so the
        reply is found even if another thread on this connection received it
        first.  Messages older than the call are ignored, which keeps a
        previous command's ACK from satisfying the next one.
        """
        t0 = time.time()
        end = t0 + timeout
        while time.time() < end:
            with self._inbox_lock:
                for ts, msg in reversed(self._inbox):
                    if ts < t0:
                        break
                    if msg.get_type() != message_type:
                        continue
                    try:
                        if predicate is None or predicate(msg):
                            return msg
                    except Exception:
                        continue
            self.recv(timeout=min(0.25, max(0.0, end - time.time())))
        return None

    def wait_link(self, timeout: float = 20.0) -> "MavConnection":
        """Wait for heartbeat plus a usable attitude estimate."""
        self.wait_heartbeat(timeout=timeout)
        return self

    # ------------------------------------------------------------------ #
    # Time base
    # ------------------------------------------------------------------ #
    def boot_to_wall(self, boot_ms: int) -> float:
        """Convert a vehicle ``time_boot_ms`` to a local wall-clock time.

        The vehicle's monotonic clock and ours are unrelated, and every
        geo-tagged detection has to be pinned to the same timeline as the
        telemetry it is fused with.  Anchoring on the most recent observed
        (boot_ms, wall) pair keeps the two aligned to within a frame period.
        """
        if not self._wall_of_boot:
            return time.time()
        nearest = min(self._wall_of_boot, key=lambda k: abs(k - boot_ms))
        return self._wall_of_boot[nearest] + (boot_ms - nearest) / 1000.0

    @property
    def now(self) -> float:
        """Best current time on the shared timeline."""
        return self.boot_to_wall(self.vehicle_time_boot_ms)

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #
    def command_long(self, command: int, p1: float = 0.0, p2: float = 0.0,
                     p3: float = 0.0, p4: float = 0.0, p5: float = 0.0,
                     p6: float = 0.0, p7: float = 0.0,
                     confirmation: int = 0, timeout: float = 3.0,
                     want_result: int = MAV_RESULT.ACCEPTED,
                     retries: int = 1) -> Any:
        """Send ``COMMAND_LONG`` and wait for the acknowledgement.

        MAVLink does not guarantee delivery, so a command that gets no ``ACK``
        is retried once before being declared failed - but only if the ACK is
        genuinely absent.  An ACK that says DENIED is *not* retried, because
        resending a refused command just wastes the timeout.

        Raises :class:`CommandError` unless the result matches ``want_result``.
        Pass ``want_result=None`` to accept anything.
        """
        if self._closed:
            raise MavError("connection is closed")
        for attempt in range(max(1, retries + 1)):
            # Drain stale acks for this command so we do not read a previous
            # command's acknowledgement and mistake it for this one's.
            with self._lock:
                self.master.mav.command_long_send(
                    self.target_system, self.target_component,
                    int(command), int(confirmation),
                    float(p1), float(p2), float(p3), float(p4),
                    float(p5), float(p6), float(p7))
                self.link.tx_packets += 1
            ack = self.wait_message(
                "COMMAND_ACK", timeout=timeout,
                predicate=lambda m: int(m.command) == int(command))
            if ack is not None:
                if want_result is None or int(ack.result) == int(want_result):
                    return ack
                raise CommandError(command, int(ack.result),
                                   detail=self.recent_statustext(2.0))
            log.debug("no ACK for cmd %d (attempt %d)", command, attempt + 1)
        raise CommandError(command, timeout=timeout)

    def recent_statustext(self, window_s: float = 5.0) -> str:
        """Recent ``STATUSTEXT`` joined into one string, for error messages."""
        cutoff = time.time() - window_s
        return "; ".join(s for t, s in self._statustext if t >= cutoff)

    def statustext_since(self, t0: float) -> List[str]:
        return [s for t, s in self._statustext if t >= t0]

    # ------------------------------------------------------------------ #
    # Mode
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str, timeout: float = 5.0) -> str:
        """Change flight mode by name.  Returns the mode actually reached.

        Raises :class:`CommandError` if the vehicle does not confirm the change
        within ``timeout``.  Confirming by observing the mode in telemetry rather
        than trusting the ACK matters: ArduPilot ACKs a mode change it then
        refuses to make (LAND from a disarm state, for example).
        """
        mode = mode.upper().replace("-", "_")
        if mode not in self.modes:
            raise MavError(f"unknown flight mode {mode!r} for "
                           f"{'copter' if self.vehicle_is_copter else 'vehicle'}; "
                           f"known: {sorted(self.modes)}")
        self.master.set_mode(self.modes[mode])
        self.link.tx_packets += 1
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.05)
            if self.telemetry.armed.mode == mode:
                log.info("mode -> %s", mode)
                return mode
        raise CommandError(MAV_CMD.DO_SET_MODE, MAV_RESULT.FAILED,
                           detail=f"vehicle still in {self.telemetry.armed.mode}, "
                                  f"wanted {mode}")

    @property
    def mode(self) -> str:
        return self.telemetry.armed.mode

    @property
    def armed(self) -> bool:
        return self.telemetry.armed.armed

    def can_accept_position_commands(self) -> bool:
        return self.telemetry.armed.mode in POSITION_MODES

    # ------------------------------------------------------------------ #
    # Arming
    # ------------------------------------------------------------------ #
    def arm(self, timeout: float = 10.0, force: bool = False,
            throttle: float = 0.0) -> ArmState:
        """Arm the vehicle, surfacing the actual pre-arm failure reasons.

        ``force`` sets the arming ``param1`` bit that tells ArduPilot to skip
        pre-arm checks.  It exists for simulation and for bench testing, and it
        is not something to fly with: the checks it skips are the ones that
        catch a disconnected compass or a battery that will not support hover.
        """
        t0 = time.time()
        param1 = 1.0 if not force else 21196.0     # 21196 = force magic
        try:
            self.command_long(MAV_CMD.COMPONENT_ARM_DISARM, p1=param1,
                              p2=0.0, timeout=min(3.0, timeout),
                              confirmation=0, retries=1)
        except CommandError as exc:
            # ArduPilot emits the reasons as STATUSTEXT, and they can land either
            # side of the COMMAND_ACK, so settle briefly and look back over the
            # whole attempt rather than only forward from its start.  A bare
            # "command 400 refused: FAILED" with no reason is the least useful
            # failure in the stack and is always avoidable with a wider window.
            self.pump(0.4)
            reasons = (self.statustext_since(t0 - 1.0)
                       or [s for s in self.recent_statustext(timeout + 2.0).split("; ")
                           if s.strip()])
            if reasons:
                raise PreArmError(reasons) from exc
            raise
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.05)
            if self.telemetry.armed.armed:
                log.info("armed (%s)", self.telemetry.armed.describe())
                if self._pending_guided_limits and self.mode == "GUIDED":
                    self.guided_limits(*self._pending_guided_limits)
                    self._pending_guided_limits = None
                return self.telemetry.armed
        raise PreArmError(self.statustext_since(t0) or ["arm not confirmed by heartbeat"])

    def disarm(self, timeout: float = 5.0, force: bool = False) -> None:
        param1 = 0.0 if not force else 21196.0
        try:
            self.command_long(MAV_CMD.COMPONENT_ARM_DISARM, p1=param1,
                              timeout=min(2.0, timeout))
        except CommandError:
            pass
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.05)
            if not self.telemetry.armed.armed:
                log.info("disarmed")
                return

    # ------------------------------------------------------------------ #
    # Streams
    # ------------------------------------------------------------------ #
    def request_streams(self, streams: Optional[Iterable[Tuple[str, int]]] = None) -> None:
        """Ask for the telemetry groups this stack needs.

        ``REQUEST_DATA_STREAM`` is the legacy mechanism and is what SITL and
        most current ArduPilot builds still honour reliably; ``MAV_CMD_SET_
        MESSAGE_INTERVAL`` is the modern one and is tried first because it is
        per-message rather than per-group.  Failing either is not fatal - a
        vehicle that streams by default simply does not need to be asked.
        """
        for name, hz in (streams or self.DEFAULT_STREAMS):
            stream_id = getattr(mavutil.mavlink, f"MAV_DATA_STREAM_{name}", None)
            if stream_id is None:
                continue
            try:
                self.master.mav.request_data_stream_send(
                    self.target_system, self.target_component,
                    stream_id, int(hz), 1)
                self.link.tx_packets += 1
            except Exception as exc:                       # pragma: no cover
                log.debug("request_data_stream(%s) failed: %r", name, exc)

    def set_message_interval(self, message_id: int, hz: float) -> bool:
        """Set one message's rate.  ``hz <= 0`` disables it."""
        interval_us = int(1e6 / hz) if hz > 0 else -1
        try:
            self.command_long(MAV_CMD.MESSAGE_INTERVAL, p1=float(message_id),
                              p2=float(interval_us), timeout=2.0, retries=0)
            return True
        except CommandError:
            return False

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    def param_set(self, name: str, value: float, timeout: float = 3.0) -> Optional[float]:
        """Write one parameter and wait for the vehicle to echo it back."""
        name = name.upper()[:16]
        # ``master.param_set`` does not exist on a mavutil connection object -
        # ``param_set_send`` is the wrapper, and it fills in target
        # system/component plus the MAVLink-1 type field itself.
        self.master.param_set_send(name, float(value))
        self.link.tx_packets += 1
        end = time.time() + timeout
        while time.time() < end:
            msg = self.recv(timeout=min(0.5, max(0.0, end - time.time())))
            if msg is not None and msg.get_type() == "PARAM_VALUE" \
                    and msg.param_id.strip().upper() == name:
                return float(msg.param_value)
        return None

    def param_get(self, name: str, timeout: float = 3.0) -> Optional[float]:
        name = name.upper()[:16]
        self.master.param_fetch_one(name)
        self.link.tx_packets += 1
        end = time.time() + timeout
        while time.time() < end:
            msg = self.recv(timeout=min(0.5, max(0.0, end - time.time())))
            if msg is not None and msg.get_type() == "PARAM_VALUE" \
                    and msg.param_id.strip().upper() == name:
                return float(msg.param_value)
        return None

    def param_set_many(self, params: Dict[str, float],
                       verify: bool = True) -> Dict[str, Optional[float]]:
        """Write a block of parameters.  Returns what was confirmed.

        Writes are issued back to back and then verified in a second pass,
        because ArduPilot processes ``PARAM_SET`` asynchronously and a naive
        set-then-immediately-read returns the old value.
        """
        for name, value in params.items():
            self.master.param_set_send(name.upper()[:16], float(value))
            self.link.tx_packets += 1
        if not verify:
            return {k: None for k in params}
        out: Dict[str, Optional[float]] = {}
        for name, want in params.items():
            got = self.param_get(name)
            out[name] = got
            if got is None:
                log.warning("param %s not confirmed (wanted %s)", name, want)
            elif abs(got - want) > 1e-6 * max(1.0, abs(want)):
                log.warning("param %s = %s, wanted %s", name, got, want)
        return out

    # ------------------------------------------------------------------ #
    # Guided movement
    # ------------------------------------------------------------------ #
    def takeoff(self, altitude_m: float, timeout: float = 4.0) -> None:
        """``MAV_CMD_NAV_TAKEOFF`` to ``altitude_m`` above home."""
        self.command_long(MAV_CMD.NAV_TAKEOFF, p7=float(altitude_m), timeout=timeout)

    def goto_global(self, lat: float, lon: float, alt_rel_m: float,
                    timeout: float = 2.0, yaw_deg: Optional[float] = None,
                    velocity_ms: Optional[float] = None) -> None:
        """Guided position target in lat/lon at an altitude relative to home.

        Relative-to-home is used rather than absolute because the survey
        altitude that sets the ground sample distance is a height above the
        ground, and an absolute altitude would put the aircraft at a different
        GSD on every hill in the search area.
        """
        lat_i, lon_i = latlon_to_int(lat, lon)
        mask = TYPE_MASK.POS_ONLY
        if yaw_deg is None:
            mask |= TYPE_MASK.IGNORE_YAW
        self.master.mav.set_position_target_global_int_send(
            0, self.target_system, self.target_component,
            MAV_FRAME.GLOBAL_REL_ALT_INT, mask,
            lat_i, lon_i, float(alt_rel_m),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            float(velocity_ms) if velocity_ms else 0.0, 0.0,
            float(yaw_deg) if yaw_deg is not None else 0.0, 0.0)
        self.link.tx_packets += 1

    def goto_local_ned(self, north: float, east: float, down: float,
                       yaw_deg: Optional[float] = None) -> None:
        """Guided position target relative to the EKF origin, in metres NED."""
        mask = TYPE_MASK.POS_ONLY | (0 if yaw_deg is not None else TYPE_MASK.IGNORE_YAW)
        self.master.mav.set_position_target_local_ned_send(
            0, self.target_system, self.target_component,
            MAV_FRAME.LOCAL_OFFSET_NED, mask,
            float(north), float(east), float(down),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            math.radians(yaw_deg) if yaw_deg is not None else 0.0, 0.0)
        self.link.tx_packets += 1

    def set_velocity_ned(self, vn: float = 0.0, ve: float = 0.0, vd: float = 0.0,
                         yaw_deg: Optional[float] = None,
                         hold_alt_rel_m: Optional[float] = None) -> None:
        """Stream a velocity setpoint.

        ``vd`` is positive *down*, following MAVLink/NED convention, so a climb
        is a negative number - the single most common sign error in guided
        flight.  :meth:`set_velocity_up` takes the intuitive sign.

        If ``hold_alt_rel_m`` is given the vertical channel becomes a position
        hold at that altitude instead of a velocity, which is what a search
        pattern wants: altitude sets the ground sample distance and must not
        drift while the horizontal target streams.
        """
        if hold_alt_rel_m is not None:
            # IGNORE_YAW is set unless a heading was explicitly asked for.  This
            # branch is the one a survey flies - altitude held for a constant
            # ground sample distance, horizontal velocity streamed - and without
            # the bit every message re-points the nose at the direction of
            # travel.  Lanes reverse at each pass, so that means a 180 deg yaw
            # slew at full speed at every lane boundary, and the velocity
            # controller saturates its tilt limit before the turn finishes.
            mask = (TYPE_MASK.IGNORE_X | TYPE_MASK.IGNORE_Y
                    | TYPE_MASK.IGNORE_VZ | TYPE_MASK.IGNORE_AX
                    | TYPE_MASK.IGNORE_AY | TYPE_MASK.IGNORE_AZ
                    | TYPE_MASK.IGNORE_YAW_RATE
                    | (0 if yaw_deg is not None else TYPE_MASK.IGNORE_YAW))
            self.master.mav.set_position_target_global_int_send(
                0, self.target_system, self.target_component,
                MAV_FRAME.GLOBAL_REL_ALT_INT, mask,
                0, 0, float(hold_alt_rel_m),
                float(vn), float(ve), 0.0,          # vx, vy, vz
                0.0, 0.0, 0.0,                      # afx, afy, afz
                math.radians(yaw_deg) if yaw_deg is not None else 0.0,
                0.0)                                # yaw, yaw_rate
        else:
            mask = TYPE_MASK.VEL_ONLY | (0 if yaw_deg is not None
                                         else TYPE_MASK.IGNORE_YAW)
            self.master.mav.set_position_target_local_ned_send(
                0, self.target_system, self.target_component,
                MAV_FRAME.LOCAL_OFFSET_NED, mask,
                0.0, 0.0, 0.0,
                float(vn), float(ve), float(vd),  # vx, vy, vz
                0.0, 0.0, 0.0,                    # afx, afy, afz
                math.radians(yaw_deg) if yaw_deg is not None else 0.0,
                0.0)                              # yaw, yaw_rate
        self.link.tx_packets += 1

    def set_velocity_up(self, vn: float = 0.0, ve: float = 0.0, up: float = 0.0,
                        yaw_deg: Optional[float] = None,
                        hold_alt_rel_m: Optional[float] = None) -> None:
        """Velocity setpoint with the intuitive sign: ``up`` positive = climb."""
        self.set_velocity_ned(vn, ve, -up, yaw_deg=yaw_deg,
                              hold_alt_rel_m=hold_alt_rel_m)

    def guided_limits(self, timeout_s: float = 0.0, alt_min_m: float = 0.0,
                      alt_max_m: float = 0.0, horiz_max_m: float = 0.0,
                      strict: bool = False) -> bool:
        """``MAV_CMD_DO_GUIDED_LIMITS`` - the runaway guard.

        Guided mode will otherwise fly to a target forever.  If the link drops
        while a velocity setpoint is being streamed, the last one keeps being
        honoured and the aircraft flies until the battery dies.  This sets a
        timeout after which the vehicle leaves guided on its own, plus absolute
        altitude and horizontal-distance bounds.  For an unattended SAR sortie
        over a disaster area this is not optional: the alternative is a 3 kg
        aircraft with spinning carbon props leaving the search area at 12 m/s.
        """
        # ``strict=False`` by default because acceptance depends on state the
        # caller may not be in yet: several builds refuse DO_GUIDED_LIMITS while
        # disarmed or outside GUIDED, returning UNSUPPORTED rather than DENIED.
        # The right pattern is to call it, arm, then call it again - so a refusal
        # is logged and returned rather than raised.
        try:
            self.command_long(MAV_CMD.DO_GUIDED_LIMITS,
                              p1=float(timeout_s), p2=float(alt_min_m),
                              p3=float(alt_max_m), p4=float(horiz_max_m),
                              timeout=3.0)
            return True
        except CommandError as exc:
            if strict:
                raise
            log.warning("guided limits not accepted (%s); will retry after arming",
                        exc)
            self._pending_guided_limits = (timeout_s, alt_min_m, alt_max_m,
                                           horiz_max_m)
            return False

    def set_yaw(self, yaw_deg: float, speed_deg_s: float = 0.0,
                direction: int = 1, relative: bool = False) -> None:
        """``MAV_CMD_CONDITION_YAW``.  ``direction`` 1 = clockwise."""
        self.command_long(MAV_CMD.CONDITION_YAW, p1=wrap_360(yaw_deg),
                          p2=float(speed_deg_s), p3=float(direction),
                          p4=1.0 if relative else 0.0, timeout=3.0)

    def set_roi(self, lat: float, lon: float, alt_m: float = 0.0) -> None:
        """Point a gimbal at a ground location - used for confirmation looks."""
        lat_i, lon_i = latlon_to_int(lat, lon)
        self.command_long(MAV_CMD.DO_SET_ROI, p5=lat_i / 1e7, p6=lon_i / 1e7,
                          p7=float(alt_m), timeout=3.0)

    def do_reposition(self, lat: float, lon: float, alt_rel_m: float,
                      speed_ms: float = 0.0, yaw_deg: float = -1.0) -> None:
        """``MAV_CMD_DO_REPOSITION`` - guided goto that ignores terrain following."""
        self.command_long(MAV_CMD.DO_REPOSITION, p1=float(speed_ms), p2=-1.0,
                          p3=0.0, p4=math.radians(yaw_deg),
                          p5=float(lat), p6=float(lon), p7=float(alt_rel_m),
                          timeout=3.0)

    # ------------------------------------------------------------------ #
    # Payload: gripper / servo release (Drop-to-Confirm)
    # ------------------------------------------------------------------ #
    def set_servo(self, channel: int, pwm_us: int) -> None:
        """``DO_SET_SERVO`` on a spare AUX channel.

        This is the release mechanism for the Drop-to-Confirm payload: a servo
        latch on ``SERVO9_FUNCTION=59`` (Gripper), driven either by this command
        from a mission item or by ``RC9_OPTION=19`` from the operator.
        """
        self.command_long(MAV_CMD.DO_SET_SERVO, p1=float(channel),
                          p2=float(pwm_us), timeout=3.0)

    def gripper(self, release: bool, channel: int = 9,
                grab_pwm: int = 1100, release_pwm: int = 1900) -> None:
        """Open or close the gripper/release servo."""
        self.set_servo(channel, release_pwm if release else grab_pwm)

    def drop_payload(self, channel: int = 9, hold_open_s: float = 1.0,
                     grab_pwm: int = 1100, release_pwm: int = 1900) -> None:
        """Release, wait, and re-latch.

        A momentary release rather than a state change: the latch must close
        again or the next payload - and the aircraft's own drag profile - are
        both wrong.
        """
        self.gripper(True, channel, grab_pwm, release_pwm)
        time.sleep(max(0.05, hold_open_s))
        self.gripper(False, channel, grab_pwm, release_pwm)

    # ------------------------------------------------------------------ #
    # Terminal actions
    # ------------------------------------------------------------------ #
    def rtl(self) -> None:
        self.set_mode("RTL")

    def land(self) -> None:
        self.set_mode("LAND")

    def loiter(self) -> None:
        self.set_mode("LOITER")

    def brake(self) -> None:
        """``BRAKE`` mode: stop as fast as aerodynamics allow and hold."""
        self.set_mode("BRAKE")

    def reboot(self, autopilot: bool = True, companion: bool = False) -> None:
        self.command_long(MAV_CMD.PREFLIGHT_REBOOT_SHUTDOWN,
                          p1=1.0 if autopilot else 0.0,
                          p2=1.0 if companion else 0.0,
                          timeout=2.0, want_result=None)

    # ------------------------------------------------------------------ #
    # Missions
    # ------------------------------------------------------------------ #
    def upload_mission(self, items: Sequence[Dict[str, Any]],
                       timeout_per_item: float = 3.0) -> int:
        """Upload a mission from plain dicts.  Returns the item count.

        Accepts dicts rather than ``MAVLink_mission_item_int_message`` objects so
        a mission can live in a YAML file and be edited by a planner that has
        never heard of MAVLink.  Keys mirror the mission-item fields; anything
        omitted defaults to zero, which is what ArduPilot expects for unused
        parameters.
        """
        loader = mavwp.MAVWPLoader()
        for it in items:
            frame = it.get("frame", MAV_FRAME.GLOBAL_REL_ALT_INT)
            msg = self.master.mav.mission_item_int(
                self.target_system, self.target_component,
                int(it.get("seq", loader.count())),
                int(frame), int(it["command"]),
                int(it.get("current", 0)), int(it.get("autocontinue", 1)),
                float(it.get("param1", 0.0)), float(it.get("param2", 0.0)),
                float(it.get("param3", 0.0)), float(it.get("param4", 0.0)),
                int(round(float(it.get("x", 0.0)) * 1e7)) if abs(it.get("x", 0.0)) < 1000
                else int(it.get("x", 0)),
                int(round(float(it.get("y", 0.0)) * 1e7)) if abs(it.get("y", 0.0)) < 1000
                else int(it.get("y", 0)),
                float(it.get("z", 0.0)))
            loader.add(msg)
        try:
            loader.upload(timeout=timeout_per_item * max(1, len(items)))
        except Exception as exc:
            raise MavError(f"mission upload failed: {exc}") from exc
        log.info("uploaded %d mission items", loader.count())
        return int(loader.count())

    def download_mission(self, timeout: float = 15.0) -> List[Dict[str, Any]]:
        """Read the mission back off the vehicle.

        Used to verify an upload round-tripped intact - MAVLink's mission
        protocol is a multi-message handshake with no single acknowledgement, and
        a truncated upload produces a vehicle that flies a partial survey and
        reports success.
        """
        try:
            loader = mavwp.MAVWPLoader()
            loader.load(self.target)
        except Exception as exc:
            raise MavError(f"mission download failed: {exc}") from exc
        out: List[Dict[str, Any]] = []
        for i in range(loader.count()):
            m = loader.wp(i)
            out.append({"seq": m.seq, "command": m.command, "frame": m.frame,
                        "param1": m.param1, "param2": m.param2, "param3": m.param3,
                        "param4": m.param4, "x": m.x / 1e7 if abs(m.x) > 1000 else m.x,
                        "y": m.y / 1e7 if abs(m.y) > 1000 else m.y, "z": m.z,
                        "autocontinue": m.autocontinue})
        return out

    def mission_set_current(self, seq: int) -> None:
        self.master.waypoint_set_current_send(int(seq))
        self.link.tx_packets += 1

    # ------------------------------------------------------------------ #
    # Waiters used by mission logic
    # ------------------------------------------------------------------ #
    def wait_gps_fix(self, min_sats: int = 8, min_fix: int = GPS_FIX.FIX_3D,
                     timeout: float = 60.0) -> bool:
        """Block until a usable GNSS fix.  Returns False on timeout."""
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            if (self.telemetry.n_satellites >= min_sats
                    and self.telemetry.gps_fix >= min_fix):
                log.info("GPS fix: %d sats, type %d, hdop %.1f",
                         self.telemetry.n_satellites, self.telemetry.gps_fix,
                         self.telemetry.hdop)
                return True
        return False

    def wait_ekf_ready(self, timeout: float = 60.0,
                       need_position: bool = True) -> bool:
        """Block until the EKF reports a consistent attitude/position solution."""
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            a = self.telemetry.armed
            if a.ekf_ok and (a.position_ok or not need_position):
                return True
        return False

    def wait_altitude(self, alt_min_m: float, alt_max_m: float,
                      timeout: float = 60.0, relative: bool = True) -> bool:
        """Block until home-relative altitude enters a band."""
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            alt = self.telemetry.alt_rel_m if relative else self.telemetry.alt_msl_m
            if alt is not None and alt_min_m <= alt <= alt_max_m:
                return True
        return False

    def wait_position(self, lat: float, lon: float, radius_m: float = 2.0,
                      timeout: float = 60.0) -> bool:
        """Block until within ``radius_m`` of a lat/lon."""
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            if self.distance_to(lat, lon) <= radius_m:
                return True
        return False

    def wait_mode(self, mode: str, timeout: float = 20.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            if self.telemetry.armed.mode == mode.upper():
                return True
        return False

    def wait_disarmed(self, timeout: float = 60.0) -> bool:
        """Block until disarmed - the normal end-of-mission condition."""
        end = time.time() + timeout
        while time.time() < end:
            self.pump(0.1)
            if not self.telemetry.armed.armed:
                return True
        return False

    def distance_to(self, lat: float, lon: float) -> float:
        """Great-circle distance from the current position, metres."""
        if not self.telemetry.has_position:
            return float("inf")
        r = 6371000.0
        p1 = math.radians(self.telemetry.lat)
        p2 = math.radians(lat)
        dp = p2 - p1
        dl = math.radians(lon - self.telemetry.lon)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))

    # ------------------------------------------------------------------ #
    # GPS-denied operation
    # ------------------------------------------------------------------ #
    def set_ekf_source_set(self, source_set: int, timeout: float = 3.0) -> None:
        """``MAV_CMD_SET_EKF_SOURCE_SET`` (42007).

        Switches the whole aiding source set in one command - position, velocity
        and yaw - which is what makes an in-flight transition from GNSS to
        visual-inertial survivable.  Doing it with parameter writes instead
        requires an EKF reset, and during that reset the vehicle has no position
        at all, which at 60 m over a flooded street ends the sortie.

        The target sets must already exist in the vehicle's parameters
        (``EK3_SRC2_POSXY`` etc.); this command only selects between them.
        """
        self.command_long(MAV_CMD.SET_EKF_SOURCE_SET, p1=float(source_set),
                          timeout=timeout)
        self.telemetry.ekf_source_set = int(source_set)
        self.telemetry.ekf_source_name = SOURCE_SET.name(source_set)
        log.info("EKF source set -> %d (%s)", source_set, SOURCE_SET.name(source_set))

    def send_external_position(self, north_m: float, east_m: float, down_m: float,
                               yaw_deg: float = 0.0,
                               t_usec: Optional[int] = None,
                               sigma_m: float = 0.1,
                               reset_counter: int = 0) -> None:
        """``VISION_POSITION_ESTIMATE`` (102) - feed the EKF an external nav fix.

        Local NED metres from the EKF origin, which is what a visual-inertial
        odometry stack naturally produces and what ``EK3_SRCn_POSXY=6``
        (ExternalNav) expects.

        Two details that are easy to get wrong and produce a filter that silently
        rejects everything.  The timestamp is in **microseconds** on the vehicle's
        boot clock, not milliseconds - a ms value looks like a sample from 1970
        and EKF3 discards it as stale.  And the covariance is the 21-element
        upper triangle of a 6x6, of which only the three position diagonals are
        filled here; the rest are NaN, which the receiver is required to ignore.

        ``sigma_m`` becomes those diagonals.  Claiming the aiding is better than
        it is makes the EKF trust a diverging VIO solution over its own dead
        reckoning, which is worse than not aiding at all.
        """
        t_usec = int(t_usec if t_usec is not None
                     else self.vehicle_time_boot_ms * 1000)
        var = float(sigma_m) ** 2
        cov = [float("nan")] * 21
        cov[0] = cov[6] = cov[11] = var
        self.master.mav.vision_position_estimate_send(
            t_usec, float(north_m), float(east_m), float(down_m),
            0.0, 0.0, math.radians(float(yaw_deg)),
            cov, int(reset_counter))
        self.link.tx_packets += 1

    def send_odometry(self, north_m: float, east_m: float, down_m: float,
                      vn: float = 0.0, ve: float = 0.0, vd: float = 0.0,
                      q: Optional[Sequence[float]] = None,
                      yaw_deg: Optional[float] = None,
                      t_usec: Optional[int] = None,
                      pos_sigma_m: float = 0.1,
                      vel_sigma_ms: float = 0.15,
                      ang_sigma_rad: float = 0.10,
                      reset_counter: int = 0,
                      quality: int = 100) -> None:
        """``ODOMETRY`` (331) - the preferred external nav message.

        Carries a full covariance and explicit parent/child frames, so the EKF
        knows what it is being given rather than having to assume.  EKF3 accepts
        it at the same ExternalNav source setting as
        :meth:`send_external_position`, and it is what a real VIO stack publishes.

        **The frames are not negotiable.**  ``GCS_MAVLINK::handle_odometry``
        begins with::

            if (m.frame_id != MAV_FRAME_LOCAL_FRD ||
                m.child_frame_id != MAV_FRAME_BODY_FRD) {
                return;          // "only support local FRD frame data"
            }

        So ``LOCAL_FRD`` (20) and ``BODY_FRD`` (12) exactly - not ``LOCAL_NED``
        (1), not ``BODY_NED`` (8), which is what every other part of MAVLink
        uses and therefore what one reaches for.  The rejection is a bare
        ``return`` before anything is recorded, so the VisOdom backend's
        last-update timestamp never advances, ``AP_VisualOdom_Backend::healthy()``
        (which requires a sample within ``AP_VISUALODOM_TIMEOUT_MS`` = 300 ms)
        stays false, and the only symptom the operator ever sees is
        ``Arm: VisOdom: not healthy`` - no STATUSTEXT, no MAVLink error, nothing
        in the logs to say the frames were the problem.  Debugging that from the
        symptom alone means reading the flight-stack source, so the constants are
        written out here with their numeric values.

        Because the child frame is the *body*, ``vx/vy/vz`` must be body-relative;
        the handler does ``vel = q * vel`` to bring them back to earth axes.
        Passing NED velocities with an attitude quaternion therefore double-
        rotates them, and the aircraft flies at an error that tracks its own
        attitude.  :class:`~sar.nav.TelemetryVioSource` does the inverse rotation
        before publishing, which is the only reason this is safe to call with the
        body velocity it produces.

        The two covariance arrays are both 21 elements but have *different*
        diagonal indices: ``pose_covariance`` is the upper triangle of the 6x6
        position-plus-orientation block (position diagonals 0, 6, 11; orientation
        diagonals 15, 18, 20), while ``velocity_covariance`` is the upper triangle
        of a 3x3 (diagonals 0, 3, 5).  The orientation diagonals must be filled
        - see the comment below.
        """
        t_usec = int(t_usec if t_usec is not None
                     else self.vehicle_time_boot_ms * 1000)
        if q is None:
            if yaw_deg is None:
                q = [1.0, 0.0, 0.0, 0.0]
            else:
                h = math.radians(float(yaw_deg)) / 2.0
                q = [math.cos(h), 0.0, 0.0, math.sin(h)]
        q = [float(x) for x in q]
        if len(q) != 4:
            raise ValueError(f"quaternion must have 4 elements, got {len(q)}")

        pose_cov = [float("nan")] * 21
        pose_cov[0] = pose_cov[6] = pose_cov[11] = float(pos_sigma_m) ** 2
        # The three orientation diagonals have to be real numbers, not NaN.
        # ArduPilot's handler guards only on pose_covariance[0] and then takes
        # sqrt(cov[15]+cov[18]+cov[20]) unconditionally, so NaN there propagates
        # straight into the angular error it feeds the EKF.
        var_ang = (float(ang_sigma_rad) ** 2) / 3.0
        pose_cov[15] = pose_cov[18] = pose_cov[20] = var_ang
        vel_cov = [float("nan")] * 21
        vel_cov[0] = vel_cov[3] = vel_cov[5] = float(vel_sigma_ms) ** 2

        # quality is 0..100 where 0 means "unknown"; VISO_QUAL_MIN gates on it.
        q_pct = max(1, min(100, int(quality)))
        self.master.mav.odometry_send(
            t_usec,
            mavutil.mavlink.MAV_FRAME_LOCAL_FRD,       # parent frame  (20)
            mavutil.mavlink.MAV_FRAME_BODY_FRD,        # child frame   (12)
            float(north_m), float(east_m), float(down_m),
            q, float(vn), float(ve), float(vd),
            0.0, 0.0, 0.0,
            pose_cov, vel_cov,
            int(reset_counter),
            mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO,
            q_pct)
        self.link.tx_packets += 1

    # ------------------------------------------------------------------ #
    # Bridge into the perception stack
    # ------------------------------------------------------------------ #
    def nav_quality(self) -> Dict[str, Any]:
        """Telemetry expressed as navigation-quality inputs for geo-tagging.

        This is the seam between the flight stack and the perception stack: the
        geo-tagger needs a position sigma, an attitude sigma and a time-since-fix,
        and MAVLink does not provide those three directly.  They are derived from
        ``EKF_STATUS_REPORT`` variances and the age of the last GNSS message, so
        a detection's error ellipse widens automatically the moment the vehicle
        loses its fix - which is the mechanism that makes geo-tagged survivor
        positions honest during a GPS-denied sortie instead of confidently wrong.
        """
        tel = self.telemetry
        now = time.time()
        gps_age = tel.age("gps", now)
        denied = tel.gps_denied

        # Attitude sigma.  This board has no internal compass, so yaw does not
        # come from a magnetometer: it comes from GNSS course (or from the
        # external nav source), and GNSS course accuracy is set by how far the
        # aircraft has moved between fixes.  At hover the course is undefined and
        # yaw sigma is large; at 8 m/s with a 0.15 m/s velocity error it is about
        # 1 deg.  Roll and pitch are inertially observable and stay tight unless
        # the EKF itself is unhappy.
        vel_sigma = (NOMINAL_VEL_SIGMA_MS
                     * math.sqrt(max(tel.vel_variance_ratio, 0.0))
                     if tel.vel_variance_ratio else 0.4)
        speed = max(tel.speed_ms, 1e-3)
        # Bounded at 5 deg.  At hover the GNSS course is genuinely undefined, and
        # atan2 of a velocity error over a near-zero speed says 89 deg - true of
        # the *course* but not of the EKF's yaw estimate, which is still held by
        # the filter and drifts slowly rather than being random.  Five degrees is
        # the measured order of magnitude for a compass-less copter at hover, and
        # it is the number that should widen the geo-tag error ellipse off-nadir.
        yaw_sigma = (math.degrees(math.atan2(vel_sigma, max(speed, 0.5)))
                     if not denied else 8.0)
        rp_sigma = 0.35 if tel.armed.ekf_ok else 2.0
        att_sigma = min(5.0, max(rp_sigma, yaw_sigma))
        return {
            "pos_sigma_m": tel.horizontal_pos_sigma_m,
            "vel_sigma_ms": vel_sigma,
            "att_sigma_deg": att_sigma,
            "alt_sigma_m": (NOMINAL_ALT_SIGMA_M
                            * math.sqrt(max(tel.hgt_variance_ratio, 0.0))
                            if tel.hgt_variance_ratio else NOMINAL_ALT_SIGMA_M),
            "seconds_since_fix": (0.0 if not denied else max(0.0, gps_age)),
            "n_satellites": tel.n_satellites,
            "hdop": tel.hdop,
            "source": ("dead_reckoning" if denied and tel.ekf_source_set != 1
                       else "ekf_external_nav" if not denied and tel.ekf_source_set != 1
                       else "gnss"),
            "gps_fix": tel.gps_fix,
            "gps_denied": denied,
            "ekf_ok": tel.armed.ekf_ok,
            "rssi_dbm": tel.rssi_dbm,
            "latency_s": tel.age("position", now),
        }
