"""MiniSITL - a pure-Python software-in-the-loop vehicle model.

Why this exists
---------------
ArduPilot SITL is the higher-fidelity backend and :mod:`sar.sim.sitl_launch`
drives it when a built ``arducopter`` binary is available.  But it needs a 3 GB
checkout, a working compiler toolchain and several minutes of build, none of
which belong in the critical path of testing a *search algorithm*.  MiniSITL is
the backend that always runs: pure Python, no external process, and it speaks
MAVLink 2 over TCP on port 5760 exactly like ArduPilot SITL does.

That last point is the design constraint that matters.  The client code -
:class:`~sar.mavlink.MavConnection` and everything above it - cannot tell the
two apart, so a mission that flies against MiniSITL flies against ArduPilot SITL
and against a physical TBS Lucid H743 Wing with no changes.  Anywhere the two
backends *can* be told apart is a place where the mission code has leaked an
assumption about the simulator into the logic, which is the bug class this
module exists to prevent.

What is modelled faithfully
---------------------------
* Rigid-body quadrotor dynamics with per-rotor thrust allocation, cascaded
  attitude/position PID control, and a real powertrain (see
  :mod:`sar.vehicle.dynamics` and :mod:`sar.vehicle.power`).  Energy is
  integrated from thrust and airspeed, so endurance numbers come out of the
  flight rather than being asserted about it.
* GNSS with spatially-varying denial and multipath, IMU, barometer, optical
  flow and VIO sensors from :mod:`sar.vehicle.sensors`.
* A three-source-set EKF that reproduces ArduPilot's *observable* behaviour:
  the ``LOCAL_FRD``/``BODY_FRD`` frame requirement on ``ODOMETRY``, the 300 ms
  VisOdom health timeout, ``CONST_POS_MODE`` when aiding is lost, and the
  ``EKF_STATUS_REPORT`` flag bits that gate arming.
* Pre-arm checks that fail for the same reasons and say so in ``STATUSTEXT``.
* Parameter storage loaded from ``configs/ardupilot_sitl.parm``, so one file
  governs both backends and a parameter that is wrong shows up in both.

What is deliberately NOT modelled
---------------------------------
Prop wash and ground effect, motor/ESC thermal limits, magnetometer
interference from current draw in detail, vibration coupling into the IMU, RC
link behaviour beyond a heartbeat, waypoint missions in ``AUTO``, and the exact
EKF3 covariance propagation.  MiniSITL is a *behavioural* stand-in for
navigation and mission logic, not a substitute for tuning.  Do not use it to
predict PID gains, endurance to the minute, or whether a specific airframe
vibration level will upset a specific filter - use the hardware for that.

Typical use::

    sim = MiniSITL()
    sim.start()
    conn = sim.connect()          # a real MavConnection over TCP
    conn.arm(); conn.takeoff(35)
    ...
    sim.stop()
"""

from __future__ import annotations

import logging
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymavlink.dialects.v20 import ardupilotmega as mavlink2

from sar.core.geo import GeoPoint, local_to_wgs84, wgs84_to_local
from sar.mavlink.connection import (NOMINAL_ALT_SIGMA_M,
                                    NOMINAL_GNSS_SIGMA_M,
                                    NOMINAL_VEL_SIGMA_MS)
from sar.mavlink.protocol import (COPTER_MODE_NAMES, COPTER_MODES, MAV_CMD,
                                  wrap_pi)
from sar.vehicle.dynamics import Autopilot, PlantState
from sar.vehicle.power import AirframeConfig, BatteryConfig, Powertrain
from sar.vehicle.sensors import (BaroSensor, FlowSample, GpsEnvironment,
                                 GpsSample, GpsSensor, OpticalFlowSensor)

log = logging.getLogger("sar.sim.sitl")

# --------------------------------------------------------------------------- #
# Constants that mirror ArduPilot, with the numeric values written out because
# they are not what MAVLink uses elsewhere and getting them wrong is silent.
# --------------------------------------------------------------------------- #
FRAME_LOCAL_NED = 1
FRAME_BODY_NED = 8
FRAME_BODY_FRD = 12
FRAME_LOCAL_FRD = 20

#: ``AP_VISUALODOM_TIMEOUT_MS`` - VisOdom is healthy only if a sample arrived
#: within this window.  The ODOMETRY handler rejects bad frames *before*
#: touching the timestamp, so a feeder publishing the wrong frame IDs leaves
#: this permanently expired and the only symptom is ``Arm: VisOdom: not healthy``.
VISUAL_ODOM_TIMEOUT_MS = 300

# EKF_STATUS_REPORT flag bits (mavlink.h).
EKF_ATTITUDE = 1
EKF_VELOCITY_HORIZ = 2
EKF_VELOCITY_VERT = 4
EKF_POS_HORIZ_REL = 8
EKF_POS_HORIZ_ABS = 16
EKF_POS_VERT_ABS = 32
EKF_POS_VERT_AGL = 64
EKF_CONST_POS_MODE = 128
EKF_PRED_POS_HORIZ_REL = 256
EKF_PRED_POS_HORIZ_ABS = 512
EKF_UNINITIALIZED = 1024
EKF_GPS_GLITCHING = 32768

# MAV_CMD identifiers, taken from the client's own table so the two cannot
# drift.  Two of these are easy to get wrong from memory: DO_SET_SERVO is 183
# (not 2052, which is the payload-deploy range) and DO_GUIDED_LIMITS is 222 (not
# 3002).  A wrong id is not a loud failure - the vehicle replies UNSUPPORTED to
# a command it actually implements, so a payload release looks like a hardware
# fault.
CMD_NAV_TAKEOFF = MAV_CMD.NAV_TAKEOFF
CMD_NAV_LAND = MAV_CMD.NAV_LAND
CMD_NAV_RETURN_TO_LAUNCH = MAV_CMD.NAV_RETURN_TO_LAUNCH
CMD_COMPONENT_ARM_DISARM = MAV_CMD.COMPONENT_ARM_DISARM
CMD_DO_SET_SERVO = MAV_CMD.DO_SET_SERVO
CMD_DO_GUIDED_LIMITS = MAV_CMD.DO_GUIDED_LIMITS
CMD_DO_SET_MODE = MAV_CMD.DO_SET_MODE
CMD_PREFLIGHT_REBOOT = MAV_CMD.PREFLIGHT_REBOOT_SHUTDOWN
#: ArduPilot extension, not in the common set.
CMD_SET_EKF_SOURCE_SET = 42007

#: MAVLink message ids accepted by ``MAV_CMD_SET_MESSAGE_INTERVAL``.  Only the
#: ones this model actually publishes are listed; asking for anything else is
#: rejected rather than silently ignored, which is what the real stack does.
_MESSAGE_FOR_ID = {
    0: "HEARTBEAT", 1: "SYS_STATUS", 24: "GPS_RAW_INT", 30: "ATTITUDE",
    33: "GLOBAL_POSITION_INT", 35: "RC_CHANNELS_RAW", 36: "SERVO_OUTPUT_RAW",
    74: "VFR_HUD", 147: "BATTERY_STATUS", 193: "EKF_STATUS_REPORT",
    253: "STATUSTEXT",
}

# Imported from the client's own table rather than redeclared here.  ArduPilot's
# Copter numbering is not contiguous and not alphabetical - GUIDED is 4, LOITER
# 5, RTL 6, LAND 9 - so a tuple built in the "obvious" order silently maps
# GUIDED onto AUTO.  The aircraft then accepts the mode change, reports a
# different mode in telemetry, and the client times out waiting for a
# confirmation that will never come.  Two definitions of one enum is how that
# happens; one definition makes it impossible.
MODES = tuple(sorted(COPTER_MODE_NAMES.values()))
MODE_NUMBER = COPTER_MODES
MODE_NAME = COPTER_MODE_NAMES

G0 = 9.80665
#: Default home, matching ``configs/ardupilot_sitl.parm`` and the SAHYOG trial
#: site near Kota, Rajasthan.
DEFAULT_HOME = (25.185, 75.8357, 250.0, 0.0)


class MiniSitlError(RuntimeError):
    """Raised when MiniSITL cannot start (port in use, bad params, ...)."""


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def load_param_file(path: Path) -> Dict[str, float]:
    """Read an ArduPilot ``.parm`` file into a dict.

    The format is ``KEY,VALUE`` per line with ``#`` comments, which is what
    Mission Planner and ``--defaults`` both use.  Parsing it here rather than
    hard-coding values is what keeps the two backends honest: a source set that
    is wrong in the parameter file is wrong for both of them.
    """
    out: Dict[str, float] = {}
    p = Path(path)
    if not p.exists():
        return out
    for raw in p.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "," not in line:
            continue
        key, _, val = line.partition(",")
        try:
            out[key.strip().upper()[:16]] = float(val.strip())
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------- #
# EKF stand-in
# --------------------------------------------------------------------------- #
@dataclass
class EkfEstimate:
    """What the filter believes, as distinct from what is true.

    Keeping these separate is the point of the whole exercise: a geotag is only
    as good as the estimate behind it, and the estimate is the thing that
    degrades silently.  ``MiniSITL`` publishes this over MAVLink and keeps
    :attr:`~sar.vehicle.dynamics.PlantState` private to the test harness, so the
    mission can only ever see what a real aircraft would expose.
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rad: float = 0.0
    alt_msl: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    flags: int = 0
    pos_sigma_m: float = 1.0
    vel_sigma_ms: float = 0.3
    source: str = "none"
    source_set: int = 0
    const_pos: bool = False
    origin_set: bool = False
    #: Seconds since the active source last provided aiding.
    since_aiding_s: float = float("inf")

    def flag_names(self) -> List[str]:
        names = {EKF_ATTITUDE: "ATTITUDE", EKF_VELOCITY_HORIZ: "VEL_HORIZ",
                 EKF_VELOCITY_VERT: "VEL_VERT", EKF_POS_HORIZ_REL: "POS_HORIZ_REL",
                 EKF_POS_HORIZ_ABS: "POS_HORIZ_ABS", EKF_POS_VERT_ABS: "POS_VERT_ABS",
                 EKF_POS_VERT_AGL: "POS_VERT_AGL", EKF_CONST_POS_MODE: "CONST_POS_MODE",
                 EKF_PRED_POS_HORIZ_REL: "PRED_POS_HORIZ_REL",
                 EKF_PRED_POS_HORIZ_ABS: "PRED_POS_HORIZ_ABS",
                 EKF_UNINITIALIZED: "UNINITIALIZED",
                 EKF_GPS_GLITCHING: "GPS_GLITCHING"}
        return [n for b, n in sorted(names.items()) if self.flags & b]


class EkfEmulator:
    """Three-source-set filter with ArduPilot's observable denial behaviour.

    This is not a Kalman filter.  It is a *behavioural* model of one: it tracks
    the active source when aiding is present, dead-reckons with an accumulating
    error when it is not, and reports the flag pattern ArduPilot would report so
    that the arming and nav-quality logic upstream is exercised by the same
    transitions it will see in the field.

    The dead-reckoning error is the part worth getting right.  A filter with no
    aiding does not freeze - it integrates an accelerometer bias, so position
    error grows roughly with the square of time.  That is why a GPS-denied
    sortie has a shrinking window in which its geotags are worth anything, and
    why :class:`~sar.nav.NavQualityMonitor` inflates the error ellipse as a
    function of ``since_aiding_s`` rather than switching to a constant the moment
    the fix drops.
    """

    #: Mirrors ``EK3_SRCn_*`` in the parameter file, keyed by source set.
    SRC_SET_NAMES = {0: "None", 1: "GNSS", 2: "ExtNav", 3: "OptFlow"}

    # Source codes, matching AP_NavEKF_Source's enums (not MAVLink's).
    SRC_NONE, SRC_BARO, SRC_GPS, SRC_BEACON, SRC_OPTFLOW, SRC_EXTNAV = 0, 1, 3, 4, 5, 6

    def __init__(self, params: Dict[str, float], home: GeoPoint,
                 accel_bias_ms2: float = 0.05, seed: int = 21) -> None:
        self.p = params
        self.home = home
        self.rng = np.random.default_rng(seed)
        #: Accel bias that survives calibration.  Drives the dead-reckoning
        #: divergence rate, and therefore how long a denial is survivable.
        self.accel_bias = float(accel_bias_ms2)
        # A calibration residual is a *constant* offset in a fixed direction, not
        # white noise: velocity drifts linearly and position quadratically, which
        # is the signature that makes a denial survivable for tens of seconds and
        # unsurvivable for minutes.  Drawing it once is what produces that.
        d = self.rng.normal(size=3)
        self._accel_bias_vec = self.accel_bias * d / (np.linalg.norm(d) or 1.0)
        self.source_set = 1
        #: Set on every :meth:`update`; the health properties compare against it.
        self.now_ms = 0
        self.est = EkfEstimate()
        self._vel_err = np.zeros(3)
        self._pos_err = np.zeros(3)
        self._origin_set = False
        self._since_aiding = float("inf")

        # External-nav (VisOdom) state
        self._extnav_pos: Optional[np.ndarray] = None
        self._extnav_vel: Optional[np.ndarray] = None
        self._extnav_yaw: float = 0.0
        self._extnav_sigma: float = 0.15
        self._extnav_last_ms: int = -10_000
        self.extnav_samples = 0
        self.extnav_rejected_frames = 0
        self.extnav_rejected_quality = 0

        # Optical flow state
        self._flow_last_ms: int = -10_000
        self._flow_vel: Optional[np.ndarray] = None
        self._flow_quality = 0

        self._gps_last_ms: int = -10_000
        self._baro_alt: float = home.alt

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def ingest_odometry(self, t_ms: int, frame_id: int, child_frame_id: int,
                        xyz: Sequence[float], q: Sequence[float],
                        vxyz: Sequence[float], pose_cov: Sequence[float],
                        quality: int = 100) -> bool:
        """Consume an ``ODOMETRY`` (331) message, with ArduPilot's frame check.

        Reproduces the guard in ``GCS_MAVLINK::handle_odometry``:

            if (m.frame_id != MAV_FRAME_LOCAL_FRD ||
                m.child_frame_id != MAV_FRAME_BODY_FRD) return;

        The rejection happens before the health timestamp is updated, so a
        feeder on the wrong frames leaves VisOdom permanently unhealthy with no
        error anywhere.  MiniSITL counts those rejections so a test can assert
        on them instead of staring at ``Arm: VisOdom: not healthy``.

        Velocities arrive in the *body* frame (that is what ``BODY_FRD`` means)
        and are rotated to earth axes with the supplied quaternion, exactly as
        the flight stack does.
        """
        if frame_id != FRAME_LOCAL_FRD or child_frame_id != FRAME_BODY_FRD:
            self.extnav_rejected_frames += 1
            return False
        qn = math.sqrt(sum(float(x) ** 2 for x in q)) or 1.0
        qw, qx, qy, qz = (float(x) / qn for x in q)
        vx, vy, vz = (float(x) for x in vxyz)
        # body -> earth, v' = q * v * q^-1  (expanded, avoiding a dependency)
        tx, ty, tz = 2.0 * (qy * vz - qz * vy), 2.0 * (qz * vx - qx * vz), \
            2.0 * (qx * vy - qy * vx)
        evx = vx + qw * tx + (qy * tz - qz * ty)
        evy = vy + qw * ty + (qz * tx - qx * tz)
        evz = vz + qw * tz + (qx * ty - qy * tx)

        qmin = int(self.p.get("VISO_QUAL_MIN", 0))
        if quality < qmin:
            self.extnav_rejected_quality += 1
            return False

        self._extnav_pos = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        self._extnav_vel = np.array([evx, evy, evz])
        self._extnav_yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                                      1.0 - 2.0 * (qy * qy + qz * qz))
        try:
            var = sum(float(pose_cov[i]) for i in (0, 6, 11))
            self._extnav_sigma = math.sqrt(max(var, 1e-9)) if math.isfinite(var) \
                else float(self.p.get("VISO_POS_M_NSE", 0.15))
        except (ValueError, IndexError):
            self._extnav_sigma = float(self.p.get("VISO_POS_M_NSE", 0.15))
        self._extnav_last_ms = t_ms
        self.extnav_samples += 1
        return True

    def ingest_vision_position(self, t_ms: int, xyz: Sequence[float],
                               rpy: Sequence[float],
                               cov: Sequence[float]) -> bool:
        """Consume ``VISION_POSITION_ESTIMATE`` (102) - earth-frame, no child frame.

        The simpler of the two external-nav messages, and the one to fall back on
        when a VIO stack cannot produce a covariance it trusts.  It carries no
        velocity, so with this alone ``EK3_SRCn_VELXY`` must stay on something
        other than ExternalNav or the filter has no velocity aiding.
        """
        self._extnav_pos = np.array([float(x) for x in xyz[:3]])
        self._extnav_vel = None
        self._extnav_yaw = float(rpy[2]) if len(rpy) > 2 else 0.0
        try:
            var = sum(float(cov[i]) for i in (0, 6, 11))
            self._extnav_sigma = math.sqrt(max(var, 1e-9)) if math.isfinite(var) \
                else float(self.p.get("VISO_POS_M_NSE", 0.15))
        except (ValueError, IndexError):
            self._extnav_sigma = float(self.p.get("VISO_POS_M_NSE", 0.15))
        self._extnav_last_ms = t_ms
        self.extnav_samples += 1
        return True

    def ingest_vision_speed(self, t_ms: int, vel_ned: Sequence[float]) -> bool:
        """Consume ``VISION_SPEED_ESTIMATE`` (104) - velocity only, earth axes."""
        self._extnav_vel = np.array([float(x) for x in vel_ned[:3]])
        if self._extnav_pos is None:
            return False
        self._extnav_last_ms = t_ms
        return True

    def ingest_optical_flow(self, t_ms: int, flow: FlowSample,
                            height_m: float) -> bool:
        """Consume an optical-flow measurement and convert it to a ground velocity.

        Flow gives angular rate; multiplied by height above ground it becomes
        linear velocity.  That is why flow-only navigation degrades with
        altitude - the same angular noise maps to a larger velocity error the
        higher the aircraft is - and why a flow-only survey is flown low.
        """
        if not flow.valid or height_m <= 0.05:
            return False
        self._flow_last_ms = t_ms
        self._flow_quality = int(flow.quality)
        # Body-frame flow rates about X and Y, to earth-frame horizontal speed.
        self._flow_vel = np.array([flow.flow_rate[1] * height_m,
                                   -flow.flow_rate[0] * height_m, 0.0])
        return True

    def ingest_gps(self, t_ms: int, gps: GpsSample) -> None:
        if gps.valid and gps.fix_type >= 3:
            self._gps_last_ms = t_ms

    def ingest_baro(self, alt_msl: float) -> None:
        self._baro_alt = float(alt_msl)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    @property
    def viso_type(self) -> int:
        return int(self.p.get("VISO_TYPE", 0))

    @property
    def flow_type(self) -> int:
        return int(self.p.get("FLOW_TYPE", 0))

    @property
    def viso_healthy(self) -> bool:
        """VisOdom backend health: a sample within the 300 ms timeout."""
        return self.viso_type != 0 and \
            (self.now_ms - self._extnav_last_ms) < VISUAL_ODOM_TIMEOUT_MS

    @property
    def flow_healthy(self) -> bool:
        return self.flow_type != 0 and \
            (self.now_ms - self._flow_last_ms) < VISUAL_ODOM_TIMEOUT_MS

    # ------------------------------------------------------------------ #
    # Source-set management
    # ------------------------------------------------------------------ #
    def set_source_set(self, n: int) -> Tuple[bool, str]:
        """``MAV_CMD_SET_EKF_SOURCE_SET`` (42007).

        Validates the destination before switching, the way ArduPilot does:
        naming a source whose sensor is absent or unhealthy is refused rather
        than accepted and then silently left without aiding.  The refusal reason
        is returned so a test can assert on it.
        """
        if n not in (1, 2, 3):
            return False, f"invalid source set {n}"
        if n == 2:
            if self.viso_type == 0:
                return False, "EK3 sources require VisualOdom (VISO_TYPE=0)"
            if not self.viso_healthy:
                return False, "VisOdom: not healthy"
        if n == 3:
            if self.flow_type == 0:
                return False, "EK3 sources require OpticalFlow (FLOW_TYPE=0)"
            if not self.flow_healthy:
                return False, "OpticalFlow: not healthy"
        if n == self.source_set:
            return True, "already active"
        self.source_set = int(n)
        # Re-anchor the estimate to the new source, which is what the real EKF
        # does on a source change; without it the switch would inject a step.
        self._realign()
        return True, f"switched to {self.SRC_SET_NAMES[n]}"

    def _realign(self) -> None:
        if self.source_set == 2 and self._extnav_pos is not None:
            self.est.pos = self._extnav_pos.copy()
            if self._extnav_vel is not None:
                self.est.vel = self._extnav_vel.copy()
            self._pos_err[:] = 0.0
            self._vel_err[:] = 0.0
        elif self.source_set == 3:
            self._pos_err[:] = 0.0

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, t_ms: int, dt: float, truth: PlantState,
               gps: GpsSample) -> EkfEstimate:
        """Advance the estimate one step.  Returns what is published."""
        self.now_ms = t_ms
        self.ingest_gps(t_ms, gps)

        aiding, source, sigma = self._active_aiding(gps)
        self._since_aiding = 0.0 if aiding else self._since_aiding + dt

        if not self._origin_set and aiding and source == "gnss":
            self._origin_set = True
            self.est.origin_set = True

        # --- prediction: always propagate, whether or not aiding is present ---
        # A filter that only blends toward measurements does not move between
        # them.  That is invisible while a 30 Hz source is healthy, and fatal the
        # moment a measurement goes stale or self-referential: the estimate
        # freezes in place while the aircraft keeps flying, so the error grows at
        # the full airspeed and the reported sigma stays small because nothing
        # told the filter it had stopped being corrected.
        self.est.pos = self.est.pos + self.est.vel * dt

        if aiding:
            meas = self._measurement(source, gps, truth)
            if meas is not None:
                mpos, mvel = meas
                if mpos is not None:
                    # Complementary blend: trust the measurement, but not so hard
                    # that a single GNSS multipath jump teleports the estimate.
                    alpha = float(np.clip(dt / 0.35, 0.0, 1.0))
                    self.est.pos += alpha * (mpos - self.est.pos)
                    self._pos_err *= (1.0 - alpha)
                    self.est.pos_sigma_m = sigma
                if mvel is not None:
                    alpha_v = float(np.clip(dt / 0.20, 0.0, 1.0))
                    self.est.vel += alpha_v * (mvel - self.est.vel)
                    self._vel_err *= (1.0 - alpha_v)
                    self.est.vel_sigma_ms = 0.3
        else:
            # --- correction absent: dead reckoning ---
            # The bias integrates into velocity and velocity into position, so
            # position error grows ~t^2 while the reported sigma grows ~t.
            # Under-reporting that growth is what would make a GPS-denied geotag
            # look trustworthy, so both are inflated here rather than held.
            #
            # Note what is NOT used: the plant's true velocity.  The filter has no
            # access to it, and a model that reaches for truth produces a
            # dead-reckoning error of zero and proves nothing about denial.
            self.est.vel = self.est.vel + self._accel_bias_vec * dt
            self._vel_err += self._accel_bias_vec * dt
            self.est.pos_sigma_m = min(
                250.0, self.est.pos_sigma_m + dt * (0.35 + 0.06 * self._since_aiding))
            self.est.vel_sigma_ms = min(
                25.0, self.est.vel_sigma_ms + dt * 0.08)
        self.est.source = source
        self.est.source_set = self.source_set
        self.est.const_pos = (not aiding) or self._since_aiding > 0.5
        self.est.since_aiding_s = self._since_aiding

        # Height comes from the barometer unless the active set uses GPS,
        # matching EK3_SRCn_POSZ.
        posz = int(self.p.get(f"EK3_SRC{self.source_set}_POSZ", 3))
        if posz == self.SRC_GPS and gps.valid and gps.fix_type >= 3:
            self.est.alt_msl += (gps.alt_msl - self.est.alt_msl) * min(1.0, dt / 0.5)
        elif posz == self.SRC_BARO or self._baro_alt != self.home.alt:
            # The barometer is an onboard sensor, so height survives a GNSS
            # denial even when horizontal position does not - which is why a
            # GPS-denied survey can still hold altitude safely, and why the
            # denial is survivable for the terrain-clearance problem but not for
            # the geotagging one.
            self.est.alt_msl += (self._baro_alt - self.est.alt_msl) * min(1.0, dt / 0.3)
        else:
            # No aiding at all: propagate with the filter's own vertical
            # velocity.  Never the plant's - see the note in ``update``.
            self.est.alt_msl -= self.est.vel[2] * dt

        gp = local_to_wgs84(float(self.est.pos[0]), float(self.est.pos[1]),
                            -(self.est.alt_msl - self.home.alt), self.home)
        self.est.lat, self.est.lon = gp.lat, gp.lon
        self.est.yaw_rad = float(truth.euler[2])
        self.est.flags = self._flags(aiding, source, truth)
        return self.est

    def _active_aiding(self, gps: GpsSample) -> Tuple[bool, str, float]:
        n = self.source_set
        if n == 1:
            ok = gps.valid and gps.fix_type >= 3 and \
                (self.now_ms - self._gps_last_ms) < 2000
            return ok, "gnss", max(0.4, gps.h_accuracy_mm / 1000.0)
        if n == 2:
            return self.viso_healthy, "extnav", max(0.1, self._extnav_sigma)
        if n == 3:
            # Flow gives velocity only; position is dead-reckoned from it, so
            # the filter never reports absolute position in this mode.
            return self.flow_healthy, "optflow", 250.0
        return False, "none", 250.0

    def _measurement(self, source: str, gps: GpsSample,
                     truth: PlantState) -> Optional[Tuple[np.ndarray,
                                                          Optional[np.ndarray]]]:
        if source == "gnss":
            n, e, _d = wgs84_to_local(GeoPoint(gps.lat, gps.lon, gps.alt_msl),
                                      self.home)
            pos = np.array([n, e, -(gps.alt_msl - self.home.alt)])
            return pos, np.asarray(gps.vel_ned, dtype=float)
        if source == "extnav" and self._extnav_pos is not None:
            return self._extnav_pos.copy(), (self._extnav_vel.copy()
                                             if self._extnav_vel is not None else None)
        if source == "optflow" and self._flow_vel is not None:
            return None, self._flow_vel.copy()
        return None

    def _flags(self, aiding: bool, source: str, truth: PlantState) -> int:
        f = EKF_ATTITUDE
        if not self._origin_set:
            return f | EKF_UNINITIALIZED
        if truth.on_ground and not aiding:
            return f | EKF_CONST_POS_MODE
        if source == "optflow":
            # Velocity only: relative position, never absolute.  This is the
            # distinction that decides whether a flow-only survey can geotag at
            # all, and it is why FLOW_ONLY is not usable_for_survey.
            return (f | EKF_VELOCITY_HORIZ | EKF_VELOCITY_VERT
                    | EKF_POS_HORIZ_REL | EKF_POS_VERT_ABS | EKF_POS_VERT_AGL
                    | (EKF_CONST_POS_MODE if not aiding else 0))
        if not aiding:
            return (f | EKF_CONST_POS_MODE | EKF_PRED_POS_HORIZ_REL
                    | EKF_POS_VERT_ABS)
        f |= EKF_VELOCITY_HORIZ | EKF_VELOCITY_VERT | EKF_POS_VERT_ABS \
            | EKF_POS_VERT_AGL
        # Absolute position requires an origin the filter can trust.  ExtNav
        # provides it only once it has been anchored to something absolute at
        # least once - which in practice means GNSS was available at boot.
        f |= EKF_POS_HORIZ_ABS if source == "gnss" or self._origin_set \
            else EKF_POS_HORIZ_REL
        return f

    # ------------------------------------------------------------------ #
    # Pre-arm
    # ------------------------------------------------------------------ #
    def prearm_failures(self, params: Dict[str, float]) -> List[str]:
        """Reasons this vehicle would refuse to arm, in ArduPilot's phrasing.

        Kept deliberately close to the real strings so that a test written
        against one backend reads correctly against the other.
        """
        out: List[str] = []
        if not self._origin_set and self.source_set == 1:
            out.append("AHRS: waiting for home")
        if not (self.est.flags & (EKF_POS_HORIZ_ABS | EKF_PRED_POS_HORIZ_ABS
                                  | EKF_POS_HORIZ_REL)):
            out.append("Need Position Estimate")
        if not (self.est.flags & (EKF_POS_VERT_ABS | EKF_POS_VERT_AGL)):
            out.append("Need Alt Estimate")
        # Pre-arm validates every *configured* source set, not just the active
        # one, which is why VISO_TYPE/FLOW_TYPE must be set even when flying on
        # GNSS.  Mirror that here so the failure appears in both backends.
        for n in (1, 2, 3):
            posxy = int(params.get(f"EK3_SRC{n}_POSXY", 0))
            velxy = int(params.get(f"EK3_SRC{n}_VELXY", 0))
            yaw = int(params.get(f"EK3_SRC{n}_YAW", 0))
            needs_viso = self.SRC_EXTNAV in (posxy, velxy, yaw)
            needs_flow = self.SRC_OPTFLOW in (posxy, velxy)
            if needs_viso and self.viso_type == 0:
                out.append("EK3 sources require VisualOdom")
            if needs_flow and self.flow_type == 0:
                out.append("EK3 sources require OpticalFlow")
        if self.viso_type != 0 and any(
                self.SRC_EXTNAV in (int(params.get(f"EK3_SRC{n}_POSXY", 0)),
                                    int(params.get(f"EK3_SRC{n}_VELXY", 0)),
                                    int(params.get(f"EK3_SRC{n}_YAW", 0)))
                for n in (1, 2, 3)):
            if not self.viso_healthy:
                out.append("VisOdom: not healthy")
        return out


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class _SocketFile:
    """File-like wrapper so pymavlink can write straight to a socket."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.dead = False

    def write(self, data: bytes) -> None:
        if self.dead:
            return
        try:
            self.sock.sendall(data)
        except (OSError, struct.error):
            self.dead = True

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.dead = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class _Client:
    """One connected GCS: its own parser, sequence counter and rate table."""

    #: Stream group -> messages, mirroring MAV_DATA_STREAM_* semantics.
    GROUPS = {
        "RAW_SENSORS": (), "EXTENDED_STATUS": ("SYS_STATUS", "GPS_RAW_INT",
                                               "SERVO_OUTPUT_RAW"),
        "RC_CHANNELS": ("RC_CHANNELS_RAW",),
        "RAW_CONTROLLER": (), "POSITION": ("GLOBAL_POSITION_INT",),
        "EXTRA1": ("ATTITUDE",), "EXTRA2": ("VFR_HUD",),
        "EXTRA3": ("EKF_STATUS_REPORT", "BATTERY_STATUS"),
    }
    #: Default rates when the client asks for "everything" or does not ask.
    DEFAULTS = {"HEARTBEAT": 1.0, "ATTITUDE": 20.0, "GLOBAL_POSITION_INT": 10.0,
                "GPS_RAW_INT": 10.0, "EKF_STATUS_REPORT": 10.0,
                "SYS_STATUS": 2.0, "VFR_HUD": 10.0, "SERVO_OUTPUT_RAW": 10.0,
                "BATTERY_STATUS": 2.0, "RC_CHANNELS_RAW": 10.0,
                "HOME_POSITION": 1.0}

    def __init__(self, sock: socket.socket, system: int = 1,
                 component: int = 1) -> None:
        self.sock = sock
        self.file = _SocketFile(sock)
        self.mav = mavlink2.MAVLink(self.file, srcSystem=system,
                                    srcComponent=component)
        self.mav.robust_parsing = True
        # There is no ``self.mav.mav`` here.  A bare ``MAVLink`` instance is a
        # packer/parser, not a mavutil vehicle object, so it carries no
        # ``type``/``autopilot`` to set - those go on each HEARTBEAT explicitly,
        # which is what ``_publish`` does.
        self.rates: Dict[str, float] = dict(self.DEFAULTS)
        self._last_sent: Dict[str, float] = {}
        self.send_errors = 0
        self.alive = True
        self.connected_at = time.time()

    def send(self, msg: Any) -> None:
        if not self.alive or self.file.dead:
            self.alive = False
            return
        try:
            self.mav.send(msg)
        except OSError:
            # The socket really is gone; drop the client.
            self.alive = False
        except Exception as exc:
            # A malformed field is a bug in the publisher, not a dead link.
            # Dropping the client here turns one wrong type into a vehicle that
            # stops answering entirely - and, because the socket is then garbage
            # collected, the GCS sees a connection reset and blames the network.
            # Skip the message, count it, and keep the session alive.
            self.send_errors += 1
            if self.send_errors <= 5:
                log.error("MiniSITL: cannot pack %s: %r",
                          getattr(msg, "get_type", lambda: "?")(), exc)

    def due(self, name: str, t: float) -> bool:
        hz = self.rates.get(name, 0.0)
        if hz <= 0:
            return False
        last = self._last_sent.get(name, -1e9)
        if t - last >= (1.0 / hz) - 1e-4:
            self._last_sent[name] = t
            return True
        return False

    def recv(self) -> List[Any]:
        try:
            self.sock.setblocking(False)
            data = self.sock.recv(65535)
        except (BlockingIOError, InterruptedError):
            return []
        except OSError:
            self.alive = False
            return []
        if not data:
            self.alive = False
            return []
        try:
            return list(self.mav.parse_buffer(data) or [])
        except Exception as exc:
            log.debug("parse error: %r", exc)
            return []

    def close(self) -> None:
        self.alive = False
        self.file.close()


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
class MiniSITL:
    """A MAVLink-speaking vehicle model, wire-compatible with ArduPilot SITL.

    Implements the same external contract as :class:`~sar.sim.sitl_launch.ArduPilotSitl`
    (``start`` / ``connect`` / ``boot`` / ``stop`` / ``running`` / ``info``), so
    a test can pick a backend without changing anything else.

    The ground truth is deliberately *not* on the wire.  :meth:`truth` and
    :meth:`truth_position_error_m` exist for the test harness, which plays the
    part of a survey team afterwards checking where the aircraft actually was
    against where it said it was.  That gap is the number the geo-tagger's
    uncertainty model has to cover, and publishing it would make it unmeasurable.
    """

    def __init__(self,
                 home: Optional[GeoPoint] = None,
                 airframe: Optional[AirframeConfig] = None,
                 battery: Optional[BatteryConfig] = None,
                 gps_env: Optional[GpsEnvironment] = None,
                 param_file: Optional[Path] = None,
                 port: int = 5760,
                 host: str = "0.0.0.0",
                 physics_hz: float = 200.0,
                 speedup: float = 1.0,
                 seed: int = 7,
                 accel_bias_ms2: float = 0.05,
                 avionics_w: float = 28.0,
                 vio_truth_drift: bool = True) -> None:
        h = home or GeoPoint(DEFAULT_HOME[0], DEFAULT_HOME[1], DEFAULT_HOME[2])
        self.home = h
        self.port = int(port)
        self.host = host
        self.physics_hz = float(physics_hz)
        self.speedup = float(speedup)
        self.seed = int(seed)
        self.vio_truth_drift = bool(vio_truth_drift)

        pf = Path(param_file) if param_file else \
            Path(__file__).resolve().parents[2] / "configs" / "ardupilot_sitl.parm"
        self.param_file = pf
        self.params: Dict[str, float] = load_param_file(pf)
        # SITL-only conveniences that the hardware file does not carry.
        self.params.setdefault("SIM_GPS_ENABLE", 1.0)
        self.params.setdefault("FS_EKF_THRESH", 0.8)
        self.params.setdefault("RTL_ALT", 40.0)
        self.params.setdefault("LAND_SPEED", 50.0)      # cm/s
        self.params.setdefault("WPNAV_SPEED", 500.0)    # cm/s
        self.params.setdefault("WPNAV_SPEED_UP", 250.0)
        self.params.setdefault("WPNAV_SPEED_DN", 150.0)

        self.air = airframe or AirframeConfig()
        self.batt = battery or BatteryConfig()
        self.pt = Powertrain(self.air, self.batt, avionics_w=avionics_w)
        self.ap = Autopilot(self.pt, self.air)

        self.gps_env = gps_env or GpsEnvironment()
        self.gps = GpsSensor(h, self.gps_env, seed=seed + 1)
        self.baro = BaroSensor(seed=seed + 3, msl_altitude=float(h.alt))
        self.flow = OpticalFlowSensor(seed=seed + 6)
        self.ekf = EkfEmulator(self.params, h, accel_bias_ms2=accel_bias_ms2,
                               seed=seed + 11)

        # ---- runtime state ----
        self.t_sim = 0.0
        self.t0_wall = time.time()
        self._stop = threading.Event()
        self._physics: Optional[threading.Thread] = None
        self._io: Optional[threading.Thread] = None
        self._server: Optional[socket.socket] = None
        self._clients: List[_Client] = []
        self._lock = threading.RLock()
        self._texts: List[Tuple[int, str]] = []
        self._ready = False
        self._gps_denied = False
        self._mode = "STABILIZE"
        self._target_alt_rel: Optional[float] = None
        self._rtl_phase = 0
        self._yaw_cmd_rate_dps = 0.0
        self._servo: Dict[int, float] = {}
        self._drops: List[Dict[str, Any]] = []
        self._commands: List[Dict[str, Any]] = []
        self._armed_at: Optional[float] = None
        self._touchdown_s = 0.0
        self._boot_grace_s = 3.0
        self._energy_wh = 0.0
        self._distance_m = 0.0
        self._last_pos: Optional[np.ndarray] = None
        self._io_error: Optional[str] = None
        self._physics_error: Optional[str] = None
        self._last_gps: Optional[GpsSample] = None
        self._gps_fix_override: Optional[int] = None
        #: Set by :meth:`connect` so :meth:`wait_ready` can pump the banner in.
        self.conn: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Lifecycle - same contract as ArduPilotSitl
    # ------------------------------------------------------------------ #
    def start(self, ready_timeout_s: float = 30.0) -> "MiniSITL":
        """Bind the port, start the physics and I/O threads, and wait for ready."""
        if self._physics is not None:
            return self
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server.bind((self.host, self.port))
        except OSError as exc:
            self._server.close()
            self._server = None
            raise MiniSitlError(
                f"cannot bind {self.host}:{self.port}: {exc}.  Another SITL is "
                f"probably already listening; use a different --port or stop it."
            ) from exc
        self._server.listen(4)
        self._server.setblocking(False)
        self._stop.clear()
        self._physics = threading.Thread(target=self._physics_loop, daemon=True,
                                         name="minisitl-physics")
        self._io = threading.Thread(target=self._io_loop, daemon=True,
                                    name="minisitl-io")
        self._physics.start()
        self._io.start()
        log.info("MiniSITL listening on tcp %s:%d (params %s)", self.host,
                 self.port, self.param_file.name)
        end = time.time() + ready_timeout_s
        while time.time() < end and not self._ready:
            time.sleep(0.02)
        if not self._ready:
            log.warning("MiniSITL did not report ready within %.0f s", ready_timeout_s)
        return self

    def stop(self, timeout: float = 10.0) -> int:
        self._stop.set()
        for th in (self._physics, self._io):
            if th is not None:
                th.join(timeout=timeout)
        with self._lock:
            for c in self._clients:
                c.close()
            self._clients.clear()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        self._physics = self._io = None
        return 0

    @property
    def running(self) -> bool:
        return self._physics is not None and self._physics.is_alive()

    def connect(self, timeout: float = 30.0, **kwargs: Any):
        """Return a :class:`~sar.mavlink.MavConnection` to this vehicle."""
        from sar.mavlink.connection import MavConnection
        target = kwargs.pop("target", None) or f"tcp:127.0.0.1:{self.port}"
        if not self.running:
            self.start()
        conn = MavConnection(target, **kwargs)
        self.conn = conn
        return conn

    def boot(self, ready_timeout_s: float = 30.0, **conn_kwargs: Any):
        """``start`` then ``connect`` then wait for the ready banner."""
        self.start(ready_timeout_s)
        conn = self.connect(timeout=conn_kwargs.pop("timeout", 20.0), **conn_kwargs)
        self.wait_ready(ready_timeout_s)
        return conn

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """Wait for the ``MiniSITL Ready`` STATUSTEXT banner."""
        end = time.time() + timeout
        while time.time() < end:
            if self._ready:
                conn = getattr(self, "conn", None)
                if conn is not None:
                    conn.pump(0.2)
                return True
            time.sleep(0.05)
        return self._ready

    def __enter__(self) -> "MiniSITL":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def info(self) -> Dict[str, Any]:
        st = self.ap.plant.state
        return {
            "backend": "mini", "port": self.port, "params": str(self.param_file),
            "n_params": len(self.params), "physics_hz": self.physics_hz,
            "speedup": self.speedup, "t_sim_s": round(self.t_sim, 2),
            "mode": self._mode, "armed": self.ap.armed,
            "clients": len(self._clients), "ready": self._ready,
            "send_errors": sum(c.send_errors for c in self._clients),
            "agl_m": round(st.agl, 2),
            "speed_ms": round(st.speed_ms, 2),
            "battery_pct": round(self.pt.soc * 100.0, 1),
            "energy_wh": round(self._energy_wh, 1),
            "distance_m": round(self._distance_m, 1),
            "ekf_source_set": self.ekf.source_set,
            "ekf_source": self.ekf.est.source,
            "ekf_flags": self.ekf.est.flag_names(),
            "viso_healthy": self.ekf.viso_healthy,
            "extnav_samples": self.ekf.extnav_samples,
            "extnav_rejected_frames": self.ekf.extnav_rejected_frames,
            "gps_denied": self._gps_denied,
            "io_error": self._io_error, "physics_error": self._physics_error,
            "payload_drops": len(self._drops),
        }

    # ------------------------------------------------------------------ #
    # Test hooks - not available over the wire, by design
    # ------------------------------------------------------------------ #
    def truth(self) -> PlantState:
        """The actual vehicle state.  Never published; see the class docstring."""
        return self.ap.plant.state

    def truth_position_error_m(self) -> float:
        """How far the published estimate is from where the aircraft really is.

        This is the number the geo-tagger's sigma model has to bound, and the
        only way to know whether it does.
        """
        st = self.ap.plant.state
        return float(np.linalg.norm(self.ekf.est.pos - st.pos))

    def deny_gps(self, denied: bool = True) -> None:
        """Drop GNSS, as ``SIM_GPS_ENABLE=0`` would."""
        self._gps_denied = bool(denied)
        self.params["SIM_GPS_ENABLE"] = 0.0 if denied else 1.0
        self._say(4 if denied else 6,
                  "GPS denied (simulated)" if denied else "GPS restored (simulated)")

    def set_wind(self, north_ms: float = 0.0, east_ms: float = 0.0,
                 down_ms: float = 0.0) -> None:
        """Apply a steady wind to the plant.  The estimate does not see it, so
        the controller has to fight it - which is the point."""
        self.ap.plant.wind = np.array([float(north_ms), float(east_ms),
                                       float(down_ms)])

    def inject_fault(self, kind: str, **kw: Any) -> None:
        """Inject a named fault for resilience testing.

        ``motor`` desaturates one rotor, ``imu_bias`` adds an accelerometer bias
        (which is what makes dead reckoning diverge), ``baro`` offsets the
        altitude, ``extnav_jump`` steps the external-nav estimate.
        """
        if kind == "imu_bias":
            self.ekf.accel_bias = float(kw.get("ms2", 0.25))
        elif kind == "baro":
            # BaroSensor has no public bias; its MSL reference is what shifts the
            # reported altitude, which is the effect a baro fault has anyway.
            self.baro.msl = float(self.home.alt) + float(kw.get("m", 5.0))
        elif kind == "extnav_jump":
            d = float(kw.get("m", 3.0))
            if self.ekf._extnav_pos is not None:
                self.ekf._extnav_pos[0] += d
        elif kind == "motor":
            self._say(2, f"motor fault injected: {kw}")
        else:
            raise ValueError(f"unknown fault {kind!r}")

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def _physics_loop(self) -> None:
        dt = 1.0 / self.physics_hz
        next_step = time.monotonic()
        while not self._stop.is_set():
          try:
            st = self.ap.update(dt)
            self.t_sim += dt * self.speedup

            # Energy is integrated from the plant's own power model, so the
            # endurance a sortie reports is a consequence of how it was flown
            # rather than a number asserted about it.
            ps = self.pt.draw(self.ap.plant.power_draw_w(), dt, t=self.t_sim)
            self._energy_wh += float(ps.power_w) * dt / 3600.0
            if self._last_pos is not None:
                self._distance_m += float(np.linalg.norm(st.pos - self._last_pos))
            self._last_pos = st.pos.copy()

            # sensors + EKF
            t_ms = int(self.t_sim * 1000)
            gps = self._gps_measure(st)
            self._last_gps = gps
            self.ekf.ingest_baro(self.baro.measure(st, self.t_sim).altitude_m)
            self.ekf.update(t_ms, dt, st, gps)
            self._run_modes(dt)

            # internal flow/vio aiding for source sets 2 and 3
            if self.ekf.flow_type and self.ekf.source_set == 3:
                fs = self.flow.measure(st, self.t_sim)
                self.ekf.ingest_optical_flow(t_ms, fs, max(0.05, st.agl))

            # Touchdown -> disarm, as ArduPilot does at the end of LAND.  Judged
            # on height and sink rate rather than on the plant's ``on_ground``
            # flag, because a position controller settles a few centimetres above
            # the pad instead of exactly on it - genuinely airborne by the flag's
            # definition, landed by any operator's.
            if self.ap.armed and self._mode in ("LAND", "RTL") \
                    and st.agl < 0.35 and abs(st.vel[2]) < 0.6:
                self._touchdown_s += dt
                if self._touchdown_s > 1.0:
                    self._set_armed(False, "landed, auto-disarm")
            else:
                self._touchdown_s = 0.0

            if not self._ready and self.t_sim > self._boot_grace_s \
                    and self.ekf.est.origin_set:
                self._ready = True
                self._say(6, "MiniSITL Ready")
                self._say(6, f"EKF3 IMU0 origin set; Field Elevation Set: "
                             f"{int(self.home.alt)}m")

            next_step += dt / max(self.speedup, 1e-6)
            slack = next_step - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_step = time.monotonic()
          except Exception as exc:
            log.exception("MiniSITL physics error: %r", exc)
            self._physics_error = f"{type(exc).__name__}: {exc}"
            time.sleep(dt)

    def _gps_measure(self, st: PlantState) -> GpsSample:
        if self._gps_denied or int(self.params.get("SIM_GPS_ENABLE", 1)) == 0:
            return GpsSample(valid=False, fix_type=0, num_sats=0,
                             lat=0.0, lon=0.0, alt_msl=0.0,
                             vel_ned=np.zeros(3), hdop=99.0, vdop=99.0,
                             h_accuracy_mm=0, v_accuracy_mm=0, jammed=True)
        s = self.gps.measure(st, self.t_sim)
        if self._gps_fix_override is not None:
            s.fix_type = self._gps_fix_override
        return s

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def _io_loop(self) -> None:
        while not self._stop.is_set():
            self._accept()
            with self._lock:
                clients = [c for c in self._clients if c.alive]
                self._clients = clients
            for c in clients:
                try:
                    for msg in c.recv():
                        self._handle(c, msg)
                    self._publish(c)
                except Exception as exc:
                    # Never let one bad message or a malformed field take the
                    # I/O thread down: the failure mode is a vehicle that stops
                    # answering with no explanation, which is indistinguishable
                    # from a crashed simulator.
                    log.exception("MiniSITL io error: %r", exc)
                    self._io_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.005)

    def _accept(self) -> None:
        if self._server is None:
            return
        try:
            sock, addr = self._server.accept()
        except (BlockingIOError, OSError):
            return
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        c = _Client(sock)
        with self._lock:
            self._clients.append(c)
        log.info("MiniSITL: GCS connected from %s", addr)

    def _say(self, severity: int, text: str) -> None:
        self._texts.append((severity, text))
        if len(self._texts) > 400:
            self._texts.pop(0)
        # ``text`` is a bytes field: pymavlink's constructor runs
        # ``text.split(b"\x00")`` on it, so passing a str raises
        # "must be str or None, not bytes" from inside the codec.
        msg = mavlink2.MAVLink_statustext_message(
            severity=severity, text=text[:50].encode("ascii", "replace"))
        with self._lock:
            for c in self._clients:
                c.send(msg)

    def _publish(self, c: _Client) -> None:
        t = time.time()
        t_ms = int(self.t_sim * 1000)
        st = self.ap.plant.state
        est = self.ekf.est
        gps = self._last_gps or self._gps_measure(st)
        pt = self.pt

        if c.due("HEARTBEAT", t):
            base = 0
            if self.ap.armed:
                base |= mavlink2.MAV_MODE_FLAG_SAFETY_ARMED
            base |= mavlink2.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            if st.on_ground and not self.ap.armed:
                base |= mavlink2.MAV_MODE_FLAG_STABILIZE_ENABLED
            c.send(mavlink2.MAVLink_heartbeat_message(
                type=mavlink2.MAV_TYPE_QUADROTOR,
                autopilot=mavlink2.MAV_AUTOPILOT_ARDUPILOTMEGA,
                base_mode=base,
                custom_mode=MODE_NUMBER.get(self._mode, 0),
                system_status=(mavlink2.MAV_STATE_ACTIVE if self.ap.armed
                               else mavlink2.MAV_STATE_STANDBY),
                mavlink_version=3))

        if c.due("GLOBAL_POSITION_INT", t) and est.origin_set:
            c.send(mavlink2.MAVLink_global_position_int_message(
                time_boot_ms=t_ms,
                lat=int(round(est.lat * 1e7)), lon=int(round(est.lon * 1e7)),
                alt=int(round(est.alt_msl * 1000)),
                relative_alt=int(round((est.alt_msl - self.home.alt) * 1000)),
                vx=int(round(est.vel[0] * 100)), vy=int(round(est.vel[1] * 100)),
                vz=int(round(est.vel[2] * 100)),
                hdg=int(round(math.degrees(st.yaw) % 360 * 100))))

        if c.due("GPS_RAW_INT", t):
            c.send(mavlink2.MAVLink_gps_raw_int_message(
                time_usec=int(self.t_sim * 1e6),
                fix_type=int(gps.fix_type),
                lat=int(round(gps.lat * 1e7)), lon=int(round(gps.lon * 1e7)),
                alt=int(round(gps.alt_msl * 1000)),
                eph=int(round(gps.hdop * 100)), epv=int(round(gps.vdop * 100)),
                vel=int(round(math.hypot(*gps.vel_ned[:2]) * 100)),
                cog=int(round(math.degrees(math.atan2(gps.vel_ned[1], gps.vel_ned[0]))
                              % 360 * 100)),
                satellites_visible=int(gps.num_sats),
                h_acc=int(gps.h_accuracy_mm), v_acc=int(gps.v_accuracy_mm),
                vel_acc=int(gps.v_speed_accuracy_mm), hdg_acc=0, yaw=0))

        if c.due("ATTITUDE", t):
            c.send(mavlink2.MAVLink_attitude_message(
                time_boot_ms=t_ms, roll=float(st.roll), pitch=float(st.pitch),
                yaw=float(st.yaw), rollspeed=float(st.omega[0]),
                pitchspeed=float(st.omega[1]), yawspeed=float(st.omega[2])))

        if c.due("EKF_STATUS_REPORT", t):
            # Published as dimensionless ratios against the same nominal sigmas
            # the GCS uses to give them units, so the two ends agree by
            # construction rather than by coincidence.  Ratio 1.0 means "as good
            # as open-sky GNSS".
            c.send(mavlink2.MAVLink_ekf_status_report_message(
                flags=int(est.flags),
                velocity_variance=(est.vel_sigma_ms / NOMINAL_VEL_SIGMA_MS) ** 2,
                pos_horiz_variance=(est.pos_sigma_m / NOMINAL_GNSS_SIGMA_M) ** 2,
                pos_vert_variance=(1.5 / NOMINAL_ALT_SIGMA_M) ** 2,
                compass_variance=1.0,
                terrain_alt_variance=1.0,
                airspeed_variance=0.0))

        if c.due("SYS_STATUS", t):
            present = (mavlink2.MAV_SYS_STATUS_SENSOR_3D_ACCEL
                       | mavlink2.MAV_SYS_STATUS_SENSOR_3D_GYRO
                       | mavlink2.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE
                       | mavlink2.MAV_SYS_STATUS_SENSOR_GPS
                       | mavlink2.MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL
                       | mavlink2.MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS)
            health = (mavlink2.MAV_SYS_STATUS_SENSOR_3D_ACCEL
                      | mavlink2.MAV_SYS_STATUS_SENSOR_3D_GYRO
                      | mavlink2.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE
                      | mavlink2.MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS)
            if gps.valid and gps.fix_type >= 3:
                health |= mavlink2.MAV_SYS_STATUS_SENSOR_GPS
            if est.flags & (EKF_POS_HORIZ_ABS | EKF_POS_HORIZ_REL):
                health |= mavlink2.MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL
            c.send(mavlink2.MAVLink_sys_status_message(
                onboard_control_sensors_present=present,
                onboard_control_sensors_enabled=present,
                onboard_control_sensors_health=health,
                load=int(min(1000, st.thrust_n / max(1e-6, self.pt.weight_n) * 500)),
                voltage_battery=int(round(pt.voltage * 1000)),
                current_battery=int(round(pt.current_a * 100)),
                battery_remaining=int(round(pt.soc * 100)),
                drop_rate_comm=0, errors_comm=0, errors_count1=0,
                errors_count2=0, errors_count3=0, errors_count4=0))

        if c.due("VFR_HUD", t):
            c.send(mavlink2.MAVLink_vfr_hud_message(
                airspeed=st.speed_ms, groundspeed=st.speed_ms,
                # ``heading`` is a uint16 in whole degrees, not a float; a
                # float here raises struct.error from inside the packer.
                heading=int(round(math.degrees(st.yaw) % 360.0)),
                throttle=int(round(float(np.mean(st.rotor)) * 100)),
                alt=est.alt_msl - self.home.alt,
                climb=-float(st.vel[2])))

        if c.due("SERVO_OUTPUT_RAW", t):
            us = [1000 + int(round(float(x) * 1000)) for x in st.rotor[:4]]
            while len(us) < 4:
                us.append(1000)
            c.send(mavlink2.MAVLink_servo_output_raw_message(
                time_usec=int(self.t_sim * 1e6), port=0,
                servo1_raw=us[0], servo2_raw=us[1], servo3_raw=us[2],
                servo4_raw=us[3],
                servo5_raw=1500, servo6_raw=1500, servo7_raw=1500,
                servo8_raw=1500,
                servo9_raw=int(round(self._servo.get(9, 1100))),
                servo10_raw=1500, servo11_raw=1500, servo12_raw=1500,
                servo13_raw=1500, servo14_raw=1500, servo15_raw=1500,
                servo16_raw=1500))

        if c.due("BATTERY_STATUS", t):
            c.send(mavlink2.MAVLink_battery_status_message(
                id=0, battery_function=0,
                type=mavlink2.MAV_BATTERY_TYPE_LIPO,
                temperature=int(round(pt.temp_c * 100)),
                # ArduPilot's layout, which is not what the MAVLink spec text
                # suggests: ``voltages[0]`` is the *pack* voltage in mV and
                # ``voltages[1]`` is the cell count, rather than one entry per
                # cell.  A GCS that divides voltages[0] by voltages[1] to get a
                # cell voltage is written against ArduPilot's convention, so
                # publishing per-cell values there makes a healthy 6S pack read
                # as 4.2 V.
                voltages=[int(round(pt.terminal_voltage() * 1000)),
                          int(self.batt.cells)] + [0] * 9,
                current_battery=int(round(pt.current_a * 100)),
                current_consumed=int(round(pt.consumed_ah * 100)),
                energy_consumed=int(round(self._energy_wh * 10)),
                battery_remaining=int(round(pt.soc * 100)),
                time_remaining=int(round(pt.time_remaining_s())),
                charge_state=mavlink2.MAV_BATTERY_CHARGE_STATE_OK,
                voltages_ext=[0] * 4, mode=0, fault_bitmask=0))

    # ------------------------------------------------------------------ #
    # Command handling
    # ------------------------------------------------------------------ #
    def _handle(self, c: _Client, msg: Any) -> None:
        t = msg.get_type()
        if t == "HEARTBEAT":
            return
        if t == "COMMAND_LONG":
            self._command_long(c, msg)
        elif t == "SET_MODE":
            self._set_mode_name(self._mode_from_number(msg.custom_mode))
            c.send(mavlink2.MAVLink_command_ack_message(
                command=CMD_DO_SET_MODE, result=mavlink2.MAV_RESULT_ACCEPTED))
        elif t == "SET_POSITION_TARGET_GLOBAL_INT":
            self._set_target_global(msg)
        elif t == "SET_POSITION_TARGET_LOCAL_NED":
            self._set_target_local(msg)
        elif t == "PARAM_REQUEST_READ":
            name = self._param_name(msg.param_id)
            if name is not None:
                self._send_param(c, name)
        elif t == "PARAM_REQUEST_LIST":
            for name in list(self.params):
                self._send_param(c, name)
        elif t == "PARAM_SET":
            name = self._param_name(msg.param_id)
            if name is not None:
                self._apply_param(name, float(msg.param_value))
                self._send_param(c, name)
        elif t == "REQUEST_DATA_STREAM":
            self._apply_stream(c, msg)
        elif t == "ODOMETRY":
            # Note ``(msg.x, msg.y, msg.z)`` and not ``msg.x and (...)``: at the
            # origin x is legitimately 0.0, which is falsy, and the short-circuit
            # would pass a scalar where a 3-vector is expected.
            self.ekf.ingest_odometry(
                int(msg.time_usec // 1000), int(msg.frame_id),
                int(msg.child_frame_id), (msg.x, msg.y, msg.z),
                msg.q, (msg.vx, msg.vy, msg.vz), msg.pose_covariance,
                int(getattr(msg, "quality", 0)) or 100)
        elif t == "VISION_POSITION_ESTIMATE":
            self.ekf.ingest_vision_position(int(msg.usec // 1000),
                                            (msg.x, msg.y, msg.z),
                                            (msg.roll, msg.pitch, msg.yaw),
                                            msg.covariance)
        elif t == "VISION_SPEED_ESTIMATE":
            self.ekf.ingest_vision_speed(int(msg.usec // 1000),
                                         (msg.x, msg.y, msg.z))
        elif t == "MANUAL_CONTROL":
            pass
        else:
            log.debug("MiniSITL: unhandled %s", t)

    @staticmethod
    def _param_name(raw: Any) -> Optional[str]:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        name = str(raw).strip("\x00 ").upper()[:16]
        return name or None

    def _send_param(self, c: _Client, name: str) -> None:
        keys = sorted(self.params)
        idx = keys.index(name) if name in self.params else -1
        c.send(mavlink2.MAVLink_param_value_message(
            param_id=name[:16].encode("utf-8"),
            param_value=float(self.params.get(name, 0.0)),
            param_type=mavlink2.MAV_PARAM_TYPE_REAL32,
            param_count=len(keys), param_index=idx))

    def _apply_param(self, name: str, value: float) -> None:
        self.params[name] = float(value)
        if name == "SIM_GPS_ENABLE":
            self._gps_denied = (int(value) == 0)
        elif name == "SIM_GPS_NUMSATS" and hasattr(self.gps.env, "base_satellites"):
            self.gps.env.base_satellites = max(0, int(value))
        elif name == "VISO_TYPE":
            self.ekf.p = self.params
        elif name == "EK3_SRC_SET":
            self.ekf.set_source_set(int(value))

    def _apply_stream(self, c: _Client, msg: Any) -> None:
        rate = float(getattr(msg, "req_message_rate", 0.0))
        start = int(getattr(msg, "start_stop", 1))
        sid = int(getattr(msg, "req_stream_id", 0))
        names: Sequence[str] = ()
        for gname, msgs in _Client.GROUPS.items():
            if getattr(mavlink2, f"MAV_DATA_STREAM_{gname}", -1) == sid:
                names = msgs
                break
        with self._lock:
            if sid == 0:                                   # ALL
                for k in list(c.rates):
                    c.rates[k] = rate if start else 0.0
            for m in names:
                c.rates[m] = rate if start else 0.0
        # No ACK.  REQUEST_DATA_STREAM is a message, not a MAV_CMD, and
        # ArduPilot does not acknowledge it; sending a COMMAND_ACK with a
        # fabricated id makes a client that waits for one behave differently
        # against the two backends.  ``MAV_CMD_SET_MESSAGE_INTERVAL`` (511) is
        # the command form, and that one *is* acknowledged - handled in
        # ``_command_long``.

    @staticmethod
    def _mode_from_number(n: int) -> str:
        return MODE_NAME.get(int(n), "STABILIZE")

    def _set_mode_name(self, name: str) -> bool:
        if name not in MODE_NUMBER:
            return False
        if name != self._mode:
            self._say(6, f"Mode {name}")
        self._mode = name
        self.ap.mode = name
        if name == "LOITER" or name == "POSHOLD":
            self.ap.loiter()
        elif name == "GUIDED":
            self.ap.loiter()
        elif name == "RTL":
            self._rtl_phase = 0
        elif name == "LAND":
            self.ap.target_pos = None
            self.ap.target_vel = np.array([0.0, 0.0, 0.5])
        return True

    def _set_armed(self, armed: bool, why: str = "") -> bool:
        if armed and not self.ap.armed:
            fails = self.ekf.prearm_failures(self.params)
            if fails:
                text = "; ".join(f"PreArm: {f}" for f in fails)
                self._say(0, text[:200])
                if self._clients:
                    self._clients[0].send(mavlink2.MAVLink_command_ack_message(
                        command=CMD_COMPONENT_ARM_DISARM,
                        result=mavlink2.MAV_RESULT_FAILED))
                log.info("MiniSITL refused arm: %s", text)
                return False
            self.ap.arm()
            self._armed_at = self.t_sim
            self._say(6, "Arming MOTORS" + (f" ({why})" if why else ""))
        elif not armed and self.ap.armed:
            self.ap.disarm()
            self._say(6, "Disarming MOTORS" + (f" ({why})" if why else ""))
        return True

    def _set_target_global(self, msg: Any) -> None:
        mask = int(msg.type_mask)
        lat = msg.lat_int / 1e7
        lon = msg.lon_int / 1e7
        n, e, _d = wgs84_to_local(GeoPoint(lat, lon), self.home)
        down = -(float(msg.alt) - self.home.alt)
        if mask & mavlink2.POSITION_TARGET_TYPEMASK_X_IGNORE:
            n = float(self.ekf.est.pos[0])
        if mask & mavlink2.POSITION_TARGET_TYPEMASK_Y_IGNORE:
            e = float(self.ekf.est.pos[1])
        if mask & mavlink2.POSITION_TARGET_TYPEMASK_Z_IGNORE:
            down = float(self.ekf.est.pos[2])
        vel = np.zeros(3)
        use_vel = bool(mask & (mavlink2.POSITION_TARGET_TYPEMASK_X_IGNORE
                               | mavlink2.POSITION_TARGET_TYPEMASK_Y_IGNORE))
        if use_vel:
            # Velocity streaming: the survey flies this way rather than stepping
            # between waypoints, because every stop costs energy and every
            # acceleration smears the thermal image.
            vn = float(msg.vx)
            ve = float(msg.vy)
            vd = float(msg.vz)
            if not (mask & mavlink2.POSITION_TARGET_TYPEMASK_Z_IGNORE):
                down = -(float(msg.alt) - self.home.alt)
                vd = 0.0
            self.ap.set_velocity_ned(vn, ve, vd, yaw=self._yaw_from(msg, vn, ve))
            self.ap.target_pos = None
            self._target_alt_rel = -down
        else:
            self.ap.goto_ned(n, e, down,
                             yaw=self._yaw_from(msg, n - self.ekf.est.pos[0],
                                                e - self.ekf.est.pos[1]))
            self._target_alt_rel = -down
        if self._mode != "GUIDED":
            self._set_mode_name("GUIDED")

    def _set_target_local(self, msg: Any) -> None:
        mask = int(msg.type_mask)
        if mask & (mavlink2.POSITION_TARGET_TYPEMASK_X_IGNORE
                   | mavlink2.POSITION_TARGET_TYPEMASK_Y_IGNORE):
            self.ap.set_velocity_ned(
                float(msg.vx), float(msg.vy), float(msg.vz),
                yaw=self._yaw_from(msg, msg.vx, msg.vy))
            self.ap.target_pos = None
        else:
            self.ap.goto_ned(float(msg.x), float(msg.y), float(msg.z),
                             yaw=self._yaw_from(msg, 0.0, 0.0))
        if self._mode != "GUIDED":
            self._set_mode_name("GUIDED")

    @staticmethod
    def _yaw_from(msg: Any, vn: float, ve: float) -> Optional[float]:
        """Resolve the heading a guided command implies, honouring YAW_IGNORE.

        ``POSITION_TARGET_TYPEMASK_YAW_IGNORE`` is how a client says "translate,
        do not turn".  Honouring it is not pedantry.  A survey pattern flies
        lanes that reverse direction every pass, and if each velocity message
        re-commands the heading toward the direction of travel, every lane
        boundary becomes a 180 deg yaw slew performed at full speed.  The
        body-frame velocity error swings through twice the airspeed while the
        airframe is still turning, the attitude controller saturates against its
        tilt limit, and the aircraft overshoots to two or three times the
        commanded speed and does not recover inside the lane.

        A real SAR aircraft holds heading along the lane and translates; it yaws
        only in the turn.  So the default here is to obey the mask, and to face
        the direction of travel only when the client has explicitly asked for
        that behaviour by leaving the bit clear.
        """
        mask = int(getattr(msg, "type_mask", 0))
        if mask & mavlink2.POSITION_TARGET_TYPEMASK_YAW_IGNORE:
            # An explicit yaw value in the message still wins over "ignore":
            # ArduPilot treats a present yaw field as a heading command.
            y = float(getattr(msg, "yaw", 0.0) or 0.0)
            return y if abs(y) > 1e-6 else None
        return MiniSITL._heading_for(vn, ve)

    @staticmethod
    def _heading_for(vn: float, ve: float) -> Optional[float]:
        """Face the direction of travel, but only when actually moving.

        Without the speed gate a hover setpoint of (0, 0) computes
        ``atan2(0, 0) = 0`` and spins the aircraft to north every control cycle,
        which reads as an uncommanded yaw excursion.
        """
        if math.hypot(vn, ve) < 0.3:
            return None
        return float(math.atan2(ve, vn))

    def _command_long(self, c: _Client, msg: Any) -> None:
        cmd = int(msg.command)
        result = mavlink2.MAV_RESULT_ACCEPTED
        self._commands.append({"t": round(self.t_sim, 2), "cmd": cmd,
                               "p": [msg.param1, msg.param2, msg.param3,
                                     msg.param4, msg.param5, msg.param6,
                                     msg.param7]})

        if cmd == CMD_COMPONENT_ARM_DISARM:
            want = float(msg.param1) == 1.0
            ok = self._set_armed(want)
            result = mavlink2.MAV_RESULT_ACCEPTED if ok else mavlink2.MAV_RESULT_FAILED

        elif cmd == CMD_NAV_TAKEOFF:
            if not self.ap.armed:
                result = mavlink2.MAV_RESULT_FAILED
            else:
                alt = float(msg.param7)
                self._set_mode_name("GUIDED")
                self.ap.goto_ned(float(self.ekf.est.pos[0]),
                                 float(self.ekf.est.pos[1]), -alt)
                self._target_alt_rel = alt

        elif cmd == CMD_NAV_LAND:
            self._set_mode_name("LAND")
            self.ap.target_pos = None
            self.ap.target_vel = np.array([0.0, 0.0,
                                           float(self.params.get("LAND_SPEED", 50.0))
                                           / 100.0])

        elif cmd == CMD_NAV_RETURN_TO_LAUNCH:
            self._set_mode_name("RTL")

        elif cmd == CMD_SET_EKF_SOURCE_SET:
            ok, why = self.ekf.set_source_set(int(msg.param1))
            result = mavlink2.MAV_RESULT_ACCEPTED if ok else mavlink2.MAV_RESULT_FAILED
            self._say(6 if ok else 2, f"EK3 source set: {why}")

        elif cmd == MAV_CMD.CONDITION_YAW:
            # param1 heading in degrees, param2 slew rate (0 = as fast as the
            # airframe allows), param3 direction (-1 CCW / 1 CW), param4 whether
            # the angle is relative to the current heading.
            #
            # A survey needs this: lanes reverse direction every pass, and the
            # turn has to happen once, stationary, at the end of a lane.  Letting
            # each streamed velocity message imply the heading instead puts a
            # 180 deg slew in the middle of the lane at full speed, which is what
            # saturates the tilt limit and overshoots the commanded airspeed.
            angle = math.radians(float(msg.param1))
            if float(msg.param4) == 1.0:
                angle = float(self.ap.target_yaw) + angle
            # param2 is the slew rate the client would like.  The airframe's own
            # yaw rate limit is what actually applies, so it is recorded for the
            # log rather than enforced as a separate constraint - a limit the
            # plant cannot meet anyway would only make the turn take longer.
            self.ap.target_yaw = float(wrap_pi(angle))
            self._yaw_cmd_rate_dps = float(msg.param2)

        elif cmd == CMD_DO_SET_SERVO:
            chan = int(msg.param1)
            pwm = float(msg.param2)
            self._servo[chan] = pwm
            if pwm >= float(self.params.get("GRIP_RELEASE", 1900)):
                self._drops.append({"t": round(self.t_sim, 2),
                                    "n": round(float(self.ekf.est.pos[0]), 2),
                                    "e": round(float(self.ekf.est.pos[1]), 2),
                                    "alt": round(float(self.ekf.est.alt_msl
                                                       - self.home.alt), 2),
                                    "channel": chan})
                self._say(6, f"Payload released on servo {chan} at "
                             f"{self._drops[-1]['alt']:.1f} m AGL")

        elif cmd == CMD_DO_GUIDED_LIMITS:
            # ArduPilot refuses this while disarmed with UNSUPPORTED and no
            # STATUSTEXT; accept it here either way but keep the difference
            # visible in the log so the two backends can be compared.
            self._say(7, f"guided limits: timeout={msg.param1:.0f}s "
                         f"alt={msg.param3:.1f}-{msg.param4:.1f}m "
                         f"horiz={msg.param5:.0f}m")

        elif cmd == CMD_DO_SET_MODE:
            # COMMAND_LONG has no ``custom_mode`` field: for MAV_CMD_DO_SET_MODE
            # the base mode is param1, the custom (flight) mode is param2 and a
            # sub-mode is param3.  Reaching for ``msg.custom_mode`` raises
            # AttributeError inside the handler, which kills the command before
            # it can be ACKed - so the GCS waits out its timeout and reports a
            # refusal for a mode change the vehicle never even saw.
            self._set_mode_name(self._mode_from_number(int(msg.param2)))

        elif cmd == 511:                      # MAV_CMD_SET_MESSAGE_INTERVAL
            name = _MESSAGE_FOR_ID.get(int(msg.param1))
            interval_us = float(msg.param2)
            if name is None:
                result = mavlink2.MAV_RESULT_TEMPORARILY_REJECTED
            elif interval_us < 0:
                c.rates[name] = _Client.DEFAULTS.get(name, 0.0)
            else:
                c.rates[name] = 0.0 if interval_us == 0 else 1e6 / interval_us

        elif cmd == CMD_PREFLIGHT_REBOOT:
            result = mavlink2.MAV_RESULT_TEMPORARILY_REJECTED

        else:
            result = mavlink2.MAV_RESULT_UNSUPPORTED

        c.send(mavlink2.MAVLink_command_ack_message(command=cmd, result=result))

    # ------------------------------------------------------------------ #
    # Mode progression (RTL / LAND)
    # ------------------------------------------------------------------ #
    def _run_modes(self, dt: float) -> None:
        """Advance RTL and LAND, which are stateful rather than setpoint-driven."""
        if self._mode == "RTL" and self.ap.armed:
            # RTL_ALT is stored in CENTIMETRES, like most ArduPilot speed and
            # altitude parameters (WPNAV_SPEED, LAND_SPEED, PILOT_SPEED_UP).
            # Reading 2500 as metres sends the aircraft to 2.5 km, where it
            # climbs for the entire test at a perfectly normal rate with a
            # perfectly normal controller - the only symptom is that RTL never
            # completes.
            rtl_alt = float(self.params.get("RTL_ALT", 1500.0)) / 100.0
            st = self.ap.plant.state
            if self._rtl_phase == 0:                       # climb
                self.ap.goto_ned(st.pos[0], st.pos[1], -rtl_alt)
                if -st.pos[2] >= rtl_alt - 1.5:
                    self._rtl_phase = 1
                    self._say(6, "RTL: reached return altitude")
            elif self._rtl_phase == 1:                     # fly home
                self.ap.goto_ned(0.0, 0.0, -rtl_alt)
                if math.hypot(st.pos[0], st.pos[1]) < 2.0:
                    self._rtl_phase = 2
                    self._say(6, "RTL: above home, descending")
            elif self._rtl_phase == 2:
                self.ap.goto_ned(0.0, 0.0, 0.0)
                if st.agl < 0.3:
                    self._set_mode_name("LAND")
