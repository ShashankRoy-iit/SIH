"""MAVLink protocol constants, dialect handling and message helpers.

Everything in :mod:`sar.mavlink` imports its IDs from here rather than
scattering magic numbers through the flight logic.  The reason is practical:
MAVLink command and frame IDs are the sort of constant that is easy to
misremember and expensive to debug, because the vehicle silently ignores a
command it does not recognise rather than returning an error for a wrong
number.

Dialect
-------
``ardupilotmega`` in the MAVLink 2 (``v20``) namespace.  ArduPilot-specific
messages - ``EKF_STATUS_REPORT``, ``BATTERY_STATUS``, ``RADIO``, ``GPS_RAW_INT``
extensions - only decode with this dialect; the generic ``common`` one silently
drops them.  PX4 works with ``common`` and ignores the extras, so the same
connection class serves both stacks.

The hardware target is a TBS Lucid H743 Wing running ArduPilot Copter, with the
onboard computer attached to SERIAL4/7 as MAVLink 2, which is why the v20
namespace is the default and not an option.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from pymavlink.dialects.v20 import ardupilotmega as mavlink2  # noqa: F401

__all__ = [
    "mavlink2", "MAV_CMD", "MAV_FRAME", "MAV_MODE", "MAV_STATE", "MAV_TYPE",
    "MAV_RESULT", "MAV_COMPONENT", "TYPE_MASK", "SOURCE_SET", "GPS_FIX",
    "SourceXY", "SourceZ", "SourceYaw",
    "result_name", "mask", "clamp_i16", "wrap_360",
]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class MAV_CMD:
    """Command IDs used by ``COMMAND_LONG`` / ``COMMAND_INT``.

    Kept as a class of ints rather than an ``IntEnum`` because MAVLink IDs are
    an open set - a firmware update can add one, and an enum would refuse to
    represent it.
    """

    NAV_WAYPOINT = 16
    NAV_LOITER_UNLIM = 17
    NAV_RETURN_TO_LAUNCH = 20
    NAV_LAND = 21
    NAV_TAKEOFF = 22
    NAV_LOITER_TURNS = 23
    NAV_CONTINUE_AND_CHANGE_ALT = 30
    NAV_SET_YAW_SPEED = 45
    NAV_PAYLOAD_PREPARE_DEPLOY = 30001
    NAV_PAYLOAD_DEPLOY = 30002

    CONDITION_DELAY = 112
    CONDITION_CHANGE_ALT = 113
    CONDITION_DISTANCE = 114
    CONDITION_YAW = 115

    DO_SET_MODE = 176
    DO_JUMP = 177
    DO_CHANGE_SPEED = 178
    DO_SET_HOME = 179
    DO_SET_RELAY = 181
    DO_REPEAT_RELAY = 182
    DO_SET_SERVO = 183
    DO_LAND_START = 189
    DO_REPEAT_SERVO = 190
    DO_CONTROL_VIDEO = 200
    DO_SET_ROI = 201
    DO_DIGICAM_CONTROL = 203
    DO_MOUNT_CONTROL = 205
    DO_GRIPPER = 3001
    DO_GUIDED_LIMITS = 3002
    DO_ENGINE_CONTROL = 3003

    COMPONENT_ARM_DISARM = 400
    PREFLIGHT_CALIBRATION = 241
    PREFLIGHT_SET_SENSOR_OFFSETS = 242
    PREFLIGHT_UAVCAN = 243
    PREFLIGHT_REBOOT_SHUTDOWN = 246
    REQUEST_DATA_STREAM = 66
    MESSAGE_INTERVAL = 511
    SET_EKF_SOURCE_SET = 42007
    #: ArduPilot's guided-mode "go straight here, ignore terrain" command.
    DO_REPOSITION = 192
    #: Set a position target in GUIDED mode without a mission.
    GUIDED_ENABLE = 4000
    TERMINATE_FLIGHT = 4001


class MAV_FRAME:
    GLOBAL = 0                    # lat/lon, absolute (WGS84) altitude
    LOCAL_NED = 1
    MISSION = 2
    GLOBAL_REL_ALT = 3            # lat/lon, altitude relative to home
    LOCAL_ENU = 4
    GLOBAL_INT = 5
    GLOBAL_REL_ALT_INT = 6
    LOCAL_OFFSET_NED = 7          # relative to current position
    BODY_NED = 8
    LOCAL_TANGENT_ENU = 9
    GLOBAL_TERRAIN_ALT = 10
    GLOBAL_TERRAIN_ALT_INT = 11
    BODY_OFFSET_NED = 12
    LOCAL_FLU = 13


class MAV_MODE:
    """``base_mode`` bitfield."""

    CUSTOM_MODE_ENABLED = 1
    TEST_ENABLED = 2
    AUTO_ENABLED = 4
    GUIDED_ENABLED = 8
    STABILIZE_ENABLED = 16
    HIL_ENABLED = 32
    MANUAL_INPUT_ENABLED = 64
    SAFETY_ARMED = 128


class MAV_STATE:
    UNINIT = 0
    BOOT = 1
    CALIBRATING = 2
    STANDBY = 3
    ACTIVE = 4
    CRITICAL = 5
    EMERGENCY = 6
    POWEROFF = 7
    FLIGHT_TERMINATION = 8


class MAV_TYPE:
    GENERIC = 0
    FIXED_WING = 1
    QUADROTOR = 2
    COAXIAL = 3
    HELICOPTER = 4
    ANTENNA_TRACKER = 5
    GCS = 6
    AIRSHIP = 7
    FIXED_WING_VTAIL = 19
    VTOL_DUOROTOR = 20
    VTOL_QUADROTOR = 21


class MAV_RESULT:
    """``MAV_RESULT`` from a command acknowledgement."""

    ACCEPTED = 0
    TEMPORARILY_REJECTED = 1
    DENIED = 2
    UNSUPPORTED = 3
    FAILED = 4
    IN_PROGRESS = 5
    CANCELLED = 6


class MAV_COMPONENT:
    AUTOPILOT1 = 1
    COMPANION_COMPUTER = 191
    MISSIONPLANNER = 190
    MAV_COMP_ID_CAMERA = 100
    MAV_COMP_ID_GIMBAL = 140


class RESULT_NAME:
    pass


_RESULT_NAMES = {
    0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED", 3: "UNSUPPORTED",
    4: "FAILED", 5: "IN_PROGRESS", 6: "CANCELLED",
}


def result_name(code: int) -> str:
    """Human-readable ``MAV_RESULT``, so logs say DENIED rather than 2."""
    return _RESULT_NAMES.get(int(code), f"UNKNOWN({code})")


# --------------------------------------------------------------------------- #
# Copter flight modes
# --------------------------------------------------------------------------- #
#: ArduPilot Copter ``FLTMODE`` numbers.  These are what goes in the
#: ``custom_mode`` field of ``SET_MODE`` / ``DO_SET_MODE`` for a copter; the
#: numbering is different for Plane and Rover, which is a classic source of
#: "the vehicle ignored my mode change" confusion.
COPTER_MODES: Dict[str, int] = {
    "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4,
    "LOITER": 5, "RTL": 6, "CIRCLE": 7, "LAND": 9, "DRIFT": 11,
    "SPORT": 13, "FLIP": 14, "AUTOTUNE": 15, "POSHOLD": 16, "BRAKE": 17,
    "THROW": 18, "AVOID_ADSB": 19, "GUIDED_NOGPS": 20, "SMART_RTL": 21,
    "FLOWHOLD": 22, "FOLLOW": 23, "ZIGZAG": 24, "SYSTEMID": 25,
    "AUTOROTATE": 26, "AUTO_RTL": 27, "TURTLE": 28,
}

#: Reverse map for decoding ``custom_mode`` in telemetry.
COPTER_MODE_NAMES: Dict[int, str] = {v: k for k, v in COPTER_MODES.items()}

#: Modes in which position and velocity setpoints are honoured.  Sending a
#: GUIDED target to a vehicle in LAND is not an error at the protocol level - it
#: is accepted and then ignored - so the connection layer checks first.
POSITION_MODES = frozenset({"GUIDED", "GUIDED_NOGPS", "LOITER", "POSHOLD",
                            "AUTO", "FOLLOW", "SMART_RTL", "AVOID_ADSB"})

#: Modes that hold altitude without a position fix.  A GPS-denied sortie has to
#: live in one of these or in GUIDED_NOGPS with an external nav source.
ALTITUDE_MODES = frozenset({"ALT_HOLD", "LOITER", "POSHOLD", "GUIDED_NOGPS"})


# --------------------------------------------------------------------------- #
# SET_POSITION_TARGET masks
# --------------------------------------------------------------------------- #
class TYPE_MASK:
    """Bit masks for ``SET_POSITION_TARGET_GLOBAL_INT`` / ``_LOCAL_NED``.

    MAVLink's mask convention is inverted relative to intuition: a *set* bit
    means "ignore this field".  Getting one bit wrong produces a vehicle that
    flies to the right place at the wrong speed, or holds a velocity target as
    if it were a position, so the masks are spelled out rather than composed
    inline at the call site.
    """

    IGNORE_X = 1 << 0
    IGNORE_Y = 1 << 1
    IGNORE_Z = 1 << 2
    IGNORE_VX = 1 << 3
    IGNORE_VY = 1 << 4
    IGNORE_VZ = 1 << 5
    IGNORE_AX = 1 << 6
    IGNORE_AY = 1 << 7
    IGNORE_AZ = 1 << 8
    IGNORE_YAW = 1 << 10
    IGNORE_YAW_RATE = 1 << 11
    FORCE = 1 << 12               # use for acceleration-only targets
    USE_ALT = 1 << 13             # altitude field is terrain-relative

    #: Position target: ignore everything except x/y/z.  This is what a survey
    #: waypoint uses.
    POS_ONLY = (IGNORE_VX | IGNORE_VY | IGNORE_VZ
                | IGNORE_AX | IGNORE_AY | IGNORE_AZ | IGNORE_YAW_RATE)
    #: Velocity target: ignore position and acceleration.  Streaming this at
    #: 4-10 Hz is how a search pattern is flown smoothly rather than as a
    #: sequence of stops, which matters because every stop costs energy and
    #: every acceleration smears the thermal image.
    VEL_ONLY = (IGNORE_X | IGNORE_Y | IGNORE_Z
                | IGNORE_AX | IGNORE_AY | IGNORE_AZ | IGNORE_YAW_RATE)
    #: Velocity in x/y but hold an explicit altitude - the normal search
    #: profile, since altitude is what sets the ground sample distance and must
    #: not be allowed to wander while the horizontal target streams.
    VEL_XY_ALT_Z = (IGNORE_X | IGNORE_Y
                    | IGNORE_VZ | IGNORE_AX | IGNORE_AY | IGNORE_AZ
                    | IGNORE_YAW_RATE)


def mask(*names: str) -> int:
    """Compose a type mask from ``TYPE_MASK`` field names."""
    out = 0
    for n in names:
        out |= int(getattr(TYPE_MASK, n))
    return out


# --------------------------------------------------------------------------- #
# EKF source sets (GPS-denied operation)
# --------------------------------------------------------------------------- #
class SourceXY:
    """``EK3_SRCn_POSXY`` and ``EK3_SRCn_VELXY`` values.

    Read out of ``libraries/AP_NavEKF/AP_NavEKF_Source.h`` in the ArduPilot
    checkout rather than from documentation, because the three source axes use
    *different* numberings and the gaps are not where intuition puts them: in
    this enum 1 and 2 are reserved slots (``// BARO = 1 (not applicable)``), so
    GPS is 3 and not 1.  Setting ``EK3_SRC1_POSXY=1`` fails the pre-arm check
    with a bare "Check EK3_SRC1_POSXY" and no further explanation, which is how
    this was found.
    """

    NONE = 0
    GPS = 3
    BEACON = 4
    OPTFLOW = 5
    EXTNAV = 6
    WHEEL_ENCODER = 7

    NAMES = {0: "NONE", 3: "GPS", 4: "BEACON", 5: "OPTFLOW",
             6: "EXTNAV", 7: "WHEEL_ENCODER"}

    @classmethod
    def name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"INVALID({code})")


class SourceZ:
    """``EK3_SRCn_POSZ`` and ``EK3_SRCn_VELZ`` values.

    The vertical axis shares one enum between height and vertical velocity, but
    the *validation* does not: ArduPilot accepts BARO, RANGEFINDER and BEACON for
    POSZ and explicitly rejects all three for VELZ, where only NONE, GPS and
    EXTNAV are legal.  A barometer gives height, not vertical speed, so the
    restriction is physical rather than arbitrary - but it means a source set
    written as "all ones" fails at the VELZ check after passing POSZ.
    """

    NONE = 0
    BARO = 1
    RANGEFINDER = 2
    GPS = 3
    BEACON = 4
    EXTNAV = 6

    #: Legal for ``VELZ``.  Everything else is rejected at pre-arm.
    VALID_VELZ = frozenset({0, 3, 6})

    NAMES = {0: "NONE", 1: "BARO", 2: "RANGEFINDER", 3: "GPS",
             4: "BEACON", 6: "EXTNAV"}

    @classmethod
    def name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"INVALID({code})")


class SourceYaw:
    """``EK3_SRCn_YAW`` values.

    COMPASS is accepted by the validator even when no compass exists - ArduPilot
    skips that check "for easier user setup of compass-less operation" - so it is
    a silent misconfiguration rather than an error.  The TBS Lucid H743 Wing has
    no internal magnetometer, so every set here uses GPS, EXTNAV or GSF and never
    COMPASS.  GSF (Gaussian Sum Filter) is worth knowing about: it derives yaw
    from GPS *velocity* rather than position, so it still produces a heading when
    the position solution has degraded past usefulness, which is exactly the
    regime an optical-flow confirmation pass operates in.
    """

    NONE = 0
    COMPASS = 1
    GPS = 2
    GPS_COMPASS_FALLBACK = 3
    EXTNAV = 6
    GSF = 8

    NAMES = {0: "NONE", 1: "COMPASS", 2: "GPS", 3: "GPS_COMPASS_FALLBACK",
             6: "EXTNAV", 8: "GSF"}

    @classmethod
    def name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"INVALID({code})")


class SOURCE_SET:
    """Which EKF3 source set is active, addressed by ``MAV_CMD_SET_EKF_SOURCE_SET``.

    This numbers the *sets* (1, 2, 3), not the sources within them; the sources
    are :class:`SourceXY`, :class:`SourceZ` and :class:`SourceYaw`.  Set 0 is the
    default and cannot be switched to at runtime.

    The vehicle is configured with three sets so that losing GNSS is one command
    rather than a parameter write plus a filter reset.  That distinction is the
    whole basis of surviving a denial in flight: during an EKF reset the vehicle
    has no position at all, and at 60 m over a flooded street that ends the
    sortie.
    """

    #: Normal operation: GNSS position, velocity, height and course.
    GNSS = 1
    #: GPS-denied: the onboard computer supplies position, velocity and attitude
    #: from visual-inertial odometry.
    EXTERNAL_NAV = 2
    #: No position aiding at all: optical-flow velocity with baro height, for a
    #: low confirmation pass under canopy or inside a structure.
    FLOW = 3

    NAMES = {1: "GNSS", 2: "EXTERNAL_NAV", 3: "FLOW"}

    @classmethod
    def name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"SET({code})")


#: MAVLink message IDs accepted as an external navigation source.  ``ODOMETRY``
#: (331) is preferred because it carries a full covariance and a child frame;
#: ``VISION_POSITION_ESTIMATE`` (102) is the legacy path and is still what most
#: VIO stacks emit.
EXTERNAL_NAV_MESSAGES = ("ODOMETRY", "VISION_POSITION_ESTIMATE",
                         "GLOBAL_VISION_POSITION_ESTIMATE",
                         "VISION_SPEED_ESTIMATE", "ATT_POS_MOCAP")

#: Rate at which an external nav source must be refreshed before EKF3 declares
#: it timed out.  ArduPilot's ``EK3_SRCn_*`` aiding times out at roughly 0.5 s,
#: so 20 Hz is the practical minimum and 50 Hz leaves margin for a lossy link.
EXTERNAL_NAV_MIN_HZ = 20.0


class GPS_FIX:
    """``GPS_RAW_INT.fix_type``."""

    NO_FIX = 0
    NO_FIX_2 = 1
    FIX_2D = 2
    FIX_3D = 3
    DGPS = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def clamp_i16(value: float) -> int:
    """Clamp to the signed 16-bit range MAVLink's ``int16`` fields accept.

    Used for centimetre and centidegree fields, where an unclamped value wraps
    rather than saturating and produces a target on the other side of the world.
    """
    return int(max(-32768, min(32767, round(value))))


def wrap_360(angle_deg: float) -> float:
    """Normalise an angle to [0, 360)."""
    return float(angle_deg % 360.0)


def wrap_pi(angle_rad: float) -> float:
    """Normalise an angle to [-pi, pi)."""
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def latlon_to_int(lat_deg: float, lon_deg: float) -> tuple:
    """MAVLink encodes latitude and longitude as degrees * 1e7."""
    return int(round(lat_deg * 1e7)), int(round(lon_deg * 1e7))


def int_to_latlon(lat_int: int, lon_int: int) -> tuple:
    return lat_int / 1e7, lon_int / 1e7


def describe_message(msg: Any) -> str:
    """One-line description of a MAVLink message, for logs and the dashboard."""
    t = msg.get_type()
    if t == "GLOBAL_POSITION_INT":
        return (f"GPS lat={msg.lat/1e7:.6f} lon={msg.lon/1e7:.6f} "
                f"alt={msg.alt/1000:.1f}m rel={msg.relative_alt/1000:.1f}m "
                f"hdg={msg.hdg/100:.0f}deg")
    if t == "ATTITUDE":
        return (f"ATT roll={math.degrees(msg.roll):+.1f} "
                f"pitch={math.degrees(msg.pitch):+.1f} "
                f"yaw={math.degrees(msg.yaw):+.1f}")
    if t == "SYS_STATUS":
        return (f"SYS v={msg.voltage_battery/1000:.2f}V "
                f"a={msg.current_battery/100:.1f}A batt={msg.battery_remaining}% "
                f"load={msg.load}%")
    if t == "HEARTBEAT":
        return (f"HB type={msg.type} base=0x{msg.base_mode:02x} "
                f"custom={msg.custom_mode} sys={msg.system_status}")
    if t == "COMMAND_ACK":
        return (f"ACK cmd={msg.command} result={result_name(msg.result)}")
    return f"{t}: {msg.to_dict()}"
