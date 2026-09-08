"""Rigid-body attitude maths: Euler angles, DCMs and quaternions.

Convention (identical to ArduPilot/PX4):

* Body frame  - X forward, Y right, Z down (FRD).
* World frame - NED (North, East, Down).
* Euler order - ZYX intrinsic, i.e. ``R_body->NED = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

Everything is radians here.  The MAVLink layer converts to/from degrees.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

Vec3 = np.ndarray  # shape (3,)
Mat3 = np.ndarray  # shape (3, 3)


def euler_to_dcm(roll: float, pitch: float, yaw: float) -> Mat3:
    """Body -> NED direction cosine matrix from ZYX Euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=float,
    )


def dcm_to_euler(dcm: Mat3) -> Tuple[float, float, float]:
    """NED->body or body->NED DCM (body->NED expected) -> (roll, pitch, yaw)."""
    pitch = math.asin(np.clip(-dcm[2, 0], -1.0, 1.0))
    if abs(math.cos(pitch)) < 1e-9:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-dcm[0, 1], dcm[1, 1])
    else:
        roll = math.atan2(dcm[2, 1], dcm[2, 2])
        yaw = math.atan2(dcm[1, 0], dcm[0, 0])
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->NED attitude as ``[w, x, y, z]`` (MAVLink ordering)."""
    cr, cp, cy = math.cos(roll / 2), math.cos(pitch / 2), math.cos(yaw / 2)
    sr, sp, sy = math.sin(roll / 2), math.sin(pitch / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def quaternion_to_dcm(q: np.ndarray) -> Mat3:
    w, x, y, z = q / (np.linalg.norm(q) or 1.0)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def body_to_ned(vec: Vec3, roll: float, pitch: float, yaw: float) -> Vec3:
    return euler_to_dcm(roll, pitch, yaw) @ np.asarray(vec, dtype=float)


def ned_to_body(vec: Vec3, roll: float, pitch: float, yaw: float) -> Vec3:
    return euler_to_dcm(roll, pitch, yaw).T @ np.asarray(vec, dtype=float)


def wrap_pi(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_error(yaw: float, target: float) -> float:
    """Signed shortest angular error ``yaw -> target`` in radians."""
    return wrap_pi(target - yaw)


def skew(v: Vec3) -> Mat3:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def look_at_yaw(north: float, east: float) -> float:
    """Yaw (rad, NED) that points the nose at a NED offset."""
    return math.atan2(east, north)


def ground_distance(north: float, east: float) -> float:
    return float(math.hypot(north, east))
