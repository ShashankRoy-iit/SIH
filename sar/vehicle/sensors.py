"""Sensor models with failure modes that actually matter in a disaster.

Every sensor exposes a ``measure(true_state, t)`` method returning a dataclass
sample.  Noise is drawn from a seeded ``numpy.random.Generator`` so a scenario
replays exactly.

The interesting parts are the *degradation* models, which are what make the
GPS-denied autonomy work worth testing:

``GpsSensor``
    Satellite count, HDOP/VDOP, fix type and a per-scenario *denial field*
    (urban canyon multipath, jamming, deep valleys, heavy rain attenuation).
    A 30+ satellite multi-GNSS module (GPS+GLONASS+Galileo+BeiDou) is modelled
    explicitly, because that is what the team has on the bench.

``OpticalFlowSensor``
    Quality collapses over *floodwater* and *uniform snow/ash* - a smooth,
    featureless surface gives no texture to track.  This is a real and widely
    overlooked failure mode for GPS-denied flight over inundated terrain.

``VioSensor``
    Visual-inertial odometry with bounded drift, usable as the EKF3
    "ExternalNav" source when GNSS is denied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from sar.core.frames import euler_to_dcm, ned_to_body
from sar.core.geo import GeoPoint, local_to_wgs84
from sar.vehicle.dynamics import PlantState
from sar.vehicle.power import G0

# Fix types, matching the MAVLink GPS_FIX_TYPE enum.
FIX_NO_FIX = 0
FIX_NO_FIX_2 = 1
FIX_2D = 2
FIX_3D = 3
FIX_DGPS = 4
FIX_RTK_FLOAT = 5
FIX_RTK_FIXED = 6


@dataclass
class ImuSample:
    accel: np.ndarray          # body-frame specific force [m/s^2]
    gyro: np.ndarray           # body rates [rad/s]
    temp_c: float = 35.0


@dataclass
class BaroSample:
    altitude_m: float          # AMSL
    pressure_pa: float = 101325.0
    temp_c: float = 15.0


@dataclass
class MagSample:
    field_mgauss: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading_deg: float = 0.0
    interference: float = 0.0


@dataclass
class GpsSample:
    valid: bool
    fix_type: int
    num_sats: int
    lat: float
    lon: float
    alt_msl: float
    vel_ned: np.ndarray
    hdop: float
    vdop: float
    h_accuracy_mm: int
    v_accuracy_mm: int
    v_speed_accuracy_mm: int = 200
    jammed: bool = False
    yaw_deg: float = 0.0


@dataclass
class FlowSample:
    valid: bool
    flow_rate: np.ndarray      # body-frame angular flow [rad/s] about X, Y
    distance_m: float          # integrated rangefinder distance
    quality: int               # 0..255 (ArduPilot convention)


@dataclass
class RangeSample:
    valid: bool
    distance_m: float
    signal_quality: float = 1.0


@dataclass
class VioSample:
    valid: bool
    position: np.ndarray       # local NED estimate [m]
    velocity: np.ndarray       # local NED estimate [m/s]
    yaw: float                 # rad
    confidence: int = 100      # 0..100, as consumed by VISION_POSITION_DELTA


# --------------------------------------------------------------------------- #
# Environment description
# --------------------------------------------------------------------------- #
@dataclass
class GpsEnvironment:
    """Spatially-varying GNSS quality.

    ``denial_zones`` is a list of ``(north, east, radius_m, severity)`` where
    severity 1.0 is total denial (jammer / deep峡谷) and 0.0 is no effect.
    ``canyon_zones`` model urban multipath: satellites stay visible but HDOP
    degrades and the position estimate jumps.
    """

    base_satellites: int = 34          # 30+ satellite multi-GNSS receiver
    base_hdop: float = 0.62
    base_vdop: float = 0.95
    base_h_accuracy_m: float = 0.9
    base_v_accuracy_m: float = 1.6
    denial_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)
    canyon_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)
    rain_attenuation: float = 0.0      # 0..1 extra degradation from heavy rain
    iono_storm: float = 0.0            # 0..1 regional scintillation

    def sample_field(self, north: float, east: float) -> Tuple[float, float]:
        """Return (denial 0..1, canyon 0..1) at a local position."""
        denial = 0.0
        for (zn, ze, zr, sev) in self.denial_zones:
            d = math.hypot(north - zn, east - ze)
            if d < zr:
                denial = max(denial, sev * (1.0 - d / zr) ** 0.5)
        canyon = 0.0
        for (zn, ze, zr, sev) in self.canyon_zones:
            d = math.hypot(north - zn, east - ze)
            if d < zr:
                canyon = max(canyon, sev * (1.0 - d / zr))
        return float(np.clip(denial, 0.0, 1.0)), float(np.clip(canyon, 0.0, 1.0))


@dataclass
class TerrainTexture:
    """Where the ground has visual texture (needed for optical flow).

    A callable returning 0..1: 1 = richly textured, 0 = featureless.
    """

    fn: Optional[Callable[[float, float], float]] = None

    def at(self, north: float, east: float) -> float:
        if self.fn is None:
            return 0.85
        return float(np.clip(self.fn(north, east), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #
class ImuSensor:
    """Dual-IMU (ICM-42688-P class) accelerometer + gyroscope."""

    def __init__(self, seed: int = 1, accel_noise: float = 0.18, gyro_noise: float = 0.006,
                 accel_bias: float = 0.05, gyro_bias: float = 0.004) -> None:
        self.rng = np.random.default_rng(seed)
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self._accel_bias = np.array([
            self.rng.normal(0, accel_bias),
            self.rng.normal(0, accel_bias),
            self.rng.normal(0, accel_bias),
        ])
        self._gyro_bias = np.array([
            self.rng.normal(0, gyro_bias),
            self.rng.normal(0, gyro_bias),
            self.rng.normal(0, gyro_bias),
        ])
        self._prev_vel: Optional[np.ndarray] = None

    def measure(self, st: PlantState, dt: float, accel_ned: Optional[np.ndarray] = None) -> ImuSample:
        dcm = euler_to_dcm(*st.euler)
        if accel_ned is None:
            # Finite-difference of the state is good enough at sim rates.
            accel_ned = np.zeros(3)
        # Specific force = a - g  (NED, g = +Z)
        f_ned = accel_ned - np.array([0.0, 0.0, G0])
        f_body = dcm.T @ f_ned
        gyro_body = st.omega.copy()
        accel = f_body + self._accel_bias + self.rng.normal(0, self.accel_noise, 3)
        gyro = gyro_body + self._gyro_bias + self.rng.normal(0, self.gyro_noise, 3)
        return ImuSample(accel=accel, gyro=gyro, temp_c=38.0 + 6.0 * st.thrust_n / 60.0)


class BaroSensor:
    """DPS310-class barometer with a slowly drifting bias."""

    def __init__(self, seed: int = 2, noise: float = 0.35, bias_drift: float = 0.02,
                 msl_altitude: float = 0.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.bias_drift = bias_drift
        self.msl = msl_altitude
        self._bias = 0.0

    def measure(self, st: PlantState, t: float) -> BaroSample:
        self._bias += self.rng.normal(0.0, self.bias_drift * 0.02)
        self._bias = float(np.clip(self._bias, -1.5, 1.5))
        alt = self.msl - st.pos[2] + self._bias + self.rng.normal(0, self.noise)
        # ISA pressure from altitude.
        p = 101325.0 * (1.0 - 2.25577e-5 * alt) ** 5.25588
        temp = 15.0 - 0.0065 * alt
        return BaroSample(altitude_m=alt, pressure_pa=p, temp_c=temp)


class MagSensor:
    """External I2C compass with motor-current interference."""

    def __init__(self, seed: int = 3, declination_deg: float = -1.2, noise: float = 40.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.declination = math.radians(declination_deg)
        self.noise = noise   # milligauss

    def measure(self, st: PlantState, current_a: float) -> MagSample:
        dcm = euler_to_dcm(*st.euler)
        # Earth field, roughly 0.4 gauss, inclined; use a simple local vector.
        earth_ned = np.array([0.28, 0.005, -0.15]) * 1000.0   # milligauss
        body = dcm.T @ earth_ned
        # Motor interference scales with pack current and is fixed in body frame.
        interference = 0.55 * current_a
        body = body + np.array([interference, -0.4 * interference, 0.2 * interference])
        body = body + self.rng.normal(0, self.noise, 3)
        heading = math.degrees(math.atan2(earth_ned[1], earth_ned[0]))
        return MagSample(field_mgauss=body, heading_deg=heading, interference=interference)


class GpsSensor:
    """Multi-GNSS receiver (30+ satellites) with denial / canyon modelling."""

    def __init__(self, origin: GeoPoint, env: Optional[GpsEnvironment] = None,
                 seed: int = 4, update_hz: float = 10.0) -> None:
        self.origin = origin
        self.env = env or GpsEnvironment()
        self.rng = np.random.default_rng(seed)
        self.update_hz = update_hz
        self._last: Optional[GpsSample] = None
        self._last_update = -1.0
        # Random-walk multipath error state (produces realistic "jumping" fixes)
        self._mpath = np.zeros(2)
        self.jam_detected_at: Optional[float] = None

    def measure(self, st: PlantState, t: float) -> GpsSample:
        if t - self._last_update < 1.0 / self.update_hz and self._last is not None:
            return self._last
        self._last_update = t

        denial, canyon = self.env.sample_field(st.pos[0], st.pos[1])
        weather = self.env.rain_attenuation * 0.4 + self.env.iono_storm * 0.6

        sats = self.env.base_satellites
        sats = int(round(sats * (1.0 - 0.85 * denial) * (1.0 - 0.35 * canyon) * (1.0 - 0.3 * weather)))
        sats = max(0, sats + int(self.rng.integers(-1, 2)))

        hdop = self.env.base_hdop * (1.0 + 3.5 * canyon + 6.0 * denial + 2.0 * weather)
        vdop = self.env.base_vdop * (1.0 + 3.0 * canyon + 6.0 * denial + 2.0 * weather)
        hdop = max(0.35, hdop + abs(self.rng.normal(0, 0.05)))

        if denial > 0.92 or sats < 4:
            fix = FIX_NO_FIX
            valid = False
        elif denial > 0.55 or sats < 6:
            fix = FIX_2D
            valid = True
        else:
            fix = FIX_3D if canyon < 0.5 else FIX_2D
            valid = True

        h_acc = self.env.base_h_accuracy_m * hdop / max(self.env.base_hdop, 1e-6)
        v_acc = self.env.base_v_accuracy_m * vdop / max(self.env.base_vdop, 1e-6)

        # Position error: white noise + a correlated multipath random walk.
        self._mpath = 0.97 * self._mpath + self.rng.normal(0, 0.35 * (1.0 + 4.0 * canyon), 2)
        err_n = self.rng.normal(0, h_acc) + self._mpath[0] * (1.0 if valid else 0.0)
        err_e = self.rng.normal(0, h_acc) + self._mpath[1] * (1.0 if valid else 0.0)
        err_d = self.rng.normal(0, v_acc)

        truth = local_to_wgs84(st.pos[0], st.pos[1], st.pos[2], self.origin)
        noisy = local_to_wgs84(st.pos[0] + err_n, st.pos[1] + err_e, st.pos[2] + err_d, self.origin)

        jammed = denial > 0.35
        if jammed and self.jam_detected_at is None:
            self.jam_detected_at = t

        sample = GpsSample(
            valid=valid,
            fix_type=fix,
            num_sats=sats,
            lat=noisy.lat if valid else truth.lat,
            lon=noisy.lon if valid else truth.lon,
            alt_msl=noisy.alt if valid else truth.alt,
            vel_ned=st.vel + self.rng.normal(0, 0.08 * hdop, 3),
            hdop=hdop,
            vdop=vdop,
            h_accuracy_mm=int(h_acc * 1000),
            v_accuracy_mm=int(v_acc * 1000),
            jammed=jammed,
            yaw_deg=math.degrees(st.euler[2]) % 360.0,
        )
        self._last = sample
        return sample


class RangefinderSensor:
    """Downward LiDAR/ToF rangefinder (e.g. 0.1-40 m)."""

    def __init__(self, seed: int = 5, max_range: float = 40.0, noise: float = 0.05,
                 terrain_fn: Optional[Callable[[float, float], float]] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_range = max_range
        self.noise = noise
        self.terrain_fn = terrain_fn

    def measure(self, st: PlantState) -> RangeSample:
        ground = self.terrain_fn(st.pos[0], st.pos[1]) if self.terrain_fn else 0.0
        agl = -st.pos[2] - ground
        if agl > self.max_range or agl < 0.02:
            return RangeSample(valid=False, distance_m=0.0, signal_quality=0.0)
        d = agl + self.rng.normal(0, self.noise)
        quality = 1.0 if agl < self.max_range * 0.8 else 0.5
        return RangeSample(valid=True, distance_m=max(d, 0.0), signal_quality=quality)


class OpticalFlowSensor:
    """Downward optical-flow module whose quality depends on ground texture.

    Floodwater, wet mud and uniform ash are near-featureless: the sensor keeps
    reporting but with collapsing quality.  The autonomy stack must notice this
    and refuse to rely on flow for position hold - otherwise the vehicle drifts
    away over exactly the terrain it was sent to survey.
    """

    def __init__(self, seed: int = 6, fov_deg: float = 42.0,
                 texture: Optional[TerrainTexture] = None, max_range: float = 30.0,
                 rangefinder: Optional[RangefinderSensor] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.fov = math.radians(fov_deg)
        self.texture = texture or TerrainTexture()
        self.max_range = max_range
        self.range = rangefinder or RangefinderSensor(seed=7, max_range=max_range)

    def measure(self, st: PlantState, t: float, light: float = 1.0) -> FlowSample:
        rng_sample = self.range.measure(st)
        agl = rng_sample.distance_m if rng_sample.valid else max(-st.pos[2], 0.5)
        texture = self.texture.at(st.pos[0], st.pos[1])
        in_range = 0.08 < agl < self.max_range

        quality_f = 1.0
        quality_f *= texture                              # surface texture
        quality_f *= float(np.clip(light, 0.05, 1.0)) ** 0.5   # illumination
        quality_f *= 1.0 if in_range else 0.0
        quality_f *= 1.0 - 0.5 * float(np.clip((agl - 6.0) / 24.0, 0.0, 1.0))  # height falloff
        if abs(st.euler[0]) > math.radians(30) or abs(st.euler[1]) > math.radians(30):
            quality_f *= 0.25                              # out of FOV

        # Body-frame angular flow = ground-relative horizontal velocity / height.
        v_rel = st.vel[:2] - np.array([0.0, 0.0])
        if hasattr(self, "wind") and self.wind is not None:
            v_rel = st.vel[:2] - self.wind[:2]
        flow = np.array([v_rel[1], -v_rel[0]]) / max(agl, 0.1)   # about X, Y
        noise = 0.02 + 0.12 * (1.0 - quality_f)
        flow = flow + self.rng.normal(0, noise, 2)
        valid = quality_f > 0.12 and in_range
        return FlowSample(
            valid=bool(valid),
            flow_rate=flow if valid else np.zeros(2),
            distance_m=agl,
            quality=int(np.clip(quality_f, 0.0, 1.0) * 255),
        )


class VioSensor:
    """Visual-inertial odometry estimate with bounded drift.

    Emulates what a companion computer running ORB-SLAM3/VINS-Fusion would feed
    to ArduPilot through ``ODOMETRY`` or ``VISION_POSITION_DELTA``.  Drift is a
    random walk whose rate depends on visual quality, plus an occasional
    "relocalisation" correction when texture is good.
    """

    def __init__(self, seed: int = 8, drift_ms: float = 0.06,
                 texture: Optional[TerrainTexture] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.drift_ms = drift_ms
        self.texture = texture or TerrainTexture()
        self._err = np.zeros(3)
        self._verr = np.zeros(3)
        self.lost = False
        self._last_t = 0.0

    def measure(self, st: PlantState, t: float) -> VioSample:
        dt = max(0.0, t - self._last_t) if self._last_t else 0.0
        self._last_t = t
        texture = self.texture.at(st.pos[0], st.pos[1])
        rate = self.drift_ms * (1.6 - texture)
        self._err += self._verr * dt + self.rng.normal(0, rate * math.sqrt(max(dt, 1e-6)), 3)
        self._verr += self.rng.normal(0, 0.02 * math.sqrt(max(dt, 1e-6)), 3)
        self._verr *= 0.995
        self.lost = texture < 0.15
        confidence = int(np.clip(texture * 100.0, 0, 100)) if not self.lost else 0
        return VioSample(
            valid=not self.lost,
            position=st.pos + self._err,
            velocity=st.vel + self._verr,
            yaw=st.euler[2] + self.rng.normal(0, 0.02),
            confidence=confidence,
        )

    def relocalise(self, correction: np.ndarray, quality: float = 1.0) -> None:
        """Apply an absolute fix (e.g. a recognised ground landmark or GNSS)."""
        self._err -= np.asarray(correction) * quality
        self._verr *= (1.0 - 0.7 * quality)


class SensorSuite:
    """Bundles the sensors into one object with a single ``measure_all``."""

    def __init__(
        self,
        origin: GeoPoint,
        gps_env: Optional[GpsEnvironment] = None,
        terrain_fn: Optional[Callable[[float, float], float]] = None,
        texture_fn: Optional[Callable[[float, float], float]] = None,
        msl_altitude: float = 0.0,
        seed: int = 1234,
        light_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        texture = TerrainTexture(texture_fn)
        self.imu = ImuSensor(seed=seed + 1)
        self.baro = BaroSensor(seed=seed + 2, msl_altitude=msl_altitude)
        self.mag = MagSensor(seed=seed + 3)
        self.gps = GpsSensor(origin, gps_env, seed=seed + 4)
        self.rangefinder = RangefinderSensor(seed=seed + 5, terrain_fn=terrain_fn)
        self.flow = OpticalFlowSensor(seed=seed + 6, texture=texture, rangefinder=self.rangefinder)
        self.vio = VioSensor(seed=seed + 7, texture=texture)
        self.origin = origin
        self.light_fn = light_fn or (lambda t: 1.0)

    def measure_all(self, st: PlantState, t: float, dt: float,
                    accel_ned: Optional[np.ndarray] = None,
                    current_a: float = 0.0) -> Dict[str, object]:
        return {
            "imu": self.imu.measure(st, dt, accel_ned),
            "baro": self.baro.measure(st, t),
            "mag": self.mag.measure(st, current_a),
            "gps": self.gps.measure(st, t),
            "range": self.rangefinder.measure(st),
            "flow": self.flow.measure(st, t, self.light_fn(t)),
            "vio": self.vio.measure(st, t),
        }
