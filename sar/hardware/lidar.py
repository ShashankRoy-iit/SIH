"""LiDAR: true height above ground + low-altitude obstacle stop.

Two very different jobs share this file, because they share one sensor bus:

1. **1D rangefinder (Benewake TF-Luna / TFmini-S, ~8–12 m, ~Rs 3k, 5 g).**
   Gives *true AGL* — the one number the whole detection geometry depends on.
   Baro altitude drifts with weather; GPS altitude is +/-5 m fiction; the
   rangefinder reads the ground.  It also drives the landing flare and the
   "too low over water" guard.  Wired to a spare SERIAL port as
   ``RNGFND1_TYPE=20`` (serial) or over I2C; ArduPilot natively understands it.

2. **2D scanning LiDAR (RPLidar A1/A2 class, optional).**  Gives a 360 deg
   obstacle polar map for the confirmation pass, where the aircraft descends
   to 22 m between buildings, wires and trees.  Consumed as MAVLink
   ``OBSTACLE_DISTANCE`` (ArduPilot PRX) for automatic slowdown/stop, and by
   the planner as a local cost bump.

Design rules followed here:
* The rangefinder is *trusted* only inside its honest envelope (0.3–8 m for
  TF-Luna, quality flag good, variance small).  Outside it, AGL falls back to
  baro+terrain with a widened sigma — never a frozen last reading.
* Water is a liar: still flood water at nadir can absorb/reflect the beam and
  read long or drop out.  ``water_likely`` (from the RGB water mask) widens
  the AGL sigma and refuses landing on water.
* No serial/I2C dependency is imported at module scope; drivers are injected,
  so bench tests and the simulators run dependency-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 1D rangefinder
# --------------------------------------------------------------------------- #
@dataclass
class RangeSample:
    t: float
    range_m: float          # raw reading, metres
    quality: int = 255      # 0..255 driver quality (TF-Luna signal strength)
    valid: bool = True


@dataclass
class AglEstimate:
    agl_m: float            # fused height above ground, metres
    sigma_m: float          # 1-sigma uncertainty
    source: str             # 'lidar' | 'baro_terrain' | 'baro_only'
    lidar_valid: bool = False
    water_likely: bool = False


class Rangefinder1D:
    """TF-Luna class driver front-end: median filtering + envelope gating.

    ``read`` is injected: on the aircraft it parses the 9-byte TF-Luna UART
    frame (0x59 0x59); on the bench it returns scripted samples.
    """

    MIN_RANGE_M = 0.30
    MAX_RANGE_M = 8.0
    MIN_QUALITY = 40

    def __init__(self, read: Optional[Callable[[], Optional[RangeSample]]] = None,
                 window: int = 5) -> None:
        self._read = read or (lambda: None)
        self.window = max(1, int(window))
        self._hist: List[RangeSample] = []
        self.last: Optional[RangeSample] = None

    def poll(self) -> Optional[RangeSample]:
        s = self._read()
        if s is None:
            return self.last
        s.valid = bool(self.MIN_RANGE_M <= s.range_m <= self.MAX_RANGE_M
                       and s.quality >= self.MIN_QUALITY)
        self.last = s
        if s.valid:
            self._hist.append(s)
            self._hist = self._hist[-self.window:]
        return s

    @property
    def filtered_m(self) -> Optional[float]:
        if not self._hist:
            return None
        return float(np.median([s.range_m for s in self._hist]))

    @property
    def variance_m2(self) -> float:
        if len(self._hist) < 2:
            return 1e6
        return float(np.var([s.range_m for s in self._hist]))

    def healthy(self) -> bool:
        return (self.last is not None and self.last.valid
                and self.variance_m2 < 1.0)


class AglFuser:
    """Fuse rangefinder + baro + terrain into one honest AGL number.

    * Inside the LiDAR envelope with a healthy driver: trust it (sigma 0.15 m,
      widened to 0.6 m when ``water_likely``).
    * Outside: baro altitude minus terrain height, sigma 2.5 m (terrain + baro
      drift).  Never better than the sensors deserve.
    """

    LIDAR_SIGMA_M = 0.15
    LIDAR_WATER_SIGMA_M = 0.60
    BARO_TERRAIN_SIGMA_M = 2.5

    def __init__(self, rf: Optional[Rangefinder1D] = None) -> None:
        self.rf = rf or Rangefinder1D()

    def update(self, baro_alt_amsl_m: float, terrain_amsl_m: float,
               water_likely: bool = False) -> AglEstimate:
        self.rf.poll()
        if self.rf.healthy():
            m = self.rf.filtered_m
            assert m is not None
            sigma = (self.LIDAR_WATER_SIGMA_M if water_likely else self.LIDAR_SIGMA_M)
            return AglEstimate(agl_m=m, sigma_m=sigma, source="lidar",
                               lidar_valid=True, water_likely=water_likely)
        agl = max(baro_alt_amsl_m - terrain_amsl_m, 0.0)
        return AglEstimate(agl_m=agl, sigma_m=self.BARO_TERRAIN_SIGMA_M,
                           source="baro_terrain", lidar_valid=False,
                           water_likely=water_likely)


# --------------------------------------------------------------------------- #
# 2D obstacle map (optional scanning LiDAR)
# --------------------------------------------------------------------------- #
@dataclass
class PolarScan:
    t: float
    angles_deg: np.ndarray      # shape (N,), 0 = forward, clockwise
    ranges_m: np.ndarray        # shape (N,), inf = no return
    max_range_m: float = 12.0

    @property
    def closest_m(self) -> float:
        finite = self.ranges_m[np.isfinite(self.ranges_m)]
        return float(finite.min()) if finite.size else math.inf

    def sector_min(self, center_deg: float, half_width_deg: float) -> float:
        d = np.abs((self.angles_deg - center_deg + 180.0) % 360.0 - 180.0)
        sel = self.ranges_m[d <= half_width_deg]
        sel = sel[np.isfinite(sel)]
        return float(sel.min()) if sel.size else math.inf


@dataclass
class AvoidDecision:
    action: str                 # 'clear' | 'slow' | 'stop'
    closest_m: float
    ahead_m: float              # clearance in the direction of travel
    scale_velocity: float = 1.0  # multiply the guidance velocity by this


class ObstacleGuard:
    """Turns a polar scan into a velocity scale + stop decision.

    Thresholds are in metres of *ahead* clearance (direction of travel):
    >6 m clear, 3–6 m slow proportionally, <3 m stop.  The guard never steers
    — it only scales/stops; steering stays with the planner and the pilot.
    """

    SLOW_M = 6.0
    STOP_M = 3.0

    def __init__(self, slow_m: float = 6.0, stop_m: float = 3.0) -> None:
        self.SLOW_M = slow_m
        self.STOP_M = stop_m

    def decide(self, scan: PolarScan, heading_deg: float = 0.0) -> AvoidDecision:
        ahead = scan.sector_min(heading_deg, 30.0)
        closest = scan.closest_m
        if ahead < self.STOP_M:
            return AvoidDecision("stop", closest, ahead, 0.0)
        if ahead < self.SLOW_M:
            scale = (ahead - self.STOP_M) / max(self.SLOW_M - self.STOP_M, 1e-6)
            return AvoidDecision("slow", closest, ahead, float(np.clip(scale, 0.2, 1.0)))
        return AvoidDecision("clear", closest, ahead, 1.0)

    @staticmethod
    def to_mavlink_obstacle_distance(scan: PolarScan) -> List[int]:
        """Pack a scan into 72 x 5-degree OBSTACLE_DISTANCE bins (cm)."""
        bins = np.full(72, 0, dtype=int)  # 0 = unknown per MAVLink spec
        idx = np.floor(((scan.angles_deg % 360.0) / 5.0)).astype(int) % 72
        for i, r in zip(idx, scan.ranges_m):
            if np.isfinite(r):
                cm = int(r * 100)
                bins[i] = cm if bins[i] == 0 else min(bins[i], cm)
        return bins.tolist()


class SimulatedLidar:
    """Scripted 1D + 2D LiDAR for bench tests and the flood sims."""

    def __init__(self, agl_fn: Optional[Callable[[float], float]] = None,
                 obstacles: Optional[List[Tuple[float, float, float]]] = None,
                 seed: int = 11) -> None:
        self.agl_fn = agl_fn or (lambda t: 45.0)
        self.obstacles = obstacles or []  # (x_n, x_e, radius_m) world poles
        self._rng = np.random.default_rng(seed)
        self.t = 0.0

    def range_sample(self, t: Optional[float] = None) -> RangeSample:
        t = self.t if t is None else t
        agl = self.agl_fn(t)
        # TF-Luna honest envelope: reads only below 8 m; above that, dropout.
        if agl > Rangefinder1D.MAX_RANGE_M:
            return RangeSample(t=t, range_m=9.9, quality=0, valid=False)
        noisy = agl + self._rng.normal(0, 0.03)
        return RangeSample(t=t, range_m=max(noisy, 0.0), quality=220, valid=True)

    def scan(self, pos_n: float, pos_e: float, heading_deg: float = 0.0,
             n_rays: int = 72, max_range_m: float = 12.0,
             t: Optional[float] = None) -> PolarScan:
        t = self.t if t is None else t
        angles = np.linspace(0, 360, n_rays, endpoint=False)
        ranges = np.full(n_rays, np.inf)
        for i, a in enumerate(angles):
            th = math.radians(a + heading_deg)
            dx, dy = math.sin(th), math.cos(th)
            best = max_range_m
            for (ox, oy, r) in self.obstacles:
                # Ray-circle intersection in the NE plane.
                ox0, oy0 = ox - pos_n, oy - pos_e
                proj = ox0 * dx + oy0 * dy
                if proj < 0:
                    continue
                perp2 = ox0 * ox0 + oy0 * oy0 - proj * proj
                if perp2 > r * r:
                    continue
                hit = proj - math.sqrt(max(r * r - perp2, 0.0))
                if 0 < hit < best:
                    best = hit
            if best < max_range_m:
                ranges[i] = best + self._rng.normal(0, 0.02)
        return PolarScan(t=t, angles_deg=angles, ranges_m=ranges,
                         max_range_m=max_range_m)


__all__ = ["RangeSample", "AglEstimate", "Rangefinder1D", "AglFuser",
           "PolarScan", "AvoidDecision", "ObstacleGuard", "SimulatedLidar"]
