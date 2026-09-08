"""Geodesy helpers for the SAHYOG SAR drone stack.

The mission area for a disaster sortie is at most a few tens of kilometres
across, so we use the standard ArduPilot/PX4 convention: a local
North-East-Down (NED) tangent plane anchored at an EKF origin, plus exact
great-circle helpers for longer baselines (e.g. rally points, relay drops).

All angles are in **degrees** at this module's public boundary (matching
MAVLink and GPS conventions) and metres for distances.  Internally we work in
radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# WGS-84 constants
# --------------------------------------------------------------------------- #
WGS84_A = 6378137.0                 # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563       # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
MEAN_EARTH_RADIUS = 6371008.8       # IUGG mean radius [m]


@dataclass(frozen=True)
class GeoPoint:
    """A WGS-84 position.

    Attributes
    ----------
    lat, lon : float
        Geodetic latitude / longitude in degrees.
    alt : float
        Altitude above mean sea level (AMSL) in metres.  Note MAVLink uses
        *millimetres* for this on the wire; conversion happens in the MAVLink
        layer, not here.
    """

    lat: float
    lon: float
    alt: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.lat, self.lon, self.alt)

    def distance_to(self, other: "GeoPoint") -> float:
        """Great-circle surface distance in metres (ignores altitude)."""
        return haversine_m(self.lat, self.lon, other.lat, other.lon)

    def distance_3d(self, other: "GeoPoint") -> float:
        """Slant distance in metres including altitude difference."""
        d = self.distance_to(other)
        dz = self.alt - other.alt
        return float(math.hypot(d, dz))

    def offset(self, north_m: float, east_m: float, up_m: float = 0.0) -> "GeoPoint":
        return local_to_wgs84(north_m, east_m, -up_m, self)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.lat:.7f},{self.lon:.7f},{self.alt:.2f}"


# --------------------------------------------------------------------------- #
# Great-circle / small-offset conversions
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * MEAN_EARTH_RADIUS * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees true."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.degrees(math.atan2(y, x)) % 360.0


def destination_point(origin: GeoPoint, bearing: float, distance_m: float) -> GeoPoint:
    """Point reached by travelling ``distance_m`` along ``bearing`` (deg true)."""
    d_ang = distance_m / MEAN_EARTH_RADIUS
    b = math.radians(bearing)
    p1, l1 = math.radians(origin.lat), math.radians(origin.lon)
    p2 = math.asin(
        math.sin(p1) * math.cos(d_ang) + math.cos(p1) * math.sin(d_ang) * math.cos(b)
    )
    l2 = l1 + math.atan2(
        math.sin(b) * math.sin(d_ang) * math.cos(p1),
        math.cos(d_ang) - math.sin(p1) * math.sin(p2),
    )
    return GeoPoint(math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0, origin.alt)


def meters_per_degree(lat: float) -> Tuple[float, float]:
    """(metres per degree latitude, metres per degree longitude) at ``lat``."""
    p = math.radians(lat)
    denom = math.sqrt(1.0 - WGS84_E2 * math.sin(p) ** 2)
    m_lat = math.pi * WGS84_A * (1.0 - WGS84_E2) / (180.0 * denom ** 3)
    m_lon = math.pi * WGS84_A * math.cos(p) / (180.0 * denom)
    return m_lat, m_lon


def wgs84_to_local(point: GeoPoint, origin: GeoPoint) -> Tuple[float, float, float]:
    """Convert ``point`` into the NED tangent plane anchored at ``origin``.

    Returns ``(north_m, east_m, down_m)``.  Uses the local meridian/parallel
    radii, which keeps errors below ~10 cm at 10 km offset - far better than
    the sensor noise floor of a hobby-grade GNSS receiver.
    """
    m_lat, m_lon = meters_per_degree(origin.lat)
    north = (point.lat - origin.lat) * m_lat
    east = (point.lon - origin.lon) * m_lon
    down = -(point.alt - origin.alt)
    return north, east, down


def local_to_wgs84(north: float, east: float, down: float, origin: GeoPoint) -> GeoPoint:
    """Inverse of :func:`wgs84_to_local`."""
    m_lat, m_lon = meters_per_degree(origin.lat)
    lat = origin.lat + north / m_lat
    lon = origin.lon + east / m_lon
    alt = origin.alt - down
    return GeoPoint(lat, lon, alt)


def wgs84_to_local_array(
    points: Iterable[GeoPoint], origin: GeoPoint
) -> np.ndarray:
    """Vectorised :func:`wgs84_to_local` -> ``(N, 3)`` NED array."""
    arr = np.asarray([wgs84_to_local(p, origin) for p in points], dtype=float)
    return arr.reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Geometry utilities used by the planner and the mapping layer
# --------------------------------------------------------------------------- #
def polygon_area_m2(vertices: Sequence[GeoPoint]) -> float:
    """Shoelace area of a small polygon, projected into the local NED plane."""
    if len(vertices) < 3:
        return 0.0
    origin = vertices[0]
    pts = [wgs84_to_local(v, origin) for v in vertices]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % len(pts)][0], pts[(i + 1) % len(pts)][1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def point_in_polygon(north: float, east: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test on local (north, east) coordinates."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > east) != (yj > east)) and (
            north < (xj - xi) * (east - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def bounding_box(vertices: Sequence[GeoPoint]) -> Tuple[GeoPoint, GeoPoint]:
    """(south-west, north-east) corner of the axis-aligned bbox of ``vertices``."""
    lats = [v.lat for v in vertices]
    lons = [v.lon for v in vertices]
    alts = [v.alt for v in vertices]
    return (
        GeoPoint(min(lats), min(lons), min(alts)),
        GeoPoint(max(lats), max(lons), max(alts)),
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_deg(angle: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def angle_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference ``a - b`` in degrees, in [-180, 180)."""
    return normalize_deg(a - b)
