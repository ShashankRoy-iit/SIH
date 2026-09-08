"""Pixel -> geodetic, with an honest uncertainty budget.

Every survivor the system reports is a *place*, and a place with no error bar is
not actionable: a rescue boat sent to a 300 m error ellipse searches an area
100x larger than one sent to a 30 m ellipse.  So this module treats geo-tagging
as an estimation problem rather than a coordinate transform, and propagates every
error source it can see:

=========================  =====================================================
platform position          GNSS fix quality, or dead-reckoning drift since the
                           last fix.  In GPS denial this term dominates.
attitude                   AHRS/EKF roll-pitch-yaw sigma.  A 0.5 deg attitude
                           error at 60 m slant range is 0.52 m on the ground -
                           and it grows linearly with range.
altitude above ground      baro + rangefinder + terrain model disagreement.
terrain height uncertainty  we do not have a DEM of a post-disaster site, so the
                           ground plane assumption itself carries error, and it
                           couples into lateral position through the off-nadir
                           angle.
pixel quantisation         half a pixel times the GSD.
timestamp / latency        the frame was captured some milliseconds before the
                           pose we paired it with; at 12 m/s that is metres.
=========================  =====================================================

The output is a :class:`GeoTag` carrying a 2x2 horizontal covariance, so the
tracker can fuse tags with correct weighting and the command centre can draw a
real error ellipse instead of a pin.

Nadir versus oblique
--------------------
At nadir the altitude error barely affects lateral position, which is why the
survey passes fly nadir.  Off-nadir, altitude error and terrain error project
into lateral error through ``tan(theta)``.  The tagger computes this exactly, so
the planner can be shown why confirmation orbits should be flown low *and*
level.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from sar.core.frames import euler_to_dcm
from sar.core.geo import GeoPoint, local_to_wgs84

__all__ = ["GeoTag", "NavQuality", "PixelGeoTagger", "GeoTagError"]


class GeoTagError(RuntimeError):
    """Raised when a pixel cannot be projected to the ground."""


@dataclass
class NavQuality:
    """How much to trust the navigation solution that produced a pose.

    This is the bridge between the navigation filter and the geo-tagger.  The
    two numbers that matter are the horizontal position sigma and the *age* of
    the last absolute fix, because in GPS denial the error grows with time since
    the fix rather than being constant.
    """

    pos_sigma_m: float = 1.2          # 1-sigma horizontal position error
    vel_sigma_ms: float = 0.25        # 1-sigma velocity error
    att_sigma_deg: float = 0.8        # 1-sigma roll/pitch/yaw error
    alt_sigma_m: float = 1.0          # 1-sigma height-above-ground error
    seconds_since_fix: float = 0.0    # 0 = fresh GNSS fix
    n_satellites: int = 34
    source: str = "gnss"              # gnss | ekf_external_nav | dead_reckoning | vision
    hdop: float = 0.7

    #: Dead-reckoning drift rate when no absolute fix is available [m/s].
    #: A visual-inertial EKF3 with external nav holds ~0.15 m/s; a pure
    #: IMU/air-data dead reckoning on a small multirotor drifts far faster.
    drift_rate_ms: float = 0.15

    @property
    def denied(self) -> bool:
        return self.source != "gnss" or self.seconds_since_fix > 3.0

    def effective_pos_sigma(self) -> float:
        """Position sigma including dead-reckoning drift since the last fix."""
        drift = self.drift_rate_ms * max(self.seconds_since_fix, 0.0)
        return float(math.hypot(self.pos_sigma_m, drift))

    def describe(self) -> str:
        return (f"{self.source} sats={self.n_satellites} hdop={self.hdop:.1f} "
                f"pos_sigma={self.effective_pos_sigma():.2f}m "
                f"since_fix={self.seconds_since_fix:.1f}s")


@dataclass
class GeoTag:
    """A detection pinned to the Earth, with its error ellipse."""

    point: GeoPoint
    sigma_north_m: float
    sigma_east_m: float
    #: Local tangent-plane coordinates in metres from ``PixelGeoTagger.origin``.
    #: Carried alongside the geodetic point because every downstream consumer -
    #: tracker, hazard map, planner - works in metres, and round-tripping through
    #: lat/lon per detection is both slower and lossier.
    north_m: float = 0.0
    east_m: float = 0.0
    covariance: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
    #: Ground sample distance at the target, m/px.  Determines the floor on how
    #: precisely anything can be located from this frame.
    gsd_m: float = 0.0
    slant_range_m: float = 0.0
    off_nadir_deg: float = 0.0
    terrain_z_m: float = 0.0
    t: float = 0.0
    #: Which error source dominates.  Surfaced in the dashboard because it tells
    #: the operator what to fix ("fly lower", "wait for GNSS", "re-zero the EKF").
    dominant_term: str = ""
    terms: Dict[str, float] = field(default_factory=dict)
    nav: Optional[NavQuality] = None

    # ------------------------------------------------------------------ #
    @property
    def sigma_m(self) -> float:
        """Total 1-sigma horizontal radius (isotropic equivalent)."""
        return float(math.sqrt(max(self.sigma_north_m ** 2 + self.sigma_east_m ** 2, 0.0)))

    @property
    def cep_m(self) -> float:
        """Circular error probable: radius of the 50% containment circle."""
        return 0.7071 * self.sigma_m

    @property
    def r95_m(self) -> float:
        """Radius of the 95% containment circle (2.45 sigma for 2-D Gaussian)."""
        return 2.4477 * self.sigma_m

    def ellipse(self, n_points: int = 48, confidence: float = 2.4477
                ) -> Sequence[GeoPoint]:
        """Polygon of the error ellipse, for drawing on the command centre map."""
        cov = np.asarray(self.covariance, dtype=float)
        if not np.all(np.isfinite(cov)) or cov.sum() <= 0:
            return [self.point]
        try:
            w, V = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:  # pragma: no cover
            return [self.point]
        w = np.maximum(w, 0.0)
        a, b = confidence * math.sqrt(w[1]), confidence * math.sqrt(w[0])
        ang = math.atan2(V[1, 1], V[0, 1])
        out = []
        for i in range(n_points):
            th = 2.0 * math.pi * i / n_points
            dn = a * math.cos(th) * math.cos(ang) - b * math.sin(th) * math.sin(ang)
            de = a * math.cos(th) * math.sin(ang) + b * math.sin(th) * math.cos(ang)
            out.append(self.point.offset(dn, de))
        return out

    def contains(self, other: GeoPoint, k: float = 2.0) -> bool:
        return self.point.distance_to(other) <= k * self.sigma_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.point.lat, "lon": self.point.lon, "alt": self.point.alt,
            "north_m": round(self.north_m, 2), "east_m": round(self.east_m, 2),
            "sigma_north_m": round(self.sigma_north_m, 3),
            "sigma_east_m": round(self.sigma_east_m, 3),
            "sigma_m": round(self.sigma_m, 3),
            "r95_m": round(self.r95_m, 2),
            "gsd_m": round(self.gsd_m, 4),
            "slant_range_m": round(self.slant_range_m, 2),
            "off_nadir_deg": round(self.off_nadir_deg, 2),
            "dominant_term": self.dominant_term,
            "nav_source": self.nav.source if self.nav else None,
            "nav_denied": self.nav.denied if self.nav else None,
            "t": self.t,
        }


# --------------------------------------------------------------------------- #
class PixelGeoTagger:
    """Projects pixel coordinates onto the ground and propagates uncertainty.

    Parameters
    ----------
    width, height : int
        Sensor resolution in pixels.
    hfov_deg : float
        Horizontal field of view.  Focal length in pixels is derived from it, so
        a lens change is a config change and not a code change.
    mount_pitch_deg : float
        0 = straight down (nadir survey), 90 = straight ahead.
    mount_roll_deg : float
        Roll of the camera relative to the vehicle body.
    origin : GeoPoint
        Local tangent-plane origin.  Everything internal is in metres NED from
        here, which keeps the arithmetic well conditioned.
    terrain_fn : callable
        ``(north, east) -> ground height above datum``.  In simulation this is
        the world's terrain; in flight it is a coarse DEM plus the rangefinder.
    terrain_sigma_m : float
        How much to trust ``terrain_fn``.  A post-disaster site has no DEM at
        all, so this is set from the difference between the rangefinder and the
        model, and it is one of the largest terms off-nadir.
    """

    def __init__(self, width: int, height: int, hfov_deg: float,
                 origin: GeoPoint,
                 mount_pitch_deg: float = 0.0, mount_roll_deg: float = 0.0,
                 terrain_fn: Optional[Callable[[float, float], float]] = None,
                 terrain_sigma_m: float = 1.5,
                 latency_s: float = 0.06,
                 sensor_offset_down_m: float = 0.12) -> None:
        self.width = int(width)
        self.height = int(height)
        self.hfov_deg = float(hfov_deg)
        self.origin = origin
        self.mount_pitch = math.radians(mount_pitch_deg)
        self.mount_roll = math.radians(mount_roll_deg)
        self.terrain_fn = terrain_fn or (lambda n, e: 0.0)
        self.terrain_sigma_m = float(terrain_sigma_m)
        self.latency_s = float(latency_s)
        #: The sensor hangs below the CG on a gimbal or a 3D-printed mount.  Half
        #: a percent of AGL at 35 m, which is 0.3 m of lateral error at the frame
        #: edge - worth modelling because it is a known, fixed number.
        self.sensor_offset_down_m = float(sensor_offset_down_m)
        self.focal_px = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        # ---------------------------------------------------------------
        # Camera frame convention, matching CameraRenderer.camera_dcm exactly:
        #   X_cam = body +Y   (image right = starboard)
        #   Y_cam = body -X   (image down  = aft, so image up = nose)
        #   Z_cam = body +Z   (optical axis straight down at mount_pitch 0)
        # with the mount tilt about X_cam and the mount roll about Z_cam applied
        # on top.  An earlier version took camera == body, which rotates every
        # ray 90 deg about the optical axis: dead accurate at the frame centre
        # and systematically wrong everywhere else (12 m of bias at the edge of a
        # 35 m survey pass, which is exactly what showed up as a constant error
        # across survivors at different off-nadir angles).
        # ---------------------------------------------------------------
        r_nadir = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        cp, sp_ = math.cos(self.mount_pitch), math.sin(self.mount_pitch)
        cr, sr = math.cos(self.mount_roll), math.sin(self.mount_roll)
        r_tilt = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp_], [0.0, -sp_, cp]])
        r_roll = np.array([[cr, sr, 0.0], [-sr, cr, 0.0], [0.0, 0.0, 1.0]])
        self._dcm_cam_body = r_tilt @ r_roll @ r_nadir

    # ------------------------------------------------------------------ #
    def gsd_at(self, agl_m: float) -> float:
        """Nadir ground sample distance [m/px] at a given AGL."""
        return float(agl_m / self.focal_px)

    def ray(self, u: float, v: float, dcm_body_ned: np.ndarray) -> np.ndarray:
        """Unit viewing ray in NED for pixel ``(u, v)``."""
        dx = (float(u) - self.width / 2.0) / self.focal_px
        dy = (float(v) - self.height / 2.0) / self.focal_px
        cam = np.array([dx, dy, 1.0])
        cam /= np.linalg.norm(cam)
        body = self._dcm_cam_body.T @ cam
        ned = dcm_body_ned @ body
        n = np.linalg.norm(ned)
        return ned / n if n > 1e-9 else ned

    def off_nadir_deg(self, u: float, v: float) -> float:
        """Angle between this pixel's ray and the optical axis, in degrees."""
        dx = (float(u) - self.width / 2.0) / self.focal_px
        dy = (float(v) - self.height / 2.0) / self.focal_px
        return float(math.degrees(math.atan(math.hypot(dx, dy))))

    # ------------------------------------------------------------------ #
    def tag(self, u: float, v: float, pos_ned: np.ndarray, euler: np.ndarray,
            velocity_ned: Optional[np.ndarray] = None,
            nav: Optional[NavQuality] = None, t: Optional[float] = None,
            agl_override_m: Optional[float] = None) -> GeoTag:
        """Geo-tag pixel ``(u, v)`` captured from ``pos_ned`` with attitude ``euler``.

        Raises :class:`GeoTagError` if the ray does not meet the ground (looking
        above the horizon), which happens for oblique mounts near the frame edge
        and must be handled by the caller rather than silently extrapolated.
        """
        nav = nav or NavQuality()
        dcm = euler_to_dcm(float(euler[0]), float(euler[1]), float(euler[2]))
        ray = self.ray(u, v, dcm)

        # Iterate the ray/plane intersection against the terrain height: one pass
        # assumes a flat plane at the nadir ground height, which is wrong by
        # metres over a 12 m building or a riverbank.  Two or three passes
        # converge for anything a small UAV actually flies over.
        north, east = float(pos_ned[0]), float(pos_ned[1])
        ground_z = float(self.terrain_fn(north, east))
        # Sensor, not CG: the renderer hangs the camera below the vehicle.
        cam_z = float(pos_ned[2]) + self.sensor_offset_down_m
        if agl_override_m is not None:
            cam_h = float(agl_override_m)
        else:
            cam_h = float(-cam_z - ground_z)
        for _ in range(3):
            # NED: +z is DOWN.  A ray reaches the ground only if its z component
            # is positive.  An earlier version tested the opposite sign, which is
            # the natural mistake when thinking in ENU, and it rejected every
            # single pixel of every nadir frame.
            if ray[2] <= 1e-6:
                raise GeoTagError(
                    f"pixel ({u:.1f},{v:.1f}) ray points at or above the horizon "
                    f"(ray_z={ray[2]:+.4f}); cannot intersect the ground plane")
            cam_h = max(float(-cam_z - ground_z), 0.2)
            s = cam_h / ray[2]
            north = float(pos_ned[0] + ray[0] * s)
            east = float(pos_ned[1] + ray[1] * s)
            ground_z = float(self.terrain_fn(north, east))
        slant = float(cam_h / ray[2])

        # ---------------- uncertainty budget -------------------------------
        # Attitude: an angular error swings the ray about the camera, so the
        # ground displacement is range x angle.  Decomposed into the component
        # that moves the point cross-track (roll/yaw) and along-track (pitch).
        att = math.radians(nav.att_sigma_deg)
        sig_att_cross = slant * att
        sig_att_along = slant * att * max(math.tan(self.off_nadir_deg(u, v) * math.pi / 180.0),
                                          1e-3)

        # Height: for an off-nadir ray, a height error moves the ground point by
        # h_err * tan(theta).  At nadir it barely moves it laterally at all.
        tan_off = math.tan(math.radians(self.off_nadir_deg(u, v)))
        sig_alt_along = nav.alt_sigma_m * tan_off
        sig_terrain_along = self.terrain_sigma_m * tan_off

        # Pixel quantisation and lens distortion residual.  GSD scales with
        # slant range, not with AGL: at 30 deg off-nadir the same pixel covers
        # 15% more ground, and the renderer reports it that way too.
        gsd = slant / self.focal_px
        sig_pixel = 0.5 * gsd
        sig_distortion = 0.35 * gsd          # uncalibrated lens, 1/3 px

        # Latency: the pose is sampled when the frame is *processed*, not when it
        # was *captured*.  At 12 m/s a 60 ms pipeline latency is 0.72 m.
        vel = velocity_ned if velocity_ned is not None else np.zeros(3)
        sig_latency_n = abs(float(vel[0])) * self.latency_s
        sig_latency_e = abs(float(vel[1])) * self.latency_s

        # Platform position.  In GNSS denial this grows with time since the fix.
        sig_pos = nav.effective_pos_sigma()

        # Combine in quadrature.  "Along" is the direction of the off-nadir lean,
        # which for a nadir survey camera is arbitrary per pixel, so the two
        # horizontal axes are treated symmetrically except for latency, which is
        # genuinely anisotropic (along-track).
        radial = math.hypot(sig_att_cross, sig_alt_along, sig_terrain_along,
                            sig_pixel, sig_distortion, sig_att_along)
        sig_north = float(math.hypot(radial, sig_pos, sig_latency_n))
        sig_east = float(math.hypot(radial, sig_pos, sig_latency_e))

        terms = {
            "platform_pos": sig_pos,
            "attitude": sig_att_cross,
            "altitude_offnadir": sig_alt_along,
            "terrain_model": sig_terrain_along,
            "pixel_quantisation": sig_pixel,
            "lens_distortion": sig_distortion,
            "latency": math.hypot(sig_latency_n, sig_latency_e),
        }
        dominant = max(terms, key=lambda k: terms[k])

        # Off-nadir lean direction, to build a real (not axis-aligned) covariance.
        dx = float(u) - (self.width - 1) / 2.0
        dy = float(v) - (self.height - 1) / 2.0
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            en, ee = dx / norm, dy / norm
        else:
            en, ee = 1.0, 0.0
        iso = math.hypot(sig_pixel, sig_distortion, sig_pos)
        a_along = math.hypot(iso, radial)
        a_cross = iso
        cov = np.array([[a_cross ** 2, 0.0], [0.0, a_cross ** 2]])
        R = np.array([[en, -ee], [ee, en]])
        cov = R @ np.array([[a_along ** 2, 0.0], [0.0, a_cross ** 2]]) @ R.T

        point = local_to_wgs84(north, east, -ground_z, self.origin)
        return GeoTag(
            point=GeoPoint(point.lat, point.lon, ground_z),
            sigma_north_m=sig_north, sigma_east_m=sig_east,
            north_m=north, east_m=east,
            covariance=((float(cov[0, 0]), float(cov[0, 1])),
                        (float(cov[1, 0]), float(cov[1, 1]))),
            gsd_m=gsd, slant_range_m=slant,
            off_nadir_deg=self.off_nadir_deg(u, v), terrain_z_m=ground_z,
            t=float(t if t is not None else time.time()),
            dominant_term=dominant, terms=terms, nav=nav,
        )

    # ------------------------------------------------------------------ #
    def tag_with_offset(self, u: float, v: float, pos_ned: np.ndarray,
                        euler: np.ndarray, **kwargs: Any) -> GeoTag:
        """Same as :meth:`tag` but never raises; returns a NaN tag on failure.

        Useful inside per-frame loops where a horizon-crossing pixel is normal
        and should not abort the frame.
        """
        try:
            return self.tag(u, v, pos_ned, euler, **kwargs)
        except GeoTagError:
            nan = GeoPoint(float("nan"), float("nan"), float("nan"))
            return GeoTag(point=nan, sigma_north_m=float("inf"),
                          sigma_east_m=float("inf"),
                          north_m=float("nan"), east_m=float("nan"),
                          dominant_term="no_ground_intersection",
                          nav=kwargs.get("nav"))

    # ------------------------------------------------------------------ #
    def monte_carlo_sigma(self, u: float, v: float, pos_ned: np.ndarray,
                          euler: np.ndarray, nav: NavQuality,
                          samples: int = 400, seed: int = 0) -> Tuple[float, float]:
        """Empirical check of the analytic budget.

        Perturbs position, attitude, altitude and terrain, re-projects, and
        returns the measured standard deviations.  Used by
        ``tests/test_geotag.py`` to prove the closed-form propagation is not
        optimistic - which is the failure mode that matters, since an
        under-reported error ellipse sends rescuers to the wrong place with
        false confidence.
        """
        rng = np.random.default_rng(seed)
        pts = []
        for _ in range(samples):
            p = np.asarray(pos_ned, dtype=float).copy()
            p[0] += rng.normal(0, nav.effective_pos_sigma())
            p[1] += rng.normal(0, nav.effective_pos_sigma())
            p[2] += rng.normal(0, nav.alt_sigma_m)
            e = np.asarray(euler, dtype=float).copy()
            e += rng.normal(0, math.radians(nav.att_sigma_deg), 3)
            tf = self.terrain_fn

            def noisy_terrain(n, ee, _tf=tf, _s=self.terrain_sigma_m):
                return float(_tf(n, ee)) + rng.normal(0, _s)

            self.terrain_fn = noisy_terrain
            try:
                g = self.tag(u, v, p, e, nav=nav)
                pts.append((g.point.lat, g.point.lon))
            except GeoTagError:
                pass
            finally:
                self.terrain_fn = tf
        if len(pts) < 8:
            return float("nan"), float("nan")
        arr = np.array(pts)
        m_per_deg_lat, m_per_deg_lon = 111_320.0, 111_320.0 * math.cos(
            math.radians(float(arr[:, 0].mean())))
        dn = (arr[:, 0] - arr[:, 0].mean()) * m_per_deg_lat
        de = (arr[:, 1] - arr[:, 1].mean()) * m_per_deg_lon
        return float(dn.std()), float(de.std())
