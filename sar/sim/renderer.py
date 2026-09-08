"""Physics-aware synthetic camera: renders RGB + LWIR frames from the world.

Why a hand-written renderer instead of Gazebo/AirSim?

* The problem statement demands *on-device* AI in a communication-denied
  disaster.  What must be tested is the **perception-to-decision chain**, not
  photorealism.  A deterministic analytic renderer gives exact ground truth
  (so precision/recall are measurable), runs headless at ~20 fps on a 2-core
  laptop, and lets us dial in conditions - sun angle, cloud, rain, smoke,
  inundation - that would take weeks to wait for in the field.
* It still models the physics that decide whether a detector works: emissivity,
  reflected sky radiance, atmospheric transmission, solar loading, ground
  sample distance vs target size, occlusion by structures, and the collapse of
  LWIR contrast for a person standing in cold flood water.

Camera model
------------
Pinhole with a gimbal mount.  ``mount_pitch_deg = 0`` is straight down (nadir,
used for survey) and ``90`` is straight ahead.  Pixel -> world uses exact
ray/plane intersection against a height-corrected ground plane, so oblique
views show correct parallax: a 12 m building leans away from the nadir point.

Outputs
-------
``Frame.image``
    * LWIR: ``float32`` in **degrees Celsius** (radiometric).  Detectors may
      internally quantise to 8-bit AGP, but radiometry is preserved so the
      survivor-viability estimator can read real temperatures.
    * RGB: ``uint8`` ``(H, W, 3)``.
``Frame.truth``
    Ground-truth objects geometrically visible in this frame with projected
    pixel boxes and a physics-based detection ceiling (TTP model).  Used by the
    evaluation harness only - the perception stack never sees it.

Note on resolution
------------------
The simulation renders at 320x240 by default where the flight sensor is
640x512.  Measured detection performance is therefore a **conservative lower
bound** on the real payload: at equal field of view the simulated target
occupies a quarter of the pixels.  Set ``resolution_scale`` in
``config/perception.yaml`` to 1.0 for a faithful-but-slower render.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.frames import euler_to_dcm
from sar.sim.world import (
    HUMAN_THERMAL_BAND,
    DisasterWorld,
    Distractor,
    Victim,
    VictimPosture,
    solar_loading,
)
from sar.vehicle.dynamics import PlantState

# Emissivity by surface type (LWIR 8-14 um).
EMISSIVITY = {
    "soil": 0.95,
    "vegetation": 0.97,
    "water": 0.98,
    "concrete": 0.93,
    "brick": 0.93,
    "wood": 0.92,
    "metal": 0.28,      # low emissivity -> reflects cold sky -> reads COLD
    "human_skin": 0.98,
    "asphalt": 0.94,
    "rubble": 0.92,
}

# Base sRGB colour per surface type.
RGB_BASE = {
    "soil": (128, 106, 78),
    "vegetation": (66, 96, 48),
    "water": (48, 68, 84),
    "concrete": (168, 166, 160),
    "brick": (154, 96, 78),
    "wood": (140, 112, 78),
    "metal": (150, 158, 166),
    "asphalt": (96, 96, 98),
    "rubble": (142, 128, 112),
    "human_clothing": (214, 62, 58),
}

# ---------------------------------------------------------------------------
# Material classification uses integer codes + LUTs rather than object-dtype
# string arrays: the string version cost ~90 ms per 320x240 frame, the LUT
# version ~2 ms.  The material semantics are identical.
# ---------------------------------------------------------------------------
MAT_SOIL, MAT_VEGETATION, MAT_RUBBLE, MAT_ASPHALT, MAT_WATER, MAT_CONCRETE, \
    MAT_METAL_ROOF = range(7)

EPS_LUT = np.array([
    EMISSIVITY["soil"], EMISSIVITY["vegetation"], EMISSIVITY["rubble"],
    EMISSIVITY["asphalt"], EMISSIVITY["water"], EMISSIVITY["concrete"],
    EMISSIVITY["metal"],
], dtype=np.float32)

RGB_LUT = np.array([
    RGB_BASE["soil"], RGB_BASE["vegetation"], RGB_BASE["rubble"],
    RGB_BASE["asphalt"], RGB_BASE["water"], RGB_BASE["concrete"],
    RGB_BASE["metal"],
], dtype=np.float32)

#: Fine-scale thermal/visual clutter sigma [K] on a nominally smooth surface.
#: Calibrated so that open water stays within ~0.05 K (a real flood sheet is a
#: mirror-flat radiator) while rubble and canopy reach ~2 K, which is the regime
#: where thermal false alarms actually come from.
CLUTTER_SIGMA_K = 0.85

#: Layer stack order used by :meth:`DisasterWorld.sample_stack`.  Sampling many
#: layers through one shared :class:`~sar.sim.world.GridIndex` is what makes
#: real-time synthetic imaging affordable; adding a layer here costs ~1 ms/frame.
_STACK = ("surface_temp", "structure_height", "damage", "flood", "vegetation",
          "debris", "fire", "smoke", "road", "metal_roof", "elevation", "albedo",
          "texture")
_S = {name: i for i, name in enumerate(_STACK)}


def classify_material(height: np.ndarray, damage: np.ndarray, water: np.ndarray,
                      veg: np.ndarray, debris: np.ndarray, road: np.ndarray,
                      metal_roof: np.ndarray) -> np.ndarray:
    """Per-pixel surface material code (vectorised, integer)."""
    idx = np.zeros(height.shape, dtype=np.int8)
    idx = np.where(veg > 0.35, MAT_VEGETATION, idx)
    idx = np.where(debris > 0.35, MAT_RUBBLE, idx)
    idx = np.where(road > 0.5, MAT_ASPHALT, idx)
    idx = np.where(height > 0.5,
                   np.where(metal_roof > 0.5, MAT_METAL_ROOF,
                            np.where(damage > 0.6, MAT_RUBBLE, MAT_CONCRETE)), idx)
    # Open water wins over everything: it is the visible surface.
    idx = np.where(water > 0.08, MAT_WATER, idx)
    return idx


@dataclass
class CameraSpec:
    """Sensor description.

    Defaults match the reference payload in ``config/perception.yaml``: a
    640x512-class uncooled LWIR microbolometer core (simulated at 320x240) and
    a 1080p-class RGB camera.
    """

    name: str = "lwir"
    kind: str = "lwir"                       # 'lwir' | 'rgb'
    width: int = 320
    height: int = 240
    hfov_deg: float = 42.0
    mount_pitch_deg: float = 0.0             # 0 = straight down, 90 = forward
    mount_roll_deg: float = 0.0
    netd_mk: float = 45.0                    # LWIR noise-equivalent delta-T
    rgb_noise_dn: float = 3.0
    fps: float = 8.0
    haze_k: float = 0.0016                   # RGB extinction [1/m]
    lwir_extinction_k: float = 0.00035       # LWIR extinction [1/m]
    atm_temp_c: float = 22.0
    max_range_m: float = 400.0
    native_width: int = 640                  # real sensor resolution (metadata)
    native_height: int = 512

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def resolution_scale(self) -> float:
        """Simulated pixels / real sensor pixels (documented conservatism)."""
        return self.width / max(self.native_width, 1)

    def gsd_at(self, slant_range_m: float) -> float:
        """Ground sample distance [m/px] for a nadir view at this range."""
        return slant_range_m / self.focal_px

    def swath_width_m(self, agl_m: float) -> float:
        """Width of the nadir footprint at height ``agl_m``."""
        return 2.0 * agl_m * math.tan(math.radians(self.hfov_deg) / 2.0)


@dataclass
class CameraPose:
    position: np.ndarray        # NED [m]
    euler: np.ndarray           # vehicle roll, pitch, yaw [rad]
    dcm_cam_ned: np.ndarray     # 3x3, NED -> camera
    altitude_agl: float
    t: float


@dataclass
class TruthObject:
    """Ground truth for one object in one frame."""

    oid: str
    category: str               # 'victim' | 'distractor' | 'hazard' | 'structure'
    label: str                  # detector taxonomy label
    pixel: Tuple[float, float]
    width_px: float             # bounding-box width  (2a)
    height_px: float            # bounding-box height (2b)
    distance_m: float
    world_ned: Tuple[float, float, float]
    apparent_temp_c: Optional[float] = None
    background_temp_c: Optional[float] = None
    occlusion: float = 0.0
    visible: bool = True
    #: Apparent area actually covered by the object, in pixels.  This is the
    #: ellipse area ``pi*a*b`` - *not* ``width_px * height_px``, which is the
    #: bounding box and overstates an ellipse by 4/pi = 1.27x.  Getting this
    #: wrong scales the TTP size term by ~1.9x in equivalent pixels, which is
    #: the difference between "detectable" and "not" at survey altitude.
    area_px: float = 0.0
    area_m2: float = 0.0
    gsd_m: float = 0.0
    attributes: Dict[str, object] = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        u, v = self.pixel
        return (u - self.width_px / 2.0, v - self.height_px / 2.0,
                u + self.width_px / 2.0, v + self.height_px / 2.0)

    @property
    def bbox_area_px(self) -> float:
        return max(self.width_px, 0.0) * max(self.height_px, 0.0)

    @property
    def equivalent_pixels(self) -> float:
        """Side of the equal-area square: 'target size in pixels' for TTP."""
        a = self.area_px if self.area_px > 0 else self.bbox_area_px
        return float(math.sqrt(max(a, 0.0)))

    @property
    def thermal_contrast_k(self) -> float:
        if self.apparent_temp_c is None or self.background_temp_c is None:
            return 0.0
        return abs(self.apparent_temp_c - self.background_temp_c)

    def ttp_detection(self, n50: float = 2.0) -> float:
        """Probability an ideal sensor+observer detects this target.

        Standard Targeting Task Performance function (Driggers et al.), the
        modern replacement for the Johnson criteria::

            E = 1.5074 + 0.2477 (n / n50)
            P = (n/n50)^E / (1 + (n/n50)^E)

        ``n50 = 2.0`` equivalent pixels is the classic detection threshold for
        a human-shaped target.  This is the physically-implied ceiling on
        probability-of-detection (POD) consumed by the search-theory layer, so
        the planner is never credited with finding a target that was not
        resolvable.
        """
        ratio = self.equivalent_pixels / max(n50, 1e-6)
        if ratio <= 0.0:
            return 0.0
        e = 1.5074 + 0.2477 * ratio
        return float(ratio ** e / (1.0 + ratio ** e))

    @property
    def detectability(self) -> float:
        """POD ceiling = TTP(size) x contrast x fragmentation.

        Occlusion enters **once**, through ``area_px``: the renderer shrinks a
        survivor's apparent area by occlusion, so the TTP size term is already
        the occluded one.  Multiplying by a second ``(1 - occ)`` term here - as
        an earlier version did - double-counted it and pushed in-water survivors
        to a POD of 0.2 at 35 m, well below what the rendered frame supports.

        What occlusion *does* still cost beyond lost area is silhouette
        fragmentation: a body seen through gaps in canopy or rubble is harder to
        recognise than a solid blob of the same area.  That residual is modelled
        as a deliberately weak term.
        """
        size_term = self.ttp_detection()
        occ = float(np.clip(self.occlusion, 0.0, 1.0))
        frag_term = 1.0 - 0.35 * occ ** 1.2
        if self.apparent_temp_c is not None:
            # Below ~1 K contrast a person is lost in LWIR noise; above ~6 K
            # contrast stops being the limiting factor.
            contrast_term = float(np.clip((self.thermal_contrast_k - 1.0) / 5.0, 0.05, 1.0))
        else:
            contrast_term = 0.9
        return float(np.clip(size_term * frag_term * contrast_term, 0.0, 1.0))

    @property
    def in_band(self) -> bool:
        if self.apparent_temp_c is None:
            return False
        return HUMAN_THERMAL_BAND[0] <= self.apparent_temp_c <= HUMAN_THERMAL_BAND[1]


@dataclass
class Frame:
    t: float
    camera: str
    kind: str
    image: np.ndarray
    pose: CameraPose
    truth: List[TruthObject] = field(default_factory=list)
    gsd_m: float = 0.0
    footprint: Tuple[Tuple[float, float], ...] = ()


# --------------------------------------------------------------------------- #
class CameraRenderer:
    """Renders :class:`Frame` objects from a :class:`DisasterWorld`."""

    def __init__(self, world: DisasterWorld, spec: CameraSpec,
                 sun_azimuth_deg: float = 160.0,
                 sun_elevation_deg: Optional[float] = None,
                 seed: Optional[int] = None) -> None:
        self.world = world
        self.spec = spec
        self._u, self._v = np.meshgrid(
            np.arange(spec.width, dtype=np.float32),
            np.arange(spec.height, dtype=np.float32),
            indexing="xy",
        )
        self.sun_az = math.radians(sun_azimuth_deg)
        if sun_elevation_deg is None:
            sun_elevation_deg = 12.0 + 66.0 * solar_loading(
                world.spec.time_of_day_h, world.spec.cloud_cover)
        self.sun_el = math.radians(sun_elevation_deg)
        # NED direction pointing from the surface towards the sun (up = -Z).
        self.sun_dir = np.array([
            math.cos(self.sun_el) * math.cos(self.sun_az),
            math.cos(self.sun_el) * math.sin(self.sun_az),
            -math.sin(self.sun_el),
        ])
        if seed is None:
            seed = abs(hash(spec.name + world.spec.name)) % (2 ** 31)
        self._rng = np.random.default_rng(seed)
        self._d_cam_cache = self._make_direction_grid()

    # ------------------------------------------------------------------ #
    def _make_direction_grid(self) -> np.ndarray:
        """Unit ray directions in camera coordinates, shape (3, H*W)."""
        sp = self.spec
        f = sp.focal_px
        x = (self._u - sp.width / 2.0) / f
        y = (self._v - sp.height / 2.0) / f
        ones = np.ones_like(x)
        return np.stack([x, y, ones], axis=0).reshape(3, -1)

    def camera_dcm(self, vehicle_euler: np.ndarray) -> np.ndarray:
        """NED -> camera rotation matrix for the gimbal-mounted sensor."""
        r_b2n = euler_to_dcm(*vehicle_euler)
        # Nadir reference frame:
        #   X_cam = body +Y   (image right  = starboard)
        #   Y_cam = body -X   (image down   = aft, so image up = nose)
        #   Z_cam = body +Z   (optical axis = straight down)
        r_nadir = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        tp = math.radians(self.spec.mount_pitch_deg)
        tr = math.radians(self.spec.mount_roll_deg)
        cp, sp_ = math.cos(tp), math.sin(tp)
        cr, sr = math.cos(tr), math.sin(tr)
        r_tilt = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp_], [0.0, -sp_, cp]])
        r_roll = np.array([[cr, sr, 0.0], [-sr, cr, 0.0], [0.0, 0.0, 1.0]])
        return (r_tilt @ r_roll @ r_nadir) @ r_b2n.T

    def pose(self, st: PlantState, t: float, ground_z: float) -> CameraPose:
        pos = st.pos.copy()
        pos[2] = pos[2] + 0.12          # sensor sits slightly below the CG
        dcm = self.camera_dcm(st.euler)
        return CameraPose(position=pos, euler=st.euler.copy(), dcm_cam_ned=dcm,
                          altitude_agl=-st.pos[2] - ground_z, t=t)

    # ------------------------------------------------------------------ #
    def pixel_to_ground(self, pose: CameraPose, ground_z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised pixel -> (north, east, slant_range) on a flat plane."""
        sp = self.spec
        d_ned = pose.dcm_cam_ned.T @ self._d_cam_cache
        dz = d_ned[2]
        plane = ground_z - pose.position[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(dz > 1e-6, plane / dz, np.nan)
        north = pose.position[0] + s * d_ned[0]
        east = pose.position[1] + s * d_ned[1]
        rng = np.abs(s) * np.linalg.norm(d_ned, axis=0)
        shp = (sp.height, sp.width)
        return north.reshape(shp), east.reshape(shp), rng.reshape(shp)

    def ground_to_pixel(self, pose: CameraPose, north: np.ndarray, east: np.ndarray,
                        up_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points into pixels -> (u, v, slant_range)."""
        sp = self.spec
        p = np.stack([np.asarray(north, float).ravel(),
                      np.asarray(east, float).ravel(),
                      -np.asarray(up_z, float).ravel()], axis=0)
        d = p - pose.position.reshape(3, 1)
        cam = pose.dcm_cam_ned @ d
        z = cam[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = sp.focal_px * cam[0] / z + sp.width / 2.0
            v = sp.focal_px * cam[1] / z + sp.height / 2.0
        rng = np.linalg.norm(d, axis=0)
        return np.where(z > 0.05, u, np.nan), np.where(z > 0.05, v, np.nan), rng

    # ------------------------------------------------------------------ #
    def render(self, st: PlantState, t: float, extra_light: float = 1.0) -> Frame:
        w = self.world
        sp = self.spec
        ground_z = float(np.nan_to_num(w.terrain_z(st.pos[0], st.pos[1])))
        pose = self.pose(st, t, ground_z)

        north, east, slant = self.pixel_to_ground(pose, ground_z)
        valid = np.isfinite(north) & np.isfinite(east) & (slant < sp.max_range_m)
        nn = np.nan_to_num(north)
        ee = np.nan_to_num(east)

        # Parallax correction for elevated surfaces: a roof of height h appears
        # displaced radially from the nadir point by h/(H-h).
        gi_probe = w.grid_index(nn, ee)
        h_obj = np.nan_to_num(w.building_height.sample_idx(gi_probe))
        cam_h = max(-pose.position[2] - ground_z, 1.0)
        scale = cam_h / np.maximum(cam_h - h_obj, 0.5)
        nn_c = pose.position[0] + (nn - pose.position[0]) * scale
        ee_c = pose.position[1] + (ee - pose.position[1]) * scale

        gi = w.grid_index(nn_c, ee_c)
        L = w.sample_stack(_STACK, gi)
        if sp.kind == "lwir":
            image = self._render_lwir(L, nn_c, ee_c, slant, valid, pose, t, gi)
        else:
            image = self._render_rgb(L, nn_c, ee_c, slant, valid, pose, t, extra_light)

        truth = self._build_truth(pose, t, ground_z)
        axis = pose.dcm_cam_ned.T @ np.array([0.0, 0.0, 1.0])
        # Height above the local ground plane in NED is -z_down - z_ground, not
        # z_ground - z_down: with terrain at +13 m this used to report a GSD of
        # 0.161 m/px instead of the true 0.096 m/px at 40 m AGL, a 1.7x error
        # that propagated into the detector's measurement aperture and into the
        # geo-tagging uncertainty budget.
        cam_h = max(-pose.position[2] - ground_z, 0.5)
        if axis[2] > 1e-6:
            gsd = float(cam_h / axis[2] / sp.focal_px)
        else:
            gsd = float("inf")
        return Frame(t=t, camera=sp.name, kind=sp.kind, image=image, pose=pose,
                     truth=truth, gsd_m=gsd, footprint=self._footprint(pose, ground_z))

    # ------------------------------------------------------------------ #
    def _render_lwir(self, L: np.ndarray, nn_c, ee_c, slant, valid, pose, t,
                     gi) -> np.ndarray:
        w = self.world
        sp = self.spec
        solar = solar_loading(w.spec.time_of_day_h, w.spec.cloud_cover)

        base_t = L[_S["surface_temp"]].astype(np.float64)
        texture = L[_S["texture"]]
        height = L[_S["structure_height"]]
        damage = L[_S["damage"]]
        water = L[_S["flood"]]
        veg = L[_S["vegetation"]]
        debris = L[_S["debris"]]
        fire = L[_S["fire"]]
        smoke = L[_S["smoke"]]
        road = L[_S["road"]]
        metal = L[_S["metal_roof"]]

        mat = classify_material(height, damage, water, veg, debris, road, metal)
        eps = EPS_LUT[mat].astype(np.float64)

        # Sub-cell thermal clutter.  Amplitude is material-dependent: open water
        # is a near-perfectly uniform radiator, while rubble, canopy and damaged
        # masonry are a patchwork of sunlit and shaded facets, wet and dry
        # grains, metal and stone.  This is the dominant source of realistic
        # false alarms and it must be in the simulator, or the measured
        # false-alarm rate describes our renderer instead of our detector.
        rough = (0.30 + 1.05 * texture + 1.10 * debris + 0.85 * veg
                 + 0.60 * damage - 1.05 * (water > 0.08))
        rough = np.clip(rough, 0.04, 2.4)
        base_t = base_t + w.micro_texture.sample(nn_c, ee_c) * rough * CLUTTER_SIGMA_K

        sky_t = 6.0 + 8.0 * w.spec.cloud_cover
        radiance_t = eps * base_t + (1.0 - eps) * sky_t

        # Water is thermally very stable: damp terrain texture so a flood sheet
        # reads as a smooth cold region - the cue the hazard classifier uses.
        radiance_t = np.where(water > 0.08,
                              0.6 * radiance_t + 0.4 * (12.0 + 5.0 * solar), radiance_t)
        # Open fire saturates the band.
        radiance_t = np.where(fire > 0.02,
                              np.maximum(radiance_t, 320.0 + 620.0 * fire), radiance_t)
        # Smouldering rubble beneath collapsed structures.
        radiance_t = np.where((damage > 0.55) & (height > 0.3),
                              radiance_t + 9.0 * damage, radiance_t)

        # Atmospheric transmission + path radiance (humidity rises over water).
        hum = 0.45 + 0.45 * float(np.clip(water.mean(), 0, 1)) + 0.3 * w.spec.rain
        tau = np.exp(-sp.lwir_extinction_k * (1.0 + 1.6 * hum) * slant)
        img = tau * radiance_t + (1.0 - tau) * sp.atm_temp_c

        # Smoke is semi-opaque in LWIR: attenuates AND radiates warm.
        smoke_t = sp.atm_temp_c + 14.0 * smoke
        img = (1.0 - 0.85 * smoke) * img + 0.85 * smoke * smoke_t

        img = np.where(valid, img, sp.atm_temp_c)
        img = self._stamp_instances_lwir(img, pose, t)

        netd_c = sp.netd_mk / 1000.0
        img += self._rng.normal(0.0, netd_c * 2.2, img.shape)
        img += self._rng.normal(0.0, netd_c * 3.0, (1, img.shape[1]))
        return img.astype(np.float32)

    def _stamp_instances_lwir(self, img: np.ndarray, pose: CameraPose, t: float) -> np.ndarray:
        w = self.world
        sp = self.spec
        items: List[Tuple[float, float, float, float, float, float]] = []
        for v in w.victims:
            items.append((v.north, v.east, v.elevation, v.body_temp_c,
                          _human_apparent_area_m2(v), 1.0 - v.occlusion))
        for d in w.distractors:
            items.append((d.north, d.east, d.elevation, d.temp_c,
                          math.pi * (d.size_m / 2.0) ** 2 * 0.6, 1.0))
        if not items:
            return img
        norths = np.array([i[0] for i in items])
        easts = np.array([i[1] for i in items])
        ups = np.array([i[2] for i in items])
        u, v, rng = self.ground_to_pixel(pose, norths, easts, ups)
        m_per_px = rng / sp.focal_px
        for k, (_n, _e, _z, temp, area, vis) in enumerate(items):
            if not np.isfinite(u[k]) or rng[k] > sp.max_range_m or rng[k] < 0.5:
                continue
            px_radius = math.sqrt(area / math.pi) / max(m_per_px[k], 1e-6)
            if px_radius < 0.15:
                continue
            tau = math.exp(-sp.lwir_extinction_k * rng[k] * 1.6)
            t_app = tau * temp + (1.0 - tau) * sp.atm_temp_c
            # Occlusion is already applied to ``area`` (see
            # _human_apparent_area_m2), so it must NOT also scale the amplitude:
            # doing both halved the contrast of every partly-hidden survivor.
            # Only the atmosphere attenuates the radiance that does reach us.
            _add_blob(img, u[k], v[k], px_radius, t_app, tau, _blob_aspect(area))
        return img

    # ------------------------------------------------------------------ #
    def _render_rgb(self, L: np.ndarray, nn, ee, slant, valid, pose, t, light) -> np.ndarray:
        w = self.world
        sp = self.spec
        solar = solar_loading(w.spec.time_of_day_h, w.spec.cloud_cover)

        albedo = L[_S["albedo"]].astype(np.float64)
        height = L[_S["structure_height"]]
        damage = L[_S["damage"]]
        water = L[_S["flood"]]
        veg = L[_S["vegetation"]]
        debris = L[_S["debris"]]
        road = L[_S["road"]]
        smoke = L[_S["smoke"]]
        fire = L[_S["fire"]]
        metal = L[_S["metal_roof"]]
        elev = L[_S["elevation"]]
        texture = L[_S["texture"]]

        mat = classify_material(height, damage, water, veg, debris, road, metal)
        rgb = RGB_LUT[mat].astype(np.float64) * (0.78 + 0.72 * albedo)[..., None]
        # Sub-cell visual texture: individual stones, leaves and rubble facets.
        # Scaled down over open water, which really is mirror-flat at this range.
        rough_rgb = np.clip(0.42 + 0.75 * texture + 0.85 * debris + 0.55 * veg
                            - 0.75 * (water > 0.08), 0.03, 1.7)
        # Neutral (achromatic) texture: real micro-facets change brightness and
        # shadow, not hue.  A red-biased term here made sunlit soil read as fire
        # to any chromaticity rule downstream.
        micro = w.micro_texture.sample(nn, ee)[..., None] * rough_rgb[..., None]
        rgb = rgb * (1.0 + 0.13 * micro) + 7.0 * micro

        # Lambertian shading from terrain slope (central differences).
        dn = np.pad(elev[2:, 1:-1] - elev[:-2, 1:-1], 1, mode="edge") * 0.5
        de = np.pad(elev[1:-1, 2:] - elev[1:-1, :-2], 1, mode="edge") * 0.5
        norm = np.sqrt(dn * dn + de * de + 1.0)
        cos_i = np.clip((self.sun_dir[0] * dn + self.sun_dir[1] * de - self.sun_dir[2]) / norm,
                        0.0, 1.0)
        shade = 0.42 + 1.05 * cos_i * (0.45 + 0.55 * solar) * light
        shade = shade * np.where(height > 0.5, 1.0 - 0.25 * damage, 1.0)
        rgb = rgb * shade[..., None]

        # Specular glint on open water - the dominant RGB false-positive source.
        glint = np.clip((water > 0.05) *
                        (0.5 + 0.5 * np.sin(nn * 0.9 + t * 1.7) * np.cos(ee * 1.1)), 0, 1) * solar
        rgb = rgb + np.array([60.0, 66.0, 72.0]) * glint[..., None]

        f = fire[..., None]
        rgb = rgb + np.concatenate([210.0 * f, 110.0 * f ** 1.4, 30.0 * f ** 2], axis=-1)

        tau = np.exp(-sp.haze_k * (1.0 + 2.2 * w.spec.rain) * slant)[..., None]
        sky = np.array([186.0, 196.0, 205.0]) * (0.5 + 0.5 * light)
        sm = (0.75 * smoke)[..., None]
        rgb = rgb * tau + (1.0 - tau) * sky
        rgb = rgb * (1.0 - sm) + sm * np.array([120.0, 116.0, 112.0])

        img = self._stamp_instances_rgb(rgb, pose, t)
        img = np.where(valid[..., None], img, 0.0)
        img += self._rng.normal(0.0, sp.rgb_noise_dn + 6.0 * (1.0 - light), img.shape)
        img = np.clip(img, 0, 255)

        # Automatic gain control: a real camera exposes to a target mid-grey,
        # which is what makes dusk frames usable - and what amplifies read noise.
        mean = float(img[valid].mean()) if valid.any() else float(img.mean())
        gain = float(np.clip(118.0 / max(mean, 1.0), 0.5, 4.0))
        img = np.clip(img * gain, 0, 255)
        if gain > 1.6:
            img += self._rng.normal(0.0, 3.2 * (gain - 1.0), img.shape)
        img = np.clip(img, 0, 255)
        img = 255.0 * (img / 255.0) ** 0.94
        return np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)

    def _stamp_instances_rgb(self, img: np.ndarray, pose: CameraPose, t: float) -> np.ndarray:
        w = self.world
        sp = self.spec
        items = []
        for v in w.victims:
            col = RGB_BASE["human_clothing"]
            if v.posture == VictimPosture.IN_WATER_CLINGING:
                col = (205, 175, 140)     # mostly head and shoulders above water
            items.append((v.north, v.east, v.elevation, col,
                          _human_apparent_area_m2(v), 1.0 - v.occlusion, v.visual_salience))
        decoy_col = {"animal": (120, 96, 70), "vehicle": (190, 190, 200),
                     "mannequin": (210, 70, 60), "hot_rock": (110, 100, 92),
                     "wet_cloth": (150, 160, 170), "fire_barrel": (90, 80, 70)}
        for d in w.distractors:
            items.append((d.north, d.east, d.elevation, decoy_col.get(d.kind, (150, 150, 150)),
                          math.pi * (d.size_m / 2.0) ** 2, 1.0, d.visual_salience))
        if not items:
            return img
        norths = np.array([i[0] for i in items])
        easts = np.array([i[1] for i in items])
        ups = np.array([i[2] for i in items])
        u, v, rng = self.ground_to_pixel(pose, norths, easts, ups)
        m_per_px = rng / sp.focal_px
        for k, (_n, _e, _z, col, area, vis, sal) in enumerate(items):
            if not np.isfinite(u[k]) or rng[k] > sp.max_range_m:
                continue
            px_radius = math.sqrt(area / math.pi) / max(m_per_px[k], 1e-6)
            if px_radius < 0.12:
                continue
            tau = math.exp(-sp.haze_k * rng[k])
            _add_blob_rgb(img, u[k], v[k], px_radius, col, vis * sal * tau,
                          _blob_aspect(area))
        return img

    # ------------------------------------------------------------------ #
    def _build_truth(self, pose: CameraPose, t: float, ground_z: float) -> List[TruthObject]:
        w = self.world
        sp = self.spec
        out: List[TruthObject] = []
        m, n = 8.0, 8.0

        def project(north, east, up):
            u, v, rng = self.ground_to_pixel(pose, np.array([north]), np.array([east]),
                                             np.array([up]))
            return float(u[0]), float(v[0]), float(rng[0])

        # Batch-project everything at once, then build objects.
        all_items: List[Tuple[str, float, float, float]] = []
        for vic in w.victims:
            all_items.append(("victim", vic.north, vic.east, vic.elevation))
        for dis in w.distractors:
            all_items.append(("decoy", dis.north, dis.east, dis.elevation))
        for b in w.buildings:
            all_items.append(("bld", b.north, b.east, b.height))
        for pl in w.powerlines:
            all_items.append(("pl", (pl.p1[0] + pl.p2[0]) / 2.0,
                              (pl.p1[1] + pl.p2[1]) / 2.0, 0.4))
        if all_items:
            uu, vv, rr = self.ground_to_pixel(
                pose,
                np.array([i[1] for i in all_items]),
                np.array([i[2] for i in all_items]),
                np.array([i[3] for i in all_items]),
            )
        else:
            uu = vv = rr = np.zeros(0)

        def in_frame(k: int) -> bool:
            return bool(np.isfinite(uu[k]) and np.isfinite(vv[k])
                        and 0.4 < rr[k] <= sp.max_range_m
                        and -m <= uu[k] <= sp.width + m
                        and -n <= vv[k] <= sp.height + n)

        for k, vic in enumerate(w.victims):
            if not in_frame(k):
                continue
            u, v, rng = float(uu[k]), float(vv[k]), float(rr[k])
            gsd = rng / sp.focal_px
            area = _human_apparent_area_m2(vic)
            area_px = area / max(gsd * gsd, 1e-12)
            a, b = _semi_axes(area_px, _blob_aspect(area))
            px = math.sqrt(area_px / math.pi)          # equal-area radius, px
            occ = max(float(np.clip(_occlusion_to(w, pose, vic.north, vic.east, vic.elevation),
                                    0, 1)), vic.occlusion)
            tau = math.exp(-sp.lwir_extinction_k * rng * 1.6)
            t_app = tau * vic.body_temp_c + (1.0 - tau) * sp.atm_temp_c
            bg = float(np.nan_to_num(w.surface_temperature(vic.north, vic.east)))
            out.append(TruthObject(
                oid=vic.vid, category="victim", label="person", pixel=(u, v),
                width_px=2.0 * a, height_px=2.0 * b, distance_m=rng,
                world_ned=(vic.north, vic.east, -vic.elevation),
                apparent_temp_c=t_app, background_temp_c=bg, occlusion=occ,
                visible=(occ < 0.92 and px >= 0.9),
                area_px=area_px, area_m2=area, gsd_m=gsd,
                attributes={"posture": vic.posture.value, "group": vic.group,
                            "needs": vic.needs, "in_water": vic.in_water,
                            "on_rooftop": vic.on_rooftop, "moving": vic.moving,
                            "visual_salience": vic.visual_salience,
                            "category": vic.category},
            ))

        nv = len(w.victims)
        for j, dis in enumerate(w.distractors):
            k = nv + j
            if not in_frame(k):
                continue
            u, v, rng = float(uu[k]), float(vv[k]), float(rr[k])
            gsd = rng / sp.focal_px
            area = math.pi * (dis.size_m / 2.0) ** 2 * 0.6
            area_px = area / max(gsd * gsd, 1e-12)
            px = math.sqrt(area_px / math.pi)
            tau = math.exp(-sp.lwir_extinction_k * rng * 1.6)
            t_app = tau * dis.temp_c + (1.0 - tau) * sp.atm_temp_c
            bg = float(np.nan_to_num(w.surface_temperature(dis.north, dis.east)))
            label = {"animal": "animal", "vehicle": "vehicle", "mannequin": "person",
                     "wet_cloth": "cloth", "hot_rock": "hot_rock",
                     "fire_barrel": "fire"}[dis.kind]
            out.append(TruthObject(
                oid=dis.did, category="distractor", label=label, pixel=(u, v),
                width_px=2.0 * px, height_px=2.0 * px, distance_m=rng,
                world_ned=(dis.north, dis.east, -dis.elevation),
                apparent_temp_c=t_app, background_temp_c=bg, occlusion=0.0,
                visible=(px >= 0.9), area_px=area_px, area_m2=area, gsd_m=gsd,
                attributes={"kind": dis.kind, "true_label": label,
                            "looks_like_person": label == "person"},
            ))

        nb = nv + len(w.distractors)
        for j, b in enumerate(w.buildings):
            k = nb + j
            if not in_frame(k):
                continue
            u, v, rng = float(uu[k]), float(vv[k]), float(rr[k])
            gsd = rng / sp.focal_px
            area_px = (b.width * b.length) / max(gsd * gsd, 1e-12)
            label = {0: "structure", 1: "damaged_structure", 2: "damaged_structure",
                     3: "collapsed_structure"}[b.damage]
            out.append(TruthObject(
                oid=b.bid, category="hazard" if b.damage >= 2 else "structure",
                label=label, pixel=(u, v), width_px=b.width / max(gsd, 1e-9),
                height_px=b.length / max(gsd, 1e-9), distance_m=rng,
                world_ned=(b.north, b.east, -b.height), occlusion=0.0, visible=True,
                area_px=area_px, area_m2=b.width * b.length, gsd_m=gsd,
                attributes={"damage": b.damage, "material": b.material,
                            "height": b.height, "occupied": b.occupied},
            ))

        np_ = nb + len(w.buildings)
        for j, pl in enumerate(w.powerlines):
            k = np_ + j
            hazardous = pl.downed or bool(np.nan_to_num(w.water_depth(*pl.p1)) > 0.2)
            if not hazardous or not in_frame(k):
                continue
            u, v, rng = float(uu[k]), float(vv[k]), float(rr[k])
            gsd = rng / sp.focal_px
            length = math.dist(pl.p1, pl.p2)
            length_px = length / max(gsd, 1e-9)
            out.append(TruthObject(
                oid=pl.pid, category="hazard", label="exposed_powerline",
                pixel=(u, v), width_px=length_px, height_px=2.0, distance_m=rng,
                world_ned=((pl.p1[0] + pl.p2[0]) / 2, (pl.p1[1] + pl.p2[1]) / 2, -0.4),
                occlusion=0.0, visible=True,
                # A conductor is a line, not a blob: its "area" is span x the
                # ~1 px the core can resolve.  Reported so the TTP size term is
                # not silently inherited from a bounding box.
                area_px=max(length_px, 1.0), area_m2=length * 0.03, gsd_m=gsd,
                attributes={"downed": pl.downed, "energized": pl.energized,
                            "in_water": bool(np.nan_to_num(w.water_depth(*pl.p1)) > 0.2)},
            ))
        return out

    def _footprint(self, pose: CameraPose, ground_z: float) -> Tuple[Tuple[float, float], ...]:
        sp = self.spec
        corners = [(0, 0), (sp.width - 1, 0), (sp.width - 1, sp.height - 1), (0, sp.height - 1)]
        out = []
        for (cu, cv) in corners:
            x = (cu - sp.width / 2.0) / sp.focal_px
            y = (cv - sp.height / 2.0) / sp.focal_px
            d_ned = pose.dcm_cam_ned.T @ np.array([x, y, 1.0])
            if d_ned[2] <= 1e-6:
                out.append((float("nan"), float("nan")))
                continue
            s = (ground_z - pose.position[2]) / d_ned[2]
            out.append((float(pose.position[0] + s * d_ned[0]),
                        float(pose.position[1] + s * d_ned[1])))
        return tuple(out)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _human_apparent_area_m2(v: Victim) -> float:
    """Projected LWIR-visible area of a person [m^2], by posture.

    A standing adult presents ~0.62 m^2 to a nadir camera (head/shoulders);
    lying down ~0.34 m^2 of top surface; chest-deep in water only head and
    shoulders, ~0.11 m^2.  These are the geometric cross-sections used in
    aerial LWIR detection work and they are the reason flood victims are the
    hardest class to find.
    """
    base = {
        VictimPosture.STANDING: 0.62,
        VictimPosture.WAVING: 0.78,
        VictimPosture.SITTING: 0.46,
        VictimPosture.LYING: 0.34,
        VictimPosture.PRONE_PARTIAL_BURIAL: 0.16,
        VictimPosture.IN_WATER_CLINGING: 0.11,
    }.get(v.posture, 0.5)
    return base * (1.0 - 0.55 * v.occlusion)


def _blob_aspect(area_m2: float) -> float:
    """Width/height of the rendered blob (people are taller than wide)."""
    if area_m2 < 0.14:
        return 0.9          # head and shoulders: nearly round
    if area_m2 < 0.4:
        return 0.72
    return 0.55


def _semi_axes(area_px: float, aspect: float) -> Tuple[float, float]:
    """Ellipse semi-axes (a, b) in pixels for a given area and width/height ratio.

    ``pi*a*b = area_px`` and ``a/b = aspect`` -> ``a = r*sqrt(aspect)``,
    ``b = r/sqrt(aspect)`` with ``r = sqrt(area_px/pi)``.  Used by both the
    renderer (what to stamp) and the truth builder (what box to report), so the
    image and the metadata can never disagree about a target's size.
    """
    r = math.sqrt(max(area_px, 0.0) / math.pi)
    a = max(aspect, 1e-3)
    return r * math.sqrt(a), r / math.sqrt(a)


#: Sensor point-spread-function sigma, in pixels.  An uncooled microbolometer
#: with a 12 um pitch and a fast germanium lens is close to diffraction limited
#: at f/1.0-1.2, i.e. about 1 pixel FWHM.  This is what makes a sub-pixel target
#: lose peak contrast while conserving energy - the effect that decides whether a
#: distant survivor is a 2 K bump or an invisible one.
PSF_SIGMA_PX = 0.62


def _gauss_matrix(n: int, sigma: float) -> np.ndarray:
    """Normalised, truncated 1-D Gaussian convolution matrix (n x n)."""
    if sigma <= 0.05 or n < 2:
        return np.eye(n)
    i = np.arange(n)
    d = (i[:, None] - i[None, :]).astype(np.float64)
    k = np.exp(-0.5 * (d / sigma) ** 2)
    k[np.abs(d) > 4.0 * sigma] = 0.0
    k /= np.maximum(k.sum(axis=1, keepdims=True), 1e-12)
    return k


def _footprint(a: float, b: float, u: float, v: float, u0: int, u1: int,
               v0: int, v1: int, strength: float, supersample: int = 3) -> np.ndarray:
    """Pixel-integrated elliptical footprint, blurred by the sensor PSF.

    A target is a *disk*, not a Gaussian.  Rendering it as a Gaussian with
    ``sigma = radius`` - what an earlier version did - spreads the same energy
    over ~3.6x the area at a reduced peak, which halves the apparent contrast of
    every survivor and makes the image inconsistent with the truth metadata.
    Here the ellipse is rasterised at ``supersample``x, convolved with a
    truncated Gaussian PSF, then block-averaged back to pixel centres, so:

    * a target much larger than the PSF reaches its full apparent temperature;
    * a sub-pixel target conserves energy but loses peak amplitude, exactly as a
      real microbolometer does;
    * total emitted energy is preserved, so radiometric thresholds stay valid.
    """
    ss = max(1, int(supersample))
    H, W = v1 - v0, u1 - u0
    xs = (np.arange(u0 * ss, u1 * ss) + 0.5) / ss - u
    ys = (np.arange(v0 * ss, v1 * ss) + 0.5) / ss - v
    ell = ((xs[None, :] / max(a, 1e-6)) ** 2 + (ys[:, None] / max(b, 1e-6)) ** 2) <= 1.0
    if not ell.any():
        return np.zeros((H, W), dtype=np.float64)
    if ss > 1:
        e = ell.astype(np.float64)
        ky = _gauss_matrix(H * ss, PSF_SIGMA_PX * ss)
        kx = _gauss_matrix(W * ss, PSF_SIGMA_PX * ss)
        e = ky @ e @ kx.T
        prof = e.reshape(H, ss, W, ss).mean(axis=(1, 3))
    else:
        prof = _gauss_matrix(H, PSF_SIGMA_PX) @ ell.astype(np.float64) @ \
            _gauss_matrix(W, PSF_SIGMA_PX).T
    return np.clip(prof * strength, 0.0, 1.0)


def _blob_window(u: float, v: float, a: float, b: float, shape: Tuple[int, int]
                 ) -> Tuple[int, int, int, int]:
    pad = 3.0 * PSF_SIGMA_PX + 1.0
    h, w = shape
    u0, u1 = max(0, int(math.floor(u - a - pad))), min(w, int(math.ceil(u + a + pad)) + 1)
    v0, v1 = max(0, int(math.floor(v - b - pad))), min(h, int(math.ceil(v + b + pad)) + 1)
    return u0, u1, v0, v1


def _add_blob(img: np.ndarray, u: float, v: float, radius_px: float, value: float,
              strength: float, aspect: float = 1.0) -> None:
    """Blend an elliptical thermal target into a 2-D float image."""
    if strength <= 0.005:
        return
    aspect = max(aspect, 0.2)
    area_px = math.pi * max(radius_px, 0.0) ** 2
    a, b = _semi_axes(area_px, aspect)
    u0, u1, v0, v1 = _blob_window(u, v, a, b, img.shape)
    if u0 >= u1 or v0 >= v1:
        return
    prof = _footprint(a, b, u, v, u0, u1, v0, v1, strength)
    sub = img[v0:v1, u0:u1].astype(np.float64)
    img[v0:v1, u0:u1] = (sub + (value - sub) * prof).astype(img.dtype)


def _add_blob_rgb(img: np.ndarray, u: float, v: float, radius_px: float,
                  colour: Sequence[float], strength: float, aspect: float = 1.0) -> None:
    """Blend an elliptical visual target into an RGB image (same footprint model)."""
    if strength <= 0.01:
        return
    aspect = max(aspect, 0.2)
    area_px = math.pi * max(radius_px, 0.0) ** 2
    a, b = _semi_axes(area_px, aspect)
    u0, u1, v0, v1 = _blob_window(u, v, a, b, img.shape[:2])
    if u0 >= u1 or v0 >= v1:
        return
    g = _footprint(a, b, u, v, u0, u1, v0, v1, strength)[..., None]
    sub = img[v0:v1, u0:u1].astype(np.float64)
    col = np.asarray(colour, dtype=np.float64).reshape(1, 1, 3)
    img[v0:v1, u0:u1] = sub * (1 - g) + col * g


def _gradient(grid: np.ndarray, res: float, nn: np.ndarray, ee: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour terrain gradient (kept for external analysis tools)."""
    r = np.clip(np.round(nn / res).astype(int), 1, grid.shape[0] - 2)
    c = np.clip(np.round(ee / res).astype(int), 1, grid.shape[1] - 2)
    dn = (grid[r + 1, c] - grid[r - 1, c]) / (2.0 * res)
    de = (grid[r, c + 1] - grid[r, c - 1]) / (2.0 * res)
    return dn.astype(np.float64), de.astype(np.float64)


def _occlusion_to(world: DisasterWorld, pose: CameraPose,
                  north: float, east: float, up_z: float, samples: int = 6) -> float:
    """Fraction of the line of sight blocked by structures (0 = clear)."""
    cam_n, cam_e, cam_z = pose.position
    tgt_z = -up_z
    dn_total, de_total, dz_total = north - cam_n, east - cam_e, tgt_z - cam_z
    blocked = 0
    ns = np.linspace(0.2, 0.85, samples)
    sn = cam_n + dn_total * ns
    se = cam_e + de_total * ns
    sz = cam_z + dz_total * ns
    ground = np.nan_to_num(world.terrain_z(sn, se))
    h = np.nan_to_num(world.structure_height(sn, se))
    blocked = int(np.sum(h > (ground - sz) + 0.4))
    return blocked / samples
