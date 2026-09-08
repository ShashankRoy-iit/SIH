"""The synthetic disaster world: ground truth for the whole simulation.

``DisasterWorld`` owns everything that is *true* about the scene - terrain,
inundation, fire, structures, victims, distractors, wind and the GNSS denial
field.  The renderer turns slices of it into RGB + LWIR frames; the perception
stack tries to recover it; and the evaluation harness scores the difference.
Because the world is the single source of truth, precision/recall, localisation
error and coverage are all measurable rather than asserted.

Worlds are built from :class:`ScenarioSpec` objects (see ``sar.sim.scenario``)
and are fully deterministic given a seed.

Coordinate convention
---------------------
Local NED metres relative to ``world.origin`` throughout.  ``north`` increases
with +X, ``east`` with +Y.  Elevation is stored as metres *above the origin's
MSL altitude* so that ``terrain_z(n, e)`` can be compared directly with the
NED ``-z`` of the vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint, local_to_wgs84
from sar.vehicle.power import air_density


class VictimPosture(str, Enum):
    """Body posture - strongly affects both the visible silhouette and the
    thermal cross-section, and is the basis of the 'waving for help' cue."""

    STANDING = "standing"
    WAVING = "waving"
    SITTING = "sitting"
    LYING = "lying"
    PRONE_PARTIAL_BURIAL = "partial_burial"
    IN_WATER_CLINGING = "in_water"


class HazardClass(str, Enum):
    """Disaster hazards the stack must classify (mirrors the problem statement)."""

    FIRE = "fire"
    SMOKE = "smoke"
    FLOOD_WATER = "flood_water"
    FLASH_FLOOD_CHANNEL = "flash_flood_channel"
    DEBRIS_FIELD = "debris_field"
    COLLAPSED_STRUCTURE = "collapsed_structure"
    DAMAGED_STRUCTURE = "damaged_structure"
    EXPOSED_POWERLINE = "exposed_powerline"
    LANDSLIDE_ZONE = "landslide_zone"
    CHEMICAL_PLUME = "chemical_plume"
    GAS_CYLINDER = "gas_cylinder"
    BLOCKED_ROAD = "blocked_road"
    VEHICLE_WRECK = "vehicle_wreck"
    UNSTABLE_SLOPE = "unstable_slope"


# Human-relevant thermal bands (apparent, at the sensor) in degrees Celsius.
HUMAN_THERMAL_BAND = (24.0, 39.0)
ANIMAL_THERMAL_BAND = (26.0, 41.0)


@dataclass
class Victim:
    """A ground-truth survivor (or, in evaluation scenarios, a decoy)."""

    vid: str
    north: float
    east: float
    elevation: float = 0.0
    posture: VictimPosture = VictimPosture.STANDING
    alive: bool = True
    group: str = "solo"
    # Placement category, kept for reporting: water_edge | rooftop | dry_refuge
    # | structure | road.  Detection difficulty is dominated by this, so the
    # evaluation harness breaks recall down by it.
    category: str = "dry_refuge"
    # Apparent LWIR temperature of the brightest part of the body [degC].
    body_temp_c: float = 33.0
    # 0 = fully visible from the air, 1 = completely occluded by canopy/rubble.
    occlusion: float = 0.0
    # Whether a visual (RGB) cue exists: bright clothing, waving, etc.
    visual_salience: float = 0.7
    moving: bool = False
    speed_ms: float = 0.0
    heading_deg: float = 0.0
    needs: str = "medical"          # medical | flotation | evacuation | none
    in_water: bool = False
    on_rooftop: bool = False
    discovered_at: Optional[float] = None

    def advance(self, dt: float) -> None:
        if not self.moving:
            return
        h = math.radians(self.heading_deg)
        self.north += math.cos(h) * self.speed_ms * dt
        self.east += math.sin(h) * self.speed_ms * dt

    @property
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east, -self.elevation])


@dataclass
class Distractor:
    """A false-positive source: animal, hot rock, warm vehicle, sun-warmed slab."""

    did: str
    kind: str                      # 'animal' | 'hot_rock' | 'vehicle' | 'mannequin' | 'wet_cloth'
    north: float
    east: float
    elevation: float = 0.0
    temp_c: float = 30.0
    size_m: float = 0.8
    visual_salience: float = 0.4
    moving: bool = False
    speed_ms: float = 0.0
    heading_deg: float = 0.0

    def advance(self, dt: float) -> None:
        if not self.moving:
            return
        h = math.radians(self.heading_deg)
        self.north += math.cos(h) * self.speed_ms * dt
        self.east += math.sin(h) * self.speed_ms * dt


@dataclass
class Building:
    """A rectangular structure with a Joint-Damage-Scale style damage grade."""

    bid: str
    north: float
    east: float
    width: float
    length: float
    height: float
    heading_deg: float = 0.0
    damage: int = 0                # 0 none, 1 minor, 2 major, 3 destroyed
    material: str = "concrete"     # concrete | metal | brick | wood
    occupied: bool = False


@dataclass
class PowerLine:
    """A transmission/distribution span; ``downed`` makes it an acute hazard."""

    pid: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    downed: bool = False
    energized: bool = True
    pole_height: float = 9.0


@dataclass
class WindField:
    """Altitude-varying wind with gusts.

    Uses a power-law wind profile plus a deterministic gust field so that
    runs are reproducible.  Disaster-relevant: cyclone and post-blast
    environments have strong shear, and shear is what breaks a naive survey.
    """

    speed_ms: float = 5.0
    direction_deg: float = 210.0     # direction the wind blows *towards*
    shear_exponent: float = 0.16
    reference_height_m: float = 10.0
    gust_ms: float = 2.5
    turbulence_ms: float = 0.8

    def sample(self, north: float, east: float, altitude_m: float, t: float) -> np.ndarray:
        h = max(1.0, altitude_m)
        speed = self.speed_ms * (h / self.reference_height_m) ** self.shear_exponent
        # Deterministic spatial gust field.
        gust = self.gust_ms * math.sin(north * 0.011 + t * 0.21) * math.cos(east * 0.013 - t * 0.17)
        turb = self.turbulence_ms * math.sin(t * 1.7 + north * 0.05 + east * 0.03)
        brg = math.radians(self.direction_deg)
        s = speed + gust + turb * 0.5
        wn = s * math.cos(brg) + turb * 0.3 * math.sin(t * 2.3)
        we = s * math.sin(brg) + turb * 0.3 * math.cos(t * 1.9)
        return np.array([wn, we, turb * 0.15])

    def at_altitude(self, altitude_m: float) -> float:
        return self.speed_ms * (max(1.0, altitude_m) / self.reference_height_m) ** self.shear_exponent


class GridIndex:
    """Pre-computed bilinear sampling indices shared across many layers.

    A camera frame samples ~10 world layers at the same ground coordinates.
    Computing the integer indices and fractional weights once (instead of once
    per layer) is a ~5x speed-up and is what makes real-time synthetic imaging
    possible on a 2-core machine.
    """

    __slots__ = ("r0", "r1", "c0", "c1", "fr", "fc", "valid", "shape")

    def __init__(self, north: np.ndarray, east: np.ndarray, res: float,
                 shape: Tuple[int, int]) -> None:
        nr, er = shape
        rn = (np.asarray(north, dtype=np.float64)) / res - 0.5
        re = (np.asarray(east, dtype=np.float64)) / res - 0.5
        r0f = np.floor(rn)
        c0f = np.floor(re)
        self.fr = (rn - r0f).astype(np.float32)
        self.fc = (re - c0f).astype(np.float32)
        r0 = r0f.astype(np.int64)
        c0 = c0f.astype(np.int64)
        self.r0 = np.clip(r0, 0, nr - 1)
        self.r1 = np.clip(r0 + 1, 0, nr - 1)
        self.c0 = np.clip(c0, 0, er - 1)
        self.c1 = np.clip(c0 + 1, 0, er - 1)
        self.valid = (rn >= -0.5) & (rn <= nr - 0.5) & (re >= -0.5) & (re <= er - 0.5)
        self.shape = (nr, er)


class MicroTextureField:
    """Sub-cell procedural texture, stable in world coordinates.

    The world's raster layers are 2 m cells, but real ground has thermal and
    visual structure at 5-50 cm: individual stones, leaf litter, puddles, rebar,
    ash, sunlit and shaded facets.  Without it the simulated background is
    *smoother than reality* and every false-alarm number we report would be
    optimistic - precisely the kind of error that sinks a field deployment.

    Two wrapping value-noise tiles with incommensurate cell sizes and periods
    (0.55 m / 35 m and 1.35 m / 55 m) are summed, so the result has no visible
    repeat at survey scale and costs about 1 ms per frame.  Because it is a pure
    function of (north, east) it does not swim as the aircraft moves, which
    matters: the temporal tracker assumes the background is stationary in world
    coordinates and only the aircraft changes.
    """

    __slots__ = ("_ta", "_tb", "_cell_a", "_cell_b", "_scale")

    def __init__(self, seed: int = 0, cell_a: float = 0.55, tile_a: int = 64,
                 cell_b: float = 1.35, tile_b: int = 41) -> None:
        rng = np.random.default_rng(int(seed) * 977 + 13)
        self._ta = rng.standard_normal((tile_a, tile_a)).astype(np.float32)
        self._tb = rng.standard_normal((tile_b, tile_b)).astype(np.float32)
        self._cell_a = float(cell_a)
        self._cell_b = float(cell_b)
        self._scale = 1.0 / math.sqrt(2.0)

    @staticmethod
    def _sample_tile(tile: np.ndarray, north: np.ndarray, east: np.ndarray,
                     cell: float) -> np.ndarray:
        n = tile.shape[0]
        rx = np.asarray(north, dtype=np.float64) / cell
        ry = np.asarray(east, dtype=np.float64) / cell
        i0 = np.floor(rx)
        j0 = np.floor(ry)
        # Smoothstep weights give C1 continuity, so there are no grid seams for a
        # detector's local-maximum filter to lock onto as false targets.
        fx = (rx - i0).astype(np.float32)
        fy = (ry - j0).astype(np.float32)
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        i0 = np.mod(i0.astype(np.int64), n)
        j0 = np.mod(j0.astype(np.int64), n)
        i1 = (i0 + 1) % n
        j1 = (j0 + 1) % n
        return (tile[i0, j0] * (1 - fx) * (1 - fy) + tile[i1, j0] * fx * (1 - fy)
                + tile[i0, j1] * (1 - fx) * fy + tile[i1, j1] * fx * fy)

    def sample(self, north: np.ndarray, east: np.ndarray) -> np.ndarray:
        """Unit-variance signed texture field."""
        a = self._sample_tile(self._ta, north, east, self._cell_a)
        b = self._sample_tile(self._tb, north, east, self._cell_b)
        return ((a + b) * self._scale).astype(np.float32)


class _Raster:
    """Float32 grid over the world with bilinear sampling."""

    __slots__ = ("data", "res", "n0", "e0", "shape")

    def __init__(self, data: np.ndarray, res: float, n0: float, e0: float) -> None:
        self.data = data.astype(np.float32)
        self.res = float(res)
        self.n0 = float(n0)
        self.e0 = float(e0)
        self.shape = data.shape

    def sample_idx(self, gi: GridIndex) -> np.ndarray:
        """Bilinear sample using pre-computed indices (fast path)."""
        d = self.data
        fc = gi.fc
        top = d[gi.r0, gi.c0] * (1.0 - fc) + d[gi.r0, gi.c1] * fc
        bot = d[gi.r1, gi.c0] * (1.0 - fc) + d[gi.r1, gi.c1] * fc
        out = top * (1.0 - gi.fr) + bot * gi.fr
        return np.where(gi.valid, out, np.nan).astype(np.float32)

    def sample(self, north: np.ndarray, east: np.ndarray) -> np.ndarray:
        """Bilinear sample at arbitrary (vectorised) local coordinates."""
        gi = GridIndex(north, east, self.res, self.shape)
        return self.sample_idx(gi)


@dataclass
class ScenarioSpec:
    """Declarative recipe for a world (see ``sar.sim.scenario`` for presets)."""

    name: str = "flood_basin"
    disaster: str = "flood"
    origin: GeoPoint = field(default_factory=lambda: GeoPoint(26.9124, 75.7873, 240.0))
    extent_north_m: float = 900.0
    extent_east_m: float = 900.0
    grid_res_m: float = 2.0
    seed: int = 7
    n_victims: int = 9
    n_distractors: int = 7
    n_buildings: int = 40
    flood_fraction: float = 0.45
    fire_sites: int = 2
    landslide_sites: int = 0
    downed_powerlines: int = 2
    wind_speed_ms: float = 5.0
    wind_direction_deg: float = 210.0
    time_of_day_h: float = 16.5      # local solar hour, drives thermal contrast
    cloud_cover: float = 0.35
    rain: float = 0.0                # 0..1 -> visibility loss + GNSS attenuation
    smoke_density: float = 0.25
    vegetation_fraction: float = 0.18
    gps_denial_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)
    gps_canyon_zones: List[Tuple[float, float, float, float]] = field(default_factory=list)
    base_satellites: int = 34
    victim_prior_weights: Dict[str, float] = field(default_factory=dict)


class DisasterWorld:
    """Rasterised + vectorised ground truth for a disaster sortie."""

    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self.origin = spec.origin
        self.rng = np.random.default_rng(spec.seed)
        self.north_m = spec.extent_north_m
        self.east_m = spec.extent_east_m
        res = spec.grid_res_m
        nr = int(math.ceil(self.north_m / res)) + 1
        er = int(math.ceil(self.east_m / res)) + 1
        self.res = res
        self.grid_shape = (nr, er)
        nn = np.linspace(0.0, self.north_m, nr)[:, None] * np.ones((1, er))
        ee = np.ones((nr, 1)) * np.linspace(0.0, self.east_m, er)[None, :]
        self._nn, self._ee = nn, ee

        self.victims: List[Victim] = []
        self.distractors: List[Distractor] = []
        self.buildings: List[Building] = []
        self.powerlines: List[PowerLine] = []
        self.roads: List[List[Tuple[float, float]]] = []
        self.reliefs: List[Dict[str, object]] = []     # dropped payloads / beacons

        #: Sub-cell thermal/visual clutter (see MicroTextureField).
        self.micro_texture = MicroTextureField(seed=spec.seed)

        self.wind = WindField(
            speed_ms=spec.wind_speed_ms,
            direction_deg=spec.wind_direction_deg,
            gust_ms=max(0.6, spec.wind_speed_ms * 0.45),
            turbulence_ms=0.5 + spec.wind_speed_ms * 0.12,
        )

        self._build_layers(nn, ee)
        self._place_roads()
        self._place_buildings()
        self._place_powerlines()
        self._place_victims()
        self._place_distractors()

    # ------------------------------------------------------------------ #
    # Layer construction
    # ------------------------------------------------------------------ #
    def _layer(self, fill: float = 0.0) -> np.ndarray:
        return np.full(self.grid_shape, fill, dtype=np.float32)

    def _make_raster(self, data: np.ndarray) -> _Raster:
        return _Raster(data, self.res, 0.0, 0.0)

    def _fbm(self, nn: np.ndarray, ee: np.ndarray, octaves: int = 4,
             base_scale: float = 400.0, seed_offset: int = 0) -> np.ndarray:
        """Cheap deterministic value-noise fractal sum (no scipy dependency)."""
        rng = np.random.default_rng(self.spec.seed * 17 + seed_offset)
        out = np.zeros_like(nn, dtype=np.float64)
        amp, scale, total = 1.0, base_scale, 0.0
        for _ in range(octaves):
            gx = int(max(3, self.north_m / scale)) + 1
            gy = int(max(3, self.east_m / scale)) + 1
            g = rng.standard_normal((gx, gy))
            # smooth it once so it does not look like TV static
            k = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
            gp = np.pad(g, 1, mode="edge")
            gs = sum(k[i, j] * gp[i:i + g.shape[0], j:j + g.shape[1]]
                     for i in range(3) for j in range(3))
            rn = nn / scale
            re = ee / scale
            i0 = np.clip(np.floor(rn).astype(int), 0, gs.shape[0] - 2)
            j0 = np.clip(np.floor(re).astype(int), 0, gs.shape[1] - 2)
            fr = rn - i0
            fc = re - j0
            a = gs[i0, j0] * (1 - fr) * (1 - fc) + gs[i0, j0 + 1] * (1 - fr) * fc
            b = gs[i0 + 1, j0] * fr * (1 - fc) + gs[i0 + 1, j0 + 1] * fr * fc
            out += amp * (a + b)
            total += amp
            amp *= 0.5
            scale *= 0.5
        return out / total

    def _build_layers(self, nn: np.ndarray, ee: np.ndarray) -> None:
        spec = self.spec
        # --- terrain elevation ------------------------------------------------
        base = self._fbm(nn, ee, octaves=4, base_scale=520.0, seed_offset=1)
        ridge = self._fbm(nn, ee, octaves=3, base_scale=260.0, seed_offset=2)
        if spec.disaster in ("landslide", "earthquake"):
            elev = 30.0 * base + 22.0 * np.clip(ridge, -1, 1) + 0.012 * nn
        elif spec.disaster == "flood":
            # A river valley: low in the middle band, rising to the sides.
            centre = self.east_m * 0.45
            valley = np.exp(-((ee - centre) ** 2) / (2 * (self.east_m * 0.22) ** 2))
            elev = 14.0 * base + 26.0 * (1.0 - valley) + 0.004 * nn
        else:
            elev = 12.0 * base + 6.0 * ridge
        elev -= elev.min()
        self.elevation = self._make_raster(elev)

        # --- a river / drainage channel --------------------------------------
        centre = self.east_m * 0.45 + 40.0 * np.sin(nn / 260.0)
        channel = np.exp(-((ee - centre) ** 2) / (2 * 55.0 ** 2))
        self.channel = channel

        # --- flood inundation -------------------------------------------------
        flood = self._layer()
        if spec.disaster in ("flood", "cyclone", "tsunami", "flash_flood"):
            water_level = float(np.percentile(elev, 100.0 * spec.flood_fraction * 0.62))
            depth = np.clip(water_level - elev, 0.0, None)
            depth = np.where(channel > 0.15, np.maximum(depth, 1.2 + 2.0 * channel), depth)
            flood = depth
        self.flood_depth = self._make_raster(flood)
        self.water_level = float(water_level) if spec.disaster in ("flood", "cyclone", "tsunami", "flash_flood") else 0.0

        # --- fire and smoke ---------------------------------------------------
        fire = self._layer()
        smoke = self._layer()
        self.fire_sites: List[Tuple[float, float, float]] = []
        for i in range(max(0, spec.fire_sites)):
            fn = float(self.rng.uniform(0.15, 0.85) * self.north_m)
            fe = float(self.rng.uniform(0.15, 0.85) * self.east_m)
            if flood_at_point(flood, fn, fe, self.res) > 0.4:
                continue
            strength = float(self.rng.uniform(0.7, 1.0))
            radius = float(self.rng.uniform(14.0, 34.0))
            self.fire_sites.append((fn, fe, strength))
            d2 = (nn - fn) ** 2 + (ee - fe) ** 2
            fire += strength * np.exp(-d2 / (2 * radius ** 2))
            # Smoke drifts downwind.
            wd = math.radians(spec.wind_direction_deg + 90.0)
            dn = (nn - fn) * math.cos(wd) + (ee - fe) * math.sin(wd)
            dp = -(nn - fn) * math.sin(wd) + (ee - fe) * math.cos(wd)
            plume = np.exp(-(dp ** 2) / (2 * (radius * 1.6) ** 2)) * np.clip(dn / (radius * 2), 0, 1)
            smoke += 0.85 * strength * plume * np.exp(-np.clip(dn, 0, None) / (radius * 12))
        self.fire = self._make_raster(np.clip(fire, 0, 1))
        self.smoke = self._make_raster(np.clip(smoke * spec.smoke_density / max(spec.smoke_density, 1e-6), 0, 1))

        # --- landslide scar / unstable slope ----------------------------------
        slide = self._layer()
        self.landslide_sites = []
        for _ in range(max(0, spec.landslide_sites)):
            sn = float(self.rng.uniform(0.2, 0.8) * self.north_m)
            se = float(self.rng.uniform(0.2, 0.8) * self.east_m)
            self.landslide_sites.append((sn, se))
            d2 = ((nn - sn) / 90.0) ** 2 + ((ee - se) / 45.0) ** 2
            slide += np.exp(-d2)
        self.landslide = self._make_raster(np.clip(slide, 0, 1))

        # --- debris field (rubble) --------------------------------------------
        debris = self._fbm(nn, ee, octaves=5, base_scale=90.0, seed_offset=5)
        debris = np.clip((debris - 0.35) * 2.2, 0, 1)
        if spec.disaster in ("earthquake", "cyclone", "landslide"):
            debris = np.clip(debris * 1.7, 0, 1)
        debris = np.where(flood > 0.5, debris * 0.25, debris)
        self.debris = self._make_raster(debris)

        # --- vegetation / canopy ---------------------------------------------
        veg = self._fbm(nn, ee, octaves=3, base_scale=170.0, seed_offset=9)
        veg = np.clip((veg + 0.4) * spec.vegetation_fraction * 1.6, 0, 1)
        veg = np.where(flood > 0.8, veg * 0.15, veg)
        self.vegetation = self._make_raster(veg)

        # --- soil / surface temperature ---------------------------------------
        solar = solar_loading(spec.time_of_day_h, spec.cloud_cover)
        soil_t = 14.0 + 18.0 * solar + 3.0 * base
        soil_t = np.where(flood > 0.05, 12.0 + 6.0 * solar, soil_t)     # water is cold & stable
        soil_t = np.where(fire > 0.02, soil_t + 260.0 * np.clip(fire, 0, 1), soil_t)
        soil_t = np.where(smoke > 0.25, soil_t + 6.0 * smoke, soil_t)
        self.surface_temp = self._make_raster(soil_t)

        # --- visual texture (drives optical-flow quality) ----------------------
        texture = 0.15 + 0.55 * np.clip(debris, 0, 1) + 0.35 * np.clip(veg, 0, 1)
        texture = np.where(flood > 0.15, 0.04, texture)      # open water: no texture
        texture = np.where(smoke > 0.55, 0.10, texture)      # heavy smoke: no features
        self.texture = self._make_raster(np.clip(texture, 0.0, 1.0))

        # --- material albedo / RGB base colour ---------------------------------
        alb = 0.22 + 0.18 * np.clip(veg, 0, 1) + 0.14 * np.clip(debris, 0, 1)
        alb = np.where(flood > 0.15, 0.10, alb)
        self.albedo_raster = self._make_raster(np.clip(alb, 0.05, 0.85))

        # --- chemical plume (optional, industrial area) --------------------------
        chem = self._layer()
        if spec.disaster in ("industrial", "chemical", "earthquake"):
            cn, ce = 0.7 * self.north_m, 0.3 * self.east_m
            chem = 0.8 * np.exp(-(((nn - cn) / 60.0) ** 2 + ((ee - ce) / 30.0) ** 2))
        self.chemical = self._make_raster(np.clip(chem, 0, 1))

    # ------------------------------------------------------------------ #
    def _place_roads(self) -> None:
        """Two crossing roads; used for victim priors and access routing."""
        n1 = self.north_m * 0.38
        self.roads.append([(n1, 0.0), (n1 + 40.0, self.east_m)])
        self.roads.append([(0.0, self.east_m * 0.62), (self.north_m, self.east_m * 0.55)])
        self.road_raster = self._rasterize_polylines(self.roads, width_m=7.0)

    def _rasterize_polylines(self, lines: Sequence[Sequence[Tuple[float, float]]],
                             width_m: float) -> _Raster:
        out = self._layer()
        nn, ee = self._nn, self._ee
        for line in lines:
            for i in range(len(line) - 1):
                (n0, e0), (n1, e1) = line[i], line[i + 1]
                seg = math.hypot(n1 - n0, e1 - e0)
                if seg < 1e-6:
                    continue
                # distance from each grid node to the segment
                t = np.clip(((nn - n0) * (n1 - n0) + (ee - e0) * (e1 - e0)) / (seg * seg), 0, 1)
                pn = n0 + t * (n1 - n0)
                pe = e0 + t * (e1 - e0)
                d = np.hypot(nn - pn, ee - pe)
                out = np.maximum(out, np.clip(1.0 - d / (width_m / 2.0), 0, 1).astype(np.float32))
        return self._make_raster(out)

    def _place_buildings(self) -> None:
        spec = self.spec
        flood = self.flood_depth.data
        nn, ee = self._nn, self._ee
        attempts = 0
        while len(self.buildings) < spec.n_buildings and attempts < spec.n_buildings * 40:
            attempts += 1
            bn = float(self.rng.uniform(15, self.north_m - 15))
            be = float(self.rng.uniform(15, self.east_m - 15))
            # Cluster along roads / in a settlement core.
            if self.rng.random() < 0.65:
                road_n = self.north_m * 0.38
                bn = road_n + float(self.rng.normal(0, 55))
            w = float(self.rng.uniform(6, 16))
            l = float(self.rng.uniform(6, 20))
            h = float(self.rng.choice([3.2, 3.6, 6.5, 9.0, 12.0], p=[0.3, 0.3, 0.2, 0.12, 0.08]))
            mat = str(self.rng.choice(["concrete", "brick", "metal", "wood"], p=[0.45, 0.3, 0.15, 0.10]))
            if spec.disaster == "earthquake":
                dmg = int(self.rng.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.28, 0.22]))
            elif spec.disaster in ("flood", "cyclone"):
                dmg = int(self.rng.choice([0, 1, 2, 3], p=[0.45, 0.3, 0.18, 0.07]))
            else:
                dmg = int(self.rng.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05]))
            self.buildings.append(Building(
                bid=f"B{len(self.buildings):03d}", north=bn, east=be, width=w, length=l,
                height=h, heading_deg=float(self.rng.uniform(-25, 25)), damage=dmg,
                material=mat, occupied=bool(self.rng.random() < 0.55),
            ))
        # Rasterise building footprints (value = height) and a damage layer.
        heights = self._layer()
        damage = self._layer()
        occupied = self._layer()
        metal = self._layer()
        for b in self.buildings:
            ang = math.radians(b.heading_deg)
            ca, sa = math.cos(ang), math.sin(ang)
            dn = (nn - b.north) * ca + (ee - b.east) * sa
            de = -(nn - b.north) * sa + (ee - b.east) * ca
            mask = (np.abs(dn) <= b.length / 2) & (np.abs(de) <= b.width / 2)
            eff_h = b.height * (1.0 - 0.55 * (b.damage / 3.0))
            heights = np.where(mask, np.maximum(heights, eff_h), heights)
            damage = np.where(mask, np.maximum(damage, b.damage / 3.0), damage)
            occupied = np.where(mask, np.maximum(occupied, 1.0 if b.occupied else 0.0), occupied)
            if b.material == "metal":
                metal = np.where(mask, 1.0, metal)
            # Rubble apron around destroyed structures.
            if b.damage >= 2:
                apron = (np.abs(dn) <= b.length / 2 + 7) & (np.abs(de) <= b.width / 2 + 7)
                self.debris.data = np.where(apron, np.maximum(self.debris.data, 0.75 * (b.damage / 3.0)),
                                            self.debris.data)
        heights = np.where(self.flood_depth.data > 0.1, heights, heights)
        self.building_height = self._make_raster(heights)
        self.building_damage = self._make_raster(damage)
        self.building_occupied = self._make_raster(occupied)
        self.metal_roof = self._make_raster(metal)

    def _place_powerlines(self) -> None:
        for i in range(max(0, self.spec.downed_powerlines)):
            n0 = float(self.rng.uniform(0.1, 0.9) * self.north_m)
            e0 = float(self.rng.uniform(0.1, 0.9) * self.east_m)
            length = float(self.rng.uniform(60, 160))
            ang = math.radians(float(self.rng.uniform(0, 180)))
            p1 = (n0, e0)
            p2 = (n0 + length * math.cos(ang), e0 + length * math.sin(ang))
            downed = bool(self.rng.random() < 0.75)
            in_water = flood_at_point(self.flood_depth.data, n0, e0, self.res) > 0.2
            self.powerlines.append(PowerLine(
                pid=f"PL{i:02d}", p1=p1, p2=p2, downed=downed,
                energized=downed or in_water, pole_height=9.0,
            ))
        self.powerline_raster = self._rasterize_polylines(
            [pl.p1 for pl in self.powerlines] and [[pl.p1, pl.p2] for pl in self.powerlines],
            width_m=2.0,
        )

    # ------------------------------------------------------------------ #
    # Victim / distractor placement uses a *hazard-aware prior*, which is the
    # same prior the search planner is given.  Placing victims from the prior
    # (with noise) makes the evaluation meaningful: a planner that exploits the
    # prior should find them sooner, and we can measure exactly how much sooner.
    # ------------------------------------------------------------------ #
    def prior_density(self, resolution_m: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Probability-of-containment (POC) density over the search area.

        Encodes what a SAR duty officer actually knows before launch:
        people live along roads and in settlements; in a flood they end up on
        rooftops and on the high ground at the inundation edge; in an earthquake
        they are inside/near damaged structures; near a landslide scar they are
        downhill of it.
        """
        nr = int(self.north_m / resolution_m) + 1
        er = int(self.east_m / resolution_m) + 1
        nn = np.linspace(0, self.north_m, nr)[:, None] * np.ones((1, er))
        ee = np.ones((nr, 1)) * np.linspace(0, self.east_m, er)[None, :]

        def rs(raster: _Raster) -> np.ndarray:
            return raster.sample(nn, ee)

        road = self.road_raster.sample(nn, ee)
        bld = self.building_occupied.sample(nn, ee)
        flood = self.flood_depth.sample(nn, ee)
        dmg = self.building_damage.sample(nn, ee)
        slide = self.landslide.sample(nn, ee)
        elev = self.elevation.sample(nn, ee)

        d = self.spec.disaster
        w = dict(road=0.25, building=0.30, water_edge=0.25, dry_high=0.10,
                 damage=0.20, slide=0.10, uniform=0.06)
        w.update({k: float(v) for k, v in self.spec.victim_prior_weights.items() if k in w})

        density = np.zeros_like(nn)
        density += w["road"] * np.nan_to_num(road)
        density += w["building"] * np.nan_to_num(bld)
        # inundation edge: 0.05 m < depth < 1.2 m is where people get trapped
        edge = np.clip(1.0 - np.abs(np.nan_to_num(flood) - 0.45) / 0.7, 0, 1)
        density += (w["water_edge"] if d in ("flood", "cyclone", "flash_flood", "tsunami") else 0.0) * edge
        # Dry high ground: where people flee to when the water rises.
        elev = np.nan_to_num(elev)
        flood = np.nan_to_num(flood)
        if np.any(flood > 0.1):
            flood_line = float(np.max(elev[flood > 0.1]))
        else:
            flood_line = float(np.max(elev))
        dry = np.clip((elev - flood_line) / 6.0, 0.0, 1.0)
        density += (w["dry_high"] if d in ("flood", "cyclone", "flash_flood") else 0.0) * dry
        density += (w["damage"] if d in ("earthquake", "cyclone", "landslide") else 0.0) * np.nan_to_num(dmg)
        if slide.size and self.landslide_sites:
            dn = np.zeros_like(nn)
            for (sn, se) in self.landslide_sites:
                dn += np.exp(-(((nn - sn) / 120.0) ** 2 + ((ee - se) / 90.0) ** 2))
            density += w["slide"] * np.clip(dn, 0, 1)
        density += w["uniform"]
        density = np.nan_to_num(density, nan=0.0)
        density = np.clip(density, 1e-6, None)
        total = density.sum()
        return nn, ee, density / total

    def _place_victims(self) -> None:
        """Place survivors into operationally meaningful categories.

        Rather than pure prior sampling (which tends to dump everyone on dry
        high ground), each victim is assigned a *category* first and then a
        location that satisfies it.  The category mix is what makes the
        benchmark hard and realistic:

        ====================  ==================================================
        ``water_edge``        chest/shoulder deep at the inundation boundary -
                              only head and arms are visible in RGB, and LWIR
                              contrast is reduced by evaporative cooling.
        ``rooftop``           climbed onto a flooded structure - a bright,
                              unoccluded LWIR target but with no ground context.
        ``dry_refuge``        on the high ground people flee to.
        ``structure``         in/next to a damaged building (earthquake, cyclone).
        ``road``              along a road or track (the classic last-known-
                              position prior).
        ====================  ==================================================
        """
        spec = self.spec
        flood = self.flood_depth.data
        heights = self.building_height.data
        damage = self.building_damage.data
        road = self.road_raster.data
        elev = self.elevation.data
        res = self.res

        # Pre-compute candidate index sets (cheap, done once at world build).
        yy, xx = np.indices(flood.shape)
        cand = {
            "water_edge": np.argwhere((flood > 0.35) & (flood < 1.35) & (heights < 0.5)),
            "rooftop": np.argwhere((flood > 0.7) & (heights > 2.5)),
            "dry_refuge": np.argwhere((flood < 0.05) & (elev > np.percentile(elev, 68))),
            "structure": np.argwhere(damage > 0.4),
            "road": np.argwhere(road > 0.6),
        }
        # Any candidate set that came out empty falls back to "anywhere dry".
        fallback = np.argwhere(flood < 0.2)
        if fallback.size == 0:
            fallback = np.stack([yy.ravel(), xx.ravel()], axis=1)
        for k in list(cand):
            if cand[k].shape[0] < 4:
                cand[k] = fallback

        if spec.disaster in ("flood", "cyclone", "tsunami", "flash_flood"):
            mix = {"water_edge": 0.30, "rooftop": 0.26, "dry_refuge": 0.24,
                   "structure": 0.08, "road": 0.12}
        elif spec.disaster == "earthquake":
            mix = {"water_edge": 0.03, "rooftop": 0.05, "dry_refuge": 0.12,
                   "structure": 0.62, "road": 0.18}
        elif spec.disaster == "landslide":
            mix = {"water_edge": 0.05, "rooftop": 0.05, "dry_refuge": 0.20,
                   "structure": 0.45, "road": 0.25}
        else:
            mix = {"water_edge": 0.15, "rooftop": 0.15, "dry_refuge": 0.30,
                   "structure": 0.20, "road": 0.20}
        mix.update({k: float(v) for k, v in spec.victim_prior_weights.items() if k in mix})
        keys = list(mix)
        weights = np.array([mix[k] for k in keys], dtype=float)
        weights = weights / weights.sum()

        categories = list(self.rng.choice(keys, size=spec.n_victims, p=weights))
        used: Dict[str, List[int]] = {k: [] for k in keys}

        for i, cat in enumerate(categories):
            pool = cand[cat]
            order = self.rng.permutation(pool.shape[0])
            pick = None
            for oi in order:
                idx = int(oi)
                if idx in used[cat]:
                    continue
                # keep victims at least 25 m apart so they are distinct targets
                r, c = pool[idx]
                n_, e_ = r * res, c * res
                if any(math.hypot(n_ - v.north, e_ - v.east) < 25.0 for v in self.victims):
                    continue
                pick = idx
                break
            if pick is None:
                pick = int(order[0])
            used[cat].append(pick)
            r, c = pool[pick]
            vn = float(np.clip(r * res + self.rng.normal(0, 3.0), 3, self.north_m - 3))
            ve = float(np.clip(c * res + self.rng.normal(0, 3.0), 3, self.east_m - 3))

            ground_elev = float(np.nan_to_num(height_at_point(elev, vn, ve, res)))
            flood_d = float(np.nan_to_num(flood_at_point(flood, vn, ve, res)))
            bld_h = float(np.nan_to_num(height_at_point(heights, vn, ve, res)))
            veg = float(np.nan_to_num(height_at_point(self.vegetation.data, vn, ve, res)))

            on_rooftop = cat == "rooftop"
            in_water = cat == "water_edge"
            z = ground_elev + (bld_h if on_rooftop else 0.0)

            if in_water:
                posture = VictimPosture.IN_WATER_CLINGING
            elif on_rooftop:
                posture = VictimPosture.WAVING if self.rng.random() < 0.62 else VictimPosture.STANDING
            elif cat == "structure" and self.rng.random() < 0.4:
                posture = VictimPosture.PRONE_PARTIAL_BURIAL
            else:
                posture = VictimPosture(str(self.rng.choice([
                    VictimPosture.STANDING.value, VictimPosture.WAVING.value,
                    VictimPosture.SITTING.value, VictimPosture.LYING.value,
                ], p=[0.3, 0.3, 0.25, 0.15])))

            # --- visibility budget ------------------------------------------------
            occ = 0.10 + 0.70 * veg
            if posture == VictimPosture.PRONE_PARTIAL_BURIAL:
                occ = max(occ, 0.55)
            if in_water:
                occ = max(occ, 0.30 + 0.25 * float(np.clip(flood_d / 1.5, 0, 1)))
            if on_rooftop:
                occ = min(occ, 0.12)
            if cat == "structure":
                occ = max(occ, 0.35)

            # --- thermal budget ---------------------------------------------------
            body_temp = 33.5
            if in_water:
                # Evaporative + convective cooling: wet clothing reads much colder,
                # which is exactly why LWIR-only pipelines miss flood victims.
                body_temp = 28.5 - 2.0 * float(np.clip(flood_d / 1.5, 0, 1))
            if posture == VictimPosture.PRONE_PARTIAL_BURIAL:
                body_temp = 29.5
            if posture == VictimPosture.LYING:
                body_temp = 31.5
            if on_rooftop:
                body_temp = 32.0     # sun-warm roof reduces apparent contrast

            self.victims.append(Victim(
                vid=f"V{i:02d}", north=vn, east=ve, elevation=z, posture=posture,
                alive=True, group=f"G{i // 3}", category=cat,
                body_temp_c=body_temp + float(self.rng.normal(0, 0.7)),
                occlusion=float(np.clip(occ, 0.0, 0.95)),
                visual_salience=float(np.clip(0.80 - 0.55 * occ + self.rng.normal(0, 0.08), 0.05, 1.0)),
                moving=bool(self.rng.random() < 0.16),
                speed_ms=float(self.rng.uniform(0.2, 0.8)),
                heading_deg=float(self.rng.uniform(0, 360)),
                needs=("flotation" if in_water else
                       "evacuation" if on_rooftop else
                       "medical" if posture == VictimPosture.PRONE_PARTIAL_BURIAL else "medical"),
                in_water=in_water, on_rooftop=on_rooftop,
            ))

    def _place_distractors(self) -> None:
        spec = self.spec
        for i in range(spec.n_distractors):
            kind = str(self.rng.choice(
                ["animal", "hot_rock", "vehicle", "mannequin", "wet_cloth", "fire_barrel"],
                p=[0.22, 0.18, 0.16, 0.14, 0.12, 0.18]))
            dn = float(self.rng.uniform(10, self.north_m - 10))
            de = float(self.rng.uniform(10, self.east_m - 10))
            elev = float(np.nan_to_num(height_at_point(self.elevation.data, dn, de, self.res)))
            if kind == "animal":
                temp, size, sal, move, spd = 37.5, 0.7, 0.55, True, 1.6
            elif kind == "hot_rock":
                temp, size, sal, move, spd = 44.0 + float(self.rng.uniform(0, 20)), 1.1, 0.05, False, 0.0
            elif kind == "vehicle":
                temp, size, sal, move, spd = 41.0, 4.2, 0.8, False, 0.0
            elif kind == "mannequin":
                temp, size, sal, move, spd = 21.0, 0.6, 0.85, False, 0.0
            elif kind == "wet_cloth":
                temp, size, sal, move, spd = 17.0, 0.9, 0.6, False, 0.0
            else:  # fire_barrel - hot, small, flickering
                temp, size, sal, move, spd = 220.0, 0.6, 0.3, False, 0.0
            self.distractors.append(Distractor(
                did=f"D{i:02d}", kind=kind, north=dn, east=de, elevation=elev,
                temp_c=temp + float(self.rng.normal(0, 1.0)), size_m=size,
                visual_salience=sal, moving=move, speed_ms=spd,
                heading_deg=float(self.rng.uniform(0, 360)),
            ))

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def advance(self, dt: float, t: float) -> None:
        """Move dynamic entities and evolve the fire slightly."""
        for v in self.victims:
            v.advance(dt)
        for d in self.distractors:
            d.advance(dt)

    #: layers available to :meth:`sample_stack`, in a fixed order.
    LAYER_NAMES = (
        "surface_temp", "structure_height", "damage", "flood", "vegetation",
        "debris", "fire", "smoke", "road", "elevation", "texture", "albedo",
        "metal_roof", "chemical", "landslide", "powerline",
    )

    def _layer_map(self) -> Dict[str, "_Raster"]:
        return {
            "surface_temp": self.surface_temp,
            "structure_height": self.building_height,
            "damage": self.building_damage,
            "flood": self.flood_depth,
            "vegetation": self.vegetation,
            "debris": self.debris,
            "fire": self.fire,
            "smoke": self.smoke,
            "road": self.road_raster,
            "elevation": self.elevation,
            "texture": self.texture,
            "albedo": self.albedo_raster,
            "metal_roof": self.metal_roof,
            "chemical": self.chemical,
            "landslide": self.landslide,
            "powerline": self.powerline_raster,
        }

    def grid_index(self, north: np.ndarray, east: np.ndarray) -> GridIndex:
        return GridIndex(north, east, self.res, self.grid_shape)

    def sample_stack(self, names: Sequence[str], gi: GridIndex) -> np.ndarray:
        """Sample several layers at once -> ``(len(names), H, W)`` float32."""
        lm = self._layer_map()
        return np.stack([np.nan_to_num(lm[n].sample_idx(gi)) for n in names])

    def terrain_z(self, north: float | np.ndarray, east: float | np.ndarray) -> np.ndarray:
        return np.nan_to_num(self.elevation.sample(np.asarray(north), np.asarray(east)))

    def water_depth(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.flood_depth.sample(np.asarray(north), np.asarray(east)))

    def structure_height(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.building_height.sample(np.asarray(north), np.asarray(east)))

    def fire_intensity(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.fire.sample(np.asarray(north), np.asarray(east)))

    def smoke_density(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.smoke.sample(np.asarray(north), np.asarray(east)))

    def debris_density(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.debris.sample(np.asarray(north), np.asarray(east)))

    def vegetation_density(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.vegetation.sample(np.asarray(north), np.asarray(east)))

    def landslide_density(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.landslide.sample(np.asarray(north), np.asarray(east)))

    def chemical_density(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.chemical.sample(np.asarray(north), np.asarray(east)))

    def surface_temperature(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.surface_temp.sample(np.asarray(north), np.asarray(east)))

    def visual_texture(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.texture.sample(np.asarray(north), np.asarray(east)))

    def albedo(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.albedo_raster.sample(np.asarray(north), np.asarray(east)))

    def damage_level(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.building_damage.sample(np.asarray(north), np.asarray(east)))

    def is_road(self, north, east) -> np.ndarray:
        return np.nan_to_num(self.road_raster.sample(np.asarray(north), np.asarray(east)))

    def clearance(self, north, east) -> float:
        """Height of the tallest obstacle at a point (for obstacle avoidance)."""
        h = float(self.structure_height(north, east))
        return h

    def obstacle_field(self, resolution_m: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Regular grid of structure heights for the obstacle-avoidance layer."""
        n_axis = np.arange(0.0, self.north_m + 1e-9, resolution_m)
        e_axis = np.arange(0.0, self.east_m + 1e-9, resolution_m)
        nn, ee = np.meshgrid(n_axis, e_axis, indexing="ij")
        struct = np.nan_to_num(self.building_height.sample(nn, ee))
        return nn, ee, struct

    def passability_field(self, resolution_m: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cost of ground access for rescue teams (lower is better).

        Combines inundation depth, debris, landslide scar, fire and damaged
        structures into a single normalised 0..1 cost surface.  The routing
        layer turns this into safe access corridors - the "safe access routes"
        deliverable in the problem statement.
        """
        n_axis = np.arange(0.0, self.north_m + 1e-9, resolution_m)
        e_axis = np.arange(0.0, self.east_m + 1e-9, resolution_m)
        nn, ee = np.meshgrid(n_axis, e_axis, indexing="ij")
        flood = np.nan_to_num(self.flood_depth.sample(nn, ee))
        debris = np.nan_to_num(self.debris.sample(nn, ee))
        slide = np.nan_to_num(self.landslide.sample(nn, ee))
        fire = np.nan_to_num(self.fire.sample(nn, ee))
        chem = np.nan_to_num(self.chemical.sample(nn, ee))
        dmg = np.nan_to_num(self.building_damage.sample(nn, ee))
        struct = np.nan_to_num(self.building_height.sample(nn, ee))
        cost = np.zeros_like(nn)
        cost += 3.0 * np.clip(flood / 1.5, 0, 1)          # wading > 1.5 m is impassable
        cost += np.where(flood > 1.5, 5.0, 0.0)
        cost += 1.2 * np.clip(debris, 0, 1)
        cost += 2.0 * np.clip(slide, 0, 1)
        cost += 4.0 * np.clip(fire, 0, 1)
        cost += 3.0 * np.clip(chem, 0, 1)
        cost += 0.8 * np.clip(dmg, 0, 1)
        cost += np.where(struct > 1.0, 2.5, 0.0)            # cannot walk through a building
        cost += 0.25                                          # baseline effort
        return nn, ee, cost.astype(np.float32)

    def geo(self, north: float, east: float, up: float = 0.0) -> GeoPoint:
        return local_to_wgs84(north, east, -up, self.origin)

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.spec.name,
            "disaster": self.spec.disaster,
            "area_m2": self.north_m * self.east_m,
            "victims": len(self.victims),
            "victims_in_water": sum(1 for v in self.victims if v.in_water),
            "victims_on_rooftop": sum(1 for v in self.victims if v.on_rooftop),
            "distractors": len(self.distractors),
            "buildings": len(self.buildings),
            "buildings_damaged": sum(1 for b in self.buildings if b.damage >= 2),
            "powerlines_downed": sum(1 for p in self.powerlines if p.downed),
            "fire_sites": len(self.fire_sites),
            "inundated_fraction": float((self.flood_depth.data > 0.1).mean()),
            "wind_ms": self.wind.speed_ms,
        }


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _index(north: float, east: float, res: float) -> Tuple[int, int]:
    return int(round(north / res)), int(round(east / res))


def _at(grid: np.ndarray, north: float, east: float, res: float) -> float:
    r, c = _index(north, east, res)
    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        return float(grid[r, c])
    return 0.0


flood_at_point = _at
height_at_point = _at


def solar_loading(hour: float, cloud_cover: float) -> float:
    """Normalised 0..1 solar heating of surfaces at local solar hour ``hour``.

    Peaks at local noon and is attenuated by cloud.  This single scalar drives
    the thermal contrast of the scene, which is why dawn/dusk sorties are the
    gold standard for LWIR person detection: the background is cold and the
    body is warm.
    """
    x = math.sin(math.pi * (hour - 6.0) / 12.0)
    sun = max(0.0, x) ** 1.25
    return float(sun * (1.0 - 0.65 * float(np.clip(cloud_cover, 0, 1))))
