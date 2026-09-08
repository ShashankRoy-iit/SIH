"""Mission execution: fly the survey, see the ground, report what matters.

This module is the join between three subsystems that were each developed
against their own tests, and its job is to make sure the seams are honest.

What it does
------------
Flies a :class:`~sar.decision.coverage.SurveyPlan` by sending MAVLink velocity
and position commands over a :class:`~sar.mavlink.MavConnection`.  It never
touches the vehicle plant directly.  That is a deliberate constraint rather than
an inconvenience: a runner that moved the aircraft by assigning to its state
would still produce a convincing log, and would tell you nothing about whether
the guidance, the link budget or the prearm checks work.  Everything here goes
over the wire, so the same runner drives MiniSITL, ArduPilot SITL and the TBS
Lucid airframe without modification.

At each perception tick it renders an RGB and an LWIR frame from the world at the
aircraft's current pose, runs the full
:class:`~sar.perception.pipeline.PerceptionPipeline`, and publishes whatever
survivors and hazards that produced onto the
:class:`~sar.comms.link.TelemetryUplink`.

The two states that are not the same state
------------------------------------------
Rendering uses the aircraft's **true** pose; geo-tagging uses the **estimated**
one.  Conflating them is the most tempting shortcut in this module and it erases
the result the GPS-denied work exists to produce.

Photons arrive at the sensor from where the aircraft physically is.  The
geo-tag attached to a detection is computed from where the aircraft *believes* it
is.  In normal flight those differ by a metre or two and nobody cares.  Under
denial they diverge, and the divergence is exactly the geotagging error a
rescue team would inherit when they walk to a reported coordinate.  A runner that
rendered from the estimate would produce a perfect geotag under total denial,
which would be a simulation artifact and not a finding.

So: ``state_fn`` supplies truth (from MiniSITL, or unavailable on hardware),
``nav_fn`` supplies the estimate-derived :class:`~sar.perception.geotag.NavQuality`
from telemetry.  On real hardware there is no truth, and ``state_fn`` falls back
to the telemetry-derived pose - at which point the geotag error is not measurable
in flight and has to come from the pre-flight characterization instead.  That
fallback is explicit in :func:`plant_from_telemetry` rather than hidden.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.comms.link import Message, MessageKind, PriorityTier, TelemetryUplink
from sar.core.geo import GeoPoint, local_to_wgs84
from sar.decision.coverage import (BeliefMap, BoustrophedonPlanner, CoverageGrid,
                                   Lane, SurveyPlan)
from sar.perception.geotag import NavQuality, PixelGeoTagger
from sar.perception.pipeline import PerceptionPipeline, PerceptionProduct
from sar.vehicle.dynamics import PlantState

log = logging.getLogger("sar.mission")

__all__ = ["MissionRunner", "MissionReport", "plant_from_telemetry",
           "build_pipeline"]


# --------------------------------------------------------------------------- #
# Telemetry -> a pose the renderer can use
# --------------------------------------------------------------------------- #
def _is_armed(tel: Any) -> bool:
    """Whether the vehicle is armed, from a telemetry snapshot.

    ``Telemetry.armed`` is an :class:`ArmState` dataclass carrying the arm bits,
    the ECS status and the prearm reasons - not a boolean.  ``bool(tel.armed)`` is
    therefore always True, and ``not tel.armed`` never fires.  Written as a
    standalone guard because the failure it prevents is invisible: an RTL loop
    waiting for disarm simply runs to its timeout, which looks like slow landing
    rather than a condition that can never be met.
    """
    a = getattr(tel, "armed", None)
    if a is None:
        return False
    if isinstance(a, bool):
        return a
    return bool(getattr(a, "armed", False))


def plant_from_telemetry(tel: Any, home: GeoPoint,
                         last: Optional[PlantState] = None) -> PlantState:
    """Build a :class:`PlantState` from MAVLink telemetry alone.

    Used when there is no ground truth - which is every real flight, and any
    simulation run against a genuine ArduPilot binary rather than MiniSITL.  The
    result is the aircraft's *estimate* of its own state dressed up as a pose, so
    rendering from it produces frames consistent with what the aircraft believes
    rather than with what is physically true.

    That is the correct thing to do in the absence of truth, and it is also why
    geotag error cannot be measured this way: the error would be identically
    zero by construction.  Callers that have truth should use it and reserve this
    for the hardware path.
    """
    st = last.copy() if last is not None else PlantState()
    lat = getattr(tel, "lat", None)
    lon = getattr(tel, "lon", None)
    if lat is not None and lon is not None and abs(lat) > 1e-6:
        from sar.core.geo import wgs84_to_local
        n, e, d = wgs84_to_local(GeoPoint(float(lat), float(lon), 0.0), home)
        alt_rel = float(getattr(tel, "alt_rel_m", 0.0) or 0.0)
        st.pos = np.array([n, e, -alt_rel], dtype=np.float64)
    st.vel = np.array([float(getattr(tel, "vn", 0.0) or 0.0),
                       float(getattr(tel, "ve", 0.0) or 0.0),
                       float(getattr(tel, "vd", 0.0) or 0.0)], dtype=np.float64)
    st.euler = np.array([math.radians(float(getattr(tel, "roll_deg", 0.0) or 0.0)),
                         math.radians(float(getattr(tel, "pitch_deg", 0.0) or 0.0)),
                         math.radians(float(getattr(tel, "yaw_deg", 0.0) or 0.0))],
                        dtype=np.float64)
    st.altitude_msl = float(getattr(tel, "alt_msl_m", home.alt) or home.alt)
    st.on_ground = not _is_armed(tel)
    return st


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def build_pipeline(world: Any, rgb_spec: Any, lwir_spec: Any,
                   origin: Optional[GeoPoint] = None,
                   **kwargs: Any) -> PerceptionPipeline:
    """Assemble the perception stack against a world and two camera specs.

    Factored out of :class:`MissionRunner` because the evaluation scripts need
    exactly the same pipeline without a vehicle attached, and a divergence
    between the two would mean the published detector numbers came from a
    different configuration than the one that flies.
    """
    origin = origin or world.origin
    terrain_fn = None
    if hasattr(world, "terrain_z"):
        terrain_fn = lambda n, e: float(np.nan_to_num(world.terrain_z(n, e)))
    tagger = PixelGeoTagger(width=rgb_spec.width, height=rgb_spec.height,
                            hfov_deg=rgb_spec.hfov_deg, origin=origin,
                            terrain_fn=terrain_fn,
                            terrain_sigma_m=kwargs.pop("terrain_sigma_m", 1.5))
    pipe = PerceptionPipeline(
        origin=origin, tagger=tagger,
        world_extent_m=(world.north_m, world.east_m), **kwargs)
    if terrain_fn is not None:
        pipe.terrain_fn = terrain_fn
    if hasattr(world, "flood_depth"):
        pipe.water_fn = lambda n, e: float(
            np.nan_to_num(world.flood_depth.at(n, e)))
    pipe.set_environment(
        ambient_c=getattr(world.spec, "ambient_c", 22.0),
        wind_speed_ms=getattr(world.wind, "speed_ms", 3.0),
        wind_from_deg=getattr(world.wind, "direction_deg", 210.0),
        water_temp_c=getattr(world.spec, "water_temp_c", None))
    return pipe


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class MissionReport:
    """Everything one sortie produced, in a form that can be diffed.

    Written to JSON under ``artifacts/`` so two runs of the same scenario and
    seed can be compared numerically rather than by eye.  Fields are rounded on
    the way out: a report full of 17-significant-figure floats is unreadable and
    the extra digits are noise from the integrator, not signal.
    """

    scenario: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    armed: bool = False
    landed: bool = False
    crashed: bool = False
    flight_time_s: float = 0.0
    distance_m: float = 0.0
    energy_wh: float = 0.0
    battery_pct: float = 100.0
    plan: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    belief: Dict[str, Any] = field(default_factory=dict)
    comms: Dict[str, Any] = field(default_factory=dict)
    perception: Dict[str, Any] = field(default_factory=dict)
    survivors_reported: List[Dict[str, Any]] = field(default_factory=list)
    hazards_reported: List[Dict[str, Any]] = field(default_factory=list)
    #: Ground truth from the world model, for scoring.  Never transmitted - this
    #: exists only so the report can say whether the mission worked.
    truth: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["duration_s"] = round(self.duration_s, 2)
        d["flight_time_s"] = round(self.flight_time_s, 2)
        d["distance_m"] = round(self.distance_m, 1)
        d["energy_wh"] = round(self.energy_wh, 2)
        d["battery_pct"] = round(self.battery_pct, 1)
        return d

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return p


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class MissionRunner:
    """Executes a survey sortie end to end.

    Parameters
    ----------
    conn
        A connected :class:`~sar.mavlink.MavConnection`.  Commands go out and
        telemetry comes in through it; nothing else touches the vehicle.
    world, rgb_spec, lwir_spec
        The scenario.  Built by :func:`sar.sim.scenario.build_reference_scenario`.
    state_fn
        ``(t) -> PlantState`` supplying the aircraft's TRUE pose for rendering.
        ``None`` falls back to :func:`plant_from_telemetry`, which is what real
        hardware does.
    uplink
        The comms path.  ``None`` builds a default LoRa uplink, which is the
        narrowest alert-capable link and therefore the honest default.
    gcs_position
        Where the ground station is, in local NED metres from the mission origin.
        Drives the link range calculation, so out-of-range behaviour emerges from
        the flight path rather than being scripted.
    """

    def __init__(self, conn: Any, world: Any, rgb_spec: Any, lwir_spec: Any,
                 pipeline: Optional[PerceptionPipeline] = None,
                 state_fn: Optional[Callable[[float], PlantState]] = None,
                 uplink: Optional[TelemetryUplink] = None,
                 plan: Optional[SurveyPlan] = None,
                 gcs_position: Tuple[float, float] = (0.0, 0.0),
                 origin: Optional[GeoPoint] = None,
                 perception_hz: float = 2.0,
                 control_hz: float = 5.0,
                 arrival_tolerance_m: float = 6.0,
                 scenario_name: str = "",
                 record_frames: bool = False,
                 artifacts_dir: Optional[Path] = None,
                 vehicle_info_fn: Optional[Callable[[], Dict[str, Any]]] = None,
                 ) -> None:
        from sar.sim.renderer import CameraRenderer

        self.conn = conn
        self.world = world
        self.origin = origin or world.origin
        self.rgb_spec = rgb_spec
        self.lwir_spec = lwir_spec
        self.state_fn = state_fn
        self.pipeline = pipeline or build_pipeline(world, rgb_spec, lwir_spec,
                                                   self.origin)
        self.uplink = uplink if uplink is not None else TelemetryUplink()
        self.gcs = np.array(gcs_position, dtype=np.float64)
        self.perception_hz = float(perception_hz)
        self.control_hz = float(control_hz)
        self.arrival_tol = float(arrival_tolerance_m)
        self.scenario_name = scenario_name
        self.record_frames = bool(record_frames)
        self.artifacts = Path(artifacts_dir) if artifacts_dir else None
        self.vehicle_info_fn = vehicle_info_fn

        self.rgb_render = CameraRenderer(world, rgb_spec, seed=11)
        self.lwir_render = CameraRenderer(world, lwir_spec, seed=12)

        self.coverage = CoverageGrid(world.north_m, world.east_m,
                                     resolution_m=4.0)
        self.belief = BeliefMap(world.north_m, world.east_m, resolution_m=6.0,
                                expected_survivors=float(len(world.victims) or 8))
        self.planner = BoustrophedonPlanner()
        self.plan = plan

        self.report = MissionReport(scenario=scenario_name)
        self._t0 = 0.0
        self._last_perception = -1e9
        self._last_telemetry_pub = -1e9
        self._last_nav_pub = -1e9
        self._last_coverage_pub = -1e9
        self._reported_survivors: Dict[int, float] = {}
        #: Distinct survivor *locations* reported, as opposed to distinct track
        #: ids.  See :meth:`_resolve_site`.
        self._sites: List[Dict[str, Any]] = []
        self.min_merge_radius_m = 15.0
        self.merge_sigma_k = 3.0
        #: How far a merged site may drift from its first observation before
        #: further observations are folded in without moving it.  See
        #: :meth:`_resolve_site`.
        self.max_site_drift_m = 40.0
        self._rejected_out_of_area = 0
        self._reported_hazards: set = set()
        self._lane_idx = 0
        self._lane_s = 0.0
        #: "transit" to the lane start, then "survey" along it.  See
        #: :meth:`_advance_lane` for why the two are separate legs.
        self._lane_phase = "transit"
        self.transit_speed_ms = 10.0
        self._phase = "init"
        self._frames_rendered = 0
        self._perception_ms: List[float] = []
        self._track_first_report: Dict[int, float] = {}
        self._stop = False
        self._nav_monitor = None
        self._source_manager = None
        self._feeder = None
        self._last_pos: Optional[np.ndarray] = None
        self._distance_m = 0.0
        self._lane_heading_set = -1

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def make_plan(self, perception_hz: Optional[float] = None,
                  altitude_agl_m: Optional[float] = None,
                  max_duration_s: Optional[float] = None,
                  area_north_m: Optional[float] = None,
                  area_east_m: Optional[float] = None,
                  area_origin: Optional[Tuple[float, float]] = None,
                  ) -> SurveyPlan:
        """Derive a survey plan from the camera footprint at survey altitude.

        The swath and along-track extents come from the camera spec at the chosen
        altitude, so the lane spacing and the ground speed both follow from the
        sensor rather than being tuned per scenario.  Change the lens or the
        altitude and the plan changes with it, which is the property that makes
        it usable on a different airframe.
        """
        hz = perception_hz or self.perception_hz
        alt = altitude_agl_m
        if alt is None:
            alt = self._choose_altitude(hz, max_duration_s,
                                        area_north_m or self.world.north_m,
                                        area_east_m or self.world.east_m)
        # Centre the search box on the belief peak when the caller asks for a
        # sub-area but not for a specific corner.  Anchoring at (0, 0) searches
        # whatever happens to sit at the map origin, which in the reference flood
        # basin is empty water: a 200 m box there contains none of the twelve
        # survivors, and the sortie returns a perfect flight with zero detections
        # and no indication that the search was simply in the wrong place.
        on, oe = 0.0, 0.0
        an = area_north_m or self.world.north_m
        ae = area_east_m or self.world.east_m
        if area_origin is not None:
            on, oe = float(area_origin[0]), float(area_origin[1])
        elif an < self.world.north_m or ae < self.world.east_m:
            peak = self.belief.peak()
            if peak is not None:
                on = float(np.clip(peak[0] - an / 2.0, 0.0,
                                   max(0.0, self.world.north_m - an)))
                oe = float(np.clip(peak[1] - ae / 2.0, 0.0,
                                   max(0.0, self.world.east_m - ae)))
                log.info("search area centred on belief peak (%.0f, %.0f) -> "
                         "box origin (%.0f, %.0f), %.0f x %.0f m",
                         peak[0], peak[1], on, oe, an, ae)

        swath = self.lwir_spec.swath_width_m(alt)
        # The LWIR array is coarser than the RGB one, so it sets the spacing:
        # planning to the RGB footprint would leave thermal-only gaps between
        # lanes, and a survivor found by RGB alone at night is not found at all.
        along = swath * (self.lwir_spec.height / max(1, self.lwir_spec.width))
        self.plan = self.planner.plan(
            extent_north_m=an, extent_east_m=ae,
            origin_north_m=on, origin_east_m=oe,
            swath_width_m=swath, along_track_m=max(along, swath * 0.5),
            perception_hz=hz, altitude_agl_m=alt, belief=self.belief)

        # --- truncate to what the airframe can actually fly ----------------
        # The 900 x 900 m reference basin needs ~25 km of lane at a spacing that
        # gives useful thermal GSD, which is 50+ minutes at survey speed.  A
        # small multirotor on 6S does not have that.  Pretending otherwise
        # produces a plan the runner cannot finish and a coverage number that
        # never arrives, so the plan is cut to the endurance budget instead - and
        # because the lanes were ordered by belief, what gets cut is the ground
        # least likely to hold anyone.
        if max_duration_s:
            kept: List[Lane] = []
            t = 0.0
            for ln in self.plan.lanes:
                if t + ln.duration_s > max_duration_s:
                    break
                kept.append(ln)
                t += ln.duration_s
            cut = len(self.plan.lanes) - len(kept)
            if cut:
                log.warning("plan truncated to endurance: dropped %d of %d lanes "
                            "(%.0f s of %.0f s needed); the dropped lanes are the "
                            "lowest-belief ones because ordering is belief-"
                            "weighted", cut, len(self.plan.lanes),
                            self.plan.total_duration_s, max_duration_s)
                self.plan.lanes = kept
                self.report.faults.append(
                    f"plan truncated: {cut} lanes exceed the {max_duration_s:.0f} s "
                    f"endurance budget")

        log.info("plan: %d lanes, %.0f m spacing, %.1f m/s, alt %.0f m AGL, "
                 "%.0f m total, ~%.0f s", len(self.plan.lanes),
                 self.plan.lane_spacing_m,
                 self.plan.lanes[0].speed_ms if self.plan.lanes else 0.0,
                 alt, self.plan.total_length_m, self.plan.total_duration_s)
        self.plan.search_box = {
            "north_m": [round(on, 1), round(on + an, 1)],
            "east_m": [round(oe, 1), round(oe + ae, 1)],
            "area_m2": round(an * ae, 1),
            "victims_inside": sum(1 for v in getattr(self.world, "victims", [])
                                  if on <= v.north <= on + an and oe <= v.east <= oe + ae),
        }
        n_in = self.plan.search_box["victims_inside"]
        log.info("search box n %.0f-%.0f m, e %.0f-%.0f m (%.0f m^2): %d of %d "
                 "survivors in the world are inside it - recall is bounded by "
                 "that, not by the detector", on, on + an, oe, oe + ae, an * ae,
                 n_in, len(getattr(self.world, "victims", [])))
        self.report.plan = self.plan.to_dict()
        return self.plan

    # ------------------------------------------------------------------ #
    # Nav quality from telemetry
    #: Survey altitude ladder, best detection first.
    #:
    #: Ordered low-to-high because the measured trade is not symmetric.  From the
    #: project's own aimed-mode calibration (``artifacts/detector_aimed.json``):
    #: ``recall_of_resolvable`` is 1.00 at 35, 50 and 70 m - the detector finds
    #: every survivor it can resolve at any of them.  What changes with altitude
    #: is how many survivors are resolvable at all (12 of 36 at 35 m, 9 of 36 at
    #: 70 m) and how much junk is detected alongside them (background false
    #: alarms rise 0.017 -> 0.117 per frame, a factor of seven).
    #:
    #: So altitude buys coverage rate and pays for it in both recall and
    #: precision, and the previous heuristic - "the highest altitude whose GSD is
    #: still under 0.20 m/px" - landed on 75 m, which is the worst available
    #: choice on both axes at once.  The ladder starts where detection is best
    #: and only climbs when endurance forces it.
    ALTITUDE_LADDER = (35.0, 45.0, 55.0, 70.0, 90.0, 110.0)

    def _choose_altitude(self, hz: float, budget_s: Optional[float],
                         area_n: float, area_e: float) -> float:
        """Lowest altitude whose survey fits the endurance budget.

        The constraint is genuinely binding and the trade should be visible
        rather than buried in a default.  Lane count scales as 1/altitude, so
        halving the altitude roughly doubles the sortie time; a small multirotor
        on 6S cannot lawnmower a 900 m basin at 35 m.  When nothing on the ladder
        fits, this returns the highest rung and lets the plan truncator cut lanes
        - and logs that the coverage number about to be reported is going to be
        low because of endurance, not because the sensor failed.
        """
        chosen = self.ALTITUDE_LADDER[0]
        for alt in self.ALTITUDE_LADDER:
            swath = self.lwir_spec.swath_width_m(alt)
            spacing = swath * (1.0 - self.planner.side_overlap)
            n_lanes = max(1, math.ceil(area_e / spacing))
            v = min(self.planner.max_speed_ms,
                    hz * swath * 0.75 * (1.0 - self.planner.forward_overlap))
            v = max(v, 0.5)
            need = n_lanes * area_n / v
            if budget_s is None or need <= budget_s:
                chosen = alt
                log.info("survey altitude %.0f m AGL: GSD %.3f m/px, %d lanes, "
                         "%.0f m, ~%.0f s (budget %.0f s)", alt,
                         self.lwir_spec.gsd_at(alt), n_lanes, n_lanes * area_n,
                         need, budget_s or float("inf"))
                return chosen
            chosen = alt
        log.warning("no altitude on the ladder fits the %.0f s endurance budget "
                    "over %.0f x %.0f m; using %.0f m AGL and the plan will be "
                    "truncated.  Low coverage in this sortie is an endurance "
                    "limit, not a sensor failure - a real operation would fly "
                    "this area as a coarse pass followed by low-altitude "
                    "confirmation over the hot spots.",
                    budget_s or 0, area_n, area_e, chosen)
        return chosen

    # ------------------------------------------------------------------ #
    def nav_quality(self, tel: Any, assessment: Any = None) -> NavQuality:
        """Turn a telemetry snapshot into what the geo-tagger needs to know.

        The important field is ``seconds_since_fix``, because under denial the
        position error is a function of how long the aircraft has been
        dead-reckoning rather than a constant.  A geo-tagger that reports a
        fixed sigma during a two-minute denial produces error ellipses that are
        confidently wrong, and a rescue team walking to the centre of one is
        worse off than one told the position is uncertain.
        """
        sigma = float(getattr(tel, "horizontal_pos_sigma_m",
                              lambda: 100.0)()) \
            if callable(getattr(tel, "horizontal_pos_sigma_m", None)) \
            else float(getattr(tel, "horizontal_pos_sigma_m", 100.0) or 100.0)
        denied = bool(getattr(tel, "gps_denied", False))
        state = assessment.state.value if assessment is not None else \
            ("gnss" if not denied else "dead_reckoning")
        source = {"gnss": "gnss", "external_nav": "ekf_external_nav",
                  "optical_flow": "vision", "dead_reckoning": "dead_reckoning",
                  "denied": "dead_reckoning"}.get(state, state)
        since = 0.0
        if assessment is not None:
            since = float(getattr(assessment, "since_fix_s", 0.0) or 0.0)
        elif denied:
            since = float(getattr(tel, "seconds_since_gps", 0.0) or 0.0)
        # Drift rate differs by an order of magnitude between a visual-inertial
        # solution holding ~0.15 m/s and pure inertial dead reckoning, and using
        # the wrong one here is the difference between an honest ellipse and a
        # fiction.
        drift = {"ekf_external_nav": 0.15, "vision": 0.20,
                 "dead_reckoning": 0.80}.get(source, 0.05)
        return NavQuality(
            pos_sigma_m=sigma,
            vel_sigma_ms=float(getattr(tel, "vel_sigma_ms", 0.3) or 0.3),
            att_sigma_deg=float(getattr(tel, "att_sigma_deg", 0.8) or 0.8),
            alt_sigma_m=float(getattr(tel, "alt_sigma_m", 1.0) or 1.0),
            seconds_since_fix=since,
            n_satellites=int(getattr(tel, "n_satellites", 0) or 0),
            source=source, hdop=float(getattr(tel, "hdop", 99.0) or 99.0),
            drift_rate_ms=drift)

    # ------------------------------------------------------------------ #
    # Perception tick
    # ------------------------------------------------------------------ #
    def perceive(self, t_mission: float, state: PlantState,
                 nav: NavQuality) -> PerceptionProduct:
        """Render both modalities and run the pipeline once."""
        t0 = time.perf_counter()
        frames = [self.lwir_render.render(state, t_mission),
                  self.rgb_render.render(state, t_mission)]
        self._frames_rendered += 2
        product = self.pipeline.process(
            frames, pos_ned=state.pos.copy(), euler=state.euler.copy(),
            t=t_mission, velocity_ned=state.vel.copy(), nav=nav)
        self._perception_ms.append((time.perf_counter() - t0) * 1000.0)

        # --- coverage and belief from the frames themselves ----------------
        # Driven by the rendered footprint rather than by the lane geometry: a
        # lane that was flown but produced a smoke-obscured frame has not
        # covered anything, and bookkeeping from the plan alone would say it had.
        self._update_coverage(frames, product, nav, t_mission)

        if self.record_frames and self.artifacts is not None:
            self._save_frames(frames, t_mission)
        return product

    def _update_coverage(self, frames: Sequence[Any],
                         product: PerceptionProduct, nav: NavQuality,
                         t_mission: float) -> None:
        """Fold one frame pair into the coverage grid and the belief map."""
        pts_n: List[float] = []
        pts_e: List[float] = []
        p_det: List[float] = []
        gsd: List[float] = []
        for f in frames:
            fp = getattr(f, "footprint", None) or ()
            if len(fp) < 4:
                continue
            g = float(getattr(f, "gsd_m", 0.0) or 0.0)
            # Detection capability falls off with ground sample distance.  The
            # exponent is calibrated against the detector's own measured recall
            # versus GSD rather than chosen: at 0.4 m/px a person is a handful of
            # pixels and recall collapses, and the coverage map has to say so.
            cap = float(np.clip((0.22 / max(g, 1e-3)) ** 1.3, 0.02, 0.97))
            # Denial does not reduce the ability to *see* someone; it reduces the
            # ability to say where they are.  Those are different quantities and
            # conflating them makes a GPS-denied pass look like a blind one.
            for (pn, pe) in self._fill_footprint(fp):
                pts_n.append(pn)
                pts_e.append(pe)
                p_det.append(cap)
                gsd.append(g)
        if not pts_n:
            return
        self.coverage.mark(pts_n, pts_e, p_det, gsd, t_mission)

        detected_cells = set()
        for obs, tag in zip(product.observations, product.geo_tags):
            if tag is None:
                continue
            detected_cells.add((round(tag.north_m / self.belief.res),
                                round(tag.east_m / self.belief.res)))
        det_flags = [((round(n / self.belief.res), round(e / self.belief.res))
                      in detected_cells) for n, e in zip(pts_n, pts_e)]
        self.belief.observe(pts_n, pts_e, p_det, detected=det_flags)

    def _fill_footprint(self, fp: Sequence[Tuple[float, float]]
                        ) -> List[Tuple[float, float]]:
        """Sample the interior of a ground footprint, not just its corners.

        ``Frame.footprint`` is four corner points.  Marking only those would
        touch four cells of a 4 m grid per frame - roughly 64 m^2 out of a 2100
        m^2 swath - and the coverage map would report the sortie as having
        searched almost nothing, or (worse, if the grid were coarser) as having
        searched everything.  The interior is the coverage.

        Bilinear interpolation across the corners is exact for the nadir case,
        where the footprint is a rectangle possibly rotated by heading, and is a
        good approximation for the mild perspective of a slightly pitched camera.
        Sampled to the coverage grid resolution so the count scales with the
        swath rather than being a fixed oversample of a small frame.
        """
        q = np.asarray(fp[:4], dtype=np.float64)
        span = max(float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1])))
        n = int(np.clip(math.ceil(span / self.coverage.res), 2, 24))
        u = np.linspace(0.0, 1.0, n)
        # Corners ordered so that (0,0)->(1,0) and (0,1)->(1,1) are the two
        # across-track edges; the renderer emits them consistently around the
        # quadrilateral.
        top = q[0] * (1 - u)[:, None] + q[1] * u[:, None]
        bot = q[3] * (1 - u)[:, None] + q[2] * u[:, None]
        out: List[Tuple[float, float]] = []
        for a, b in zip(top, bot):
            for v in u:
                pt = a * (1 - v) + b * v
                out.append((float(pt[0]), float(pt[1])))
        return out

    def _save_frames(self, frames: Sequence[Any], t_mission: float) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            d = self.artifacts / "frames"
            d.mkdir(parents=True, exist_ok=True)
            for f in frames:
                img = np.asarray(f.image)
                if img.ndim == 3:
                    img = img[..., :3]
                lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
                norm = ((img - lo) / max(hi - lo, 1e-6))
                plt.imsave(d / f"{f.kind}_{int(t_mission*1000):07d}.png",
                           np.clip(norm, 0, 1))
        except Exception as exc:                       # pragma: no cover
            log.debug("frame save failed: %r", exc)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def publish(self, product: PerceptionProduct, t_mission: float,
                nav: NavQuality) -> None:
        """Push a perception product onto the uplink.

        The filtering here is what keeps a low-bandwidth link usable.  A survivor
        seen continuously for a minute produces hundreds of products; the uplink
        gets one DISCOVER and an UPDATE only when something a responder would act
        on has changed.  Without that, the first person found consumes the entire
        link budget and the second is never reported at all.
        """
        for s in product.survivors:
            if float(s.confidence) < 0.45:
                continue
            site = self._resolve_site(s)
            if site is None:
                continue
            tid = int(s.track_id)
            first = site["is_new"]
            # Keyed by location, not by track: fragments of one survivor become
            # revisions of one report rather than several reports.
            self.uplink.publish_survivor(
                s, first_report=first, key=f"p{site['site']['site_id']}")
            if first:
                self._reported_survivors[tid] = t_mission
                self._track_first_report[tid] = t_mission
                self.report.timeline.append({
                    "t": round(t_mission, 1), "event": "survivor_reported",
                    "track_id": tid, "lat": round(s.point.lat, 6),
                    "lon": round(s.point.lon, 6), "sigma_m": round(s.sigma_m, 2),
                    "priority": s.priority.value,
                    "nav_source": nav.source,
                    "nav_sigma_m": round(nav.effective_pos_sigma(), 2),
                })
                log.info("SURVIVOR track=%d prio=%s conf=%.2f sigma=%.1fm "
                         "nav=%s(%.1fm) @ n=%.0f e=%.0f", tid, s.priority.value,
                         s.confidence, s.sigma_m, nav.source,
                         nav.effective_pos_sigma(), s.north_m, s.east_m)

        for h in product.new_hazards:
            key = self._hazard_key(h)
            if key in self._reported_hazards:
                continue
            self._reported_hazards.add(key)
            self.uplink.publish_hazard(h, key=key, alert=True)
            self.report.timeline.append({
                "t": round(t_mission, 1), "event": "hazard_reported",
                "key": key,
                "hazard_class": getattr(h, "hazard_class", None)
                or (h.get("hazard_class") if isinstance(h, dict) else None),
            })

        if t_mission - self._last_nav_pub >= 2.0:
            self._last_nav_pub = t_mission
            self.uplink.publish_nav({
                "source": nav.source,
                "pos_sigma_m": round(nav.effective_pos_sigma(), 2),
                "since_fix_s": round(nav.seconds_since_fix, 1),
                "n_satellites": nav.n_satellites, "hdop": round(nav.hdop, 2),
                "denied": nav.denied, "drift_rate_ms": nav.drift_rate_ms,
            })
        if t_mission - self._last_coverage_pub >= 5.0:
            self._last_coverage_pub = t_mission
            self.uplink.publish_coverage({
                **self.coverage.summary(t_mission),
                "expected_remaining": round(self.belief.expected_remaining(), 2),
                "progress": round(self.belief.progress(), 4),
            })

    def _resolve_site(self, s: Any) -> Optional[Dict[str, Any]]:
        """Decide whether a survivor is a new location or one already reported.

        Track identity is not location identity.  The tracker assigns a new id
        whenever association fails - which happens routinely when the platform
        moves 8 m/s and a perception cycle takes most of a second, so the same
        person re-acquires a fresh id several times over one lane.  Reporting per
        track therefore puts six or eight markers on the dashboard for one
        survivor, and each one costs an alert on a link that cannot afford it.

        A command centre consumes *locations*, not track ids.  So reports are
        merged spatially: a survivor within the merge radius of one already
        reported is an update to that report, not a new one.  The radius scales
        with the reported uncertainty rather than being a fixed number, because a
        merge distance that is right for a 1.5 m geo-tag is wrong for a 25 m one.

        Also rejected here: a geo-tag outside the world extent.  Those are not
        marginal detections, they are projection failures - a survivor reported at
        n = 2151 m in a 900 m basin is a bug wearing a coordinate, and sending it
        to a rescue team is worse than sending nothing.
        """
        n, e = float(s.north_m), float(s.east_m)
        sigma = float(s.sigma_m)
        extent = (self.world.north_m, self.world.east_m)
        if not (-50.0 <= n <= extent[0] + 50.0 and -50.0 <= e <= extent[1] + 50.0):
            self._rejected_out_of_area += 1
            log.warning("discarded survivor track=%d: geo-tag n=%.0f e=%.0f lies "
                        "outside the %.0f x %.0f m world - projection failure, "
                        "not a marginal detection", int(s.track_id), n, e,
                        extent[0], extent[1])
            return None

        radius = max(self.min_merge_radius_m, self.merge_sigma_k * sigma)
        for site in self._sites:
            if math.hypot(n - site["north_m"], e - site["east_m"]) <= radius:
                # Move the site toward the new observation, but bound how far it
                # can travel from its anchor.  Merging is transitive: A matches B
                # and B matches C, so without a bound a chain of marginal
                # detections walks a single report hundreds of metres from where
                # it started while every individual step looks reasonable.  The
                # anchor is the first observation - not necessarily the best, but
                # the one with the least accumulated drift - and the site is
                # pinned within ``max_site_drift_m`` of it.
                drift = math.hypot(n - site["anchor_n"], e - site["anchor_e"])
                if drift > self.max_site_drift_m:
                    site["drift_rejected"] += 1
                    site["n_reports"] += 1
                    site["tracks"].add(int(s.track_id))
                    return {"is_new": False, "site": site}
                w = 1.0 / site["n_reports"]
                site["north_m"] += (n - site["north_m"]) * w
                site["east_m"] += (e - site["east_m"]) * w
                site["sigma_m"] = min(site["sigma_m"], sigma) if sigma > 0 \
                    else site["sigma_m"]
                site["tracks"].add(int(s.track_id))
                site["n_reports"] += 1
                site["last_t"] = t_now = s.last_seen_t
                return {"is_new": False, "site": site}

        site = {"site_id": len(self._sites) + 1,
                "north_m": n, "east_m": e, "anchor_n": n, "anchor_e": e,
                "drift_rejected": 0, "sigma_m": sigma,
                "tracks": {int(s.track_id)}, "n_reports": 1,
                "first_t": s.first_seen_t, "last_t": s.last_seen_t}
        self._sites.append(site)
        if len(self._sites) > 1:
            log.info("survivor site %d at n=%.0f e=%.0f (+/-%.1f m); %d tracks "
                     "merged into %d distinct locations so far",
                     len(self._sites), n, e, sigma,
                     sum(len(x["tracks"]) for x in self._sites), len(self._sites))
        return {"is_new": True, "site": site}

    @staticmethod
    def _hazard_key(h: Any) -> str:
        cls = getattr(h, "hazard_class", None) or \
            (h.get("hazard_class") if isinstance(h, dict) else "hazard")
        n = getattr(h, "north_m", None)
        e = getattr(h, "east_m", None)
        if isinstance(h, dict):
            n = h.get("north_m", n)
            e = h.get("east_m", e)
        if n is None or e is None:
            return f"hazard:{cls}"
        # Quantised to a 25 m cell so the same hazard seen across several frames
        # does not become several alerts.
        return f"hazard:{cls}:{int(round(float(n) / 25))}:{int(round(float(e) / 25))}"

    # ------------------------------------------------------------------ #
    # Guidance
    # ------------------------------------------------------------------ #
    def _current_lane(self) -> Optional[Lane]:
        if self.plan is None or self._lane_idx >= len(self.plan.lanes):
            return None
        return self.plan.lanes[self._lane_idx]

    def _set_lane_heading(self, lane: Lane) -> None:
        """Yaw onto the lane heading once, at the turn, then hold it.

        A survey aircraft translates along its lane; it does not continuously
        re-point at the direction of travel.  Commanding the heading here rather
        than letting the velocity stream imply it is what makes a serpentine plan
        flyable: the 180 deg reversal happens once, stationary, at the end of a
        lane, instead of at full speed in the middle of one.
        """
        if self._lane_heading_set == lane.index:
            return
        self._lane_heading_set = lane.index
        self.conn.set_velocity_ned()                    # brake before the turn
        self.conn.set_yaw(lane.heading_deg)
        log.info("lane %d: heading %.0f deg", lane.index, lane.heading_deg)

    def _advance_lane(self, state: PlantState) -> bool:
        """Advance the lane state machine: transit to its start, then survey it.

        The transit leg is explicit rather than folded into the survey command,
        and that separation is not cosmetic.  Lanes are ordered by belief, so the
        next one to fly can start anywhere in the area - the highest-value ground
        is not usually adjacent to the last lane flown.  Merging the transit into
        the survey command produces a diagonal ground track that is neither: it
        crosses un-imaged ground at an angle the lane spacing was not computed
        for, and progress along the lane is measured by projection onto a line the
        aircraft is not on, so a lane can register as complete while the aircraft
        is 200 m off it.

        Flying the transit as its own leg also puts the yaw slew where it belongs:
        stationary at the start of the survey leg, not at speed in the middle of
        one.
        """
        lane = self._current_lane()
        if lane is None:
            return False
        p = state.pos[:2]

        if self._lane_phase == "transit":
            gap = float(np.linalg.norm(p - np.array(lane.start)))
            if gap <= self.arrival_tol:
                self._lane_phase = "survey"
                self._lane_s = 0.0
                self._set_lane_heading(lane)
                log.info("lane %d: on station, surveying %.0f deg for %.0f m",
                         lane.index, lane.heading_deg, lane.length_m)
            return True

        d = np.array([lane.end[0] - lane.start[0],
                      lane.end[1] - lane.start[1]], dtype=np.float64)
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            self._next_lane()
            return self._current_lane() is not None
        u = d / L
        self._lane_s = float(np.dot(p - np.array(lane.start), u))
        lateral = float(abs(u[0] * (p[1] - lane.start[1])
                            - u[1] * (p[0] - lane.start[0])))
        if self._lane_s >= L - self.arrival_tol:
            done = self._lane_idx
            self._next_lane()
            nxt = self._current_lane()
            log.info("lane %d/%d complete (lateral err %.1f m)%s", done,
                     len(self.plan.lanes), lateral,
                     f"; next heading {nxt.heading_deg:.0f} deg, "
                     f"weight {nxt.weight:.2f}" if nxt else "; plan complete")
            self.report.timeline.append({
                "t": round(self._elapsed(), 1), "event": "lane_complete",
                "lane": done, "lateral_err_m": round(lateral, 2),
                "transit_m": round(float(np.linalg.norm(
                    p - np.array(lane.start))), 1),
            })
            return nxt is not None
        return True

    def _next_lane(self) -> None:
        self._lane_idx += 1
        self._lane_s = 0.0
        self._lane_phase = "transit"

    def _command_transit(self, state: PlantState, lane: Lane) -> None:
        """Fly to the start of the next lane at transit speed.

        Faster than the survey speed, because nothing is being imaged: the
        perception rate was chosen for the survey leg's ground sample interval and
        does not apply here.  Altitude is held at the survey altitude so the
        transit does not cost a climb the sortie has to pay for twice.
        """
        err = np.array(lane.start) - state.pos[:2]
        dist = float(np.linalg.norm(err))
        if dist < 0.5:
            self.conn.set_velocity_ned(hold_alt_rel_m=lane.altitude_agl_m)
            return
        v = err / dist * min(self.transit_speed_ms, max(dist * 0.5, 1.0))
        self.conn.set_velocity_ned(vn=float(v[0]), ve=float(v[1]),
                                   hold_alt_rel_m=lane.altitude_agl_m)

    def _command_lane(self, state: PlantState, lane: Lane) -> None:
        """Velocity command along the lane with lateral correction.

        Velocity guidance rather than waypoint guidance, because a lane is a
        continuous sweep: stepping between discrete waypoints produces a
        stop-start ground track whose along-track sampling is uneven, and the
        perception rate was chosen for a constant speed.
        """
        target = np.array(lane.point_at(self._lane_s + 8.0), dtype=np.float64)
        err = target - state.pos[:2]
        dist = float(np.linalg.norm(err))
        if dist < 0.5:
            self.conn.set_velocity_ned(vn=0.0, ve=0.0,
                                       hold_alt_rel_m=lane.altitude_agl_m)
            return
        d = np.array([lane.end[0] - lane.start[0],
                      lane.end[1] - lane.start[1]], dtype=np.float64)
        L = max(float(np.linalg.norm(d)), 1e-6)
        along = d / L
        lateral = np.array([-along[1], along[0]])
        # Split the command between making along-lane progress and killing
        # cross-track error.  Commanding straight at the target point instead
        # cuts the corner of every lane and leaves an unimaged wedge at each turn.
        v_along = lane.speed_ms
        v_lat = float(np.clip(np.dot(err, lateral) * 0.35,
                              -lane.speed_ms * 0.6, lane.speed_ms * 0.6))
        cmd = along * v_along + lateral * v_lat
        speed = float(np.linalg.norm(cmd))
        if speed > lane.speed_ms * 1.35:
            cmd *= (lane.speed_ms * 1.35) / speed
        self.conn.set_velocity_ned(vn=float(cmd[0]), ve=float(cmd[1]),
                                   hold_alt_rel_m=lane.altitude_agl_m)

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _state(self, t_mission: float, tel: Any) -> PlantState:
        if self.state_fn is not None:
            return self.state_fn(t_mission)
        return plant_from_telemetry(tel, self.origin)

    def _track_distance(self, state: PlantState) -> None:
        """Integrate ground track from the pose.

        Done here rather than taken from the vehicle, because the aircraft reports
        distance only if something on it is integrating, and a runner that assumes
        that will silently report zero on any flight stack that does not.
        """
        p = state.pos[:2]
        if self._last_pos is not None:
            d = float(np.linalg.norm(p - self._last_pos))
            if d < 5.0:                    # reject a teleport as a real leg
                self._distance_m += d
        self._last_pos = p.copy()

    def _update_link(self, state: PlantState) -> None:
        """Range to the GCS decides whether anything can be transmitted.

        Computed from the flight path rather than scripted, so a survey lane that
        happens to run away from the command post produces a real outage and the
        store-and-forward queue is exercised by the geometry of the sortie.
        """
        d = float(np.hypot(state.pos[0] - self.gcs[0], state.pos[1] - self.gcs[1]))
        self.uplink.set_range(d)

    def run(self, max_duration_s: float = 900.0,
            takeoff_altitude_agl_m: Optional[float] = None,
            return_home: bool = True) -> MissionReport:
        """Fly the sortie.  Returns the report.

        Phases are explicit rather than emergent because each has a different
        failure mode and a different thing to log: climb is about the prearm and
        the position estimate being good enough to take off, survey is about
        coverage and detection, return is about endurance.  Collapsing them into
        one loop makes a sortie that failed to arm look like a sortie that found
        nobody.
        """
        rep = self.report
        rep.started_at = time.time()
        self._t0 = rep.started_at
        self.uplink.start()

        if self.plan is None:
            # Budget the survey to roughly half the mission clock: the rest is
            # climb, return and the margin that decides whether the aircraft gets
            # home.  A runner that spends the whole budget surveying returns home
            # on fumes or not at all.
            self.make_plan(max_duration_s=max(60.0, max_duration_s * 0.5))
        alt = takeoff_altitude_agl_m or self.plan.survey_altitude_agl_m

        try:
            self._phase = "connect"
            tel = self._wait_for_telemetry()
            if tel is None:
                rep.faults.append("no telemetry: vehicle never came up")
                return self._finish()

            self._phase = "arm"
            if not self._arm_and_takeoff(alt, rep):
                return self._finish()

            self._phase = "survey"
            self._survey_loop(max_duration_s - self._elapsed(), rep)

            if return_home:
                self._phase = "rtl"
                self._return_home(rep)
        except KeyboardInterrupt:                        # pragma: no cover
            rep.faults.append("interrupted")
            log.warning("mission interrupted at %.1f s", self._elapsed())
        except Exception as exc:                         # pragma: no cover
            rep.faults.append(f"runner exception: {exc!r}")
            log.exception("mission failed")
        finally:
            self._phase = "finished"
            self._finish()
        return rep

    # ------------------------------------------------------------------ #
    def _wait_for_telemetry(self, timeout: float = 30.0) -> Any:
        self.conn.request_streams()
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.conn.pump(0.25)
            tel = self.conn.telemetry
            if tel.has_position:
                log.info("telemetry up: %s", tel.summary())
                return tel
        log.error("no usable telemetry after %.0f s", timeout)
        return None

    def _arm_and_takeoff(self, alt: float, rep: MissionReport) -> bool:
        self.conn.set_mode("GUIDED")
        ok = self.conn.arm(timeout=25.0)
        if not ok:
            rep.faults.append(f"arm refused: {self.conn.recent_statustext(20.0)}")
            log.error("ARM REFUSED: %s", self.conn.recent_statustext(20.0))
            return False
        rep.armed = True
        log.info("armed; taking off to %.0f m AGL", alt)
        self.conn.takeoff(alt)
        t0 = time.time()
        while time.time() - t0 < 90.0:
            self.conn.pump(0.2)
            if (self.conn.telemetry.alt_rel_m or 0.0) >= alt - 2.0:
                break
        climb = self.conn.telemetry.alt_rel_m or 0.0
        if climb < alt * 0.6:
            rep.faults.append(f"takeoff incomplete: reached {climb:.1f} m of "
                              f"{alt:.0f} m")
            log.error("takeoff incomplete at %.1f m", climb)
            return False
        log.info("at %.1f m AGL after %.1f s", climb, time.time() - t0)
        rep.timeline.append({"t": round(time.time() - t0, 1),
                             "event": "survey_altitude_reached",
                             "alt_agl_m": round(climb, 1)})
        return True

    def _survey_loop(self, budget_s: float, rep: MissionReport) -> None:
        """Fly every lane, perceiving and reporting as it goes."""
        t_end = time.time() + max(budget_s, 5.0)
        ctrl_dt = 1.0 / max(self.control_hz, 1.0)
        perc_dt = 1.0 / max(self.perception_hz, 0.1)
        tel = self.conn.telemetry

        while time.time() < t_end and not self._stop:
            tick = time.time()
            self.conn.pump(ctrl_dt)
            tel = self.conn.telemetry
            t_mission = self._elapsed()

            if not _is_armed(tel):
                rep.faults.append(f"disarmed in flight at {t_mission:.1f} s")
                log.error("vehicle disarmed during survey at %.1f s", t_mission)
                break

            state = self._state(t_mission, tel)
            self._update_link(state)

            lane = self._current_lane()
            if lane is None:
                log.info("all %d lanes complete at %.1f s", len(self.plan.lanes),
                         t_mission)
                rep.timeline.append({"t": round(t_mission, 1),
                                     "event": "survey_complete",
                                     "lanes": len(self.plan.lanes)})
                break
            self._advance_lane(state)
            lane = self._current_lane()
            if lane is not None:
                if self._lane_phase == "transit":
                    self._command_transit(state, lane)
                else:
                    self._command_lane(state, lane)

            # --- perception at its own, slower rate ----------------------
            # Only on the survey leg.  A transit frame images ground that is
            # between lanes at an angle the spacing was not computed for, so
            # counting it as coverage would inflate the number without adding
            # detection capability - and running the pipeline on it costs the
            # 280 ms the control loop needs for the turn.
            if (self._lane_phase == "survey"
                    and t_mission - self._last_perception >= perc_dt):
                self._last_perception = t_mission
                nav = self.nav_quality(tel)
                try:
                    product = self.perceive(t_mission, state, nav)
                    self.publish(product, t_mission, nav)
                except Exception as exc:                 # pragma: no cover
                    rep.faults.append(f"perception failed at {t_mission:.1f}s: "
                                      f"{exc!r}")
                    log.exception("perception cycle failed")

            if t_mission - self._last_telemetry_pub >= 1.0:
                self._last_telemetry_pub = t_mission
                self.uplink.publish_vehicle({
                    "mode": self.conn.mode, "armed": _is_armed(tel),
                    "alt_rel_m": round(tel.alt_rel_m or 0.0, 1),
                    "gs_ms": round(tel.speed_ms, 2),
                    "n_m": round(float(state.pos[0]), 1),
                    "e_m": round(float(state.pos[1]), 1),
                    "batt_pct": round(float(
                        getattr(tel, "battery_remaining_pct", None) or 100.0), 1),
                    "lane": self._lane_idx,
                    "queue_bytes": self.uplink.queue.bytes_queued,
                })
            self.uplink.pump_once()

            self._track_distance(state)

        info = self._vehicle_info()
        rep.flight_time_s = self._elapsed()
        rep.distance_m = info.get("distance_m") or self._distance_m
        rep.energy_wh = info.get("energy_wh", 0.0)
        rep.battery_pct = info.get("battery_pct", 100.0)

    def _vehicle_info(self) -> Dict[str, Any]:
        """Energy and distance, from the simulator when there is one.

        Injected rather than sniffed off the connection.  The hardware path has no
        truth-derived energy figure at all - it has a battery percentage from the
        power monitor and an odometer reading, and nothing that knows how far the
        aircraft actually flew.  Reaching into the simulator for those numbers
        from inside the runner would put a simulated value in a real flight
        report with no indication of where it came from.
        """
        tel = self.conn.telemetry
        fallback = {"distance_m": 0.0, "energy_wh": 0.0,
                    "battery_pct": float(
                        getattr(tel, "battery_remaining_pct", None) or 100.0)}
        if self.vehicle_info_fn is None:
            return fallback
        try:
            got = dict(self.vehicle_info_fn() or {})
        except Exception as exc:                         # pragma: no cover
            log.debug("vehicle_info_fn failed: %r", exc)
            return fallback
        fallback.update({k: v for k, v in got.items() if k in fallback})
        return fallback

    def _return_home(self, rep: MissionReport) -> None:
        log.info("returning to launch")
        self.conn.set_velocity_ned()
        self.conn.rtl()
        t0 = time.time()
        while time.time() - t0 < 300.0:
            self.conn.pump(0.3)
            self.uplink.pump_once()
            if not self.conn.armed:
                break
        rep.landed = not self.conn.armed
        if rep.landed:
            log.info("landed and disarmed after %.0f s of RTL", time.time() - t0)
        else:
            rep.faults.append(f"RTL did not complete in {time.time()-t0:.0f} s")
            log.error("RTL timed out")

    def _finish(self) -> MissionReport:
        """Close out the report: drain the queue, score against truth, write out."""
        rep = self.report
        rep.finished_at = time.time()
        rep.flight_time_s = max(rep.flight_time_s, self._elapsed())

        # Drain whatever is left.  A sortie that lands with survivors still in
        # the queue has not reported them, and the report has to say so rather
        # than quietly counting queued messages as delivered.
        if self.uplink.link.in_range:
            for _ in range(2000):
                if not self.uplink.pump_once():
                    break
        self.uplink.stop()

        info = self._vehicle_info()
        if info.get("distance_m"):
            rep.distance_m = info["distance_m"]
        elif self._distance_m:
            rep.distance_m = self._distance_m
        if info.get("energy_wh"):
            rep.energy_wh = info["energy_wh"]
        rep.battery_pct = info.get("battery_pct", rep.battery_pct)
        rep.coverage = self.coverage.summary(rep.flight_time_s)
        box = rep.plan.get("search_box")
        if box:
            rep.coverage["of_search_box"] = self.coverage_in_box(box)
            rep.coverage["search_box"] = box
        rep.belief = self.belief.summary()
        rep.comms = self.uplink.summary()
        rep.comms["ground"] = self.uplink.report.to_dict()
        rep.perception = {
            "frames_rendered": self._frames_rendered,
            "cycles": len(self._perception_ms),
            "mean_cycle_ms": (round(float(np.mean(self._perception_ms)), 1)
                              if self._perception_ms else None),
            "p95_cycle_ms": (round(float(np.percentile(self._perception_ms, 95)), 1)
                             if len(self._perception_ms) > 4 else None),
            "summary": self.pipeline.summary(),
        }

        ground = self.uplink.report.to_dict()
        rep.survivors_reported = ground["survivors"]
        rep.hazards_reported = ground["hazards"]
        rep.perception["distinct_survivor_sites"] = len(self._sites)
        rep.perception["tracks_merged_into_sites"] = sum(
            len(x["tracks"]) for x in self._sites)
        rep.perception["rejected_out_of_area"] = self._rejected_out_of_area
        rep.perception["sites"] = [
            {"site_id": x["site_id"],
             "north_m": round(x["north_m"], 1), "east_m": round(x["east_m"], 1),
             "sigma_m": round(x["sigma_m"], 2), "tracks": sorted(x["tracks"]),
             "n_reports": x["n_reports"],
             "drift_rejected": x["drift_rejected"],
             "drift_from_anchor_m": round(math.hypot(
                 x["north_m"] - x["anchor_n"], x["east_m"] - x["anchor_e"]), 1),
             } for x in self._sites]
        rep.scoring = self._score(ground)
        if self.artifacts is not None:
            try:
                rep.write(self.artifacts / f"mission_{rep.scenario or 'run'}.json")
            except Exception as exc:                     # pragma: no cover
                log.warning("could not write report: %r", exc)
        log.info("mission complete: %d/%d survivors reported to the ground, "
                 "coverage %.1f%%, %.0f m, %.1f Wh",
                 rep.scoring.get("n_reported", 0), rep.scoring.get("n_truth", 0),
                 100 * rep.coverage.get("effective_coverage", 0.0),
                 rep.distance_m, rep.energy_wh)
        return rep

    # ------------------------------------------------------------------ #
    def coverage_in_box(self, box: Dict[str, Any]) -> Dict[str, float]:
        """Coverage restricted to the area the plan actually intended to search.

        Reported alongside the whole-world figure rather than instead of it.  The
        world number answers "what fraction of the disaster area has been
        searched", which is what an incident commander asks; the box number
        answers "did the sortie do what it set out to do", which is what a test
        needs.  Without the second, a 200 m box flown perfectly inside a 900 m
        basin reports 8% coverage and looks like a failure of the sensor rather
        than a choice about the search area.
        """
        g = self.coverage
        n0, n1 = box["north_m"]
        e0, e1 = box["east_m"]
        sub = g.p_detected[int(n0 / g.res):int(n1 / g.res) + 1,
                           int(e0 / g.res):int(e1 / g.res) + 1]
        if sub.size == 0:
            return {}
        return {
            "fraction_looked_at": round(float(np.mean(sub > 0)), 4),
            "fraction_covered_p50": round(float(np.mean(sub >= 0.5)), 4),
            "effective_coverage": round(float(np.mean(sub)), 4),
            "area_m2": round(float(sub.size) * g.res ** 2, 1),
        }

    def _score(self, ground: Dict[str, Any]) -> Dict[str, Any]:
        """Compare what reached the ground with what was actually there.

        Association is by distance with a generous gate, because the point is not
        to penalise the geo-tag for being a few metres off - it is to establish
        that a real person produced a report a responder could act on.  Position
        error is reported separately so the two questions stay distinguishable.
        """
        victims = getattr(self.world, "victims", [])
        truth = [{"vid": v.vid, "north_m": float(v.north), "east_m": float(v.east),
                  "alive": bool(v.alive), "category": v.category,
                  "in_water": bool(v.in_water), "on_rooftop": bool(v.on_rooftop),
                  "needs": v.needs, "group": v.group}
                 for v in victims]
        self.report.truth = {"n_victims": len(truth), "victims": truth}

        reported: List[Dict[str, Any]] = ground.get("survivors", [])
        matched: List[Dict[str, Any]] = []
        used: set = set()
        from sar.core.geo import wgs84_to_local
        for r in reported:
            # The wire form is compact - "la"/"lo"/"sg" - because it has to fit a
            # 222-byte LoRa packet.  Read both spellings: scoring against the
            # verbose names silently matched nothing at all and reported 0% recall
            # on a sortie that had in fact delivered eight survivors, which is
            # the worst kind of test failure because the number looks plausible.
            lat = r.get("la", r.get("lat"))
            lon = r.get("lo", r.get("lon"))
            rn = r.get("north_m")
            re_ = r.get("east_m")
            if (rn is None or re_ is None) and lat is not None and lon is not None:
                rn, re_, _ = wgs84_to_local(GeoPoint(lat, lon, 0.0), self.origin)
            if rn is None or re_ is None:
                continue
            sigma = r.get("sg", r.get("sigma_m"))
            best, best_d = None, float("inf")
            for i, tv in enumerate(truth):
                if i in used or not tv["alive"]:
                    continue
                d = float(math.hypot(rn - tv["north_m"], re_ - tv["east_m"]))
                if d < best_d:
                    best, best_d = i, d
            # A 60 m gate is generous by design: this measures whether the person
            # was found and reported at all.  Localisation accuracy is a separate
            # number, reported below, and gating tightly would fold a geotagging
            # error into a detection miss and hide which one happened.
            if best is not None and best_d <= 60.0:
                used.add(best)
                matched.append({"vid": truth[best]["vid"],
                                "distance_m": round(best_d, 1),
                                "reported_sigma_m": sigma,
                                "category": truth[best]["category"],
                                "priority": r.get("pr", r.get("priority"))})

        false_positives = len(reported) - len(matched)
        errs = [m["distance_m"] for m in matched]
        # An error ellipse that does not contain the truth is worse than a large
        # one: it is confidently wrong.  Reporting the containment rate alongside
        # the raw error is what distinguishes "uncertain but honest" from
        # "precise and misleading".
        contained = sum(1 for m in matched
                        if m["reported_sigma_m"] is not None
                        and m["distance_m"] <= 2.0 * float(m["reported_sigma_m"]))
        n_alive = sum(1 for t in truth if t["alive"])
        return {
            "n_truth": n_alive,
            "n_reported": len(reported),
            "n_matched": len(matched),
            "false_positives": false_positives,
            "recall": round(len(matched) / n_alive, 4) if n_alive else None,
            "precision": (round(len(matched) / len(reported), 4)
                          if reported else None),
            "mean_position_error_m": round(float(np.mean(errs)), 2) if errs else None,
            "max_position_error_m": round(float(np.max(errs)), 2) if errs else None,
            "sigma_contains_truth_pct": (round(100.0 * contained / len(matched), 1)
                                         if matched else None),
            "matched": matched,
            "missed": [t for i, t in enumerate(truth)
                       if t["alive"] and i not in used],
        }

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._stop = True
