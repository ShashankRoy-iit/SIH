"""Coverage planning and the survivor belief map.

Two things live here, and they are coupled on purpose.

:class:`BoustrophedonPlanner` produces the lawnmower lane plan: the standard
coverage path for a convex-ish search area, and the right default because it is
provably complete, needs no map, and degrades gracefully.  Lane spacing comes
from the camera footprint and a required side overlap, not from a fixed number,
which is what makes the same plan work at 40 m over a flood basin and 90 m over
an earthquake zone.

:class:`BeliefMap` is the reason the plan is allowed to be incomplete.  A
lawnmower treats every square metre as equally worth visiting, which is false in
a disaster: survivors cluster near roads, on rooftops, above the waterline, and
downstream of where people were last known to be.  The belief map holds a
probability per cell that an un-detected survivor is there, updated by what the
sensor actually saw, and the planner is asked for the next lane *weighted by it*.

The update is Bayesian rather than a heuristic score, and the distinction is not
cosmetic.  A cell the thermal camera looked at and found nothing has had its
probability multiplied by the probability of *missing* someone who is there -
which depends on ground sample distance, contrast and occlusion, all of which the
renderer reports.  Looking at a cell from 120 m through smoke at 4 cm/px barely
reduces belief; looking at it from 40 m in clear air at 8 mm/px reduces it a lot.
A coverage grid that only records "seen / not seen" reports 100% coverage over an
area it demonstrably could not have detected anyone in, and that number is the
one an incident commander would act on.

Detection probability per cell therefore comes from the same
:class:`~sar.sim.renderer.TruthObject` detectability model the detector is
evaluated against, so the belief map cannot become more optimistic than the
sensor physically is.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("sar.decision")

__all__ = [
    "Lane", "SurveyPlan", "BoustrophedonPlanner", "BeliefMap", "CoverageGrid",
]


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #
@dataclass
class Lane:
    """One straight survey pass, in local NED metres from the mission origin."""

    index: int
    start: Tuple[float, float]          # (north, east)
    end: Tuple[float, float]
    altitude_agl_m: float
    #: Expected ground speed along the lane.  Set from the perception rate and
    #: the along-track footprint: flying faster than the sensor can sample leaves
    #: gaps no amount of coverage bookkeeping will reveal.
    speed_ms: float = 4.0
    weight: float = 1.0                 # belief-weighted priority

    @property
    def length_m(self) -> float:
        return float(math.hypot(self.end[0] - self.start[0],
                                self.end[1] - self.start[1]))

    @property
    def heading_deg(self) -> float:
        dn = self.end[0] - self.start[0]
        de = self.end[1] - self.start[1]
        return (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0

    @property
    def duration_s(self) -> float:
        return self.length_m / max(self.speed_ms, 0.1)

    def point_at(self, s: float) -> Tuple[float, float]:
        """Position at arc length ``s`` along the lane."""
        L = self.length_m
        if L <= 0:
            return self.start
        u = min(max(s / L, 0.0), 1.0)
        return (self.start[0] + u * (self.end[0] - self.start[0]),
                self.start[1] + u * (self.end[1] - self.start[1]))

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "start": list(self.start),
                "end": list(self.end), "alt_agl_m": self.altitude_agl_m,
                "length_m": round(self.length_m, 1),
                "heading_deg": round(self.heading_deg, 1),
                "speed_ms": self.speed_ms, "duration_s": round(self.duration_s, 1),
                "weight": round(self.weight, 3)}


@dataclass
class SurveyPlan:
    """A complete sortie: lanes in flying order, plus what it should achieve."""

    lanes: List[Lane] = field(default_factory=list)
    survey_altitude_agl_m: float = 60.0
    lane_spacing_m: float = 20.0
    side_overlap: float = 0.30
    area_m2: float = 0.0
    origin: Tuple[float, float] = (0.0, 0.0)
    #: The box this plan intends to search, and how many survivors the world
    #: model says are inside it.  Carried on the plan rather than only in the
    #: mission report because the dashboard draws it and because coverage has to
    #: be reported against it: a 220 m box flown perfectly inside a 900 m basin
    #: is 5% of the world and 100% of the sortie.
    search_box: Optional[Dict[str, Any]] = None
    extent_north_m: float = 0.0
    extent_east_m: float = 0.0

    @property
    def total_length_m(self) -> float:
        return sum(ln.length_m for ln in self.lanes)

    @property
    def total_duration_s(self) -> float:
        """Lane time plus the transit between lanes, which is not free.

        Ignoring transit understates a sortie by 10-20% on a lane plan with many
        short lanes, and that error lands directly on the endurance estimate -
        which is the number that decides whether the aircraft comes home.
        """
        t = sum(ln.duration_s for ln in self.lanes)
        for a, b in zip(self.lanes, self.lanes[1:]):
            gap = math.hypot(b.start[0] - a.end[0], b.start[1] - a.end[1])
            t += gap / max(a.speed_ms * 1.5, 0.5)
        return t

    def waypoints(self) -> List[Tuple[float, float, float]]:
        """Flattened (north, east, -alt_agl) list for a waypoint-style upload."""
        out: List[Tuple[float, float, float]] = []
        for ln in self.lanes:
            out.append((ln.start[0], ln.start[1], -ln.altitude_agl_m))
            out.append((ln.end[0], ln.end[1], -ln.altitude_agl_m))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_lanes": len(self.lanes),
            "lane_spacing_m": round(self.lane_spacing_m, 2),
            "side_overlap": self.side_overlap,
            "survey_altitude_agl_m": self.survey_altitude_agl_m,
            "total_length_m": round(self.total_length_m, 1),
            "total_duration_s": round(self.total_duration_s, 1),
            "area_m2": round(self.area_m2, 1),
            "origin_north_m": round(self.origin[0], 1),
            "origin_east_m": round(self.origin[1], 1),
            "search_box": self.search_box,
            "lanes": [ln.to_dict() for ln in self.lanes],
        }


class BoustrophedonPlanner:
    """Lawnmower lanes with belief-weighted ordering.

    Lane spacing is derived, not configured.  Given the swath width at the survey
    altitude and a required side overlap, the spacing that guarantees the overlap
    is ``swath * (1 - overlap)``.  Flying that spacing means every point in the
    area is imaged at least once and the lane boundaries are imaged twice, which
    is what makes a detection near a lane edge recoverable from the adjacent pass.

    The along-track speed is derived the same way, from the perception rate: at
    ``f`` Hz and along-track footprint ``L``, flying at ``v`` samples the ground
    every ``v/f`` metres, so ``v <= f * L * (1 - forward_overlap)``.  Exceeding it
    produces a coverage grid that reports cells as seen between two frames that
    never actually looked at them.
    """

    def __init__(self, side_overlap: float = 0.30,
                 forward_overlap: float = 0.60,
                 max_speed_ms: float = 8.0,
                 min_altitude_agl_m: float = 25.0,
                 max_altitude_agl_m: float = 120.0) -> None:
        self.side_overlap = float(np.clip(side_overlap, 0.0, 0.9))
        self.forward_overlap = float(np.clip(forward_overlap, 0.0, 0.9))
        self.max_speed_ms = float(max_speed_ms)
        self.min_altitude_agl_m = float(min_altitude_agl_m)
        self.max_altitude_agl_m = float(max_altitude_agl_m)

    # ------------------------------------------------------------------ #
    def plan(self, extent_north_m: float, extent_east_m: float,
             swath_width_m: float, along_track_m: float,
             perception_hz: float = 2.0,
             altitude_agl_m: Optional[float] = None,
             belief: Optional["BeliefMap"] = None,
             lane_axis: str = "north",
             origin_north_m: float = 0.0,
             origin_east_m: float = 0.0) -> SurveyPlan:
        """Build a lane plan for a rectangular area.

        ``belief``, when given, reorders and reweights lanes by the expected
        survivor density along them.  The lanes themselves do not move - a
        lawnmower whose spacing varies is not complete - but the order in which
        they are flown does, so a sortie cut short by endurance has covered the
        most promising ground first.
        """
        swath_width_m = max(float(swath_width_m), 1.0)
        along_track_m = max(float(along_track_m), 1.0)
        spacing = swath_width_m * (1.0 - self.side_overlap)

        # Along-track speed from the perception rate, not a wish.
        v_sample = perception_hz * along_track_m * (1.0 - self.forward_overlap)
        speed = float(np.clip(v_sample, 0.5, self.max_speed_ms))

        alt = altitude_agl_m
        if alt is None:
            alt = swath_width_m / max(1.0, math.tan(math.radians(25.0)))
        alt = float(np.clip(alt, self.min_altitude_agl_m,
                            self.max_altitude_agl_m))

        along = extent_north_m if lane_axis == "north" else extent_east_m
        across = extent_east_m if lane_axis == "north" else extent_north_m
        n_lanes = max(1, int(math.ceil(across / spacing)))

        lanes: List[Lane] = []
        for i in range(n_lanes):
            off = min(i * spacing, across)
            if lane_axis == "north":
                a, b = (0.0, off), (along, off)
            else:
                a, b = (off, 0.0), (off, along)
            if i % 2 == 1:                       # serpentine, so no long transit
                a, b = b, a
            # Offset into the search area.  The area is a sub-region of the
            # scenario, not necessarily the corner it starts at: an operation
            # searches where the prior says people are, and anchoring every plan
            # at (0, 0) searches whatever happens to be at the map origin.
            a = (a[0] + origin_north_m, a[1] + origin_east_m)
            b = (b[0] + origin_north_m, b[1] + origin_east_m)
            lanes.append(Lane(index=i, start=a, end=b, altitude_agl_m=alt,
                              speed_ms=speed))

        if belief is not None:
            self._weight_by_belief(lanes, belief, lane_axis)

        return SurveyPlan(
            lanes=lanes, survey_altitude_agl_m=alt, lane_spacing_m=spacing,
            side_overlap=self.side_overlap,
            area_m2=float(extent_north_m * extent_east_m),
            origin=(float(origin_north_m), float(origin_east_m)),
            extent_north_m=float(extent_north_m),
            extent_east_m=float(extent_east_m))

    # ------------------------------------------------------------------ #
    def _weight_by_belief(self, lanes: List[Lane], belief: "BeliefMap",
                          lane_axis: str) -> None:
        """Score each lane by the belief mass it would sweep, then reorder.

        Reordering is a real trade and not a free optimisation: flying the
        high-belief lanes first means the *transit* legs cross low-belief ground,
        which is slightly longer overall.  It is worth it because the binding
        constraint on a SAR sortie is endurance, and a sortie that covers the
        most promising 60% before it must return home finds more people than one
        that covers 100% of the least promising ground in a fixed order.
        """
        for ln in lanes:
            pts = [ln.point_at(s) for s in np.linspace(0.0, ln.length_m, 24)]
            if lane_axis == "north":
                mass = belief.mass_at([(p[0], p[1]) for p in pts])
            else:
                mass = belief.mass_at([(p[0], p[1]) for p in pts])
            ln.weight = float(mass)

        # Stable sort on descending weight, keeping the serpentine within ties so
        # that lanes of equal belief are still flown in a low-transit order.
        order = sorted(range(len(lanes)),
                       key=lambda i: (-lanes[i].weight, abs(i - len(lanes) / 2)))
        lanes.sort(key=lambda ln: order.index(ln.index))
        for i, ln in enumerate(lanes):
            ln.index = i

    # ------------------------------------------------------------------ #
    def refine(self, plan: SurveyPlan, north: float, east: float,
               radius_m: float = 60.0,
               altitude_agl_m: Optional[float] = None) -> SurveyPlan:
        """A tighter, lower plan around a point of interest.

        Used by the confirm pass: a survivor detected at survey altitude gets a
        second look from lower and slower, which both improves the ground sample
        distance and gives the tracker enough frames to resolve posture and
        movement.  Returning a new plan rather than mutating the existing one
        keeps the original survey resumable afterwards.
        """
        alt = altitude_agl_m or max(self.min_altitude_agl_m,
                                    plan.survey_altitude_agl_m * 0.5)
        lanes: List[Lane] = []
        n = 4
        for i in range(n):
            e = east - radius_m + (2 * radius_m) * i / max(1, n - 1)
            a = (north - radius_m, e)
            b = (north + radius_m, e)
            if i % 2 == 1:
                a, b = b, a
            lanes.append(Lane(index=i, start=a, end=b, altitude_agl_m=alt,
                              speed_ms=max(1.0, plan.lanes[0].speed_ms * 0.6)
                              if plan.lanes else 2.0, weight=1.0))
        return SurveyPlan(lanes=lanes, survey_altitude_agl_m=alt,
                          lane_spacing_m=2 * radius_m / max(1, n - 1),
                          side_overlap=plan.side_overlap,
                          area_m2=(2 * radius_m) ** 2,
                          origin=(north - radius_m, east - radius_m),
                          extent_north_m=2 * radius_m,
                          extent_east_m=2 * radius_m)


# --------------------------------------------------------------------------- #
# Coverage bookkeeping
# --------------------------------------------------------------------------- #
class CoverageGrid:
    """What the sensor has actually looked at, and how well.

    Separate from :class:`BeliefMap` because they answer different questions and
    get reported to different people.  Coverage is the operator's question -
    "have we searched the area" - and is a statement about the aircraft's sensors.
    Belief is the analyst's question - "where is someone still likely to be" -
    and is a statement about the world.

    The grid stores *detection capability*, not a boolean.  A cell swept at
    0.15 m GSD through clear air and a cell swept at 0.4 m GSD through smoke both
    count as "covered" in a boolean grid, and the difference between them is
    whether a person lying in that cell would have been found.
    """

    def __init__(self, extent_north_m: float, extent_east_m: float,
                 resolution_m: float = 4.0) -> None:
        self.res = float(resolution_m)
        self.north_m = float(extent_north_m)
        self.east_m = float(extent_east_m)
        self.nr = max(1, int(math.ceil(self.north_m / self.res)))
        self.er = max(1, int(math.ceil(self.east_m / self.res)))
        #: Cumulative probability that a present survivor would have been
        #: detected, over all passes so far.  Composed as 1 - prod(1 - p_i),
        #: which is the correct way to combine independent looks and is what
        #: makes a second pass at better GSD worth more than a first pass at
        #: worse GSD rather than merely repeating it.
        self.p_detected = np.zeros((self.nr, self.er), dtype=np.float32)
        self.n_looks = np.zeros((self.nr, self.er), dtype=np.uint16)
        self.best_gsd_m = np.full((self.nr, self.er), np.inf, dtype=np.float32)
        self.last_look_s = np.full((self.nr, self.er), -1.0, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _idx(self, north, east):
        n = np.clip((np.asarray(north, dtype=np.float64) / self.res).astype(int),
                    0, self.nr - 1)
        e = np.clip((np.asarray(east, dtype=np.float64) / self.res).astype(int),
                    0, self.er - 1)
        return n, e

    def mark(self, north: Sequence[float], east: Sequence[float],
             p_detect: Sequence[float], gsd_m: Sequence[float],
             t_s: float) -> int:
        """Record a look at a set of ground points.  Returns cells updated."""
        if len(north) == 0:
            return 0
        n, e = self._idx(north, east)
        p = np.clip(np.asarray(p_detect, dtype=np.float32), 0.0, 1.0)
        g = np.asarray(gsd_m, dtype=np.float32)

        prev = self.p_detected[n, e]
        self.p_detected[n, e] = 1.0 - (1.0 - prev) * (1.0 - p)
        self.n_looks[n, e] += 1
        self.best_gsd_m[n, e] = np.minimum(self.best_gsd_m[n, e], g)
        self.last_look_s[n, e] = t_s
        return int(len(n))

    # ------------------------------------------------------------------ #
    def fraction_covered(self, threshold: float = 0.5) -> float:
        """Cells whose cumulative detection probability exceeds ``threshold``.

        Reported at a threshold rather than as "looked at least once" because
        those two numbers diverge exactly when it matters: in smoke, at night, or
        at high altitude, where a pass happens and detects nothing.
        """
        return float(np.mean(self.p_detected >= threshold))

    def mean_detection_probability(self) -> float:
        return float(np.mean(self.p_detected))

    def effective_coverage(self, threshold: float = 0.5) -> float:
        """Area-weighted coverage: sum of per-cell detection probability.

        This is the number to quote.  ``fraction_covered`` is a step function of
        a threshold and can jump from 0.4 to 0.9 on a single good pass; the
        expected detected fraction is continuous and is what a search's
        probability of success actually is.
        """
        return float(np.mean(self.p_detected))

    def uncovered_mass(self) -> float:
        return float(np.sum(1.0 - self.p_detected)) * self.res * self.res

    def summary(self, t_s: float = 0.0) -> Dict[str, Any]:
        looks = self.n_looks.astype(np.float64)
        return {
            "cells": int(self.nr * self.er),
            "resolution_m": self.res,
            "area_m2": round(self.north_m * self.east_m, 1),
            "fraction_looked_at": round(float(np.mean(looks > 0)), 4),
            "fraction_covered_p50": round(self.fraction_covered(0.5), 4),
            "fraction_covered_p90": round(self.fraction_covered(0.9), 4),
            "effective_coverage": round(self.effective_coverage(), 4),
            "mean_p_detected": round(self.mean_detection_probability(), 4),
            "mean_best_gsd_m": (round(float(np.mean(
                self.best_gsd_m[np.isfinite(self.best_gsd_m)])), 4)
                if np.any(np.isfinite(self.best_gsd_m)) else None),
            "uncovered_area_m2": round(float(np.sum(
                (self.p_detected < 0.5).astype(np.float64)) * self.res ** 2), 1),
        }

    def as_image(self) -> np.ndarray:
        """Coverage as a 0-255 array, for the dashboard and for artifacts."""
        return (np.clip(self.p_detected, 0.0, 1.0) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# The belief map
# --------------------------------------------------------------------------- #
@dataclass
class PriorSource:
    """One contributor to the initial survivor prior, with a weight.

    Kept as data rather than folded into :class:`BeliefMap` because the priors
    that matter change with the disaster: flood survivors are above the
    waterline and on rooftops, earthquake survivors are near collapsed
    structures and along roads, wildfire survivors are downstream of the front.
    A single hardcoded prior would silently favour whichever scenario it was
    written against.
    """

    name: str
    field: np.ndarray                  # same shape as the belief grid, 0..1
    weight: float = 1.0
    description: str = ""


class BeliefMap:
    """Bayesian P(undetected survivor in cell) over the search area.

    The state is the probability that a survivor is present *and has not been
    detected yet*, which is the quantity a search acts on.  Two things update it:

    **A look that found nothing** multiplies by ``(1 - p_detect)``, where
    ``p_detect`` is the sensor's actual probability of detecting someone in that
    cell on that pass.  This is where the coupling to the renderer matters: a
    look from high altitude in smoke has a low ``p_detect`` and barely reduces
    belief, so the map keeps sending the aircraft back.  A boolean coverage grid
    would have marked the cell done.

    **A detection** sets the cell's undetected-survivor belief toward zero, but
    not to exactly zero - the group-size estimate is uncertain and there may be
    a second person in the same cell who was occluded on this frame.

    Prior mass is normalised to an *expected number of survivors*, not to 1.  A
    probability distribution over cells says "the one survivor is somewhere";
    an expected count says "we believe there are about nine people in this basin
    and here is where they are", which is what makes the map composable with
    detections - finding three people should reduce the remaining expectation by
    three, not reset it.
    """

    def __init__(self, extent_north_m: float, extent_east_m: float,
                 resolution_m: float = 6.0,
                 expected_survivors: float = 8.0,
                 floor_per_cell: Optional[float] = None) -> None:
        self.res = float(resolution_m)
        self.north_m = float(extent_north_m)
        self.east_m = float(extent_east_m)
        self.nr = max(1, int(math.ceil(self.north_m / self.res)))
        self.er = max(1, int(math.ceil(self.east_m / self.res)))
        #: Expected number of *undetected* survivors per cell.
        self.belief = np.zeros((self.nr, self.er), dtype=np.float64)
        self.priors: List[PriorSource] = []
        self.expected_survivors = float(expected_survivors)
        # The floor keeps a cell from becoming permanently unsearchable, so it
        # must be small relative to the uniform prior - not an absolute constant.
        # At 0.002 over a 6 m grid on a 900 m basin the floor alone sums to 45
        # expected survivors against a prior of 12, so the map starts by
        # believing there are four times more people than the scenario contains
        # and ``progress`` can never move.  Scaled to 2% of the uniform prior it
        # does its job without outweighing the thing it is flooring.
        uniform = self.expected_survivors / max(1, self.nr * self.er)
        self.floor_per_cell = (float(floor_per_cell) if floor_per_cell is not None
                               else 0.02 * uniform)
        self._initial_total = float(expected_survivors)
        self.detections_absorbed = 0
        self.n_updates = 0
        self._flat_prior()

    # ------------------------------------------------------------------ #
    def _flat_prior(self) -> None:
        n = self.nr * self.er
        self.belief[:] = self.expected_survivors / max(1, n)

    def add_prior(self, source: PriorSource) -> None:
        """Fold a spatial prior in and renormalise to the expected count.

        Multiplying rather than replacing keeps a flat floor everywhere: a prior
        that is exactly zero over a cell would make that cell permanently
        unsearchable, and priors built from a damage map are wrong somewhere.
        """
        if source.field.shape != self.belief.shape:
            raise ValueError(f"prior {source.name!r} has shape "
                             f"{source.field.shape}, belief is "
                             f"{self.belief.shape}")
        self.priors.append(source)
        shape = np.clip(source.field, 0.0, 1.0) ** max(source.weight, 1e-6)
        self.belief *= (0.25 + 0.75 * shape)
        total = float(self.belief.sum())
        if total > 0:
            self.belief *= self.expected_survivors / total
        self.belief = np.maximum(self.belief, self.floor_per_cell)

    # ------------------------------------------------------------------ #
    def _idx(self, north, east):
        n = np.clip((np.asarray(north, dtype=np.float64) / self.res).astype(int),
                    0, self.nr - 1)
        e = np.clip((np.asarray(east, dtype=np.float64) / self.res).astype(int),
                    0, self.er - 1)
        return n, e

    def observe(self, north: Sequence[float], east: Sequence[float],
                p_detect: Sequence[float],
                detected: Optional[Sequence[bool]] = None) -> float:
        """Update belief from one sensor pass.  Returns belief mass removed."""
        if len(north) == 0:
            return 0.0
        n, e = self._idx(north, east)
        p = np.clip(np.asarray(p_detect, dtype=np.float64), 0.0, 1.0)
        before = float(self.belief[n, e].sum())

        det = (np.zeros(len(n), dtype=bool) if detected is None
               else np.asarray(detected, dtype=bool))

        # Negative observation: multiply by the probability of having missed.
        miss = np.where(det, 1.0, 1.0 - p)
        self.belief[n, e] *= miss

        # Positive observation: this cell's survivor is now accounted for.  Not
        # set to zero, because group size is an estimate and a second person in
        # the same cell may have been occluded on this frame.
        if det.any():
            self.belief[n[det], e[det]] *= 0.15
            self.detections_absorbed += int(det.sum())

        self.belief = np.maximum(self.belief, self.floor_per_cell)
        self.n_updates += 1
        return max(0.0, before - float(self.belief[n, e].sum()))

    # ------------------------------------------------------------------ #
    def expected_remaining(self) -> float:
        """How many survivors the map still thinks are out there, unfound."""
        return float(self.belief.sum())

    def progress(self) -> float:
        """Fraction of the prior expectation accounted for, 0..1."""
        if self._initial_total <= 0:
            return 0.0
        return float(np.clip(1.0 - self.expected_remaining() / self._initial_total,
                             0.0, 1.0))

    def mass_at(self, points: Iterable[Tuple[float, float]]) -> float:
        """Belief mass along a path - how the planner scores a lane."""
        pts = list(points)
        if not pts:
            return 0.0
        n, e = self._idx([p[0] for p in pts], [p[1] for p in pts])
        return float(self.belief[n, e].sum())

    def peak(self) -> Optional[Tuple[float, float, float]]:
        """Highest-belief cell as ``(north, east, belief)``."""
        if self.belief.size == 0:
            return None
        i = int(np.argmax(self.belief))
        r, c = divmod(i, self.er)
        return ((r + 0.5) * self.res, (c + 0.5) * self.res,
                float(self.belief[r, c]))

    def top_cells(self, k: int = 8) -> List[Dict[str, Any]]:
        """The ``k`` most promising unsearched cells, for a re-tasking pass."""
        flat = self.belief.ravel()
        k = int(min(k, flat.size))
        idx = np.argpartition(-flat, k - 1)[:k] if k > 0 else np.array([], int)
        idx = idx[np.argsort(-flat[idx])]
        out = []
        for i in idx:
            r, c = divmod(int(i), self.er)
            out.append({"north_m": round((r + 0.5) * self.res, 1),
                        "east_m": round((c + 0.5) * self.res, 1),
                        "belief": round(float(flat[i]), 5)})
        return out

    def summary(self) -> Dict[str, Any]:
        peak = self.peak()
        return {
            "cells": int(self.nr * self.er),
            "resolution_m": self.res,
            "expected_remaining": round(self.expected_remaining(), 3),
            "initial_expected": round(self._initial_total, 3),
            "progress": round(self.progress(), 4),
            "detections_absorbed": self.detections_absorbed,
            "updates": self.n_updates,
            "peak": ({"north_m": round(peak[0], 1), "east_m": round(peak[1], 1),
                      "belief": round(peak[2], 5)} if peak else None),
            "n_priors": len(self.priors),
            "priors": [p.name for p in self.priors],
        }

    def as_image(self) -> np.ndarray:
        """Belief as 0-255, normalised to its own maximum."""
        m = float(self.belief.max())
        if m <= 0:
            return np.zeros_like(self.belief, dtype=np.uint8)
        return (np.clip(self.belief / m, 0.0, 1.0) * 255.0).astype(np.uint8)
