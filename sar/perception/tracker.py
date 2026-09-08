"""Geographic multi-target tracker with evidence-based confirmation.

Why track in *world* coordinates and not in the image
-----------------------------------------------------
A survey UAV at 12 m/s with an 8 Hz thermal core moves one full swath between
consecutive frames.  Image-space trackers (SORT, DeepSORT, ByteTrack) assume
frame-to-frame overlap and are therefore the wrong tool for a mapping sortie:
the same survivor is re-acquired minutes later from a different pass, a
different altitude and a different heading.  So tracks live in metres NED,
measurements arrive as geo-tags, and association is geometric rather than
appearance-based.  This also means a track survives GPS denial, camera restarts
and complete loss of the target between passes - it is a place on the ground
with an accumulating body of evidence, not a bounding box with an ID.

Confirmation is the part that makes the whole system usable
-----------------------------------------------------------
A single frame at 110 m AGL cannot distinguish a survivor from a sun-warmed
rock: ``scripts/experiment_subpixel_radiometry.py`` measures the mean
radiometric error growing to -7.5 K at that height, which pulls a 46 C rock
inside the human band and a chest-deep survivor outside it.  So no single-frame
score is allowed to raise an alert.  Instead:

* a detection opens a **candidate** track;
* each subsequent observation adds a label vote weighted by the *information
  content* of the frame it came from, ``1 / gsd^2`` - a 35 m look is worth
  nine 110 m looks, which is exactly right and is what makes a low confirmation
  pass able to overrule a high survey pass;
* a track becomes **confirmed** only after ``n_confirm`` observations spanning
  at least ``min_confirm_span_s`` seconds, with a weighted person vote above
  threshold.

That is the mechanism by which the two-pass search strategy actually pays off:
the broad pass generates candidates cheaply, the low pass decides them.

Motion model
------------
Constant velocity with a posture-dependent process noise.  A survivor who is
immobile has near-zero process noise, so their position estimate keeps
improving; a walking survivor, an animal or drifting debris needs a larger
allowance.  The tracker infers which from observed displacement, and reports
immobility as a triage input (an immobile survivor in flood water is a much
higher priority than one who is waving).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint, local_to_wgs84, wgs84_to_local
from sar.perception.detector import PERSON_LABELS, Detection
from sar.perception.geotag import GeoTag

__all__ = ["Track", "TrackStatus", "Tracker", "TrackObservation", "TRACK_LABELS"]

TRACK_LABELS = PERSON_LABELS + ("animal", "vehicle", "fire", "hot_rock", "cloth",
                                "flood_water", "debris_field", "collapsed_structure",
                                "damaged_structure", "exposed_powerline", "smoke",
                                "unknown")


class TrackStatus:
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    MERGED = "merged"
    LOST = "lost"


@dataclass
class TrackObservation:
    """One detection fused into one track."""

    t: float
    north: float
    east: float
    label: str
    score: float
    sigma_n: float
    sigma_e: float
    gsd_m: float
    modality: str
    peak_temp_c: Optional[float] = None
    mean_temp_c: Optional[float] = None
    background_temp_c: Optional[float] = None
    area_px: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    detection_uid: int = -1

    @property
    def weight(self) -> float:
        """Information content of the frame this came from, ``1/gsd^2``.

        Normalised so a 0.084 m/px frame (35 m AGL, 42 deg HFOV, 320 px core)
        scores 1.0.  A 110 m look at 0.264 m/px scores 0.10 - it is allowed to
        open a track but not to decide one.
        """
        g = max(self.gsd_m, 1e-4)
        return float(min((0.0842 / g) ** 2, 6.0))


@dataclass
class Track:
    """A persistent object on the ground with an accumulating evidence record."""

    tid: int
    north: float
    east: float
    vn: float = 0.0
    ve: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.eye(4) * 25.0)
    status: str = TrackStatus.CANDIDATE
    label: str = "unknown"
    confidence: float = 0.0
    created_t: float = 0.0
    updated_t: float = 0.0
    #: Last time the state was propagated forward.  Kept separate from
    #: ``updated_t`` (last *measurement*) because predicting with the time since
    #: the last measurement instead of the time since the last prediction applies
    #: the process noise cumulatively and inflates the covariance without bound:
    #: a track not re-observed for 20 frames ended up with a 53 m sigma, which
    #: then made the merge gate 140 m wide and swallowed the whole victim map.
    propagated_t: float = 0.0
    n_obs: int = 0
    n_modalities: set = field(default_factory=set)
    last_gsd_m: float = 0.0
    observations: List[TrackObservation] = field(default_factory=list)
    votes: Dict[str, float] = field(default_factory=dict)
    #: Weighted person evidence, normalised to 0..1 by total weight.
    person_evidence: float = 0.0
    #: Displacement-based immobility estimate, 0..1 (1 = never moved).
    immobility: float = 0.0
    #: Observations the fuser was willing to stand behind, versus ones it flagged
    #: as needing a closer look.  A track is only *confirmed* when it has at
    #: least one resolved observation; otherwise it stays a candidate and goes on
    #: the confirmation queue.  This is what stops a persistent hot rock from
    #: accumulating ten looks and promoting itself to a survivor.
    n_resolved: int = 0
    n_unresolved: int = 0
    #: Low confirmation passes flown specifically for this track.
    confirmation_looks: int = 0
    displacement_m: float = 0.0
    speed_ms: float = 0.0
    peak_temp_c: Optional[float] = None
    origin: Optional[GeoPoint] = None
    merged_into: int = -1
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def age_s(self) -> float:
        return max(self.updated_t - self.created_t, 0.0)

    @property
    def sigma_m(self) -> float:
        return float(math.sqrt(max(self.cov[0, 0] + self.cov[1, 1], 0.0)))

    @property
    def position_sigma_m(self) -> float:
        return float(math.sqrt(max(self.cov[0, 0], 0.0) + max(self.cov[1, 1], 0.0)) / math.sqrt(2.0))

    @property
    def is_person(self) -> bool:
        return self.label in PERSON_LABELS

    @property
    def cross_modal(self) -> bool:
        """Confirmed by more than one sensor modality.

        A fused observation arrives with ``modality='fused'``, which already means
        both sensors agreed in the same frame - that counts, and is stronger
        evidence than one sensor seeing the target twice.
        """
        mods = {m for m in self.n_modalities if m}
        return "fused" in mods or len(mods) > 1

    def geo(self, origin: GeoPoint) -> GeoPoint:
        return local_to_wgs84(self.north, self.east, 0.0, origin)

    def label_ranking(self, k: int = 3) -> List[Tuple[str, float]]:
        tot = sum(self.votes.values()) or 1.0
        return sorted(((lab, w / tot) for lab, w in self.votes.items()),
                      key=lambda kv: -kv[1])[:k]

    def to_dict(self, origin: Optional[GeoPoint] = None) -> Dict[str, Any]:
        g = self.geo(origin) if origin else None
        return {
            "tid": self.tid, "status": self.status, "label": self.label,
            "confidence": round(self.confidence, 3),
            "person_evidence": round(self.person_evidence, 3),
            "north_m": round(self.north, 2), "east_m": round(self.east, 2),
            "lat": g.lat if g else None, "lon": g.lon if g else None,
            "sigma_m": round(self.sigma_m, 2),
            "n_obs": self.n_obs, "modalities": sorted(self.n_modalities),
            "cross_modal": self.cross_modal,
            "speed_ms": round(self.speed_ms, 2),
            "immobility": round(self.immobility, 3),
            "n_resolved": self.n_resolved, "n_unresolved": self.n_unresolved,
            "confirmation_looks": self.confirmation_looks,
            "displacement_m": round(self.displacement_m, 1),
            "age_s": round(self.age_s, 1),
            "peak_temp_c": self.peak_temp_c,
            "labels": [(l, round(w, 3)) for l, w in self.label_ranking()],
            "last_gsd_m": round(self.last_gsd_m, 4),
            "notes": list(self.notes[-6:]),
        }


# --------------------------------------------------------------------------- #
class Tracker:
    """Kalman-filtered geographic tracks with weighted label voting."""

    def __init__(self,
                 origin: GeoPoint,
                 n_confirm: int = 2,
                 min_confirm_span_s: float = 0.5,
                 person_evidence_threshold: float = 0.50,
                 gate_sigma: float = 3.0,
                 max_gate_m: float = 30.0,
                 coast_limit_s: float = 90.0,
                 merge_radius_m: float = 6.0,
                 #: Ceiling on an unobserved track's reported position sigma.
                 sigma_cap_m: float = 25.0,
                 #: A frame finer than this has already had the close look that a
                 #: confirmation pass exists to provide.
                 gsd_resolve_m: float = 0.16,
                 #: Independent fine-GSD observations needed before a
                 #: single-modality track may confirm on its own.
                 n_resolve_fine: int = 4,
                 #: Initial 1-sigma velocity per label [m/s].  Drives how fast an
                 #: unobserved track's position circle opens up.
                 velocity_prior_ms: Optional[Dict[str, float]] = None,
                 velocity_prior_default_ms: float = 2.0,
                 process_noise_ms: float = 0.35,
                 immobile_process_noise_ms: float = 0.04,
                 max_tracks: int = 400) -> None:
        self.origin = origin
        self.n_confirm = n_confirm
        self.min_confirm_span_s = min_confirm_span_s
        self.person_threshold = person_evidence_threshold
        self.gate_sigma = gate_sigma
        self.max_gate_m = max_gate_m
        self.coast_limit_s = coast_limit_s
        self.merge_radius_m = merge_radius_m
        self.sigma_cap_m = float(sigma_cap_m)
        self.gsd_resolve_m = float(gsd_resolve_m)
        self.n_resolve_fine = int(n_resolve_fine)
        #: A casualty who is still moving is the exception, not the rule: the
        #: people this system is looking for are injured, trapped, hypothermic or
        #: in the water.  The prior is tight and the filter is free to widen it if
        #: the measurements say otherwise.
        self.velocity_prior_ms: Dict[str, float] = {
            "person": 0.15, "animal": 1.5, "vehicle": 8.0, "boat": 3.0,
            "fire": 0.0, "flood_water": 0.0, "smoke": 0.5,
        }
        if velocity_prior_ms:
            self.velocity_prior_ms.update(velocity_prior_ms)
        self.velocity_prior_default_ms = float(velocity_prior_default_ms)
        self.process_noise_ms = process_noise_ms
        self.immobile_process_noise_ms = immobile_process_noise_ms
        self.max_tracks = max_tracks
        self.tracks: List[Track] = []
        self._ids = itertools.count(1)
        self._det_uid = itertools.count(1)
        self.stats: Dict[str, Any] = {"created": 0, "confirmed": 0, "merged": 0,
                                      "lost": 0, "observations": 0}

    # ------------------------------------------------------------------ #
    def _new_track(self, obs: TrackObservation) -> Track:
        # Velocity prior per class.  Every track used to start with sigma_v =
        # 2 m/s, which is right for a car and absurd for a person lying on a
        # rooftop.  It mattered far more than it looks: position-only
        # measurements over a 1.25 s dwell cannot observe velocity at all, so
        # cov[v,v] stayed at 4 m^2/s^2 and the F cov F^T propagation pumped
        # dt^2 * 4 = 0.06 m^2 into the position variance on EVERY step, whether
        # or not the target had moved.  A survivor located to 0.14 m therefore
        # reported a 24 m circle a few seconds after the aircraft flew on.
        sv = self.velocity_prior_ms.get(obs.label, self.velocity_prior_default_ms)
        tr = Track(tid=next(self._ids), north=obs.north, east=obs.east,
                   cov=np.diag([max(obs.sigma_n, 0.5) ** 2, max(obs.sigma_e, 0.5) ** 2,
                                sv * sv, sv * sv]),
                   status=TrackStatus.CANDIDATE, created_t=obs.t, updated_t=obs.t,
                   propagated_t=obs.t, origin=self.origin)
        self._absorb(tr, obs)
        self.tracks.append(tr)
        self.stats["created"] += 1
        return tr

    def _absorb(self, tr: Track, obs: TrackObservation) -> None:
        """Add one observation's label vote and radiometry to a track."""
        w = obs.weight * max(obs.score, 0.02)
        tr.votes[obs.label] = tr.votes.get(obs.label, 0.0) + w
        tr.observations.append(obs)
        tr.n_obs += 1
        tr.n_modalities.add(obs.modality)
        attrs = obs.attributes or {}
        if attrs.get("needs_confirmation"):
            tr.n_unresolved += 1
        else:
            tr.n_resolved += 1
        if attrs.get("confirmation_pass"):
            tr.confirmation_looks += 1
            tr.n_resolved += 1
        tr.updated_t = obs.t
        tr.last_gsd_m = obs.gsd_m
        if obs.peak_temp_c is not None:
            # Keep the *best-informed* temperature: the one measured at the
            # smallest GSD, since sub-pixel radiometry biases peak temperature
            # toward the background (see experiment_subpixel_radiometry.py).
            if tr.peak_temp_c is None or obs.gsd_m <= min(
                    o.gsd_m for o in tr.observations if o.peak_temp_c is not None):
                tr.peak_temp_c = obs.peak_temp_c
        total = sum(tr.votes.values()) or 1.0
        person_w = sum(w_ for lab, w_ in tr.votes.items() if lab in PERSON_LABELS)
        tr.person_evidence = float(person_w / total)
        top, top_w = max(tr.votes.items(), key=lambda kv: kv[1])
        tr.label = top
        tr.confidence = float(top_w / total)

    # ------------------------------------------------------------------ #
    def _predict(self, tr: Track, dt: float) -> None:
        if dt <= 0:
            return
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        # An immobile survivor's estimate should keep tightening; a walking one
        # must be allowed to move.
        #
        # Scaling continuously on the filter's own speed estimate rather than
        # gating on the immobility flag matters a lot.  Under a constant-velocity
        # model an unobserved track's position variance grows as q^2 T^3 / 3, so
        # at q = 0.35 m/s^2 a survivor located to 0.14 m reported a 24 m circle
        # fourteen seconds after the aircraft flew on - a 170x overstatement that
        # would send a rescue boat to the wrong riverbank.  The immobility gate
        # did not fire because rooftop parallax alone scatters successive fixes by
        # more than a metre, which is measurement geometry, not target motion.
        # Speed is the quantity that actually answers "can this target have
        # moved": stationary means it cannot accelerate either.
        speed = math.hypot(tr.vn, tr.ve)
        q = 0.75 * speed
        if tr.immobility > 0.8:
            q = self.immobile_process_noise_ms
        q = float(np.clip(q, self.immobile_process_noise_ms,
                          self.process_noise_ms))
        q = max(q, 0.02)
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = (q * dt ** 2 / 2.0) ** 2 + 1e-6
        Q[2, 2] = Q[3, 3] = (q * dt) ** 2 + 1e-6
        tr.cov = F @ tr.cov @ F.T + Q
        tr.north += tr.vn * dt
        tr.east += tr.ve * dt
        # Ceiling on the reported circle.  Beyond it the honest statement is "we
        # lost them, go and look again", not a 100 m radius that quietly tells the
        # operator the target is somewhere in this district.  Capping rescales the
        # velocity covariance consistently so the next measurement still fuses.
        sig = tr.position_sigma_m
        if sig > self.sigma_cap_m:
            k = (self.sigma_cap_m / sig) ** 2
            tr.cov[:2, :2] *= k
            tr.cov[2:, :2] *= math.sqrt(k)
            tr.cov[:2, 2:] *= math.sqrt(k)

    def _update(self, tr: Track, obs: TrackObservation) -> float:
        """Kalman update.  Returns the normalised innovation (Mahalanobis)."""
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        R = np.diag([max(obs.sigma_n, 0.25) ** 2, max(obs.sigma_e, 0.25) ** 2])
        y = np.array([obs.north - tr.north, obs.east - tr.east])
        S = H @ tr.cov @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:  # pragma: no cover
            return float("inf")
        maha2 = float(y @ S_inv @ y)
        K = tr.cov @ H.T @ S_inv
        x = np.array([tr.north, tr.east, tr.vn, tr.ve]) + K @ y
        prev_n, prev_e = tr.north, tr.east
        tr.north, tr.east, tr.vn, tr.ve = (float(x[0]), float(x[1]),
                                           float(x[2]), float(x[3]))
        I = np.eye(4)
        # Joseph form: numerically stable when R is much smaller than P, which
        # happens exactly when a low-altitude pass measures a track opened by a
        # high-altitude one.
        A = I - K @ H
        tr.cov = A @ tr.cov @ A.T + K @ R @ K.T
        # Immobility from observed displacement relative to what measurement
        # noise could explain.
        d = math.hypot(obs.north - prev_n, obs.east - prev_e)
        span = max(obs.t - tr.created_t, 1e-3)
        tr.displacement_m = float(math.hypot(tr.north - tr.observations[0].north,
                                             tr.east - tr.observations[0].east))
        tr.speed_ms = tr.displacement_m / span
        noise_floor = 2.0 * math.hypot(obs.sigma_n, obs.sigma_e) + 0.5
        tr.immobility = float(np.clip(1.0 - tr.displacement_m / max(noise_floor * 3.0, 4.0),
                                      0.0, 1.0))
        return math.sqrt(max(maha2, 0.0))

    # ------------------------------------------------------------------ #
    def update(self, detections: Sequence[Detection], tags: Sequence[Optional[GeoTag]],
               t: float) -> List[Track]:
        """Fuse one frame's detections (with their geo-tags) into the track set.

        ``tags[i]`` is the :class:`GeoTag` for ``detections[i]``, or ``None`` if
        the pixel could not be projected (horizon, out of the terrain model).
        Unprojectable detections are dropped rather than guessed at.
        """
        for tr in self.tracks:
            if tr.status != TrackStatus.MERGED:
                # Propagate by the increment since the last propagation, never by
                # the age of the last measurement.
                last = tr.propagated_t if tr.propagated_t > 0 else tr.updated_t
                self._predict(tr, t - last)
                tr.propagated_t = t

        obs_list: List[TrackObservation] = []
        for det, tag in zip(detections, tags):
            if tag is None or not math.isfinite(tag.point.lat):
                continue
            # Prefer the local coordinates the tagger already computed.  Going
            # back through lat/lon per detection is slower, loses precision, and
            # silently produces (0, 0) for any tag whose geodetic point was not
            # filled in - which turns a 0.1 m fix into a 223 m one.
            if tag.north_m or tag.east_m:
                n, e = tag.north_m, tag.east_m
            else:
                n, e, _ = wgs84_to_local(tag.point, self.origin)
            obs_list.append(TrackObservation(
                t=t, north=float(n), east=float(e), label=det.label,
                score=float(det.score), sigma_n=tag.sigma_north_m,
                sigma_e=tag.sigma_east_m, gsd_m=max(tag.gsd_m, 1e-4),
                modality=det.modality, peak_temp_c=det.peak_temp_c,
                mean_temp_c=det.mean_temp_c, background_temp_c=det.background_temp_c,
                area_px=det.area_px, attributes=dict(det.attributes),
                detection_uid=next(self._det_uid),
            ))
        self.stats["observations"] += len(obs_list)

        # ---- greedy gated nearest-neighbour association -------------------
        live = [tr for tr in self.tracks if tr.status != TrackStatus.MERGED]
        pairs: List[Tuple[float, int, int]] = []
        for oi, obs in enumerate(obs_list):
            for ti, tr in enumerate(live):
                d = math.hypot(obs.north - tr.north, obs.east - tr.east)
                gate = min(self.gate_sigma * max(tr.position_sigma_m, 0.6), self.max_gate_m)
                if d <= max(gate, 2.5):
                    # Prefer the closest, but break ties toward the higher-scoring
                    # observation so a confident detection wins the association.
                    pairs.append((d - 0.5 * obs.score, ti, oi))
        pairs.sort()
        used_t: set = set()
        used_o: set = set()
        for _, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            self._update(live[ti], obs_list[oi])
            self._absorb(live[ti], obs_list[oi])
            self._review_status(live[ti])

        for oi, obs in enumerate(obs_list):
            if oi not in used_o:
                if len(self.tracks) >= self.max_tracks:
                    self._prune()
                self._review_status(self._new_track(obs))

        self._merge_duplicates(t)
        self._age_out(t)
        return [tr for tr in self.tracks if tr.status != TrackStatus.MERGED]

    # ------------------------------------------------------------------ #
    def _review_status(self, tr: Track) -> None:
        if tr.status == TrackStatus.MERGED:
            return
        person_like = tr.person_evidence >= self.person_threshold
        enough = tr.n_obs >= self.n_confirm
        span = tr.age_s >= self.min_confirm_span_s
        # Confirmation requires evidence some sensor was willing to stand behind:
        # genuine cross-modal agreement, a deliberate low confirmation look, or a
        # single-modality observation made when the other modality was blind
        # (night, smoke) and therefore had nothing to disagree with.  Repeated
        # looks from one sensor at one unconfirmed point target do not qualify -
        # persistence is exactly what a hot rock also has.
        resolved = tr.n_resolved > 0 or tr.confirmation_looks > 0
        # Escape hatch, and without it a whole survivor category is unreachable.
        #
        # The needs_confirmation flag exists because the frame was too coarse to
        # decide.  If it was NOT too coarse, decide.  A chest-deep survivor is
        # cold and invisible to the visible camera no matter how low the aircraft
        # flies, so cross-modal agreement is physically impossible for them; at
        # 35 m AGL those tracks accumulated four resolved-by-nobody observations
        # and stayed candidates for ever, while the 110 m hot rocks they were
        # meant to hold back were being correctly queued.  Several independent
        # looks from different geometry at a GSD where the LWIR radiometric gate
        # is decisive - inside the human thermal band, human-sized extent,
        # persistent across viewpoints - is the evidence a confirmation pass
        # would have produced, so it is accepted as such.
        best = min((o.gsd_m for o in tr.observations), default=float("inf"))
        fine_enough = tr.n_obs >= self.n_resolve_fine and best <= self.gsd_resolve_m
        if not resolved and fine_enough:
            resolved = True
            tr.notes.append(f"resolved_by_fine_gsd@{best:.3f}m/px n={tr.n_obs}")
        # The dwell-time guard exists to stop a single-frame glitch confirming.
        # It should not stop a *fast* aircraft confirming: at 12 m/s a survivor is
        # inside a 35 m swath for under a second, so requiring 0.5 s of track age
        # penalised exactly the sorties that cover the most ground per joule.  Four
        # independent looks at 0.084 m/px from different geometry is the evidence
        # the guard was asking for; wall-clock dwell is only a proxy for it, and
        # cross-modal agreement (waived above) is another.
        if fine_enough or tr.cross_modal:
            span = True
        # Two sensors agreeing in ONE frame is stronger evidence than one sensor
        # seeing the target twice, so cross-modal agreement waives the dwell-time
        # requirement.  Without this a fast 7 m/s flyover of a survivor never
        # accumulates enough seconds to confirm, and the fastest passes - the ones
        # that cover the most ground - would be the least able to raise an alert.
        if person_like and enough and tr.cross_modal:
            span = True
        if person_like and enough and span and resolved:
            if tr.status != TrackStatus.CONFIRMED:
                tr.status = TrackStatus.CONFIRMED
                self.stats["confirmed"] += 1
                tr.notes.append(f"confirmed@{tr.updated_t:.1f}s "
                                f"ev={tr.person_evidence:.2f} n={tr.n_obs} "
                                f"gsd={tr.last_gsd_m:.3f}")
        elif not person_like and tr.status == TrackStatus.CONFIRMED:
            # A low pass has overruled an earlier high pass - exactly the
            # behaviour the 1/gsd^2 vote weighting exists to produce.
            tr.status = TrackStatus.CANDIDATE
            tr.notes.append(f"demoted@{tr.updated_t:.1f}s ev={tr.person_evidence:.2f} "
                            f"label={tr.label}")

    def _merge_duplicates(self, t: float) -> None:
        """Collapse tracks that have converged onto the same object.

        The same survivor is routinely re-acquired by a later pass with a
        different geometry; without merging the victim map fills with duplicates
        and the sortie re-visits the same person, burning the endurance that
        should be spent on unsearched area.
        """
        live = [tr for tr in self.tracks if tr.status != TrackStatus.MERGED]
        live.sort(key=lambda x: (-x.n_obs, -x.person_evidence))
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if b.status == TrackStatus.MERGED:
                    continue
                d = math.hypot(a.north - b.north, a.east - b.east)
                # Bounded above: an uncertain track must not be allowed to
                # swallow a confident one just because its own ellipse is huge.
                tol = min(max(self.merge_radius_m,
                              2.0 * math.hypot(a.position_sigma_m, b.position_sigma_m)),
                          self.merge_radius_m * 3.0)
                # Never merge across the person / non-person boundary.  A survivor
                # and a fire 15 m apart are two facts, not one, and merging them
                # destroyed four survivor tracks in the first end-to-end run
                # because the merge predicate accepted any pair of nearby tracks.
                same_class = (a.is_person == b.is_person)
                if d <= tol and same_class and (a.label == b.label
                                                or a.person_evidence > 0.3
                                                or b.person_evidence > 0.3):
                    b.status = TrackStatus.MERGED
                    b.merged_into = a.tid
                    a.observations.extend(b.observations)
                    a.n_obs = len(a.observations)
                    a.n_modalities |= b.n_modalities
                    for lab, w in b.votes.items():
                        a.votes[lab] = a.votes.get(lab, 0.0) + w
                    total = sum(a.votes.values()) or 1.0
                    a.person_evidence = float(sum(w for lab, w in a.votes.items()
                                                  if lab in PERSON_LABELS) / total)
                    top, top_w = max(a.votes.items(), key=lambda kv: kv[1])
                    a.label, a.confidence = top, float(top_w / total)
                    a.notes.append(f"merged T{b.tid} at d={d:.1f}m")
                    self.stats["merged"] += 1

    def _age_out(self, t: float) -> None:
        for tr in self.tracks:
            if tr.status in (TrackStatus.MERGED, TrackStatus.LOST):
                continue
            dt = t - tr.updated_t
            if dt > self.coast_limit_s:
                tr.status = TrackStatus.LOST
                tr.notes.append(f"lost after {dt:.0f}s without observation")
                self.stats["lost"] += 1
            elif dt > 2.0 * max(self.min_confirm_span_s, 2.0):
                tr.status = TrackStatus.COASTING

    def _prune(self) -> None:
        """Drop the least promising tracks when the budget is full."""
        if len(self.tracks) < self.max_tracks:
            return
        self.tracks.sort(key=lambda tr: (
            tr.status == TrackStatus.CONFIRMED,
            tr.person_evidence, tr.n_obs, tr.updated_t))
        drop = len(self.tracks) - self.max_tracks + self.max_tracks // 4
        self.tracks = self.tracks[drop:]

    # ------------------------------------------------------------------ #
    def confirmed_persons(self) -> List[Track]:
        return [tr for tr in self.tracks
                if tr.status == TrackStatus.CONFIRMED and tr.is_person]

    def candidates_needing_confirmation(self, gsd_threshold: float = 0.16) -> List[Track]:
        """Tracks whose only evidence came from frames too coarse to decide.

        This is the work queue for the low confirmation pass.  A track opened at
        0.26 m/px (110 m AGL) has not been looked at properly yet; the planner
        routes a narrow-FOV low pass over it before either committing a rescue
        asset or discarding it.
        """
        out = []
        for tr in self.tracks:
            if tr.status not in (TrackStatus.CANDIDATE, TrackStatus.CONFIRMED):
                continue
            if not tr.is_person:
                continue
            best = min((o.gsd_m for o in tr.observations), default=float("inf"))
            if best > gsd_threshold:
                out.append(tr)
        return out

    def person_tracks(self) -> List[Track]:
        return [tr for tr in self.tracks if tr.is_person
                and tr.status != TrackStatus.MERGED]

    def hazard_tracks(self) -> List[Track]:
        return [tr for tr in self.tracks if not tr.is_person
                and tr.status in (TrackStatus.CONFIRMED, TrackStatus.CANDIDATE)]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_tracks": len([t for t in self.tracks if t.status != TrackStatus.MERGED]),
            "n_confirmed": len(self.confirmed_persons()),
            "n_candidates": len([t for t in self.tracks
                                 if t.status == TrackStatus.CANDIDATE]),
            "n_needing_confirmation": len(self.candidates_needing_confirmation()),
            "stats": dict(self.stats),
        }
