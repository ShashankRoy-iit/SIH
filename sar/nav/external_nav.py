"""GPS-denied navigation: external-nav feeding and EKF source-set management.

A search-and-rescue sortie goes to exactly the places where GNSS is worst -
urban canyons after an earthquake, flood plains under heavy rain attenuation,
valleys with jamming or debris on the horizon.  Losing position at 60 m over a
street with people in it is not a degraded mode to be tolerated, it is the end
of the sortie, so the system is built to keep flying through it.

Three things have to be right, and they interact.

**The vehicle must have somewhere to switch to.**  ArduPilot's EKF3 supports
several *source sets*, and ``MAV_CMD_SET_EKF_SOURCE_SET`` (42007) selects one in
a single command.  Configuring a set means writing ``EK3_SRCn_*``, and those
parameters use three different value spaces - see
:class:`~sar.mavlink.protocol.SourceXY`, :class:`~sar.mavlink.protocol.SourceZ`
and :class:`~sar.mavlink.protocol.SourceYaw`.  They are also validated at
pre-arm: a source set naming optical flow or external nav will not arm unless
the corresponding backend exists and is healthy, which is why the feeder below
has to be running *before* arming rather than started when the denial happens.

**Something has to supply the external navigation.**  That is the onboard
computer: the same unit running perception already has the cameras, so visual
odometry is computed from imagery it is capturing anyway and published as
MAVLink ``ODOMETRY`` / ``VISION_POSITION_ESTIMATE``.  :class:`ExternalNavFeeder`
does the publishing, at a rate EKF3 will accept (its aiding times out at roughly
0.5 s, so 20 Hz is the floor and this defaults to 30).

**The switch has to be made deliberately, and taken back deliberately.**
:class:`EkfSourceManager` owns that policy.  The two failure modes it exists to
avoid are *flapping* and *switching to something dead*.  Flapping is worse than
staying on a degraded source because every transition resets part of the filter;
switching to an unhealthy source is worse still, because the EKF then has no
position at all.  So transitions are hysteresis-gated in both directions and a
switch is refused outright if the destination source is not currently healthy
and being received at rate.

The output that matters downstream is not the switch itself but the *uncertainty*
it implies.  :meth:`NavQualityMonitor.nav_quality` turns the current state into
the :class:`~sar.perception.geotag.NavQuality` the geo-tagger consumes, so a
survivor detected during a denial is reported with a 25 m error ellipse rather
than a 1.2 m one.  A confident wrong position sends a rescue boat to the wrong
riverbank; an honest wide one sends it to the right bank with a search pattern.
"""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.mavlink.protocol import SOURCE_SET, SourceXY, SourceYaw

log = logging.getLogger("sar.nav")

__all__ = [
    "NavState", "NavAssessment", "NavQualityMonitor", "ExternalNavFeeder",
    "EkfSourceManager", "VioSource", "TelemetryVioSource", "SimulatedVioSource",
    "ExternalNavSample",
]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class NavState(enum.Enum):
    """How the vehicle currently knows where it is."""

    #: GNSS with a good fix.  The normal case.
    GNSS_OK = "gnss_ok"
    #: GNSS present but degraded - few satellites, high HDOP, or a canyon
    #: multipath solution.  Still usable, but the geo-tag error ellipse widens
    #: and the manager starts preparing the fallback rather than waiting for the
    #: fix to disappear.
    GNSS_DEGRADED = "gnss_degraded"
    #: No usable GNSS.  Position and velocity are coming from external nav.
    EXTERNAL_NAV = "external_nav"
    #: No GNSS and no external nav position: optical-flow velocity with baro
    #: height only.  The aircraft can hold attitude and stop itself, but it is
    #: dead-reckoning position and must not be relied on to survey.
    FLOW_ONLY = "flow_only"
    #: Nothing.  Inertial only, which diverges within seconds.  Land.
    LOST = "lost"

    @property
    def position_absolute(self) -> bool:
        """True when position is anchored to something that does not drift."""
        return self in (NavState.GNSS_OK, NavState.GNSS_DEGRADED)

    @property
    def usable_for_survey(self) -> bool:
        """True when the coverage record can still be trusted.

        ``EXTERNAL_NAV`` qualifies because visual-inertial odometry drifts slowly
        and is re-anchored whenever GNSS returns.  ``FLOW_ONLY`` does not: with no
        position aiding the aircraft can complete a survey pattern and have no
        idea where any of it was, which is worse than not flying it.
        """
        return self in (NavState.GNSS_OK, NavState.GNSS_DEGRADED,
                        NavState.EXTERNAL_NAV)


def euler_to_quat(roll_deg: float, pitch_deg: float,
                  yaw_deg: float) -> Tuple[float, float, float, float]:
    """Aerospace ZYX (yaw-pitch-roll) Euler angles in degrees to ``(w, x, y, z)``.

    Same convention as MAVLink ``ATTITUDE`` and ArduPilot's ``Quaternion``, which
    matters because the ODOMETRY quaternion is used both as the reported attitude
    and to rotate the body-frame velocity back to earth axes.  A different
    convention would be accepted without complaint and quietly misalign both.
    """
    r, pt, y = (math.radians(float(a)) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r / 2.0), math.sin(r / 2.0)
    cp, sp = math.cos(pt / 2.0), math.sin(pt / 2.0)
    cy, sy = math.cos(y / 2.0), math.sin(y / 2.0)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def quat_rotate(q: Sequence[float], v: Sequence[float],
                inverse: bool = False) -> Tuple[float, float, float]:
    """Rotate ``v`` by quaternion ``q`` (or by its conjugate if ``inverse``).

    ``inverse=True`` takes earth-frame to body-frame, which is the direction
    needed before publishing velocities in an ODOMETRY message whose child frame
    is ``BODY_FRD``: the receiver multiplies by ``q`` to get back to earth, so
    sending earth-frame values there would rotate them twice.
    """
    w, x, y, z = (float(a) for a in q)
    if inverse:
        x, y, z = -x, -y, -z
    vx, vy, vz = (float(a) for a in v)
    # t = 2 * (q_vec x v); v' = v + q_w*t + q_vec x t   (the compact form)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


@dataclass
class ExternalNavSample:
    """One external-navigation estimate, ready to publish."""

    t: float
    north_m: float
    east_m: float
    down_m: float
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    #: 1-sigma position error the *source* believes it has.  Published in the
    #: ODOMETRY covariance, so telling the EKF the aiding is better than it is
    #: makes it trust a diverging estimate over its own dead reckoning - which is
    #: worse than not aiding at all.
    pos_sigma_m: float = 0.15
    vel_sigma_ms: float = 0.15
    valid: bool = True
    #: 0..100.  Published as ODOMETRY ``quality``, which the flight stack gates
    #: against ``VISO_QUAL_MIN``; 0 means "unknown" in MAVLink, so a source that
    #: reports a fraction rather than a percentage gets clamped at the floor and
    #: silently stops aiding whenever ``VISO_QUAL_MIN`` is above zero.
    confidence: int = 100
    source: str = "vio"

    # ---- derived, and what actually goes on the wire ----
    @property
    def quat(self) -> Tuple[float, float, float, float]:
        """Attitude as ``(w, x, y, z)``, body-to-earth."""
        return euler_to_quat(self.roll_deg, self.pitch_deg, self.yaw_deg)

    @property
    def v_body(self) -> Tuple[float, float, float]:
        """Velocity rotated earth-to-body.

        ODOMETRY with ``child_frame_id = BODY_FRD`` requires body-relative
        velocity, because the receiver rotates it back with the quaternion.  NED
        values are carried on ``vn/ve/vd`` for the log and for
        :meth:`~sar.mavlink.MavConnection.send_external_position`, which has no
        child frame and takes earth axes.
        """
        return quat_rotate(self.quat, (self.vn, self.ve, self.vd), inverse=True)


class VioSource:
    """Interface for anything that can produce an external-nav estimate.

    On the aircraft this is a real visual-inertial odometry stack (ORB-SLAM3,
    VINS-Fusion, or the vendor's own on a Qualcomm Flight board).  In simulation
    it is one of the two implementations below.  The feeder does not care which,
    which is what lets the same denial-handling code be tested against a model
    with realistic drift and then flown unchanged.
    """

    def sample(self, t: float) -> Optional[ExternalNavSample]:
        raise NotImplementedError

    @property
    def healthy(self) -> bool:
        return True

    def relocalise(self, north_m: float, east_m: float, down_m: float,
                   quality: float = 1.0) -> None:
        """Anchor the estimate to an absolute fix, if the source supports it."""


class TelemetryVioSource(VioSource):
    """External nav derived from the vehicle's own telemetry.

    Used when flying against ArduPilot SITL, or against hardware where a
    separate VIO stack is not running.  The estimate is the EKF's current
    solution re-published with added noise and latency - which sounds circular,
    and is: it cannot improve the navigation solution, only keep the
    ``VisOdom`` backend healthy so that a source set naming ExternalNav is
    armable and testable.

    That circularity is worth being explicit about, because it bounds what this
    configuration can prove.  It exercises the *plumbing* - message rates, frame
    conventions, timestamps on the boot clock, source-set switching, the
    geo-tagger's response to a widened error ellipse.  It does not exercise VIO
    drift.  For that use :class:`SimulatedVioSource`, which wraps the in-house
    :class:`~sar.vehicle.sensors.VioSensor` and accumulates real random-walk
    error against a ground truth the vehicle does not have.
    """

    def __init__(self, conn: Any, pos_sigma_m: float = 0.15,
                 vel_sigma_ms: float = 0.15, latency_s: float = 0.03,
                 origin_north_m: float = 0.0, origin_east_m: float = 0.0,
                 seed: int = 11) -> None:
        self.conn = conn
        self.pos_sigma_m = float(pos_sigma_m)
        self.vel_sigma_ms = float(vel_sigma_ms)
        self.latency_s = float(latency_s)
        self.origin = (float(origin_north_m), float(origin_east_m))
        self.rng = np.random.default_rng(seed)
        self._buffer: List[Tuple[float, Tuple[float, float, float,
                                              float, float, float, float]]] = []
        self._healthy = False
        self._last_local: Optional[Tuple[float, float, float]] = None

    # ------------------------------------------------------------------ #
    def set_origin_from_telemetry(self) -> bool:
        """Latch the current position as the local-frame origin.

        External nav is published in metres from an origin, and that origin has
        to be the EKF's own or the two frames disagree by however far the
        aircraft has travelled.  Called once, before flight.
        """
        tel = self.conn.telemetry
        if not tel.has_position:
            return False
        self.origin = (0.0, 0.0)
        self._origin_latlon = (tel.lat, tel.lon)
        return True

    def _local(self, lat: Optional[float], lon: Optional[float]
               ) -> Optional[Tuple[float, float]]:
        """Convert telemetry lat/lon to metres from the latched origin."""
        if lat is None or lon is None:
            return None
        olat, olon = getattr(self, "_origin_latlon", (None, None))
        if olat is None:
            return None
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(olat))
        return ((lat - olat) * m_per_deg_lat, (lon - olon) * m_per_deg_lon)

    def sample(self, t: float) -> Optional[ExternalNavSample]:
        # Refresh from the link: this runs on the feeder thread and nobody else
        # is pumping while it does, so without this the estimate is built from
        # whatever the main loop last happened to read.
        try:
            self.conn.pump(0.02)
        except Exception:
            pass
        tel = self.conn.telemetry
        local = self._local(tel.lat, tel.lon)
        if local is None or tel.alt_rel_m is None:
            self._healthy = False
            return None
        # Delay the estimate by the configured latency, which is what a real VIO
        # does: it reports where the aircraft *was*, and a feeder that publishes
        # instantaneous positions teaches the EKF a fiction that collapses the
        # moment real hardware is attached.
        self._buffer.append((t, (local[0], local[1], -tel.alt_rel_m,
                                 tel.vn, tel.ve, tel.vd,
                                 tel.roll_deg, tel.pitch_deg, tel.yaw_deg)))
        cutoff = t - self.latency_s
        while len(self._buffer) > 2 and self._buffer[1][0] <= cutoff:
            self._buffer.pop(0)
        pick = self._buffer[0][1] if self._buffer[0][0] <= cutoff \
            else self._buffer[-1][1]
        n, e, d, vn, ve, vd, roll, pitch, yaw = pick
        s = ExternalNavSample(
            t=t,
            north_m=n + float(self.rng.normal(0.0, self.pos_sigma_m)),
            east_m=e + float(self.rng.normal(0.0, self.pos_sigma_m)),
            down_m=d + float(self.rng.normal(0.0, self.pos_sigma_m)),
            vn=vn + float(self.rng.normal(0.0, self.vel_sigma_ms)),
            ve=ve + float(self.rng.normal(0.0, self.vel_sigma_ms)),
            vd=vd + float(self.rng.normal(0.0, self.vel_sigma_ms)),
            yaw_deg=yaw, roll_deg=roll, pitch_deg=pitch,
            pos_sigma_m=self.pos_sigma_m,
            vel_sigma_ms=self.vel_sigma_ms, valid=True,
            confidence=95, source="telemetry_echo")
        self._healthy = True
        self._last_local = (n, e, d)
        return s

    @property
    def healthy(self) -> bool:
        return self._healthy


class SimulatedVioSource(VioSource):
    """External nav from the in-house VIO model, with real drift.

    Wraps :class:`~sar.vehicle.sensors.VioSensor`, which accumulates a
    random-walk position error whose rate depends on the visual texture below
    the aircraft and goes invalid entirely over featureless ground.  Over a flood
    that is not an edge case, it is the normal case: open water has no features,
    so VIO is lost precisely where the survivors are.  Any denial-handling logic
    that has not been tested against that will fail in the field.

    ``truth_fn`` supplies the plant state the VIO measures; without it the source
    reports itself permanently lost.
    """

    def __init__(self, vio_sensor: Any, truth_fn: Callable[[float], Any],
                 pos_sigma_m: float = 0.15, vel_sigma_ms: float = 0.15) -> None:
        self.vio = vio_sensor
        self.truth_fn = truth_fn
        self.pos_sigma_m = float(pos_sigma_m)
        self.vel_sigma_ms = float(vel_sigma_ms)
        self._healthy = False
        self.lost_fraction = 0.0
        self._n = 0
        self._n_lost = 0

    def sample(self, t: float) -> Optional[ExternalNavSample]:
        st = self.truth_fn(t)
        if st is None:
            self._healthy = False
            return None
        s = self.vio.measure(st, t)
        self._n += 1
        self._n_lost += 0 if s.valid else 1
        self.lost_fraction = self._n_lost / max(self._n, 1)
        self._healthy = bool(s.valid)
        if not s.valid:
            return None
        return ExternalNavSample(
            t=t, north_m=float(s.position[0]), east_m=float(s.position[1]),
            down_m=float(s.position[2]),
            vn=float(s.velocity[0]), ve=float(s.velocity[1]),
            vd=float(s.velocity[2]),
            roll_deg=math.degrees(float(getattr(s, "roll", 0.0))),
            pitch_deg=math.degrees(float(getattr(s, "pitch", 0.0))),
            yaw_deg=math.degrees(float(s.yaw)),
            pos_sigma_m=self.pos_sigma_m, vel_sigma_ms=self.vel_sigma_ms,
            valid=True,
            # The sensor model reports a 0..1 fraction; the wire field is a
            # 0..100 percentage gated by VISO_QUAL_MIN.
            confidence=max(1, min(100, int(float(s.confidence) * 100.0))),
            source="vio_model")

    @property
    def healthy(self) -> bool:
        return self._healthy

    def relocalise(self, north_m: float, east_m: float, down_m: float,
                   quality: float = 1.0) -> None:
        if hasattr(self.vio, "relocalise"):
            self.vio.relocalise(np.array([north_m, east_m, down_m]), quality)


# --------------------------------------------------------------------------- #
# Quality assessment
# --------------------------------------------------------------------------- #
@dataclass
class NavAssessment:
    """The navigation state at one instant, with the reasoning attached."""

    t: float
    state: NavState
    reason: str
    #: Seconds spent continuously in this state.  Used for hysteresis.
    dwell_s: float = 0.0
    n_satellites: int = 0
    hdop: float = 99.0
    gps_fix: int = 0
    pos_sigma_m: float = 100.0
    external_nav_hz: float = 0.0
    external_nav_healthy: bool = False
    external_nav_age_s: float = float("inf")
    flow_healthy: bool = False
    #: Set when a denial is expected from the mission geometry rather than
    #: observed from telemetry - entering a mapped canyon or jamming zone.  The
    #: manager uses this to switch *before* the fix is lost rather than after,
    #: because switching after means switching during the transition.
    denial_predicted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"t": round(self.t, 3), "state": self.state.value,
                "reason": self.reason, "dwell_s": round(self.dwell_s, 2),
                "n_satellites": self.n_satellites, "hdop": round(self.hdop, 2),
                "gps_fix": self.gps_fix,
                "pos_sigma_m": round(self.pos_sigma_m, 3),
                "external_nav_hz": round(self.external_nav_hz, 1),
                "external_nav_healthy": self.external_nav_healthy,
                "flow_healthy": self.flow_healthy,
                "denial_predicted": self.denial_predicted}


class NavQualityMonitor:
    """Classifies telemetry into a :class:`NavState`, with hysteresis.

    Parameters
    ----------
    min_sats_ok, min_sats_degraded : int
        Satellite counts for a good and a marginal fix.  The reference receiver
        is a 30+ satellite multi-GNSS module, so 10 is already a poor sky.
    hdop_ok, hdop_bad : float
        Dilution of precision thresholds.  Below ``hdop_ok`` the fix is good;
        above ``hdop_bad`` it is not to be trusted for position even though the
        receiver still reports a fix type - which is the situation in an urban
        canyon, and the reason fix type alone is not a sufficient test.
    hysteresis_s : float
        How long a condition must persist before the reported state changes.
        Without it a single bad epoch flips the state and the manager switches
        source sets, and switching source sets is far more disruptive than
        tolerating one bad epoch.
    denial_zones : sequence
        ``(north, east, radius_m, severity)`` of mapped GNSS-denied areas, so a
        denial can be predicted from position rather than only detected from
        telemetry.
    """

    def __init__(self,
                 min_sats_ok: int = 10,
                 min_sats_degraded: int = 6,
                 hdop_ok: float = 1.6,
                 hdop_bad: float = 4.0,
                 min_fix_type: int = 3,
                 hysteresis_s: float = 1.5,
                 external_nav_min_hz: float = 15.0,
                 external_nav_max_age_s: float = 0.5,
                 denial_zones: Sequence[Tuple[float, float, float, float]] = (),
                 origin: Any = None) -> None:
        self.min_sats_ok = int(min_sats_ok)
        self.min_sats_degraded = int(min_sats_degraded)
        self.hdop_ok = float(hdop_ok)
        self.hdop_bad = float(hdop_bad)
        self.min_fix_type = int(min_fix_type)
        self.hysteresis_s = float(hysteresis_s)
        self.external_nav_min_hz = float(external_nav_min_hz)
        self.external_nav_max_age_s = float(external_nav_max_age_s)
        self.denial_zones = list(denial_zones)
        self.origin = origin

        self._state = NavState.GNSS_OK
        self._candidate = NavState.GNSS_OK
        self._candidate_since = 0.0
        self._history: List[NavAssessment] = []
        self.transitions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> NavState:
        return self._state

    def history(self, limit: int = 500) -> List[NavAssessment]:
        return self._history[-limit:]

    def denial_at(self, north: float, east: float) -> Tuple[float, float]:
        """``(denial 0..1, canyon 0..1)`` at a local position, from the map."""
        denial = canyon = 0.0
        for (zn, ze, zr, sev) in self.denial_zones:
            d = math.hypot(north - zn, east - ze)
            if d < zr:
                f = sev * (1.0 - d / zr) ** 0.5
                if sev >= 0.85:
                    denial = max(denial, f)
                else:
                    canyon = max(canyon, f)
        return min(1.0, denial), min(1.0, canyon)

    # ------------------------------------------------------------------ #
    def assess(self, telemetry: Any, t: Optional[float] = None,
               external_nav_hz: float = 0.0,
               external_nav_healthy: bool = False,
               external_nav_age_s: float = float("inf"),
               flow_healthy: bool = False,
               local_north: Optional[float] = None,
               local_east: Optional[float] = None) -> NavAssessment:
        """Classify one telemetry snapshot and advance the hysteresis."""
        now = t if t is not None else time.time()
        tel = telemetry
        fix = int(getattr(tel, "gps_fix", 0) or 0)
        sats = int(getattr(tel, "n_satellites", 0) or 0)
        hdop = float(getattr(tel, "hdop", 99.0) or 99.0)
        pos_sigma = float(getattr(tel, "horizontal_pos_sigma_m", 100.0))
        src_set = int(getattr(tel, "ekf_source_set", 1) or 1)

        predicted_denial = False
        if local_north is not None and local_east is not None:
            d, _c = self.denial_at(float(local_north), float(local_east))
            predicted_denial = d > 0.5

        extnav_usable = (external_nav_healthy
                         and external_nav_hz >= self.external_nav_min_hz
                         and external_nav_age_s <= self.external_nav_max_age_s)

        # Instantaneous classification, before hysteresis.
        gps_usable = fix >= self.min_fix_type and sats >= self.min_sats_degraded
        if gps_usable and not predicted_denial:
            if sats >= self.min_sats_ok and hdop <= self.hdop_ok:
                raw, reason = NavState.GNSS_OK, f"{sats} sats hdop {hdop:.1f}"
            elif hdop >= self.hdop_bad or sats < self.min_sats_ok:
                raw = NavState.GNSS_DEGRADED
                reason = (f"degraded: {sats} sats hdop {hdop:.1f}"
                          + (f" > {self.hdop_bad}" if hdop >= self.hdop_bad else ""))
            else:
                raw, reason = NavState.GNSS_OK, f"{sats} sats hdop {hdop:.1f}"
        elif extnav_usable:
            raw = NavState.EXTERNAL_NAV
            reason = (f"gnss unusable (fix={fix} sats={sats} hdop={hdop:.1f}"
                      f"{', predicted denial zone' if predicted_denial else ''}); "
                      f"external nav at {external_nav_hz:.0f} Hz")
        elif flow_healthy:
            raw = NavState.FLOW_ONLY
            reason = ("gnss unusable and external nav not available "
                      f"(hz={external_nav_hz:.0f} healthy={external_nav_healthy} "
                      f"age={external_nav_age_s:.2f}s); optical flow only")
        else:
            raw = NavState.LOST
            reason = "no gnss, no external nav, no flow"

        # Hysteresis: a new state has to persist before it is adopted.  Losing
        # GNSS is acted on immediately, because waiting costs the fix; regaining
        # it is not, because a fix that comes back for one epoch and goes again
        # is worse than useless.
        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = now
        dwell = now - self._candidate_since
        degrading = _severity(raw) > _severity(self._state)
        if raw != self._state and (degrading or dwell >= self.hysteresis_s):
            old = self._state
            self._state = raw
            self.transitions.append({"t": round(now, 3), "from": old.value,
                                     "to": raw.value, "reason": reason,
                                     "dwell_s": round(dwell, 2)})
            log.info("nav state %s -> %s: %s", old.value, raw.value, reason)

        a = NavAssessment(t=now, state=self._state, reason=reason, dwell_s=dwell,
                          n_satellites=sats, hdop=hdop, gps_fix=fix,
                          pos_sigma_m=pos_sigma,
                          external_nav_hz=external_nav_hz,
                          external_nav_healthy=external_nav_healthy,
                          external_nav_age_s=external_nav_age_s,
                          flow_healthy=flow_healthy,
                          denial_predicted=predicted_denial)
        self._history.append(a)
        if len(self._history) > 5000:
            del self._history[:2500]
        return a

    # ------------------------------------------------------------------ #
    def nav_quality(self, telemetry: Any, assessment: Optional[NavAssessment] = None,
                    vio_drift_ms: float = 0.15) -> Dict[str, Any]:
        """Navigation state expressed as geo-tagging uncertainty inputs.

        This is the seam into the perception stack.  The geo-tagger needs a
        position sigma, an attitude sigma and a time-since-absolute-fix, and none
        of those is a single MAVLink field.  In particular ``seconds_since_fix``
        is what makes a denial honest downstream: the geo-tagger multiplies it by
        a drift rate, so a survivor detected 60 s into a denial is reported with a
        10 m ellipse rather than the 1.2 m ellipse the last good fix would imply.

        For ``EXTERNAL_NAV`` the sigma is built from the VIO's own drift rate
        rather than from the EKF's reported variance, because the EKF's variance
        reflects how consistent the aiding is with the inertial solution, not how
        far the aiding has drifted from the truth - and a smoothly diverging VIO
        is perfectly self-consistent.
        """
        tel = telemetry
        a = assessment
        state = a.state if a else self._state
        base = float(getattr(tel, "horizontal_pos_sigma_m", 1.5))
        seconds_since_fix = 0.0
        source = "gnss"
        drift = 0.0

        if state == NavState.GNSS_OK:
            source, drift = "gnss", 0.0
        elif state == NavState.GNSS_DEGRADED:
            source = "gnss"
            drift = 0.0
            base = max(base, 2.5)
        elif state == NavState.EXTERNAL_NAV:
            source = "ekf_external_nav"
            drift = float(vio_drift_ms)
            # VIO error is a random walk: it grows with the square root of time,
            # not linearly, until re-anchored.  The linear term the geo-tagger
            # applies is therefore conservative at long denial times, which is
            # the right direction to be wrong in.
            base = max(base, 0.25)
        elif state == NavState.FLOW_ONLY:
            source = "dead_reckoning"
            drift = 0.6                      # flow gives velocity, not position
            base = max(base, 5.0)
        else:
            source = "dead_reckoning"
            drift = 1.5
            base = max(base, 20.0)

        if a and a.dwell_s and state in (NavState.EXTERNAL_NAV, NavState.FLOW_ONLY,
                                         NavState.LOST):
            seconds_since_fix = float(a.dwell_s)

        att_sigma = getattr(tel, "_att_sigma_deg", None)
        if att_sigma is None:
            # Yaw is the weak axis on this airframe: no compass, so heading comes
            # from GNSS course or from the VIO.  Both are unavailable in the
            # states that matter, and an attitude error swings a geo-tagged point
            # by range x angle, which at 60 m and 3 deg is 3 m of position error.
            att_sigma = {NavState.GNSS_OK: 0.9, NavState.GNSS_DEGRADED: 2.0,
                         NavState.EXTERNAL_NAV: 1.5, NavState.FLOW_ONLY: 4.0,
                         NavState.LOST: 8.0}[state]

        return {
            "pos_sigma_m": base,
            "att_sigma_deg": float(att_sigma),
            "alt_sigma_m": 1.5 if state.position_absolute else 3.0,
            "seconds_since_fix": seconds_since_fix,
            "drift_rate_ms": drift,
            "n_satellites": int(getattr(tel, "n_satellites", 0) or 0),
            "hdop": float(getattr(tel, "hdop", 99.0) or 99.0),
            "source": source,
            "nav_state": state.value,
            "gps_denied": not state.position_absolute,
            "usable_for_survey": state.usable_for_survey,
        }


def _severity(state: NavState) -> int:
    """Ordering used to decide whether a change is a degradation."""
    return {NavState.GNSS_OK: 0, NavState.GNSS_DEGRADED: 1,
            NavState.EXTERNAL_NAV: 2, NavState.FLOW_ONLY: 3,
            NavState.LOST: 4}[state]


# --------------------------------------------------------------------------- #
# Feeder
# --------------------------------------------------------------------------- #
class ExternalNavFeeder:
    """Publishes external navigation to the vehicle at a rate EKF3 accepts.

    Runs on its own thread.  It has to be started *before* arming when a source
    set naming ExternalNav is configured, because ArduPilot's pre-arm check
    requires the ``VisOdom`` backend to be healthy and it only becomes healthy
    once samples arrive - a chicken-and-egg that reads as an unexplained
    "VisOdom: not healthy" refusal if the feeder is started on denial instead of
    on boot.

    Parameters
    ----------
    rate_hz : float
        Publish rate.  EKF3's aiding times out at roughly 0.5 s, so anything
        below 2 Hz is pointless; 20 Hz is the documented floor and 30 leaves
        margin for a lossy link.  Above ~50 Hz the messages cost more link
        bandwidth than they add accuracy.
    dropout : float
        Fraction of samples deliberately not sent, to test that the manager
        notices a degrading source before the EKF times out.
    use_odometry : bool
        Prefer ``ODOMETRY`` (331), which carries a full covariance and explicit
        parent/child frames, over the legacy ``VISION_POSITION_ESTIMATE``.
        Requires ``VISO_TYPE`` to be set for MAVLink visual odometry either way.
    """

    def __init__(self, conn: Any, source: VioSource, rate_hz: float = 30.0,
                 dropout: float = 0.0, use_odometry: bool = True,
                 ang_sigma_rad: float = 0.10,
                 name: str = "extnav") -> None:
        self.conn = conn
        self.source = source
        self.rate_hz = float(rate_hz)
        self.dropout = float(np.clip(dropout, 0.0, 0.95))
        self.use_odometry = bool(use_odometry)
        #: 1-sigma attitude error published in the ODOMETRY pose covariance.  The
        #: receiver takes ``sqrt`` of the three orientation diagonals and clamps
        #: the result to ``[VISO_YAW_NSE, 1.5]``, so this only matters above that
        #: floor - understating it is what lets a badly-aligned heading in.
        self.ang_sigma_rad = float(ang_sigma_rad)
        self.name = name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._sent = 0
        self._dropped = 0
        self._invalid = 0
        self._last_sent_t = 0.0
        self._rate_window: List[float] = []
        self.rng = np.random.default_rng(4)
        self.running = False

    # ------------------------------------------------------------------ #
    @property
    def sent(self) -> int:
        return self._sent

    @property
    def measured_hz(self) -> float:
        """Actual publish rate over the last second."""
        with self._lock:
            now = time.time()
            self._rate_window = [t for t in self._rate_window if now - t <= 1.0]
            return float(len(self._rate_window))

    @property
    def age_s(self) -> float:
        """Seconds since the last sample was actually published."""
        return (time.time() - self._last_sent_t) if self._last_sent_t else float("inf")

    @property
    def healthy(self) -> bool:
        """Source is producing valid samples AND they are reaching the vehicle.

        Both halves matter.  A VIO that reports itself healthy while the link is
        dropping every message is not a usable nav source, and a manager that
        switched to it would leave the EKF with nothing.
        """
        return (bool(self.source.healthy) and self.measured_hz >= 5.0
                and self.age_s < 0.5)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"name": self.name, "running": self.running,
                    "sent": self._sent, "dropped": self._dropped,
                    "invalid": self._invalid, "measured_hz": round(self.measured_hz, 1),
                    "age_s": round(self.age_s, 3), "healthy": self.healthy,
                    "source_healthy": bool(self.source.healthy),
                    "target_hz": self.rate_hz, "dropout": self.dropout}

    # ------------------------------------------------------------------ #
    def start(self) -> "ExternalNavFeeder":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self.running = True

        def _loop() -> None:
            period = 1.0 / max(self.rate_hz, 1.0)
            next_t = time.time()
            while not self._stop.is_set():
                next_t += period
                delay = next_t - time.time()
                if delay > 0:
                    self._stop.wait(delay)
                if self._stop.is_set():
                    break
                self.publish_once()

        self._thread = threading.Thread(target=_loop, name=f"extnav-{self.name}",
                                        daemon=True)
        self._thread.start()
        log.info("%s feeder started at %.0f Hz", self.name, self.rate_hz)
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------ #
    def publish_once(self, t: Optional[float] = None) -> bool:
        """Publish one sample.  Returns whether it was sent.

        Public and synchronous so a deterministic test can drive the feeder by
        hand instead of racing its thread, which is how the source-manager tests
        below work.
        """
        now = t if t is not None else time.time()
        s = self.source.sample(now)
        with self._lock:
            if s is None or not s.valid:
                self._invalid += 1
                return False
            if self.dropout > 0 and self.rng.random() < self.dropout:
                self._dropped += 1
                return False
            # Timestamps go out in microseconds on the vehicle's boot clock.  The
            # connection fills that in itself when given None, which is the right
            # default: a feeder that guesses the clock sends samples the EKF
            # rejects as being from the future or from 1970.
            try:
                if self.use_odometry:
                    bx, by, bz = s.v_body
                    self.conn.send_odometry(
                        s.north_m, s.east_m, s.down_m, bx, by, bz,
                        q=s.quat, pos_sigma_m=s.pos_sigma_m,
                        vel_sigma_ms=s.vel_sigma_ms,
                        ang_sigma_rad=self.ang_sigma_rad,
                        quality=int(s.confidence))
                else:
                    self.conn.send_external_position(
                        s.north_m, s.east_m, s.down_m, s.yaw_deg,
                        sigma_m=s.pos_sigma_m)
            except Exception as exc:
                log.warning("%s publish failed: %r", self.name, exc)
                return False
            self._sent += 1
            self._last_sent_t = time.time()
            self._rate_window.append(self._last_sent_t)
            if len(self._rate_window) > 400:
                del self._rate_window[:200]
        return True

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "ExternalNavFeeder":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# Source-set policy
# --------------------------------------------------------------------------- #
@dataclass
class SourceSetPolicy:
    """Which source set to use for which navigation state.

    Kept as data rather than code so a different airframe - one with a compass,
    or a rangefinder, or a UWB beacon network - is a configuration change rather
    than an edit to the switching logic.
    """

    gnss_ok: int = SOURCE_SET.GNSS
    gnss_degraded: int = SOURCE_SET.GNSS
    external_nav: int = SOURCE_SET.EXTERNAL_NAV
    flow_only: int = SOURCE_SET.FLOW
    lost: int = SOURCE_SET.FLOW

    #: Refuse to switch *away* from a working source on a single bad assessment.
    switches_to_degrade: int = 1
    #: Require this many consecutive good assessments before switching back to
    #: GNSS.  A fix that returns for one epoch in a canyon and then disappears
    #: again is worse than no fix at all, because each switch disturbs the
    #: filter.
    switches_to_recover: int = 4
    #: Minimum seconds between any two switches, in either direction.
    min_switch_interval_s: float = 2.0


class EkfSourceManager:
    """Decides and executes EKF source-set switches.

    The policy is deliberately asymmetric.  Degrading - leaving GNSS for external
    nav - happens on a single assessment, because the cost of waiting is that the
    switch happens *during* the transition, when neither source is good.
    Recovering - going back to GNSS - requires several consecutive good
    assessments and a minimum interval, because a fix that flickers in and out is
    common in a canyon and each switch disturbs the filter.

    Nothing is switched to unless it is healthy.  That single rule is what stops
    the manager from making things worse, and it is the reason the feeder's
    ``healthy`` property checks the measured publish rate rather than just
    asking the source whether it thinks it is working.
    """

    def __init__(self, conn: Any, monitor: NavQualityMonitor,
                 extnav: Optional[ExternalNavFeeder] = None,
                 policy: Optional[SourceSetPolicy] = None,
                 flow_healthy_fn: Optional[Callable[[], bool]] = None,
                 dry_run: bool = False) -> None:
        self.conn = conn
        self.monitor = monitor
        self.extnav = extnav
        self.policy = policy or SourceSetPolicy()
        self.flow_healthy_fn = flow_healthy_fn or (lambda: False)
        self.dry_run = bool(dry_run)
        self._consecutive_good = 0
        self._consecutive_bad = 0
        self._last_switch_t = 0.0
        self._current_set: Optional[int] = None
        self.decisions: List[Dict[str, Any]] = []
        self.refusals: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    @property
    def current_set(self) -> int:
        if self._current_set is None:
            self._current_set = int(getattr(self.conn.telemetry, "ekf_source_set", 1)
                                    or 1)
        return self._current_set

    def _destination_healthy(self, target_set: int,
                             assessment: NavAssessment) -> Tuple[bool, str]:
        """Is the set we want to switch to actually usable right now?"""
        if target_set == self.policy.external_nav:
            if self.extnav is None:
                return False, "no external-nav feeder configured"
            if not self.extnav.healthy:
                return False, (f"external nav unhealthy (hz="
                               f"{self.extnav.measured_hz:.0f}, age="
                               f"{self.extnav.age_s:.2f}s, source="
                               f"{self.extnav.source.healthy})")
            return True, f"external nav at {self.extnav.measured_hz:.0f} Hz"
        if target_set == self.policy.flow_only:
            ok = bool(self.flow_healthy_fn()) or assessment.flow_healthy
            return ok, ("optical flow healthy" if ok else "optical flow unhealthy")
        if target_set == self.policy.gnss_ok:
            return (assessment.gps_fix >= self.monitor.min_fix_type
                    and assessment.n_satellites >= self.monitor.min_sats_degraded), \
                f"gnss fix={assessment.gps_fix} sats={assessment.n_satellites}"
        return True, "no aiding required"

    # ------------------------------------------------------------------ #
    def update(self, assessment: NavAssessment, t: Optional[float] = None) -> Dict[str, Any]:
        """Feed one assessment in; switch source set if the policy says so."""
        now = t if t is not None else time.time()
        desired = self._desired_set(assessment)
        current = self.current_set
        decision: Dict[str, Any] = {
            "t": round(now, 3), "state": assessment.state.value,
            "current_set": current, "desired_set": desired, "switched": False,
            "reason": assessment.reason,
        }

        if assessment.state == NavState.GNSS_OK:
            self._consecutive_good += 1
            self._consecutive_bad = 0
        else:
            self._consecutive_bad += 1
            self._consecutive_good = 0

        if desired == current:
            self.decisions.append(decision)
            return decision

        recovering = _set_severity(desired) < _set_severity(current)
        if recovering and self._consecutive_good < self.policy.switches_to_recover:
            decision["reason"] = (f"holding on set {current}: recovery needs "
                                  f"{self.policy.switches_to_recover} consecutive "
                                  f"good assessments, have {self._consecutive_good}")
            self.decisions.append(decision)
            return decision
        if not recovering and self._consecutive_bad < self.policy.switches_to_degrade:
            decision["reason"] = (f"holding on set {current}: degradation needs "
                                  f"{self.policy.switches_to_degrade} assessment(s), "
                                  f"have {self._consecutive_bad}")
            self.decisions.append(decision)
            return decision

        since = now - self._last_switch_t if self._last_switch_t else float("inf")
        if since < self.policy.min_switch_interval_s:
            decision["reason"] = (f"rate-limited: last switch {since:.1f}s ago, "
                                  f"minimum {self.policy.min_switch_interval_s}s")
            self.refusals.append(dict(decision))
            self.decisions.append(decision)
            return decision

        ok, why = self._destination_healthy(desired, assessment)
        if not ok:
            decision["reason"] = f"refused switch to set {desired}: {why}"
            self.refusals.append(dict(decision))
            self.decisions.append(decision)
            log.warning("source-set switch refused: %s", why)
            return decision

        if not self.dry_run:
            try:
                self.conn.set_ekf_source_set(desired)
                self._current_set = desired
                decision["switched"] = True
                self._last_switch_t = now
                decision["reason"] = f"switched to set {desired}: {why}"
                log.info("EKF source set -> %d (%s): %s", desired,
                         SOURCE_SET.name(desired), why)
            except Exception as exc:
                decision["reason"] = f"switch to set {desired} failed: {exc!r}"
                self.refusals.append(dict(decision))
                log.error("source-set switch failed: %r", exc)
        else:
            decision["switched"] = True
            decision["reason"] = f"(dry run) would switch to set {desired}: {why}"
            self._current_set = desired
            self._last_switch_t = now

        self.decisions.append(decision)
        if len(self.decisions) > 2000:
            del self.decisions[:1000]
        return decision

    def _desired_set(self, assessment: NavAssessment) -> int:
        p = self.policy
        return {NavState.GNSS_OK: p.gnss_ok,
                NavState.GNSS_DEGRADED: p.gnss_degraded,
                NavState.EXTERNAL_NAV: p.external_nav,
                NavState.FLOW_ONLY: p.flow_only,
                NavState.LOST: p.lost}[assessment.state]

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        switches = [d for d in self.decisions if d.get("switched")]
        return {"current_set": self.current_set,
                "current_set_name": SOURCE_SET.name(self.current_set),
                "n_decisions": len(self.decisions),
                "n_switches": len(switches),
                "n_refusals": len(self.refusals),
                "switches": switches[-12:],
                "refusals": [r["reason"] for r in self.refusals[-6:]],
                "nav_state": self.monitor.state.value,
                "transitions": self.monitor.transitions[-8:]}


def _set_severity(source_set: int) -> int:
    """Ordering over source sets, mirroring :func:`_severity` over nav states."""
    return {SOURCE_SET.GNSS: 0, SOURCE_SET.EXTERNAL_NAV: 1,
            SOURCE_SET.FLOW: 2}.get(int(source_set), 3)
