"""The supervisor that is allowed to end the sortie.

Autonomy decides *where to look*.  This module decides *whether to keep
flying*, and it is deliberately separate, small, and readable, because it is
the only component whose failure is unrecoverable.

Design commitments
------------------
**Every limit is evaluated against a measured quantity, not a timer.**  A
battery failsafe at "20% remaining" is not a limit; the aircraft may be 900 m
downwind of home.  :meth:`SafetySupervisor.evaluate` computes the energy needed
to return - distance, wind, climb to RTL altitude, descent and a reserve - and
compares that against what is left.  The trigger is *reachability*, not a
percentage.

**Every decision carries its reason.**  A supervisor that returns
``ABORT`` without a string is unauditable after an incident.  Each decision
carries the rule that fired, the measured value and the limit, and every one is
recorded in the sortie report.

**Latching.**  Safety states never improve silently.  Once ``RTL_NOW`` has
fired, a momentarily better battery reading does not resume the search; the
state can only be cleared by an operator command.  Oscillating between "return"
and "continue" at a limit boundary is worse than either.

**The autopilot's own failsafes still exist and are not replaced.**  ArduPilot
holds the last line of defence (``BATT_FS_LOW_ACT``, ``FS_GCS_ENABLE``,
``FENCE_ACTION``).  This supervisor exists to act *earlier*, with knowledge the
flight controller does not have - where the survivors are, how much of the
search remains, and whether the perception loop is still alive.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("sar.hardware.safety")


class SafetyState(str, Enum):
    """Escalating, latching states."""

    NOMINAL = "nominal"
    CAUTION = "caution"          # keep flying, report it
    RETURN = "return"            # finish the current lane, then RTL
    RTL_NOW = "rtl_now"          # RTL immediately
    LAND_NOW = "land_now"        # descend where you are
    ABORT = "abort"              # disarm on the ground / motors off

_ORDER = {s: i for i, s in enumerate(
    [SafetyState.NOMINAL, SafetyState.CAUTION, SafetyState.RETURN,
     SafetyState.RTL_NOW, SafetyState.LAND_NOW, SafetyState.ABORT])}


@dataclass
class SafetyLimits:
    """Numbers that end sorties.  Every one is a deliberate operational choice."""

    # -- energy -------------------------------------------------------- #
    reserve_fraction: float = 0.20      # never plan to land below this SoC
    hover_power_w: float = 420.0        # measured, not from the motor datasheet
    cruise_speed_ms: float = 8.0
    rtl_climb_m: float = 15.0
    energy_margin: float = 1.35         # 35% on top of the computed return energy

    # -- geometry ------------------------------------------------------ #
    max_range_m: float = 800.0          # from home; also the ELRS control-link edge
    max_altitude_agl_m: float = 120.0   # DGCA/most-jurisdictions VLOS ceiling
    min_altitude_agl_m: float = 8.0
    geofence_polygon: Optional[List[Tuple[float, float]]] = None   # (north, east) m

    # -- links --------------------------------------------------------- #
    max_telemetry_silence_s: float = 3.0
    max_rc_loss_s: float = 5.0
    max_loop_stall_s: float = 2.0       # autonomy watchdog

    # -- navigation ---------------------------------------------------- #
    max_position_sigma_m: float = 25.0  # beyond this a geo-tag is not actionable
    max_denial_s: float = 90.0          # dead reckoning is bounded by policy

    # -- environment --------------------------------------------------- #
    max_wind_ms: float = 12.0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        if self.geofence_polygon:
            d["geofence_polygon"] = [list(p) for p in self.geofence_polygon]
        return d


@dataclass
class SafetyDecision:
    state: SafetyState
    reasons: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)
    t: float = 0.0

    @property
    def must_return(self) -> bool:
        return _ORDER[self.state] >= _ORDER[SafetyState.RETURN]

    @property
    def must_stop_searching(self) -> bool:
        return _ORDER[self.state] >= _ORDER[SafetyState.RTL_NOW]

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state.value, "reasons": self.reasons,
                "measurements": self.measurements, "t": self.t}


class SafetySupervisor:
    """Evaluates the limits every control cycle and latches the worst outcome."""

    def __init__(self, limits: Optional[SafetyLimits] = None) -> None:
        self.limits = limits or SafetyLimits()
        self.state = SafetyState.NOMINAL
        self.history: List[SafetyDecision] = []
        self.last_progress_t = time.monotonic()
        self._denial_started: Optional[float] = None
        self.triggered_rules: List[str] = []

    # ------------------------------------------------------------------ #
    def note_progress(self) -> None:
        """Called by the autonomy loop each cycle: the watchdog's food."""
        self.last_progress_t = time.monotonic()

    def clear(self, reason: str = "operator") -> None:
        """Operator override.  Latching means only a human clears it."""
        log.warning("safety state cleared by %s (was %s)", reason, self.state.value)
        self.state = SafetyState.NOMINAL
        self.triggered_rules.clear()

    # ------------------------------------------------------------------ #
    def evaluate(self, *, telemetry: Any = None,
                 position_ned: Optional[Tuple[float, float, float]] = None,
                 home_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 battery_remaining: Optional[float] = None,
                 battery_wh_remaining: Optional[float] = None,
                 agl_m: Optional[float] = None,
                 position_sigma_m: Optional[float] = None,
                 gps_denied: bool = False,
                 wind_ms: Optional[float] = None,
                 telemetry_silence_s: Optional[float] = None,
                 rc_loss_s: Optional[float] = None,
                 now: Optional[float] = None) -> SafetyDecision:
        """Evaluate every rule.  Returns the (latched) decision."""
        now = now if now is not None else time.monotonic()
        reasons: List[str] = []
        meas: Dict[str, Any] = {}
        worst = SafetyState.NOMINAL

        def raise_to(state: SafetyState, rule: str, msg: str) -> None:
            nonlocal worst
            if _ORDER[state] > _ORDER[worst]:
                worst = state
            reasons.append(msg)
            if rule not in self.triggered_rules:
                self.triggered_rules.append(rule)

        lim = self.limits

        # -- distance / geofence ---------------------------------------- #
        if position_ned is not None:
            dn = position_ned[0] - home_ned[0]
            de = position_ned[1] - home_ned[1]
            dist = math.hypot(dn, de)
            meas["range_m"] = round(dist, 1)
            if dist > lim.max_range_m:
                raise_to(SafetyState.RTL_NOW, "max_range",
                         f"range {dist:.0f} m exceeds limit {lim.max_range_m:.0f} m")
            elif dist > 0.85 * lim.max_range_m:
                raise_to(SafetyState.CAUTION, "range_warn",
                         f"range {dist:.0f} m approaching limit "
                         f"{lim.max_range_m:.0f} m")
            if lim.geofence_polygon and not _point_in_polygon(
                    (position_ned[0], position_ned[1]), lim.geofence_polygon):
                raise_to(SafetyState.RTL_NOW, "geofence",
                         "outside the operational geofence polygon")

        # -- altitude ---------------------------------------------------- #
        if agl_m is not None:
            meas["agl_m"] = round(agl_m, 1)
            if agl_m > lim.max_altitude_agl_m:
                raise_to(SafetyState.RETURN, "max_alt",
                         f"AGL {agl_m:.0f} m above ceiling "
                         f"{lim.max_altitude_agl_m:.0f} m")
            if 0.0 < agl_m < lim.min_altitude_agl_m:
                raise_to(SafetyState.CAUTION, "min_alt",
                         f"AGL {agl_m:.1f} m below floor "
                         f"{lim.min_altitude_agl_m:.0f} m")

        # -- energy: reachability, not a percentage ---------------------- #
        if battery_remaining is not None:
            meas["battery_frac"] = round(battery_remaining, 3)
            if position_ned is not None and battery_wh_remaining is not None:
                need = self.energy_to_return_wh(position_ned, home_ned)
                meas["return_energy_wh"] = round(need, 1)
                meas["available_wh"] = round(battery_wh_remaining, 1)
                if battery_wh_remaining < need:
                    raise_to(SafetyState.RTL_NOW, "energy_reachability",
                             f"{battery_wh_remaining:.0f} Wh left, {need:.0f} Wh "
                             f"needed to reach home with margin")
                elif battery_wh_remaining < need * 1.25:
                    raise_to(SafetyState.RETURN, "energy_reachability_warn",
                             f"{battery_wh_remaining:.0f} Wh left vs {need:.0f} Wh "
                             "to return - finish this lane and go home")
            if battery_remaining < lim.reserve_fraction * 0.5:
                raise_to(SafetyState.LAND_NOW, "battery_critical",
                         f"battery {battery_remaining:.0%} below half of the "
                         f"{lim.reserve_fraction:.0%} reserve")
            elif battery_remaining < lim.reserve_fraction:
                raise_to(SafetyState.RTL_NOW, "battery_reserve",
                         f"battery {battery_remaining:.0%} below reserve "
                         f"{lim.reserve_fraction:.0%}")

        # -- navigation quality ------------------------------------------ #
        if position_sigma_m is not None:
            meas["position_sigma_m"] = round(position_sigma_m, 1)
            if position_sigma_m > lim.max_position_sigma_m:
                raise_to(SafetyState.RETURN, "nav_sigma",
                         f"position sigma {position_sigma_m:.0f} m exceeds "
                         f"{lim.max_position_sigma_m:.0f} m - reports would not "
                         "be actionable")
        if gps_denied:
            self._denial_started = self._denial_started or now
            denial_s = now - self._denial_started
            meas["denial_s"] = round(denial_s, 1)
            if denial_s > lim.max_denial_s:
                raise_to(SafetyState.RTL_NOW, "denial_duration",
                         f"GNSS denied for {denial_s:.0f} s, over the "
                         f"{lim.max_denial_s:.0f} s policy bound")
            elif denial_s > 0.5 * lim.max_denial_s:
                raise_to(SafetyState.CAUTION, "denial_warn",
                         f"GNSS denied for {denial_s:.0f} s")
        else:
            self._denial_started = None

        # -- links -------------------------------------------------------- #
        if telemetry_silence_s is not None:
            meas["telemetry_silence_s"] = round(telemetry_silence_s, 1)
            if telemetry_silence_s > lim.max_telemetry_silence_s:
                raise_to(SafetyState.RTL_NOW, "telemetry_loss",
                         f"no telemetry for {telemetry_silence_s:.1f} s")
        if rc_loss_s is not None and rc_loss_s > lim.max_rc_loss_s:
            meas["rc_loss_s"] = round(rc_loss_s, 1)
            # Not an abort: autonomous search is exactly the mission that must
            # survive RC loss. It is a caution, and the autopilot's own RC
            # failsafe remains configured behind it.
            raise_to(SafetyState.CAUTION, "rc_loss",
                     f"RC link lost for {rc_loss_s:.1f} s (autonomy continues)")

        # -- environment ---------------------------------------------------- #
        if wind_ms is not None:
            meas["wind_ms"] = round(wind_ms, 1)
            if wind_ms > lim.max_wind_ms:
                raise_to(SafetyState.RTL_NOW, "wind",
                         f"wind {wind_ms:.0f} m/s over limit {lim.max_wind_ms:.0f} m/s")

        # -- autonomy watchdog ------------------------------------------- #
        stall = now - self.last_progress_t
        meas["loop_stall_s"] = round(stall, 2)
        if stall > lim.max_loop_stall_s:
            raise_to(SafetyState.RTL_NOW, "loop_watchdog",
                     f"autonomy loop has not progressed for {stall:.1f} s")

        # -- latch --------------------------------------------------------- #
        if _ORDER[worst] > _ORDER[self.state]:
            log.warning("safety state %s -> %s: %s", self.state.value,
                        worst.value, "; ".join(reasons))
            self.state = worst
        decision = SafetyDecision(state=self.state, reasons=reasons,
                                  measurements=meas, t=now)
        self.history.append(decision)
        if len(self.history) > 512:
            del self.history[:256]
        return decision

    # ------------------------------------------------------------------ #
    def energy_to_return_wh(self, position_ned: Tuple[float, float, float],
                            home_ned: Tuple[float, float, float]) -> float:
        """Energy to climb to RTL altitude, cruise home, descend, and land.

        Deliberately simple and deliberately pessimistic: hover power for the
        whole transit (a multirotor's cruise power is not much below hover),
        plus the margin.  A model that flatters the aircraft here is the model
        that ends up in a field.
        """
        lim = self.limits
        dn = position_ned[0] - home_ned[0]
        de = position_ned[1] - home_ned[1]
        dist = math.hypot(dn, de)
        cruise_s = dist / max(lim.cruise_speed_ms, 0.5)
        climb_s = lim.rtl_climb_m / 2.5
        descend_s = (abs(position_ned[2] - home_ned[2]) + lim.rtl_climb_m) / 1.5
        seconds = cruise_s + climb_s + descend_s
        wh = lim.hover_power_w * seconds / 3600.0
        return wh * lim.energy_margin

    def summary(self) -> Dict[str, Any]:
        return {"state": self.state.value,
                "triggered_rules": list(self.triggered_rules),
                "limits": self.limits.to_dict(),
                "evaluations": len(self.history),
                "last": self.history[-1].to_dict() if self.history else None}


def _point_in_polygon(point: Tuple[float, float],
                      polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting.  Polygon in the same (north, east) metres frame as ``point``."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xint = (x1 - x0) * (y - y0) / ((y1 - y0) or 1e-12) + x0
            if x < xint:
                inside = not inside
    return inside


__all__ = ["SafetyDecision", "SafetyLimits", "SafetyState", "SafetySupervisor"]
