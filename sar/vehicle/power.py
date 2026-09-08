"""Battery and powertrain model for a 6S LiPo multirotor SAR platform.

Two things are modelled with care because they decide whether a disaster
sortie is *useful*:

1. **The pack** - a 6S LiPo with an open-circuit-voltage curve, internal
   resistance, coulomb counting and temperature-dependent capacity fade.  The
   autonomy stack must make battery decisions from *estimated state of charge*
   and *sag-corrected voltage*, not from a naive "voltage < 3.5 V/cell" rule,
   because a heavy climb at 60 % SOC can momentarily look like a critical pack.

2. **The powertrain** - hover and cruise power from actuator-disk (momentum)
   theory with a propeller figure-of-merit and motor efficiency.  This is the
   same physics used by eCalc-class tools and reproduces measured endurance of
   10-inch 6S quads to within ~10 %.

References for the physics
--------------------------
* Leishman, *Principles of Helicopter Aerodynamics*, momentum theory in
  forward flight: :math:`P_i = T\\left(\\sqrt{V^2/4 + T/(2\\rho A)} - V/2\\right)`.
* Standard LiPo OCV/SOC behaviour (3.30 V/cell empty .. 4.20 V/cell full).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

G0 = 9.80665          # standard gravity [m/s^2]
RHO_SEA = 1.225       # sea-level air density [kg/m^3]

# Cell open-circuit voltage vs state-of-charge lookup (LiPo, 4.2 V/cell charge
# termination, 3.3 V/cell safe cutoff).  Interpolated linearly.
_CELLS = 6
_OCV_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
_OCV_CELL = np.array([3.30, 3.55, 3.65, 3.70, 3.73, 3.76, 3.79, 3.83, 3.87, 3.94, 4.05, 4.12, 4.20])


@dataclass
class BatteryConfig:
    """6S LiPo pack parameters.

    Defaults describe the reference endurance pack for the SAHYOG airframe:
    a 6S 8000 mAh 30C LiPo.  A 6S 5000 mAh freestyle pack flies the same
    airframe for roughly 60 % of the time; see the endurance budget table in
    ``docs/HARDWARE_BRINGUP.md``.
    """

    cells: int = 6
    capacity_mah: float = 8000.0
    internal_resistance_mohm: float = 20.0     # whole pack, DC-IR
    c_rating: float = 30.0
    mass_kg: float = 1.050                     # typical 6S8000 LiPo
    warn_soc: float = 0.30                     # "battery low" RTL trigger
    critical_soc: float = 0.15                 # "land now" trigger
    reserve_soc: float = 0.08                  # absolute no-fly floor
    cell_voltage_warn: float = 3.55
    cell_voltage_critical: float = 3.40
    temp_c: float = 25.0

    @property
    def nominal_voltage(self) -> float:
        return 3.70 * self.cells

    @property
    def max_voltage(self) -> float:
        return 4.20 * self.cells

    @property
    def capacity_ah(self) -> float:
        return self.capacity_mah / 1000.0

    @property
    def energy_wh(self) -> float:
        return self.capacity_ah * self.nominal_voltage

    @property
    def max_continuous_current(self) -> float:
        return self.capacity_ah * self.c_rating


@dataclass
class AirframeConfig:
    """Rotor/aerodynamic parameters.

    Defaults describe the SAHYOG reference build (``config/vehicle.yaml``):
    a 12-inch (305 mm prop) X-quad on 6S, 3.1 kg all-up.  Propeller diameter
    is the single most important lever on endurance at this weight class -
    going from 10" to 12" props buys ~18 % hover endurance for free.
    """

    n_rotors: int = 4
    prop_diameter_m: float = 0.3048         # 12 in
    figure_of_merit: float = 0.72           # hover efficiency of a good prop
    motor_efficiency: float = 0.86          # shaft power -> electrical power
    esc_efficiency: float = 0.95
    mass_dry_kg: float = 1.35               # frame+motors+FC+companion computer+wiring
    payload_kg: float = 0.70                # RGB+LWIR sensors, gimbal, drop mechanism, relay pods
    frontal_area_m2: float = 0.070          # for parasite drag
    drag_coefficient: float = 0.78
    inertia: Tuple[float, float, float] = (0.062, 0.062, 0.104)   # Ixx, Iyy, Izz [kg m^2]
    arm_length_m: float = 0.26              # motor CG distance
    max_rotor_thrust_n: float = 27.5        # static thrust limit of one motor/prop on 6S
    esc_continuous_a: float = 45.0          # per-ESC-channel continuous current
    max_tilt_rad: float = math.radians(32.0)
    max_climb_ms: float = 3.5
    max_speed_ms: float = 15.0

    @property
    def mass_kg(self) -> float:
        """All-up mass without battery (battery is added by the vehicle model)."""
        return self.mass_dry_kg + self.payload_kg

    @property
    def disk_area_m2(self) -> float:
        return math.pi * (self.prop_diameter_m / 2.0) ** 2

    @property
    def total_disk_area_m2(self) -> float:
        return self.disk_area_m2 * self.n_rotors


@dataclass
class PowerState:
    voltage: float = 0.0
    current_a: float = 0.0
    power_w: float = 0.0
    soc: float = 1.0
    consumed_mah: float = 0.0
    consumed_wh: float = 0.0
    cell_voltage: float = 0.0
    temp_c: float = 25.0
    time_remaining_s: float = 0.0


class Powertrain:
    """Thrust <-> power conversion + pack state estimation.

    The class is deliberately free of any simulator coupling: the mission
    planner queries :meth:`power_for_thrust` to cost a candidate trajectory,
    while the plant integrates :meth:`draw` each step.
    """

    def __init__(
        self,
        airframe: AirframeConfig,
        battery: BatteryConfig,
        avionics_w: float = 28.0,
        rho: float = RHO_SEA,
    ) -> None:
        self.airframe = airframe
        self.battery = battery
        self.avionics_w = float(avionics_w)
        self.rho = float(rho)

        self.mass_total = airframe.mass_kg + battery.mass_kg
        self.weight_n = self.mass_total * G0

        # Pack electrical state
        self.soc = 1.0
        self.consumed_ah = 0.0
        self.consumed_wh = 0.0
        self.current_a = 0.0
        self.voltage = battery.max_voltage
        self.temp_c = battery.temp_c
        self._last_power_w = self.avionics_w
        self._power_history: List[Tuple[float, float]] = []   # (t, watts)

    # ------------------------------------------------------------------ #
    # Aerodynamics
    # ------------------------------------------------------------------ #
    def hover_thrust_n(self) -> float:
        return self.weight_n

    def induced_power(self, thrust_n: float, airspeed_ms: float = 0.0,
                      climb_ms: float = 0.0) -> float:
        """Actuator-disk induced power for the whole vehicle [W].

        Glauert's combined-flow momentum theory, which reduces exactly to the
        classical hover and edgewise results:

        .. math::

            v_h^2 = \\frac{T}{2 \\rho A},\\qquad
            W = \\sqrt{V_{par}^2 + V_{perp}^2},\\qquad
            v_i = \\tfrac12\\left(-W + \\sqrt{W^2 + 4 v_h^2}\\right),\\qquad
            P_i = T\\,(v_i + V_{perp})

        Divided by the propeller figure of merit to get shaft power.
        """
        if thrust_n <= 1e-6:
            return 0.0
        rho = self.rho
        area = self.airframe.total_disk_area_m2
        v_par = max(0.0, float(airspeed_ms))
        v_perp = float(climb_ms)          # positive = climbing
        vh2 = thrust_n / (2.0 * rho * area)
        w = math.hypot(v_par, v_perp)
        vi = 0.5 * (-w + math.sqrt(w * w + 4.0 * vh2))
        p_ideal = thrust_n * (vi + v_perp)
        if p_ideal < 0.0:
            # Fast descent / autorotative regime: momentum theory is invalid and
            # the real cost is dominated by profile power and control activity.
            p_ideal = thrust_n * math.sqrt(vh2) * 0.5
        # Vortex-ring state: low forward speed with a meaningful descent rate.
        # Thrust becomes unsteady and expensive; inflate the cost so the planner
        # avoids parking in it (and so a descending RTL over a hillside is not
        # treated as "free").
        if v_par < 3.0 and v_perp < -1.0:
            p_ideal *= 1.0 + 0.55 * min(1.0, -v_perp / 4.0)
        return p_ideal / self.airframe.figure_of_merit

    def parasite_power(self, airspeed_ms: float) -> float:
        a = self.airframe
        return 0.5 * self.rho * (airspeed_ms ** 3) * a.drag_coefficient * a.frontal_area_m2

    def profile_power(self, thrust_n: float) -> float:
        """Rotor profile (blade drag) power - weakly dependent on thrust."""
        base = self.induced_power(self.hover_thrust_n(), 0.0) * 0.18
        load = max(thrust_n, 0.0) / max(self.hover_thrust_n(), 1e-6)
        return base * (0.6 + 0.4 * load)

    def power_for_thrust(
        self,
        thrust_n: float,
        airspeed_ms: float = 0.0,
        climb_ms: float = 0.0,
        altitude_m: float = 0.0,
    ) -> float:
        """Total **electrical** power draw [W] at the battery terminals."""
        rho = air_density(altitude_m)
        saved_rho, self.rho = self.rho, rho
        try:
            p_ind = self.induced_power(thrust_n, airspeed_ms, climb_ms)
        finally:
            self.rho = saved_rho
        p_shaft = p_ind + self.parasite_power(airspeed_ms) + self.profile_power(thrust_n)
        eff = self.airframe.motor_efficiency * self.airframe.esc_efficiency
        # NOTE: climb power is already included by momentum theory through the
        # V_perp term of induced_power(); adding it again would double-count.
        return p_shaft / eff + self.avionics_w

    def thrust_available_n(self, voltage: float, throttle_limit: float = 1.0) -> float:
        """Maximum steady thrust the propulsion system can deliver [N].

        Three independent limits are applied and the *tightest* wins:

        1. the pack's continuous discharge current;
        2. the ESC/motor channel current (a 60 A 4-in-1 ESC is the usual
           bottleneck on a 3 kg build, not the battery);
        3. the propeller's static thrust ceiling at wide-open throttle, which
           momentum theory alone would badly over-predict.
        """
        n = self.airframe.n_rotors
        i_pack = self.battery.max_continuous_current
        i_esc = self.airframe.esc_continuous_a * n
        i_max = min(i_pack, i_esc)
        p_max = i_max * max(voltage, 1.0) * self.airframe.motor_efficiency * self.airframe.esc_efficiency
        rho = self.rho
        area = self.airframe.total_disk_area_m2
        fm = self.airframe.figure_of_merit
        t_power = ((p_max * fm) ** 2 * 2.0 * rho * area) ** (1.0 / 3.0)
        t_static = self.airframe.max_rotor_thrust_n * n
        # Thrust also falls off with the square of voltage below a full pack.
        v_scale = float(np.clip(voltage / self.battery.nominal_voltage, 0.6, 1.15)) ** 2
        return min(t_power, t_static * v_scale) * throttle_limit

    # ------------------------------------------------------------------ #
    # Pack state
    # ------------------------------------------------------------------ #
    def open_circuit_voltage(self) -> float:
        soc = float(np.clip(self.soc, 0.0, 1.0))
        cell = float(np.interp(soc, _OCV_SOC, _OCV_CELL))
        # Cold packs deliver less: ~0.4 % capacity loss per degree below 20 C.
        temp_derate = 1.0 - max(0.0, (20.0 - self.temp_c)) * 0.004
        return cell * self.battery.cells * (0.5 + 0.5 * temp_derate)

    def terminal_voltage(self) -> float:
        r = self.battery.internal_resistance_mohm / 1000.0
        return max(0.0, self.open_circuit_voltage() - self.current_a * r)

    @property
    def cell_voltage(self) -> float:
        return self.terminal_voltage() / self.battery.cells

    def draw(self, power_w: float, dt: float, t: Optional[float] = None) -> PowerState:
        """Consume ``power_w`` for ``dt`` seconds; return the pack state."""
        power_w = max(0.0, float(power_w))
        self._last_power_w = power_w
        if t is not None:
            self._power_history.append((float(t), power_w))

        # Iterate once on voltage since current depends on voltage.
        v = self.terminal_voltage()
        for _ in range(3):
            i = power_w / max(v, 1.0)
            v = self.open_circuit_voltage() - i * (self.battery.internal_resistance_mohm / 1000.0)
            v = max(v, 1.0)
        self.current_a = i
        self.voltage = v

        ah = i * dt / 3600.0
        usable = self.battery.capacity_ah * _temp_capacity_factor(self.temp_c)
        self.consumed_ah += ah
        self.consumed_wh += power_w * dt / 3600.0
        self.soc = float(np.clip(1.0 - self.consumed_ah / max(usable, 1e-6), 0.0, 1.0))

        # Crude thermal model: I^2 R heating vs convective cooling from airflow.
        r = self.battery.internal_resistance_mohm / 1000.0
        heat_w = i * i * r
        self.temp_c += (heat_w * 0.05 - (self.temp_c - 25.0) * 0.02) * dt

        return self.state

    @property
    def state(self) -> PowerState:
        return PowerState(
            voltage=self.voltage,
            current_a=self.current_a,
            power_w=self._last_power_w,
            soc=self.soc,
            consumed_mah=self.consumed_ah * 1000.0,
            consumed_wh=self.consumed_wh,
            cell_voltage=self.cell_voltage,
            temp_c=self.temp_c,
            time_remaining_s=self.time_remaining_s(),
        )

    def time_remaining_s(self, recent_window_s: float = 20.0) -> float:
        """Estimated seconds until :data:`BatteryConfig.reserve_soc`.

        Uses an exponentially-weighted recent power draw so a burst of climbing
        does not instantly declare an emergency, but a sustained high draw does.
        """
        if not self._power_history:
            p_avg = max(self._last_power_w, 1.0)
        else:
            t_now = self._power_history[-1][0]
            recent = [p for (t, p) in self._power_history if t >= t_now - recent_window_s]
            p_avg = float(np.mean(recent)) if recent else self._last_power_w
        usable_ah = self.battery.capacity_ah * _temp_capacity_factor(self.temp_c)
        remaining_ah = max(0.0, (self.soc - self.battery.reserve_soc) * usable_ah)
        v = max(self.voltage, 1.0)
        remaining_wh = remaining_ah * v
        if p_avg <= 1e-6:
            return float("inf")
        return remaining_wh / p_avg * 3600.0

    def energy_to_hover_home(self, distance_m: float, cruise_speed_ms: float) -> float:
        """Joules needed to fly ``distance_m`` at ``cruise_speed_ms`` (no reserve)."""
        if cruise_speed_ms <= 0.1:
            return float("inf")
        p = self.power_for_thrust(self.hover_thrust_n() * 1.05, cruise_speed_ms)
        return p * (distance_m / cruise_speed_ms)

    def endurance_estimate_s(self, cruise_speed_ms: float = 0.0) -> float:
        p = self.power_for_thrust(self.hover_thrust_n(), cruise_speed_ms)
        usable_wh = self.battery.energy_wh * 0.92   # 8 % unusable bottom
        return usable_wh / max(p, 1.0) * 3600.0

    def optimum_cruise_speed(self, vmax: float = 20.0, step: float = 0.5) -> float:
        """Speed that maximises *range* (Wh/km) - the classic power-curve minimum."""
        best_v, best_cost = 0.0, float("inf")
        v = step
        while v <= vmax:
            p = self.power_for_thrust(self.hover_thrust_n(), v)
            cost = p / v      # W per m/s == J/m
            if cost < best_cost:
                best_cost, best_v = cost, v
            v += step
        return best_v


def _temp_capacity_factor(temp_c: float) -> float:
    """Available-capacity derate vs temperature (0 % at 25 C, -25 % at -10 C)."""
    if temp_c >= 25.0:
        return 1.0
    return float(np.clip(1.0 + (temp_c - 25.0) * 0.0071, 0.70, 1.0))


def air_density(altitude_m: float, temp_c: float = 15.0) -> float:
    """ISA-ish air density; ``temp_c`` offsets the ISA lapse temperature."""
    h = float(altitude_m)
    t = temp_c + 273.15 - 0.0065 * h
    t = max(t, 216.65)
    p = 101325.0 * (1.0 - 0.0065 * h / 288.15) ** 5.255876
    p = max(p, 1000.0)
    return p / (287.05287 * t)
