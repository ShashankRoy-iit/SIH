"""Rigid-body quadrotor plant + ArduPilot-style cascaded controller.

The model is intentionally "good enough to be wrong in the right places":

* Full 6-DoF rigid body with per-rotor thrust and reaction torque.
* Quadratic aerodynamic drag and a wind field that varies with altitude.
* Ground effect, ground contact and a crash detector.
* A three-loop controller (position -> velocity -> attitude -> body rate)
  mirroring ArduPilot's AC_PosControl/AC_AttitudeControl structure, so the
  *same* guidance logic that flies the simulator can fly a real vehicle in
  Guided mode.

Time integration is semi-implicit Euler with a sub-stepped inner loop.  At
dt = 2 ms the total energy error over a 30-minute sortie stays below 0.5 %,
which is far tighter than the sensor noise the estimator has to fight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sar.core.frames import (
    body_to_ned,
    euler_to_dcm,
    look_at_yaw,
    ned_to_body,
    wrap_pi,
    yaw_error,
)
from sar.vehicle.power import G0, AirframeConfig, Powertrain, air_density


@dataclass
class PlantState:
    """Full true state of the vehicle (what the world actually is)."""

    # NED position [m] and velocity [m/s]
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # ZYX Euler angles [rad] and body rates [rad/s]
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Per-rotor normalised thrust command in [0, 1]
    rotor: np.ndarray = field(default_factory=lambda: np.zeros(4))
    thrust_n: float = 0.0
    on_ground: bool = True
    crashed: bool = False
    altitude_msl: float = 0.0

    def copy(self) -> "PlantState":
        return PlantState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            euler=self.euler.copy(),
            omega=self.omega.copy(),
            rotor=self.rotor.copy(),
            thrust_n=self.thrust_n,
            on_ground=self.on_ground,
            crashed=self.crashed,
            altitude_msl=self.altitude_msl,
        )

    @property
    def roll(self) -> float:
        return float(self.euler[0])

    @property
    def pitch(self) -> float:
        return float(self.euler[1])

    @property
    def yaw(self) -> float:
        return float(self.euler[2])

    @property
    def speed_ms(self) -> float:
        return float(math.hypot(self.vel[0], self.vel[1]))

    @property
    def agl(self) -> float:
        """Height above the local ground plane (negative if below)."""
        return -float(self.pos[2])


# --------------------------------------------------------------------------- #
# Controllers
# --------------------------------------------------------------------------- #
@dataclass
class PID:
    """Textbook PID with anti-windup clamp and derivative-on-measurement."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    imax: float = 1.0
    out_min: float = -float("inf")
    out_max: float = float("inf")
    d_filter: float = 0.0
    _i: float = field(default=0.0, init=False, repr=False)
    _d_prev: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._i = 0.0
        self._d_prev = 0.0

    def update(self, error: float, dt: float, measurement: Optional[float] = None) -> float:
        if dt <= 0:
            return 0.0
        p = self.kp * error
        self._i += self.ki * error * dt
        self._i = float(np.clip(self._i, -self.imax, self.imax))
        if self.kd > 0.0:
            d_src = error if measurement is None else -measurement
            raw = (d_src - self._d_prev) / dt
            if self.d_filter > 0.0:
                alpha = dt / (self.d_filter + dt)
                raw = alpha * raw + (1.0 - alpha) * getattr(self, "_d_filt", 0.0)
                self._d_filt = raw
            self._d_prev = d_src
            d = self.kd * raw
        else:
            d = 0.0
        return float(np.clip(p + self._i + d, self.out_min, self.out_max))


class AttitudeController:
    """Angle P + body-rate PID (ArduPilot AC_AttitudeControl equivalent)."""

    #: Rate-loop proportional gain [1/s].  The body-rate plant is a pure
    #: integrator (torque/I -> omega_dot -> omega), so a P controller closes it
    #: with time constant 1/kp.  At 40 that is 25 ms, comparable to what a real
    #: 8 kHz gyro-driven rate loop achieves.  The gains this replaces were
    #: ~0.135, i.e. a 7.4 s time constant: the attitude responded over seconds,
    #: so the velocity loop - which runs at the same rate and assumes attitude is
    #: a fast inner loop - wound up and overshot, and the aircraft was still
    #: holding a 24 deg pitch when its velocity error had already passed through
    #: zero.  A cascade only works if each loop is several times faster than the
    #: one outside it.
    RATE_P = 40.0
    RATE_I = 20.0
    RATE_IMAX = 15.0
    RATE_D = 0.40
    YAW_RATE_P = 30.0
    YAW_RATE_I = 10.0
    #: Angle-loop proportional gain [1/s]; its output is a body-rate target.
    ANGLE_P = 6.0
    #: Rate target limit [rad/s] - about 200 deg/s, well inside what the
    #: airframe can achieve and enough that the angle loop does not saturate on
    #: a normal attitude correction.
    ANGLE_RATE_LIMIT = 3.5

    def __init__(self, airframe: AirframeConfig) -> None:
        self.airframe = airframe
        self.angle_p = PID(kp=self.ANGLE_P, ki=0.0, kd=0.0,
                           out_min=-self.ANGLE_RATE_LIMIT,
                           out_max=self.ANGLE_RATE_LIMIT)
        self.rate_pid = [
            PID(kp=self.RATE_P, ki=self.RATE_I, kd=self.RATE_D,
                imax=self.RATE_IMAX),                       # roll
            PID(kp=self.RATE_P, ki=self.RATE_I, kd=self.RATE_D,
                imax=self.RATE_IMAX),                       # pitch
            PID(kp=self.YAW_RATE_P, ki=self.YAW_RATE_I, kd=0.0,
                imax=10.0),                                 # yaw
        ]

    def reset(self) -> None:
        self.angle_p.reset()
        for pid in self.rate_pid:
            pid.reset()

    def moment(self, euler: np.ndarray, omega: np.ndarray,
               target_euler: np.ndarray, dt: float, inertia: np.ndarray) -> np.ndarray:
        err = np.array([
            wrap_pi(target_euler[0] - euler[0]),
            wrap_pi(target_euler[1] - euler[1]),
            wrap_pi(target_euler[2] - euler[2]),
        ])
        rate_target = np.array([self.angle_p.update(err[i], dt) for i in range(3)])
        rate_err = rate_target - omega
        torque = np.zeros(3)
        for i in range(3):
            torque[i] = self.rate_pid[i].update(rate_err[i], dt, measurement=omega[i]) * inertia[i]
        return torque


class PositionController:
    """NED position/velocity controller producing an acceleration demand."""

    def __init__(self, airframe: AirframeConfig) -> None:
        self.airframe = airframe
        self.pos_p = PID(kp=1.6, ki=0.0, kd=0.0, out_min=-airframe.max_speed_ms,
                         out_max=airframe.max_speed_ms)
        self.vel_pid = PID(kp=3.2, ki=1.1, kd=0.0, imax=4.0, out_min=-14.0, out_max=14.0)
        # Output is a NED vertical acceleration, so out_min is the climb limit and
        # out_max the descent limit; they match the bounds Autopilot re-applies.
        self.vel_z_pid = PID(kp=4.0, ki=1.4, kd=0.0, imax=2.0,
                             out_min=-4.0, out_max=2.5)
        self.max_accel = 6.0

    def reset(self) -> None:
        self.pos_p.reset()
        self.vel_pid.reset()
        self.vel_z_pid.reset()

    def accel_demand(self, state: PlantState, target_pos: Optional[np.ndarray],
                     target_vel: Optional[np.ndarray], dt: float) -> np.ndarray:
        """Acceleration demand in NED [m/s^2] (D positive = accelerate down)."""
        demand = np.zeros(3)
        vel_target = np.zeros(3)
        if target_pos is not None:
            for axis in (0, 1):
                err = target_pos[axis] - state.pos[axis]
                vel_target[axis] = self.pos_p.update(err, dt)
            vel_target[2] = self.pos_p.update(target_pos[2] - state.pos[2], dt)
            vel_target[2] = float(np.clip(vel_target[2], -self.airframe.max_climb_ms,
                                          self.airframe.max_climb_ms))
        if target_vel is not None:
            vel_target = np.asarray(target_vel, dtype=float) + vel_target * (target_pos is None)
        for axis in (0, 1):
            demand[axis] = self.vel_pid.update(vel_target[axis] - state.vel[axis], dt)
        demand[2] = self.vel_z_pid.update(vel_target[2] - state.vel[2], dt)
        mag = float(np.linalg.norm(demand))
        if mag > self.max_accel:
            demand *= self.max_accel / mag
        return demand


# --------------------------------------------------------------------------- #
# Plant
# --------------------------------------------------------------------------- #
class QuadrotorPlant:
    """6-DoF quadrotor in NED with per-rotor thrust allocation."""

    def __init__(
        self,
        powertrain: Powertrain,
        airframe: Optional[AirframeConfig] = None,
        ground_elevation_m: float = 0.0,
    ) -> None:
        self.air = airframe or powertrain.airframe
        self.pt = powertrain
        self.state = PlantState()
        self.state.rotor = np.zeros(self.air.n_rotors)
        self.ground_elevation = float(ground_elevation_m)
        self.inertia = np.array(self.air.inertia, dtype=float)
        self.wind = np.zeros(3)
        self._drag_coef = self.air.drag_coefficient * self.air.frontal_area_m2
        # X-configuration rotor geometry: (x, y) offsets and spin direction.
        self._rotor_pos, self._rotor_dir = self._build_geometry()
        self.max_thrust_per_rotor_n = self._estimate_max_rotor_thrust()

    def _build_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.air.n_rotors
        l = self.air.arm_length_m
        angles = [(math.pi / 4.0) + i * (2.0 * math.pi / n) for i in range(n)]
        pos = np.array([[l * math.cos(a), l * math.sin(a)] for a in angles])
        direction = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        return pos, direction

    def _estimate_max_rotor_thrust(self) -> float:
        """Static thrust of one rotor at wide-open throttle [N].

        Taken from the configured propeller ceiling (measured on a thrust
        stand for the reference build) rather than extrapolated from momentum
        theory, which ignores the motor's RPM limit and over-predicts by ~2x.
        """
        return float(self.air.max_rotor_thrust_n)

    # ------------------------------------------------------------------ #
    def set_rotor_throttle(self, throttle: np.ndarray) -> None:
        self.state.rotor = np.clip(np.asarray(throttle, dtype=float), 0.0, 1.0)

    def allocate(self, thrust_n: float, torque: np.ndarray) -> np.ndarray:
        """Least-squares thrust/torque -> per-rotor normalised thrust."""
        n = self.air.n_rotors
        tmax = self.max_thrust_per_rotor_n
        # Build mixing matrix M: [Fz, Lx, My, Nz] = M @ t_i
        rows = []
        for i in range(n):
            x, y = self._rotor_pos[i]
            kq = 0.028   # torque-to-thrust ratio for a hobby prop
            rows.append([1.0, -y, x, self._rotor_dir[i] * kq])
        m = np.array(rows).T                     # 4 x n
        desired = np.array([max(thrust_n, 0.0) / max(tmax, 1e-6),
                            torque[0], torque[1], torque[2]])
        sol, *_ = np.linalg.lstsq(m, desired, rcond=None)

        # Desaturate rather than clip.  For a symmetric X-quad the null space of
        # the three torque rows is the collective direction [1,1,1,1], so the
        # solution separates into a mean (thrust) and a deviation (torque).
        # Scaling the deviation down toward zero sheds attitude authority while
        # preserving total thrust, which is the trade a real allocator makes.
        #
        # Clipping instead is what produces the failure this replaces: a large
        # attitude error drives the least-squares solution far outside [0, 1],
        # and clipping pins two rotors to 1.0 and two to 0.0.  The total thrust
        # then locks at exactly half of maximum - 55 N here, against a 30 N
        # weight - *independently of what was asked for*, so the aircraft climbs
        # away at a constant rate no controller can arrest, and the commanded
        # thrust reading looks perfectly reasonable the whole time.
        mean = float(sol.mean())
        dev = sol - mean
        hi, lo = float(dev.max()), float(dev.min())
        alpha = 1.0
        if hi > 1e-9:
            alpha = min(alpha, (1.0 - mean) / hi)
        if lo < -1e-9:
            alpha = min(alpha, (0.0 - mean) / lo)
        sol = mean + max(0.0, alpha) * dev
        return np.clip(sol, 0.0, 1.0)

    def step(self, dt: float, altitude_msl: Optional[float] = None) -> PlantState:
        """Advance the plant by ``dt`` seconds (sub-stepped internally)."""
        steps = max(1, int(math.ceil(dt / 0.002)))
        h = dt / steps
        for _ in range(steps):
            self._integrate(h)
        if altitude_msl is not None:
            self.state.altitude_msl = altitude_msl
        return self.state

    def _integrate(self, h: float) -> None:
        st = self.state
        roll, pitch, yaw = st.euler
        dcm = euler_to_dcm(roll, pitch, yaw)

        # --- forces ---
        rotor_thrust = st.rotor * self.max_thrust_per_rotor_n     # [N] each
        total_thrust = float(rotor_thrust.sum())
        st.thrust_n = total_thrust

        thrust_body = np.array([0.0, 0.0, -total_thrust])
        thrust_ned = dcm @ thrust_body

        mass = self.pt.mass_total
        gravity_ned = np.array([0.0, 0.0, mass * G0])

        # Aerodynamic drag relative to airmass.
        v_rel = st.vel - self.wind
        speed = float(np.linalg.norm(v_rel[:2]))
        rho = self.pt.rho
        drag_ned = -0.5 * rho * self._drag_coef * speed * np.array([v_rel[0], v_rel[1], 0.0])
        # Vertical drag is much smaller (frame is thin in Z).
        drag_ned[2] = -0.25 * rho * self._drag_coef * abs(v_rel[2]) * v_rel[2]

        forces = thrust_ned + gravity_ned + drag_ned
        accel = forces / mass

        # Ground effect: extra thrust efficiency close to the surface.
        agl = -st.pos[2] - self.ground_elevation
        if 0.0 < agl < 1.2 and total_thrust > 0:
            ges = 1.0 + 0.18 * max(0.0, (1.2 - agl) / 1.2)
            accel += dcm @ np.array([0.0, 0.0, -total_thrust * (ges - 1.0)]) / mass

        # --- moments ---
        torque = np.zeros(3)
        kq = 0.028
        for i in range(self.air.n_rotors):
            x, y = self._rotor_pos[i]
            t = float(rotor_thrust[i])
            torque[0] += -y * t
            torque[1] += x * t
            torque[2] += self._rotor_dir[i] * kq * t
        # Aerodynamic damping + gyroscopic coupling
        torque -= 0.02 * st.omega * np.linalg.norm(st.omega)
        omega_dot = self.inertia ** -1 * (
            torque - np.cross(st.omega, self.inertia * st.omega)
        )

        # --- integrate (semi-implicit) ---
        st.omega = st.omega + omega_dot * h
        st.euler = st.euler + _euler_rates(st.euler, st.omega) * h
        st.euler[0] = wrap_pi(st.euler[0])
        st.euler[1] = wrap_pi(st.euler[1])
        st.euler[2] = wrap_pi(st.euler[2])
        st.vel = st.vel + accel * h
        st.pos = st.pos + st.vel * h

        # --- ground contact / crash ---
        # The ground can only push, never pull: the clamp must arrest a
        # *descent*, and leave a climb alone.  Applying it unconditionally while
        # inside the tolerance band pins the vehicle to the pad, because at
        # h = 2 ms a takeoff acceleration of ~9 m/s^2 moves it only ~4e-5 m per
        # sub-step - less than the band is wide - so the upward velocity is
        # zeroed every single step and the aircraft never leaves the ground no
        # matter how much thrust it has.  Rotor saturation and controller output
        # both look perfectly normal while this happens.
        z_ground = self.ground_elevation
        if st.pos[2] >= -z_ground - 1e-4:
            st.pos[2] = -z_ground
            if accel[2] < -0.5:
                # Net upward acceleration: thrust exceeds weight by enough to
                # actually leave, so the pad is no longer constraining anything.
                # The margin matters - with exactly hover throttle the residual
                # is integration noise either side of zero, and a bare
                # ``< 0.0`` makes ``on_ground`` flicker, so a landing is never
                # registered and LAND/RTL never auto-disarm.
                # Testing *velocity* here instead cannot work - it accumulates
                # from zero, so at a 2 ms sub-step the aircraft is inside the
                # tolerance band for the first several milliseconds of every
                # takeoff, any threshold small enough to release it is also
                # small enough to be noise, and any threshold large enough to be
                # meaningful pins it to the pad forever.  Acceleration is the
                # quantity the normal force actually opposes.
                st.on_ground = False
            else:
                if st.vel[2] > 4.5:
                    st.crashed = True
                st.vel = np.array([st.vel[0] * 0.2, st.vel[1] * 0.2, 0.0])
                st.omega *= 0.1
                st.euler[0] *= 0.5
                st.euler[1] *= 0.5
                st.on_ground = True
        else:
            st.on_ground = False
        if abs(st.euler[0]) > math.radians(75) or abs(st.euler[1]) > math.radians(75):
            st.crashed = True

    # ------------------------------------------------------------------ #
    def power_draw_w(self) -> float:
        """Electrical power for the current rotor state and flight condition."""
        st = self.state
        if st.rotor.sum() < 1e-6:
            return self.pt.avionics_w
        speed = float(math.hypot(st.vel[0] - self.wind[0], st.vel[1] - self.wind[1]))
        climb = float(-st.vel[2])
        return self.pt.power_for_thrust(st.thrust_n, speed, climb)


def _euler_rates(euler: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Body rates -> ZYX Euler angle rates."""
    roll, pitch, _ = euler
    p, q, r = omega
    cp = math.cos(pitch)
    if abs(cp) < 1e-6:
        cp = 1e-6 * math.copysign(1.0, cp)
    sp, cr, sr = math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([
        p + (q * sr + r * cr) * sp / cp,
        q * cr - r * sr,
        (q * sr + r * cr) / cp,
    ])


# --------------------------------------------------------------------------- #
# Top-level autopilot wrapper (used by MiniSITL)
# --------------------------------------------------------------------------- #
class Autopilot:
    """Position/attitude autopilot that drives :class:`QuadrotorPlant`.

    This is *not* ArduPilot - it is a behaviour-compatible stand-in with the
    same external contract (Guided velocity/position setpoints, Auto waypoint
    following, RTL, LAND, LOITER, failsafes).  Swapping in real ArduPilot SITL
    or a physical TBS Lucid H743 changes only the transport layer.
    """

    def __init__(self, powertrain: Powertrain, airframe: Optional[AirframeConfig] = None) -> None:
        self.air = airframe or powertrain.airframe
        self.plant = QuadrotorPlant(powertrain, self.air)
        self.att = AttitudeController(self.air)
        self.pos = PositionController(self.air)
        self.target_pos: Optional[np.ndarray] = None
        self.target_vel: Optional[np.ndarray] = None
        self.target_yaw: float = 0.0
        self.target_altitude: Optional[float] = None
        self.desired_climb_ms: float = 0.0
        #: Vertical acceleration bounds [m/s^2], NED positive-down.  A search
        #: aircraft climbs briskly and descends gently: the payload hangs below
        #: the CG and a hard descent swings it, and the descent rate also sets how
        #: much warning the rangefinder gets before touchdown.
        self.max_climb_accel_ms2: float = 4.0
        self.max_descent_accel_ms2: float = 2.5
        self.armed = False
        self.mode = "STABILIZE"
        self.dt = 0.004

    # -- high-level commands ------------------------------------------- #
    def arm(self) -> None:
        self.armed = True
        self.att.reset()
        self.pos.reset()

    def disarm(self) -> None:
        self.armed = False
        self.plant.set_rotor_throttle(np.zeros(self.air.n_rotors))

    def goto_ned(self, north: float, east: float, down: float,
                 yaw: Optional[float] = None) -> None:
        self.target_pos = np.array([north, east, down], dtype=float)
        self.target_vel = None
        if yaw is not None:
            self.target_yaw = float(yaw)

    def set_velocity_ned(self, vn: float, ve: float, vd: float,
                         yaw: Optional[float] = None) -> None:
        self.target_vel = np.array([vn, ve, vd], dtype=float)
        self.target_pos = None
        if yaw is not None:
            self.target_yaw = float(yaw)

    def loiter(self) -> None:
        st = self.plant.state
        self.target_pos = st.pos.copy()
        self.target_vel = None

    # -- control loop -------------------------------------------------- #
    def update(self, dt: float) -> PlantState:
        self.plant.step(dt)
        st = self.plant.state
        if not self.armed or st.crashed:
            self.plant.set_rotor_throttle(np.zeros(self.air.n_rotors))
            return st

        accel = np.zeros(3)
        if self.target_pos is not None or self.target_vel is not None:
            accel = self.pos.accel_demand(st, self.target_pos, self.target_vel, dt)
        elif self.mode in ("LAND",):
            accel = np.array([0.0, 0.0, 0.9])      # gentle descent demand
        elif self.mode == "STABILIZE":
            accel = np.zeros(3)

        mass = self.plant.pt.mass_total

        # Vertical demand, NED positive-down.  It has to be bounded *away* from
        # +G0: the specific force the rotors must produce is (G0 - a_z) upward,
        # and as a_z approaches G0 that goes to zero, so the tilt needed for any
        # horizontal acceleration at all approaches 90 deg and the thrust
        # magnitude approaches zero.  That is a degenerate operating point the
        # allocator cannot represent, and reaching it is how a controller ends up
        # commanding 60 deg of pitch to hold a hover.
        az_dem = float(np.clip(accel[2], -self.max_climb_accel_ms2,
                               self.max_descent_accel_ms2))
        specific_up = G0 - az_dem                    # always > 0 by construction

        # Horizontal acceleration is limited by the tilt the airframe allows and
        # by the vertical thrust margin left over, so the tilt computed below can
        # never need clipping - and if it never needs clipping, the thrust
        # magnitude reproduces the desired vector exactly instead of being
        # silently inconsistent with it.
        h_dem = np.array([accel[0], accel[1]], dtype=float)
        h_max = specific_up * math.tan(self.air.max_tilt_rad)
        h_mag = float(np.linalg.norm(h_dem))
        if h_mag > h_max:
            h_dem *= h_max / max(h_mag, 1e-9)

        # Desired thrust vector: cancel gravity, then apply demanded accel.
        desired_ned = np.array([h_dem[0], h_dem[1], az_dem - G0]) * mass
        thrust_n = float(np.clip(np.linalg.norm(desired_ned), 0.0,
                                 self.plant.max_thrust_per_rotor_n * self.air.n_rotors))
        # Desired tilt from the horizontal components of the demand.
        #
        # The signs here follow from euler_to_dcm's ZYX/NED convention, checked
        # numerically rather than from memory: body thrust is [0, 0, -T], so
        #   pitch +10 deg -> NED force ax = -5.28   (nose up flies SOUTH)
        #   roll  +10 deg -> NED force ay = +5.28   (right wing down flies EAST)
        # Northward acceleration therefore needs NEGATIVE pitch and eastward
        # needs POSITIVE roll.  Getting either one backwards is not a small
        # error: the tilt fights the demand, the velocity error grows, the
        # controller commands more tilt in the same wrong direction, and the
        # result is positive feedback that accelerates the aircraft away from
        # its target until it hits the tilt or speed limit.  It reads as a
        # runaway, not as a sign error, which is why the convention is written
        # down here with the numbers that establish it.
        ax, ay = h_dem[0], h_dem[1]
        az = desired_ned[2] / mass                   # == -(specific_up) < 0
        # The demand above is in NED, but the thrust vector lies along the BODY
        # -Z axis, so it has to be rotated by the ACTUAL yaw before it becomes a
        # roll and a pitch.  Solving the small-angle thrust projection
        #   a_north = -g(cos(psi) theta + sin(psi) phi)
        #   a_east  =  g(-sin(psi) theta + cos(psi) phi)
        # for (theta, phi) gives exactly this rotation.
        #
        # Omitting it is invisible while the aircraft points north, which is why
        # every north-only test in this project passed: at psi = 0 the rotation
        # is the identity.  At psi = 180 deg it is a sign flip on both axes, so
        # the tilt drives the aircraft AWAY from its own demand, the error grows,
        # the controller commands more tilt in the same wrong direction, and the
        # airframe accelerates to the tilt limit and stays there.  It reads as an
        # unarrestable runaway rather than as a missing rotation.
        #
        # Actual yaw, not the target: during a slew the thrust is along the
        # airframe's real axis, and using the commanded one reintroduces the
        # error for the duration of every turn.
        psi = float(st.euler[2])
        c, sn = math.cos(psi), math.sin(psi)
        a_fwd = c * ax + sn * ay                     # body +X, forward
        a_right = -sn * ax + c * ay                  # body +Y, starboard
        pitch_d = -math.atan2(a_fwd, -az)
        roll_d = math.atan2(a_right, -az) * math.cos(pitch_d)
        pitch_d = float(np.clip(pitch_d, -self.air.max_tilt_rad, self.air.max_tilt_rad))
        roll_d = float(np.clip(roll_d, -self.air.max_tilt_rad, self.air.max_tilt_rad))

        target_euler = np.array([roll_d, pitch_d, self.target_yaw])
        torque = self.att.moment(st.euler, st.omega, target_euler, dt, self.plant.inertia)
        throttle = self.plant.allocate(thrust_n, torque)
        self.plant.set_rotor_throttle(throttle)
        return st
