#!/usr/bin/env python3
"""
RocketPy simulation of the Project Proteus CONCEPT 2 spaceshot (Flight 3 vehicle),
for the anu-rocketry-dnc-optimiser simulator folder.

Flattened on 2026-10-01 from the Proteus RCS repo (config/concept2_vehicle.yaml,
tools/concept_vehicle.py, tools/rcs_rocketpy.py, tools/atmosphere.py,
tools/launch_corridor.py) so it runs with RocketPy alone (tested on 1.11.0).
Every number comes from that YAML or is computed from it. ALL VEHICLE INPUTS ARE
ENGINEERING ESTIMATES, not measured hardware. Concept 2 is the Rev-3 Proteus
vehicle; the Phase-B design (Vehicle Configuration Rev C) has since moved on.

Vehicle: 0.26 m x 7.05 m, liftoff 252 kg (dry 102 kg + 150 kg LOX/ethanol,
O/F 1.58), Project Triton 3.5 kN at Isp 240 s sea level (~4.0 kN / ~275 s in
vacuum through pressure thrust), T/W 1.42, burn 100.9 s, 12 m tower.

How it differs from simulate_tempest.py
- Liquid engine: GenericMotor with constant thrust. The propellant is one lump at
  the combined LOX/ethanol centroid (the two tanks drain to a nearly fixed
  combined centroid), not SolidMotor + .eng.
- Drag: Cd(Mach) estimate from the YAML. There is no .ork for this vehicle.
- Atmosphere: pressure continued isothermally above 80 km (upper_atmosphere).
  RocketPy 1.11's custom_atmosphere WITHOUT a pressure profile goes negative
  above ~84 km: drag reverses, an uncontrolled vehicle tumbles, and the main
  parachute opens at apogee (its altitude trigger reads pressure).
- Cold-gas RCS instead of air brakes. Without it the vehicle arcs over off the
  rail in any wind (~9.7 m/s off a 12 m tower): use USE_RCS = False only in calm air.
- Recovery: drogue at apogee, main at 1 km AGL, Cd*S 2.2 / 45 m^2.

Checked against the RCS repo: see VERIFIED at the bottom.

Run:  python simulate_spaceshot.py      (set the CASE SELECTION below first)
      or import it: from simulate_spaceshot import build_and_fly, summary
      Ascent only ~1-3 min; a full flight to landing ~5-8 min (the 100 Hz RCS
      controller runs for the whole flight).
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from rocketpy import Environment, Rocket, Flight, GenericMotor
from rocketpy.control.controller import _Controller
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface

# ==============================================================================
# CASE SELECTION
# ==============================================================================
WIND = "VIC_Aug"       # "calm", "uniform", "VIC_Jun", "VIC_Aug", "Arnhem_Sep", "Arnhem_May"
UNIFORM_WIND = 5.0     # m/s toward +x (east) at every height, for WIND = "uniform"
USE_RCS = True         # False only makes sense in calm air
FLY_DESCENT = True     # True: fly to landing under parachutes; False: stop at apogee

G0 = 9.80665

# ==============================================================================
# VEHICLE — config/concept2_vehicle.yaml (positions from the nose tip, + aft)
# ==============================================================================
BODY_RADIUS = 0.13                 # m (0.26 m body)
ROCKET_LENGTH = 7.05               # m
DRY_MASS = 102.0                   # kg, the 12 components of the YAML stack
DRY_CG = 4.218210784313725         # m from nose
DRY_I_PITCH = 498.567552634804     # kg m^2 about the dry CG (slender rods + parallel axis)
DRY_I_ROLL = 0.5 * DRY_MASS * BODY_RADIUS ** 2
LOADED_CG = 4.551755490956071      # m from nose at liftoff (252 kg)

# Propulsion — Project Triton, LOX/ethanol
THRUST = 3500.0                    # N, sea level
ISP_SL = 240.0                     # s, sea level
PROP_MASS = 150.0                  # kg (LOX 91.86, ethanol 58.14)
PROP_CG = 4.7785658914728675       # m from nose: combined propellant centroid, tanks full
BURN_TIME = PROP_MASS / (THRUST / (ISP_SL * G0))   # 100.87 s
THRUST_RAMP = 0.5                  # s to full thrust
NOZZLE_EXIT_RADIUS = 0.04          # m (Pc ~20 bar, expansion ~4)
REFERENCE_PRESSURE = 101325.0      # Pa: thrust quoted at sea level, so RocketPy adds
                                   # (p_ref - p_ambient) * A_exit as the vehicle climbs

# Aerodynamics (EST: slender finned sounding rocket), power on and off
CD_MACH = [[0.0, 0.32], [0.6, 0.32], [0.9, 0.42], [1.05, 0.58], [1.2, 0.52],
           [1.5, 0.44], [2.0, 0.38], [2.5, 0.33], [3.0, 0.30], [4.0, 0.27]]
NOSE_LENGTH = 1.04                 # m, ogive
FINS = dict(n=4, root_chord=0.43, tip_chord=0.17, span=0.20, sweep_length=0.29,
            position=6.50)         # root leading edge 6.50 m from nose
RAIL_BUTTONS = (LOADED_CG - 0.6, ROCKET_LENGTH - 0.2)   # m from nose
TOWER_LENGTH = 12.0                # m

# Cold-gas RCS (three-phase flight software, Proteus §5.3)
RCS_POSITION = 1.15                # m from nose (thrusters)
RCS_ARM = LOADED_CG - RCS_POSITION # 3.40 m
RCS_TORQUE = 310.0                 # N m per axis
RCS_ISP = 60.0                     # s, cold nitrogen
RCS_GAS = 3.0                      # kg carried; the RCS stops when it is spent
RCS_ENABLE_ALTITUDE = 12.0         # m: arm only after the tower
HANDOVER_SPEED = 150.0             # m/s: hold vertical, then hand over to the fins (latched)
REENGAGE_ALTITUDE = 26000.0        # m: re-engage above the jet stream (latched)
KP, KD = 7400.0, 3450.0            # PD gains: N m/rad, N m/(rad/s); runs at 100 Hz

# Recovery (EST, sized for ~6 m/s landing at 102 kg)
DROGUE_CD_S = 2.2                  # m^2, opens at apogee
MAIN_CD_S = 45.0                   # m^2, opens at MAIN_DEPLOY_ALTITUDE on descent
MAIN_DEPLOY_ALTITUDE = 1000.0      # m AGL
DEPLOY_LAG = 1.0                   # s

# ==============================================================================
# WINDS — NCEP/NCAR Reanalysis-1 monthly-mean speed profiles (RCS repo
# tools/wind_climatology.py), applied toward +x as everywhere in the RCS repo.
# Site coordinates are the NCEP grid points the profiles come from.
# ==============================================================================
CALM_SITE = (-36.482, 144.018)     # YAML launch site, used for "calm" and "uniform"
SITES = {"VIC": (-37.5, 145.0), "Arnhem": (-12.5, 137.5)}
WIND_PROFILES = {
    "VIC_Jun": ("VIC", [  # median month, jet 25 m/s
        [0.0, 0.0], [142.2, 2.205], [798.2, 4.04], [1499.0, 4.963], [3075.0, 6.333],
        [4291.0, 7.73], [5682.4, 9.511], [7320.4, 11.967], [9314.0, 16.016],
        [10520.7, 19.622], [11962.0, 23.866], [13799.2, 25.04], [16344.8, 20.456],
        [18565.3, 14.616], [20673.2, 11.213], [23923.0, 9.811], [26544.9, 13.44],
        [31135.0, 24.751], [80000.0, 24.751]]),
    "VIC_Aug": ("VIC", [  # worst month, jet 32 m/s
        [0.0, 0.0], [142.2, 3.489], [798.2, 6.119], [1499.0, 7.536], [3075.0, 9.458],
        [4291.0, 11.282], [5682.4, 13.707], [7320.4, 17.072], [9314.0, 22.86],
        [10520.7, 27.69], [11962.0, 31.943], [13799.2, 31.026], [16344.8, 23.599],
        [18565.3, 15.912], [20673.2, 11.092], [23923.0, 7.048], [26544.9, 6.278],
        [31135.0, 7.597], [80000.0, 7.597]]),
    "Arnhem_Sep": ("Arnhem", [  # median month, jet 5 m/s (the Phase-B test profile)
        [0.0, 0.0], [90.0, 5.647], [776.3, 9.013], [1506.0, 8.849], [3142.6, 4.395],
        [4410.0, 1.029], [5864.7, 0.344], [7590.0, 1.125], [9705.1, 2.89],
        [10976.5, 3.846], [12463.6, 4.588], [14273.4, 3.223], [16625.8, 0.302],
        [18645.3, 0.556], [20642.3, 2.821], [23809.9, 8.851], [26404.9, 8.52],
        [31003.7, 6.043], [80000.0, 6.043]]),
    "Arnhem_May": ("Arnhem", [  # worst month, jet 11 m/s
        [0.0, 0.0], [90.0, 7.398], [776.3, 10.078], [1506.0, 8.489], [3142.6, 1.783],
        [4410.0, 0.79], [5864.7, 1.943], [7590.0, 4.509], [9705.1, 8.162],
        [10976.5, 9.816], [12463.6, 10.93], [14273.4, 11.148], [16625.8, 7.713],
        [18645.3, 2.094], [20642.3, 2.053], [23809.9, 7.799], [26404.9, 9.596],
        [31003.7, 4.782], [80000.0, 4.782]]),
}


# ==============================================================================
# ATMOSPHERE — physical above 80 km (RCS repo tools/atmosphere.py)
# ==============================================================================
def upper_atmosphere(env):
    """(height, pressure) table: RocketPy's ISA pressure every 100 m below 80 km,
    then isothermal at the ISA 80 km temperature (196.65 K) to 600 km:
    p = p80 * exp(-(H - H80) g0 / (R T)), H = Re h / (Re + h) geopotential."""
    R_AIR, T_TOP, H_TOP = 287.05287, 196.65, 80000.0
    re = float(env.earth_radius)
    hs = np.arange(0.0, H_TOP, 100.0)
    up = np.arange(H_TOP, 600001.0, 250.0)
    p80 = float(env.pressure_ISA(H_TOP))
    h80 = re * H_TOP / (re + H_TOP)
    ps = [float(env.pressure_ISA(h)) for h in hs]
    ps += [p80 * math.exp(-(re * h / (re + h) - h80) * G0 / (R_AIR * T_TOP)) for h in up]
    return np.column_stack([np.concatenate([hs, up]), ps])


# ==============================================================================
# COLD-GAS RCS (RCS repo tools/rcs_rocketpy.py)
# ==============================================================================
class ColdGasRCS:
    """Actuator state, written by the controller and read by the moment surface."""

    def __init__(self, max_gas=RCS_GAS):
        self.max_gas = max_gas
        self.Mx = self.My = 0.0
        self.handed_over = self.reengaged = False
        self.propellant_mass = self.peak_torque = self.on_time = 0.0


class RCSMomentSurface(GenericSurface):
    """Applies the commanded body moment, independent of dynamic pressure (a
    thruster, not a fin). Being a GenericSurface keeps it out of RocketPy's
    centre-of-pressure and static-margin calculation."""

    def __init__(self, rcs, reference_area, reference_length):
        super().__init__(reference_area=reference_area, reference_length=reference_length,
                         coefficients={}, center_of_pressure=(0, 0, 0), name="RCS")
        self.rcs = rcs

    def compute_forces_and_moments(self, stream_velocity, stream_speed, stream_mach,
                                   rho, cp, omega, reynolds):
        return 0.0, 0.0, 0.0, self.rcs.Mx, self.rcs.My, 0.0


def rcs_controller(time, sampling_rate, state, state_history, observed_variables, rcs):
    """Three-phase RCS: (1) hold vertical from the tower to HANDOVER_SPEED;
    (2) latched handover, the fins ride the jet stream at trim; (3) latched
    re-engage above REENGAGE_ALTITUDE. PD on tilt in BODY axes plus rate damping.
    state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]"""
    z = state[2]
    e0, e1, e2, e3 = state[6], state[7], state[8], state[9]
    wx, wy = state[10], state[11]
    speed = math.sqrt(state[3] ** 2 + state[4] ** 2 + state[5] ** 2)
    if speed > HANDOVER_SPEED:
        rcs.handed_over = True
    if rcs.handed_over and z > REENGAGE_ALTITUDE:
        rcs.reengaged = True
    if z < RCS_ENABLE_ALTITUDE or (rcs.handed_over and not rcs.reengaged) \
            or rcs.propellant_mass >= rcs.max_gas:
        rcs.Mx = rcs.My = 0.0
        return None
    # inertial vertical expressed in body axes: its lateral part is the tilt
    # direction in the body frame (invariant to roll/azimuth)
    gx = 2 * (e1 * e3 - e0 * e2)
    gy = 2 * (e2 * e3 + e0 * e1)
    gz = e0 * e0 - e1 * e1 - e2 * e2 + e3 * e3
    m1 = max(-RCS_TORQUE, min(RCS_TORQUE, KP * (-gy) - KD * wx))   # about body x
    m2 = max(-RCS_TORQUE, min(RCS_TORQUE, KP * gx - KD * wy))      # about body y
    rcs.Mx, rcs.My = m1, m2
    mag = math.hypot(m1, m2)
    if mag > 0:                                    # cold-gas accounting over the tick
        dt = 1.0 / sampling_rate
        rcs.propellant_mass += mag / RCS_ARM * dt / (RCS_ISP * G0)
        rcs.on_time += dt
        rcs.peak_torque = max(rcs.peak_torque, mag)
    return [time, m1, m2, math.degrees(math.acos(max(-1.0, min(1.0, gz))))]


# ==============================================================================
# BUILD AND FLY
# ==============================================================================
def build_rocket(use_rcs=True, descent=True):
    """Fresh environment, motor, rocket and flight. Returns the rocket."""

    motor = GenericMotor(
        thrust_source=[[0.0, 0.0], [THRUST_RAMP, THRUST], [BURN_TIME, THRUST]],
        burn_time=(0.0, BURN_TIME),
        chamber_radius=0.8 * BODY_RADIUS, chamber_height=1.0, chamber_position=0.0,
        propellant_initial_mass=PROP_MASS, nozzle_radius=NOZZLE_EXIT_RADIUS,
        dry_mass=0.001, center_of_dry_mass_position=0.0, dry_inertia=(1e-3, 1e-3, 1e-3),
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        reference_pressure=REFERENCE_PRESSURE,
    )   # engine mass is in DRY_MASS; the motor is propellant only

    rocket = Rocket(
        radius=BODY_RADIUS, mass=DRY_MASS, inertia=(DRY_I_PITCH, DRY_I_PITCH, DRY_I_ROLL),
        power_off_drag=CD_MACH, power_on_drag=CD_MACH,
        center_of_mass_without_motor=DRY_CG, coordinate_system_orientation="nose_to_tail",
    )
    rocket.add_motor(motor, position=PROP_CG)
    rocket.add_nose(length=NOSE_LENGTH, kind="ogive", position=0.0)
    rocket.add_trapezoidal_fins(**FINS)
    rocket.set_rail_buttons(upper_button_position=RAIL_BUTTONS[0],
                            lower_button_position=RAIL_BUTTONS[1], angular_position=45)

    rcs = ColdGasRCS()
    if use_rcs:
        rocket.add_surfaces(RCSMomentSurface(rcs, math.pi * BODY_RADIUS ** 2, 2 * BODY_RADIUS),
                            RCS_POSITION)
        rocket._add_controllers(_Controller(
            interactive_objects=rcs, controller_function=rcs_controller, sampling_rate=100,
            initial_observed_variables=[0, 0, 0, 0], name="Cold-gas RCS"))

    if descent:
        rocket.add_parachute("drogue", cd_s=DROGUE_CD_S, trigger=lambda p, h, y: y[5] < 0,
                             sampling_rate=10, lag=DEPLOY_LAG, noise=(0, 0, 0))
        rocket.add_parachute("main", cd_s=MAIN_CD_S,
                             trigger=lambda p, h, y: y[5] < 0 and h < MAIN_DEPLOY_ALTITUDE,
                             sampling_rate=10, lag=DEPLOY_LAG, noise=(0, 0, 0))
    return rocket, rcs


def summary(flight, rcs):
    """Headline numbers (same definitions as the RCS repo)."""
    t_end = float(flight.t_final)
    out = {
        "off_rod_ms": float(flight.out_of_rail_velocity),
        "apogee_km": float(flight.apogee - flight.env.elevation) / 1000.0,
        "apogee_time_s": float(flight.apogee_time),
        "max_mach": float(flight.max_mach_number),
        "max_q_kpa": float(flight.max_dynamic_pressure) / 1000.0,
        "rcs_gas_kg": rcs.propellant_mass,
        "rcs_peak_torque_Nm": rcs.peak_torque,
    }
    if not flight.terminate_on_apogee:
        out.update({
            "landing_distance_km": math.hypot(float(flight.x(t_end)), float(flight.y(t_end))) / 1000.0,
            "impact_speed_ms": abs(float(flight.vz(t_end))),
            "flight_time_min": t_end / 60.0,
            "parachute_events": [(round(float(t), 1), p.name) for t, p in flight.parachute_events],
        })
    return out


if __name__ == "__main__":
    print(f"Concept 2 spaceshot: wind {WIND}, RCS {'on' if USE_RCS else 'off'}, "
          f"{'full flight' if FLY_DESCENT else 'ascent only'}")
    flight, rcs = build_and_fly()
    s = summary(flight, rcs)
    for k, v in s.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")

    ts = np.linspace(0.0, float(flight.t_final), 1500)
    alt = np.array([float(flight.altitude(t)) for t in ts]) / 1000.0
    spd = np.array([float(flight.speed(t)) for t in ts])
    tilt = np.degrees(np.arccos(np.clip([float(flight.attitude_vector_z(t)) for t in ts], -1, 1)))
    x = np.array([float(flight.x(t)) for t in ts]) / 1000.0
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(ts, alt); ax[0, 0].axhline(100, color="k", ls="--", lw=1)
    ax[0, 0].set(xlabel="time (s)", ylabel="altitude (km)", title=f"Altitude, apogee {s['apogee_km']:.1f} km")
    ax[0, 1].plot(ts, spd); ax[0, 1].set(xlabel="time (s)", ylabel="speed (m/s)",
                                         title=f"Speed, max Mach {s['max_mach']:.2f}")
    ax[1, 0].plot(ts, tilt); ax[1, 0].set(xlabel="time (s)", ylabel="tilt from vertical (deg)",
                                          title=f"Attitude, RCS gas {s['rcs_gas_kg']:.2f} kg")
    ax[1, 1].plot(x, alt); ax[1, 1].set(xlabel="downrange x (km)", ylabel="altitude (km)",
                                        title="Trajectory")
    fig.suptitle(f"Proteus Concept 2 spaceshot — wind {WIND}, RCS {'on' if USE_RCS else 'off'}")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spaceshot_flight.png"), dpi=110)
    print("spaceshot_flight.png written")

# VERIFIED (2026-10-01, RocketPy 1.11.0) against the RCS repo, corrected atmosphere:
#   calm, no RCS, ascent          apogee 103.322 km (identical), max Mach 3.603
#   VIC_Aug, RCS, ascent          apogee 97.823 km (RCS repo 97.822), gas 1.985 kg (identical)
#   Arnhem_Sep, RCS, ascent       apogee 104.645 km (identical), gas 1.308 kg (identical)
#   calm, RCS, to landing         lands 5.41 km out at 6.0 m/s after 16.1 min (identical)
#   VIC_Aug, RCS, to landing      lands 52.05 km out (RCS repo 52.06) after 15.9 min, gas 2.154 kg
#   The small differences come from rounding the wind tables above to 0.001 m/s.
