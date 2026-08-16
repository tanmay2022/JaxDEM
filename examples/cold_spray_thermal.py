"""
Reduced-order cold-spray impact with thermal energy accounting.

A Cu particle impacts a rigid flat substrate at high velocity.

Mechanical model
----------------
    substrate contact = spring + dashpot

Thermal model
-------------
    dissipated dashpot work -> particle heat

    Q = integral(c * v_n^2 dt)

    Delta T = Q / (m * cp)

The model is intentionally reduced-order. It does NOT resolve:
    - spatial temperature gradients
    - plastic strain fields
    - thermal conduction
    - phase change
    - oxide fracture
    - true adiabatic shear localization

It does provide a useful energy-consistent first model for:
    kinetic energy -> contact dissipation -> temperature rise

Run:

    python examples/cold_spray_thermal.py

Outputs:

    cold_spray_output/
        cold_spray_thermal.csv
        cold_spray_thermal.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import jaxdem as jdem


# ============================================================
# Physical parameters
# ============================================================

# Copper particle
DIAMETER = 20.0e-6          # m
RADIUS = DIAMETER / 2.0
DENSITY = 8960.0            # kg/m^3
CP = 385.0                  # J/(kg K)

INITIAL_TEMPERATURE = 300.0 # K

# Cold-spray impact velocity
IMPACT_VELOCITY = 600.0     # m/s

# ============================================================
# Reduced-order contact parameters
# ============================================================

# Effective normal stiffness.
#
# This is deliberately an effective reduced-order stiffness rather
# than a literal Young's modulus. It is chosen together with dt
# to resolve the impact transient.
K_CONTACT = 5.0e8           # N/m

# Damping ratio.
#
# zeta > 1 produces strongly damped impact and therefore deposition.
# Change this to explore rebound/deposition.
DAMPING_RATIO = 1.10

# Dashpot coefficient:
#
# c = 2*zeta*sqrt(k*m)
#
# calculated below after mass is known.

# Fraction of dissipated mechanical energy assigned to the particle.
#
# 1.0 = all contact dissipation becomes particle heat.
#
# For a more realistic particle/substrate thermal partition this
# should eventually be replaced with an effusivity-based partition.
HEAT_FRACTION_PARTICLE = 1.0


# ============================================================
# Numerical parameters
# ============================================================

# The particle mass is ~3.76e-12 kg.
#
# With k = 5e8 N/m the contact frequency is very high, so SI
# cold-spray simulations require a very small timestep.
DT = 2.0e-12               # seconds

TOTAL_TIME = 20.0e-9       # 20 ns

SAVE_EVERY = 10

N_STEPS = int(TOTAL_TIME / DT)
N_FRAMES = N_STEPS // SAVE_EVERY


# ============================================================
# Particle properties
# ============================================================

volume = (4.0 / 3.0) * np.pi * RADIUS**3
mass = DENSITY * volume

initial_ke = 0.5 * mass * IMPACT_VELOCITY**2

omega = np.sqrt(K_CONTACT / mass)
contact_period = 2.0 * np.pi / omega

dashpot_c = (
    2.0
    * DAMPING_RATIO
    * np.sqrt(K_CONTACT * mass)
)


print()
print("Reduced-order cold-spray impact")
print("--------------------------------")
print(f"Particle diameter : {DIAMETER * 1e6:.2f} um")
print(f"Particle mass     : {mass:.6e} kg")
print(f"Impact velocity   : {IMPACT_VELOCITY:.1f} m/s")
print(f"Initial KE        : {initial_ke:.6e} J")
print(f"Contact stiffness : {K_CONTACT:.3e} N/m")
print(f"Damping ratio     : {DAMPING_RATIO:.3f}")
print(f"Dashpot c         : {dashpot_c:.6e} N s/m")
print(f"Contact period    : {contact_period * 1e9:.4f} ns")
print(f"Time step         : {DT * 1e12:.4f} ps")
print(f"Number of steps   : {N_STEPS}")
print()


# ============================================================
# Initial state
# ============================================================

# Substrate is z = 0.
#
# Particle starts sufficiently far above it that:
#
#     z - R > 0
#
# and travels downward.

initial_height = 2.0 * RADIUS + IMPACT_VELOCITY * 5.0e-9

state = jdem.State.create(
    pos=jnp.array([
        [0.0, 0.0, initial_height],
    ]),
    vel=jnp.array([
        [0.0, 0.0, -IMPACT_VELOCITY],
    ]),
    rad=jnp.array([RADIUS]),
    mass=jnp.array([mass]),
)


# ============================================================
# Flat substrate force
# ============================================================

def substrate_force(
    pos: jax.Array,
    state: jdem.State,
    system: jdem.System,
):
    """
    Reduced-order rigid substrate.

    The substrate occupies z <= 0.

    Particle penetration:

        delta = R - z

    Contact force:

        F = k*delta - c*v_n

    where v_n is the particle velocity in the substrate-normal
    direction.

    The force is unilateral: the substrate cannot pull the particle.
    """

    del system

    k = K_CONTACT
    c = dashpot_c

    # Upward substrate normal.
    normal = jnp.array([0.0, 0.0, 1.0])

    # Surface gap measured from the particle surface.
    #
    # gap > 0 : particle above substrate
    # gap = 0 : touching
    # gap < 0 : penetration
    gap = pos[:, 2] - state.rad

    penetration = jnp.maximum(-gap, 0.0)

    # Normal velocity.
    vn = jnp.sum(state.vel * normal, axis=-1)

    # Kelvin-Voigt contact.
    #
    # During approach vn < 0, so -c*vn is positive.
    normal_force = (
        k * penetration
        - c * vn
    )

    # Substrate can push but cannot pull.
    normal_force = jnp.maximum(normal_force, 0.0)

    force = normal_force[:, None] * normal[None, :]

    torque = jnp.zeros_like(state.torque)

    return force, torque


def substrate_energy(
    pos: jax.Array,
    state: jdem.State,
    system: jdem.System,
):
    """
    Elastic energy stored temporarily in the substrate contact spring.
    """

    del system

    penetration = jnp.maximum(
        state.rad - pos[:, 2],
        0.0,
    )

    return 0.5 * K_CONTACT * penetration**2


# ============================================================
# System
# ============================================================

system = jdem.System.create(
    state=state,
    dt=DT,

    # We only have one particle, so naive collision detection is fine.
    collider_type="naive",

    domain_type="free",

    force_model_type="spring",

    force_manager_kw={
        "force_functions": [
            (
                substrate_force,
                substrate_energy,
            ),
        ],
    },
)


# ============================================================
# Save only quantities needed for analysis.
#
# trajectory_rollout uses lax.scan internally, so the expensive
# simulation remains JIT compiled.
# ============================================================

def save_frame(
    state: jdem.State,
    system: jdem.System,
):
    return (
        state.pos[0],
        state.vel[0],
        system.time,
    )


# Initial frame
initial_frame = save_frame(state, system)

# Run
final_state, final_system, trajectory = system.trajectory_rollout(
    state,
    system,
    n=N_FRAMES,
    stride=SAVE_EVERY,
    save_fn=save_frame,
)


positions = trajectory[0]
velocities = trajectory[1]
times = trajectory[2]

# Include t = 0.
times = jnp.concatenate([
    jnp.asarray([0.0]),
    times,
])

positions = jnp.concatenate([
    initial_frame[0][None, :],
    positions,
])

velocities = jnp.concatenate([
    initial_frame[1][None, :],
    velocities,
])


# ============================================================
# Thermal / energy post-processing
# ============================================================

# Translational kinetic energy.
speed_squared = jnp.sum(
    velocities**2,
    axis=-1,
)

ke = 0.5 * mass * speed_squared


# ------------------------------------------------------------
# Contact penetration
# ------------------------------------------------------------

penetration = jnp.maximum(
    RADIUS - positions[:, 2],
    0.0,
)


# Elastic energy temporarily stored in contact.
elastic_energy = (
    0.5
    * K_CONTACT
    * penetration**2
)


# ------------------------------------------------------------
# Dashpot dissipation
# ------------------------------------------------------------
#
# Dashpot force magnitude:
#
#     F_d = c*v_n
#
# Dissipated power:
#
#     P_heat = c*v_n^2
#
# Only count this while contact exists.

vn = velocities[:, 2]

contact_mask = penetration > 0.0

dissipation_power = (
    dashpot_c
    * vn**2
    * contact_mask
)


# Numerical integration of dissipated power.
#
# cumulative trapezoidal integration.
dt_frames = times[1:] - times[:-1]

heat_increment = (
    0.5
    * (
        dissipation_power[1:]
        + dissipation_power[:-1]
    )
    * dt_frames
)

cumulative_dissipated_energy = jnp.concatenate([
    jnp.asarray([0.0]),
    jnp.cumsum(heat_increment),
])


# Heat assigned to particle.
particle_heat = (
    HEAT_FRACTION_PARTICLE
    * cumulative_dissipated_energy
)


# ------------------------------------------------------------
# Particle temperature
# ------------------------------------------------------------

temperature = (
    INITIAL_TEMPERATURE
    + particle_heat / (mass * CP)
)


# ------------------------------------------------------------
# Energy accounting
# ------------------------------------------------------------
#
# Initial KE should approximately equal:
#
#     current KE
#   + elastic contact energy
#   + dissipated heat
#
# after the particle has finished impacting.

energy_residual = (
    initial_ke
    - ke
    - elastic_energy
    - cumulative_dissipated_energy
)


# ============================================================
# Convert to NumPy for output
# ============================================================

t_np = np.asarray(times)
z_np = np.asarray(positions[:, 2])
v_np = np.asarray(velocities[:, 2])

ke_np = np.asarray(ke)
elastic_np = np.asarray(elastic_energy)
heat_np = np.asarray(cumulative_dissipated_energy)
temp_np = np.asarray(temperature)
residual_np = np.asarray(energy_residual)


# ============================================================
# Determine deposition
# ============================================================

final_velocity = abs(v_np[-1])

if final_velocity < 0.01 * IMPACT_VELOCITY:
    outcome = "DEPOSITED"
else:
    outcome = "REBOUND"


# ============================================================
# Output directory
# ============================================================

output = Path("cold_spray_output")
output.mkdir(exist_ok=True)


# ============================================================
# CSV
# ============================================================

csv_data = np.column_stack([
    t_np,
    z_np,
    v_np,
    ke_np,
    elastic_np,
    heat_np,
    temp_np,
    residual_np,
])

header = (
    "time_s,"
    "z_m,"
    "vz_m_per_s,"
    "kinetic_energy_J,"
    "elastic_contact_energy_J,"
    "dissipated_heat_J,"
    "particle_temperature_K,"
    "energy_residual_J"
)

np.savetxt(
    output / "cold_spray_thermal.csv",
    csv_data,
    delimiter=",",
    header=header,
    comments="",
)


# ============================================================
# Print result
# ============================================================

final_ke = ke_np[-1]
final_heat = heat_np[-1]
final_temperature = temp_np[-1]

print()
print("RESULT")
print("------")
print(f"Outcome                : {outcome}")
print(f"Final |velocity|       : {final_velocity:.4f} m/s")
print(f"Initial kinetic energy : {initial_ke:.6e} J")
print(f"Final kinetic energy   : {final_ke:.6e} J")
print(f"Dissipated heat        : {final_heat:.6e} J")
print(f"Final particle temp.   : {final_temperature:.2f} K")
print(
    f"Energy residual        : "
    f"{residual_np[-1]:.6e} J"
)
print()
print(f"CSV: {output / 'cold_spray_thermal.csv'}")


# ============================================================
# Plot
# ============================================================

time_ns = t_np * 1e9

fig, axes = plt.subplots(
    3,
    1,
    figsize=(8, 9),
    sharex=True,
)

# ------------------------------------------------------------
# Velocity
# ------------------------------------------------------------

axes[0].plot(
    time_ns,
    v_np,
    color="tab:blue",
    linewidth=2,
)

axes[0].axhline(
    0.0,
    color="black",
    linewidth=0.8,
)

axes[0].set_ylabel("Velocity (m/s)")
axes[0].set_title(
    f"Reduced-order cold spray: {IMPACT_VELOCITY:.0f} m/s Cu impact"
)
axes[0].grid(alpha=0.25)


# ------------------------------------------------------------
# Energy
# ------------------------------------------------------------

axes[1].plot(
    time_ns,
    ke_np * 1e6,
    label="Kinetic energy",
    linewidth=2,
)

axes[1].plot(
    time_ns,
    heat_np * 1e6,
    label="Dissipated heat",
    linewidth=2,
)

axes[1].plot(
    time_ns,
    elastic_np * 1e6,
    label="Elastic contact energy",
    linewidth=1.5,
)

axes[1].set_ylabel("Energy (µJ)")
axes[1].legend()
axes[1].grid(alpha=0.25)


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

axes[2].plot(
    time_ns,
    temp_np,
    color="tab:red",
    linewidth=2.5,
)

axes[2].axhline(
    INITIAL_TEMPERATURE,
    color="black",
    linestyle="--",
    linewidth=1,
)

axes[2].set_xlabel("Time (ns)")
axes[2].set_ylabel("Particle temperature (K)")
axes[2].grid(alpha=0.25)


fig.tight_layout()

fig.savefig(
    output / "cold_spray_thermal.png",
    dpi=180,
)

plt.close(fig)

print(f"Plot: {output / 'cold_spray_thermal.png'}")
