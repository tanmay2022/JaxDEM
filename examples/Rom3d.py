#!/usr/bin/env python3
"""
3-D reduced-order cold-spray impact example based on JaxDEM.

Physics:
    - one spherical particle
    - rigid flat substrate z = 0
    - nonlinear/capped elastic normal contact
    - viscous contact damping
    - no gravity
    - no thermal model
    - no substrate deformation
    - no adhesion

Outputs:
    output/cold_spray_rom/
        particle_000000.vtp
        particle_000001.vtp
        ...
        substrate.vtp
        cold_spray.pvd
        energy.csv

Open cold_spray.pvd in ParaView.

Dependencies:
    pip install jax jaxdem
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxdem as jdem


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    # Particle: 20 um diameter Al sphere
    diameter: float = 20.0e-6
    density: float = 2700.0

    # Cold-spray impact velocity
    impact_velocity: float = 700.0

    # Initial particle center height
    # A few radii above the substrate.
    initial_gap: float = 4.0

    # Contact model
    #
    # kn is intentionally an effective ROM parameter, not an Al Young's
    # modulus converted directly into a Hertz contact.
    kn: float = 1.0e8

    # Yield force for the capped elastic contact.
    #
    # This gives:
    #     F_el = min(kn * delta, yield_force)
    #
    # It is best interpreted as an effective reduced-order contact strength.
    yield_force: float = 2.0e-5

    # Contact damping.
    damping: float = 1.0e-6

    # Time integration
    dt: float = 1.0e-11
    n_steps: int = 8000

    # VTK output interval
    output_every: int = 40

    # Deposition criterion
    deposited_velocity_fraction: float = 0.02

    # Output directory
    output_dir: str = "output/cold_spray_rom"


P = Params()


# ---------------------------------------------------------------------------
# Derived particle quantities
# ---------------------------------------------------------------------------

RADIUS = 0.5 * P.diameter
VOLUME = (4.0 / 3.0) * math.pi * RADIUS**3
MASS = P.density * VOLUME

# Initial kinetic energy
IMPACT_ENERGY = 0.5 * MASS * P.impact_velocity**2

# Elastic yield overlap
YIELD_OVERLAP = P.yield_force / P.kn


# ---------------------------------------------------------------------------
# Custom JaxDEM collider
# ---------------------------------------------------------------------------
#
# The JaxDEM collider API allows a custom module to implement contact
# detection and force/energy calculation.
#
# Here the "substrate" is analytically represented by z = 0 rather than
# by thousands of DEM particles.
# ---------------------------------------------------------------------------

@jdem.Collider.register("cold_spray_plane")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ColdSprayPlaneCollider(jdem.Collider):
    radius: float = RADIUS
    kn: float = P.kn
    yield_force: float = P.yield_force
    damping: float = P.damping

    @staticmethod
    @jax.jit(inline=True)
    def compute_force(
        state: jdem.State,
        system: jdem.System,
    ) -> tuple[jdem.State, jdem.System]:
        """
        Compute normal particle/rigid-plane contact.

        Plane:
            z = 0

        Particle:
            contact if z_center < radius

        Force:
            F = [kn * delta capped at yield_force]
                + damping contribution

        where:
            delta = radius - z_center
        """
        model = system.collider

        z = state.pos[:, 2]
        vz = state.vel[:, 2]

        delta = jnp.maximum(model.radius - z, 0.0)

        # Capped elastic force.
        elastic_force = jnp.minimum(
            model.kn * delta,
            model.yield_force,
        )

        # Damping only opposes penetration.
        #
        # Downward velocity => vz < 0.
        # Therefore -vz is positive during compression.
        damping_force = model.damping * jnp.maximum(-vz, 0.0)

        normal_force = elastic_force + damping_force

        active = (delta > 0.0).astype(state.pos.dtype)

        force = jnp.zeros_like(state.force)
        force = force.at[:, 2].set(
            active * normal_force
        )

        state.force = force
        state.torque = jnp.zeros_like(state.torque)

        return state, system

    @staticmethod
    @jax.jit(inline=True)
    def compute_potential_energy(
        state: jdem.State,
        system: jdem.System,
    ) -> tuple[jdem.State, jdem.System, jax.Array]:
        """
        Elastic contact energy.

        For delta <= delta_y:
            U = 1/2 kn delta^2

        For delta > delta_y:
            U = 1/2 kn delta_y^2
                + Fy (delta - delta_y)

        This corresponds to the capped-force reduced-order contact law.
        """
        model = system.collider

        z = state.pos[:, 2]
        delta = jnp.maximum(model.radius - z, 0.0)

        delta_y = model.yield_force / model.kn

        elastic_energy = jnp.where(
            delta <= delta_y,
            0.5 * model.kn * delta**2,
            (
                0.5 * model.kn * delta_y**2
                + model.yield_force * (delta - delta_y)
            ),
        )

        return state, system, jnp.sum(elastic_energy)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    def create_neighbor_list(
        state: jdem.State,
        system: jdem.System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jdem.State, jdem.System, jax.Array, jax.Array]:
        """
        This example has no particle-particle neighbors.

        The substrate is analytical, so the neighbor list is empty.
        """
        del cutoff

        nl = -jnp.ones(
            (state.N, max_neighbors),
            dtype=int,
        )

        overflow = jnp.asarray(False)

        return state, system, nl, overflow


# ---------------------------------------------------------------------------
# Pure energy helpers
# ---------------------------------------------------------------------------

def kinetic_energy(state: jdem.State) -> jax.Array:
    """Particle translational kinetic energy."""
    return 0.5 * jnp.sum(
        state.mass * jnp.sum(state.vel**2, axis=-1)
    )


def contact_energy(
    state: jdem.State,
    system: jdem.System,
) -> jax.Array:
    """Get contact potential energy from JaxDEM."""
    _, _, pe = system.collider.compute_potential_energy(
        state,
        system,
    )
    return pe


def contact_force_and_overlap(
    state: jdem.State,
    system: jdem.System,
) -> tuple[jax.Array, jax.Array]:
    """Return scalar normal contact force and overlap."""
    z = state.pos[0, 2]
    vz = state.vel[0, 2]

    delta = jnp.maximum(
        system.collider.radius - z,
        0.0,
    )

    elastic_force = jnp.minimum(
        system.collider.kn * delta,
        system.collider.yield_force,
    )

    damping_force = (
        system.collider.damping
        * jnp.maximum(-vz, 0.0)
    )

    force = elastic_force + damping_force

    return force, delta


# ---------------------------------------------------------------------------
# VTK writer
# ---------------------------------------------------------------------------

def _xml_data_array(
    name: str,
    values,
    vtk_type: str = "Float64",
    n_components: int | None = None,
) -> str:
    """
    Create an ASCII VTK XML DataArray.
    """
    arr = jnp.asarray(values)
    arr = jax.device_get(arr)

    if hasattr(arr, "reshape"):
        arr = arr.reshape(-1)

    text = " ".join(
        f"{float(x):.10e}"
        for x in arr
    )

    components = ""
    if n_components is not None:
        components = f' NumberOfComponents="{n_components}"'

    return (
        f'        <DataArray type="{vtk_type}" '
        f'Name="{name}" format="ascii"{components}>\n'
        f"          {text}\n"
        f"        </DataArray>\n"
    )


def write_particle_vtp(
    filename: Path,
    state: jdem.State,
    *,
    kinetic: float,
    contact: float,
    dissipated: float,
    total: float,
    contact_force: float,
    overlap: float,
    deposited: int,
):
    """
    Write one particle as a VTK PolyData vertex.

    ParaView can then apply:
        Filters -> Glyph
    and choose Sphere to visualize the particle radius.
    """
    pos = jax.device_get(state.pos[0])
    vel = jax.device_get(state.vel[0])

    speed = float(jnp.linalg.norm(state.vel[0]))

    radius = float(state.rad[0])

    xml = f"""<?xml version="1.0"?>
<VTKFile type="PolyData"
         version="0.1"
         byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="1"
           NumberOfVerts="1"
           NumberOfLines="0"
           NumberOfStrips="0"
           NumberOfPolys="0">

      <PointData Scalars="kinetic_energy">

{_xml_data_array("kinetic_energy", [kinetic])}
{_xml_data_array("contact_energy", [contact])}
{_xml_data_array("dissipated_energy", [dissipated])}
{_xml_data_array("total_energy", [total])}
{_xml_data_array("contact_force", [contact_force])}
{_xml_data_array("overlap", [overlap])}
{_xml_data_array("speed", [speed])}
{_xml_data_array("deposited", [deposited], vtk_type="Int32")}
{_xml_data_array("radius", [radius])}
{_xml_data_array("velocity", vel, n_components=3)}

      </PointData>

      <Points>
{_xml_data_array("Points", pos, n_components=3)}
      </Points>

      <Verts>
        <DataArray type="Int32"
                   Name="connectivity"
                   format="ascii">
          0
        </DataArray>
        <DataArray type="Int32"
                   Name="offsets"
                   format="ascii">
          1
        </DataArray>
      </Verts>

    </Piece>
  </PolyData>
</VTKFile>
"""

    filename.write_text(xml)


def write_substrate_vtp(
    filename: Path,
    half_width: float,
):
    """
    Write a large rectangular substrate as one quad.

    This is visualization only. The actual contact surface in the
    dynamics is the analytical plane z=0.
    """
    vertices = [
        [-half_width, -half_width, 0.0],
        [ half_width, -half_width, 0.0],
        [ half_width,  half_width, 0.0],
        [-half_width,  half_width, 0.0],
    ]

    flat_vertices = [
        x
        for vertex in vertices
        for x in vertex
    ]

    xml = f"""<?xml version="1.0"?>
<VTKFile type="PolyData"
         version="0.1"
         byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="4"
           NumberOfVerts="0"
           NumberOfLines="0"
           NumberOfStrips="0"
           NumberOfPolys="1">

      <Points>
{_xml_data_array("Points", flat_vertices, n_components=3)}
      </Points>

      <Polys>
        <DataArray type="Int32"
                   Name="connectivity"
                   format="ascii">
          0 1 2 3
        </DataArray>
        <DataArray type="Int32"
                   Name="offsets"
                   format="ascii">
          4
        </DataArray>
      </Polys>

    </Piece>
  </PolyData>
</VTKFile>
"""

    filename.write_text(xml)


def write_pvd(
    filename: Path,
    particle_files: list[tuple[float, str]],
):
    """
    Write a ParaView time-series collection.

    Each timestep contains:
        particle VTP
        substrate VTP
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]

    for timestep, particle_file in particle_files:
        lines.append(
            f'    <DataSet timestep="{timestep:.12e}" '
            f'group="" part="0" file="{particle_file}"/>'
        )

        lines.append(
            f'    <DataSet timestep="{timestep:.12e}" '
            f'group="substrate" part="0" file="substrate.vtp"/>'
        )

    lines.extend(
        [
            "  </Collection>",
            "</VTKFile>",
        ]
    )

    filename.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    out = Path(P.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Remove old frame files.
    for path in out.glob("particle_*.vtp"):
        path.unlink()

    for path in out.glob("*.csv"):
        path.unlink()

    pvd_path = out / "cold_spray.pvd"

    if pvd_path.exists():
        pvd_path.unlink()

    # -----------------------------------------------------------------------
    # Initial particle state
    # -----------------------------------------------------------------------

    initial_z = RADIUS + P.initial_gap * RADIUS

    state = jdem.State.create(
        pos=jnp.array(
            [[0.0, 0.0, initial_z]],
            dtype=jnp.float64,
        ),
        vel=jnp.array(
            [[0.0, 0.0, -P.impact_velocity]],
            dtype=jnp.float64,
        ),
        rad=jnp.array(
            [RADIUS],
            dtype=jnp.float64,
        ),
        mass=jnp.array(
            [MASS],
            dtype=jnp.float64,
        ),
        fixed=jnp.array(
            [False],
        ),
    )

    # -----------------------------------------------------------------------
    # JaxDEM system
    # -----------------------------------------------------------------------

    collider = ColdSprayPlaneCollider(
        radius=RADIUS,
        kn=P.kn,
        yield_force=P.yield_force,
        damping=P.damping,
    )

    system = jdem.System.create(
        state=state,
        dt=P.dt,
        collider=collider,

        # We don't use JaxDEM's particle-particle force model because
        # the substrate is represented analytically by our collider.
        force_model_type="",

        # Translational motion is what matters here.
        rotation_integrator_type="",

        # Free space domain.
        domain_type="free",
    )

    # -----------------------------------------------------------------------
    # Initial energies
    # -----------------------------------------------------------------------

    initial_kinetic = float(kinetic_energy(state))

    print("=" * 72)
    print("3-D REDUCED-ORDER COLD-SPRAY IMPACT")
    print("=" * 72)
    print(f"Particle diameter : {P.diameter * 1e6:.2f} um")
    print(f"Particle density  : {P.density:.1f} kg/m^3")
    print(f"Particle mass     : {MASS:.6e} kg")
    print(f"Impact velocity   : {P.impact_velocity:.1f} m/s")
    print(f"Impact energy     : {IMPACT_ENERGY:.6e} J")
    print(f"Contact stiffness : {P.kn:.6e} N/m")
    print(f"Yield force       : {P.yield_force:.6e} N")
    print(f"Yield overlap     : {YIELD_OVERLAP * 1e6:.4f} um")
    print(f"Damping           : {P.damping:.6e} N s/m")
    print(f"dt                : {P.dt:.3e} s")
    print(f"Number of steps   : {P.n_steps}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Static substrate visualization
    # -----------------------------------------------------------------------

    substrate_width = 10.0 * P.diameter

    write_substrate_vtp(
        out / "substrate.vtp",
        half_width=substrate_width,
    )

    # -----------------------------------------------------------------------
    # CSV energy history
    # -----------------------------------------------------------------------

    energy_csv = out / "energy.csv"

    particle_frames: list[tuple[float, str]] = []

    dissipated_energy = 0.0

    previous_state = state

    with energy_csv.open("w", newline="") as f:
        csv_writer = csv.writer(f)

        csv_writer.writerow(
            [
                "step",
                "time_s",
                "z_m",
                "vz_m_s",
                "speed_m_s",
                "kinetic_J",
                "contact_J",
                "dissipated_J",
                "total_J",
                "energy_error_J",
                "contact_force_N",
                "overlap_m",
                "deposited",
            ]
        )

        # -------------------------------------------------------------------
        # Save initial frame
        # -------------------------------------------------------------------

        ke = float(kinetic_energy(state))
        pe = float(contact_energy(state, system))
        force, overlap = contact_force_and_overlap(state, system)

        force = float(force)
        overlap = float(overlap)

        total = ke + pe + dissipated_energy

        frame_name = "particle_000000.vtp"

        write_particle_vtp(
            out / frame_name,
            state,
            kinetic=ke,
            contact=pe,
            dissipated=dissipated_energy,
            total=total,
            contact_force=force,
            overlap=overlap,
            deposited=0,
        )

        particle_frames.append(
            (0.0, frame_name)
        )

        csv_writer.writerow(
            [
                0,
                0.0,
                float(state.pos[0, 2]),
                float(state.vel[0, 2]),
                float(jnp.linalg.norm(state.vel[0])),
                ke,
                pe,
                dissipated_energy,
                total,
                total - initial_kinetic,
                force,
                overlap,
                0,
            ]
        )

        # -------------------------------------------------------------------
        # Time integration
        # -------------------------------------------------------------------

        for step in range(1, P.n_steps + 1):

            # ---------------------------------------------------------------
            # Advance using JaxDEM
            # ---------------------------------------------------------------

            state, system = jdem.System.step(
                state,
                system,
                n=1,
            )

            # ---------------------------------------------------------------
            # Contact damping dissipation
            # ---------------------------------------------------------------
            #
            # P_diss = c * vn^2
            #
            # Only active during penetration/compression.
            #
            # Trapezoidal integration is more accurate than simply using
            # the final value.
            # ---------------------------------------------------------------

            old_z = float(previous_state.pos[0, 2])
            old_vz = float(previous_state.vel[0, 2])

            old_delta = max(
                RADIUS - old_z,
                0.0,
            )

            old_compression_speed = max(
                -old_vz,
                0.0,
            )

            new_z = float(state.pos[0, 2])
            new_vz = float(state.vel[0, 2])

            new_delta = max(
                RADIUS - new_z,
                0.0,
            )

            new_compression_speed = max(
                -new_vz,
                0.0,
            )

            old_power = (
                P.damping
                * old_compression_speed**2
                * float(old_delta > 0.0)
            )

            new_power = (
                P.damping
                * new_compression_speed**2
                * float(new_delta > 0.0)
            )

            dissipated_energy += (
                0.5
                * (old_power + new_power)
                * P.dt
            )

            previous_state = state

            # ---------------------------------------------------------------
            # Diagnostics
            # ---------------------------------------------------------------

            ke = float(kinetic_energy(state))
            pe = float(contact_energy(state, system))

            force, overlap = contact_force_and_overlap(
                state,
                system,
            )

            force = float(force)
            overlap = float(overlap)

            total = (
                ke
                + pe
                + dissipated_energy
            )

            energy_error = (
                total
                - initial_kinetic
            )

            speed = float(
                jnp.linalg.norm(state.vel[0])
            )

            # ---------------------------------------------------------------
            # Deposition criterion
            # ---------------------------------------------------------------
            #
            # This is deliberately a ROM classification criterion, not a
            # physical adhesion model.
            # ---------------------------------------------------------------

            deposited = int(
                (
                    abs(new_vz)
                    < P.deposited_velocity_fraction
                    * P.impact_velocity
                )
                and (
                    abs(overlap)
                    < 0.25 * RADIUS
                )
            )

            # ---------------------------------------------------------------
            # VTK output
            # ---------------------------------------------------------------

            if (
                step % P.output_every == 0
                or step == P.n_steps
            ):
                frame_number = (
                    step // P.output_every
                    if step != P.n_steps
                    else math.ceil(P.n_steps / P.output_every)
                )

                frame_name = (
                    f"particle_{frame_number:06d}.vtp"
                )

                write_particle_vtp(
                    out / frame_name,
                    state,
                    kinetic=ke,
                    contact=pe,
                    dissipated=dissipated_energy,
                    total=total,
                    contact_force=force,
                    overlap=overlap,
                    deposited=deposited,
                )

                particle_frames.append(
                    (
                        step * P.dt,
                        frame_name,
                    )
                )

            # ---------------------------------------------------------------
            # CSV output
            # ---------------------------------------------------------------

            csv_writer.writerow(
                [
                    step,
                    step * P.dt,
                    new_z,
                    new_vz,
                    speed,
                    ke,
                    pe,
                    dissipated_energy,
                    total,
                    energy_error,
                    force,
                    overlap,
                    deposited,
                ]
            )

            # ---------------------------------------------------------------
            # Progress
            # ---------------------------------------------------------------

            if (
                step % (P.output_every * 10) == 0
                or step == 1
            ):
                print(
                    f"step={step:6d} "
                    f"t={step * P.dt * 1e9:9.3f} ns "
                    f"z={new_z * 1e6:9.4f} um "
                    f"vz={new_vz:10.3f} m/s "
                    f"F={force:10.3e} N "
                    f"KE={ke:10.3e} J "
                    f"Ediss={dissipated_energy:10.3e} J"
                )

    # -----------------------------------------------------------------------
    # PVD time series
    # -----------------------------------------------------------------------

    write_pvd(
        out / "cold_spray.pvd",
        particle_frames,
    )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    final_vz = float(state.vel[0, 2])
    final_speed = float(
        jnp.linalg.norm(state.vel[0])
    )

    final_ke = float(
        kinetic_energy(state)
    )

    final_pe = float(
        contact_energy(state, system)
    )

    final_total = (
        final_ke
        + final_pe
        + dissipated_energy
    )

    final_overlap = max(
        RADIUS - float(state.pos[0, 2]),
        0.0,
    )

    final_deposited = int(
        abs(final_vz)
        < P.deposited_velocity_fraction
        * P.impact_velocity
    )

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print(
        f"Final z             : "
        f"{float(state.pos[0, 2]) * 1e6:.6f} um"
    )
    print(
        f"Final normal speed  : "
        f"{final_vz:.6f} m/s"
    )
    print(
        f"Final speed         : "
        f"{final_speed:.6f} m/s"
    )
    print(
        f"Maximum/final overlap: "
        f"{final_overlap * 1e6:.6f} um"
    )
    print(
        f"Final kinetic energy: "
        f"{final_ke:.6e} J"
    )
    print(
        f"Final contact energy: "
        f"{final_pe:.6e} J"
    )
    print(
        f"Dissipated energy   : "
        f"{dissipated_energy:.6e} J"
    )
    print(
        f"Initial energy      : "
        f"{initial_kinetic:.6e} J"
    )
    print(
        f"Final energy        : "
        f"{final_total:.6e} J"
    )
    print(
        f"Energy error        : "
        f"{final_total - initial_kinetic:.6e} J"
    )
    print(
        f"Dissipation fraction: "
        f"{dissipated_energy / initial_kinetic:.4f}"
    )
    print(
        f"Deposited (ROM)     : "
        f"{'YES' if final_deposited else 'NO'}"
    )
    print()
    print(f"ParaView file: {out / 'cold_spray.pvd'}")
    print(f"Energy CSV   : {energy_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
