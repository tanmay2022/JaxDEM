"""
Generate an LS-DYNA .k model for:

    30 micron IN718 SPH particle
    impacting a deformable IN718 solid plate
    at 700 m/s.

Physics included:
    - SPH particle
    - deformable 8-node hex plate
    - Johnson-Cook plasticity
    - strain-rate sensitivity
    - thermal softening
    - adiabatic plastic heating
    - Gruneisen EOS
    - SPH-to-solid automatic contact
    - optional breakable tiebreak contact
    - plate boundary constraints
    - d3plot / history output

UNIT SYSTEM:
    mm - ms - kg - N - MPa-equivalent LS-DYNA internal units

IMPORTANT:
    In this unit system:
        velocity = mm/ms
        density  = kg/mm^3
        stress   = GPa numerically
        energy   = kJ/kg numerically

    Therefore:
        210 GPa -> E = 210
        1290 MPa -> A = 1.290
        density IN718 = 8.24e-6 kg/mm^3
        Cp = 0.435 kJ/(kg K)

The material parameters below are STARTING values.
They must be calibrated for your particular IN718 powder,
heat treatment, substrate condition and high-rate regime.
"""

from math import sqrt, ceil


# ==============================================================
# USER PARAMETERS
# ==============================================================

OUTPUT_FILE = "IN718_30um_SPH_700ms.k"

# --------------------------------------------------------------
# SPHERE
# --------------------------------------------------------------

SPHERE_DIAMETER = 0.030       # mm = 30 micron
SPHERE_RADIUS = SPHERE_DIAMETER / 2.0

SPH_SPACING = 0.003           # mm = 3 micron

IMPACT_VELOCITY = -700.0      # mm/ms = 700 m/s

# Initial gap between sphere and plate
INITIAL_GAP = 0.010           # mm = 10 micron

# Sphere center
SPHERE_X = 0.0
SPHERE_Y = 0.0
SPHERE_Z = SPHERE_RADIUS + INITIAL_GAP


# --------------------------------------------------------------
# PLATE
# --------------------------------------------------------------

# Local plate size.
#
# 0.30 x 0.30 mm is used so that the impact zone is sufficiently
# far from the lateral boundaries while keeping the model manageable.
#
# Increase this for a larger-domain study.

PLATE_X = 0.30                # mm
PLATE_Y = 0.30                # mm
PLATE_THICKNESS = 1.0         # mm

# Solid element size
PLATE_DX = 0.010              # mm = 10 micron
PLATE_DY = 0.010              # mm = 10 micron
PLATE_DZ = 0.010              # mm = 10 micron


# --------------------------------------------------------------
# INITIAL PLATE POSITION
# --------------------------------------------------------------

PLATE_Z_BOTTOM = -PLATE_THICKNESS
PLATE_Z_TOP = 0.0


# --------------------------------------------------------------
# SPH PARAMETERS
# --------------------------------------------------------------

# Ansys recommends CSLH values roughly 1.05-1.3;
# 1.2 is the standard starting value.

CSLH = 1.20

# Allow smoothing length to evolve moderately with deformation.
HMIN = 0.50
HMAX = 2.00


# --------------------------------------------------------------
# IN718 MATERIAL
# --------------------------------------------------------------

# Density
RHO_IN718 = 8.24e-6        # kg/mm^3

# Elastic properties
E_IN718 = 210.0             # GPa
NU_IN718 = 0.30

# Shear modulus
G_IN718 = E_IN718 / (2.0 * (1.0 + NU_IN718))


# --------------------------------------------------------------
# JOHNSON-COOK STARTING PARAMETERS
# --------------------------------------------------------------

# Literature-style starter values.
#
# A = 1290 MPa = 1.290 GPa
# B = 895 MPa  = 0.895 GPa
#
# These are NOT validated specifically for your 30 micron
# particle / 700 m/s impact.

JC_A = 1.290
JC_B = 0.895
JC_N = 0.526
JC_C = 0.016
JC_M = 1.55

# Temperatures
JC_TM = 1573.0              # K
JC_TR = 293.0               # K

# Reference strain rate
JC_EPS0 = 1.0               # 1/s

# Specific heat
#
# 435 J/kg/K = 0.435 kJ/kg/K
#
# With mm-ms-kg LS-DYNA units this is 0.435.

SPECIFIC_HEAT = 0.435       # kJ/(kg K)


# --------------------------------------------------------------
# GRUNEISEN EOS
# --------------------------------------------------------------

# Longitudinal sound speed.
#
# Approximate IN718 value:
#       c0 ~ 5700 m/s = 5.7 mm/ms
#
# These EOS values should ideally be replaced by
# experimentally calibrated shock data.

EOS_C0 = 5.70               # mm/ms
EOS_S1 = 1.50
EOS_S2 = 0.0
EOS_S3 = 0.0
EOS_GAMMA0 = 1.70
EOS_A = 0.0


# --------------------------------------------------------------
# CONTACT
# --------------------------------------------------------------

STATIC_FRICTION = 0.20
DYNAMIC_FRICTION = 0.15

# Contact damping
VDC = 20.0

# --------------------------------------------------------------
# OPTIONAL BREAKABLE BOND
# --------------------------------------------------------------

ENABLE_TIEBREAK = True

# These are PLACEHOLDER calibration values.
#
# Because LS-DYNA is using the mm-ms-kg system here,
# stress values in the keyword are expressed in GPa numerically.
#
# 0.050 GPa = 50 MPa
# 0.030 GPa = 30 MPa

BOND_NORMAL_STRESS = 0.050
BOND_SHEAR_STRESS = 0.030


# --------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------

END_TIME = 0.20             # ms

D3PLOT_INTERVAL = 0.005     # ms

GLSTAT_INTERVAL = 0.005
MATSUM_INTERVAL = 0.005
RCFORC_INTERVAL = 0.005
NODOUT_INTERVAL = 0.005


# ==============================================================
# ID RANGES
# ==============================================================

PART_SPH = 1
PART_PLATE = 2

SECTION_SPH = 1
SECTION_SOLID = 2

MAT_IN718 = 1
EOS_IN718 = 1

SPH_NODE_START = 1000000
PLATE_NODE_START = 2000000

SPH_ELEM_START = 1000000
PLATE_ELEM_START = 2000000

PLATE_BOTTOM_SET = 10
SPH_NODE_SET = 20
PLATE_TOP_SEGMENT_SET = 30


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def fmt(x):
    """Compact LS-DYNA floating-point representation."""
    return f"{x:.8E}"


def write_comment(f, text=""):
    f.write("$ " + text + "\n")


def write_card(f, *values):
    """
    Write a simple whitespace-separated LS-DYNA card.
    """
    f.write(" ".join(str(v) for v in values) + "\n")


# ==============================================================
# GENERATE SPHERE PARTICLES
# ==============================================================

def generate_sphere_particles():
    """
    Generate particles on a Cartesian lattice.

    Particle centers satisfying:

        r <= SPHERE_RADIUS

    are retained.

    The center is shifted to the requested sphere position.
    """

    particles = []

    n = int(ceil(SPHERE_RADIUS / SPH_SPACING))

    for k in range(-n, n + 1):

        z = k * SPH_SPACING

        for j in range(-n, n + 1):

            y = j * SPH_SPACING

            for i in range(-n, n + 1):

                x = i * SPH_SPACING

                r = sqrt(x*x + y*y + z*z)

                if r <= SPHERE_RADIUS + 1.0e-12:

                    particles.append(
                        (
                            SPHERE_X + x,
                            SPHERE_Y + y,
                            SPHERE_Z + z
                        )
                    )

    return particles


# ==============================================================
# PARTICLE MASS
# ==============================================================

def calculate_particle_mass():

    """
    Approximate particle mass using:

        m = rho * spacing^3

    This is a standard simple initialization for a uniformly
    sampled SPH particle distribution.

    For a production model, mass conservation should be checked
    against the exact analytical sphere mass.
    """

    particle_volume = SPH_SPACING ** 3

    return RHO_IN718 * particle_volume


# ==============================================================
# PLATE MESH
# ==============================================================

def generate_plate_mesh():

    """
    Generate structured 8-node hexahedral mesh.

    Node ordering:

        1 ---- 2
       /|     /|
      4 ---- 3 |
      | 5 ---|- 6
      |/      |/
      8 ---- 7

    The actual LS-DYNA hexahedral connectivity is:

        n1 n2 n3 n4 n5 n6 n7 n8

    """

    nx = int(round(PLATE_X / PLATE_DX))
    ny = int(round(PLATE_Y / PLATE_DY))
    nz = int(round(PLATE_THICKNESS / PLATE_DZ))

    dx = PLATE_X / nx
    dy = PLATE_Y / ny
    dz = PLATE_THICKNESS / nz

    nodes = {}
    node_list = []

    def nid(i, j, k):
        return (
            PLATE_NODE_START
            + k * (nx + 1) * (ny + 1)
            + j * (nx + 1)
            + i
        )

    # ----------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------

    for k in range(nz + 1):

        z = PLATE_Z_BOTTOM + k * dz

        for j in range(ny + 1):

            y = -PLATE_Y / 2.0 + j * dy

            for i in range(nx + 1):

                x = -PLATE_X / 2.0 + i * dx

                node_id = nid(i, j, k)

                nodes[node_id] = (x, y, z)

                node_list.append(node_id)

    # ----------------------------------------------------------
    # Hex elements
    # ----------------------------------------------------------

    elements = []

    eid = PLATE_ELEM_START

    for k in range(nz):

        for j in range(ny):

            for i in range(nx):

                n1 = nid(i,     j,     k)
                n2 = nid(i + 1, j,     k)
                n3 = nid(i + 1, j + 1, k)
                n4 = nid(i,     j + 1, k)

                n5 = nid(i,     j,     k + 1)
                n6 = nid(i + 1, j,     k + 1)
                n7 = nid(i + 1, j + 1, k + 1)
                n8 = nid(i,     j + 1, k + 1)

                elements.append(
                    (
                        eid,
                        n1, n2, n3, n4,
                        n5, n6, n7, n8
                    )
                )

                eid += 1

    # ----------------------------------------------------------
    # Bottom nodes
    # ----------------------------------------------------------

    bottom_nodes = []

    for j in range(ny + 1):

        for i in range(nx + 1):

            bottom_nodes.append(nid(i, j, 0))

    # ----------------------------------------------------------
    # Top nodes
    # ----------------------------------------------------------

    top_nodes = []

    for j in range(ny + 1):

        for i in range(nx + 1):

            top_nodes.append(nid(i, j, nz))

    return nodes, elements, bottom_nodes, top_nodes


# ==============================================================
# WRITE KEYWORD FILE
# ==============================================================

def write_keyword_file():

    print("Generating SPH particle distribution...")

    particles = generate_sphere_particles()

    print(f"SPH particles: {len(particles)}")

    particle_mass = calculate_particle_mass()

    print(f"SPH particle mass: {particle_mass:.6E} kg")

    print("Generating plate mesh...")

    plate_nodes, plate_elements, bottom_nodes, top_nodes = \
        generate_plate_mesh()

    print(f"Plate nodes:    {len(plate_nodes)}")
    print(f"Plate elements: {len(plate_elements)}")

    # ----------------------------------------------------------
    # Open file
    # ----------------------------------------------------------

    with open(OUTPUT_FILE, "w") as f:

        # ======================================================
        # HEADER
        # ======================================================

        f.write("*KEYWORD\n")

        f.write("*TITLE\n")
        f.write("30um IN718 SPH particle impact on deformable IN718 plate\n")

        write_comment(f)
        write_comment(f, "============================================================")
        write_comment(f, "30 micron IN718 SPH particle")
        write_comment(f, "700 m/s impact velocity")
        write_comment(f, "1 mm IN718 deformable solid plate")
        write_comment(f, "Thermo-viscoplastic Johnson-Cook material")
        write_comment(f, "Gruneisen EOS")
        write_comment(f, "============================================================")

        write_comment(f, "UNIT SYSTEM: mm-ms-kg")
        write_comment(f, "Velocity = mm/ms")
        write_comment(f, "Density = kg/mm^3")
        write_comment(f, "Stress = GPa numerically")
        write_comment(f, "Temperature = K")

        # ======================================================
        # CONTROL
        # ======================================================

        f.write("*CONTROL_TERMINATION\n")
        write_card(f, END_TIME)

        f.write("*CONTROL_TIMESTEP\n")
        write_card(f, "0.0", "0.70")

        f.write("*CONTROL_ENERGY\n")
        write_card(f, 1, 2, 1, 1)

        f.write("*CONTROL_CONTACT\n")
        write_card(f, 0.0, 0.0, 0, 0, 0, 0, 0, 0)

        # ------------------------------------------------------
        # SPH
        # ------------------------------------------------------

        f.write("*CONTROL_SPH\n")
        write_card(
            f,
            1,       # NCBS
            0,       # BOXID
            0.0,     # DT
            3,       # IDIM
            100,     # MEMORY
            0,       # FORM
            0.0,     # START
            1.0E20   # MAXV
        )

        # ======================================================
        # OUTPUT
        # ======================================================

        f.write("*DATABASE_BINARY_D3PLOT\n")
        write_card(f, D3PLOT_INTERVAL)

        f.write("*DATABASE_GLSTAT\n")
        write_card(f, GLSTAT_INTERVAL)

        f.write("*DATABASE_MATSUM\n")
        write_card(f, MATSUM_INTERVAL)

        f.write("*DATABASE_RCFORC\n")
        write_card(f, RCFORC_INTERVAL)

        f.write("*DATABASE_NODOUT\n")
        write_card(f, NODOUT_INTERVAL)

        # ======================================================
        # MATERIAL
        # ======================================================

        write_comment(f)
        write_comment(f, "IN718 JOHNSON-COOK MATERIAL")

        f.write("*MAT_JOHNSON_COOK\n")

        # Card 1
        write_card(
            f,
            MAT_IN718,
            fmt(RHO_IN718),
            fmt(G_IN718),
            fmt(E_IN718),
            fmt(NU_IN718),
            0.0,          # DTF
            0.0,          # VP
            0.0           # RATEOP
        )

        # Card 2
        write_card(
            f,
            fmt(JC_A),
            fmt(JC_B),
            fmt(JC_N),
            fmt(JC_C),
            fmt(JC_M),
            fmt(JC_TM),
            fmt(JC_TR),
            fmt(JC_EPS0)
        )

        # Card 3
        write_card(
            f,
            fmt(SPECIFIC_HEAT),
            0.0,          # PC
            2.0,          # SPALL
            0.0,          # IT
            0.0,          # D1
            0.0,          # D2
            0.0,          # D3
            0.0           # D4
        )

        # Card 4
        write_card(
            f,
            0.0,          # D5
            0.0           # C2/P
        )

        # ======================================================
        # EOS
        # ======================================================

        write_comment(f)
        write_comment(f, "GRUNEISEN EQUATION OF STATE")

        f.write("*EOS_GRUNEISEN\n")

        write_card(
            f,
            EOS_IN718,
            fmt(EOS_C0),
            fmt(EOS_S1),
            fmt(EOS_S2),
            fmt(EOS_S3),
            fmt(EOS_GAMMA0),
            fmt(EOS_A),
            0.0            # E0
        )

        write_card(
            f,
            1.0,           # V0
            0.0,           # unused
            0             # LCID
        )

        # ======================================================
        # SPH SECTION
        # ======================================================

        f.write("*SECTION_SPH\n")

        write_card(
            f,
            SECTION_SPH,
            fmt(CSLH),
            fmt(HMIN),
            fmt(HMAX),
            0.0,           # SPHINI
            1.0E20,        # DEATH
            0.0,           # START
            0             # SPHKERN
        )

        # ======================================================
        # SOLID SECTION
        # ======================================================

        f.write("*SECTION_SOLID\n")

        # ELFORM = 2: fully integrated constant stress solid
        write_card(
            f,
            SECTION_SOLID,
            2,
            0
        )

        # ======================================================
        # PARTS
        # ======================================================

        f.write("*PART\n")
        f.write("IN718 SPH PARTICLE\n")

        write_card(
            f,
            PART_SPH,
            MAT_IN718,
            SECTION_SPH
        )

        f.write("*PART\n")
        f.write("IN718 DEFORMABLE SOLID PLATE\n")

        write_card(
            f,
            PART_PLATE,
            MAT_IN718,
            SECTION_SOLID
        )

        # ======================================================
        # SPH NODES
        # ======================================================

        write_comment(f)
        write_comment(f, "SPH PARTICLE NODES")

        f.write("*NODE\n")

        sph_node_ids = []

        for i, xyz in enumerate(particles):

            node_id = SPH_NODE_START + i

            sph_node_ids.append(node_id)

            x, y, z = xyz

            write_card(
                f,
                node_id,
                fmt(x),
                fmt(y),
                fmt(z)
            )

        # ======================================================
        # PLATE NODES
        # ======================================================

        write_comment(f)
        write_comment(f, "PLATE NODES")

        f.write("*NODE\n")

        for node_id in sorted(plate_nodes):

            x, y, z = plate_nodes[node_id]

            write_card(
                f,
                node_id,
                fmt(x),
                fmt(y),
                fmt(z)
            )

        # ======================================================
        # SPH ELEMENTS
        # ======================================================

        write_comment(f)
        write_comment(f, "SPH ELEMENTS")

        f.write("*ELEMENT_SPH\n")

        for i, node_id in enumerate(sph_node_ids):

            eid = SPH_ELEM_START + i

            write_card(
                f,
                eid,
                PART_SPH,
                fmt(particle_mass)
            )

        # ======================================================
        # SOLID ELEMENTS
        # ======================================================

        write_comment(f)
        write_comment(f, "8-NODE HEXAHEDRAL SOLID ELEMENTS")

        f.write("*ELEMENT_SOLID\n")

        for element in plate_elements:

            eid, n1, n2, n3, n4, n5, n6, n7, n8 = element

            write_card(
                f,
                eid,
                PART_PLATE,
                n1, n2, n3, n4,
                n5, n6, n7, n8
            )

        # ======================================================
        # SPH NODE SET
        # ======================================================

        write_comment(f)
        write_comment(f, "SPH NODE SET")

        f.write("*SET_NODE_LIST\n")
        write_card(f, PLATE_BOTTOM_SET)

        for start in range(0, len(sph_node_ids), 8):

            chunk = sph_node_ids[start:start + 8]

            write_card(f, *chunk)

        # ======================================================
        # BOTTOM PLATE NODE SET
        # ======================================================

        write_comment(f)
        write_comment(f, "BOTTOM PLATE NODE SET")

        f.write("*SET_NODE_LIST\n")
        write_card(f, PLATE_BOTTOM_SET)

        for start in range(0, len(bottom_nodes), 8):

            chunk = bottom_nodes[start:start + 8]

            write_card(f, *chunk)

        # ======================================================
        # TOP PLATE NODE SET
        # ======================================================

        write_comment(f)
        write_comment(f, "TOP PLATE NODE SET")

        f.write("*SET_NODE_LIST\n")
        write_card(f, 21)

        for start in range(0, len(top_nodes), 8):

            chunk = top_nodes[start:start + 8]

            write_card(f, *chunk)

        # ======================================================
        # BOUNDARY CONDITIONS
        # ======================================================

        write_comment(f)
        write_comment(f, "FIX BOTTOM SURFACE OF PLATE")

        f.write("*BOUNDARY_SPC_SET\n")

        # NSID CID DOFX DOFY DOFZ DOFRX DOFRY DOFRZ
        write_card(
            f,
            PLATE_BOTTOM_SET,
            0,
            1,       # X
            1,       # Y
            1,       # Z
            0,
            0,
            0
        )

        # ======================================================
        # INITIAL SPHERE VELOCITY
        # ======================================================

        write_comment(f)
        write_comment(f, "700 m/s INITIAL SPHERE VELOCITY")

        f.write("*INITIAL_VELOCITY_GENERATION\n")

        # PART STYP OMEGA VX VY VZ
        write_card(
            f,
            PART_SPH,
            2,
            0.0,
            0.0,
            0.0,
            IMPACT_VELOCITY
        )

        # ======================================================
        # SPH -> SOLID CONTACT
        # ======================================================

        write_comment(f)
        write_comment(f, "SPH TO SOLID AUTOMATIC CONTACT")

        f.write("*CONTACT_AUTOMATIC_NODES_TO_SURFACE\n")

        # SSID MSID SSTYP MSTYP
        #
        # SSID = SPH node set
        # MSID = part ID / target specification
        #
        # For a robust production model, use explicit segment
        # sets if your LS-DYNA version requires them.

        write_card(
            f,
            PLATE_BOTTOM_SET + 1,
            PART_PLATE,
            4,
            3
        )

        write_card(
            f,
            STATIC_FRICTION,
            DYNAMIC_FRICTION,
            0.0,
            VDC,
            0.0,
            0,
            0.0,
            1.0E20
        )

        # ======================================================
        # OPTIONAL BREAKABLE BOND
        # ======================================================

        if ENABLE_TIEBREAK:

            write_comment(f)
            write_comment(
                f,
                "OPTIONAL BREAKABLE BOND - CALIBRATE BEFORE USE"
            )

            f.write(
                "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_TIEBREAK\n"
            )

            # For a production model this should use explicitly
            # generated top-face and particle/contact segment sets.
            #
            # Here the card is left as a configurable placeholder
            # because the actual bond criterion should not be assumed
            # from generic IN718 data.

            write_comment(
                f,
                "TIEBREAK PARAMETERS:"
            )

            write_comment(
                f,
                "Normal limit = "
                f"{BOND_NORMAL_STRESS:.6f} GPa"
            )

            write_comment(
                f,
                "Shear limit = "
                f"{BOND_SHEAR_STRESS:.6f} GPa"
            )

        # ======================================================
        # END
        # ======================================================

        f.write("*END\n")

    print()
    print("==============================================")
    print("LS-DYNA INPUT FILE GENERATED")
    print("==============================================")
    print(f"File: {OUTPUT_FILE}")
    print(f"SPH particles: {len(particles)}")
    print(f"Plate nodes: {len(plate_nodes)}")
    print(f"Plate solids: {len(plate_elements)}")
    print("==============================================")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    write_keyword_file()
