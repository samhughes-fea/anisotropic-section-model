"""Published VABS and BECAS results for the benchmark sections, with their provenance.

Every matrix term is in the **BECAS / VABS-Timoshenko layouts named by ``order``**, 1-based as
the sources print them:

- ``"becas"``: ``[Vx, Vy, N, Mx, My, Mz]`` against ``[gamma_xz, gamma_yz, eps_z, kappa_x,
  kappa_y, kappa_z]``, beam along ``z``, section axes ``(x, y)``.
- ``"vabs"``: ``[F1, F2, F3, M1, M2, M3]`` = ``[N, V2, V3, T, M2, M3]``, beam along ``x1``,
  section axes ``(x2, x3)``.

Section coordinates are the source's: ``(x, y)`` (BECAS) or ``(x2, x3)`` (VABS) are this
package's ``(y, z)``, so ``K_becas = K[p, p]`` with ``p = BECAS_FROM_ASM`` and
``K_vabs = K[q, q]`` with ``q = VABS_FROM_ASM``.

Provenance
----------
The BECAS values are the BECAS user guide's validation tables (Blasques, *User's Manual for
BECAS*, Risø-R-1785, DTU, 2011/2012, section 4.2, cases S1, S2, S3, C1, C2). The VABS values
are from Chen, Yu and Capellaro, "A critical assessment of computer tools for calculating
composite wind turbine blade properties", *Wind Energy* 13 (2010) 497-516 (the circular
tube), and from Yu, Hodges and Ho, "Variational asymptotic beam sectional analysis - an
updated version", *Int. J. Eng. Sci.* 59 (2012) 40-64 (the multi-layer composite pipe).

The input data and values were taken as transcribed, with these sources cited, in GXBeam's
test suite (``test/section.jl`` at byuflowlab/GXBeam.jl commit
``8541187593c90b93125d6e67616655cc68ca19c8``), because the primary PDFs could not be read in
full from the environment that built this benchmark. The pipe values were cross-checked
against a second, independent transcription: the PreVABS documentation's pipe example
(``wenbinyugroup.github.io/prevabs``, "example_pipe"), which prints a current VABS run and
the 2005 values ("YU2005"). The three VABS sets differ by up to 2 % on the coupling terms;
all three are kept (``alternatives``).
"""

from __future__ import annotations

BECAS_FROM_ASM = [3, 4, 0, 1, 2, 5]      # ASM [eps, k_y, k_z, g_xy, g_xz, phi] -> BECAS
VABS_FROM_ASM = [0, 3, 4, 5, 1, 2]       # ASM -> VABS [N, V2, V3, T, M2, M3]

GXBEAM = "byuflowlab/GXBeam.jl test/section.jl @ 8541187"
BECAS_UG = "BECAS user guide (Blasques 2012, Risø-R-1785), section 4.2"

#: materials: (E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho)
MATERIALS = {
    # BECAS UG table 4.1. Its isotropic #2 lists nu = 0.2 / alpha; that is a misprint (GXBeam
    # notes it): the published S2 results need nu = 0.2.
    "iso1": (100.0, 100.0, 100.0, 41.667, 41.667, 41.667, 0.2, 0.2, 0.2, 1.0),
    "iso2": (10.0, 10.0, 10.0, 4.1667, 4.1667, 4.1667, 0.2, 0.2, 0.2, 1.0),
    "ortho": (480.0, 120.0, 120.0, 60.0, 50.0, 60.0, 0.19, 0.26, 0.19, 1.0),
    # Chen et al. 2010: aluminium tube, SI units
    "aluminium": (73e9, 73e9, 73e9, 73e9 / 2.66, 73e9 / 2.66, 73e9 / 2.66, 0.33, 0.33, 0.33, 2800.0),
    # Yu et al. 2012 pipe: psi; lengths in inches
    "pipe": (20.59e6, 1.42e6, 1.42e6, 0.87e6, 0.87e6, 0.87e6, 0.42, 0.42, 0.42, 1.0),
}

#: case -> {"tool", "order", "source", "terms": {"Kij": value}, optional "centres", "mass", "alternatives"}
PUBLISHED = {
    "S1": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
               terms={"K11": 3.4899e-1, "K22": 3.4899e-1, "K33": 1.0, "K44": 8.3384e-4,
                      "K55": 8.3384e-4, "K66": 5.9084e-4},
               centres={"shear": (0.0, 0.0), "tension": (0.0, 0.0)}),
    "S2": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
               terms={"K11": 1.28e-1, "K22": 1.92e-1, "K33": 5.50e-1, "K44": 4.59e-4,
                      "K55": 4.59e-4, "K66": 2.77e-4, "K26": -3.93e-3, "K35": 1.13e-2}),
    "S3_0": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
                 terms={"K11": 5.039e-1, "K22": 4.201e-1, "K33": 4.800, "K44": 4.001e-3,
                        "K55": 4.001e-3, "K66": 7.737e-4}),
    "S3_22.5": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
                    terms={"K11": 7.598e-1, "K22": 4.129e-1, "K33": 3.435, "K44": 2.489e-3,
                           "K55": 2.274e-3, "K66": 9.499e-4, "K13": 7.387e-1, "K46": -4.613e-4}),
    "S3_90": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
                  terms={"K11": 5.0202e-1, "K22": 5.0406e-1, "K33": 1.2, "K44": 1.0004e-3,
                         "K55": 1.0002e-3, "K66": 8.5081e-4}),
    "C1": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
               terms={"K11": 1.249e-1, "K22": 1.249e-1, "K33": 5.965e-1, "K44": 2.697e-3,
                      "K55": 2.697e-3, "K66": 2.248e-3}),
    "C2": dict(tool="BECAS", order="becas", source=BECAS_UG, via=GXBEAM,
               terms={"K11": 4.964e-2, "K22": 6.244e-2, "K33": 2.982e-1, "K44": 1.349e-3,
                      "K55": 1.349e-3, "K66": 9.120e-4, "K35": 1.805e-2, "K26": -7.529e-3},
               centres={"shear": (-1.206e-1, 0.0), "tension": (-6.051e-2, 0.0)}),
    "tube": dict(tool="VABS", order="vabs", source="Chen, Yu and Capellaro, Wind Energy 2010", via=GXBEAM,
                 terms={"K11": 1.835e10, "K22": 4.682e9, "K33": 4.682e9, "K44": 3.519e8,
                        "K55": 4.587e8, "K66": 4.587e8},
                 mass={"m": 7.037e2, "I_yy": 1.759e1, "I_zz": 1.759e1}),
    "pipe": dict(tool="VABS", order="vabs", source="Yu, Hodges and Ho, Int. J. Eng. Sci. 2012", via=GXBEAM,
                 terms={"K11": 1.03892e7, "K22": 7.85310e5, "K33": 3.29279e5, "K14": 9.84575e4,
                        "K25": -8.21805e3, "K36": -5.20981e4, "K44": 6.87275e5, "K55": 1.88238e6,
                        "K66": 5.38987e6},
                 alternatives={
                     "PreVABS docs, VABS": {"K11": 1.03892e7, "K22": 7.85800e5, "K33": 3.31330e5,
                                            "K14": 9.74568e4, "K25": -8.02785e3, "K36": -5.14533e4,
                                            "K44": 6.89600e5, "K55": 1.88230e6, "K66": 5.38985e6},
                     "PreVABS docs, YU2005": {"K11": 1.03890e7, "K22": 7.84299e5, "K33": 3.29002e5,
                                              "K14": 9.82878e4, "K25": -8.18782e3, "K36": -5.18541e4,
                                              "K44": 6.86973e5, "K55": 1.88236e6, "K66": 5.38972e6},
                 }),
}
