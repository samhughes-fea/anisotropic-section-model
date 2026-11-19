"""Published BECAS values for the BECAS example meshes, from the BECAS user guide.

Blasques, *User's Manual for BECAS*, Risø-R-1785 (DTU), validation section: tables for C3
(two-material cylinder against E1/E2), C4 (layered cylinder), T1, T2 and T3 (three-cell
sections). The BECAS and VABS columns of those tables agree to the digits printed, so only
the BECAS values are kept. Terms are 1-based in BECAS's order ``[Vx, Vy, N, Mx, My, Mz]``.

Two of the tables' column labels do not match the matrices they print; the labels below are
the corrected ones, and the evidence is the same-mesh VABS output shipped with BECAS (C3) and
the section's own centres (T1):

- **C3**: the columns headed ``K11, K22, K33, K44 = K55, K66`` hold ``K22, K33, K11, K66,
  K44 = K55``. VABS's full-precision Timoshenko matrices on the same meshes (shipped with
  BECAS as ``C3_*.vabs.K``) give exactly the printed values in that order (e.g. E1/E2 = 10:
  K33 = EA = 0.328, the printed "K22").
- **T1**: the rows headed ``K26`` and ``K35`` are swapped. ``K35`` (extension-bending) must
  equal ``-K33 x_t``: with ``K33 = 2.92`` and the tension centre ``x_t = -1.969e-2`` printed
  in the same table that is ``+5.75e-2``, the value printed against ``K26``. T2 and T3
  (same geometry) print their ``K26`` and ``K35`` with the labels the other way round, and
  consistent with this.

Values are as printed (three or five significant digits); compare them at that rounding.
"""

from __future__ import annotations

UG = "BECAS user guide (Blasques, Risø-R-1785), validation section"

#: case -> {"terms": {Kij: value}, "centres": {"shear": (x, y), "tension": (x, y)}}
USER_GUIDE = {
    "C3_1": dict(terms={"K22": 1.25e-1, "K33": 5.96e-1, "K11": 1.25e-1, "K66": 2.25e-3, "K44": 2.70e-3,
                        "K55": 2.70e-3},
                 centres={"shear": (0.0, 0.0), "tension": (0.0, 0.0)}),
    "C3_10": dict(terms={"K22": 6.87e-2, "K33": 3.28e-1, "K11": 3.99e-2, "K66": 1.08e-3, "K44": 1.48e-3,
                         "K55": 1.48e-3, "K35": 1.62e-2, "K26": -6.78e-3},
                  centres={"shear": (-9.866e-2, 0.0), "tension": (-4.951e-2, 0.0)}),
    "C3_100": dict(terms={"K22": 6.31e-2, "K33": 3.01e-1, "K11": 3.75e-2, "K66": 9.29e-4, "K44": 1.36e-3,
                          "K55": 1.36e-3, "K35": 1.79e-2, "K26": -7.45e-3},
                   centres={"shear": (-1.182e-1, 0.0), "tension": (-5.931e-2, 0.0)}),
    "C3_1000": dict(terms={"K22": 6.25e-2, "K33": 2.99e-1, "K11": 4.74e-2, "K66": 9.14e-4, "K44": 1.35e-3,
                           "K55": 1.35e-3, "K35": 1.80e-2, "K26": -7.52e-3},
                    centres={"shear": (-1.203e-1, 0.0), "tension": (-6.039e-2, 0.0)}),
    "C3_10000": dict(terms={"K22": 6.24e-2, "K33": 2.98e-1, "K11": 4.94e-2, "K66": 9.12e-4, "K44": 1.35e-3,
                            "K55": 1.35e-3, "K35": 1.80e-2, "K26": -7.53e-3},
                     centres={"shear": (-1.206e-1, 0.0), "tension": (-6.050e-2, 0.0)}),
    "C3_100000": dict(terms={"K22": 6.24e-2, "K33": 2.98e-1, "K11": 4.96e-2, "K66": 9.12e-4, "K44": 1.35e-3,
                             "K55": 1.35e-3, "K35": 1.80e-2, "K26": -7.53e-3},
                      centres={"shear": (-1.206e-1, 0.0), "tension": (-6.051e-2, 0.0)}),
    "C4": dict(terms={"K11": 8.3114e-2, "K22": 8.3114e-2, "K33": 3.9784e-1, "K44": 1.8012e-3,
                      "K55": 1.8012e-3, "K66": 1.5010e-3}),
    "T1": dict(terms={"K11": 7.61e-1, "K22": 2.93e-1, "K33": 2.92, "K44": 3.29e-2, "K55": 2.94e-1,
                      "K66": 3.95e-2, "K35": 5.75e-2, "K26": -8.26e-3},
               centres={"shear": (-2.823e-2, 0.0), "tension": (-1.969e-2, 0.0)}),
    "T2": dict(terms={"K11": 1.79, "K22": 3.22e-1, "K33": 4.37, "K44": 5.23e-2, "K55": 3.81e-1,
                      "K66": 7.73e-2, "K13": -8.48e-1, "K15": 2.75e-3, "K24": -3.13e-3, "K26": -1.49e-2,
                      "K35": 5.64e-2, "K46": 1.83e-2},
               centres={"shear": (-4.300e-2, 0.0), "tension": (-1.456e-2, 0.0)}),
    "T3": dict(terms={"K11": 1.78, "K22": 3.19e-1, "K33": 4.37, "K44": 5.22e-2, "K55": 3.81e-1,
                      "K66": 7.71e-2, "K13": -8.46e-1, "K15": 2.37e-3, "K24": -3.11e-3, "K26": -1.48e-2,
                      "K35": 5.65e-2, "K46": 1.82e-2},
               centres={"shear": (-4.303e-2, 0.0), "tension": (-1.454e-2, 0.0)}),
}
