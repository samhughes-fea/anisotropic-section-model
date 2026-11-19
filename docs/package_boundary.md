# Package boundary: ASM and 1d-fem

ASM (anisotropic-section-model) answers *what is this cross-section?* 1d-fem answers
*what does the beam do?* A function belongs in ASM only if it can be written for one
section, with no span coordinate, element connectivity or mesh of the beam.

| ASM: one section at a station | 1d-fem: along the span |
|---|---|
| 6 × 6 Timoshenko stiffness, 4 × 4 classical, rigid-section mass | elements, shape functions, kinematics |
| shear, tension and mass centres; second-order terms (polar, Wagner) | Ke, Kg, Fe integration at its Gauss points |
| warping (Benscoter), third-order and GBT section matrices | assembly, supports, loads |
| station modifiers: pretwist rate, taper (Resal) coupling | static, nonlinear, eigen, buckling, harmonic, transient solves |
| stress recovery, ply stresses, failure exposures (`SectionStrength`) | ply checks along the beam and load-case envelopes (`ply_strength.txt`) |
| layup derivatives of the section (`dD`, `dM`) | adjoint sensitivities of beam responses |
| builders: walls + laminates, 2D meshes, geometry-kernel cuts, isotropic constants, 6 × 6 tables (`from_matrices`) | job input: `section.txt`, `element_section.txt`, `section_matrix.txt`, `section_taper.txt` ... |
| BeamDyn / windIO export of section tables at stations | blade models from a loft |

ASM is the single source of section properties for every blade section, isotropic ones,
sections with asymmetric (geometric) couplings and sections with anisotropic material
couplings alike. ASM never imports 1d-fem.

## The contract

1d-fem reads sections through `onedfem.sections` (the *section provider*). What it takes
from each `section_analysis.SectionAnalysis` is the contract, versioned as
`SECTION_API_VERSION = (major, minor)`:

- **v1.0** (now): 1d-fem's frame (beam axis x, section `(y, z)`, strains
  `[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]`) with 1d-fem's element sign on
  `kappa_y` already applied (`ELEMENT_SIGN`). Required: `stiffness`, `classical_stiffness`,
  `shear_centre`, `tension_centre`, `mass` / `mass_matrix()`. Optional, by element type:
  second-order terms, axial points, Benscoter, third-order and GBT data, taper.
- **v2.0** (planned): right-handed engine matrices, with the element sign applied by
  1d-fem; section derivatives returned as plain `dD`, `dM` arrays; `align_gbt_sections`
  (the walk over elements) removed in favour of `align_gbt_pair`.

A new optional field is a minor bump; renaming, re-signing or reshaping one is a major
bump. 1d-fem refuses a provider whose major version differs.

## Reference beams

`beam.TimoshenkoBeam`, `torsion.RestrainedTorsionBeam` and `gbt.GeneralizedBeam` stay as
frozen reference checks: prismatic, linear, used by the examples to check a section
against a shell. They get no new features. `tests/test_package_boundary.py` fails if any
other public callable takes a span, element connectivity or a node count.
