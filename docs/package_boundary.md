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
`SECTION_API_VERSION = (major, minor)`. A new optional field is a minor bump; renaming,
re-signing or reshaping one is a major bump. 1d-fem refuses a provider whose major version
differs.

### v2.0 (ASM 0.6.0)

Every quantity is per unit length, in ASM's **engine frame** and about the section origin
(the beam reference axis) unless stated: beam axis x, section `(y, z)`, right-handed strains
`[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]` and forces `[N, M_y, M_z, V_y, V_z, T]`
with `M_y = int sigma z dA`, `M_z = -int sigma y dA`. ASM applies no beam code's element
signs. 1d-fem, whose `B` matrices have `kappa_y = -theta_y'`, re-signs each section once,
after the station modifiers and GBT alignment.

| Field | Shape, units | Required |
| --- | --- | --- |
| `frame` | `"engine"`; station modifiers refuse any other | yes |
| `stiffness` | 6 x 6 Timoshenko, N, N m, N m^2 | yes |
| `classical_stiffness` | 4 x 4 on `[eps, kappa_y, kappa_z, phi]` | yes |
| `shear_centre`, `tension_centre` | `(y, z)`, m | yes |
| `mass`, `mass_matrix()` | `m` kg/m; `y_m`, `z_m` m; `I_yy`, `I_zz`, `I_yz`, `I_p` kg m about the origin | for dynamics |
| `section_constants` | `A` m^2; `I_y`, `I_z`, `I_p`, `J_t` m^4; `Gamma` m^6 (isotropic only) | optional |
| `warping_stiffness` | `E Gamma`, N m^4, about the shear centre | optional |
| `benscoter_stiffness`, `warping_inertia` | 8 x 8; kg m^3 | optional |
| `third_order_stiffness`, `third_order_rate` | 8 x 8; 2 x 6 | optional |
| `geometric_terms`, `benscoter_geometric_terms`, `third_order_geometric_terms` | 7 x 6, 7 x 2, 7 x 2 | optional |
| `axial_points` | m x 3, `[y, z, E dA]` | optional |
| `gbt_*` | GBT stiffness, mass, geometric, Brazier, moments | optional |
| `tapered`, `taper_parts` | flag; per-laminate taper | optional |

Functions in the contract: `SectionAnalysis.from_isotropic`, `from_matrices`,
`from_thin_walled`, `from_solid`, `with_pretwist`, `with_taper`;
`thin_walled.layup_sensitivity.section_analysis_derivative` (the station's derivative as a
`SectionAnalysisDerivative` of plain engine-frame arrays: `stiffness` with its taper,
`classical_stiffness`, `mass_matrix`, Benscoter, second-order, axial points, GBT);
`gbt_alignment.align_gbt_pair` and `mass_transform` (1d-fem walks its elements).

**Torsion and polar constants.** `J_t` (Saint-Venant torsion constant, m^4) enters only the
stiffness: `stiffness[5, 5] = G J_t` [N m^2] for an uncoupled section. For a coupled one
the torsional stiffness is `1 / F[5, 5]` (twist under a pure torque, shear forces free),
which does not depend on the reference axis. `I_p = int (y^2 + z^2) dA` (polar second
moment, m^4, about the origin) enters only the mass, as the twist rotary inertia
`mass["I_p"] = rho I_p = mass_matrix()[3, 3]` [kg m], and the Wagner term `I_p / A`. The
two are equal only for a circular section. `from_isotropic` takes them as `J_t=` and
`I_p=` (`I_x=` is the v1 name, deprecated).

### v1.0 (ASM 0.5.x)

1d-fem's element sign on `kappa_y` already applied (`ELEMENT_SIGN`, `to_element_voigt`),
`element_section_derivative` (a derivative dressed as an element-signed section) and
`align_gbt_sections` (the walk over elements). All four are gone in v2.

### v2.1 (ASM 0.7.0)

Optional 2D ply-thickness data in `section_mesh_2d.txt`: `[Section_Node_Velocity]`
(`[section_id] [node_id] [laminate_id] [ply] [dy] [dz]`, `SectionMesh.velocities`) and
`[Section_Tie_Point]` (`[section_id] [node_id] [element_id] [xi] [eta]`,
`SectionMesh.tie_points`). `write_section_mesh_2d` writes them and `parse_section_mesh_2d`
reads them, so a mesh from `SectionMesh.from_thin_walled` keeps its ply-thickness
sensitivities through the file. Since ASM 0.8.0 meshes of blade cuts
(`section_mesh_from_cut`) carry them too, with no change to the contract: ply tags,
velocities and an empty `tie_points` (the cut mesh conforms, so it has no ties). Planned
next: tapered 2D ply-thickness sensitivities in 1d-fem.

## Reference beams

`beam.TimoshenkoBeam`, `torsion.RestrainedTorsionBeam` and `gbt.GeneralizedBeam` stay as
frozen reference checks: prismatic, linear, used by the examples to check a section
against a shell. They get no new features. `tests/test_package_boundary.py` fails if any
other public callable takes a span, element connectivity or a node count.
