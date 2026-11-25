# Changelog

## [Unreleased]

### Documentation

- `docs/package_boundary.md` lists the section and taper derivatives in the contract
  (`section_derivative`, `solid_section_derivative(taper=)`, `section_taper`,
  `solid_section_taper`), which 1d-fem now uses for tapered 2D sections
  (`[Section_Taper_2D]`). They have been in ASM since 0.5.0, so the contract stays at 2.1.
  (#46)

### Changed

- CI lints with a pinned ruff (`requirements/lint.txt`, syntax errors and pyflakes):
  unused imports and four unused locals removed, and the VABS/BECAS benchmark cases
  import the `SectionMesh` they annotate. Results are unchanged. (#45)
- Follows the shared development standard 0.1.1: the bug form asks for the system
  libraries the package loads (GEOS) and where they came from. (#45)

## [0.8.0] - 2026-11-24

### Added

- Ply-thickness sensitivities of 2D blade cuts: `section_mesh_from_cut` now tags each
  element with its laminate ply (`SectionMesh.plies`) and fills `SectionMesh.velocities`
  and `tie_points` (empty: the mesh conforms), as `SectionMesh.from_thin_walled` does, and
  sets `SectionMesh.laminates` (`{laminate_id: WallLaminate}`; library laminates keep
  their ids). Skins stay centred on the skin mid-line, caps stack inward from the skin,
  webs stay centred on their mid-line and their feet slide along the inner face, carrying
  a cap edge that ends under a web with them. `solid_section_derivative` matches meshes
  moved by `h V` to 1e-5 and meshes rebuilt with the thicker ply to 0.5 % on the
  diagonal stiffnesses (`tests/test_solid_section_from_geometry_thickness.py`). No
  contract change: the v2.1 file blocks carry it. `velocities=False` leaves them out
  (they make `section_mesh_2d.txt` several times larger). (#43)

### Changed

- Follows the shared development standard 0.1.0: synced issue forms, PR template,
  labels, `pr-title`, `label-sync`, `codeql` and `release` workflows, and the standard
  in `docs/conventions/standard/`. `release.yml` is now the shared one (it reads
  `.github/repo.yml`; the backfill mode is gone), and new changelog sections use
  `## [X.Y.Z] - YYYY-MM-DD`. Added `AGENTS.md`, `SCOPE.md`, `CONTRIBUTING.md` and
  `.github/labels.yml`. (#41)

## 0.7.0 - 2026-11-24

### Added

- Section-provider contract 2.1 (`SECTION_API_VERSION = (2, 1)`): `section_mesh_2d.txt`
  takes optional `[Section_Node_Velocity]` and `[Section_Tie_Point]` blocks, which
  `write_section_mesh_2d` writes and `parse_section_mesh_2d` reads into
  `SectionMesh.velocities` and `tie_points`. A meshed section written to file keeps its
  ply-thickness derivatives (`solid_section_derivative`) to 1e-12
  (`tests/test_section_mesh_2d_velocities.py`).

## 0.6.1 - 2026-11-24

### Fixed

- `SectionAnalysis.from_isotropic(y_sc=, z_sc=)` now couples shear and twist about the
  given shear centre, where it only recorded it. With the shear strains at the shear
  centre `gamma_xy - z_sc phi`, `gamma_xz + y_sc phi`, the centroid stiffness gets
  `D[3,5] = -z_sc kGA`, `D[4,5] = y_sc kGA` and `D[5,5] = GJ_t + kGA (y_sc^2 + z_sc^2)`.
  A shear force through the shear centre then twists nothing (`from_matrices` recovers
  the centre) and the torsional stiffness about it is still `G J_t`. On a thin-walled
  channel the block matches the thin-walled model's to 1e-9. `classical_stiffness`, the
  mass and the second-order terms do not change; a zero offset gives the old section.
  It reached `main` just after the 0.6.0 tag, so 0.6.0 does not have it (its
  changelog listed it by mistake).

## 0.6.0 - 2026-11-24

### Breaking: section-provider contract 2.0

- `SECTION_API_VERSION` is `(2, 0)`. 1d-fem 1.x with contract 1.0 refuses this release;
  use a 1d-fem that needs 2.0 (`docs/package_boundary.md`).
- `SectionAnalysis` is in the engine frame: every matrix (6 x 6, classical, Benscoter,
  third-order and its rate, second-order terms, GBT stiffness, geometric and Brazier
  tensors) is right-handed, as `ThinWalledSectionAnalysis` and `SolidSectionAnalysis`
  compute it. The kappa_y row and column are no longer negated; a beam code applies its
  own element signs. New field `frame` (always `"engine"`); `with_pretwist` and
  `with_taper` refuse a section with another frame.
- `with_pretwist` uses the right-handed fibre strain `eps + z kappa_y - y kappa_z`, so its
  kappa_y coupling is `+k int E z r^2 dA` (the same physics as before, re-signed).
- Removed: `section_analysis.ELEMENT_SIGN`, `section_analysis.to_element_voigt`,
  `axes.ELEMENT_SIGN`, `axes.to_element`,
  `thin_walled.layup_sensitivity.element_section_derivative` and
  `gbt_alignment.align_gbt_sections`.

### Added

- `thin_walled.layup_sensitivity.section_analysis_derivative(sa, d)` and
  `SectionAnalysisDerivative`: the derivative of a station's section as plain engine-frame
  arrays, with its taper and, new, the classical stiffness (`dC = -C dF_c C`).
- `SectionAnalysis.section_constants` (`SECTION_CONSTANTS`): `A`, `I_y`, `I_z`, `I_p`,
  `J_t`, `Gamma` of an isotropic section, with units; `None` for composite sections.
- `from_isotropic(I_p=...)` names the polar moment; `I_x=` still works with a
  `DeprecationWarning`. `I_p <= 0` now also means `I_y + I_z`.
- `gbt_alignment.align_gbt_pair` is in `__all__`.
- `tests/test_section_contract_v2.py`.

## 0.5.1 - not released (folded into 0.6.0)

### Added

- `tests/test_isotropic_sections.py`: isotropic sections built from geometry agree with
  `SectionAnalysis.from_isotropic` and the closed forms. A solid rectangle (2D builder) and
  a thin-walled tube give the same diagonal stiffness and mass as the given constants; an
  equal-leg angle keeps its bending coupling, its shear centre at the corner and its
  principal moments `a^3 t / 3`, `a^3 t / 12`. 1D-FEM now takes every isotropic
  `section.txt` section from `from_isotropic`. No API or contract change (still 1.0) (#32).

### Changed

- The `from_isotropic` docstring defines `J_t` (Saint-Venant torsion constant, giving the
  torsional stiffness `G J_t`) and `I_x` (polar second moment `I_p` about the centroid,
  giving the rotary inertia `rho I_p`) as separate geometric constants, with units and
  reference axis. It no longer says elements keep their own shear-centre handling (#32).
- CI and the README install implicit-geometry-kernel `v0.9.5` (was `v0.9.3`;
  `v0.9.4` carries 0.9.3 code) (#34).
- CI and the README install implicit-geometry-kernel `v0.9.3` instead of the
  non-release tag `K9` (#31).
- **Releases are made by the `release` workflow.** Merging a release PR titled
  `chore(release): prepare X.Y.Z` creates the annotated `vX.Y.Z` tag and the
  GitHub release; nobody pushes a tag by hand. A hand run can backfill a tag on an
  older commit, which is how `v0.4.0` is made. Rules: `docs/RELEASING.md`.
  `__version__` on `main` is now the next release (0.5.1).

## 0.5.0 - 2026-11-19

### Breaking

- `BeamDynBlade.from_sections(twist=...)` now takes the public twist: right-handed about
  z, the same number as BeamDyn's `initial_twist` and windIO's `outer_shape.twist`. In
  0.4.0 it took the kernel's `SectionCut.twist`, which is its negative, so negate any
  twist you passed (`axes.twist_from_kernel`). `from_geometry` converts the kernel's twist
  itself, so the files it writes are unchanged. `to_beamdyn_frame(matrix, twist)` also
  takes the public twist now.

### Added

- **One axis convention.** A new `axes` module defines the public frame (beam axis z, x to
  the trailing edge, y to the suction side, BeamDyn's strain order, right-handed twist).
  It holds every conversion out of it: the engine (1d-fem) order, the 1d-fem element sign,
  BeamDyn's mirror and the geometry kernel's twist sign. `MIRROR`, `rotation6` and
  `to_beamdyn_frame` now live there and can still be imported from `beamdyn`. New
  `SolidSection`, the 2D finite-element section in the public frame
  (`timoshenko_stiffness(reference)`, `mass_matrix`, centres, `GJ_free`,
  `strain_response`). New `FormulationWriter.from_section`, which takes forces in the
  public order. Section coordinates are the same numbers in both frames.
- **Resal coupling for 2D-meshed sections.** `solid.taper.solid_section_taper` is the taper
  coupling of `thin_walled.taper.section_taper` for a `SectionMesh` (two meshes of the same
  topology a step apart), with the axial modulus of each element. New
  `thin_walled.taper.taper_derivative`, and `section_derivative(..., taper=)` /
  `solid_section_derivative(..., taper=)` return the coupling's layup derivative
  (`SectionDerivative.taper`), which `element_section_derivative` adds.
- **Ply-thickness sensitivities of 2D-meshed sections.** `solid_section_derivative` takes
  `LayupParameter(..., "thickness")`. `SectionMesh.from_thin_walled` records how its nodes
  move with each ply's thickness (`SectionMesh.velocities`, exact: the nodes are affine in
  the thicknesses) and how each tied node was tied (`SectionMesh.tie_points`). The
  derivative carries the motion through the element Jacobians, the tie weights, the
  Saint-Venant and restrained-warping solves (with the moving warping constraint that fixes
  the torsional warping's in-plane rotation), the second-order terms, the axial quadrature
  (`SectionDerivative.axial_positions`) and the mass. Checked against finite differences of
  rebuilt and of moved meshes, and against the thin-walled derivatives.
- **Fix:** `SectionMesh.from_thin_walled` merged coincident nodes by rounding them to a grid,
  so a pair either side of a grid line stayed apart (a loose node at a junction). Nodes now
  merge by distance.
- **Section strength checks.** `SectionStrength` (`section_strength`) recovers the ply
  stresses of a section under its six section forces, thin-walled (the coupled Saint-Venant
  wall strains, ply stresses through each laminate, interlaminar stresses from equilibrium)
  or meshed in 2D (`SolidSectionAnalysis`), and evaluates maximum stress, Tsai-Wu, Hashin and
  Puck at every point, with the critical wall, ply, position, criterion and mode. It solves
  six unit force cases once, so each load case is a matrix product. Example 14 checks it on a
  three-cell blade against a shell model, at cuts whose forces follow from statics: ply
  stresses within 3 % RMS, exposures within 2.2 % away from the clamp and the tip rib, the
  same critical part and criterion at every row, and the solid section within 1 % of the
  thin-walled one away from the web junctions. Running the check along a beam (forces from
  a beam solve, load-case envelopes) is 1d-fem's.
- **Section-provider contract v1.** `SECTION_API_VERSION = (1, 0)` versions what a beam code
  (1d-fem) takes from `SectionAnalysis`; `docs/package_boundary.md` spells it out, and
  `tests/test_package_boundary.py` keeps span-wise solvers out of the package.
- **Sections from 6 x 6 tables.** `SectionAnalysis.from_matrices(stiffness, mass, frame=)`
  takes a section given only as its stiffness and rigid-section mass, in the engine, public
  or BeamDyn frame, and finds the shear and tension centres the stiffness implies. On a
  coupled, off-centre composite box it reproduces the thin-walled `SectionAnalysis`.
- **`gbt_alignment.align_gbt_pair`** turns one section's GBT bases to a reference section's;
  `align_gbt_sections`, the walk over a beam's elements, is kept for contract v1 and goes in
  v2 (the walk is the beam code's).
- **Reference beams frozen.** `TimoshenkoBeam`, `RestrainedTorsionBeam` and
  `GeneralizedBeam` stay as exact checks of a section and get no new features.
- **Scope:** ASM covers one cross-section at a time. Beam solves along the span (tapered
  beams, beam-level sensitivities, modal analysis, spanwise strength) belong to 1d-fem, which
  uses these sections; see `docs/package_boundary.md`.

## 0.4.0 - 2026-11-19

- **Worked example suite.** Four new examples: isotropic sections against the closed forms,
  thin-wall and solid (08); CAS and CUS composite box beams, thin-wall against the solid
  section and a shell cantilever (09); coupled torsion, GJ against GJ_free with a shell
  cantilever with a free and a held tip (10); the solid section against every BECAS and VABS
  validation case (11). Every example now ends with self-checks against its references
  (`examples/_shared/checks.py`) and fails if one drifts: `tests/test_examples.py` re-checks
  the committed results and CI's new `examples` job solves each example from scratch.
  `run_all.py` takes example numbers and `--reuse`. The index is reorganised as a reading
  path. Examples 02 and 05 have their numbers refreshed for the coupled-torsion change.
- **BeamDyn and windIO export.** `BeamDynBlade.from_geometry` sweeps a `BladeGeometry`
  (one `ThinWalledSection.from_geometry` cut per station; laminates can be a function of
  z) and `from_sections` takes sections you have. `write` gives a complete BeamDyn deck,
  the primary file (one member through the stations' pitch-axis points, initial twist,
  solver options) and the blade file (6 × 6 stiffness and mass per station, damping),
  for OpenFAST 5 or, with `version=4`, OpenFAST 3 and 4. `to_windio` / `write_windio`
  give the windIO 2.0 `reference_axis`, `twist` and `elastic_properties`. `read`,
  `read_blade` and `from_windio` read them back. Matrices are about the pitch axis in
  the chord-aligned frame, in BeamDyn's axes (mirrored from the geometry kernel's), and
  the twist takes BeamDyn's sign. New `ThinWalledSection.mass_matrix(reference)`.
  Checked against the IEA 15 MW windIO and BeamDyn files (1e-12) and in BeamDyn itself
  (OpenFAST 4.1.2 and 5.0.0) against `TimoshenkoBeam`; example 07. New `windio` extra
  (PyYAML).
- **Same-mesh BECAS benchmark** (`benchmarks/becas_same_mesh`, gated by
  `tests/test_benchmark_becas_same_mesh.py`). A reader and writer of BECAS's 2D input
  (`becas_io`) puts ASM and BECAS on identical meshes. On BECAS's own validation meshes ASM
  matches the full 6 x 6 and both centres of the BECAS and VABS outputs shipped with them to
  1e-9 or better (S2, S3, C1, C2, C3 at E1/E2 up to 1e5, C4 layered cylinder), and the user
  guide's three-cell T1 to T3 tables to their printed digits. Adds an MH-104-style two-web
  section, CAS and CUS box beams and a layered composite cylinder, written to BECAS input; the
  BECAS runs on them are pending a MATLAB or Octave runtime. Records stale result files and
  mislabelled tables found in BECAS's examples and guide.
- **VABS and BECAS benchmark** for the solid section (`benchmarks/vabs_becas`, gated by
  `tests/test_benchmark_vabs_becas.py`). On the BECAS user guide's own 10 x 10 mesh of its
  squares (isotropic, two-material, orthotropic at 0, 22.5 and 90 degrees) ASM reproduces
  every published term to the printed digits, couplings included; converged, it matches the
  closed forms and sits about 0.85 % below BECAS's coarse-mesh shear and torsion values. The
  BECAS tube and half tube agree to 0.15 % (centres included), the Chen et al. aluminium tube
  to 0.02 % (VABS's GJ is 2 % above the exact value, which ASM matches), and the Yu et al.
  multi-layer composite pipe to 0.2 % on all nine VABS terms, extension-twist and
  shear-bending couplings included. `RESULTS.md` has the tables.
- Solid section: a bilinear `Q4` element, to run the BECAS user guide's meshes, and
  `solid.structured` (`structured_mesh`, `rectangle_patch`, `arc_patch`): conforming mapped
  meshes of rectangles and annular sectors in `Q4`, `Q8` or `T6`, with plies that follow
  curved walls.
- **Torsion from the coupled section analysis.** `ThinWalledSection.omega`, `S_warping`
  and `Ip` now come from `section.analysis` (its torsional warping and Benscoter
  stiffness), like `GJ` and `EGamma` already did; the classical values move to
  `section.classical`. `vlasov_stiffness` gains the couplings of the warping gradient
  with the six beam strains. The new `GJ_free` is the torsional stiffness with every
  other section force zero (1 / F[5, 5] about the shear centre).
  `RestrainedTorsionBeam.from_section` takes `other_strains="free"` (the default,
  `GJ_free`) or `"held"` (`GJ`, the old behaviour). Balanced walls are unchanged to
  about 1e-4. On a box with one off-axis layup ([30]4, extension-twist coupled),
  `GJ_free` is 0.44 GJ, and a shell model of the box twists within 3 % of it
  (`tests/test_coupled_torsion.py`); the old `from_section` was about 2.2 times too stiff.
  GBT's classical torsional warping mode follows the coupled `omega`.
- `FormulationWriter` returns warping, shear flow and wall strains at section, wall,
  Gauss and ply levels of a `ThinWalledSectionAnalysis`, and writes one CSV per level
  (`to_csv`). Warping is force-independent; strains, shear flow and ply stresses are
  for the given section forces.
- Solid 2D section at root and cap stations. `solid.compare_cut` and
  `compare_thick_stations` run the 2D finite-element 6 x 6 beside the
  thin-walled 6 x 6 of the same cut and report the difference. A published
  matrix is compared only when the caller supplies held named terms (or a
  `K6` already in `STRAINS` order); missing fields stay `absent`. No VABS
  or BECAS gate is added.
- Principal axes of the full section. `SectionAnalysis`, `ThinWalledSectionAnalysis`
  and `SolidSectionAnalysis` export `principal_angle` and `principal_moments` from
  the mass second moments about the mass centre, the same pair `reduced_section_from_integrals`
  already gave on the reduced section. `principal_axes_from_moments` and
  `principal_axes_from_mass` are the shared helpers. No VABS or BECAS gate is added.
- CI and the install notes pin implicit-geometry-kernel to tag K9
  (`d3f6c4fb74bff47131fd3dcbabb1228c3eea6b0b`). The package still does not
  import the kernel.
- `reduced_section_from_integrals` accepts geometric integrals (area, centroid,
  second moments, including a cap offset already in those moments) and returns
  the elastic centre, shear centre, principal axes, mass centre and warping
  constant. It does not import the geometry kernel or read a wall graph. For a
  homogeneous section the three centres are the centroid and the warping
  constant is zero.

## 0.3.0 - 2026-11-11

- Tests, README and example 06 use implicit-geometry-kernel 0.9.1: full layout codes
  (`"2D-F-LC-0R"`) and `pivot_fraction`. CI installs the v0.9.1 tag and `libgeos-dev`.
- **Eccentric walls.** `WallLaminate` takes an `offset` (mid-plane above the wall's
  reference line) and its `A`, `B`, `D` are about that line. `WallLaminate.from_laminate`
  keeps a laminate's offset instead of refusing it.
- `ThinWalledSection.from_geometry` and `thin_walled_from_cut` keep a spar cap bonded to a
  skin on the skin mid-surface, where the kernel now puts it, with the stacked laminate
  offset by half the cap thickness inward. On the example 20 section EA, EI and GJ are
  within 0.5 % of before; the shear centre moves from 5.9 mm to 4.5 mm (the solid section
  gives 5.2 mm on a 2 mm mesh).
- `ThinWalledSection.from_blade_mesh` leaves walls on the mesh nodes and keeps laminate
  offsets as eccentricity (it moved the nodes by the offset before). The classical
  closed-form values place each wall's membrane stiffness at its mid-plane.
- CI and the install notes pin laminate-constitutive-model v0.3.0 (was a commit on its main).
- Example 03 reads the IEA 15 MW sections from the private reference-blades catalogue
  (`examples/_shared/reference_blades.py`) and runs its box spar part alone without it.
  `examples/_shared/data/` (the section data and its WISDEM script) is removed, and the
  published 6 × 6 is no longer stored in `data/results.json`.

## 0.2.0 - 2026-11-05

- **One section engine.** `ThinWalledSection` keeps its API but takes `EA`, `EI_*`,
  `GJ`, the centroid and shear centre, `EGamma`, `timoshenko_stiffness` and
  `vlasov_stiffness` from the coupled `thin_walled.ThinWalledSectionAnalysis` of its
  walls (`section.analysis`), with every coupling of unbalanced walls. Results for
  balanced, symmetric walls are unchanged to about 1e-4 (the coupled analysis also
  counts the walls' own bending); unbalanced walls change by up to about 15%. The
  classical values stay available as `section.classical`,
  `classical_timoshenko_stiffness()` and `classical_summary()`, and the shear flows,
  warping function and Benscoter quantities used by `RestrainedTorsionBeam` and
  `GeneralizedBeam` are still the classical ones. `wall_bending` maps onto the
  analysis' `hoop` option.
- README section on the two axis conventions and the BeamDyn conversion
  (`BEAMDYN_FROM_ONEDFEM`).
- **laminate-constitutive-model plugs into the moved section code.** `ply_failure` now
  evaluates its criteria with LCM's array exposures (`ply_exposures`, new in LCM after
  0.2.0); results match 1d-fem's own implementation to 1e-12 (`tests/test_lcm_adapters.py`).
  `PlyStrength` keeps its defaults (`S13 = S12`, `S23 = R_TT`, `F_star = -1/2`) and gains
  `from_strength_properties`; `F_star` must now lie strictly inside (-1, 1).
  `WallLaminate.from_laminate`, `PlyMaterial.from_elastic` / `from_material_card` and
  `SolidMaterial.from_elastic` / `from_material_card` take LCM laminates and materials
  (the transverse shear order is swapped from LCM's `[gamma_yz, gamma_xz]`).
- CI and the install notes pin laminate-constitutive-model to the commit that adds the
  exposures, until it is tagged.
- **1d-fem's section code moves here** (step 6 of the split), with its git history:
  `thin_walled/` (wall laminates and condensed wall laws, Saint-Venant and Benscoter
  analysis, GBT modes, ply stress recovery, layup derivatives, taper), `solid/` (2D
  finite-element sections), `section_analysis`, `ply_failure`, `gbt_alignment`,
  `closed_form` and `geometry_to_properties`, and their tests. 1d-fem's job-file
  reader and beam-element recovery stay in 1d-fem. The moved code keeps 1d-fem's beam
  axis x and strain order; it is folded into the existing API in a later step.
- SciPy is now a required dependency (the GBT and 2D section solves use it).
- CI and the install notes pin laminate-constitutive-model v0.2.0 and shell-fem v0.1.0.
- Depends on [laminate-constitutive-model](https://github.com/samhughes-fea/laminate-constitutive-model)
  (`Laminate`, `integrate_abd`) instead of composite-laminate-model, which was split
  into that package and [shell-fem](https://github.com/samhughes-fea/shell-fem).
  The tests and examples build their shell meshes with `shell_fem`.
- `tests/test_import_boundary.py` checks that the package imports only NumPy and
  laminate-constitutive-model.

## 0.1.0 (unreleased)

- Split out of [composite-laminate-fea](https://github.com/samhughes-fea/composite-laminate-fea)
  with its history. Its examples 27 to 32 are examples 01 to 06 here.
- `ThinWalledSection`: Saint-Venant shear flows and GJ in any number of cells, the
  warping function, shear centre and EΓ, and the Timoshenko 6 × 6 and Timoshenko-Vlasov
  7 × 7 from thin-wall theory; built from wall segments, a shell blade mesh
  (`from_blade_mesh`) or a cut of the implicit geometry (`from_geometry`).
- `RestrainedTorsionBeam`: Benscoter and Vlasov torsion with the bimoment.
- `GeneralizedBeam` and `SectionModes`: generalised beam theory with warping, shear lag,
  distortion, ribs and extensible walls.
- `TimoshenkoBeam`: exact Timoshenko and Euler-Bernoulli beam elements from a 6 × 6 or
  4 × 4 section stiffness.
- Imports from the laminate model are limited to `Laminate` and `integrate_abd`.
