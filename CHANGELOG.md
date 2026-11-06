# Changelog

## Unreleased

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
