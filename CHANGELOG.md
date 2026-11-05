# Changelog

## Unreleased

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
