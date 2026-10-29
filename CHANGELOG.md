# Changelog

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
