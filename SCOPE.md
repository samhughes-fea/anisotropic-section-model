# Scope

`anisotropic-section-model` (ASM) answers *what is this cross-section?*: the
stiffness, mass and stress recovery of one beam cross-section, isotropic or
anisotropic.

## In scope

- The 6 × 6 Timoshenko stiffness (and 4 × 4 classical), rigid-section mass,
  shear, tension and mass centres, and second-order terms.
- Thin-walled and solid (2D finite-element) section models, warping, GBT and
  station modifiers (pretwist, taper).
- Stress recovery, ply stresses and failure exposures of a section, and layup
  derivatives.
- Section builders (walls and laminates, 2D meshes, geometry-kernel cuts) and
  BeamDyn / windIO export of section tables.
- Section benchmarks against published and external results.

## Out of scope

| Topic | Where it belongs |
| --- | --- |
| Blade and section geometry | [implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel) |
| Ply and laminate constitutive behaviour | [laminate-constitutive-model](https://github.com/samhughes-fea/laminate-constitutive-model) |
| Shell and solid models of the whole blade | [shell-fem](https://github.com/samhughes-fea/shell-fem) |
| Anything along the span: beam elements, solves, load cases | [1d-fem](https://github.com/samhughes-fea/1d-fem) |

The detailed line with 1d-fem is in [docs/package_boundary.md](docs/package_boundary.md).
