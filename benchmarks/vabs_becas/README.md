# ASM's solid section against VABS and BECAS

`solid.SolidSectionAnalysis` (the Giavotto / BECAS method on 2D meshes) run on the
benchmark sections that BECAS and VABS publish results for, term by term, couplings
included. The tables are in [`RESULTS.md`](RESULTS.md) and every number behind them in
`results.json`; `tests/test_benchmark_vabs_becas.py` gates the same cases in CI.

```bash
python benchmarks/vabs_becas/run.py     # about 2.5 minutes; rewrites RESULTS.md and results.json
pytest tests/test_benchmark_vabs_becas.py
```

## Summary

| Case | Tool | What is checked | ASM against the published values |
| --- | --- | --- | --- |
| S1 isotropic square | BECAS | 6 diagonal terms, centres | **Identical on BECAS's mesh** (10 × 10 Q4, within 0.002 %). Converged, shear and torsion are 0.83 to 0.86 % lower than BECAS and equal to the exact torsion constant |
| S2 two-material square | BECAS | 6 diagonal terms, shear-torsion K26, extension-bending K35 | Within the 3-digit rounding of the published values on BECAS's mesh (largest 0.44 %, K35 = 1.13e-2) |
| S3 orthotropic square, 0°, 22.5°, 90° | BECAS | diagonal terms, shear-extension K13 and bending-torsion K46 at 22.5° | Within 0.012 % on BECAS's mesh, couplings included |
| C1 isotropic tube | BECAS | 6 diagonal terms | Within 0.15 %; ASM equals the exact EA, EI and GJ of the annulus, BECAS's polygonal mesh is 0.07 to 0.15 % below them |
| C2 half tube | BECAS | 8 terms, shear and tension centres | Within 0.11 %; shear centre -0.1206 (published -0.1206), tension centre -0.06053 (-0.06051) |
| Thick aluminium tube | VABS | 6 diagonal terms, mass | Within 0.02 %, except GJ: VABS's published 3.519e8 is 2.0 % above the exact 3.449e8, which ASM matches to 1e-6 |
| Multi-layer composite pipe | VABS | 6 diagonal terms, extension-twist K14, shear-bending K25 and K36 | **Within 0.2 % on every term** of Yu et al. 2012; within 0.5 % of Yu 2005; the third published VABS set differs from the other two by up to 2.5 % |

T6 meshes (the element a section-properties port would use) give the same results as Q8
to 4 to 5 digits on every case (`RESULTS.md`, `T6 %` column).

## What the comparison shows

1. **ASM and BECAS solve the same equations.** On BECAS's own 10 × 10 bilinear mesh of
   the user-guide squares, ASM (with a `Q4` element added for this) reproduces every
   published term, including the shear-extension and bending-torsion couplings of the
   22.5° orthotropic square, to the digits BECAS prints.
2. **The published BECAS square values carry their mesh's error.** Converged ASM shear and
   torsion stiffness are about 0.85 % below them. The bilinear mesh is the cause: refining
   it (20², 40², 80²) closes the gap at second order (0.21 %, 0.05 %, 0.01 %) onto the Q8
   value, and the converged S1 torsion stiffness equals Saint-Venant's series to 1e-6.
3. **VABS agrees on a composite section with couplings.** The pipe has two ply angles in
   each wall (0/90 in the flats, ±45 in the arcs) and nonzero extension-twist and
   shear-bending couplings; ASM converges onto Yu et al. 2012's VABS values on all nine
   published terms. The quadratic meshes converge at about second order there (ply
   interfaces and the flat-to-arc junctions), and at fourth order on the smooth cases.
4. **One VABS number disagrees with the closed form.** The Chen et al. tube's GJ is 2.0 %
   above G π (r_o⁴ - r_i⁴) / 2; every other term of that tube agrees with ASM and the closed
   forms to 0.02 %. It is reported, not gated.

## Conventions

- **Order.** BECAS's matrix is `[Vx, Vy, N, Mx, My, Mz]` with the beam along z; VABS's
  Timoshenko matrix is `[F1, F2, F3, M1, M2, M3]` = `[N, V2, V3, T, M2, M3]` with the beam
  along x1. Both sources' section coordinates are ASM's `(y, z)` unchanged, so
  `K_becas = K[p, p]` with `p = [3, 4, 0, 1, 2, 5]` (the README's `BEAMDYN_FROM_ONEDFEM`)
  and `K_vabs = K[q, q]` with `q = [0, 3, 4, 5, 1, 2]`. No sign changes are needed.
- **Fibre angles.** ASM's `theta` is measured from the beam axis toward the wall direction
  `s` (`solid/material.py`). With `s` along +x for the squares, BECAS's 22.5° is ASM's
  +22.5°; with `s` running clockwise round the pipe, the pipe's layup is ASM's angles
  unchanged. Both are fixed by the data, not chosen: flipping the pipe's angles flips the
  sign of all three couplings (`test_vabs_pipe_layup_sign_is_not_free`).
- **Materials.** BECAS's isotropic #2 is listed with ν = 0.2/α; that is a misprint, and its
  published S2 results need ν = 0.2. The orthotropic constants are BECAS's table 4.1 read
  with 1 = fibre (beam axis), 2 = section x, 3 = section y (G12 = 60, G13 = 50, G23 = 60,
  ν12 = 0.19, ν13 = 0.26, ν23 = 0.19).

## Provenance and limits

- BECAS: Blasques, *User's Manual for BECAS*, Risø-R-1785 (DTU), section 4.2.
  VABS: Chen, Yu and Capellaro, *Wind Energy* 13 (2010) 497-516 (the tube); Yu, Hodges and
  Ho, *Int. J. Eng. Sci.* 59 (2012) 40-64 (the pipe).
- The primary PDFs could not be read in full from the environment that built this
  benchmark. The input data and published values are taken from GXBeam's test suite
  (`byuflowlab/GXBeam.jl`, `test/section.jl` at commit `8541187`), which cites these
  sources and reproduces them with its own BECAS-type solver. The pipe's values were
  cross-checked against a second transcription, the PreVABS documentation's pipe example.
  The BECAS user guide's material table and case list were read directly and agree.
  **Before quoting these numbers upstream, check the S2 and C1 / C2 values against the
  user guide's tables**: C1 and C2's BECAS meshes are not stated, so they are compared on
  converged meshes only.
- Not yet covered: the BECAS user guide's C3, C4 (layered cylinder) and T1, T2 (three-cell)
  sections, the Chen et al. composite airfoil (MH-104), and the CAS / CUS box beams of
  Chandra et al. Their input data was not available here.
