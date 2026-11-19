# ASM's solid section against BECAS on the same meshes

`solid.SolidSectionAnalysis` and BECAS solve the same cross-section problem (Giavotto et al.;
Blasques and Stolpe 2012). On the same mesh, with the same element and materials, they should
give the same 6 x 6 stiffness matrix and the same shear and tension centres to round-off. This
benchmark checks that, term by term and couplings included, on:

- **BECAS's own validation meshes** (`becas_cases.EXAMPLES`): the squares, the cylinders C1 to
  C4 (C3 at six stiffness contrasts, C4 the layered cylinder), the three-cell sections T1 to
  T3, the Yu et al. composite pipe (W3) and BECAS's wind-turbine airfoil section. They are read
  from a local, licensed BECAS (`BECAS_HOME`); nothing of BECAS is copied here.
- **Composite sections built here** (`becas_cases.BUILT`), written to BECAS input by
  `becas_io.write_becas`: an MH-104-style two-web blade section, CAS and CUS box beams and a
  layered composite cylinder. Their geometry, materials and layups are this benchmark's own
  (stated in each builder) and chosen to exercise every coupling; they are not reproductions of
  published specimens and are not compared with published values.

```bash
export BECAS_HOME=/path/to/BECAS        # local licensed copy; optional
python benchmarks/becas_same_mesh/run.py            # rewrites RESULTS.md and results.json
python benchmarks/becas_same_mesh/run.py --becas    # also runs BECAS (MATLAB or Octave on PATH)
pytest tests/test_benchmark_becas_same_mesh.py
```

When BECAS can be run, `run.py` calls it through our driver `run_becas.m` on every case and
stores the resulting numbers (6 x 6, centres) in `becas_reference.json`. Only those numbers are
kept: BECAS's code, documents, example inputs and output files never enter this repository
(`_work/`, where the generated input and BECAS's output go, is git-ignored).

## What the comparison shows so far

ASM agrees with every full-precision reference on BECAS's meshes to round-off
(`RESULTS.md`; "max scaled" is `|ΔK_ij| / sqrt(K_ii K_jj)` over all 36 terms):

| reference on the same mesh | cases | max scaled | centres |
| --- | --- | --- | --- |
| BECAS's output shipped with the example | S2 (E1/E2 = 10), S3 0°, C1, C2, C4 | 2e-12 to 1e-11 | 1e-13 of the section size or better |
| BECAS's output, S1 | S1 | 8e-6, explained below | 2e-16 |
| VABS's output shipped with the example | C3 at E1/E2 = 1 to 1e5 | 4e-13 to 6e-10 | 2e-11 or better |
| BECAS user guide tables (3 to 5 digits) | C3, C4, T1, T2, T3 | within the printed rounding (largest 0.26 %) | within the printed rounding |
| VABS, Yu 2005 (pipe, W3 mesh) | W3 | within 0.13 % on all nine terms | |

The fibre and fibre-plane angle conventions in `becas_io` are fixed by the data: T2's
-45° flanges and the pipe's ±45° walls give BECAS's and VABS's coupling signs only with them
(`test_angle_signs_are_fixed_by_the_data`).

**Not done yet: BECAS itself on the built sections and on the examples without a shipped
output.** The machine that built this has BECAS but no MATLAB or Octave. The built cases run in
ASM and through the writer, and their BECAS rows read "pending" until `run.py --becas` is run
where a runtime exists; the tests then gate them automatically.

## Data problems found in BECAS's example folders

Found by running ASM on every example mesh; each is evidenced in `RESULTS.md`.

- `S2_1/BECASresults.txt` is the E1/E2 = 10 result: S2_1's two materials are identical, and the
  file matches ASM on the `S2_10` mesh to 1e-11.
- `S3_22.5/` and `S3_45/BECASresults.txt` are both the 0° square's result (they match ASM on
  `S3_0` to 1e-11 and have no couplings).
- `S1/BECASresults.txt` was made with G = 41.667; `S1/MATPROPS.in` now holds G = E / 2.4 =
  41.6667. The G-dependent terms differ by exactly 41.6667 / 41.667 - 1 = -8.0e-6; the others
  agree to 1e-9.
- The user guide's C3 table columns headed `K11, K22, K33, K44 = K55, K66` hold `K22, K33,
  K11, K66, K44 = K55`, and its T1 rows `K26` and `K35` are swapped (`becas_published.py` has
  the evidence and the corrected labels).
- T2's `MATPROPS.in` swaps G13 / G23 and nu13 / nu23 of orthotropic #1 relative to T3 and the
  guide's material table (ASM uses the file as given; T2 and T3 differ by under 1 %).

## Provenance

- BECAS: Blasques, *User's Manual for BECAS*, Risø-R-1785 (DTU), validation section; results
  computed with BECAS cite Blasques J.P., Stolpe M., *Multi-material topology optimization of
  laminated composite beam cross sections*, Composite Structures 94(11) (2012) 3278-3289.
- MH 104 coordinates: UIUC airfoil coordinates database (`mh104.dat`).
- VABS pipe values: `benchmarks/vabs_becas/reference.py` (Yu 2005 via the PreVABS documentation).
