# Examples

Worked examples that exercise the package end to end, from a single plate
solve to cross-checking against CalculiX. Each one lives in its own folder with
its script, the figures it produces and a README that walks through the model
and how to read the results.

| # | Example | What it shows | Main API |
|---|---|---|---|
| 01 | [Kinematic models vs thickness](01_kinematic_models/) | CLPT, FSDT, RZT and layerwise deflections from thin to thick plates, against Navier solutions | `FEAMesh`, kinematic models, `StructuralGlobalSolver` |
| 02 | [Through-thickness stresses](02_through_thickness_stresses/) | σx and τxz profiles in a thick [0/90/0] plate for each theory, with equilibrium recovery and CalculiX overlaid | `ExactStressRecoverer`, `EquilibriumStressRecoverer` |
| 03 | [Result fidelity levels](03_result_fidelity/) | The same solution at Gauss points, elements, nodes and global totals, with convergence against Navier | `ResultsExtractor` |
| 04 | [Thermal warping](04_thermal_warping/) | Cure cool-down shapes of [0/90], [45/-45] and [0/90]s plates, against laminate theory | `HygrothermalState`, `HygrothermalEngine` |
| 05 | [Failure envelopes](05_failure_envelopes/) | Maximum stress, Tsai-Hill, Tsai-Wu and Hashin envelopes for a T300/5208 ply | failure criteria |
| 06 | [Progressive damage](06_progressive_damage/) | A [0/90]s strip in tension from first-ply failure to fibre failure | `ProgressiveDamageAnalysis` |
| 07 | [CalculiX cross-check](07_calculix_crosscheck/) | Native solver against CalculiX on the benchmark plate | kernels, `StressDistributionVerifier` |
| 08 | [Open hole with Gmsh](08_open_hole_gmsh/) | A graded Gmsh quad mesh around a hole; Kt of four laminates against Lekhnitskii | `GmshPlateBuilder`, `GmshMeshImporter` |
| 09 | [Sandwich panel](09_sandwich_panel/) | Foam-core sandwich: FSDT against RZT, layerwise and sandwich theory | `Ply`, `Laminate`, `MaterialCard` |
| 10 | [Bending-twist coupling](10_bending_twist_coupling/) | Tip twist of an off-axis cantilever against ply angle | clamped supports, `integrate_abd` |
| 11 | [Laminate failure envelope](11_laminate_failure_envelope/) | First-ply failure in N_x–N_y, with and without cure stress | `TsaiWuCriterion`, `HygrothermalState` |
| 12 | [Temperature gradient and moisture](12_temperature_gradient_and_moisture/) | Face-to-face temperature difference; moisture undoing cure warp | `HygrothermalState` gradients |
| 13 | [Stiffness polars](13_stiffness_polars/) | Ex, Gxy and νxy against direction for four stacks | `Laminate`, `integrate_abd` |
| 14 | [Ply-angle sweep](14_ply_angle_sweep/) | Best [±θ]2s angle for four load cases | failure criteria |
| 15 | [Solver scaling](15_solver_scaling/) | Runtime against DOFs for sparse, dense and CalculiX | `SparseLinearSolver`, `DenseLinearSolver` |
| 16 | [Material comparison](16_material_comparison/) | One plate in five library materials: stiffness, strength, mass | `create_default_library`, `ResultsExtractor` |
| 17 | [Pagano benchmark](17_pagano_benchmark/) | All four theories against the exact 3D solution; interlaminar shear and σz from equilibrium | `PaganoPlate`, `EquilibriumStressRecoverer` |
| 18 | [Wound pressure vessel](18_pressure_vessel_winding/) | Full 360° barrel, winding angle sweep against CLT, optimum near 54.7° | `CylindricalPanelBuilder`, `CylindricalDirection` |
| 19 | [Element comparison](19_element_comparison/) | Tri3, Quad4 and Quad8 on the open hole, Kt error against DOFs | `Tri3ShellElement`, `Quad8ShellElement`, `GmshPlateBuilder(element=...)` |
| 20 | [Load combinations](20_load_combinations/) | Body, pressure-field, line and thermal cases on a propped cantilever, solved once and combined | `solve_many`, `LoadCombination`, `add_spring` |

All examples are linear static analyses.

## Running

From the repository root:

```bash
pip install -e ".[examples]"     # the package plus matplotlib and Gmsh
python examples/run_all.py       # every example, figures into each folder
python examples/03_result_fidelity/result_fidelity.py   # or one at a time
```

The scripts also run straight from a clone without installing, because they add
`src/` to the import path. Examples 08 and 19 need Gmsh. Example 07 needs CalculiX on the `PATH`
as `ccx` (`apt install calculix-ccx`) and skips itself without it. Example 02
adds a CalculiX overlay when `ccx` is present.

## Layout

```
examples/
├── README.md                   this index
├── run_all.py                  runs every example in order
├── _shared/
│   ├── plotting.py             plot style, colours, save(), simply_supported()
│   ├── analytical.py           Navier series solutions used as references
│   └── membrane.py             one-element membrane cell for failure envelopes and sweeps
└── 0N_<name>/
    ├── README.md               walkthrough with the figures
    ├── <name>.py               the script
    └── figures/*.png           output (committed, so the READMEs render on GitHub)
```

## Plot conventions

- Colours are fixed per kinematic model across every figure: CLPT blue, FSDT
  orange, RZT green, layerwise yellow.
- Analytical and external references are drawn in black: dashed or dotted lines
  for closed-form solutions, hollow circles for CalculiX.
- Each figure's title states the finding; the grey note under the figure lists
  the model data.
