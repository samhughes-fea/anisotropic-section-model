# Examples

Worked examples that exercise the package end to end, from a single plate
solve to cross-checking against CalculiX. Each one lives in its own folder with
its script, the figures it produces and a README that walks through the model
and how to read the results.

| # | Example | What it shows | Main API |
|---|---|---|---|
| 01 | [Kinematic models vs thickness](01_kinematic_models/) | CLPT, FSDT, RZT and layerwise deflections from thin to thick plates, against Navier solutions | `FEAMesh`, kinematic models, `StructuralGlobalSolver` |
| 02 | [Through-thickness stresses](02_through_thickness_stresses/) | σx and τxz profiles in a thick [0/90/0] plate for each theory, with CalculiX overlaid | `ExactStressRecoverer` |
| 03 | [Result fidelity levels](03_result_fidelity/) | The same solution at Gauss points, elements, nodes and global totals, with convergence against Navier | `ResultsExtractor` |
| 04 | [Thermal warping](04_thermal_warping/) | Cure cool-down shapes of [0/90], [45/-45] and [0/90]s plates, against laminate theory | `HygrothermalState`, `HygrothermalEngine` |
| 05 | [Failure envelopes](05_failure_envelopes/) | Maximum stress, Tsai-Hill, Tsai-Wu and Hashin envelopes for a T300/5208 ply | failure criteria |
| 06 | [Progressive damage](06_progressive_damage/) | A [0/90]s strip in tension from first-ply failure to fibre failure | `ProgressiveDamageAnalysis` |
| 07 | [CalculiX cross-check](07_calculix_crosscheck/) | Native solver against CalculiX on the benchmark plate | kernels, `StressDistributionVerifier` |
| 08 | [Open hole with Gmsh](08_open_hole_gmsh/) | A graded Gmsh quad mesh around a hole; Kt of four laminates against Lekhnitskii | `GmshPlateBuilder`, `GmshMeshImporter` |

All examples are linear static analyses.

## Running

From the repository root:

```bash
pip install -e ".[examples]"     # the package plus matplotlib
python examples/run_all.py       # every example, figures into each folder
python examples/03_result_fidelity/result_fidelity.py   # or one at a time
```

The scripts also run straight from a clone without installing, because they add
`src/` to the import path. Example 07 needs CalculiX on the `PATH`
as `ccx` (`apt install calculix-ccx`) and skips itself without it. Example 02
adds a CalculiX overlay when `ccx` is present.

## Layout

```
examples/
├── README.md                   this index
├── run_all.py                  runs every example in order
├── _shared/
│   ├── plotting.py             plot style, colours, save(), simply_supported()
│   └── analytical.py           Navier series solutions used as references
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
