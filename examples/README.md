# Examples

Worked examples that exercise the package end to end, from a single plate
solve to blade structures. Examples 01–19 cover the laminate mechanics underneath;
20–26 are blade sections, spar caps and blade segments, with 24 to 26 validating
them on the IEA 15 MW reference blade; 27 and 28 take torsion beyond Saint-Venant,
to restrained warping and cross-section distortion, and 29 takes the section stiffness
to the Timoshenko 6 × 6 and the Timoshenko-Vlasov 7 × 7. Each one lives in its own folder with
its script, the figures it produces and a README that walks through the model
and how to read the results.

| # | Example | What it shows | Main API |
|---|---|---|---|
| 01 | [Kinematic models vs thickness](01_kinematic_models/) | CLPT, FSDT, RZT and layerwise deflections from thin to thick plates, against Navier solutions | `FEAMesh`, kinematic models, `StructuralGlobalSolver` |
| 02 | [Through-thickness stresses](02_through_thickness_stresses/) | σx and τxz in a thick [0/90/0] plate for each theory against exact 3D elasticity, with equilibrium recovery and CalculiX | `ExactStressRecoverer`, `EquilibriumStressRecoverer` |
| 03 | [Result fidelity levels](03_result_fidelity/) | The same solution at Gauss points, elements, nodes and global totals, with convergence against Navier | `ResultsExtractor` |
| 04 | [Thermal warping](04_thermal_warping/) | Cure cool-down shapes of [0/90], [45/-45] and [0/90]s plates, against laminate theory | `HygrothermalState`, `HygrothermalEngine` |
| 05 | [Failure envelopes](05_failure_envelopes/) | Maximum stress, Tsai-Hill, Tsai-Wu and Hashin envelopes for a T300/5208 ply | failure criteria |
| 06 | [Progressive damage](06_progressive_damage/) | A [0/90]s strip in tension from first-ply to fibre failure, against a CLT ply-discount analysis | `ProgressiveDamageAnalysis` |
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
| 18 | [Element comparison](18_element_comparison/) | All five shell elements on the open hole, Kt error against DOFs | `Tri3ShellElement`, `Quad8ShellElement`, `GmshPlateBuilder(element=...)` |
| 19 | [Load combinations](19_load_combinations/) | Body, pressure-field, line and thermal cases on a propped cantilever, solved once and combined | `solve_many`, `LoadCombination`, `add_spring` |

### Blade structures

| # | Example | What it shows | Main API |
|---|---|---|---|
| 20 | [Blade beam properties](20_blade_beam_properties/) | Section stiffness and centres from a shell model; bend-twist coupling from off-axis caps, against thin-walled beam theory and PreComp | `BladeBuilder`, `beam_section_properties` |
| 21 | [Spar cap ply drops](21_spar_cap_ply_drops/) | A tapered, twisted blade whose caps come from a ply book with ply drops, against a thin-walled beam; per-ply failure; ParaView output | `Layup`, `PlyDefinition`, `write_vtu` |
| 22 | [Ply drop in 3D](22_ply_drop_solid_detail/) | Interlaminar shear and peel at a cap ply drop from layered Hex20 solids inside a shell model; equilibrium checks and a mesh study of the corner singularity | `solid_block`, `merge_coincident_nodes`, `couple_shell_to_solid` |
| 23 | [Sandwich blade segment](23_blade_segment/) | Foam-cored panels, caps and webs under pressure, gravity and outboard loads; shell against beam | `distributing_link`, `beam_section_properties`, `write_vtu` |
| 24 | [IEA 15 MW section properties](24_iea15mw_section_properties/) | The published reference blade at six stations against PreComp and the published data; why skins are meshed on their mid-surface (torsion of offset skins on flat shells) | `BladeBuilder`, `beam_section_properties` |
| 25 | [CalculiX on a blade segment](25_blade_calculix_crosscheck/) | A 20 m IEA 15 MW segment under operating loads: deflection, twist and ply stresses against CalculiX S8R | `CalculiXKernelAdapter`, `AnalysisModel` |
| 26 | [Mid-surface meshing](26_mid_surface_meshing/) | Skins on the mould line with offsets or on their mid-surface: thick-tube torsion and six IEA 15 MW stations, for Quad4, flat and curved Quad8 | `BladeBuilder(mesh_surface=...)`, `BladeMesh.to_mid_surface` |
| 27 | [Box spar torsion](27_box_spar_torsion/) | Saint-Venant, Vlasov, Benscoter and GBT on a clamped composite box against closed forms and a shell model: twist, bimoment, warping stress, distortion under a web force couple, ribs | `ThinWalledSection`, `RestrainedTorsionBeam`, `GeneralizedBeam` |
| 28 | [Blade torsion hierarchy](28_blade_torsion_hierarchy/) | A three-cell blade section: Saint-Venant shear flows and GJ, the restrained-warping bimoment at the root, and GBT distortion under torque from skin loads, each against the shell | `ThinWalledSection.from_blade_mesh`, `GeneralizedBeam`, `SectionModes` |
| 29 | [Timoshenko and Vlasov section stiffness](29_timoshenko_vlasov_stiffness/) | The 6 × 6 (shear) and 7 × 7 (warping) section stiffness from thin-wall theory and from the shell: a composite box spar, shear deflection of short cantilevers, restrained twist, and six IEA 15 MW stations against the published 6 × 6 | `beam_section_properties(warping=True)`, `ThinWalledSection.timoshenko_stiffness`, `transform_beam_stiffness` |
| 30 | [Box spar bending and axial load](30_box_spar_bending_axial/) | Beyond Euler-Bernoulli on a composite box: shear deflection at L/h = 6, shear lag in the caps, distortion under a cap pressure, and the diffusion and Poisson end effects of an axial load | `TimoshenkoBeam`, `GeneralizedBeam(extensible=True)`, `ThinWalledSection.timoshenko_stiffness` |
| 31 | [Blade bending and axial hierarchy](31_blade_bending_axial_hierarchy/) | The same hierarchy on a three-cell blade section: flapwise tip force, shear lag across the spar caps at the root, skin suction and ribs, and an axial load fed into the caps alone | `TimoshenkoBeam`, `GeneralizedBeam`, `beam_section_properties` |

All examples are linear static analyses, and each one compares its results with a
reference: a closed-form or series solution, exact 3D elasticity, laminate or
thin-walled beam theory, CalculiX, PreComp or published data.

## Running

From the repository root:

```bash
pip install -e ".[examples]"     # the package plus matplotlib and Gmsh
python examples/run_all.py       # every example, figures into each folder
python examples/03_result_fidelity/result_fidelity.py   # or one at a time
```

The scripts also run straight from a clone without installing, because they add
`src/` to the import path. Examples 08 and 18 need Gmsh. Examples 07 and 25 need CalculiX on the `PATH`
as `ccx` (`apt install calculix-ccx`) and skip themselves without it. Example 24 reads
its reference data from a committed file; regenerating that file needs WISDEM
(`data/prepare_reference_data.py`). Example 02
adds a CalculiX overlay when `ccx` is present.

## Layout

```
examples/
├── README.md                   this index
├── run_all.py                  runs every example in order
├── _shared/
│   ├── plotting.py             plot style, colours, save(), simply_supported()
│   ├── analytical.py           Navier series solutions used as references
│   ├── membrane.py             one-element membrane cell for failure envelopes and sweeps
│   ├── iea15mw.py              IEA 15 MW blade stations as BladeBuilder regions and laminates
│   └── torsion.py              box spar shell, ribs, shear-flow torques, shell twist, bimoment
│                                and wall stress resultants (examples 27-31)
└── NN_<name>/
    ├── README.md               walkthrough with the figures
    ├── <name>.py               the script
    ├── figures/*.png           output (committed, so the READMEs render on GitHub)
    └── vtk/                    ParaView files from the blade examples (not committed)
```

## Plot conventions

- Colours are fixed per kinematic model across every figure: CLPT blue, FSDT
  orange, RZT green, layerwise yellow.
- Analytical and external references are drawn in black: dashed or dotted lines
  for closed-form solutions, hollow circles for CalculiX.
- Each figure's title states the finding; the grey note under the figure lists
  the model data.
