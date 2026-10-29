# Examples

Worked examples of the section theory, each checked against closed forms and shell
models built with [composite-laminate-model](https://github.com/samhughes-fea/composite-laminate-model).
01 and 02 take torsion beyond Saint-Venant, to restrained warping and cross-section
distortion; 03 takes the section stiffness to the Timoshenko 6 × 6 and the
Timoshenko-Vlasov 7 × 7; 04 and 05 do the same for bending and axial load; and 06 cuts
the sections from the implicit geometry. Each one lives in its own folder with its
script, the figures it produces and a README that walks through the model and how to
read the results. Example numbers above 06 (20, 24, 26 and so on) refer to
[composite-laminate-model's examples](https://github.com/samhughes-fea/composite-laminate-model/tree/main/examples/).

| # | Example | What it shows | Main API |
|---|---|---|---|
| 01 | [Box spar torsion](01_box_spar_torsion/) | Saint-Venant, Vlasov, Benscoter and GBT on a clamped composite box against closed forms and a shell model: twist, bimoment, warping stress, distortion under a web force couple, ribs | `ThinWalledSection`, `RestrainedTorsionBeam`, `GeneralizedBeam` |
| 02 | [Blade torsion hierarchy](02_blade_torsion_hierarchy/) | A three-cell blade section: Saint-Venant shear flows and GJ, the restrained-warping bimoment at the root, and GBT distortion under torque from skin loads, each against the shell | `ThinWalledSection.from_blade_mesh`, `GeneralizedBeam`, `SectionModes` |
| 03 | [Timoshenko and Vlasov section stiffness](03_timoshenko_vlasov_stiffness/) | The 6 × 6 (shear) and 7 × 7 (warping) section stiffness from thin-wall theory and from the shell: a composite box spar, shear deflection of short cantilevers, restrained twist, and six IEA 15 MW stations ([example 24](https://github.com/samhughes-fea/composite-laminate-model/tree/main/examples/24_iea15mw_section_properties/)) against the published 6 × 6 | `beam_section_properties(warping=True)`, `ThinWalledSection.timoshenko_stiffness`, `transform_beam_stiffness` |
| 04 | [Box spar bending and axial load](04_box_spar_bending_axial/) | Beyond Euler-Bernoulli on a composite box: shear deflection at L/h = 6, shear lag in the caps, distortion under a cap pressure, and the diffusion and Poisson end effects of an axial load | `TimoshenkoBeam`, `GeneralizedBeam(extensible=True)`, `ThinWalledSection.timoshenko_stiffness` |
| 05 | [Blade bending and axial hierarchy](05_blade_bending_axial_hierarchy/) | The same hierarchy on a three-cell blade section: flapwise tip force, shear lag across the spar caps at the root, skin suction and ribs, and an axial load fed into the caps alone | `TimoshenkoBeam`, `GeneralizedBeam`, `beam_section_properties` |
| 06 | [Blade from its implicit geometry](06_blade_from_implicit_geometry/) | Any of the 40 section layouts lofted along a blade, cut anywhere and meshed; the thin-walled section and the shell model from the same cut, against composite-laminate-model's [example 20](https://github.com/samhughes-fea/composite-laminate-model/tree/main/examples/20_blade_beam_properties/) region-based model; a tapered, twisted segment; flapwise webs under twist | `BladeGeometry.section_at`, `ThinWalledSection.from_geometry`, `BladeBuilder.from_geometry` |

All examples are linear static analyses.

## Running

From the repository root:

```bash
pip install -e ".[examples]"     # the package plus SciPy and matplotlib
python examples/run_all.py       # every example, figures into each folder
python examples/01_box_spar_torsion/box_spar_torsion.py   # or one at a time
```

The scripts also run straight from a clone without installing, because they add
`src/` to the import path (the laminate model and the geometry kernel must be
installed). Examples 03, 04 and 05 take several minutes each; `--reuse` redraws their
figures from the committed `data/results.json`.

## Layout

```
examples/
├── README.md                   this index
├── run_all.py                  runs every example in order
├── _shared/
│   ├── plotting.py             plot style, colours, save()
│   ├── iea15mw.py              IEA 15 MW blade stations as BladeBuilder regions and laminates
│   ├── data/                   the IEA 15 MW sections (and the WISDEM script that wrote them)
│   └── torsion.py              box spar shell, ribs, shear-flow torques, shell twist, bimoment
│                                and wall stress resultants
└── NN_<name>/
    ├── README.md               walkthrough with the figures
    ├── <name>.py               the script
    ├── figures/*.png           output (committed, so the READMEs render on GitHub)
    └── data/results.json       results of the slow runs, for --reuse
```

## Plot conventions

- Analytical and external references are drawn in black: dashed or dotted lines
  for closed-form solutions.
- Each figure's title states the finding; the grey note under the figure lists
  the model data.
