# Examples

Worked examples of the section theory. Each one explains the physics, shows the results
in figures, and compares them with a reference: a closed form, a published result
(BECAS, VABS, the IEA 15 MW reference files) or a shell model built with
[shell-fem](https://github.com/samhughes-fea/shell-fem). Each example also **checks
itself**. It ends with a table of checks against its references and fails if any is out
of tolerance, so running it is a test, and CI runs all of them.

Every example has its own folder holding the script, the figures it produces, and a
README that walks through the model and how to read the results. Example numbers of 20
and above (20, 24, 26 and so on) refer to
[shell-fem's examples](https://github.com/samhughes-fea/shell-fem/tree/main/examples/).

## Reading path

The numbers give the order the examples were written in. For a first read, this order
builds up from the simplest case:

| Step | # | Example | What it shows | Reference | Checks | Run time |
|---|---|---|---|---|---:|---:|
| **1 · Foundations** | 08 | [Isotropic sections](08_isotropic_sections/) | Both section engines where the answer is exact: thin-wall theory on a tube, slit tube, I-beam, channel and two-cell box (GJ, EΓ, shear centres); the solid section on rectangles and thick tubes; where thin-wall theory stops | closed forms | 39 | 25 s |
| | 11 | [Solid sections against BECAS and VABS](11_solid_sections_vs_becas_vabs/) | `SolidSectionAnalysis` on every published BECAS and VABS validation section, term by term: identical to BECAS on its own mesh; within 0.2 % of VABS on a coupled composite pipe | BECAS, VABS, closed forms | 114 | 1 min |
| **2 · Composite couplings** | 09 | [CAS and CUS box beams](09_composite_box_couplings/) | Extension-twist and bending-twist coupling from the ply angle; thin-wall against the solid section at every angle; what the couplings do to a shell cantilever | solid section, shell | 24 | 40 s |
| | 10 | [Coupled torsion](10_coupled_torsion/) | GJ against GJ_free: one stiffness with the other strains held, the other with the other forces free; a shell cantilever with a free and a held tip; restrained warping at the clamp | shell, solid section | 12 | 15 s |
| **3 · Beyond Saint-Venant** | 01 | [Box spar torsion](01_box_spar_torsion/) | Saint-Venant, Vlasov, Benscoter and GBT on a clamped composite box: twist, bimoment, warping stress, distortion under a web force couple, ribs | closed forms, shell | 10 | 20 s |
| | 02 | [Blade torsion hierarchy](02_blade_torsion_hierarchy/) | A three-cell blade section: Saint-Venant shear flows and GJ, the restrained-warping bimoment at the root, and GBT distortion under torque from skin loads | shell | 12 | 2 min |
| | 03 | [Timoshenko and Vlasov section stiffness](03_timoshenko_vlasov_stiffness/) | The 6 × 6 (shear) and 7 × 7 (warping) stiffness from thin-wall theory and from the shell: a composite box spar, short cantilevers, and six IEA 15 MW stations ([example 24](https://github.com/samhughes-fea/shell-fem/tree/main/examples/24_iea15mw_section_properties/)) against the published 6 × 6 | shell, IEA 15 MW | 16 | 30 s |
| | 04 | [Box spar bending and axial load](04_box_spar_bending_axial/) | Beyond Euler-Bernoulli on a composite box: shear deflection at L/h = 6, shear lag in the caps, distortion under a cap pressure, the diffusion and Poisson end effects of an axial load | shell | 9 | 45 s |
| | 05 | [Blade bending and axial hierarchy](05_blade_bending_axial_hierarchy/) | The same hierarchy on a three-cell blade section: flapwise tip force, shear lag across the spar caps, skin suction and ribs, an axial load fed into the caps | shell | 12 | 2 min |
| **4 · From geometry to a turbine model** | 06 | [Blade from its implicit geometry](06_blade_from_implicit_geometry/) | Any of the 40 section layouts lofted along a blade, cut anywhere and meshed; the thin-walled section and the shell model from the same cut, against shell-fem's [example 20](https://github.com/samhughes-fea/shell-fem/tree/main/examples/20_blade_beam_properties/) model; a tapered, twisted segment | shell-fem example 20 | 16 | 1 min |
| | 07 | [BeamDyn and windIO export](07_beamdyn_windio_export/) | A blade swept station by station into an OpenFAST BeamDyn deck and windIO 2.0 elastic properties: the IEA 15 MW windIO file reproduces the official BeamDyn blade file, and BeamDyn matches `TimoshenkoBeam` on a twisted, coupled section | IEA 15 MW files, BeamDyn | 14 | 10 s |
| | 12 | [Tapered beams and ply-thickness sensitivities](12_tapered_beams_layup_sensitivities/) | A box spar whose depth halves: the Resal coupling of the inclined caps (the cuts alone are 29 % too soft), beam elements whose stiffness varies along them; exact tip-deflection derivatives with respect to every ply thickness, per kg added, and the 2D-meshed section's thickness derivatives | shell, finite differences, thin-wall | 28 | 45 s |

The main API each one uses:

| # | Main API |
|---|---|
| 01 | `ThinWalledSection`, `RestrainedTorsionBeam`, `GeneralizedBeam` |
| 02 | `ThinWalledSection.from_blade_mesh`, `GeneralizedBeam`, `SectionModes` |
| 03 | `beam_section_properties(warping=True)`, `ThinWalledSection.timoshenko_stiffness`, `transform_beam_stiffness` |
| 04 | `TimoshenkoBeam`, `GeneralizedBeam(extensible=True)`, `ThinWalledSection.timoshenko_stiffness` |
| 05 | `TimoshenkoBeam`, `GeneralizedBeam`, `beam_section_properties` |
| 06 | `BladeGeometry.section_at`, `ThinWalledSection.from_geometry`, `BladeBuilder.from_geometry` |
| 07 | `BeamDynBlade.from_geometry`, `write`, `write_windio`, `from_windio` |
| 08 | `ThinWalledSection.from_segments`, `SolidSectionAnalysis`, `solid.structured` |
| 09 | `ThinWalledSection.timoshenko_stiffness`, `SolidSectionAnalysis`, `TimoshenkoBeam` |
| 10 | `ThinWalledSection.GJ_free`, `RestrainedTorsionBeam.from_section(other_strains=...)`, `GeneralizedBeam` |
| 11 | `SolidSectionAnalysis`, `solid.structured` (`arc_patch`, `rectangle_patch`) |
| 12 | `thin_walled.taper.section_taper`, `TimoshenkoBeam` (stiffness per node), `section_derivative`, `solid_section_derivative`, `BeamSolution.displacement_derivative` |

All examples are linear static analyses.

## Running

From the repository root:

```bash
pip install -e ".[examples]"         # the package plus SciPy and matplotlib (".[dev]" adds PyYAML for 07)
python examples/run_all.py           # every example: figures into each folder, then its checks
python examples/run_all.py 08 09     # only these
python examples/run_all.py --reuse   # redraw and re-check from the committed data/results.json (seconds)
python examples/08_isotropic_sections/isotropic_sections.py   # or one at a time
```

The scripts also run straight from a clone without installing, because they add
`src/` to the import path (laminate-constitutive-model, shell-fem and the geometry kernel must be
installed; the geometry kernel needs the GEOS library, `libgeos-dev`, or `GEOS_LIBRARY_PATH`).
Every example except 01 and 02 keeps its results in `data/results.json`, and `--reuse`
redraws its figures and re-runs its checks from them.

Two parts need resources CI does not have, and are skipped when they are missing:
- 03's IEA 15 MW stations need the private reference-blades catalogue;
- 07's BeamDyn runs need OpenFAST's `beamdyn_driver` (`BEAMDYN_DRIVER`). Without it, 07
  keeps the BeamDyn results of the last run that had one.

## Testing

Each example ends with `Checks` (`_shared/checks.py`): every comparison with its reference,
tolerance and source, printed as a table, and an `AssertionError` if any fails.

- `tests/test_examples.py` re-runs the checks of every example with committed results
  (`--reuse`, about 15 s), as part of `pytest`.
- CI's `examples` job solves every example from scratch, one job per example.

## Layout

```
examples/
├── README.md                   this index
├── run_all.py                  runs every example in order (or those named), fails if any check fails
├── _shared/
│   ├── checks.py               Checks: comparisons against references, with tolerances, as a test
│   ├── plotting.py             plot style, colours, save()
│   ├── composite_box.py        the CAS / CUS box beams as thin-wall, solid and shell models (09, 10)
│   ├── iea15mw.py              IEA 15 MW blade stations as BladeBuilder regions and laminates
│   ├── reference_blades.py     finds the private reference-blades checkout (IEA 15 MW sections)
│   └── torsion.py              box spar shell, ribs, shear-flow torques, shell twist, bimoment
│                                and wall stress resultants
└── NN_<name>/
    ├── README.md               walkthrough: the physics, the figures, the results and the checks
    ├── <name>.py               the script
    ├── figures/*.png           output (committed, so the READMEs render on GitHub)
    └── data/results.json       the results, for --reuse
```

## Plot conventions

- Analytical and external references are drawn in black: dashed or dotted lines
  for closed-form solutions.
- Each figure's title states the finding; the grey note under the figure lists
  the model data.
