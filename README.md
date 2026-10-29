# anisotropic-section-model

[![tests](https://github.com/samhughes-fea/anisotropic-section-model/actions/workflows/tests.yml/badge.svg)](https://github.com/samhughes-fea/anisotropic-section-model/actions/workflows/tests.yml)

Beam-section theory for anisotropic thin-walled sections: blade, wing and spar
sections whose walls are composite laminates. Linear static.

- **Saint-Venant** section properties (`ThinWalledSection`): multi-cell shear flows and
  GJ, the warping function, shear centre and EΓ, and the Timoshenko 6 × 6 and
  Timoshenko-Vlasov 7 × 7 stiffness from thin-wall theory. Sections are built from
  wall segments, from a shell blade mesh (`from_blade_mesh`) or from a cut of the
  implicit geometry (`from_geometry`).
- **Restrained warping** (`RestrainedTorsionBeam`): Benscoter and Vlasov torsion, with
  the bimoment and the split of the torque into Saint-Venant and warping parts.
- **Generalised beam theory** (`GeneralizedBeam`, `SectionModes`): warping, shear lag,
  cross-section distortion, ribs and, with extensible walls, hoop strain and the
  Poisson end effects.
- **Exact beams** (`TimoshenkoBeam`): Timoshenko and Euler-Bernoulli beam elements from
  any 6 × 6 or 4 × 4 section stiffness, all couplings kept.

```
src/anisotropic_section_model/
├── thin_wall.py    ThinWalledSection: Saint-Venant shear flows, GJ, warping function, EΓ, 6 × 6 / 7 × 7
├── torsion.py      RestrainedTorsionBeam: Benscoter / Vlasov torsion, bimoment
├── gbt.py          GeneralizedBeam, SectionModes: GBT warping and distortion, ribs
└── beam.py         TimoshenkoBeam: exact Timoshenko / Euler-Bernoulli beams from a 6 × 6 or 4 × 4
tests/              pytest suite, with shell models as references
examples/           six worked examples, each with its script, figures and README
```

## Install

The package is not on PyPI yet. It depends on
[composite-laminate-model](https://github.com/samhughes-fea/composite-laminate-model) (walls are its `Laminate`s), which depends on
[implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel). Install both, then this package from a clone:

```bash
pip install "implicit-geometry-kernel @ git+https://github.com/samhughes-fea/implicit-geometry-kernel"
pip install "composite-laminate-model @ git+https://github.com/samhughes-fea/composite-laminate-model"
pip install -e .             # NumPy and the laminate model
pip install -e ".[sparse]"   # + SciPy sparse solves for GBT
pip install -e ".[examples]" # + SciPy and matplotlib for the examples
pip install -e ".[dev]"      # everything above, plus pytest and build
pytest                       # run the test suite
```

Python 3.11 or newer.

### Related packages

| Package | Holds | Depends on |
| --- | --- | --- |
| [implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel) | `Airfoil`, `Station`, `SkinRegion`, `ShearWeb`, `BladeGeometry` and the implicit section geometry: 40 structural layouts, wall midlines and cells at any span position | numpy, shapely, numba |
| [composite-laminate-model](https://github.com/samhughes-fea/composite-laminate-model) | materials, laminates, kinematics, shell and solid FE, solver, kernels, `BladeBuilder`, `beam_section_properties` | the geometry kernel |
| **anisotropic-section-model** (this one) | `ThinWalledSection`, `RestrainedTorsionBeam`, `GeneralizedBeam`, `TimoshenkoBeam`: thin-walled beam-section theory | `Laminate` and `integrate_abd` from the laminate model |

Walls are laminates, so the only names this package takes from the laminate model are
`Laminate` and `integrate_abd` (`tests/test_import_boundary.py` checks it). It never
extracts properties from a shell model: that is the laminate model's
`beam_section_properties`. The two meet in this package's tests and examples, which
check the section theory against shell models. `from_geometry` reads a cut of the
geometry kernel's `BladeGeometry` by its attributes, so the package does not import the
kernel either.

## Torsion: Saint-Venant, restrained warping and distortion

composite-laminate-model's `beam_section_properties` gives the Saint-Venant GJ of a
shell segment. Three tools here follow the torsional response beyond it, each adding
one physical contribution:

```python
from anisotropic_section_model import GeneralizedBeam, RestrainedTorsionBeam, ThinWalledSection

section = ThinWalledSection.from_blade_mesh(blade, z=5.0)       # or from_segments(...)
section.GJ, section.shear_flow, section.omega, section.shear_centre, section.EGamma

beam = RestrainedTorsionBeam.from_section(section, length=10.0)    # Benscoter; vlasov=True for Vlasov
r = beam.cantilever_tip_torque(1e3)      # twist, warping amplitude, bimoment, Saint-Venant / warping torque
r = beam.solve(distributed_torque=50.0, supports={0.0: (True, True)})

gbt = GeneralizedBeam(section, 10.0, warping="full", distortional_modes=8,
                      z_nodes=GeneralizedBeam.graded_nodes(10.0, 40, stations=[5.0]))
gbt.clamp(0.0).diaphragm(5.0).add_wall_load(10.0, gbt.walls_named("te_panel"), (0.0, -2e3))
sol = gbt.solve()
sol.mean_twist(z), sol.bimoment(z), sol.distortion(z, 0), sol.wall_resultants(z)
```

- **Saint-Venant** (`ThinWalledSection`). The section is a network of laminate walls.
  The warping function is the one that minimises the membrane shear energy, which
  gives the Bredt-Batho shear flows in any number of cells and the sectorial
  coordinate on open branches in one solve. GJ adds each wall's plate twisting. The
  shear centre is the pole that makes ω orthogonal to bending, and EΓ = ∫ Et ω² ds
  plus the walls' own bending.
- **Restrained warping** (`RestrainedTorsionBeam`). Benscoter's theory gives the
  warping its own amplitude φ(z), with shear stiffness S = Ip − J; Vlasov's ties it to
  the twist rate (S → ∞). Solved in closed form for cantilevers and by finite elements
  for any supports. Outputs: bimoment B = −EΓ φ', warping stress N_zz = B Et ω / EΓ,
  and the split of the torque into Saint-Venant and warping parts.
- **Distortion** (`GeneralizedBeam`, Generalised Beam Theory). Every section point
  gets an axial warping displacement and an in-plane displacement and rotation. The
  walls are Kirchhoff plate strips with their full stiffness, couplings included. The
  in-plane fields split into the three rigid modes and the distortional modes, which
  are free of work with the Saint-Venant shear flows and ordered from the softest.
  Choosing the modes gives each level: `warping="classical", distortional_modes=0` is
  Benscoter's beam (with Timoshenko bending), `warping="full"` adds shear lag,
  `distortional_modes=n` adds distortion, and `extensible=True` lets the walls stretch
  round the contour. Clamps, ribs (`diaphragm`), warping restraints, Saint-Venant
  torques, point forces and wall loads.

Examples [01](examples/01_box_spar_torsion/) (box spar) and
[02](examples/02_blade_torsion_hierarchy/) (three-cell blade section) check each layer
against closed forms and the shell model.

## Bending and axial load: shear, shear lag, distortion and end effects

The same hierarchy applies to bending and extension. `TimoshenkoBeam` solves a beam
from any 6 × 6 (or 4 × 4) section stiffness, and `GeneralizedBeam` adds the layers a
beam stiffness cannot hold:

```python
from anisotropic_section_model import TimoshenkoBeam

beam = TimoshenkoBeam(K6, length=10.0, n_elements=20)        # shear=False, or a 4 x 4: Euler-Bernoulli
beam = TimoshenkoBeam.with_nodes_at(K6, 10.0, points=[4.0])  # element ends at supports and point loads
sol = beam.clamp(0.0).pin(10.0).add_force(4.0, [0, -5e3, 0]).add_distributed([0, -2e3, 0]).solve()
sol.node_displacement(4.0), sol.displacement(z), sol.section_forces(z), sol.strains(z), sol.reaction(0.0)
```

- **Euler-Bernoulli → Timoshenko.** Shear deflection is a few percent on a slender
  blade and most of the answer on a short segment (half the tip deflection at
  L/h = 6). Each element is exact for the section compliance it is given, so one
  element per load or support point is enough for point and uniform loads, and the
  response inside an element is recovered exactly. All couplings are kept.
- **Shear lag** (`GeneralizedBeam`, warping modes). The axial stress across a wide
  spar cap is not the plane-section straight line near a clamp or a load
  introduction: it peaks at the webs and dips at mid-cap.
- **Distortion** (distortional modes). A pressure on one skin bends that skin into the
  section, which no beam stiffness can show. Ribs stop it.
- **End effects** (`extensible=True`). Keeping the hoop strain and the full wall ABD
  adds the Poisson restraint at a clamp and lets a point load diffuse sideways through
  the walls, both of which die out within about one section width.

Examples [04](examples/04_box_spar_bending_axial/) (box spar) and
[05](examples/05_blade_bending_axial_hierarchy/) (blade section) check each layer
against the shell model.

### From the implicit section geometry

A cut of the geometry kernel's `BladeGeometry` gives the thin-walled section directly:

```python
from implicit_geometry_kernel import BladeGeometry
from anisotropic_section_model import ThinWalledSection

geo = BladeGeometry(stations, "2D-F", skin_thickness=0.003, spar_cap_thickness=0.008,
                    shear_web_thickness=0.002, web_positions=(0.18, 0.48))
section = ThinWalledSection.from_geometry(geo.section_at(1.0), {"skin": skin, "cap": cap, "web": web})
```

Laminates bind by label, region or kind, and spar caps are merged into the skin they
are bonded to. On composite-laminate-model's example 20 section it matches
`from_blade_mesh` within 1% in EA, EI and GJ
([example 06](examples/06_blade_from_implicit_geometry/)).

## Examples

Six worked examples in [examples/](examples/), each with its script, figures and a
README on the model and the results: torsion of a box spar and of a three-cell blade
section, the 6 × 6 and 7 × 7 section stiffness (including six IEA 15 MW stations), the
bending and axial hierarchy on both sections, and sections cut from the implicit
geometry. Each checks itself against closed forms and shell models built with
composite-laminate-model.

```bash
pip install -e ".[examples]"
python examples/run_all.py
```

## Verification

- Thin-walled section properties against textbook closed forms: an isotropic box
  (Bredt GJ, Benscoter warping function, EΓ, Ip), an I-beam (open GJ and Γ) and a channel
  (shear centre and Γ), all exact. Benscoter and Vlasov finite elements against their
  closed forms, and GBT against shell models of clamped boxes under a tip torque and a
  web force couple, within 2% (examples 01 and 02 give the full comparison).
- Timoshenko and Vlasov section stiffness (`tests/test_beam_stiffness.py`): thin-wall
  theory gives the thin tube's shear factor of 1/2 and the channel's textbook shear
  centre; on a box with no Poisson coupling the shell's 6 × 6 matches thin-wall theory
  term by term within 0.5%, and its fitted EΓ and warping shear stiffness within 6% and
  15%.
- `TimoshenkoBeam` against closed-form cantilever and simply supported beams, with and
  without shear (`tests/test_timoshenko_beam.py`).
- `from_geometry` against `from_blade_mesh` and the shell model on the same section, and
  on all 40 layouts (`tests/test_section_from_geometry.py`).

## Limitations

- Linear and small-deflection.
- `ThinWalledSection`, `RestrainedTorsionBeam` and `GeneralizedBeam` describe prismatic
  beams with straight walls between section points; the closed-form section properties
  (and the thin-wall 6 × 6 and 7 × 7) use the uncoupled wall stiffnesses, with each wall
  stretching and shearing without curving (GBT keeps all couplings). Off-axis walls'
  couplings (bend-twist, extension-twist, shear-bending) come only from a shell model
  (example 03). A tapered blade is handled one cut at a time.

## License

MIT; see [LICENSE](LICENSE).
