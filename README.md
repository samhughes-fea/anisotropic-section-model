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
- **Blade export** (`BeamDynBlade`): a section at every station of a blade, written as
  an OpenFAST BeamDyn deck (6 × 6 stiffness and mass, reference axis, twist) and as
  windIO 2.0 elastic properties, and read back from both.

```
src/anisotropic_section_model/
├── thin_wall.py    ThinWalledSection: Saint-Venant shear flows, GJ, warping function, EΓ, 6 × 6 / 7 × 7
├── torsion.py      RestrainedTorsionBeam: Benscoter / Vlasov torsion, bimoment
├── gbt.py          GeneralizedBeam, SectionModes: GBT warping and distortion, ribs
├── beam.py         TimoshenkoBeam: exact Timoshenko / Euler-Bernoulli beams from a 6 × 6 or 4 × 4
├── beamdyn.py      BeamDynBlade: spanwise sweep to OpenFAST BeamDyn and windIO 2.0 files
├── solid_section.py SolidSection: 2D finite-element sections in the public frame
└── axes.py         the one public axis convention and every conversion out of it
tests/              pytest suite, with shell models as references
examples/           eleven worked examples, each with its script, figures, README and self-checks
benchmarks/         the solid section against published VABS and BECAS results (vabs_becas/)
```

## Install

The package is not on PyPI yet. It depends on
[laminate-constitutive-model](https://github.com/samhughes-fea/laminate-constitutive-model)
(walls are its `Laminate`s). The tests and examples also build shell models with
[shell-fem](https://github.com/samhughes-fea/shell-fem), which needs
[implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel).
Install them, then this package from a clone (laminate-constitutive-model v0.3.0 or
later, which has the ply exposures `ply_failure` uses):

```bash
pip install "laminate-constitutive-model @ git+https://github.com/samhughes-fea/laminate-constitutive-model@v0.3.0"
# for the tests and examples (the kernel needs the system GEOS library: libgeos-dev):
pip install "implicit-geometry-kernel @ git+https://github.com/samhughes-fea/implicit-geometry-kernel@K9"
pip install "shell-fem @ git+https://github.com/samhughes-fea/shell-fem@v0.2.0"
pip install -e .             # NumPy and the constitutive model
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
| [laminate-constitutive-model](https://github.com/samhughes-fea/laminate-constitutive-model) | materials, `Ply`, `Laminate`, ABD, the kinematic theories' section stiffness, failure criteria | numpy |
| [shell-fem](https://github.com/samhughes-fea/shell-fem) | shell and solid FE, solver, kernels, `BladeBuilder`, `beam_section_properties` | the geometry kernel and the constitutive model |
| **anisotropic-section-model** (this one) | `ThinWalledSection`, `RestrainedTorsionBeam`, `GeneralizedBeam`, `TimoshenkoBeam`: thin-walled beam-section theory | `Laminate` and `integrate_abd` from the constitutive model |

Walls are laminates, so the only names this package takes from the constitutive model are
`Laminate` and `integrate_abd` (`tests/test_import_boundary.py` checks it). It never
extracts properties from a shell model: that is shell-fem's
`beam_section_properties`. The two meet in this package's tests and examples, which
check the section theory against shell models. `from_geometry` reads a cut of the
geometry kernel's `BladeGeometry` by its attributes, so the package does not import the
kernel either.

## Axes and strain order

One public frame, defined in `anisotropic_section_model.axes`, for every top-level class
(`ThinWalledSection`, `SolidSection`, `TimoshenkoBeam`, `BeamDynBlade`, `FormulationWriter.from_section`):

- **z** is the beam axis, root to tip; **x** points to the trailing edge and **y** to
  the suction side (right-handed). This is the geometry kernel's and shell-fem's blade frame.
- Strains and forces are in BeamDyn's order, `axes.STRAINS` against `axes.FORCES`:
  `[gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z]` against `[Fx, Fy, N, Mx, My, Mz]`,
  with `gamma_x = u_x' - theta_y` and `gamma_y = u_y' + theta_x`. A warping degree of
  freedom (`vlasov_stiffness`) is a seventh entry, and its force is the bimoment.
- **Twist** is in degrees and is right-handed about +z, like every other rotation in the
  frame: a section twisted by `t` is the untwisted one turned by `axes.rotation6(t)`, so
  positive twist moves the leading edge towards the pressure side (towards feather). It is
  the same number as BeamDyn's `initial_twist` and windIO 2.0's `outer_shape.twist`.

Three other frames appear only at the package's boundaries. `axes` holds every conversion,
so nothing else spells them out:

| Frame | Where | Conversion |
| --- | --- | --- |
| Engine: beam axis x, section (y, z), `[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]` | the solvers in `thin_walled` and `solid` (from 1d-fem), `closed_form` | `axes.from_engine` / `to_engine`: a reordering, no signs. Engine (x, y, z) = public (z, x, y), so section coordinates are the same numbers |
| 1d-fem element order: the engine order with the sign of kappa_y flipped | `section_analysis.SectionAnalysis` | `axes.to_element` |
| BeamDyn and windIO: X to suction, Y to the trailing edge (a mirror image) | `BeamDynBlade` arrays and files | `axes.MIRROR`, `axes.to_beamdyn_frame`. Twist needs no conversion |
| Geometry kernel twist: `SectionCut.twist`, `Station(twist=...)`, left-handed about z | kernel inputs | `axes.twist_from_kernel` / `twist_to_kernel` (a sign flip). `BeamDynBlade.from_geometry` applies it |

```python
from anisotropic_section_model import axes
K_public = axes.from_engine(analysis.stiffness)              # engine 6 x 6 to BeamDyn order
K_public = axes.stiffness_about(analysis.stiffness, (x, y))  # ... about a point
```

The geometry kernel's own `write_windio` writes its twist without negating it, so its
twist has the opposite sign to windIO's and to `BeamDynBlade.write_windio`'s.

`ThinWalledSection` builds a `thin_walled.ThinWalledSectionAnalysis` of the same walls
(`section.analysis`) and takes its stiffness, centres and EΓ from it, so every wall
coupling is included. The classical thin-wall values it used before are kept in
`section.classical` and `classical_timoshenko_stiffness()` as a check; the two agree
for balanced, symmetric walls. For BeamDyn's own input, the stiffness is usually
wanted about the reference axis: `section.timoshenko_stiffness(reference)`.

`FormulationWriter` exposes the in-memory Gauss cache as arrays at four levels:
section (stiffness, centres, EΓ), wall (geometry and condensed wall law), Gauss
(warping, shear flow, wall strains) and ply (through-thickness recovery):

```python
from anisotropic_section_model import FormulationWriter

w = FormulationWriter(section.analysis, forces=[N, My, Mz, Vy, Vz, T])
w.section(), w.walls(), w.gauss(), w.ply()
w.to_csv("fields/")
```

## Torsion: Saint-Venant, restrained warping and distortion

shell-fem's `beam_section_properties` gives the Saint-Venant GJ of a
shell segment. Three tools here follow the torsional response beyond it, each adding
one physical contribution:

```python
from anisotropic_section_model import GeneralizedBeam, RestrainedTorsionBeam, ThinWalledSection

section = ThinWalledSection.from_blade_mesh(blade, z=5.0)       # or from_segments(...)
section.GJ, section.GJ_free, section.shear_flow, section.omega, section.shear_centre, section.EGamma

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
  warping its own amplitude φ(z), with shear stiffness S (Ip − J in thin-wall theory,
  from the coupled analysis here); Vlasov's ties it to
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

geo = BladeGeometry(stations, "2D-F-LC-0R", skin_thickness=0.003, spar_cap_thickness=0.008,
                    shear_web_thickness=0.002, web_positions=(0.18, 0.48), pivot_fraction=0.3)
section = ThinWalledSection.from_geometry(geo.section_at(1.0), {"skin": skin, "cap": cap, "web": web})
```

Laminates bind by label, region or kind, and spar caps are merged into the skin they
are bonded to. On shell-fem's example 20 section it matches
`from_blade_mesh` within 1% in EA, EI and GJ
([example 06](examples/06_blade_from_implicit_geometry/)).

## BeamDyn and windIO export

`BeamDynBlade` cuts a `BladeGeometry` at each station, builds the section of every cut,
and writes the beam model OpenFAST's BeamDyn reads, or the windIO 2.0 blade fields
WISDEM reads:

```python
from anisotropic_section_model import BeamDynBlade

blade = BeamDynBlade.from_geometry(geo, laminates, z=np.linspace(0.0, 20.0, 11))  # laminates: dict, or z -> dict
blade.write("blade_BeamDyn.dat", damping=[3e-3] * 6)  # primary file + blade_BeamDyn_blade.dat (OpenFAST 5)
blade.write("blade_BeamDyn.dat", version=4)           # the format OpenFAST 3 and 4 read
blade.write_windio("blade.yaml")                      # reference_axis, twist, elastic_properties
BeamDynBlade.read("IEA-15-240-RWT_BeamDyn.dat"), BeamDynBlade.from_windio("IEA-15-240-RWT.yaml")
```

Each station's 6 × 6 stiffness and mass (`ThinWalledSection.mass_matrix`) are taken
about the pitch axis in the chord-aligned frame and put in BeamDyn's axes (x to the
suction side, y to the trailing edge), the mirror image of the geometry kernel's; the
twist changes sign on the way, as BeamDyn's turns the leading edge towards the pressure
side. The IEA 15 MW windIO file written this way reproduces the official BeamDyn blade
file to 1e-12, and BeamDyn (OpenFAST 4.1 and 5.0) on a written deck matches
`TimoshenkoBeam` on the section ([example 07](examples/07_beamdyn_windio_export/)).

## Examples

Eleven worked examples in [examples/](examples/). Each has its script, figures and a
README on the physics and the results, and each checks itself against its references
(closed forms, BECAS and VABS, the IEA 15 MW files, shell models built with shell-fem). The
[index](examples/README.md) suggests a reading order:
1. **Foundations:** isotropic sections against the closed forms (08), and the solid
   section against every BECAS and VABS validation case (11).
2. **Composite couplings:** CAS and CUS box beams (09), and GJ against GJ_free in coupled
   torsion (10).
3. **Beyond Saint-Venant:** torsion, bending and axial load on a box spar and a
   three-cell blade section: restrained warping, shear lag, distortion, and the 6 × 6 and
   7 × 7 stiffness (01 to 05).
4. **From geometry to a turbine model:** sections cut from the implicit geometry (06),
   and a blade written for BeamDyn and windIO (07).

```bash
pip install -e ".[dev]"
python examples/run_all.py           # solve every example and run its checks
python examples/run_all.py --reuse   # re-check from the committed results, in seconds
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
- BeamDyn export (`tests/test_beamdyn_export.py`, example 07): the mass matrix against
  a thin box's closed forms; a twisted cut gives the untwisted local matrices; files
  read back exactly; the IEA 15 MW windIO file reproduces its official BeamDyn blade
  file. With `BEAMDYN_DRIVER` set, BeamDyn on a written deck matches `TimoshenkoBeam`
  under six tip loads and self-weight.
- Every worked example's checks (`tests/test_examples.py` on the committed results; CI's
  `examples` job solves them from scratch).

## Limitations

- Linear and small-deflection.
- `ThinWalledSection`, `RestrainedTorsionBeam` and `GeneralizedBeam` describe prismatic
  beams with straight walls between section points. The 6 × 6 and 7 × 7, the warping
  function, EΓ and Benscoter's S come from the coupled section analysis, so off-axis
  walls' couplings (bend-twist, extension-twist, shear-bending) are kept; the classical
  thin-wall values are in `section.classical`. `RestrainedTorsionBeam` has the twist as
  its only degree of freedom: it takes `GJ_free` (the other strains free, the default)
  or `GJ` (held), and for coupled bending and twist along the span use `TimoshenkoBeam`
  or `GeneralizedBeam`. GBT's energy keeps the full wall stiffness; its distortional
  modes are made free of work with the classical Saint-Venant shear flows. A tapered
  blade is handled one cut at a time.

## License

MIT; see [LICENSE](LICENSE).
