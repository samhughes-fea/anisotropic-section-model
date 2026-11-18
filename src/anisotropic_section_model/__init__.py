"""anisotropic_section_model - beam section theory for anisotropic thin-walled sections.

Public API overview
-------------------
Section         ThinWalledSection (Saint-Venant shear flows and GJ, warping function,
                shear centre, EΓ; Timoshenko 6 x 6 and Timoshenko-Vlasov 7 x 7 from
                thin-wall theory), Wall, condensed_wall_stiffness
Torsion         RestrainedTorsionBeam (Benscoter / Vlasov: bimoment, warping torque),
                TorsionResponse
GBT             GeneralizedBeam and SectionModes (generalized beam theory: warping and
                cross-section distortion, ribs; shear lag and, with extensible walls,
                hoop strain and Poisson end effects), GBTSolution
Beams           TimoshenkoBeam (exact Timoshenko / Euler-Bernoulli beam elements from a
                6 x 6 or 4 x 4 section stiffness, all couplings), BeamSolution,
                euler_bernoulli_beam
Blade export    BeamDynBlade (a spanwise sweep of a BladeGeometry, or sections, into
                OpenFAST BeamDyn primary and blade files and windIO 2.0 elastic
                properties; reads both)
Closed form     rectangle_properties, i_section_properties, channel_properties,
                section_properties_from_geometry, reduced_section_from_integrals,
                principal axes of the full section (not only the reduced one)

Subpackages moved from 1d-fem's ``section_integration`` (step 6 of the split; they are
folded into the API above in a later step):

- ``anisotropic_section_model.thin_walled``: wall laminates, condensed wall laws and the
  thin-walled section analysis (Saint-Venant, Benscoter warping, GBT modes, ply stress
  recovery, layup derivatives, taper);
- ``anisotropic_section_model.solid``: 2D finite-element sections for thick, solid and
  laminated cross-sections;
- ``section_analysis`` (the common ``SectionAnalysis`` result, in 1d-fem's beam axis x
  and strain order ``[eps_x, kappa_y, kappa_z, gamma_xy, gamma_xz, phi_x]``),
  ``ply_failure``, ``gbt_alignment``, ``closed_form``, ``geometry_to_properties``.

Walls are laminates: this package takes ``Laminate`` and ``integrate_abd`` from
``laminate_constitutive_model``, and its array ply-failure exposures for ``ply_failure``;
LCM laminates and materials plug into the moved code through ``from_laminate`` and
``from_material_card``. It never
extracts properties from a shell model; that is
``shell_fem.beam_section_properties``, and the two meet only in
tests and examples.
"""

from .thin_wall import ThinWalledSection, Wall, condensed_wall_stiffness
from .thin_walled.fields import (
    FORMULATION_CSV_SCHEMA,
    FormulationWriter,
    GaussField,
    PlyField,
    SectionField,
    WallField,
)
from .torsion import RestrainedTorsionBeam, TorsionResponse
from .gbt import GBTSolution, GeneralizedBeam, SectionModes
from .beam import BeamSolution, TimoshenkoBeam, euler_bernoulli_beam
from .beamdyn import BeamDynBlade
from .closed_form import channel_properties, i_section_properties, rectangle_properties
from .geometry_to_properties import (
    ReducedBeamSection,
    principal_axes_from_mass,
    principal_axes_from_moments,
    reduced_section_from_integrals,
    section_properties_from_geometry,
)

__version__ = "0.3.0"

__all__ = [
    "BeamDynBlade",
    "BeamSolution",
    "FORMULATION_CSV_SCHEMA",
    "FormulationWriter",
    "GBTSolution",
    "GaussField",
    "GeneralizedBeam",
    "PlyField",
    "RestrainedTorsionBeam",
    "SectionField",
    "SectionModes",
    "ThinWalledSection",
    "TimoshenkoBeam",
    "TorsionResponse",
    "Wall",
    "WallField",
    "channel_properties",
    "condensed_wall_stiffness",
    "euler_bernoulli_beam",
    "i_section_properties",
    "ReducedBeamSection",
    "rectangle_properties",
    "principal_axes_from_mass",
    "principal_axes_from_moments",
    "reduced_section_from_integrals",
    "section_properties_from_geometry",
]
