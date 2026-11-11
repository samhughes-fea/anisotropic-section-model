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
Closed form     rectangle_properties, i_section_properties, channel_properties,
                section_properties_from_geometry

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
from .torsion import RestrainedTorsionBeam, TorsionResponse
from .gbt import GBTSolution, GeneralizedBeam, SectionModes
from .beam import BeamSolution, TimoshenkoBeam, euler_bernoulli_beam
from .closed_form import channel_properties, i_section_properties, rectangle_properties
from .geometry_to_properties import section_properties_from_geometry

__version__ = "0.3.0"

__all__ = [
    "BeamSolution",
    "GBTSolution",
    "GeneralizedBeam",
    "RestrainedTorsionBeam",
    "SectionModes",
    "ThinWalledSection",
    "TimoshenkoBeam",
    "TorsionResponse",
    "Wall",
    "channel_properties",
    "condensed_wall_stiffness",
    "euler_bernoulli_beam",
    "i_section_properties",
    "rectangle_properties",
    "section_properties_from_geometry",
]
