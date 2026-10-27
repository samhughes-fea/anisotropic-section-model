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

Walls are laminates: the only things this package takes from
``composite_laminate_model`` are ``Laminate`` and ``integrate_abd``. It never
extracts properties from a shell model; that is
``composite_laminate_model.beam_section_properties``, and the two meet only in
tests and examples.
"""

from .thin_wall import ThinWalledSection, Wall, condensed_wall_stiffness
from .torsion import RestrainedTorsionBeam, TorsionResponse
from .gbt import GBTSolution, GeneralizedBeam, SectionModes
from .beam import BeamSolution, TimoshenkoBeam, euler_bernoulli_beam

__version__ = "0.1.0"

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
    "condensed_wall_stiffness",
    "euler_bernoulli_beam",
]
