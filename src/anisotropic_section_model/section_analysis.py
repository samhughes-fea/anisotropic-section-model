# anisotropic_section_model/section_analysis.py
"""Section constitutive data shared by every beam element: ``SectionAnalysis``.

A ``SectionAnalysis`` holds the section stiffness that elements turn into
their ``D`` matrices. It is built from the classical isotropic inputs
(``section.txt`` and ``material.txt``), from a thin-walled anisotropic
section analysis (``laminate.txt``, ``section_geometry.txt`` and
``element_section.txt``) or from a 2D section mesh (``section_mesh_2d.txt``
and ``element_section.txt``, ``solid/section_analysis.py``).

Frame and order
---------------
Every matrix of a ``SectionAnalysis`` is in this package's engine frame (``axes``): beam
axis ``x``, section axes ``(y, z)``, right-handed strains
``[eps_x, kappa_y, kappa_z, gamma_xy, gamma_xz, phi_x]`` and forces
``[N, M_y, M_z, V_y, V_z, T]`` with ``M_y = int sigma z dA`` and ``M_z = -int sigma y dA``.
That is the frame ``thin_walled.section_analysis`` and ``solid.section_analysis`` compute
in, so the builders here copy their matrices unchanged. ``frame`` is always ``"engine"``.

A beam code whose elements use other signs (1d-fem's ``B`` matrices have
``kappa_y = -theta_y'``) applies them itself, after the station modifiers
(``with_pretwist``, ``with_taper``); those refuse a section whose ``frame`` is not
``"engine"``, so a section is never re-signed twice.

Section-provider contract
-------------------------
``SectionAnalysis`` is what this package hands a beam code (1d-fem by default) for one
section at one station: no nodes, shape functions or mesh. ``SECTION_API_VERSION`` is the
version of that contract (``docs/package_boundary.md``): a beam code checks the major
number matches and the minor is at least the one it needs. Adding an optional field is a
minor change; renaming, re-signing or reshaping one is a major change.

Torsion and polar constants
---------------------------
Two section constants in m^4 are easily confused. The Saint-Venant torsion constant
``J_t`` enters only the stiffness (``G J_t``, N m^2); the polar second moment of area
``I_p = int (y^2 + z^2) dA`` enters only the mass (``rho I_p``, kg m, the twist rotary
inertia ``mass["I_p"] = mass_matrix()[3, 3]``) and the Wagner term ``I_p / A``. They are
equal only for a circle or a circular tube. ``section_constants`` holds the geometric
values (isotropic sections only); every mass moment is about the section origin.

Public API
----------
SECTION_API_VERSION, SectionAnalysis.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

#: version of the section-provider contract (``docs/package_boundary.md``): (major, minor)
SECTION_API_VERSION: Tuple[int, int] = (2, 0)

#: the only frame this package builds sections in
ENGINE_FRAME = "engine"

CLASSICAL_INDEX = (0, 1, 2, 5)  # [eps_x, kappa_y, kappa_z, phi_x]

#: keys of ``SectionAnalysis.section_constants`` and their units
SECTION_CONSTANTS = {"A": "m^2", "I_y": "m^4", "I_z": "m^4", "I_p": "m^4", "J_t": "m^4", "Gamma": "m^6"}


def _rectangle_points(E: float, A: float, I_y: float, I_z: float) -> Optional[npt.NDArray[np.float64]]:
    """3 x 3 Gauss points ``[y, z, E dA]`` of the rectangle with the given ``A``, ``I_y``, ``I_z``."""
    if A <= 0 or I_y <= 0 or I_z <= 0:
        return None
    h, b = np.sqrt(12.0 * I_y / A), np.sqrt(12.0 * I_z / A)
    x, w = np.polynomial.legendre.leggauss(3)
    return np.array([[0.5 * b * yi, 0.5 * h * zi, E * 0.25 * b * h * wy * wz]
                     for yi, wy in zip(x, w) for zi, wz in zip(x, w)])


@dataclass(frozen=True)
class SectionAnalysis:
    """Section stiffness and properties at one station, in the engine frame (module docstring).

    Every quantity is per unit length and about the section origin (the beam reference
    axis) unless its entry says otherwise.

    Attributes
    ----------
    stiffness : ndarray (6, 6)
        Timoshenko section stiffness on the right-handed strains
        ``[eps_x, kappa_y, kappa_z, gamma_xy, gamma_xz, phi_x]``.
    classical_stiffness : ndarray (4, 4)
        Shear-rigid stiffness for ``[eps_x, kappa_y, kappa_z, phi_x]``
        (inverse of the force-flexibility block, not a sub-block of ``stiffness``).
    warping_stiffness : float
        Vlasov ``E Gamma`` about the shear centre.
    shear_centre, tension_centre : (float, float)
        In section coordinates ``(y, z)`` about the beam reference axis.
    mass : dict
        Mass per length and its moments about the origin: ``m`` [kg/m]; the mass centre
        ``y_m``, ``z_m`` [m]; the mass moments ``I_yy = int rho z^2 dA``,
        ``I_zz = int rho y^2 dA``, ``I_yz = int rho y z dA`` and the twist rotary inertia
        ``I_p = int rho (y^2 + z^2) dA`` [kg m] (``rho I_p`` of an isotropic section, never
        ``rho J_t``).
        Principal axes of those moments about the mass centre are ``principal_angle``
        and ``principal_moments``.
    source : str
        ``"isotropic"``, ``"thin_walled"`` or ``"solid"`` (a section meshed in 2D).
    section_id : int or None
        Section id, when ``source`` is ``"thin_walled"`` or ``"solid"``.
    benscoter_stiffness : ndarray (8, 8) or None
        Benscoter stiffness for ``[eps_x, kappa_y, kappa_z, gamma_xy,
        gamma_xz, phi_x, d, psi']`` with ``d = psi - phi_x`` (thin-walled sections only).
    warping_inertia : float
        ``int rho_A omega^2 ds`` for the warping amplitude DOF.
    gbt_modes : int
        Cross-section distortional modes carried, ``M`` (0 when none were asked for).
    gbt_shear_modes : int
        Extra shear-lag warping modes carried, ``P``.
    gbt_sv_modes : int
        Warping modes spanning the Saint-Venant warping, ``n_sv`` (always carried by a GBT
        section; 0 without GBT matrices).
    third_order_stiffness : ndarray (8, 8) or None
        Third-order shear stiffness (Reddy, Levinson) for
        ``[eps_x, kappa_y, kappa_z, gamma_xy, gamma_xz, phi_x, chi_y, chi_z]``, where
        ``chi = gamma' - third_order_rate @ psi6`` stretches the walls through the slaved
        shear warping. Its leading 6 x 6 block is ``stiffness``.
    third_order_rate : ndarray (2, 6) or None
        ``R``: the shear-strain rate ``gamma'`` of the Saint-Venant solution
        is ``R @ psi6`` (zero for an uncoupled section).
    geometric_terms : ndarray (7, 6) or None
        Second-order section terms ``[N, S_y, S_z, W, V_y, V_z, Q_r]`` (``GEOMETRIC_TERMS``)
        per unit section force: ``geometric_terms @ F``. The terms are
        integrals over the section in its own right-handed ``(y, z)`` and feed the
        geometric stiffness (``docs/element_library/second_order.md``).
    benscoter_geometric_terms : ndarray (7, 2) or None
        The same terms per unit Benscoter strain ``[d, psi']`` (the bimoment's part).
    third_order_geometric_terms : ndarray (7, 2) or None
        The same terms per unit third-order strain ``[chi_y, chi_z]``.
    axial_points : ndarray (m, 3) or None
        Section quadrature ``[y, z, k]`` with ``sum k f = int E f dA`` (fourth-order energy).
    gbt_stiffness : ndarray or None
        GBT stiffness, square of size ``6 + 2(n_sv + P) + 3M``, for
        ``[psi6, b'(n_sv + P), b(n_sv + P), a''(M), a'(M), a(M)]``.
    gbt_mass : ndarray or None
        Section mass, square of size ``6 + (n_sv + P) + 2M``, for
        ``[u, v, w, theta_x, theta_y, theta_z, b(n_sv + P), a(M), a'(M)]``.
    gbt_geometric : (ndarray, ndarray) or None
        GBT second-order terms ``(X, Y)``, each ``(3 + M, 3 + M, n_gbt)``: the in-plane
        motion ``q = [v, w, theta_x, a(M)]`` does the work ``1/2 q'^T X q' + q'^T Y q``
        per length against the stresses of the GBT strains (``X @ E``).
        See ``gbt_geometric_terms``.
    gbt_brazier : (ndarray, ndarray) or None
        Brazier terms ``(T, Q)`` of the modes (``CrossSectionModes.brazier_tensor``):
        ``T`` (M, 2, n_gbt) and ``Q`` (M, M, 2, 2) on ``kappa = (kappa_y, kappa_z)``. See
        ``gbt_brazier_terms``.
    gbt_moments : ndarray or None
        ``(6 + (n_sv + P) + 2M, 30)``: moments of each field of the GBT mass order over the
        section (``CrossSectionModes.shape_moments``), used to align the bases of
        neighbouring sections (``gbt_alignment``).
    tapered : bool
        The section carries a taper coupling (``with_taper``).
    taper_parts : dict or None
        The taper's split by laminate (``SectionTaper.parts``), when it has one.
    frame : str
        ``"engine"`` (module docstring). A beam code that re-signs a copy for its elements
        marks the copy with its own name; ASM's station modifiers then refuse it.
    section_constants : dict or None
        Geometric constants of an isotropic section about its centroid (the origin), keys
        and units in ``SECTION_CONSTANTS``: ``A``, ``I_y``, ``I_z``, the polar moment
        ``I_p``, the Saint-Venant torsion constant ``J_t`` and the warping constant
        ``Gamma``. ``None`` for composite sections, which have stiffness-weighted
        constants only.
    """

    stiffness: npt.NDArray[np.float64]
    classical_stiffness: npt.NDArray[np.float64]
    warping_stiffness: float = 0.0
    shear_centre: Tuple[float, float] = (0.0, 0.0)
    tension_centre: Tuple[float, float] = (0.0, 0.0)
    mass: Dict[str, float] = field(default_factory=dict)
    source: str = "isotropic"
    section_id: Optional[int] = None
    benscoter_stiffness: Optional[npt.NDArray[np.float64]] = None
    warping_inertia: float = 0.0
    gbt_modes: int = 0
    gbt_stiffness: Optional[npt.NDArray[np.float64]] = None
    gbt_mass: Optional[npt.NDArray[np.float64]] = None
    gbt_shear_modes: int = 0
    gbt_sv_modes: int = 0
    third_order_stiffness: Optional[npt.NDArray[np.float64]] = None
    third_order_rate: Optional[npt.NDArray[np.float64]] = None
    geometric_terms: Optional[npt.NDArray[np.float64]] = None
    benscoter_geometric_terms: Optional[npt.NDArray[np.float64]] = None
    third_order_geometric_terms: Optional[npt.NDArray[np.float64]] = None
    axial_points: Optional[npt.NDArray[np.float64]] = None
    gbt_geometric: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    gbt_moments: Optional[npt.NDArray[np.float64]] = None
    tapered: bool = False
    taper_parts: Optional[Dict[int, npt.NDArray[np.float64]]] = None
    gbt_brazier: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    frame: str = ENGINE_FRAME
    section_constants: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        K = np.asarray(self.stiffness, dtype=np.float64)
        C = np.asarray(self.classical_stiffness, dtype=np.float64)
        if K.shape != (6, 6) or C.shape != (4, 4):
            raise ValueError("stiffness must be (6, 6) and classical_stiffness (4, 4)")
        if not (np.allclose(K, K.T, rtol=1e-10, atol=0.0) and np.allclose(C, C.T, rtol=1e-10, atol=0.0)):
            raise ValueError("section stiffness must be symmetric")
        object.__setattr__(self, "stiffness", K)
        object.__setattr__(self, "classical_stiffness", C)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_isotropic(
        cls,
        *,
        E: float,
        G: float,
        A: float,
        I_y: float,
        I_z: float,
        J_t: float,
        kappa: float = 5.0 / 6.0,
        Gamma: float = 0.0,
        rho: float = 0.0,
        I_p: Optional[float] = None,
        y_sc: float = 0.0,
        z_sc: float = 0.0,
        I_x: Optional[float] = None,
    ) -> "SectionAnalysis":
        """Uncoupled section from classical inputs: ``diag(EA, EI_y, EI_z, kGA, kGA, GJ_t)``.

        The inputs are geometric constants of the section about its centroid, which is the
        section origin here; ``E``, ``G`` [Pa] and ``rho`` [kg/m^3] turn them into the
        stiffness and the mass per length. They are kept as ``section_constants``. Two
        torsion constants [m^4] mean different things:

        - ``J_t``, the Saint-Venant torsion constant (independent of the reference axis),
          gives only the torsional stiffness ``stiffness[5, 5] = G J_t`` [N m^2].
        - ``I_p``, the polar second moment of area ``int (y^2 + z^2) dA`` about the
          centroid, gives only the twist rotary inertia ``mass["I_p"] = rho I_p`` [kg m]
          (``mass_matrix()[3, 3]``, about the origin) and the Wagner term ``I_p / A``.
          ``None`` (or ``<= 0``) takes ``I_y + I_z``.

        They are equal only for a circular section (solid or tube); otherwise ``J_t < I_p``.
        The mass holds ``m = rho A`` [kg/m] and the bending rotary inertias
        ``I_yy = rho I_y``, ``I_zz = rho I_z`` [kg m], all about the origin. ``I_x`` is the
        contract-v1 name of ``I_p``, kept for one release (``DeprecationWarning``).

        ``y_sc``, ``z_sc`` are recorded as the shear centre only; the stiffness
        stays diagonal and the mass takes no shear-centre offset.
        The third-order shear terms assume a solid rectangle (Reddy's cubic shear
        warping): ``EI_z / 84`` for ``chi_y`` and ``EI_y / 84`` for ``chi_z``. The
        second-order terms take the centroid at the origin and a doubly symmetric
        section (Wagner term ``W = N I_p / A``); the cubic shear warping adds none.
        """
        if I_x is not None:
            import warnings

            if I_p is not None:
                raise TypeError("give the polar moment as I_p only (I_x is its old name)")
            warnings.warn("SectionAnalysis.from_isotropic(I_x=...) is now I_p=...", DeprecationWarning,
                          stacklevel=2)
            I_p = I_x
        if I_p is None or I_p <= 0:
            I_p = I_y + I_z
        D = np.diag([E * A, E * I_y, E * I_z, kappa * G * A, kappa * G * A, G * J_t])
        C = np.diag([E * A, E * I_y, E * I_z, G * J_t])
        mass = {"m": rho * A, "y_m": 0.0, "z_m": 0.0, "I_yy": rho * I_y,
                "I_zz": rho * I_z, "I_yz": 0.0, "I_p": rho * I_p}
        constants = {"A": float(A), "I_y": float(I_y), "I_z": float(I_z), "I_p": float(I_p),
                     "J_t": float(J_t), "Gamma": float(Gamma)}
        D8 = np.zeros((8, 8))
        D8[:6, :6] = D
        D8[6, 6], D8[7, 7] = E * I_z / 84.0, E * I_y / 84.0
        H = np.zeros((7, 6))
        H[0, 0] = 1.0                  # N
        H[1, 2] = -1.0                 # S_y = int sigma y = -M_z
        H[2, 1] = 1.0                  # S_z = int sigma z = M_y
        H[3, 0] = I_p / A if A > 0 else 0.0
        H[4, 3] = H[5, 4] = 1.0        # V_y, V_z
        return cls(D, C, E * Gamma, (float(y_sc), float(z_sc)), (0.0, 0.0), mass, "isotropic", None,
                   third_order_stiffness=D8, third_order_rate=np.zeros((2, 6)),
                   geometric_terms=H, third_order_geometric_terms=np.zeros((7, 2)),
                   axial_points=_rectangle_points(E, A, I_y, I_z), section_constants=constants)

    @classmethod
    def from_matrices(
        cls,
        stiffness: npt.ArrayLike,
        mass: Optional[npt.ArrayLike] = None,
        *,
        frame: str = "engine",
        section_id: Optional[int] = None,
        shear_centre: Optional[Tuple[float, float]] = None,
        tension_centre: Optional[Tuple[float, float]] = None,
        rtol: float = 1e-8,
    ) -> "SectionAnalysis":
        """A section given only as its 6 x 6 stiffness and 6 x 6 mass per length.

        For section tables from elsewhere: a BeamDyn blade file, windIO 2.0 elastic
        properties, VABS or BECAS output. Both matrices are about the beam reference axis
        (the section origin). ``frame`` says how they are written:

        - ``"engine"``: beam axis x, right-handed strains
          ``[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]`` (the order of ``thin_walled``
          and ``solid``), mass on ``[u_x, u_y, u_z, theta_x, theta_y, theta_z]`` in the form of
          ``mass_matrix``;
        - ``"public"``: this package's public frame (``axes``): beam axis z, BeamDyn's order
          ``[gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z]``, mass in the form of
          ``axes.mass_matrix``;
        - ``"beamdyn"``: BeamDyn's own (mirrored) blade frame, as a BeamDyn blade file or
          ``BeamDynBlade`` writes it, untwisted (the station's twist is the beam's, not
          the section's).

        The stiffness may carry every coupling. The mass must be a rigid-section mass (a mass
        per length, its centre and the second moments; the polar moment may differ from
        ``I_yy + I_zz``); anything else raises ``ValueError``. ``mass=None`` gives a
        massless section (static analyses only).

        The centres default to the ones the stiffness implies: the tension centre is where an
        axial force bends nothing (classical compliance), the shear centre where a shear force
        twists nothing (Timoshenko compliance). The section carries no second-order, warping,
        third-order or GBT data, so beam elements that need them refuse it.
        """
        from anisotropic_section_model import axes

        K = np.asarray(stiffness, dtype=np.float64)
        if K.shape != (6, 6):
            raise ValueError(f"stiffness must be 6 x 6, got {K.shape}")
        if not np.allclose(K, K.T, rtol=rtol, atol=rtol * np.abs(K).max()):
            raise ValueError("stiffness must be symmetric")
        K = 0.5 * (K + K.T)
        M = None if mass is None else np.asarray(mass, dtype=np.float64)
        if M is not None and M.shape != (6, 6):
            raise ValueError(f"mass must be 6 x 6, got {M.shape}")
        if frame == "beamdyn":
            K = axes.MIRROR.T @ K @ axes.MIRROR
            M = None if M is None else axes.MIRROR.T @ M @ axes.MIRROR
            frame = "public"
        if frame == "public":
            K_rh = axes.to_engine(K)
            props = None if M is None else _mass_from_public(M, rtol)
        elif frame == "engine":
            K_rh = K
            props = None if M is None else _mass_from_engine(M, rtol)
        else:
            raise ValueError(f"frame must be 'engine', 'public' or 'beamdyn', got {frame!r}")
        F = np.linalg.inv(K_rh)
        idx = np.array(CLASSICAL_INDEX)
        C_rh = np.linalg.inv(F[np.ix_(idx, idx)])
        C_rh = 0.5 * (C_rh + C_rh.T)
        if tension_centre is None:
            Fc = np.linalg.inv(C_rh)
            # an axial force N at (y, z) has M_y = N z, M_z = -N y: no curvature there
            A = np.array([[-Fc[1, 2], Fc[1, 1]], [-Fc[2, 2], Fc[2, 1]]])
            tension_centre = tuple(float(v) for v in np.linalg.solve(A, -Fc[1:3, 0]))
        if shear_centre is None:
            # a shear force V at (y, z) has the torque T = y V_z - z V_y: no twist there
            shear_centre = (float(-F[5, 4] / F[5, 5]), float(F[5, 3] / F[5, 5]))
        return cls(
            K_rh,
            C_rh,
            0.0,
            (float(shear_centre[0]), float(shear_centre[1])),
            (float(tension_centre[0]), float(tension_centre[1])),
            {} if props is None else props,
            "matrix",
            None if section_id is None else int(section_id),
        )

    @classmethod
    def from_thin_walled(
        cls,
        analysis,
        section_id: Optional[int] = None,
        gbt_modes: int = 0,
        gbt_shear_modes: int = 0,
        gbt: bool = False,
    ) -> "SectionAnalysis":
        """From a ``ThinWalledSectionAnalysis`` (its right-handed matrices, unchanged).

        ``gbt=True`` (implied by ``gbt_modes`` or ``gbt_shear_modes`` > 0) also builds the
        GBT matrices with that many distortional and extra shear-lag modes
        (``CrossSectionModes``). The modes carry the hoop bending, so a GBT section is
        analysed with ``hoop="restrained"``: an analysis with another hoop option is
        rebuilt that way first.
        """
        if (gbt or gbt_modes or gbt_shear_modes) and analysis.hoop != "restrained":
            from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis

            analysis = ThinWalledSectionAnalysis(analysis.section, analysis.laminates, hoop="restrained",
                                                 n_sub=analysis.n_sub)
        K = np.array(analysis.stiffness, dtype=np.float64)
        C = np.array(analysis.classical_stiffness, dtype=np.float64)
        try:
            KB = np.array(analysis.benscoter_stiffness, dtype=np.float64)
            w_inertia = float(analysis.warping_inertia)
            K3 = np.array(analysis.third_order_stiffness, dtype=np.float64)
            R3 = np.array(analysis.third_order_rate, dtype=np.float64)
        except ValueError:
            KB, w_inertia, K3, R3 = None, 0.0, None, None
        H = np.array(analysis.geometric_terms, dtype=np.float64)
        try:
            XB, X3 = analysis.benscoter_geometric_terms, analysis.third_order_geometric_terms
        except ValueError:
            XB = X3 = None
        K_gbt = M_gbt = G_gbt = mom = brazier = None
        n_sv = 0
        if gbt or gbt_modes or gbt_shear_modes:
            from anisotropic_section_model.thin_walled.cross_section_modes import CrossSectionModes

            modes = CrossSectionModes(analysis)
            n_sv = modes.n_sv_warping
            K_gbt = modes.stiffness(gbt_modes, gbt_shear_modes)
            M_gbt = modes.mass(gbt_modes, gbt_shear_modes)
            mom = modes.shape_moments(gbt_modes, gbt_shear_modes)
            G_gbt = tuple(modes.geometric_tensor(gbt_modes, gbt_shear_modes))
            brazier = tuple(modes.brazier_tensor(gbt_modes, gbt_shear_modes))
        out = cls(
            K,
            C,
            float(analysis.warping_constant),
            tuple(analysis.shear_centre),
            tuple(analysis.tension_centre),
            dict(analysis.mass_properties),
            "thin_walled",
            None if section_id is None else int(section_id),
            KB,
            w_inertia,
            int(gbt_modes),
            K_gbt,
            M_gbt,
            int(gbt_shear_modes),
            int(n_sv),
            K3,
            R3,
            H,
            XB,
            X3,
            analysis.axial_points,
            G_gbt,
            gbt_brazier=brazier,
        )
        if M_gbt is not None:
            # rigid block with the walls' own rotary inertia, as mass_matrix()
            M_gbt[:6, :6] = out.mass_matrix()
            out = dataclasses.replace(out, gbt_moments=mom)
        return out

    @classmethod
    def from_solid(cls, analysis, section_id: Optional[int] = None, gbt_modes: int = 0,
                   gbt_shear_modes: int = 0, gbt: bool = False) -> "SectionAnalysis":
        """From a ``SolidSectionAnalysis`` (a section meshed in 2D; its right-handed matrices,
        unchanged). Carries the 6 x 6, classical stiffness, centres, ``E Gamma``, mass,
        second-order terms, axial quadrature and the Benscoter (restrained warping) data; no
        third-order data.

        ``gbt=True`` (implied by ``gbt_modes`` > 0) also builds the GBT matrices of the 2D
        section with that many distortional modes (``solid.cross_section_modes``). 2D
        sections have no extra shear-lag modes, so ``gbt_shear_modes`` must be 0.
        """
        K_gbt = M_gbt = G_gbt = mom = brazier = None
        n_sv = 0
        if gbt or gbt_modes or gbt_shear_modes:
            from anisotropic_section_model.solid.cross_section_modes import SolidCrossSectionModes

            modes = SolidCrossSectionModes(analysis, max_modes=max(int(gbt_modes), 1))
            n_sv = modes.n_sv_warping
            K_gbt = modes.stiffness(gbt_modes, gbt_shear_modes)
            M_gbt = modes.mass(gbt_modes, gbt_shear_modes)
            mom = modes.shape_moments(gbt_modes, gbt_shear_modes)
            G_gbt = tuple(modes.geometric_tensor(gbt_modes, gbt_shear_modes))
            brazier = tuple(modes.brazier_tensor(gbt_modes, gbt_shear_modes))
        out = cls(
            np.array(analysis.stiffness, dtype=np.float64),
            np.array(analysis.classical_stiffness, dtype=np.float64),
            float(analysis.warping_constant),
            tuple(analysis.shear_centre),
            tuple(analysis.tension_centre),
            dict(analysis.mass_properties),
            "solid",
            None if section_id is None else int(section_id),
            benscoter_stiffness=np.array(analysis.benscoter_stiffness, dtype=np.float64),
            warping_inertia=float(analysis.warping_inertia),
            geometric_terms=np.array(analysis.geometric_terms, dtype=np.float64),
            benscoter_geometric_terms=analysis.benscoter_geometric_terms,
            axial_points=analysis.axial_points,
            gbt_modes=int(gbt_modes),
            gbt_stiffness=K_gbt,
            gbt_mass=M_gbt,
            gbt_shear_modes=int(gbt_shear_modes),
            gbt_sv_modes=int(n_sv),
            gbt_geometric=G_gbt,
            gbt_brazier=brazier,
        )
        if M_gbt is not None:
            M_gbt[:6, :6] = out.mass_matrix()
            out = dataclasses.replace(out, gbt_moments=mom)
        return out

    # ------------------------------------------------------------------ #
    def gbt_indices(self, n_modes: int, n_shear: int = 0) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Rows of the stored GBT stiffness and mass (or wall load) for the first modes.

        The element carries the ``gbt_sv_modes`` Saint-Venant warping modes, the first
        ``n_shear`` shear-lag modes and the first ``n_modes`` distortional modes.
        """
        M, P = self.gbt_modes, self.gbt_shear_modes
        if self.gbt_stiffness is None or not (0 <= n_modes <= M and 0 <= n_shear <= P):
            raise ValueError(
                f"section carries {M} cross-section and {P} shear-lag modes"
                f"{'' if self.gbt_stiffness is not None else ' (no GBT matrices)'}; "
                f"asked for {n_modes} and {n_shear}"
            )
        Pt = self.gbt_sv_modes + P
        j = np.arange(self.gbt_sv_modes + n_shear)
        k = np.arange(n_modes)
        oa = 6 + 2 * Pt
        ik = np.r_[np.arange(6), 6 + j, 6 + Pt + j, oa + k, oa + M + k, oa + 2 * M + k]
        im = np.r_[np.arange(6), 6 + j, 6 + Pt + k, 6 + Pt + M + k]
        return ik, im

    def gbt_matrices(self, n_modes: int, n_shear: int = 0) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """GBT stiffness and mass of the first modes (see ``gbt_indices``)."""
        ik, im = self.gbt_indices(n_modes, n_shear)
        return self.gbt_stiffness[np.ix_(ik, ik)].copy(), self.gbt_mass[np.ix_(im, im)].copy()

    def gbt_geometric_terms(self, n_modes: int, n_shear: int = 0) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """GBT second-order terms ``(X, Y)`` of the first modes, on ``q = [v, w, theta_x, a(n_modes)]``
        and the strains of ``gbt_matrices(n_modes, n_shear)``."""
        if self.gbt_geometric is None:
            raise ValueError("the section has no GBT second-order terms")
        ik, _ = self.gbt_indices(n_modes, n_shear)
        iq = np.r_[0, 1, 2, 3 + np.arange(n_modes)]
        return tuple(T[np.ix_(iq, iq, ik)].copy() for T in self.gbt_geometric)

    def gbt_brazier_terms(self, n_modes: int, n_shear: int = 0):
        """Brazier terms ``(T, Q)`` of the first modes, on the strains of ``gbt_matrices(n_modes,
        n_shear)``, or ``None`` when the section has none: the fibre strain of the beam
        curvature at the modes' in-plane motion is ``sum_k a_k g_k . kappa`` and its work is
        ``sum_k a_k kappa^T T_k E + 1/2 sum_kl a_k a_l kappa^T Q_kl kappa``."""
        if self.gbt_brazier is None:
            return None
        ik, _ = self.gbt_indices(n_modes, n_shear)
        T, Q = self.gbt_brazier
        return T[:n_modes][:, :, ik].copy(), Q[:n_modes, :n_modes].copy()

    @property
    def is_coupled(self) -> bool:
        """True when any off-diagonal stiffness term is non-zero (relative to its diagonal)."""
        K = self.stiffness
        d = np.sqrt(np.abs(np.outer(np.diag(K), np.diag(K))))
        off = np.abs(K - np.diag(np.diag(K)))
        return bool(np.any(off > 1e-12 * np.where(d > 0, d, 1.0)))

    def with_pretwist(self, rate: float) -> "SectionAnalysis":
        """This section on a beam pretwisted at ``rate`` (rad/m): the helical-fibre coupling added.

        A fibre at ``(y, z)`` of a beam twisted by ``k`` per length lies on a helix, and a
        change ``phi_x`` of the twist rate stretches it by ``k r^2 phi_x``
        (``r^2 = y^2 + z^2`` about the beam axis). With the right-handed fibre strain
        ``eps_x + z kappa_y - y kappa_z`` that adds::

            D[j, 5] += k int E a_j r^2 dA,   a = (1, z, -y) for rows (eps_x, kappa_y, kappa_z)
            D[5, 5] += k^2 int E r^4 dA

        to every stiffness carrying ``phi_x`` (the 6 x 6, classical, Benscoter, third-order and
        GBT ones). The integrals use ``axial_points``. This is the extension-twist coupling of a
        pretwisted strip (Rosen 1980; Hodges 2006, sec. 3.4); the frame's own turning along
        the beam is in the element geometry, not here. As for the Benscoter warping, the
        helical stretch of the warping and distortional fields is left out (``k`` times their
        axial strain: second order in ``k r`` for the slender blades it is meant for).
        """
        import dataclasses

        if not rate:
            return self
        self._require_engine("with_pretwist")
        if self.axial_points is None:
            raise ValueError("pretwist needs the section's axial_points (a section with area and inertias)")
        y, z, wE = np.asarray(self.axial_points, dtype=np.float64).T
        r2 = y * y + z * z
        c = rate * np.array([wE @ r2, wE @ (z * r2), -(wE @ (y * r2))])   # rows eps_x, kappa_y, kappa_z
        d = rate * rate * (wE @ (r2 * r2))

        def add(K, rows, twist):
            if K is None:
                return None
            K = np.array(K, dtype=np.float64, copy=True)
            for cj, j in zip(c, rows):
                K[j, twist] += cj
                K[twist, j] += cj
            K[twist, twist] += d
            return K

        return dataclasses.replace(
            self,
            stiffness=add(self.stiffness, (0, 1, 2), 5),
            classical_stiffness=add(self.classical_stiffness, (0, 1, 2), 3),
            benscoter_stiffness=add(self.benscoter_stiffness, (0, 1, 2), 5),
            third_order_stiffness=add(self.third_order_stiffness, (0, 1, 2), 5),
            gbt_stiffness=add(self.gbt_stiffness, (0, 1, 2), 5),
        )

    def with_taper(self, taper) -> "SectionAnalysis":
        """This section in a tapered beam: the Resal coupling of ``taper`` added.

        ``taper`` is a ``thin_walled.taper.SectionTaper`` (right-handed). Its 6 x 6
        (``taper.matrix()``, coupling the axial and bending strains with the shear strains)
        goes into every stiffness that carries the shear strains: the 6 x 6 and the leading
        blocks of the Benscoter, third-order and GBT ones. The classical (shear-rigid) stiffness
        is unchanged: the coupling acts through the shear strains only.
        """
        import dataclasses

        dK = np.asarray(taper.matrix(), dtype=np.float64)
        if not np.any(dK):
            return self
        self._require_engine("with_taper")

        def add(K):
            if K is None:
                return None
            K = np.array(K, dtype=np.float64, copy=True)
            K[:6, :6] += dK
            return K

        return dataclasses.replace(
            self,
            stiffness=add(self.stiffness),
            benscoter_stiffness=add(self.benscoter_stiffness),
            third_order_stiffness=add(self.third_order_stiffness),
            gbt_stiffness=add(self.gbt_stiffness),
            tapered=True,
            taper_parts=getattr(taper, "parts", None),
        )

    def _require_engine(self, what: str) -> None:
        if self.frame != ENGINE_FRAME:
            raise ValueError(f"{what} takes a section in the engine frame; this one is in frame {self.frame!r} "
                             f"(re-sign it back first)")

    def euler_bernoulli_D(self) -> npt.NDArray[np.float64]:
        """6 x 6 ``D`` for shear-rigid elements: ``classical_stiffness`` on rows ``[0, 1, 2, 5]``, zero shear rows."""
        D = np.zeros((6, 6))
        D[np.ix_(CLASSICAL_INDEX, CLASSICAL_INDEX)] = self.classical_stiffness
        return D

    def principal_axes(self) -> Tuple[float, Tuple[float, float]]:
        """Principal angle and centroidal mass moments ``(I_1, I_2)`` of this section.

        The angle is from ``y`` toward ``z``, in ``(-π/2, π/2]``, and ``I_1`` is
        the larger moment. The reduced section exports the same pair; this is
        that export on the full ``SectionAnalysis`` (thin-walled, solid or
        isotropic), from the mass second moments about the mass centre.
        """
        from anisotropic_section_model.geometry_to_properties import principal_axes_from_mass

        if not self.mass:
            raise ValueError("this SectionAnalysis has no mass properties")
        return principal_axes_from_mass(self.mass)

    @property
    def principal_angle(self) -> float:
        return self.principal_axes()[0]

    @property
    def principal_moments(self) -> Tuple[float, float]:
        return self.principal_axes()[1]

    def mass_matrix(self) -> npt.NDArray[np.float64]:
        """Rigid-section mass per length (6, 6) for nodal fields ``[u_x, u_y, u_z, theta_x, theta_y, theta_z]``.

        Rotations are right-handed (as the nodal DOFs). A point at ``(y, z)`` moves by
        ``(u_x - y theta_z + z theta_y, u_y - z theta_x, u_z + y theta_x)``, so with
        ``S_y = m z_m`` and ``S_z = m y_m``::

            M[0,0] = M[1,1] = M[2,2] = m
            M[0,4] = S_y,  M[0,5] = -S_z,  M[1,3] = -S_y,  M[2,3] = S_z
            M[3,3] = I_p,  M[4,4] = I_yy,  M[5,5] = I_zz,  M[4,5] = -I_yz
        """
        mp = self.mass
        if not mp:
            raise ValueError("this SectionAnalysis has no mass properties")
        m = mp["m"]
        Sy, Sz = m * mp["z_m"], m * mp["y_m"]
        M = np.zeros((6, 6))
        M[0, 0] = M[1, 1] = M[2, 2] = m
        M[0, 4] = M[4, 0] = Sy
        M[0, 5] = M[5, 0] = -Sz
        M[1, 3] = M[3, 1] = -Sy
        M[2, 3] = M[3, 2] = Sz
        M[3, 3] = mp["I_p"]
        M[4, 4] = mp["I_yy"]
        M[5, 5] = mp["I_zz"]
        M[4, 5] = M[5, 4] = -mp["I_yz"]
        return M

    def stress_resultants(self, strain: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """``S = stiffness @ strain`` (right-handed strains and forces)."""
        return self.stiffness @ np.asarray(strain, dtype=np.float64)


def _rigid_check(M: np.ndarray, rebuilt: np.ndarray, rtol: float, where: str) -> None:
    scale = max(float(np.abs(M).max()), 1e-300)
    if not np.allclose(M, M.T, rtol=0.0, atol=rtol * scale):
        raise ValueError("mass must be symmetric")
    if not np.allclose(M, rebuilt, rtol=0.0, atol=rtol * scale):
        raise ValueError(f"mass is not a rigid-section mass matrix in the {where} form "
                         f"(largest misfit {np.abs(M - rebuilt).max():.3g}); see SectionAnalysis.mass_matrix")


def _mass_from_engine(M: np.ndarray, rtol: float) -> Dict[str, float]:
    """``mass_properties`` from an engine 6 x 6 in the form of ``SectionAnalysis.mass_matrix``."""
    m = float(M[0, 0])
    if m <= 0:
        raise ValueError("mass per length must be positive")
    props = {"m": m, "y_m": float(-M[0, 5] / m), "z_m": float(M[0, 4] / m), "I_yy": float(M[4, 4]),
             "I_zz": float(M[5, 5]), "I_yz": float(-M[4, 5]), "I_p": float(M[3, 3])}
    _rigid_check(M, SectionAnalysis(np.eye(6), np.eye(4), mass=props).mass_matrix(), rtol, "engine")
    return props


def _mass_from_public(M: np.ndarray, rtol: float) -> Dict[str, float]:
    """``mass_properties`` from a public (BeamDyn-order) 6 x 6 about the origin (``axes.mass_matrix``)."""
    from anisotropic_section_model import axes

    m = float(M[0, 0])
    if m <= 0:
        raise ValueError("mass per length must be positive")
    props = {"m": m, "y_m": float(M[1, 5] / m), "z_m": float(-M[0, 5] / m), "I_yy": float(M[3, 3]),
             "I_zz": float(M[4, 4]), "I_yz": float(-M[3, 4]), "I_p": float(M[5, 5])}
    rebuilt = axes.mass_matrix(props)
    rebuilt[5, 5] = props["I_p"]
    _rigid_check(M, rebuilt, rtol, "public")
    return props
