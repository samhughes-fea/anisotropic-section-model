"""Shared helpers for the example scripts: imports, plot style and output paths.

Every example writes its figures to a ``figures/`` folder next to its script. The colours follow a
fixed categorical order (blue, orange, aqua, yellow) so a given kinematic model
or criterion keeps the same colour in every figure.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE = "#0b0b0b"  # analytical / reference solutions are drawn in ink, dashed

# Model and criterion colours are fixed across all figures.
MODEL_COLOURS = {"CLPT": SERIES[0], "FSDT": SERIES[1], "RZT": SERIES[2], "Layerwise": SERIES[3]}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32a2a"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.size": 9.5,
            "figure.dpi": 110,
        }
    )


def save(fig, script: str, name: str) -> Path:
    """Write ``fig`` to ``<folder of script>/figures/<name>``."""
    folder = Path(script).resolve().parent / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def simply_supported(mesh, a, b, hard=True):
    """w = 0 on every edge plus in-plane rigid-body restraint.

    Hard supports also fix the tangential rotation and, for zig-zag and layerwise
    models, the tangential enrichment amplitudes (the Navier / Pagano condition).
    """
    from composite_laminate_model import DOF, BoundaryConditions, EnrichedDOF

    bcs = BoundaryConditions()
    for n in mesh.nodes_on_line(0, 0.0) + mesh.nodes_on_line(0, a):
        bcs.fix([n], [DOF.UZ] + ([DOF.RX, EnrichedDOF.Y] if hard else []))
    for n in mesh.nodes_on_line(1, 0.0) + mesh.nodes_on_line(1, b):
        bcs.fix([n], [DOF.UZ] + ([DOF.RY, EnrichedDOF.X] if hard else []))
    bcs.fix([mesh.closest_node((0.0, 0.0))], [DOF.UX, DOF.UY])
    bcs.fix([mesh.closest_node((a, 0.0))], [DOF.UY])
    return bcs


apply_style()
