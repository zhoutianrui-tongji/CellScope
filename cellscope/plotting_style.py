"""Canonical publication styling and subcellular color semantics.

The palette encodes biological side before individual structure:

- nuclear-side structures use a cool blue/indigo/teal family;
- cytoplasmic-side structures use a warm gold/orange/crimson family;
- unavailable or unknown values use neutral grey.

All production plots that display these labels should import this module rather
than defining local colors, so a label has the same visual meaning everywhere.
"""

from typing import Dict, Tuple


SUBCELLULAR_LABEL_ORDER: Tuple[str, ...] = (
    "Nucleus",
    "Nucleolus",
    "Lamina",
    "Nuclear Pore",
    "Cytosol",
    "ER",
    "OMM",
    "Unknown",
)

SUBCELLULAR_LABEL_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Nuclear side": (
        "Nucleus",
        "Nucleolus",
        "Lamina",
        "Nuclear Pore",
    ),
    "Cytoplasmic side": (
        "Cytosol",
        "ER",
        "OMM",
    ),
}

# Cool nuclear family versus warm cytoplasmic family.  Saturation and luminance
# are deliberately moderated for white-background publication figures while
# retaining separation under common forms of red-green color-vision deficiency.
SUBCELLULAR_LABEL_PALETTE: Dict[str, str] = {
    "Nucleus": "#2F5D8C",
    "Nucleolus": "#6B5CA5",
    "Lamina": "#4F83B6",
    "Nuclear Pore": "#4FA7B8",
    "Cytosol": "#D6A23C",
    "ER": "#E2743E",
    "OMM": "#B84A62",
    "Unknown": "#9AA0A6",
}

PUBLICATION_PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.dpi": 500,
    "savefig.dpi": 800,
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
    "axes.linewidth": 0.9,
    "axes.labelpad": 5,
    "axes.titlepad": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
}
