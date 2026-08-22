"""Shared publication plotting style for PoseLoop R2 figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


FONT_SIZE = 10
DPI = 300
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "magenta": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#777777",
    "light_gray": "#D9D9D9",
    "black": "#222222",
    "white": "#FFFFFF",
}

matplotlib.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save a figure as vector PDF and high-resolution PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, format=suffix)
        print(f"Saved: {path}")
