"""Shared chart style for the project's notebooks (same palette/rules as notebook 1).

Categorical hues are used in fixed order, magnitude uses one blue ramp,
polarity uses blue<->red with a gray midpoint, one y-axis per chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", neutral="#d6d5cf", good="#006300", bad="#d03b3b")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b3b2", "#f0efec", "#9ec5f4", "#1c5cab"])


def setup():
    mpl.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False,
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 26,
        "axes.labelsize": 10, "xtick.color": C["muted"], "ytick.color": C["muted"],
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": C["grid"],
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "text.color": C["ink"],
    })


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color=C["ink2"], va="bottom")


def finish(fig, path, source):
    fig.text(0.01, -0.01, f"출처: {source}", fontsize=8, color=C["muted"], ha="left", va="top",
             transform=fig.transFigure)
    fig.savefig(path)
    plt.show()


def draw_map(ax, feats, fmap, values: dict, cmap, norm, edge="#ffffff"):
    """Choropleth from GeoJSON features (no geopandas). values: {row index -> value}."""
    patches, cols = [], []
    for k, f in enumerate(feats):
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        v = values.get(fmap.get(k), np.nan)
        for poly in polys:
            patches.append(MplPolygon(np.array(poly[0]), closed=True))
            cols.append(C["neutral"] if pd.isna(v) else cmap(norm(v)))
    ax.add_collection(PatchCollection(patches, facecolor=cols, edgecolor=edge, linewidth=0.25))
    ax.autoscale_view(); ax.set_aspect(1.2); ax.set_axis_off()
