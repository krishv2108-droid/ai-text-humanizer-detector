"""Shared chart tokens and axis styling (light surface, validated categorical palette)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Categorical slots 1-3 (blue, orange, aqua). The first three validate under every pair rule.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
HUMAN_C, AI_C = SERIES[0], SERIES[1]
# One-hue sequential ramp (blue), light to dark, for heatmaps.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
