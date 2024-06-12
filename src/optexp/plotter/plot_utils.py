from typing import List

import numpy as np
import pandas as pd
import torch


def normalize_y_axis(*axes):
    miny, maxy = np.inf, -np.inf
    for ax in axes:
        y1, y2 = ax.get_ylim()
        miny = np.min([miny, y1])
        maxy = np.max([maxy, y2])
    for ax in axes:
        ax.set_ylim([miny, maxy])


def normalize_x_axis(*axes):
    minx, maxx = np.inf, -np.inf
    for ax in axes:
        x1, x2 = ax.get_xlim()
        minx = np.min([minx, x1])
        maxx = np.max([maxx, x2])
    for ax in axes:
        ax.set_xlim([minx, maxx])


def subsample(xs, NMAX=200, log_only=False, linear_only=False):
    if isinstance(xs, torch.Tensor):
        xs = xs.numpy()
    if isinstance(xs, List):
        xs = np.array(xs)
    if isinstance(xs, pd.Series):
        xs = xs.values

    if not isinstance(xs, np.ndarray):
        import pdb

        pdb.set_trace()

    n = len(xs)
    if n < NMAX:
        return xs

    idx_lin = np.floor(np.linspace(0, n, int(NMAX / 2), endpoint=False)).astype(int)
    idx_log = np.floor(
        np.logspace(0, np.log10(n), int(NMAX / 2), endpoint=False)
    ).astype(int)
    base = [0, n - 1]
    indices = set(base)

    if log_only and linear_only:
        raise ValueError("Cannot have both log_only and linear_only")

    if not log_only:
        indices = indices.union(set(idx_lin))
    if not linear_only:
        indices = indices.union(set(idx_log))

    idx = np.array(sorted(list(indices))).astype(int)

    if any(idx >= n):
        import pdb

        pdb.set_trace()

    return xs[idx]


def copy_line_onto(line, ax):
    ax.plot(
        line.get_xdata(),
        line.get_ydata(),
        linewidth=line.get_linewidth(),
        linestyle=line.get_linestyle(),
        color=line.get_color(),
        gapcolor=line.get_gapcolor(),
        marker=line.get_marker(),
        markersize=line.get_markersize(),
        markeredgewidth=line.get_markeredgewidth(),
        markerfacecolor=line.get_markerfacecolor(),
        markerfacecoloralt=line.get_markerfacecoloralt(),
        fillstyle=line.get_fillstyle(),
        antialiased=line.get_antialiased(),
        dash_capstyle=line.get_dash_capstyle(),
        solid_capstyle=line.get_solid_capstyle(),
        dash_joinstyle=line.get_dash_joinstyle(),
        solid_joinstyle=line.get_solid_joinstyle(),
        pickradius=line.get_pickradius(),
        drawstyle=line.get_drawstyle(),
        markevery=line.get_markevery(),
        label=line.get_label(),
    )


def copy_axes(ax_from, ax_to):
    ax_to.set_title(ax_from.get_title())
    ax_to.set_xlabel(ax_from.get_xlabel())
    ax_to.set_ylabel(ax_from.get_ylabel())
    ax_to.set_xscale(ax_from.get_xscale())
    ax_to.set_yscale(ax_from.get_yscale())
    ax_to.set_xlim(ax_from.get_xlim())
    ax_to.set_ylim(ax_from.get_ylim())
    ax_to.set_xticks(ax_from.get_xticks())
    ax_to.set_yticks(ax_from.get_yticks())
    ax_to.set_xticklabels(ax_from.get_xticklabels())
    ax_to.set_yticklabels(ax_from.get_yticklabels())

    for line in ax_from.get_lines():
        copy_line_onto(line, ax_to)
