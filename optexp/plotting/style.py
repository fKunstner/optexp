_GOLDEN_RATIO = (1 + 5.0 ** (1 / 2)) / 2.0
_INVERSE_GOLDEN_RATIO = _GOLDEN_RATIO - 1


def matplotlib_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="sans-serif",
    height_to_width_ratio=_INVERSE_GOLDEN_RATIO,
):
    return {
        **font_config(family),
        **fontsize_config(),
        **layout_config(rel_width, ncols, nrows, height_to_width_ratio),
        **style_config(),
    }


def fontsize_config():
    fontsizes_normal = 11 - 1
    fontsizes_small = 11 - 3
    fontsizes_tiny = 11 - 4
    return {
        "font.size": fontsizes_normal,
        "axes.titlesize": fontsizes_normal,
        "axes.labelsize": fontsizes_small,
        "legend.fontsize": fontsizes_small,
        "xtick.labelsize": fontsizes_tiny,
        "ytick.labelsize": fontsizes_tiny,
    }


def layout_config(rel_width, ncols, nrows, height_to_width_ratio):
    full_width_in = 5.5
    width_in = full_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    return {
        "figure.dpi": 250,
        "figure.figsize": (width_in, height_in),
        "figure.constrained_layout.use": False,
        "figure.autolayout": False,
        # Padding around axes objects. Float representing inches.
        # Default is 3/72 inches (3 points)
        "figure.constrained_layout.h_pad": (1 / 72),
        "figure.constrained_layout.w_pad": (1 / 72),
        # Space between subplot groups. Float representing
        # a fraction of the subplot widths being separated.
        "figure.constrained_layout.hspace": 0.00,
        "figure.constrained_layout.wspace": 0.00,
    }


def style_config():
    return {
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def font_config(family):
    fontset = "stix" if family == "serif" else "stixsans"  # ptmx replacement
    return {
        "text.usetex": False,
        "font.sans-serif": ["TeX Gyre Heros"],
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": fontset,
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.family": family,
    }


def update_style(
    plt, rel_width=1.0, nrows=1, ncols=4, height_to_width_ratio=_INVERSE_GOLDEN_RATIO
):
    plt.rcParams.update(
        matplotlib_config(
            rel_width=rel_width,
            nrows=nrows,
            ncols=ncols,
            height_to_width_ratio=height_to_width_ratio,
        )
    )


def make_axes(
    plt, rel_width=1.0, nrows=1, ncols=1, height_to_width_ratio=_INVERSE_GOLDEN_RATIO
):
    update_style(
        plt,
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
    )
    return plt.subplots(nrows=nrows, ncols=ncols, squeeze=(nrows == 1 and ncols == 1))
