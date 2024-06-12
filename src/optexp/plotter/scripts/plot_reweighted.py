import os
import pickle
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt

from optexp import SGD, Adam, Experiment, LightningExperiment, NormSGD, config
from optexp.experiments.bigger_models.gpt2small_wt103 import (
    gpt2small_wt103_with_class_stats_long,
)
from optexp.experiments.imbalance import (
    PTB,
    PTB_class_weighted,
    PTB_class_weighted_per_class,
    PTB_with_class_stats,
)
from optexp.experiments.imbalance.unused import PTB_class_weighted_notsqrt
from optexp.experiments.simpler_transformers import (
    basic_one_layer_perclass,
    train_only_last_layer_perclass,
)
from optexp.experiments.toy_models import (
    balanced_x,
    balanced_x_class_weighted,
    balanced_x_class_weighted_sqrt,
    balanced_x_perclass,
)
from optexp.experiments.vision import mnist_only
from optexp.experiments.vision.barcoded import (
    mnist_and_barcoded,
    mnist_and_barcoded_long_perclass,
    mnist_and_barcoded_perclass,
    mnist_and_barcoded_reweighted_long,
    mnist_and_barcoded_reweighted_sqrt,
    mnist_barcoded_only_long,
)
from optexp.experiments.vision.unused import (
    decaying_imagenet_resnet_batchnorm_c,
    decaying_imagenet_resnet_batchnorm_c_perclass,
    decaying_imagenet_resnet_layernorm_c,
    decaying_imagenet_resnet_layernorm_c_perclass,
    small_imagenet_resnet_batchnorm_c,
    small_imagenet_resnet_layernorm_c,
)
from optexp.experiments.vision.unused.reweighted_imagenet import (
    decaying_imagenet_resnet_sqrt_weighted_batchnorm_c,
    decaying_imagenet_resnet_weighted_batchnorm_c,
)
from optexp.optimizers.normalized_opts import Sign
from optexp.plotter.best_run_plotter import plot_best
from optexp.plotter.plot_per_class import plot_per_class
from optexp.plotter.plot_utils import copy_axes
from optexp.plotter.scripts.plot_paper import (
    H_TO_W_RATIO,
    H_TO_W_RATIO_1_PLOT,
    WIDTH_1_PLOT,
    WIDTH_3_PLOTS,
    WIDTH_5_PLOTS,
    get_dir,
    select_extended_optimizers,
    select_nomom,
    select_seed_0,
    select_SGDM,
    select_SGDM_and_AdamM,
)
from optexp.plotter.style_figure import make_fig_axs, update_plt
from optexp.plotter.style_lines import COLORS


def fig_reweigthed():
    update_plt(
        plt,
        rel_width=1,
        nrows=1,
        ncols=4,
        height_to_width_ratio=H_TO_W_RATIO,
    )
    fig, axes = plt.subplots(1, 4, constrained_layout=True)
    axes = [axes]

    def copy_exp_to_ax(exp, ax, name):

        exps = list(exp)

        def postprocess(tmp_fig):
            copy_axes(tmp_fig.get_axes()[0], ax)

        plot_best(
            experiments=exps,
            where=get_dir(f"reweighted/{name}"),
            rel_width=WIDTH_1_PLOT,
            height_to_width_ratio=H_TO_W_RATIO,
            only_metric_containing="tr_CrossEntropyLoss",
            only_xscale="linear",
            only_yscale="linear",
            postprocess=postprocess,
            using_step=isinstance(exps[0], LightningExperiment),
        )

    def copyexp(base, withsqrt, nosqrt, ax, name):
        copy_exp_to_ax(filter(select_SGDM_and_AdamM, base.experiments), ax, name)
        copy_exp_to_ax(filter(select_SGDM, withsqrt.experiments), ax, f"{name}_{True}")
        copy_exp_to_ax(filter(select_SGDM, nosqrt.experiments), ax, f"{name}_{False}")

    copyexp(
        PTB,
        PTB_class_weighted,
        PTB_class_weighted_notsqrt,
        axes[0][0],
        "ptb",
    )
    copyexp(
        balanced_x_perclass,
        balanced_x_class_weighted_sqrt,
        balanced_x_class_weighted,
        axes[0][1],
        "linear",
    )
    copyexp(
        mnist_and_barcoded,
        mnist_and_barcoded_reweighted_sqrt,
        mnist_and_barcoded_reweighted_long,
        axes[0][2],
        "mnist",
    )
    copyexp(
        decaying_imagenet_resnet_batchnorm_c,
        decaying_imagenet_resnet_sqrt_weighted_batchnorm_c,
        decaying_imagenet_resnet_weighted_batchnorm_c,
        axes[0][3],
        "imagenet",
    )
    axes[0][0].set_title("PTB")
    axes[0][1].set_title("Linear")
    axes[0][2].set_title("HT MNIST")
    axes[0][3].set_title("HT ImageNet")

    for ax in axes[0]:
        ax.get_lines()[2].set_linestyle("dashed")
        ax.get_lines()[3].set_linestyle("dashed")
        ax.get_lines()[2].set_color(COLORS["PTyellow"])
        ax.get_lines()[3].set_color(COLORS["PTblue"])

    for ax in axes[0]:
        ax.get_lines()[0].set_label("Adam")
        ax.get_lines()[1].set_label("SGD")
        ax.get_lines()[2].set_label(r"rSGD $\pi^{1/2}$")
        ax.get_lines()[3].set_label(r"rSGD $\pi$")

    for ax in axes[0]:
        ax.set_ylim([0, 10])
    axes[0][0].set_xlim([0, 100])
    axes[0][1].set_xlim([0, 1000])
    axes[0][2].set_xlim([0, 300])
    axes[0][3].set_xticklabels([0, 100, 200, 300])
    axes[0][3].set_xlim([0, 1500])
    axes[0][3].set_xticks([0, 500, 1000, 1500])
    axes[0][3].set_xticklabels([0, 500, 1000, 1500])

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [1, 0]
    axes[0][0].legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        borderpad=0,
        borderaxespad=0.1,
        labelspacing=0.1,
    )

    handles, labels = axes[0][1].get_legend_handles_labels()
    order = [2, 3]
    axes[0][1].legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        borderpad=0,
        borderaxespad=0.1,
        labelspacing=0.1,
    )

    for ax in [axes[0][1], axes[0][2], axes[0][3]]:
        ax.set_ylabel("")

    fig.tight_layout(pad=0.1)
    fig.savefig(get_dir("reweighted") / "main.pdf")
    fig.savefig(get_dir("reweighted") / "main.png", dpi=600)


if __name__ == "__main__":
    fig_reweigthed()
