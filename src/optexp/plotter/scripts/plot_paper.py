import os
import pickle
from pathlib import Path
from typing import io

import matplotlib.pyplot as plt

from optexp import SGD, Adam, Experiment, NormSGD, config
from optexp.experiments.bigger_models.gpt2small_wt103 import (
    gpt2small_wt103_with_class_stats_long,
)
from optexp.experiments.imbalance import (
    PTB_class_weighted_per_class,
    PTB_with_class_stats,
)
from optexp.experiments.simpler_transformers import (
    basic_one_layer_perclass,
    train_only_last_layer_perclass,
)
from optexp.experiments.toy_models import balanced_x_perclass
from optexp.experiments.vision import mnist_only
from optexp.experiments.vision.barcoded import (
    mnist_and_barcoded_long_perclass,
    mnist_and_barcoded_perclass,
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
from optexp.optimizers.normalized_opts import Sign
from optexp.plotter.best_run_plotter import plot_best
from optexp.plotter.plot_per_class import plot_per_class
from optexp.plotter.plot_utils import copy_axes
from optexp.plotter.style_figure import make_fig_axs, update_plt


def get_dir(save_dir: str) -> Path:
    dir = config.get_plots_directory() / "paper" / save_dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def select_nomom(exp: Experiment):
    if isinstance(exp.optim, Adam):
        if exp.optim.beta1 == 0.0:
            return True
    else:
        assert (
            isinstance(exp.optim, SGD)
            or isinstance(exp.optim, NormSGD)
            or isinstance(exp.optim, Sign)
        )
        if exp.optim.momentum == 0.0:
            return True


def select_SGDM(exp: Experiment):
    if isinstance(exp.optim, SGD):
        if exp.optim.momentum > 0.0:
            return True
    return False


def select_AdamM(exp: Experiment):
    if isinstance(exp.optim, Adam):
        if exp.optim.beta1 > 0.0:
            return True
    return False


def select_SGDM_and_AdamM(exp: Experiment):
    return select_SGDM(exp) or select_AdamM(exp)


def select_extended_optimizers(exp: Experiment):
    return (
        isinstance(exp.optim, SGD)
        or isinstance(exp.optim, Adam)
        or isinstance(exp.optim, NormSGD)
        or isinstance(exp.optim, Sign)
    )


def select_seed_0(exp: Experiment):
    return exp.seed == 0


H_TO_W_RATIO = 0.8
H_TO_W_RATIO_1_PLOT = 0.7
WIDTH_5_PLOTS = 1.0
WIDTH_3_PLOTS = 0.72
WIDTH_1_PLOT = 0.275


def fig_main_gpt2_transformer():
    update_plt(
        plt,
        rel_width=1.0,
        nrows=1,
        ncols=4,
        height_to_width_ratio=H_TO_W_RATIO,
    )
    fig, axes = plt.subplots(1, 4, constrained_layout=True)
    axes = [axes]

    from optexp.plotter.scripts.plot_frequency_statistics import (
        load_data as load_frequency_data,
    )
    from optexp.plotter.scripts.plot_frequency_statistics import (
        make_figure as make_frequency_figure,
    )

    freq_fig = plt.figure()
    make_frequency_figure(
        freq_fig, load_frequency_data(problem=gpt2small_wt103_with_class_stats_long)
    )
    copy_axes(freq_fig.get_axes()[0], axes[0][0])
    plt.close(freq_fig)

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][1])
        copy_axes(tmp_fig.get_axes()[1], axes[0][2])
        copy_axes(tmp_fig.get_axes()[2], axes[0][3])

    experiments = gpt2small_wt103_with_class_stats_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig1"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
        using_step=True,
        only_tr_va="tr",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
        postprocess=postprocess,
    )

    for ax, letter in zip(axes[0], "abcd"):
        ax.set_title(f"{letter})", loc="left", fontsize="small")

    axes[0][0].set_title("    Samples/class")
    axes[0][1].set_ylabel("Train loss")
    axes[0][1].set_ylabel("Train loss", labelpad=-1)
    axes[0][2].set_title("SGD")
    axes[0][3].set_title("Adam")
    for ax in [axes[0][1], axes[0][2], axes[0][3]]:
        ax.set_ylim([0, 13])
        ax.set_xticklabels([0, "5k", "10k", "15k"])

    fig.tight_layout(pad=0.1)

    fig.savefig(get_dir("fig1_wt103") / "main_wt103.pdf")
    fig.savefig(get_dir("fig1_wt103") / "main_wt103.png", dpi=600)


def fig_linearmodel():
    update_plt(
        plt,
        rel_width=1.0,
        nrows=1,
        ncols=4,
        height_to_width_ratio=H_TO_W_RATIO,
    )
    fig, axes = plt.subplots(1, 4, constrained_layout=True)
    axes = [axes]

    from optexp.plotter.scripts.plot_frequency_statistics import (
        load_data as load_frequency_data,
    )
    from optexp.plotter.scripts.plot_frequency_statistics import (
        make_figure as make_frequency_figure,
    )

    freq_fig = plt.figure()
    make_frequency_figure(freq_fig, load_frequency_data(problem=balanced_x_perclass))
    copy_axes(freq_fig.get_axes()[0], axes[0][0])
    plt.close(freq_fig)

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][1])
        copy_axes(tmp_fig.get_axes()[1], axes[0][2])
        copy_axes(tmp_fig.get_axes()[2], axes[0][3])

    experiments = balanced_x_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)

    plot_per_class(
        experiments=experiments,
        plotting_time=150,
        where=get_dir("fig2"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
        only_tr_va="tr",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
        postprocess=postprocess,
    )

    for ax, letter in zip(axes[0], "abcd"):
        ax.set_title(f"{letter})", loc="left", fontsize="small")

    axes[0][0].set_title("    Samples/class")
    axes[0][1].set_ylabel("Train loss")
    axes[0][1].set_ylabel("Train loss", labelpad=-1)
    axes[0][2].set_title("GD")
    axes[0][3].set_title("Adam")
    for ax in [axes[0][1], axes[0][2], axes[0][3]]:
        ax.set_ylim([0, 10])

    fig.tight_layout(pad=0.1)

    fig.savefig(get_dir("fig4_linear") / "main_linear.pdf")
    fig.savefig(get_dir("fig4_linear") / "main_linear.png", dpi=600)


def fig_mnist_barcode_feasable():
    experiments = mnist_barcoded_only_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_best(
        experiments=experiments,
        where=get_dir("fig2_barcode_feasable"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=400,
    )


def mnist_additional_opts():
    experiments = mnist_and_barcoded_long_perclass.experiments
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig2/additional_opts_perclass"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=400,
    )


def fig_main_mnist():

    fig, axes = make_fig_axs(
        plt, rel_width=1.0, nrows=1, ncols=4, height_to_width_ratio=H_TO_W_RATIO
    )

    experiments = mnist_only.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(experiments)

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][0])

    plot_best(
        experiments=experiments,
        where=get_dir("mnist/balanced"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO_1_PLOT,
        postprocess=postprocess,
        only_tr_va="tr",
        only_metric_containing="loss",
        only_xscale="linear",
        only_yscale="linear",
    )

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][1])
        copy_axes(tmp_fig.get_axes()[1], axes[0][2])
        copy_axes(tmp_fig.get_axes()[2], axes[0][3])

    experiments = mnist_and_barcoded_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("mnist/imbalanced"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
        postprocess=postprocess,
        only_tr_va="tr",
        only_groups=[2],
        only_metric_containing="loss",
        only_scale="linear",
    )

    for ax, letter in zip(axes[0], "abcd"):
        ax.set_title(f"{letter})", loc="left", fontsize="small")

    for ax in axes[0]:
        ax.set_xlabel("Epoch")

    axes[0][1].set_ylabel("")

    axes[0][0].set_title("MNIST")
    axes[0][1].set_title("Imbalanced \n MNIST", verticalalignment="center", y=1.0)
    axes[0][2].set_title("GD")
    axes[0][3].set_title("Adam")

    fig.tight_layout(pad=0.1)

    fig.savefig(get_dir("mnist") / "main_mnist.pdf")
    fig.savefig(get_dir("mnist") / "main_mnist.png", dpi=600)


def fig_resnet(steps=1500, use="bn"):
    if use == "bn":
        exp_small = small_imagenet_resnet_batchnorm_c.experiments
        exp_dec = decaying_imagenet_resnet_batchnorm_c_perclass.experiments
    elif use == "ln":
        exp_small = small_imagenet_resnet_layernorm_c.experiments
        exp_dec = decaying_imagenet_resnet_layernorm_c_perclass.experiments
    else:
        raise ValueError("use must be 'bn' or 'ln'")

    fig, axes = make_fig_axs(
        plt, rel_width=1.0, nrows=1, ncols=4, height_to_width_ratio=H_TO_W_RATIO
    )

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][0])

    plot_best(
        experiments=exp_small,
        where=get_dir("imagenet_models/balanced_bn_best"),
        rel_width=WIDTH_1_PLOT,
        using_step=True,
        height_to_width_ratio=H_TO_W_RATIO_1_PLOT,
        plotting_time=steps,
        postprocess=postprocess,
        only_tr_va="tr",
        only_metric_containing="loss",
        only_xscale="linear",
        only_yscale="linear",
    )

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][1])
        copy_axes(tmp_fig.get_axes()[1], axes[0][2])
        copy_axes(tmp_fig.get_axes()[2], axes[0][3])

    plot_per_class(
        experiments=exp_dec,
        where=get_dir(f"imagenet_models/heavytailed_bn_perclass"),
        plotting_time=steps,
        using_step=True,
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
        postprocess=postprocess,
        only_tr_va="tr",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
    )

    for ax, letter in zip(axes[0], "abcd"):
        ax.set_title(f"{letter})", loc="left", fontsize="small")

    for ax in axes[0]:
        ax.set_xlabel("Step")
        ax.set_ylim([0, 10])

    axes[0][0].set_ylabel("Train loss", labelpad=-1)
    axes[0][1].set_ylabel("")
    axes[0][0].set_title("Small \n ImageNet", verticalalignment="center", y=1.0)
    axes[0][1].set_title("  Imbalanced \n ImageNet", verticalalignment="center", y=1.0)
    axes[0][2].set_title("SGD")
    axes[0][3].set_title("Adam")

    fig.tight_layout(pad=0.1)

    fig.savefig(get_dir("resnet") / "main_resnet.pdf")
    fig.savefig(get_dir("resnet") / "main_resnet.png", dpi=600)


def fig_moreopts_lastlayer(epoch=1000):
    experiments = train_only_last_layer_perclass.experiments
    experiments = filter(select_nomom, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/nomom_{epoch}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epoch,
        only_tr_va="tr",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
    )

    experiments = train_only_last_layer_perclass.experiments
    experiments = filter(lambda x: not select_nomom(x), experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/mom_{epoch}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epoch,
        only_tr_va="tr",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
    )

    experiments = train_only_last_layer_perclass.experiments
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/all_{epoch}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epoch,
        only_tr_va="tr",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
    )


def fig_linear_more_opts(epochs=500):
    experiments = balanced_x_perclass.experiments
    experiments = filter(select_extended_optimizers, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/linear_more_opts_{epochs}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epochs,
    )


def fig_ptb_class_stats(epochs=500):
    experiments = basic_one_layer_perclass.experiments
    experiments = filter(select_extended_optimizers, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/ptb_more_opts_{epochs}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epochs,
    )


def fig7():
    experiments = PTB_class_weighted_per_class.experiments
    # experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig7"),
        rel_width=WIDTH_5_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig_appendix_PTB_stochastic(epochs=100):
    experiments = PTB_with_class_stats.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"appendix_standard_training_{epochs}"),
        plotting_time=epochs,
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig_appendix_PTB_deterministic(epochs=500):
    experiments = basic_one_layer_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"appendix_fb_training_{epochs}"),
        plotting_time=epochs,
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


if __name__ == "__main__":

    fig_main_gpt2_transformer()
    fig_main_mnist()
    fig_resnet()
    fig_linearmodel()
    fig_moreopts_lastlayer(1000)

    # still need to cleanup:
    # # It also works on smaller models
    # fig_appendix_PTB_stochastic(epochs=100)
    # fig_appendix_PTB_stochastic(epochs=50)
    # # And in deterministic setting
    # fig_appendix_PTB_deterministic(epochs=500)
    # fig_appendix_PTB_deterministic(epochs=200)
    # # And on vision
    # fig2_barcode_feasable()
    # fig2_shorter_timescale()
    # fig2_additional_opts()
    # # More other optimizers
    # fig_linear_more_opts(500)
    # fig_linear_more_opts(200)
    # fig_ptb_class_stats(500)
    # fig_ptb_class_stats(200)

    # # Reweighting
    # fig7()
