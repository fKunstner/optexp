from fractions import Fraction
from typing import Type

import matplotlib.pyplot as plt

from optexp import SGD, Adam, Experiment, NormSGD, Optimizer, config
from optexp.experiments.bigger_models.gpt2small_wt103 import (
    gpt2small_wt103_with_class_stats_long,
)
from optexp.experiments.toy_models.smaller import (
    balanced_x_smaller_longer,
    balanced_x_smaller_longer_perclass,
)
from optexp.experiments.toy_models.smaller.changing_input import nnz_mean, zero_mean
from optexp.experiments.vision import mnist_only
from optexp.experiments.vision.barcoded import (
    mnist_and_barcoded_long,
    mnist_barcoded_only_long,
)
from optexp.plotter.best_run_plotter import plot_best
from optexp.plotter.plot_per_class import plot_per_class
from optexp.plotter.plot_utils import copy_axes
from optexp.plotter.scripts.plot_paper import (
    H_TO_W_RATIO,
    H_TO_W_RATIO_1_PLOT,
    WIDTH_1_PLOT,
    WIDTH_3_PLOTS,
    get_dir,
    select_nomom,
    select_seed_0,
)
from optexp.plotter.style_figure import make_fig_axs, update_plt


def select_sgd_with_lr(lr_exponent: Fraction):
    def select_sgd_with_lr_inner(exp):
        if isinstance(exp.optim, SGD):
            if exp.optim.learning_rate.exponent == lr_exponent:
                return True
        return False

    return select_sgd_with_lr_inner


def select_adam_with_lr(lr_exponent: Fraction):
    def select_adam_with_lr_inner(exp):
        if isinstance(exp.optim, Adam):
            if exp.optim.learning_rate.exponent == lr_exponent:
                return True
        return False

    return select_adam_with_lr_inner


def selector_lr(lr_exponent: Fraction):
    def select_lr(exp):
        if isinstance(exp.optim, Optimizer):
            if exp.optim.learning_rate.exponent == lr_exponent:
                return True
        return False

    return select_lr


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


def fig_large_ss():
    experiments = balanced_x_smaller_longer.experiments
    experiments = filter(select_nomom, experiments)
    experiments = list(filter(select_seed_0, experiments))
    experiments_large = list(filter(select_sgd_with_lr(Fraction(3, 1)), experiments))
    experiments_small = list(filter(select_sgd_with_lr(Fraction(-1, 1)), experiments))

    def postprocess(fig):
        for ax in fig.get_axes():
            ax.set_title("Large step-size ($10^{3}$)")
            ax.set_ylim([10**-2, 10**5.5])
        fig.tight_layout(pad=0)

    plot_best(
        experiments=experiments_large,
        where=get_dir("additional/longer/large_ss"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=0.8,
        plotting_time=1000,
        postprocess=postprocess,
    )

    def postprocess(fig):
        for ax in fig.get_axes():
            ax.set_title("Small step-size ($10^{-1}$)")
            ax.set_ylim([10**-2, 10**5.5])
        fig.tight_layout(pad=0)

    plot_best(
        experiments=experiments_small,
        where=get_dir("additional/longer/small_ss"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=0.8,
        plotting_time=1000,
        postprocess=postprocess,
    )


def fig_longterm():
    experiments = balanced_x_smaller_longer_perclass.experiments
    experiments = filter(select_nomom, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = filter(select_sgd_with_lr(Fraction(-2, 1)), experiments)
    experiments = list(experiments)

    times = [100, 1000, 10000]
    titles = ["Short (100 steps)", "Medium (1k steps)", "Long (10k steps)"]
    for time, title in zip(times, titles):

        def postprocess(fig):
            for ax in fig.get_axes():
                if ax.get_yscale() != "log":
                    ax.set_ylim([0, 7])
                ax.set_title(title)
            fig.tight_layout(pad=0)

        plot_per_class(
            experiments=experiments,
            where=get_dir(f"additional/longer/longer_run_{time}"),
            rel_width=WIDTH_1_PLOT,
            height_to_width_ratio=H_TO_W_RATIO,
            plotting_time=time,
            postprocess=postprocess,
            plot_overall_loss=False,
        )


def fig_compare_stepsizes(optimizer: Type[SGD | Adam] = SGD, start_at=-3):
    experiments = balanced_x_smaller_longer_perclass.experiments
    experiments = filter(select_nomom, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(filter(lambda _: isinstance(_.optim, optimizer), experiments))

    experiments = (
        list(filter(selector_lr(Fraction(start_at, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 1, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 2, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 3, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 4, 2)), experiments))
    )

    name = "SGD" if optimizer == SGD else "Adam"
    stopping_times = [10000, 3162, 1000, 316, 100]

    for exp, stopping_time in zip(experiments, stopping_times):

        def postprocess(fig):
            for ax in fig.get_axes():
                if ax.get_yscale() != "log":
                    ax.set_ylim([0, 7])
                ax.set_title(
                    f"{name}, $\\alpha$ = {exp.optim.learning_rate.as_latex_str()}"
                )
            fig.tight_layout(pad=0)

        plot_per_class(
            experiments=[exp],
            where=get_dir(
                f"additional/per_ss/start_at_{start_at}/{name}/{exp.optim.learning_rate}"
            ),
            rel_width=WIDTH_1_PLOT,
            height_to_width_ratio=H_TO_W_RATIO,
            plotting_time=stopping_time,
            postprocess=postprocess,
            plot_overall_loss=False,
        )


def compare_all_stepsizes():
    for start_at in [-8, -7, -6, -5, -4]:
        fig_compare_stepsizes(optimizer=SGD, start_at=start_at)
    for start_at in [-12, -11, -10, -9, -8, -7, -6]:
        fig_compare_stepsizes(optimizer=Adam, start_at=start_at)


def fig_compare_inputs():

    exps_nnz = nnz_mean.experiments
    exps_nnz = filter(select_seed_0, exps_nnz)
    exps_nnz = list(filter(select_nomom, exps_nnz))
    exps_nnz = list(filter(select_sgd_with_lr(Fraction(-4, 2)), exps_nnz)) + list(
        filter(select_adam_with_lr(Fraction(-6, 2)), exps_nnz)
    )
    exps_zero = zero_mean.experiments
    exps_zero = filter(select_seed_0, exps_zero)
    exps_zero = list(filter(select_nomom, exps_zero))
    exps_zero = list(filter(select_sgd_with_lr(Fraction(4, 2)), exps_zero)) + list(
        filter(select_adam_with_lr(Fraction(-3, 2)), exps_zero)
    )

    def postprocess(fig):
        for ax in fig.get_axes():
            if ax.get_yscale() != "log":
                ax.set_ylim([0, 7])
            # ax.set_title("")
        fig.tight_layout(pad=0.3)

    for name, exp in zip(["nnz", "zero"], [exps_nnz, exps_zero]):
        plot_per_class(
            experiments=exp,
            where=get_dir(f"additional/compare_inputs/{name}"),
            rel_width=WIDTH_3_PLOTS,
            height_to_width_ratio=H_TO_W_RATIO,
            plotting_time=25,
            postprocess=postprocess,
            plot_overall_loss=True,
        )


def fig_validation_gpt2_transformer():
    update_plt(
        plt,
        rel_width=1.0,
        nrows=1,
        ncols=3,
        height_to_width_ratio=H_TO_W_RATIO,
    )
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    axes = [axes]

    from optexp.plotter.scripts.plot_frequency_statistics import (
        make_figure as make_frequency_figure,
    )

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][0])
        copy_axes(tmp_fig.get_axes()[1], axes[0][1])
        copy_axes(tmp_fig.get_axes()[2], axes[0][2])

    experiments = gpt2small_wt103_with_class_stats_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("additional/gpt2_val"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
        using_step=True,
        only_tr_va="val",
        only_groups=[10],
        only_metric_containing="loss",
        only_scale="linear",
        postprocess=postprocess,
    )

    for ax, letter in zip(axes[0], "abcd"):
        ax.set_title(f"{letter})", loc="left", fontsize="small")

    axes[0][0].set_ylabel("Validation loss", labelpad=-1)
    axes[0][1].set_title("SGD")
    axes[0][2].set_title("Adam")
    for ax in axes[0]:
        ax.set_ylim([0, 13])
        ax.set_xticklabels([0, "5k", "10k", "15k"])

    fig.tight_layout(pad=0.1)

    fig.savefig(get_dir("additional/wt103_validation") / "validation_wt103.pdf")
    fig.savefig(
        get_dir("additional/wt103_validation") / "validation_wt103.png", dpi=600
    )


def fig_main_mnist():

    fig, axes = make_fig_axs(
        plt, rel_width=1.0, nrows=1, ncols=3, height_to_width_ratio=H_TO_W_RATIO
    )

    experiments = mnist_only.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(experiments)

    def postprocess(tmp_fig):
        copy_axes(tmp_fig.get_axes()[0], axes[0][0])

    plot_best(
        experiments=experiments,
        where=get_dir("fig2_barcode_feasable/balanced"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO_1_PLOT,
        postprocess=postprocess,
        only_tr_va="tr",
        only_metric_containing="loss",
        only_xscale="linear",
        only_yscale="linear",
        plotting_time=300,
    )

    experiments = mnist_barcoded_only_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_best(
        experiments=experiments,
        where=get_dir("fig2_barcode_feasable/barcode_only"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO_1_PLOT,
        postprocess=postprocess,
        only_tr_va="tr",
        only_metric_containing="loss",
        only_xscale="linear",
        only_yscale="linear",
        plotting_time=300,
    )

    experiments = mnist_and_barcoded_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_best(
        experiments=experiments,
        where=get_dir("fig2_barcode_feasable/mnist_and_barcode"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO_1_PLOT,
        postprocess=postprocess,
        only_tr_va="tr",
        only_metric_containing="loss",
        only_xscale="linear",
        only_yscale="linear",
        plotting_time=300,
    )


#    for ax, letter in zip(axes[0], "abc"):
#        ax.set_title(f"{letter})", loc="left", fontsize="small")
#
#    fig.savefig(get_dir("fig2_barcode_feasable") / "mnist_variants.pdf")
#    fig.savefig(get_dir("fig2_barcode_feasable") / "mnist_variants.png", dpi=600)

if __name__ == "__main__":
    # fig_large_ss()
    # fig_longterm()
    # fig_compare_all_stepsizes()
    fig_validation_gpt2_transformer()
    # fig_main_mnist()
