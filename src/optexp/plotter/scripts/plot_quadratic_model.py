import matplotlib.pyplot as plt
import numpy as np
import torch

from optexp.plotter.scripts.plot_paper import get_dir
from optexp.plotter.style_figure import update_plt
from optexp.plotter.style_lines import get_optimizers
from optexp.utils import split_frequencies_by_groups


def load_data():

    c = 1000
    alpha = 1.0
    pis = np.array([1 / k**alpha for k in range(1, c + 1)])
    pis = pis / np.sum(pis)

    def loss_per_class(w):
        return pis * w**2

    def loss(w):
        return np.sum(pis * w**2)

    def grad(w):
        return pis * w

    def gd_step(lr):
        return lambda w: w - lr * grad(w)

    def sd_step(lr):
        return lambda w: w - lr * np.sign(grad(w))

    def optimize(w0, update, T):
        w = w0
        losses, losses_per_class = [], []
        for t in range(T):
            losses.append(loss(w))
            losses_per_class.append(loss_per_class(w))

            w = update(w)

        return w, losses, losses_per_class

    def adam_opt_numpy(w0, lr):
        w_pt = torch.from_numpy(w0)
        w_pt.requires_grad = True
        optimizer = torch.optim.Adam([w_pt], lr=lr)

        def step(w):
            optimizer.zero_grad()
            w_pt.data = torch.from_numpy(w)
            w_pt.grad = torch.from_numpy(grad(w))
            optimizer.step()
            return w_pt.detach().numpy()

        return step

    def gd_opt_numpy(w0, lr):
        w_pt = torch.from_numpy(w0)
        w_pt.requires_grad = True
        optimizer = torch.optim.SGD([w_pt], lr=lr, momentum=0.0)

        def step(w):
            optimizer.zero_grad()
            w_pt.data = torch.from_numpy(w)
            w_pt.grad = torch.from_numpy(grad(w))
            optimizer.step()
            return w_pt.detach().numpy()

        return step

    np.random.seed(0)
    gd_ss = 0.5 / pis[0]
    sd_ss = 0.70
    ad_ss = 0.70
    T = 50
    w0 = np.ones(c)

    _, gd_losses, gd_losses_per_class = optimize(
        np.copy(w0), gd_opt_numpy(w0, gd_ss), T=T
    )
    _, sd_losses, sd_losses_per_class = optimize(np.copy(w0), sd_step(sd_ss), T=T)
    _, ad_losses, ad_losses_per_class = optimize(
        np.copy(w0), adam_opt_numpy(w0, ad_ss), T=T
    )

    return {
        "gd_losses": np.array(gd_losses),
        "sd_losses": np.array(sd_losses),
        "ad_losses": np.array(ad_losses),
        "gd_losses_per_class": np.stack(gd_losses_per_class),
        "sd_losses_per_class": np.stack(sd_losses_per_class),
        "ad_losses_per_class": np.stack(ad_losses_per_class),
        "pis": pis,
    }


def postprocess(data):
    return data


def settings(plt):
    update_plt(plt, rel_width=1.0, nrows=1, ncols=4, height_to_width_ratio=0.8)


def make_figure(fig, data):

    axes = [fig.add_subplot(1, 4, 1 + i) for i in range(4)]

    pis = data["pis"]
    gd_losses = data["gd_losses"]
    sd_losses = data["sd_losses"]
    ad_losses = data["ad_losses"]
    gd_losses_per_class = data["gd_losses_per_class"]
    sd_losses_per_class = data["sd_losses_per_class"]
    ad_losses_per_class = data["ad_losses_per_class"]

    opts_config = get_optimizers()
    config_gd = opts_config[2]
    config_ad = opts_config[0]
    config_sd = opts_config[6]
    config_gd.label = "GD"
    config_ad.label = "Adam"
    config_sd.label = "Sign"

    for losses, config in zip(
        [gd_losses, sd_losses, ad_losses], [config_gd, config_sd, config_ad]
    ):
        axes[0].plot(
            losses,
            color=config.line_color,
            linestyle=config.line_style,
            label=config.label,
        )

    n_splits = 10
    splits = split_frequencies_by_groups(range(len(pis)), pis, n_splits)

    colors = list(plt.cm.viridis(np.linspace(0, 1, n_splits)))

    def average_losses_per_class_for_split(losses, split):
        return np.sum(losses[:, split], axis=1) / np.sum(pis[split])

    for i, split in enumerate(splits):
        losses_gd = average_losses_per_class_for_split(gd_losses_per_class, split)
        losses_sd = average_losses_per_class_for_split(sd_losses_per_class, split)
        losses_ad = average_losses_per_class_for_split(ad_losses_per_class, split)

        axes[1].plot(losses_gd, color=colors[i])
        axes[2].plot(losses_ad, color=colors[i])
        axes[3].plot(losses_sd, color=colors[i])

    # remove

    for ax in axes:
        # ax.set_yscale("log")
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, "", 1])
        ax.set_xticks([0, 25, 50])
        ax.set_xticklabels([0, "", 50])

    axes[0].set_title("Overall loss")
    axes[1].set_title("Gradient descent")
    axes[2].set_title("Adam")
    axes[3].set_title("Sign descent")
    for ax in axes:
        ax.set_xlabel("Iteration", labelpad=-5)

    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, borderpad=0, borderaxespad=0.1, labelspacing=0.1)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.22)


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    make_figure(fig, postprocess(load_data()))
    fig.savefig(get_dir("quadratic") / "main.pdf")
    fig.savefig(get_dir("quadratic") / "main.png", dpi=600)
    plt.close(fig)
