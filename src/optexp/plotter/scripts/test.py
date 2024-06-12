import matplotlib.pyplot as plt
import numpy as np


def load_data():
    pass


def settings(plt):
    pass


def make_figure(fig, data):

    num_rows = 10
    num_cols = 1

    axes = [
        fig.add_subplot(1, 2, 1),
        fig.add_subplot(1, 2, 2),
    ]

    axes[0].set_title("Normal Title")
    axes[1].set_title("Normal Title")
    axes[0].set_title("a)", loc="left", fontsize="medium")
    axes[1].set_title("b)", loc="left", fontsize="medium")

    plt.show()
