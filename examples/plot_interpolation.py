import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import interpolation as interp


def f(x):
    return np.sin(x)


def plot_methods(xs, ys, x_dense, x_dense_pl):
    methods = [
        ("Newton", interp.newton, x_dense),
        ("Lagrange", interp.lagrange, x_dense),
        ("Piecewise Linear", interp.piecewise_linear, x_dense_pl),
        ("Spline", interp.spline, x_dense),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (name, func, x_eval) in zip(axes.flat, methods):
        y_true = f(x_eval)
        y_interp = func(xs, ys, x_eval)
        ax.plot(x_eval, y_true, "k--", linewidth=1.5, label="True f(x)")
        ax.plot(x_eval, y_interp, linewidth=1.8, label=name)
        ax.scatter(xs, ys, s=30, color="red", zorder=3, label="Nodes")
        ax.set_title(name)
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Interpolation vs. True Function", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def plot_errors(xs, ys, x_dense, x_dense_pl):
    methods = [
        ("Newton", interp.newton, x_dense),
        ("Lagrange", interp.lagrange, x_dense),
        ("Piecewise Linear", interp.piecewise_linear, x_dense_pl),
        ("Spline", interp.spline, x_dense),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (name, func, x_eval) in zip(axes.flat, methods):
        y_true = f(x_eval)
        y_interp = func(xs, ys, x_eval)
        err = np.abs(y_interp - y_true)
        ax.plot(x_eval, err, color="black", linewidth=1.6)
        ax.set_title(f"{name} | Absolute Error")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)

    fig.suptitle("Interpolation Error (log scale)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def main():
    xs = np.linspace(-2 * np.pi, 2 * np.pi, 20)
    ys = f(xs)
    x_dense = np.linspace(xs[0], xs[-1], 600)
    x_dense_pl = np.linspace(xs[0], xs[-1], 600, endpoint=False)

    plot_methods(xs, ys, x_dense, x_dense_pl)
    plot_errors(xs, ys, x_dense, x_dense_pl)

    plt.show()


if __name__ == "__main__":
    main()
