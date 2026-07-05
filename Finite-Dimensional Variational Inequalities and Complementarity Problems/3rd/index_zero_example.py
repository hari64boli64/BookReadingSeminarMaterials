from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent
EPSILONS = (-0.4, -0.2, 0.0, 0.2, 0.4)
EPSILON_LABELS = (r"-0.4", r"-0.2", r"0", r"0.2", r"0.4")
XLIM = (-0.25, 1.75)
YLIM = (-0.5, 1.0)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20,
})


def F(X, Y, epsilon):
    return X - 1, -Y - epsilon


def absolute_value_field(X, Y):
    return X - 1, -np.abs(Y)


def setup(ax):
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.axvspan(0, XLIM[1], ymin=(0 - YLIM[0]) / (YLIM[1] - YLIM[0]),
               color="#d9d9d9", alpha=0.75, zorder=0)
    ax.axhline(0, color="black", lw=1.1)
    ax.axvline(0, color="black", lw=1.1)
    ax.grid(alpha=0.18)
    ax.set_xticks([0, 1])
    ax.set_yticks([-0.4, 0, 0.4, 0.8])


x = np.linspace(0, 1.6, 9)
y = np.linspace(0, 0.9, 6)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(1, 5, figsize=(20, 4.3), sharex=True, sharey=True)

for j, (epsilon, epsilon_label) in enumerate(zip(EPSILONS, EPSILON_LABELS)):
    ax = axs[j]
    setup(ax)
    U, V = F(X, Y, epsilon)
    ax.quiver(X, Y, -U, -V, angles="xy", scale_units="xy", scale=3.8,
              width=0.006, color="#31688e", alpha=0.9)
    ax.plot(1, -epsilon, marker="*", ms=10, color="#f28e2b", zorder=5)
    if epsilon < 0:
        ax.plot([1, 1], [0, -epsilon], "o", ms=6, color="#d62728", zorder=6)
    elif epsilon == 0:
        ax.plot(1, 0, "o", ms=6, color="#d62728", zorder=6)
    ax.set_title(rf"$\varepsilon={epsilon_label}$", fontsize=24, pad=10)
    if j == 0:
        ax.set_ylabel(r"$-F_\varepsilon(x)$" + "\n" + r"$x_2$", fontsize=24)
    ax.set_xlabel(r"$x_1$")

fig.text(0.5, 0.0,
         r"gray: $K=\mathbb{R}^2_+$; orange star: $F_\varepsilon(x^*)=0$; red circle: VI solution",
         ha="center", fontsize=30)
plt.tight_layout(rect=(0.01, 0.11, 1, 1))
plt.savefig(OUTPUT_DIR / "index_zero_example.png", bbox_inches="tight", dpi=300)

x_abs = np.linspace(0, 1.6, 9)
y_abs = np.linspace(YLIM[0], 0.9, 9)
X_abs, Y_abs = np.meshgrid(x_abs, y_abs)

fig_abs, ax_abs = plt.subplots(figsize=(4, 4.3))
setup(ax_abs)
U_abs, V_abs = absolute_value_field(X_abs, Y_abs)
ax_abs.quiver(X_abs, Y_abs, U_abs, V_abs, angles="xy", scale_units="xy", scale=3.8,
              width=0.006, color="#31688e", alpha=0.9)
ax_abs.plot(1, 0, marker="*", ms=10, color="#f28e2b", zorder=5)
ax_abs.set_ylabel(r"$x_2$", fontsize=24)
ax_abs.set_xlabel(r"$x_1$")

fig_abs.tight_layout()
fig_abs.savefig(OUTPUT_DIR / "absolute_value_vector_field.png",
                bbox_inches="tight", dpi=300)
