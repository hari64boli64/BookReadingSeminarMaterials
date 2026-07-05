from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent
EPSILONS = (-0.4, -0.2, 0.0, 0.2, 0.4)
X1_LIM = (0.5, 1.5)
X2_LIM = (0.0, 0.8)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 17,
})


def objective(x1, x2, epsilon):
    return 0.5 * x1**2 - x1 - 0.5 * x2**2 - epsilon * x2


x1 = np.linspace(*X1_LIM, 100)
x2 = np.linspace(*X2_LIM, 100)
X1, X2 = np.meshgrid(x1, x2)

fig = plt.figure(figsize=(22, 4.8))

for j, epsilon in enumerate(EPSILONS, start=1):
    ax = fig.add_subplot(1, 5, j, projection="3d", computed_zorder=False)
    Z = objective(X1, X2, epsilon)
    ax.plot_surface(X1, X2, Z, cmap="viridis", alpha=0.82,
                    linewidth=0, antialiased=True, zorder=-1)

    # The unconstrained stationary point is shown only when it belongs to K.
    stationary_x2 = -epsilon
    if X2_LIM[0] <= stationary_x2 <= X2_LIM[1]:
        stationary_z = objective(1.0, stationary_x2, epsilon)
        ax.scatter(1.0, stationary_x2, stationary_z + 0.01,
                   marker="*", s=170, color="#f28e2b",
                   edgecolor="black", linewidth=0.5, depthshade=False,
                   zorder=6)

    # VI solutions, equivalently the KKT points of the constrained problem.
    if epsilon < 0:
        solution_x2 = np.array([0.0, -epsilon])
    elif epsilon == 0:
        solution_x2 = np.array([0.0])
    else:
        solution_x2 = np.array([])
    if solution_x2.size:
        solution_z = objective(1.0, solution_x2, epsilon)
        ax.scatter(np.ones_like(solution_x2), solution_x2, solution_z + 0.018,
                   marker="o", s=65, color="#d62728",
                   edgecolor="white", linewidth=0.8, depthshade=False,
                   zorder=7)

    ax.set_title(rf"$\varepsilon={epsilon:g}$", pad=8)
    ax.set_xlabel(r"$x_1$", labelpad=5)
    ax.set_ylabel(r"$x_2$", labelpad=5)
    if j == 1:
        ax.set_zlabel(r"$f_\varepsilon(x)$", labelpad=5)
    ax.set_xlim(X1_LIM)
    ax.set_ylim(X2_LIM)
    ax.set_zlim(-1.05, -0.30)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_zticks([-1.0, -0.6])
    ax.view_init(elev=45, azim=-100)
    ax.set_box_aspect((1.2, 1.0, 0.85))

fig.tight_layout(w_pad=0.2)
fig.savefig(OUTPUT_DIR / "optimization_surface_example.png",
            bbox_inches="tight", dpi=300)
