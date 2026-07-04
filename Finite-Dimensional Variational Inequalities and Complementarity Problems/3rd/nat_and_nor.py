from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.labelsize": 26,
    "axes.titlesize": 20,
    "legend.fontsize": 13,
})

OUTPUT_DIR = Path(__file__).parent
TAB10 = plt.get_cmap("tab10").colors


# Reproduce a 2D variational-inequality illustration:
# K = unit Euclidean ball, F(x)=x-a, equilibrium x*=Pi_K(a).
a = np.array([1.55, 0.55])
xstar = a / np.linalg.norm(a)

def proj_ball_vec(Y):
    # Y has shape (..., 2)
    n = np.linalg.norm(Y, axis=-1, keepdims=True)
    return np.where(n <= 1, Y, Y / n)

def F(x):
    return x - a

# Canvas/grid
xlim = (-1.35, 1.85)
ylim = (-1.35, 1.35)
x = np.linspace(xlim[0], xlim[1], 13)
y = np.linspace(ylim[0], ylim[1], 10)
X, Y = np.meshgrid(x, y)
P = np.stack([X, Y], axis=-1)

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    r"Example: $K=\{x:\Vert x\Vert\leq 1\}$, $F(x)=x-a$",
    y=0.99,
    fontsize=28,
)

def setup(ax):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=2.2, color="black"))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

# --- Left: negative flow -F(x)=a-x
ax = axs[0]
setup(ax)
U = a[0] - X
V = a[1] - Y
ax.quiver(X, Y, U, V, angles="xy", scale_units="xy",
          width=0.0048, color=TAB10[0], alpha=0.95)
ax.plot(a[0], a[1], marker="*", ms=14, color="gold", zorder=5)
ax.plot(xstar[0], xstar[1], "o", ms=8, color=TAB10[3], zorder=6)
ax.set_title("Negative flow\n" r"$-F(x)=a-x$", fontsize=20)

# --- Middle: natural map correction -F_nat(v), v in K
ax = axs[1]
setup(ax)
Vpts = np.stack([X, Y], axis=-1)
YminusF = Vpts - (Vpts - a)  # v - F(v) = a for this affine example
Proj = proj_ball_vec(YminusF)
Fnat = Vpts - Proj
Unat, Vnat = -Fnat[..., 0], -Fnat[..., 1]
inside = X**2 + Y**2 <= 1.0001
outside = ~inside
ax.quiver(X, Y, np.where(inside, Unat, np.nan),
          np.where(inside, Vnat, np.nan), angles="xy", scale_units="xy",
          width=0.0050, color=TAB10[2], alpha=0.95)
ax.quiver(X, Y, np.where(outside, Unat, np.nan),
          np.where(outside, Vnat, np.nan), angles="xy", scale_units="xy",
          width=0.0050, color=TAB10[2], alpha=0.3)
ax.plot(a[0], a[1], marker="*", ms=14, color="gold", zorder=5)
ax.plot(xstar[0], xstar[1], "o", ms=8, color=TAB10[3], zorder=6)
ax.set_title(
    "Natural map\n"
    r"$-F_K^{\mathrm{nat}}(v)=\Pi_K(v-F(v))-v$, $v\in K$",
    fontsize=20,
)

# --- Right: normal map correction -F_nor(z), z in R^2
ax = axs[2]
setup(ax)
Z = P
PiZ = proj_ball_vec(Z)
Fnor = (PiZ - a) + Z - PiZ   # F(Pi_K(z)) + z - Pi_K(z)
Unor, Vnor = -Fnor[..., 0], -Fnor[..., 1]
ax.quiver(X, Y, Unor, Vnor, angles="xy", scale_units="xy",
          width=0.0048, color=TAB10[4], alpha=0.95)
ax.plot(a[0], a[1], marker="*", ms=14, color="gold", zorder=5)
ax.plot([a[0], xstar[0]], [a[1], xstar[1]], linestyle="--",
        color=TAB10[1], lw=2.2, zorder=4)
ax.plot(xstar[0], xstar[1], "o", ms=8, color=TAB10[3], zorder=6)
ax.set_title(
    "Normal map\n"
    r"$-F_K^{\mathrm{nor}}(z)=-F(\Pi_K(z))-z+\Pi_K(z)$, $z\in\mathbb{R}^2$",
    fontsize=20,
)

plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(OUTPUT_DIR / "nat_and_nor.png", dpi=220, bbox_inches="tight")
