"""Two worked examples of the stretching argument in Proposition 4.1.

Inputs: C_path.txt and C_star.txt (one line per sample: step, t, the 12
vertex coordinates, Ritz approximation of C(K) with degree-10 Bernstein
polynomials), produced by make_samples.py and ritz_general.jl.
Writes figures/reduction_example.pdf.
"""
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "reduction_example.pdf")
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
EDGES = list(itertools.combinations(range(4), 2))

# Vertex labels chosen so that the final maximal edges read A-B-C-D (path)
# or meet at A (star).  "base" names three vertices whose plane (at the start
# of step 2) is rotated into the horizontal xy-plane, with the fourth vertex
# above it; the rotation changes no length and no value of C(K).
EXAMPLES = [
    ("path", {3: "A", 0: "B", 1: "C", 2: "D"}, (22, 30), "path A–B–C–D", (0, 1, 3), -30),
    ("star", {0: "A", 1: "B", 2: "C", 3: "D"}, (22, -62), "star at A", (0, 1, 3), -30),
]


def load(name):
    return [list(map(float, line.split()))
            for line in open(os.path.join(HERE, f"C_{name}.txt"))]


def verts(r):
    return np.array(r[2:14]).reshape(4, 3)


def lengths(P):
    return {e: np.linalg.norm(P[e[1]] - P[e[0]]) for e in EDGES}


def maximal(P):
    return [e for e, v in lengths(P).items() if v > 1 - 1e-9]


def step_xi(rows, k):
    """Stretch direction of step k, read off from the data: the moving
    vertex is displaced along xi."""
    rk = [r for r in rows if int(r[0]) == k]
    d = verts(rk[-1]) - verts(rk[0])
    v = d[np.argmax(np.linalg.norm(d, axis=1))]
    return v / np.linalg.norm(v)


def to_base_plane(rows, base):
    """Rigidly rotate all samples so that the plane of the vertices `base`
    at the start of step 2 is the xy-plane and the fourth vertex lies above."""
    P = verts(next(r for r in rows if int(r[0]) == 1))
    o, a, b = (P[i] for i in base)
    e1 = (a - o) / np.linalg.norm(a - o)
    n = np.cross(a - o, b - o)
    n /= np.linalg.norm(n)
    other = [i for i in range(4) if i not in base][0]
    if np.dot(P[other] - o, n) < 0:
        n = -n
    R = np.vstack([e1, np.cross(n, e1), n])
    out = []
    for r in rows:
        Q = (verts(r) - o) @ R.T
        out.append(r[:2] + list(Q.flatten()) + r[14:])
    return out


def turn(rows, degrees):
    """Rotate all samples about the z-axis (keeps the xy-plane)."""
    c, s_ = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    return [r[:2] + list((verts(r) @ R.T).flatten()) + r[14:] for r in rows]


def draw_frame(ax, P, size):
    """The xy-plane, drawn as a grey floor without axes."""
    x0, y0 = min(P[:, 0].min(), 0) - 0.08, min(P[:, 1].min(), 0) - 0.08
    x1, y1 = P[:, 0].max() + 0.08, P[:, 1].max() + 0.08
    floor = Poly3DCollection([[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]],
                             facecolor="0.72", alpha=0.45, edgecolor="0.55", lw=0.6)
    ax.add_collection3d(floor)
    floor.set_zorder(0)


def progress(rows):
    """Map step k with stretch t in [1, t*_k] to the interval [k, k+1]."""
    x = []
    for k in sorted({int(r[0]) for r in rows}):
        rk = [r for r in rows if int(r[0]) == k]
        t0, t1 = rk[0][1], rk[-1][1]
        x += [k + (r[1] - t0) / (t1 - t0) for r in rk]
    return x


os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig = plt.figure(figsize=(10.0, 10.6))
gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.95], hspace=0.32, wspace=1.2)
letters = iter("abcdefghi")
for row, (name, names, view, final, base, angle) in enumerate(EXAMPLES):
    rows = load(name)
    if base is not None:
        rows = to_base_plane(rows, base)
    rows = turn(rows, angle)
    steps = sorted({int(r[0]) for r in rows})
    stage = [verts(next(r for r in rows if int(r[0]) == k)) for k in steps]
    stage.append(verts(rows[-1]))
    titles = ["start: one maximal edge", "after step 1", f"after step 2: {final}"]
    for k, P in enumerate(stage):
        ax = fig.add_subplot(gs[row, 2 * k:2 * k + 2], projection="3d",
                             computed_zorder=False)
        M = maximal(P)
        for e in EDGES:
            a, b = P[e[0]], P[e[1]]
            if e in M:
                ax.plot(*zip(a, b), color="k", lw=2.6, zorder=4)
            else:
                ax.plot(*zip(a, b), color="0.45", lw=0.9, ls="--", zorder=4)
        for i, p in enumerate(P):
            ax.scatter(*p, color="k", s=12, zorder=5)
            ax.text(*(p + np.array([0.03, 0.03, 0.04])), names[i], fontsize=13)
        xi = step_xi(rows, k) if k < len(steps) else None
        if xi is not None:
            # Vertices that do not move during the step span the plane xi^perp.
            rk = [r for r in rows if int(r[0]) == k]
            move = np.linalg.norm(verts(rk[-1]) - verts(rk[0]), axis=1)
            fixed = [i for i in range(4) if move[i] < 1e-9]
            moving = [i for i in range(4) if i not in fixed]
            tri = Poly3DCollection([P[fixed]], facecolor="C0", alpha=0.22,
                                   edgecolor="none")
            tri.set_zorder(2)
            ax.add_collection3d(tri)
            c = P[fixed].mean(axis=0)
            qx = ax.quiver(*c, *(0.45 * xi), color="C3", lw=1.8, arrow_length_ratio=0.25)
            qx.set_zorder(6)
            ax.text(*(c + 0.52 * xi), r"$\xi$", color="C3", fontsize=14)
            plane = "".join(sorted(names[i] for i in fixed))
            fixed_list = ", ".join(sorted(names[i] for i in fixed))
            mover = ", ".join(sorted(names[i] for i in moving))
            ax.text2D(0.5, 0.0,
                      r"$\xi\perp$" + f" plane {plane} (shaded)\n"
                      + f"{fixed_list} fixed; {mover} moves along " + r"$\xi$",
                      transform=ax.transAxes, ha="center", va="top",
                      fontsize=10, color="C3")
        ax.set_title(f"({next(letters)}) {name}: {titles[k]}", fontsize=11, y=0.93)
        # Limits cover every stage and the floor corner where the axes start.
        allp = np.vstack(stage)
        corner = np.array([min(allp[:, 0].min(), 0) - 0.08,
                           min(allp[:, 1].min(), 0) - 0.08, 0.0])
        lo = np.minimum(allp.min(axis=0), corner) - 0.05
        hi = allp.max(axis=0) + 0.05
        span = (hi - lo).max()
        mid = (hi + lo) / 2
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        draw_frame(ax, allp, 0.35 * span)
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_axis_off()

ticks = ([0, 1, 2], ["start", "step 1", "step 2"])
for col, (name, names, _, _, _, _) in enumerate(EXAMPLES):
    rows = load(name)
    x = progress(rows)
    ax = fig.add_subplot(gs[2, 2 * col:2 * col + 2])
    for e in EDGES:
        ax.plot(x, [lengths(verts(r))[e] for r in rows], lw=1.3,
                label="".join(sorted(names[i] for i in e)))
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.axvline(1, color="0.7", lw=0.7)
    ax.set_xticks(*ticks)
    ax.set_ylabel("edge length")
    ax.set_title(f"({next(letters)}) {name}: edge lengths", fontsize=11)
    ax.legend(ncol=2, frameon=False, loc="lower right", fontsize=9)

ax = fig.add_subplot(gs[2, 4:6])
for name, _, _, _, _, _ in EXAMPLES:
    rows = load(name)
    ax.plot(progress(rows), [r[-1] for r in rows], "o-", ms=2.5, lw=1.2, label=name)
ax.axvline(1, color="0.7", lw=0.7)
ax.set_xticks(*ticks)
ax.set_ylabel(r"$C(K)$")
ax.set_title(f"({next(letters)}) $C(K)$ never decreases", fontsize=11)
ax.legend(frameon=False, loc="upper left")

fig.subplots_adjust(left=0.07, right=0.97, top=1.0, bottom=0.05)
fig.savefig(OUT)
print("written", os.path.normpath(OUT))
