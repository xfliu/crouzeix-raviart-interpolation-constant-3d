"""Figures for Appendix D from the 2026-10-30 campaign records.

Run from this directory:  python3 make_figures.py
Writes PDFs to ./figures/.
"""
import csv
import math
import os
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
STAR_TARGET = 799 / 5000
PATH_TARGET = 31 / 200
INV2PI = 1 / (2 * math.pi)

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def rows(name):
    with open(os.path.join(HERE, name)) as f:
        return [r for r in csv.reader(f, delimiter="\t") if r and not r[0].startswith("#")]


def dyadic(text):
    num, den = text.split("/")
    return Fraction(int(num), 2 ** int(den.split("^")[1]))


def leaves(name):
    result = []
    with open(os.path.join(HERE, "campaign", "proof", name)) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "L":
                continue
            box = []
            for interval in fields[3].split(","):
                lo, hi = interval.split("..")
                box.append((float(dyadic(lo)), float(dyadic(hi))))
            result.append((fields[1], fields[2], box))
    return result


# ---------------------------------------------------------------- Experiment 1
def fig_calibration():
    lo = {(r[0], int(r[5])): float(r[6]) for r in rows("aux/lower/exp1.tsv")}
    lc = {(r[0], int(r[5])): float(r[7]) for r in rows("aux/lower/exp1.tsv")}
    up = {(r[0], int(r[5])): float(r[10]) for r in rows("aux/upper/exp1.tsv")}
    ids = ["S1", "S2", "S3", "S4", "S5", "P2", "P3", "P4", "P5", "P6"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for i in ids:
        ref = lo[(i, 14)]
        ms = [2, 4, 6, 8, 10, 12]
        d = [max(abs(lo[(i, m)] / ref - 1), 1e-16) for m in ms]
        style = "-o" if i.startswith("S") else "--s"
        axes[0].semilogy(ms, d, style, ms=3.5, lw=1, label=i)
        levels = [2, 3, 4, 5]
        g = [100 * (up[(i, k)] / lc[(i, 10)] - 1) for k in levels]
        axes[1].semilogy([2 ** k for k in levels], g, style, ms=3.5, lw=1, label=i)
    axes[0].set_xlabel("polynomial degree $M$")
    axes[0].set_ylabel(r"$|L_M/L_{14}-1|$")
    axes[0].set_title("(a) lower bounds: degree")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([4, 8, 16, 32])
    axes[1].set_xticklabels(["4", "8", "16", "32"])
    axes[1].set_xlabel("CR mesh resolution $n$ (cells per edge)")
    axes[1].set_ylabel(r"$100\,(U_n/L_{10}-1)$")
    axes[1].set_title("(b) gap between the bounds")
    axes[1].legend(ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D_calibration.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Experiment 2
def fig_samples():
    data = rows("aux/lower/exp2.tsv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for fam, marker, color, label in (("star", "o", "C0", "star samples"),
                                      ("path", "s", "C1", "path samples")):
        xs, ys = [], []
        for r in data:
            if r[1] != fam:
                continue
            p = [float(v) for v in r[2:5]]
            xs.append(max(p) if fam == "star" else max(p[0], p[1]))
            ys.append(float(r[7]))
        ax.scatter(xs, ys, s=12, marker=marker, color=color, alpha=0.75, label=label)
    ax.axhline(INV2PI, color="k", lw=0.8, ls=":", label=r"$1/(2\pi)$")
    ax.axhline(STAR_TARGET, color="C3", lw=0.8, ls="--", label=r"$0.1598$ (proved $S^*$ bound)")
    ax.axhline(PATH_TARGET, color="C2", lw=0.8, ls="--", label=r"$0.155$ (proved $P^*$ bound)")
    ax.set_xlabel(r"largest face angle (degrees)")
    ax.set_ylabel(r"certified lower bound $L_{10}$")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D_samples.pdf"))
    plt.close(fig)


# ------------------------------------------------------ equal-angle star line
def fig_diagonal():
    lo = sorted((float(r[2]), float(r[7])) for r in rows("aux/lower/diag.tsv")
                if r[5] == "10" and r[11] == "ok")
    up = sorted((float(r[2]), float(r[10])) for r in rows("aux/upper/diag.tsv")
                if r[13] == "ok")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([x for x, _ in up], [y for _, y in up], "-", color="C3", lw=1,
            label=r"upper bound $U_{32}$ (CR, $n=32$)")
    ax.plot([x for x, _ in lo], [y for _, y in lo], "-", color="C0", lw=1,
            label=r"lower bound $L_{10}$")
    ax.axhline(INV2PI, color="k", lw=0.8, ls=":", label=r"$1/(2\pi)$")
    ax.axhline(STAR_TARGET, color="C3", lw=0.8, ls="--", label=r"target $0.1598$")
    ax.set_xlim(0, 20)
    ax.set_ylim(0.1570, 0.1606)
    ax.set_xlabel(r"$\varepsilon$ (degrees) for the star $(\varepsilon,\varepsilon,\varepsilon)$")
    ax.set_ylabel(r"bounds for $C(K)$")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D_diagonal.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- cover slices
def _slice(ax, boxes, axis, value, xi, yi, depth, xlabel, ylabel, title):
    sizes = []
    patches = []
    for status, _, box in boxes:
        lo, hi = box[axis]
        if not lo <= value < hi:
            continue
        if status != "retained":
            continue
        (x0, x1), (y0, y1) = box[xi], box[yi]
        patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
        sizes.append(depth(box))
    pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.3)
    pc.set_array(sizes)
    ax.add_collection(pc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    return pc


def fig_covers():
    star = leaves("star-cover.txt")
    path = leaves("path-cover.txt")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    star_depth = lambda b: round(math.log2(60 / (b[0][1] - b[0][0])))
    path_depth = lambda b: round(sum(math.log2(m / (hi - lo))
                                     for (lo, hi), m in zip(b, (60, 60, 90))))
    pc = _slice(axes[0], star, 0, 2.0, 1, 2, star_depth,
                r"$\beta$ (degrees)", r"$\gamma$ (degrees)",
                r"(a) star cover, slice $\alpha=2^\circ$")
    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(0, 60)
    axes[0].set_aspect("equal")
    t = [i * 0.5 for i in range(121)]
    axes[0].plot(t, [x + 2 for x in t], "r-", lw=0.8)
    axes[0].plot(t, [x - 2 for x in t], "r-", lw=0.8)
    cb = fig.colorbar(pc, ax=axes[0], shrink=0.8)
    cb.set_label("octree depth")
    cb.locator = MaxNLocator(integer=True)
    cb.update_ticks()

    pc = _slice(axes[1], path, 0, 30.0, 1, 2, path_depth,
                r"$\theta_2$ (degrees)", r"$\varphi$ (degrees)",
                r"(b) path cover, slice $\theta_1=30^\circ$")
    th = [i * 0.25 for i in range(1, 241)]
    cap = [math.degrees(math.acos(math.tan(math.radians(15)) * math.tan(math.radians(x / 2))))
           for x in th]
    axes[1].plot(th, cap, "r-", lw=0.8)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylim(0, 90)
    cb = fig.colorbar(pc, ax=axes[1], shrink=0.8)
    cb.set_label("number of bisections")
    cb.locator = MaxNLocator(integer=True)
    cb.update_ticks()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "D_covers.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_calibration()
    fig_samples()
    fig_diagonal()
    fig_covers()
    print("figures written to", os.path.normpath(OUT))
