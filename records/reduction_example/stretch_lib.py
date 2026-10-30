"""Stretching procedure of Proposition 4.1.

Starting from a tetrahedron of diameter 1, repeatedly stretch along a unit
vector xi orthogonal to all edges of length 1 (the maximal edges), until a
further edge reaches length 1.  Stop when the maximal edges span R^3.
"""
import itertools

import numpy as np

EDGES = list(itertools.combinations(range(4), 2))


def lengths(P):
    return {e: np.linalg.norm(P[e[1]] - P[e[0]]) for e in EDGES}


def run(P, xi1=None, verbose=True):
    """Return the list of steps (P, maximal edges, xi, t*, new edge).

    The last entry is the final shape (P, maximal edges, None, None).  When
    the maximal edges span a line, xi is the part of `xi1` (default: the
    z-axis) orthogonal to that line; when they span a plane, xi is its normal.
    """
    P = np.array(P, float)
    steps = []
    while True:
        L = lengths(P)
        M = [e for e in EDGES if L[e] > 1 - 1e-12]
        V = np.array([P[e[1]] - P[e[0]] for e in M])
        rank = np.linalg.matrix_rank(V, 1e-9)
        if rank == 3:
            steps.append((P.copy(), M, None, None))
            break
        if rank == 1:
            v = V[0] / np.linalg.norm(V[0])
            w = xi1 if xi1 is not None else np.array([0.0, 0.0, 1.0])
            xi = w - np.dot(w, v) * v
            xi /= np.linalg.norm(xi)
        else:
            xi = np.linalg.svd(V)[2][-1]
        # |Phi_t e|^2 = |e|^2 + (t^2 - 1)(e.xi)^2 reaches 1 at this t.
        reach = {}
        for e in EDGES:
            if e in M:
                continue
            d = P[e[1]] - P[e[0]]
            a = np.dot(d, xi)
            if abs(a) > 1e-12:
                reach[e] = np.sqrt(1 + (1 - np.dot(d, d)) / a**2)
        new = min(reach, key=reach.get)
        t = reach[new]
        steps.append((P.copy(), M, xi, t, new))
        Phi = np.eye(3) + (t - 1) * np.outer(xi, xi)
        P = P @ Phi.T
    if verbose:
        for st in steps:
            names = [f"{e[0]}{e[1]}" for e in st[1]]
            t = None if st[2] is None else round(st[3], 4)
            print(names, "t* =", t, "new edge", st[4] if len(st) > 4 else "-")
    return steps
