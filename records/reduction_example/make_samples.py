"""Sample the two worked examples of Proposition 4.1 along their stretches."""
import numpy as np
from stretch_lib import run

EXAMPLES = {
    "path": [(0, 0, 0), (1, 0, 0), (0.35, 0.55, 0.0), (0.7, 0.25, 0.35)],
    "star": [(0, 0, 0), (1, 0, 0), (0.65, 0.42, 0.0), (0.64, 0.27, 0.41)],
}
for name, P0 in EXAMPLES.items():
    steps = run(P0, verbose=True)
    with open(f"samples_{name}.txt", "w") as f:
        for k, st in enumerate(steps[:-1]):
            P, M, xi, t = st[:4]
            for s in np.linspace(1, t, 13):
                Phi = np.eye(3) + (s - 1) * np.outer(xi, xi)
                Q = P @ Phi.T
                f.write(f"{k} {float(s)!r} " + " ".join(repr(float(x)) for x in Q.flatten()) + "\n")
