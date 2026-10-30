# Worked examples for Proposition 4.1

This folder holds the scripts and data for the figure that illustrates the
proof of Proposition 4.1 in the paper: the reduction of all tetrahedra to the
star and path families.

The proof stretches a tetrahedron perpendicular to its maximal edges until
one more edge reaches length 1, and repeats until the maximal edges span
$\mathbb R^3$. The constant $C(K)$ never decreases during a stretch. The
figure shows two starting tetrahedra, one ending in each family.

| File | Content |
|---|---|
| `stretch_lib.py` | The stretching procedure. |
| `make_samples.py` | Runs it on the two starting tetrahedra and writes 13 samples per stretch to `samples_path.txt` and `samples_star.txt`. |
| `ritz_general.jl` | Ritz approximation of $C(K)$ with degree-10 Bernstein polynomials for each sample, written to `C_path.txt` and `C_star.txt`. These values are approximations used for illustration; they are not certified bounds. |
| `make_reduction_figure.py` | Draws the figure to `figures/reduction_example.pdf`. |

To regenerate the figure from the repository root (Python with numpy and
matplotlib, Julia 1.12):

```bash
cd records/reduction_example
python3 make_samples.py
for n in path star; do
  julia --project=../../lowerbound ritz_general.jl ../../lowerbound samples_$n.txt > C_$n.txt
done
python3 make_reduction_figure.py
```

Along the stretches, $C(K)$ increases from 0.1294 to 0.1388 for the path
example and from 0.1333 to 0.1438 for the star example.
