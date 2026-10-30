# Crouzeix–Raviart interpolation constant on tetrahedra

Code and proof records for the paper

> L. Jia and X. Liu, *Guaranteed estimation of the Crouzeix–Raviart
> interpolation constant on tetrahedra*.

For a tetrahedron $K$ with diameter $h_K$, let $C(K)$ be the optimal constant in

$$\|u-\Pi_K u\|_{0,K}\le C(K)\,|u-\Pi_K u|_{1,K},$$

where $\Pi_K$ is the Crouzeix–Raviart interpolation, and let
$\kappa_3=\sup_K C(K)/h_K$. The paper proves

$$\frac{1}{2\pi}\le\kappa_3\le 0.1598 .$$

The lower bound is analytic. The upper bound is computer-assisted, and this
repository contains that computation:

* **Star family:** $C(K)\le 0.1598$, proved with 73 reference shapes and 274
  certified boxes.
* **Path family:** $C(K)\le 0.155$, proved with 137 reference shapes and 1062
  certified boxes.

Every reference bound is an interval-arithmetic enclosure, and the box covers
leave no part of either parameter domain unchecked. An independent replay
program rechecks every record.

## Contents

| Directory | Content |
|---|---|
| `code/` | Julia package `Kappa3Certificate`: Crouzeix–Raviart upper bounds with a verified eigenvalue lower bound, Gram-box comparison, adaptive covers, proof records and replay. See `code/README.md`. |
| `lowerbound/` | Julia project `BernsteinLB`: certified lower bounds $C(K)\ge L_M$ from exact-rational Bernstein trial functions, plus the batch scripts and input lists of the experiments (`RUNBOOK.md`). |
| `examples/` | Small scripts for one shape or one box (see the quick start). |
| `records/campaign/` | The proof: `proof-bundle.txt`, the 210 point records (`points/`), the reference index, the star and path cover records, source fingerprints and the computing environment, all linked by SHA-256 digests. `queue.log` is the run log. |
| `records/aux/` | Tables of the auxiliary computations in Appendix D: calibration (`exp1`), samples (`exp2`), equal-angle stars (`diag`, `small`), lower bounds at the reference points, and the pilot catalogues. |
| `records/make_figures.py` | Regenerates the Appendix D figures from these tables. |

The two dependencies are installed from GitHub at the commits used for the
proof: [VFEM.jl](https://github.com/xfliu/vfem) and
[Veigs.jl](https://github.com/xfliu/veigs) (the `VEIGS.jl` folder).

## Setup

You need Julia 1.12; the records were produced with 1.12.6 on Linux.

```bash
git clone https://github.com/xfliu/crouzeix-raviart-interpolation-constant-3d
cd crouzeix-raviart-interpolation-constant-3d
julia --project=code code/setup.jl
julia --project=lowerbound -e 'using Pkg; Pkg.instantiate()'
```

`code/setup.jl` installs VFEM and Veigs from GitHub, exactly as recorded in
`code/Manifest.toml`.

Run everything single-threaded, as in the proof:

```bash
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

## Quick start: one shape or one box

Angles are in degrees.

* A **star** is given by the three angles $(\alpha,\beta,\gamma)$ between
  three unit edges at a common vertex.
* A **path** is given by $(\theta_1,\theta_2,\varphi)$: the angles $\theta_1$
  and $\theta_2$ between consecutive unit edges of the chain $A\!-\!B\!-\!C\!-\!D$,
  and the dihedral angle $\varphi$ about $BC$.

### Upper bound for one shape

This runs the Crouzeix–Raviart computation with a verified eigenvalue lower
bound. `KAPPA3_LEVEL` is the number of red refinements; the proof uses level
5, that is 32768 cells.

```bash
KAPPA3_FAMILY=star KAPPA3_PARAMETERS=10,10,10 KAPPA3_LEVEL=5 \
  julia --project=code code/scripts/point_certificate.jl
```

The output ends with `C(K) <= 0.15909732836886503`. Level 5 takes from
about 20 seconds to a few minutes. For a quick test use `KAPPA3_LEVEL=3`,
which gives the weaker bound `C(K) <= 0.16699...` in a few seconds.

### Lower bound for one shape

This is a certified Bernstein trial function of degree $M$, which defaults to 10.

```bash
julia --project=lowerbound examples/point_lower.jl star 10 10 10
# certified lower bound C(K) >= 0.158549577082   (a few seconds)
julia --project=lowerbound examples/point_lower.jl path 30 60 16.220114224037754
```

### Upper bound on a whole box of shapes

This uses the 210 stored reference bounds of the proof and the eight-corner
Gram comparison.

```bash
julia --project=code examples/box_bound.jl star 0 1.875 0 1.875 0 1.875
# C(K) <= 0.159774 for every feasible shape in the box
# certified with reference S001 (relabelling 123)

julia --project=code examples/box_bound.jl path 30 37.5 45 52.5 0 11.25
# C(K) <= 0.153773 ...
```

The arguments are `FAMILY lo1 hi1 lo2 hi2 lo3 hi3`, and the endpoints must be
dyadic, such as 60/2^k. The script finds, to six decimals, the smallest
bound that some stored reference certifies on the box, so a large box may
need splitting. To use a single reference, set `BOX_REFERENCE=S001`.

The corner tests are recomputed rigorously. The reference bounds themselves
are read from `records/campaign/points`; the full replay below recomputes them.

## Replaying the whole proof

```bash
julia --startup-file=no --project=code code/scripts/replay_proof.jl \
  records/campaign/proof-bundle.txt
```

The replay first checks the SHA-256 fingerprints of all sources, including
the installed VFEM and Veigs, against `records/campaign/proof/source-fingerprints.txt`.
It then recomputes all 319 reference bounds, which takes about 1.5 hours on
one core, and all 2192 + 8496 corner tests. It ends with

```
KAPPA3 PROOF REPLAY PASSED
  exact targets: star 799/5000, path 31/200
  replayed point proofs: 319
  star retained leaves/corners: 274/2192
  path retained leaves/corners: 1062/8496
```

## License

MIT, see `LICENSE`.
