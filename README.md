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
| `code/` | Julia package `Kappa3Certificate`: Crouzeix–Raviart upper bounds with a verified eigenvalue lower bound, Gram-box comparison, adaptive covers, proof records and replay (`scripts/replay_proof.jl`). See `code/README.md`. |
| `lowerbound/` | Julia project `BernsteinLB`: certified lower bounds $C(K)\ge L_M$ from exact-rational Bernstein trial functions, plus the batch scripts and input lists of the experiments (`RUNBOOK.md`). |
| `deps/` | Snapshots of the two dependencies used in the run, [VFEM.jl](https://github.com/xfliu/vfem) and [Veigs.jl](https://github.com/xfliu/veigs) (MIT licensed). |
| `records/campaign/` | The proof: `proof-bundle.txt`, the 210 point records (`points/`), the reference index, the star and path cover records, source fingerprints and the computing environment, all linked by SHA-256 digests. `queue.log` is the run log. |
| `records/aux/` | Tables of the auxiliary computations in Appendix D: calibration (`exp1`), samples (`exp2`), equal-angle stars (`diag`, `small`), lower bounds at the reference points, and the pilot catalogues. |
| `records/make_figures.py` | Regenerates the Appendix D figures from these tables. |

## Replaying the proof

The campaign ran on Linux with Julia 1.12.6, single-threaded:
`JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

```bash
cd code
julia --project=. -e 'using Pkg; Pkg.instantiate()'
env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  KAPPA3_VFEM_ROOT=../deps/VFEM.jl KAPPA3_VEIGS_ROOT=../deps/Veigs.jl \
  julia --startup-file=no --project=. scripts/replay_proof.jl \
  ../records/campaign/proof-bundle.txt
```

A successful replay ends with

```
KAPPA3 PROOF REPLAY PASSED
  exact targets: star 799/5000, path 31/200
  replayed point proofs: 319
  star retained leaves/corners: 274/2192
  path retained leaves/corners: 1062/8496
```

The replay recomputes all 319 reference bounds, which takes about 1.5 hours
on one core. It also checks the SHA-256 fingerprints of the source files
against `records/campaign/proof/source-fingerprints.txt`.

**Note on paths.** The committed `Manifest.toml` files refer to the two
dependencies by their paths on the original machine:
`/home/xfliu/Workspace/Julia_LIB/Lib/VFEM.jl` and
`/home/xfliu/Workspace/Julia_LIB/Lib/Veigs.jl`. The manifests are part of
the fingerprinted sources, so do not edit them. Instead, copy or symlink
`deps/VFEM.jl` and `deps/Veigs.jl` to these two paths before running the
commands above.

## License

MIT, see `LICENSE`.
