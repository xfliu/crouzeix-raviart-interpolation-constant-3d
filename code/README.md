# Rigorous computation for the tetrahedral CR constant

This directory contains the Julia rerun of the interval-arithmetic computation
for the universal three-dimensional Crouzeix--Raviart interpolation constant.
The target is a complete, machine-auditable certificate of

```math
\kappa_3 \le 0.160.
```

The code uses the local, auditable sparse positive-definiteness certificates in
`src/RigorousSPD.jl` and the mesh infrastructure from `VFEM.jl`.  It implements
the boundary condition required by the paper: four zero means on the
*original* tetrahedron faces. It does not use the Dirichlet boundary restriction
in `VFEM.verified_cr_laplace_3d`.

For a fixed shape, the certificate consists of:

1. an affine red-refined tetrahedral mesh;
2. interval CR mass and stiffness matrices;
3. a sparse basis for the four original-face mean constraints;
4. a verified positive-definiteness test for
   `A_h - lambda_lower M_h` on the constrained space; and
5. the Liu correction
   `C(K) <= sqrt(1/lambda_lower + (0.3804 h)^2)`.

`RigorousSPD.jl` exposes two independent one-sided tests:

- `certify_spd_residual(B)` is the preferred manuscript-style certificate. It
  factors a downward-shifted midpoint and rigorously bounds the backward error
  of the computed sparse Cholesky factor. The returned record includes the
  interval-radius, factor-product, roundoff, underflow, total-error, and strict
  margin bounds.
- `certify_spd(B; roundoff_bound=:dense)` implements Rump's a priori bound I.
  It is a useful independent replay because the bound is permutation
  independent. `roundoff_bound=:envelope` selects the sharper sparse bound II.

Both routines take the symmetric interval hull, optionally apply an interval
power-of-two diagonal congruence, round all scalar reductions outward, and fail
closed. `certified == false` is inconclusive rather than evidence that the
matrix is indefinite.

The continuous parameter cover uses outward-rounded Gram-box comparisons.
All large cover records and checkpoints are stored on the `hpc` scratch disk,
not under `/home`.

## Reproducible environment on `hpc`

```bash
cd /home/xfliu/Workspace/Paper_Jia_Lueling_CR_Constant_3D/Code
/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. setup.jl
```

Smoke test with one Julia thread while the node is busy:

```bash
env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  KAPPA3_LEVEL=3 \
  /home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. scripts/smoke_points.jl
```

Focused SPD tests:

```bash
env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. test/runtests_spd.jl
```

The publication-sized level-5 CR test is intentionally separate:

```bash
env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  KAPPA3_TEST_LEVEL=5 KAPPA3_TEST_BACKOFF=1e-4 \
  /home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. test/runtests_spd_level5.jl
```

The production cover must not be started while the unrelated
`global_cover_run1` job occupies the node. Torque/PBS was unavailable when this
project was initialized. The production launcher performs its own blocker,
load, memory, disk, pinned-Julia, and single-run-lock checks. A non-starting
check is available with

```bash
./scripts/launch_production.sh --check-only
```

It exits with status 75 while `global_cover_run1` is active. The launcher does
not kill, suspend, renice, or otherwise interact with that computation.
Setting `KAPPA3_ALLOW_CONCURRENT_GLOBAL_COVER=1` (and raising
`KAPPA3_MAX_LOAD1`, e.g. to 95) lets the single sequential runner use the idle
cores alongside it at nice 19; the memory and disk checks still apply. The
campaign started on 2026-10-29 uses this opt-in:

```bash
nohup setsid env KAPPA3_ALLOW_CONCURRENT_GLOBAL_COVER=1 KAPPA3_MAX_LOAD1=95 \
  ./scripts/queue_production.sh \
  >> /mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d.queue.log 2>&1 < /dev/null &
```

## Resumable production workflow

Do not initialize the campaign until all source edits are frozen: the first
production invocation hashes every source, script, and test, and later source
changes are rejected. Heavy modes can run only through the guarded launcher.
The runner is deliberately sequential so two CHOLMOD factorizations never
share a process. Start with the non-mutating readiness check, then initialize:

```bash
./scripts/launch_production.sh --check-only
./scripts/launch_production.sh initialize
```

Run one bounded step with, at most, eight new point solves and 200,000 search
nodes in total:

```bash
./scripts/launch_production.sh run \
  --max-point-solves=8 \
  --max-search-nodes=200000 \
  --max-enrichments=8 \
  --max-cycles=2
```

An incomplete bounded step exits with status 75 after atomically saving every
point and search checkpoint; rerun the same command to resume. `status` is
read-only and may be invoked directly or through the launcher:

```bash
/home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. scripts/run_production.jl status
```

For an unattended run on this particular shared host, the conservative queue
driver waits while the unrelated `global_cover_run1` process is present (or
while the load/memory/disk guard refuses), initializes provenance only after
the guard becomes ready, and then repeats those same bounded resumable steps:

```bash
nohup setsid ./scripts/queue_production.sh \
  >> /mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d.queue.log 2>&1 < /dev/null &
```

It holds a kernel-released singleton lock and stops on a fatal status; exit 73
or 75 is treated as transient. Three ready retries with an unchanged campaign
summary stop with status 74 instead of spinning forever. It never kills,
suspends, renices, or otherwise changes the unrelated computation. Its
readiness state can be queried without starting or initializing anything using
`./scripts/queue_production.sh --check-only`.

The initial catalogue is exactly `full_reference_plan()`: 73 star and 137
path level-5 references, namely those used by the 2026-10-29 pilot covers
(`LowerBound/RUNBOOK.md`). The cover targets are exact per family: star
`799/5000` (the theorem bound 0.1598) and path `31/200` (0.155). At the
configured depth limits, unresolved exact dyadic cells trigger deterministic
new level-5 reference points. A catalogue extension
restarts only the affected family search, ensuring the new references are
tested on coarser cells. Existing point files are parsed and provenance-checked
on resume, and every point is independently replayed before it can enter a
search catalogue.

Only completion of both canonical covers creates `proof-bundle.txt`; the
runner then launches the independent top-level replay. To replay an existing
bundle under the same blocker/resource guard, use

```bash
./scripts/launch_production.sh replay
```

## Proof records and independent replay

`src/ProofRecords.jl` keeps search heuristics out of the final proof chain.
Each point record contains the shape family and angles, refinement level,
constraint basis, mesh strategy, verified lower eigenvalue, SPD method and
audit summary, Gram enclosure, and computed correction bound. Float64 values
are shown in decimal and accompanied by their exact binary64 bits. Parsers
require the canonical field order, reject trailing input, and reject decimal/
bit disagreements.

A replay does **not** trust the stored value of `C(K)^2`. It reconstructs the
Gram matrix, refined mesh, interval mass/stiffness pencil, and four face-mean
constraints; reruns the SPD proof at the recorded `lambda_lower` without
ARPACK using the a posteriori Cholesky-residual method; checks every stored
residual audit value bit-for-bit; and recomputes `hmax^2` and `C(K)^2`. Legacy
envelope certificates are rejected by the final record parser. Only then can
the replay create a cover reference. A SHA-256 reference index binds every
reference to its point file.

Reference entries may also carry an exact `3 x 3` integer change of edge
basis. Replay accepts exactly the 24 matrices induced by permuting the four
tetrahedron vertices (not arbitrary unimodular matrices), and forms `T*G*T'`
by signed interval additions. This records path reversal and other vertex
relabellings without pretending that a second eigenproblem was solved.

The guarded `initialize` mode hashes `Project.toml`, `Manifest.toml`, all
verifier scripts/sources/tests, and the Julia sources and package metadata of
the locally developed VFEM and Veigs packages. The lower-level
`prepare_provenance.jl` helper is retained for format diagnostics but should
not be mixed with a production campaign. Large and generated files live only under
`/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d`, in `checkpoints`, `logs`, `points`,
`proof`, and `tmp` subdirectories.

The final `proof-bundle.txt` contains SHA-256 links to the environment,
fingerprints, reference index, and the two cover records. Its top-level replay
is

```bash
env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/xfliu/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project=. scripts/replay_proof.jl \
  /mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d/proof-bundle.txt
```

Before replaying any point matrices, this verifier requires the exact targets
`799/5000` (star) and `31/200` (path) and the canonical star `[0,60]^3` and path `[0,60]^2 x [0,90]` roots.
