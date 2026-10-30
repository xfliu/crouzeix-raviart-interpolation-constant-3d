# src/core/eigensolve2d/lambda_h_bernstein.jl
#
# Inverse-free evaluation of
#
#     lambda_{h,B} = 1 / max_i (B A^{-1} B')_ii ,
#
# the constant of the L^inf Lagrange interpolation error estimate of
# J. Galindo, K. Ike, X. Liu, over the Fujino-Morley space assembled by
# `create_matrix_fujino_morley`. This file is the replacement for Lemma 3.2 of
# that manuscript, in both a float and a rigorously certified version.
#
# THE POINT
#   The published route forms the ENTIRE dense N x N matrix D = B A^{-1} B' and
#   keeps only its diagonal. At the paper's production size (M = 8382,
#   N = 24576) that is 4.50 GB for D alone, ~9.63 GB peak.
#   But every row beta_i of B has at most 6 nonzeros, and
#
#       1 / lambda_{h,B} = max_i beta_i' A^{-1} beta_i ,                    (G)
#
#   so only sparse-right-hand-side triangular solves against one Cholesky factor
#   are needed -- the inverse is never formed. Memory is O(nnz(L)) (a few MB;
#   nnz(L) = 350083 at M = 8382, mean reach set per solve 393 entries = 0.11% of
#   L), and the measured peak RSS for the whole certified computation at n = 64
#   is 562 MB against the published route's 4.50 GB.
#
# RIGOUR
#   Two-sided variational bounds on g_i = beta_i' A^{-1} beta_i, evaluated on a
#   converged float trial vector z (`certified_g_lower` / `certified_g_upper`),
#   need only a cheap rigorous LOWER bound on lambda_min(A) -- supplied by
#   `lambda_min_lower`, which certifies a shift by interval Cholesky (Alefeld's
#   criterion) or by Rump's backward-error bound. No interval inverse, and hence
#   none of the O(M^3) interval width accumulation of the published route.
#
# SCREENING IS A NEGATIVE RESULT ON THESE MATRICES -- reported honestly
#   `lambda_hb_certified` screens rows with the crude rigorous bound
#   g_i <= ||beta_i||_2^2 / lambda_min(A). That is tight only when beta_i aligns
#   with the lowest eigenvector of A. Here beta_i has 6 nonzeros and is strongly
#   localised while the lowest eigenmode is smooth, so the bound overshoots
#   g_max by ~760x and rejects 3 of 24576 rows -- and those 3 only because they
#   are empty (all six of their columns are deleted corner DOFs). The headline
#   win comes from the sparse-RHS solves, NOT from screening.
#   `screen_sharp` is a sharper rigorous alternative using the certified
#   diagonal of A^{-1} restricted to each row's own support; it cuts survivors
#   to 59/24576 (0.24%) and reproduces the certified upper bound BIT-IDENTICALLY
#   for a 2.9x speedup, at the price of M solves up front.
#
# ACCURACY EXPECTATIONS -- cond(A) grows like n^4
#   Measured cond(A) at theta = pi/6: 8.4e1, 5.5e2, 6.9e3, 9.7e4, 1.5e6 for
#   n = 2, 4, 8, 16, 32. Both this route and a dense-D route are backward
#   stable, so each carries a forward error O(cond(A)*eps) in lambda and so does
#   their DIFFERENCE: ~3e-10 at n = 32, ~2e-8 at n = 64. Demanding agreement to
#   1e-12 at n = 32 would be a demand on double precision that no implementation
#   can meet. The acceptance criterion used in the tests is instead: both values
#   lie inside the certified enclosure, and their relative difference is
#   <= 100*cond(A)*eps.
#   Note also that the argmax row is frequently GENUINELY TIED (top two g_i
#   agreeing to 1e-14..7e-12 with overlapping rigorous enclosures), so argmax
#   equality between two routes is NOT a valid test by itself.
#
# Conventions: 1-based indices, `SparseMatrixCSC`, plain `.jl` file. The float
# path is threaded over row chunks (`nchunks`); the certified path routes the
# same arithmetic through `Interval{Float64}` with outward rounding.

# `diam` is the only IntervalArithmetic name this file needs beyond the set the
# module header of src/VFEM.jl already brings in (Interval, interval, inf, sup,
# mid, hull, mag), so it is imported here rather than by widening that shared
# header.
using IntervalArithmetic: diam

# ---------------------------------------------------------------------------
#  lambda_hb_fast.jl -- inverse-free floating-point evaluation of
#
#        lambda_{h,B} = min { x'A x : ||B x||_inf >= 1 } = 1 / max_i g_i,
#        g_i = beta_i' A^{-1} beta_i,   beta_i' = row i of B.
#
#  Provenance
#     Replacement for Lemma 3.2 (\label{lem:opti_problem_lower_bound}) of
#     J. Galindo, K. Ike, X. Liu, "L^infty error estimates for Lagrange
#     interpolation" (manuscript).  The authors' Octave reference
#     implementation forms
#           A_global : DENSE  M x M
#           D        = B*(A\B')   DENSE  N x N        (N = 6*nt)
#           lambda   = 1/max(diag(D))
#     i.e. O(M^3 + N^2 M) flops and O(N^2) memory to extract N numbers.
#     At h = 1/64 (M = 8382, N = 24576) that is 4.83 GB for D plus 1.65 GB for
#     the intermediate A\B'.  This file computes the same N numbers from a
#     sparse Cholesky factorization and N sparse-right-hand-side triangular
#     solves, using O(nnz(L)) memory.
#
#  Conventions (VFEM.jl style)
#     * 1-based indices, SparseMatrixCSC for all global matrices.
#     * routines generic in the element type via the keyword `T::Type = Float64`
#       so that `Interval{Float64}` data flows through unchanged; the certified
#       driver in `lambda_hb_certified.jl` uses that mode.
#     * plain .jl file, no module wrapper: `include("lambda_hb_fast.jl")`.
#
#  Mathematical content (proved in `inverse_free_lemma.md`)
#     Since  max_{||Bx||_inf >= 1} is attained on a single coordinate of Bx,
#           1/lambda_{h,B} = max_i beta_i' A^{-1} beta_i
#     and with the Cholesky factorization  A[p,p] = L L'  (CHOLMOD, AMD order)
#           beta' A^{-1} beta = || L^{-1} beta[p] ||_2^2 .
#     beta_i has at most 6 nonzeros (one element's local Fujino-Morley DOFs), so
#     by the Gilbert-Peierls reachability theorem the solution L^{-1}beta[p] is
#     nonzero only on the set reachable from the pattern of beta[p] in the graph
#     of L; each solve therefore touches far fewer than nnz(L) entries.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Sparse triangular solve with sparse right-hand side (Gilbert-Peierls)
# ---------------------------------------------------------------------------

"""
    ReachWorkspace(n; T = Float64)

Per-thread scratch space for [`solve_sparse_rhs!`](@ref).

Fields
* `x`      : dense accumulator of length `n` (kept zero outside the reach set)
* `xi`     : integer stack of length `n`, receives the topological order
* `stack`  : DFS node stack, length `n`
* `pstack` : DFS "next child" pointer stack, length `n`
* `marked` : `Bool` marker array, length `n` (reset after every solve)
"""
struct ReachWorkspace{T}
    x::Vector{T}
    xi::Vector{Int}
    stack::Vector{Int}
    pstack::Vector{Int}
    marked::Vector{Bool}
end

function ReachWorkspace(n::Integer; T::Type = Float64)
    return ReachWorkspace{T}(zeros(T, n), zeros(Int, n), zeros(Int, n),
                             zeros(Int, n), falses(n))
end

# depth-first search from node j0 in the graph of the lower-triangular L,
# pushing finished nodes at xi[top], top decreasing.  Faithful port of
# csparse's cs_dfs, iterative (no recursion depth limit).
@inline function _dfs!(j0::Int, Lp::Vector{Int}, Li::Vector{Int}, top::Int,
                       xi::Vector{Int}, stack::Vector{Int},
                       pstack::Vector{Int}, marked::Vector{Bool})
    head = 1
    stack[1] = j0
    while head >= 1
        j = stack[head]
        if !marked[j]
            marked[j] = true
            pstack[head] = Lp[j]        # first entry of column j is the diagonal
        end
        done = true
        p = pstack[head]
        pend = Lp[j + 1] - 1
        while p <= pend
            i = Li[p]
            if i <= j || marked[i]
                p += 1
                continue
            end
            pstack[head] = p + 1        # resume after this child
            head += 1
            stack[head] = i
            done = false
            break
        end
        if done
            pstack[head] = pend + 1
            head -= 1
            top -= 1
            xi[top] = j
        end
    end
    return top
end

"""
    solve_sparse_rhs!(ws, L, idx, val) -> (top, nreach)

Solve `L y = b` where `b` is the sparse vector with `b[idx[k]] = val[k]`, using
the reachability (Gilbert-Peierls) theorem: the nonzero pattern of `y` is the
set of nodes reachable from `pattern(b)` in the directed graph of `L`.

On return `ws.xi[top:end]` lists the nonzero positions of `y` in topological
order and `ws.x` holds the corresponding values; `nreach = n - top + 1`.
The caller must consume the result and then call [`clear_workspace!`](@ref).

`L` must be lower triangular in CSC form with sorted row indices and an
explicit diagonal (the format returned by `sparse(cholesky(A).L)`).
"""
function solve_sparse_rhs!(ws::ReachWorkspace{T}, L::SparseMatrixCSC{T,Int},
                           idx, val) where {T}
    n = L.n
    Lp = L.colptr; Li = L.rowval; Lx = L.nzval
    x = ws.x; xi = ws.xi
    top = n + 1
    @inbounds for k in eachindex(idx)
        j = idx[k]
        if !ws.marked[j]
            top = _dfs!(j, Lp, Li, top, xi, ws.stack, ws.pstack, ws.marked)
        end
    end
    @inbounds for k in eachindex(idx)
        x[idx[k]] += val[k]
    end
    @inbounds for px in top:n
        ws.marked[xi[px]] = false          # unmark for the next solve
    end
    @inbounds for px in top:n
        j = xi[px]
        xj = x[j] / Lx[Lp[j]]              # divide by L[j,j]
        x[j] = xj
        for p in (Lp[j] + 1):(Lp[j + 1] - 1)
            x[Li[p]] -= Lx[p] * xj
        end
    end
    return top, n - top + 1
end

"""
    clear_workspace!(ws, top)

Zero the accumulator on the reach set produced by the last
[`solve_sparse_rhs!`](@ref) call, leaving `ws` reusable in O(nreach).
"""
@inline function clear_workspace!(ws::ReachWorkspace{T}, top::Int) where {T}
    n = length(ws.x)
    @inbounds for px in top:n
        ws.x[ws.xi[px]] = zero(T)
    end
    return nothing
end

# ---------------------------------------------------------------------------
#  Row extraction from B
# ---------------------------------------------------------------------------

"""
    csr_rows(B) -> (rowptr, colidx, nzval)

Compressed-row view of `B::SparseMatrixCSC` (i.e. `transpose` in CSC form),
so that row `i` of `B` is `colidx[rowptr[i]:rowptr[i+1]-1]` with values
`nzval[...]`.  Rows of the Fujino-Morley `B` carry at most 6 nonzeros.
"""
function csr_rows(B::SparseMatrixCSC{T,Int}) where {T}
    Bt = sparse(transpose(B))
    return Bt.colptr, Bt.rowval, Bt.nzval
end

# ---------------------------------------------------------------------------
#  Fast path 1: sparse Cholesky + one reach-set triangular solve per row
# ---------------------------------------------------------------------------

"""
    g_all_trisolve(A, B; nchunks = Threads.nthreads(), F = nothing)
        -> (g, diagnostics)

All `N` quantities `g_i = beta_i' A^{-1} beta_i` from one sparse Cholesky
factorization of `A` and one sparse-right-hand-side triangular solve per row of
`B`.  Threaded over chunks of rows, each chunk with its own workspace.

Returns the vector `g` of length `size(B,1)` and a NamedTuple with the
factorization time, solve time, `nnz(L)`, the mean reach-set size (the
efficiency figure of merit: mean reach / `nnz(L)` per column) and the peak
working memory of the factor.
"""
function g_all_trisolve(A::SparseMatrixCSC{Float64,Int},
                        B::SparseMatrixCSC{Float64,Int};
                        nchunks::Integer = Threads.nthreads(),
                        F = nothing)
    M = size(A, 1); N = size(B, 1)
    @assert size(B, 2) == M "B has $(size(B,2)) columns, A is $M x $M"
    t0 = time()
    Fc = F === nothing ? cholesky(A) : F
    L = sparse(Fc.L)
    invp = invperm(Fc.p)
    tfact = time() - t0

    rowptr, colidx, nzval = csr_rows(B)
    g = Vector{Float64}(undef, N)
    reach_tot = Threads.Atomic{Int}(0)

    t0 = time()
    nch = max(1, Int(nchunks))
    bounds = [1 + div((c - 1) * N, nch) for c in 1:(nch + 1)]
    bounds[end] = N + 1
    tasks = map(1:nch) do c
        Threads.@spawn begin
            ws = ReachWorkspace(M; T = Float64)
            idxbuf = Int[]; valbuf = Float64[]
            rloc = 0
            for i in bounds[c]:(bounds[c + 1] - 1)
                resize!(idxbuf, 0); resize!(valbuf, 0)
                for q in rowptr[i]:(rowptr[i + 1] - 1)
                    push!(idxbuf, invp[colidx[q]])
                    push!(valbuf, nzval[q])
                end
                if isempty(idxbuf)
                    g[i] = 0.0
                    continue
                end
                top, nr = solve_sparse_rhs!(ws, L, idxbuf, valbuf)
                s = 0.0
                @inbounds for px in top:M
                    v = ws.x[ws.xi[px]]
                    s += v * v
                end
                g[i] = s
                rloc += nr
                clear_workspace!(ws, top)
            end
            Threads.atomic_add!(reach_tot, rloc)
        end
    end
    foreach(wait, tasks)
    tsolve = time() - t0

    return g, (t_factor = tfact, t_solve = tsolve, nnzL = nnz(L),
               mean_reach = reach_tot[] / max(N, 1), M = M, N = N,
               mem_bytes = nnz(L) * 12, nchunks = nch, method = :trisolve)
end

# ---------------------------------------------------------------------------
#  Fast path 2: selected inversion (Takahashi / SelInv)
# ---------------------------------------------------------------------------

"""
    selinv(A; F = nothing) -> (Zp, Zi, Zx, invp)

Selected inversion: the entries of `A^{-1}` (in the CHOLMOD permutation) on the
**lower** part of the symbolic pattern of the Cholesky factor, computed by the
Takahashi/SelInv backward recursion.  With `A[p,p] = L L'` and
`I_j = {i > j : L[i,j] != 0}`,

    Z[I_j, j] = -Z[I_j, I_j] * L[I_j, j] / L[j,j]
    Z[j, j]   = 1/L[j,j]^2 - L[I_j, j]' * Z[I_j, j] / L[j,j]

evaluated for `j = n, n-1, ..., 1`.  Every `I_j` is a clique of the symbolic
lower pattern (the classical fill-path property), so all entries needed on the
right-hand side are already available in lower storage: for `ia, ib in I_j` the
required value is `Z[max(ia,ib), min(ia,ib)]`, which lies in the pattern.

The pattern comes from [`symbolic_cholesky`](@ref), **not** from `sparse(F.L)`:
CHOLMOD prunes numerically zero entries, and on a pruned pattern `I_j` is no
longer a clique and the recursion fails.

Returns the pattern `(Zp, Zi)`, the values `Zx` of the lower triangle of
`A[p,p]^{-1}`, and the inverse permutation, so that `A^{-1}[a,b]` equals
`Z[max(u,v), min(u,v)]` with `u = invp[a]`, `v = invp[b]`.  Because the 6 local
DOFs of an element form a clique of the graph of `A`, every 6x6 block that `B`
requires is present.
"""
function selinv(A::SparseMatrixCSC{Float64,Int}; F = nothing)
    Fc = F === nothing ? cholesky(A) : F
    Ap = A[Fc.p, Fc.p]
    Zp, Zi = symbolic_cholesky(Ap)
    n = Ap.n
    # numeric factor on the symbolic pattern (left-looking, same pattern)
    ok, Lx, _ = sparse_chol_shift(Ap, 0.0, Zp, Zi; T = Float64)
    ok || error("selinv: Cholesky of A failed; A must be positive definite")
    Zx = zeros(Float64, length(Zi))
    loc = zeros(Int, n)          # row -> local index within I_j
    S = Float64[]                # dense nI x nI gather buffer
    y = Float64[]
    @inbounds for j in n:-1:1
        p0 = Zp[j]
        djj = Lx[p0]
        nI = Zp[j + 1] - 1 - p0
        if nI == 0
            Zx[p0] = 1.0 / (djj * djj)
            continue
        end
        length(S) < nI * nI && (S = zeros(Float64, nI * nI))
        length(y) < nI && (y = zeros(Float64, nI))
        for a in 1:nI
            loc[Zi[p0 + a]] = a
        end
        fill!(view(S, 1:(nI * nI)), 0.0)
        # gather Z[I_j, I_j]: walk each column ib in I_j once, keep marked rows
        for b in 1:nI
            ib = Zi[p0 + b]
            for q in Zp[ib]:(Zp[ib + 1] - 1)
                a = loc[Zi[q]]
                a == 0 && continue
                S[(b - 1) * nI + a] = Zx[q]      # S[a,b], a >= b (lower part)
            end
        end
        for b in 1:nI, a in 1:(b - 1)            # mirror to the upper part
            S[(b - 1) * nI + a] = S[(a - 1) * nI + b]
        end
        for a in 1:nI
            loc[Zi[p0 + a]] = 0
        end
        # y = S * L[I_j, j]
        for a in 1:nI
            acc = 0.0
            for b in 1:nI
                acc += S[(b - 1) * nI + a] * Lx[p0 + b]
            end
            y[a] = acc
        end
        zjj = 1.0 / (djj * djj)
        for a in 1:nI
            zij = -y[a] / djj
            Zx[p0 + a] = zij
            zjj -= Lx[p0 + a] * zij / djj
        end
        Zx[p0] = zjj
    end
    return Zp, Zi, Zx, invperm(Fc.p)
end

"""
    g_all_selinv(A, B; F = nothing) -> (g, diagnostics)

All `g_i = beta_i' A^{-1} beta_i` from a selected inversion of `A`: each row of
`B` needs only the small symmetric block of `A^{-1}` on its own support, and
that block is inside the pattern returned by [`selinv`](@ref).
"""
function g_all_selinv(A::SparseMatrixCSC{Float64,Int},
                      B::SparseMatrixCSC{Float64,Int}; F = nothing)
    M = size(A, 1); N = size(B, 1)
    t0 = time()
    Zp, Zi, Zx, invp = selinv(A; F = F)
    tsel = time() - t0

    rowptr, colidx, nzval = csr_rows(B)
    g = Vector{Float64}(undef, N)
    t0 = time()
    @inbounds for i in 1:N
        rng = rowptr[i]:(rowptr[i + 1] - 1)
        s = 0.0
        for q in rng
            a = invp[colidx[q]]; va = nzval[q]
            for r in rng
                b = invp[colidx[r]]; vb = nzval[r]
                hi = max(a, b); lo = min(a, b)
                pos = find_in_column(Zp, Zi, lo, hi)
                pos == 0 && error("selinv: A^{-1}[$a,$b] outside the pattern")
                s += va * vb * Zx[pos]
            end
        end
        g[i] = s
    end
    tform = time() - t0
    return g, (t_factor = tsel, t_solve = tform, nnzL = length(Zi), M = M,
               N = N, mem_bytes = length(Zi) * 12, method = :selinv)
end

# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

"""
    lambda_hb_fast(A, B; method = :trisolve, nchunks = Threads.nthreads())
        -> (lambda, imax, diagnostics::NamedTuple)

Floating-point value of

    lambda_{h,B} = min { x'A x : ||B x||_inf >= 1 } = 1 / max_i g_i,
    g_i = beta_i' A^{-1} beta_i,  beta_i' = row i of B,

computed without ever forming `A^{-1}` or `D = B A^{-1} B'`.

Arguments
* `A :: SparseMatrixCSC{Float64}` : `M x M` symmetric positive definite
* `B :: SparseMatrixCSC{Float64}` : `N x M`, few nonzeros per row
* `method` : `:trisolve` (sparse Cholesky + reach-set solves, default),
             `:selinv` (selected inversion), or `:auto`
* `nchunks` : number of row chunks for threading (`:trisolve` only)

Returns
* `lambda`      : `1 / max_i g_i`
* `imax`        : the maximizing row index `i*` of `B`
* `diagnostics` : `(g, gmax, t_factor, t_solve, nnzL, mean_reach, M, N,
                   mem_bytes, method)`; `g` is the full vector, so the certified
                   driver can screen without recomputation.
"""
function lambda_hb_fast(A::SparseMatrixCSC{Float64,Int},
                        B::SparseMatrixCSC{Float64,Int};
                        method::Symbol = :trisolve,
                        nchunks::Integer = Threads.nthreads(),
                        F = nothing)
    if method === :auto
        method = size(B, 1) > 4 * size(A, 1) ? :selinv : :trisolve
    end
    g, d = method === :selinv ? g_all_selinv(A, B; F = F) :
                                g_all_trisolve(A, B; nchunks = nchunks, F = F)
    gmax, imax = findmax(g)
    return 1 / gmax, imax, merge(d, (g = g, gmax = gmax, imax = imax))
end

# ---------------------------------------------------------------------------
#  Reference baselines (for benchmarking only -- O(N^2) memory)
# ---------------------------------------------------------------------------

"""
    lambda_hb_baseline_denseD(A, B) -> (lambda, imax, diagnostics)

Verbatim port of the authors' Octave route: densify `A`, form the **entire**
`N x N` matrix `D = B*(A\\B')`, and take `1/max(diag(D))`.  Allocates
`8 N^2 + 8 M N` bytes and is here only to define correctness and to measure
what the sparse path replaces.  Refuses to run above `maxN` (default 6000) to
avoid exhausting memory.
"""
function lambda_hb_baseline_denseD(A::SparseMatrixCSC{Float64,Int},
                                   B::SparseMatrixCSC{Float64,Int};
                                   maxN::Integer = 6000)
    M = size(A, 1); N = size(B, 1)
    need = 8 * (N^2 + M * N) / 2^30
    N <= maxN || error("baseline needs $(round(need, digits=2)) GB for N = $N; " *
                       "raise maxN only if that fits in RAM")
    t0 = time()
    Ad = Matrix(A)
    Bd = Matrix(B)
    X = Ad \ transpose(Bd)          # M x N dense
    D = Bd * X                      # N x N dense
    dg = diag(D)
    gmax, imax = findmax(dg)
    t = time() - t0
    return 1 / gmax, imax, (t_total = t, g = dg, gmax = gmax, imax = imax,
                            mem_bytes = 8 * (N^2 + M * N), method = :denseD)
end

"""
    lambda_hb_baseline_denseinv(A, B) -> (lambda, imax, diagnostics)

Intermediate baseline: dense `inv(A)` followed by the `N` quadratic forms
`beta_i' A^{-1} beta_i` on the sparse rows.  `O(M^3)` time, `O(M^2)` memory --
cheaper than forming `D`, still cubic.
"""
function lambda_hb_baseline_denseinv(A::SparseMatrixCSC{Float64,Int},
                                     B::SparseMatrixCSC{Float64,Int})
    M = size(A, 1); N = size(B, 1)
    t0 = time()
    Ai = inv(Matrix(A))
    rowptr, colidx, nzval = csr_rows(B)
    g = Vector{Float64}(undef, N)
    for i in 1:N
        rng = rowptr[i]:(rowptr[i + 1] - 1)
        s = 0.0
        for q in rng, r in rng
            s += nzval[q] * nzval[r] * Ai[colidx[q], colidx[r]]
        end
        g[i] = s
    end
    gmax, imax = findmax(g)
    t = time() - t0
    return 1 / gmax, imax, (t_total = t, g = g, gmax = gmax, imax = imax,
                            mem_bytes = 8 * M^2, method = :denseinv)
end

"""
    baseline_cost_model(M, N) -> NamedTuple

Memory and flop counts of the authors' route, for extrapolation beyond what
fits in RAM: `D` needs `8N^2` bytes, the intermediate `A\\B'` needs `8MN`,
the dense Cholesky is `M^3/3` flops and the two products `2M^2N + 2MN^2`.
"""
function baseline_cost_model(M::Integer, N::Integer)
    return (bytes_D = 8.0 * N^2, bytes_X = 8.0 * M * N,
            bytes_A = 8.0 * M^2,
            gflops = (M^3 / 3 + 2.0 * M^2 * N + 2.0 * M * N^2) / 1e9)
end

# ---------------------------------------------------------------------------
#  lambda_min_lower_bound.jl -- rigorous lower bounds for lambda_min(A),
#                               A sparse symmetric positive definite.
#
#  Provenance
#     Supporting routine for the inverse-free replacement of Lemma 3.2 in
#     J. Galindo, K. Ike, X. Liu (manuscript on L^infty Lagrange interpolation
#     error constants).  In the certified evaluation of
#         g = beta' A^{-1} beta <= z'A z + 2 z'r + ||r||_2^2 / lmin,
#         r := beta - A z,
#     the bound `lmin <= lambda_min(A)` multiplies only ||r||^2.  Since z comes
#     from a sparse Cholesky solve plus iterative refinement, ||r||^2 is at the
#     10^{-30} level, so `lmin` may be loose by orders of magnitude and still
#     contribute nothing to the final width.  That is the efficiency argument
#     for using a cheap verified bound here instead of a verified eigensolver.
#
#  Two independent rigorous routes, both certificate-style (they either return a
#  proven bound or report failure; they never return an unproven number):
#
#     (1) `verify_pd_interval`  -- sparse Cholesky of A - sigma*I carried out in
#         interval arithmetic on the exact symbolic fill pattern.  If every
#         pivot has a positive lower bound the factorization exists for every
#         symmetric matrix in the interval hull, in particular for the point
#         matrix A - sigma*I; hence A - sigma*I is positive definite and
#         lambda_min(A) > sigma  (Alefeld's interval Cholesky criterion).
#
#     (2) `verify_pd_rump`      -- Rump/Higham style backward-error bound.  If
#         the floating-point Cholesky of the computed Ah = fl(A - sigma*I) runs
#         to completion then Ah + Delta = R'R exactly with
#             |Delta| <= gamma_{n+1} |R'||R|,  gamma_k = k*u/(1 - k*u),
#         so lambda_min(A - sigma*I) >= -||Delta||_2 >= -gamma_{n+1} || |R'||R| ||_1
#         and lambda_min(A) >= sigma - gamma_{n+1} || |R'||R| ||_1 - u*max_j|Ah_jj|.
#         Cost: one float factorization plus O(nnz(L)); no interval arithmetic.
#
#  Conventions (VFEM.jl style)
#     * 1-based indices, SparseMatrixCSC throughout.
#     * routines generic in the element type via `T::Type = Float64`; passing
#       `T = Interval{Float64}` selects the verified mode of the factorization.
#     * plain .jl file, no module wrapper.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Certificate 1: interval Cholesky
# ---------------------------------------------------------------------------

"""
    verify_pd_interval(A, sigma; perm = nothing, Lp = nothing, Li = nothing)
        -> (ok, minpivot)

Certify `A - sigma*I` positive definite (hence `lambda_min(A) > sigma`) by a
sparse interval Cholesky factorization.  `perm` is an optional fill-reducing
permutation; `A[perm,perm] - sigma*I` is factored, which is positive definite
exactly when `A - sigma*I` is (a symmetric permutation preserves the spectrum).
"""
function verify_pd_interval(A::SparseMatrixCSC{Float64,Int}, sigma::Real;
                            perm = nothing, Lp = nothing, Li = nothing)
    Ap = perm === nothing ? A : A[perm, perm]
    if Lp === nothing || Li === nothing
        Lp, Li = symbolic_cholesky(Ap)
    end
    ok, _, mp = sparse_chol_shift(Ap, sigma, Lp, Li; T = Interval{Float64})
    return ok, mp
end

# ---------------------------------------------------------------------------
#  Certificate 2: Rump/Higham backward-error bound
# ---------------------------------------------------------------------------

"""
    verify_pd_rump(A, sigma; perm = nothing) -> (ok, bound)

Rump/Higham style certificate.  If the floating-point Cholesky of
`Ah = fl(A - sigma*I)` completes, then `Ah + Delta = R'R` exactly with
`|Delta| <= gamma_{n+1} |R'||R|`, `gamma_k = k u/(1 - k u)`, so

    lambda_min(A) >= sigma - gamma_{n+1} * || |R'||R| ||_1 - u * max_j |Ah_jj| ,

the last term absorbing the rounding of the shifted diagonal.  The 1-norm is
computed in `O(nnz(L))` from the identity

    ( |R'||R| )  column sums_i  =  sum_k |L[i,k]| * ||L[:,k]||_1 ,

and the result is rounded outward.  Returns `ok = true` when the resulting
`bound` is positive, together with that proven lower bound for
`lambda_min(A)`.

This route needs no interval arithmetic and costs one float factorization; it
is the fallback when the interval Cholesky is too slow or breaks down.
"""
function verify_pd_rump(A::SparseMatrixCSC{Float64,Int}, sigma::Real;
                        perm = nothing)
    Ap = perm === nothing ? A : A[perm, perm]
    n = Ap.n
    Ah = Ap - sigma * I
    F = try
        cholesky(Ah; check = true)
    catch
        return false, -Inf
    end
    L = sparse(F.L)
    u = eps(Float64) / 2
    gam = (n + 1) * u / (1 - (n + 1) * u)
    (n + 1) * u >= 1 && return false, -Inf
    # s[k] = ||L[:,k]||_1 ; colsum[i] = sum_k |L[i,k]| s[k]
    s = zeros(Float64, n)
    @inbounds for k in 1:n
        acc = 0.0
        for p in L.colptr[k]:(L.colptr[k + 1] - 1)
            acc += abs(L.nzval[p])
        end
        s[k] = acc
    end
    colsum = zeros(Float64, n)
    @inbounds for k in 1:n
        sk = s[k]
        for p in L.colptr[k]:(L.colptr[k + 1] - 1)
            colsum[L.rowval[p]] += abs(L.nzval[p]) * sk
        end
    end
    nrm1 = maximum(colsum)
    dmax = maximum(abs.(diag(Ah)))
    # outward rounding: inflate the perturbation estimate by a few ulps
    pert = gam * nrm1 * (1 + 8u) + u * dmax * (1 + 8u)
    bound = sigma - pert
    return bound > 0, bound
end

# ---------------------------------------------------------------------------
#  Approximate lambda_min (to place the shift; not part of any certificate)
# ---------------------------------------------------------------------------

"""
    lambda_min_estimate(A; F = nothing, iters = 60, tol = 1e-10) -> Float64

Inverse power iteration on a sparse Cholesky factorization: the Rayleigh
quotient of the dominant eigenvector of `A^{-1}` approximates
`lambda_min(A)`.  Purely a heuristic used to place the trial shift; every
returned bound is certified independently.

`x0` sets the starting vector. The default `nothing` uses `randn`, so the
returned estimate — and hence the trial shift, and hence the exact value of the
rigorous bound `lambda_min_lower` ends up certifying — varies slightly between
runs. Pass an explicit `x0` (the unit tests pass `ones(n)`) for a reproducible
bound. Rigour never depends on this: the estimate only places the shift, and
every bound is certified independently of how it was placed.
"""
function lambda_min_estimate(A::SparseMatrixCSC{Float64,Int}; F = nothing,
                             iters::Integer = 60, tol::Real = 1e-10,
                             x0::Union{Nothing,AbstractVector} = nothing)
    Fc = F === nothing ? cholesky(A) : F
    n = size(A, 1)
    x = x0 === nothing ? randn(n) : collect(float.(x0))
    length(x) == n ||
        throw(DimensionMismatch("x0 must have length $n, got $(length(x))"))
    x ./= norm(x)
    lam = Inf
    for _ in 1:iters
        y = Fc \ x
        ny = norm(y)
        ny == 0 && break
        x = y ./ ny
        lnew = dot(x, A * x)
        abs(lnew - lam) <= tol * abs(lnew) && (lam = lnew; break)
        lam = lnew
    end
    return lam
end

# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

"""
    lambda_min_lower(A; method = :auto, F = nothing, safety = 0.9,
                     backoff = 0.5, max_tries = 24, refine = 0,
                     verbose = false)
        -> (lmin, diagnostics::NamedTuple)

Rigorous lower bound `lmin <= lambda_min(A)` for a sparse symmetric positive
definite `A`.

Strategy: estimate `lambda_min` by inverse iteration, then attempt to certify
the shift `sigma = safety * estimate`, multiplying `sigma` by `backoff` on
failure until a certificate succeeds.  Optionally `refine` bisection steps
tighten the bound between the last failure and the first success (only cosmetic
here -- see the note in the file header on why looseness is harmless).

Arguments
* `method`    : `:interval` (interval Cholesky), `:rump` (backward-error bound),
                or `:auto` (try `:interval`, fall back to `:rump`)
* `safety`    : first shift as a fraction of the estimate
* `backoff`   : shrink factor applied to `sigma` after a failed attempt
* `max_tries` : maximum number of shifts tried
* `refine`    : number of bisection steps after the first success
* `x0`        : starting vector for the inverse-iteration estimate; `nothing`
                (default) uses `randn`, so the certified value varies slightly
                between runs. Pass e.g. `ones(size(A, 1))` for reproducibility.
                Rigour is unaffected either way — see
                [`lambda_min_estimate`](@ref).

Returns `lmin` (or `0.0`, itself a valid but useless bound, if nothing could be
certified) and a NamedTuple `(method, sigma, estimate, tries, ratio, t_total,
minpivot, nnzL)` where `ratio = lmin / estimate` measures the sharpness.
"""
function lambda_min_lower(A::SparseMatrixCSC{Float64,Int};
                          method::Symbol = :auto, F = nothing,
                          safety::Real = 0.9, backoff::Real = 0.5,
                          max_tries::Integer = 24, refine::Integer = 0,
                          verbose::Bool = false,
                          x0::Union{Nothing,AbstractVector} = nothing)
    t0 = time()
    Fc = F === nothing ? cholesky(A) : F
    perm = Fc.p
    est = lambda_min_estimate(A; F = Fc, x0 = x0)
    Ap = A[perm, perm]
    Lp, Li = symbolic_cholesky(Ap)
    nnzL = length(Li)

    function certify(sigma)
        if method === :rump
            ok, b = verify_pd_rump(A, sigma; perm = perm)
            return ok, b, -Inf
        end
        ok, mp = verify_pd_interval(Ap, sigma; Lp = Lp, Li = Li)
        if ok
            return true, Float64(sigma), mp
        elseif method === :auto
            ok2, b2 = verify_pd_rump(A, sigma; perm = perm)
            return ok2, b2, mp
        end
        return false, -Inf, mp
    end

    sigma = max(safety * est, 0.0)
    lmin = 0.0
    tries = 0
    mp = -Inf
    lastfail = 0.0
    while tries < max_tries && sigma > 0
        tries += 1
        ok, b, mpi = certify(sigma)
        verbose && println("  sigma = $sigma -> ", ok ? "certified $b" : "failed")
        if ok
            lmin = max(lmin, b)
            break
        end
        mp = mpi
        lastfail = sigma
        sigma *= backoff
    end
    # optional bisection between the successful sigma and the last failure
    if lmin > 0 && refine > 0 && lastfail > sigma
        lo, hi = sigma, lastfail
        for _ in 1:refine
            mid = (lo + hi) / 2
            ok, b, _ = certify(mid)
            if ok
                lo = mid
                lmin = max(lmin, b)
            else
                hi = mid
            end
        end
    end
    return lmin, (method = method, sigma = sigma, estimate = est,
                  tries = tries, ratio = est > 0 ? lmin / est : NaN,
                  t_total = time() - t0, minpivot = mp, nnzL = nnzL)
end

# ---------------------------------------------------------------------------
#  lambda_hb_certified.jl -- certified two-sided enclosure of
#
#        lambda_{h,B} = min { x'A x : ||B x||_inf >= 1 } = 1 / max_i g_i,
#        g_i = beta_i' A^{-1} beta_i,   beta_i' = row i of B,
#
#  with NO interval matrix inverse anywhere.
#
#  Provenance
#     Certified replacement for Lemma 3.2 (\label{lem:opti_problem_lower_bound})
#     of J. Galindo, K. Ike, X. Liu (manuscript on L^infty Lagrange
#     interpolation error constants).  The authors' verified route needs an
#     interval dense inverse of A and then the full N x N interval matrix
#     D = B A^{-1} B'.  Here every certified number comes from interval
#     evaluation of two variational identities on float trial vectors.
#
#  The two identities (proved in `inverse_free_lemma.md`)
#     For any z and any beta, with r := beta - A z and g := beta' A^{-1} beta:
#        (L)  g >= 2 beta'z - z'A z                                (any z)
#        (U)  g  = z'A z + 2 z'r + r' A^{-1} r
#                <= z'A z + 2 z'r + ||r||_2^2 / lmin,     lmin <= lambda_min(A).
#     (L) needs no spectral information at all; (U) degrades only
#     quadratically in ||r||, so a loose `lmin` costs essentially nothing once
#     z is a converged solve.  Both are evaluated in interval arithmetic on the
#     ORIGINAL A and beta, so the float factorization is used only as an oracle
#     producing candidate vectors -- it never enters the certificate.
#
#  Screening
#     Since max_i g_i >= g_{i*} >= Glo for the single row i*, any row with
#         g_i <= ||beta_i||_2^2 / lmin  <  Glo
#     cannot attain the maximum and is discarded after O(nnz(beta_i)) work.
#     Only the survivors get the (more expensive) interval upper bound.
#
#  Conventions (VFEM.jl style)
#     * 1-based indices, SparseMatrixCSC for all global matrices.
#     * routines generic in the element type via `T::Type = Float64`; the
#       certified path instantiates the same expressions at
#       `T = Interval{Float64}`.
#     * plain .jl file, no module wrapper.  Requires `lambda_hb_fast.jl` and
#       `lambda_min_lower_bound.jl` to be included first.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Rigorous kernels (hand-written; see the note on `mul!` below)
# ---------------------------------------------------------------------------
#
#  Julia's generic `SparseMatrixCSC * Vector` for an `Interval{Float64}` element
#  type dispatches to a fallback that is ~200x slower than an explicit loop
#  (measured: 1.18 s vs 0.0059 s for nnz(A) = 94944).  Since one rigorous matrix
#  vector product per certified row is the dominant cost of the whole driver,
#  the products below are written out.  They also avoid materialising an
#  interval copy of `A`: every entry of `A` and of the trial vector `z` is a
#  Float64 and therefore exactly representable, so `interval(a)` is a degenerate
#  (zero-width) enclosure and all width in the result comes from the correctly
#  rounded arithmetic itself.
# ---------------------------------------------------------------------------

"""
    _lhb_imat(A)

Interval copy of a float sparse matrix.  Used only by the dense reference
baseline at the bottom of this file; the certified path works directly from `A`.
File-local: underscore-prefixed so it is not injected into the library namespace.
"""
_lhb_imat(A::SparseMatrixCSC{Float64,Int}) =
    SparseMatrixCSC{Interval{Float64},Int}(A.m, A.n, A.colptr, A.rowval,
                                           interval.(A.nzval))

_lhb_ivec(x::AbstractVector{Float64}) = interval.(x)

"""
    rigorous_Az(A, z) -> Vector{Interval{Float64}}

Enclosure of the exact product `A*z` for float sparse `A` and float `z`,
accumulated column by column in interval arithmetic.  Every rounding is
outward, so the result rigorously contains the exact real product.
"""
function rigorous_Az(A::SparseMatrixCSC{Float64,Int}, z::Vector{Float64})
    y = fill(interval(0.0), A.m)
    @inbounds for j in 1:A.n
        zj = z[j]
        zj == 0 && continue
        zji = interval(zj)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            y[i] += interval(A.nzval[p]) * zji
        end
    end
    return y
end

"""
    rigorous_dot(z, y) -> Interval{Float64}

Enclosure of `z' * y` for a float vector `z` and an interval vector `y`.
"""
function rigorous_dot(z::Vector{Float64}, y::Vector{Interval{Float64}})
    s = interval(0.0)
    @inbounds for k in eachindex(z)
        zk = z[k]
        zk == 0 && continue
        s += interval(zk) * y[k]
    end
    return s
end

"""
    rigorous_sumsq(r) -> Interval{Float64}

Enclosure of `||r||_2^2` for an interval vector `r`.
"""
function rigorous_sumsq(r::Vector{Interval{Float64}})
    s = interval(0.0)
    @inbounds for k in eachindex(r)
        rk = r[k]
        s += rk * rk
    end
    return s
end

# ---------------------------------------------------------------------------
#  Certified one-row bounds
# ---------------------------------------------------------------------------

"""
    certified_g_lower(A, bidx, bval, z) -> Interval

Rigorous lower bound for `g = beta' A^{-1} beta` from identity (L),

    g - (2 beta'z - z'A z) = (z - A^{-1}beta)' A (z - A^{-1}beta) >= 0,

evaluated in interval arithmetic.  Valid for **every** trial vector `z`; no
information whatsoever about the spectrum of `A` is used.  `beta` is given by
its nonzero positions `bidx` and values `bval` (at most 6 of them for the
Fujino-Morley `B`), `z` is a float trial vector.
"""
function certified_g_lower(A::SparseMatrixCSC{Float64,Int}, bidx, bval,
                           z::Vector{Float64})
    Az = rigorous_Az(A, z)
    bz = interval(0.0)
    @inbounds for k in eachindex(bidx)
        bz += interval(bval[k]) * interval(z[bidx[k]])
    end
    return interval(2.0) * bz - rigorous_dot(z, Az)
end

"""
    certified_g_upper(A, bidx, bval, z, lmin) -> (Interval, Float64)

Rigorous upper bound for `g = beta' A^{-1} beta` from identity (U),

    g = z'A z + 2 z'r + r' A^{-1} r,   r := beta - A z,
    r' A^{-1} r <= ||r||_2^2 / lambda_min(A) <= ||r||_2^2 / lmin,

evaluated in interval arithmetic.  `lmin > 0` must be a rigorous lower bound of
`lambda_min(A)` (see `lambda_min_lower_bound.jl`).  Returns the interval bound
and `sup(||r||_2)`, the diagnostic that decides whether further refinement of
`z` would pay.
"""
function certified_g_upper(A::SparseMatrixCSC{Float64,Int}, bidx, bval,
                           z::Vector{Float64}, lmin::Float64)
    lmin > 0 || error("lmin must be a positive rigorous lower bound")
    Az = rigorous_Az(A, z)
    r = -Az                                  # r = beta - A z, beta sparse
    @inbounds for k in eachindex(bidx)
        r[bidx[k]] += interval(bval[k])
    end
    zAz = rigorous_dot(z, Az)
    zr = rigorous_dot(z, r)
    r2 = rigorous_sumsq(r)
    tail = r2 / interval(lmin)
    return zAz + interval(2.0) * zr + tail, sqrt(sup(r2))
end

"""
    refine_solve(A, F, b; z0 = nothing, steps = 2) -> z

Solve `A z = b` with the sparse Cholesky factorization `F` plus `steps` rounds
of iterative refinement, so that the residual `b - A z` reaches the level of
the working precision.  Because the certified upper bound depends on the
residual only through `||r||^2`, driving `||r||` down to ~1e-16 makes the
`lmin` term utterly negligible.
"""
function refine_solve(A::SparseMatrixCSC{Float64,Int}, F, b::Vector{Float64};
                      z0 = nothing, steps::Integer = 2)
    z = z0 === nothing ? F \ b : copy(z0)
    for _ in 1:steps
        r = b - A * z
        z += F \ r
    end
    return z
end

# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

"""
    lambda_hb_certified(A, B; lmin = nothing, lmin_method = :auto,
                        refine_steps = 2, screen = true, method = :trisolve,
                        verbose = false)
        -> (enclosure::Interval, diagnostics::NamedTuple)

Certified interval enclosure of

    lambda_{h,B} = min { x'A x : ||B x||_inf >= 1 } = 1 / max_i g_i .

Pipeline
 1. **float path** -- `lambda_hb_fast(A, B)` gives every `g_i`, the argmax `i*`
    and the value `g*` (no certificate, just an oracle);
 2. **certified lower bound on the max** -- identity (L) at `i*` in interval
    arithmetic gives `Glo <= g_{i*} <= max_i g_i`.  One row suffices, because
    the maximum dominates any individual entry;
 3. **screening** -- rigorous `g_i <= ||beta_i||_2^2 / lmin`; every row whose
    screen value is below `Glo` provably cannot attain the maximum and is
    discarded without a solve;
 4. **certified upper bound on the max** -- for the survivors only, identity (U)
    with `z` refined until `||r||` is near machine epsilon; `Gup` is the maximum
    of the row-wise upper bounds;
 5. **inversion** -- `lambda in [1/Gup, 1/Glo]`, computed with interval division
    so the rounding direction is automatic: the *lower* bound of `lambda` comes
    from the *upper* bound of the max.

Arguments
* `A :: SparseMatrixCSC{Float64}` : `M x M` symmetric positive definite
* `B :: SparseMatrixCSC{Float64}` : `N x M`, few nonzeros per row
* `lmin`         : rigorous lower bound of `lambda_min(A)`; computed by
                   `lambda_min_lower` when `nothing`
* `lmin_method`  : passed to `lambda_min_lower` (`:auto`, `:interval`, `:rump`)
* `refine_steps` : iterative-refinement rounds per survivor row
* `screen`       : enable step 3 (set `false` to certify all `N` rows, for
                   measuring what screening saves)

Returns
* `enclosure` : `Interval{Float64}` containing `lambda_{h,B}`
* `diagnostics` : `(lambda_float, imax, Glo, Gup, lmin, n_survivors, N,
                   survivor_frac, width, rel_width, max_resid, t_float,
                   t_lmin, t_lower, t_screen, t_upper, t_total, lmin_diag)`
"""
function lambda_hb_certified(A::SparseMatrixCSC{Float64,Int},
                             B::SparseMatrixCSC{Float64,Int};
                             lmin = nothing, lmin_method::Symbol = :auto,
                             refine_steps::Integer = 2, screen::Bool = true,
                             method::Symbol = :trisolve, verbose::Bool = false)
    t_start = time()
    M = size(A, 1); N = size(B, 1)
    @assert size(B, 2) == M "B has $(size(B,2)) columns, A is $M x $M"

    # ---- 1. float path -------------------------------------------------
    t0 = time()
    F = cholesky(A)
    lam_f, imax, df = lambda_hb_fast(A, B; method = method, F = F)
    t_float = time() - t0
    verbose && println("float: lambda = $lam_f at row $imax  ($(round(t_float,digits=3)) s)")

    # ---- rigorous lambda_min(A) ---------------------------------------
    t0 = time()
    lminv, ldiag = lmin === nothing ?
        lambda_min_lower(A; method = lmin_method, F = F) :
        (Float64(lmin), (method = :given, ratio = NaN, tries = 0))
    t_lmin = time() - t0
    lminv > 0 || error("no positive rigorous lower bound for lambda_min(A)")
    verbose && println("lmin = $lminv  ($(round(t_lmin,digits=3)) s)")

    rowptr, colidx, nzval = csr_rows(B)

    # ---- 2. certified lower bound on max_i g_i, via row i* -------------
    t0 = time()
    b_star = zeros(Float64, M)
    for q in rowptr[imax]:(rowptr[imax + 1] - 1)
        b_star[colidx[q]] = nzval[q]
    end
    z_star = refine_solve(A, F, b_star; steps = refine_steps)
    rs = rowptr[imax]:(rowptr[imax + 1] - 1)
    Glo_iv = certified_g_lower(A, view(colidx, rs), view(nzval, rs), z_star)
    Glo = inf(Glo_iv)
    t_lower = time() - t0
    Glo > 0 || error("certified lower bound on max_i g_i is not positive")
    verbose && println("Glo = $Glo  ($(round(t_lower,digits=3)) s)")

    # ---- 3. screening --------------------------------------------------
    t0 = time()
    survivors = Int[]
    if screen
        for i in 1:N
            s = 0.0
            for q in rowptr[i]:(rowptr[i + 1] - 1)
                v = nzval[q]
                s = muladd(v, v, s)
            end
            # rigorous, outward-rounded: g_i <= ||beta_i||^2 / lmin
            if sup(interval(s) / interval(lminv)) >= Glo
                push!(survivors, i)
            end
        end
    else
        survivors = collect(1:N)
    end
    imax in survivors || push!(survivors, imax)
    t_screen = time() - t0
    verbose && println("screening: $(length(survivors))/$N survive  ($(round(t_screen,digits=3)) s)")

    # ---- 4. certified upper bound over the survivors -------------------
    t0 = time()
    Gup = -Inf
    max_resid = 0.0
    i_up = imax
    bbuf = zeros(Float64, M)
    for i in survivors
        fill!(bbuf, 0.0)
        for q in rowptr[i]:(rowptr[i + 1] - 1)
            bbuf[colidx[q]] = nzval[q]
        end
        z = refine_solve(A, F, bbuf; steps = refine_steps)
        ri = rowptr[i]:(rowptr[i + 1] - 1)
        ub, rn = certified_g_upper(A, view(colidx, ri), view(nzval, ri), z, lminv)
        max_resid = max(max_resid, rn)
        s = sup(ub)
        if s > Gup
            Gup = s
            i_up = i
        end
    end
    t_upper = time() - t0
    Gup >= Glo || error("inconsistent certificate: Gup = $Gup < Glo = $Glo")

    # ---- 5. invert, with interval division fixing the rounding ---------
    encl = interval(1.0) / interval(Glo, Gup)
    w = diam(encl)
    return encl, (lambda_float = lam_f, imax = imax, imax_upper = i_up,
                  Glo = Glo, Gup = Gup, lmin = lminv,
                  n_survivors = length(survivors), N = N,
                  survivor_frac = length(survivors) / N,
                  width = w, rel_width = w / abs(mid(encl)),
                  max_resid = max_resid, t_float = t_float, t_lmin = t_lmin,
                  t_lower = t_lower, t_screen = t_screen, t_upper = t_upper,
                  t_total = time() - t_start, lmin_diag = ldiag,
                  g = df.g, M = M)
end

# ---------------------------------------------------------------------------
#  Reference certified baseline (dense interval inverse) -- benchmarking only
# ---------------------------------------------------------------------------

"""
    lambda_hb_certified_denseinv(A, B; maxM = 900) -> (enclosure, diagnostics)

The route this file replaces: a dense **interval** inverse of `A` obtained by
Rump's verified-inverse residual iteration

    inv(A) in R + R*E + R*E^2*inv(I - E),   E := I - A*R,   ||E||_inf < 1,

followed by the interval quadratic forms `beta_i' A^{-1} beta_i`.  `O(M^3)`
interval operations and `O(M^2)` interval storage; restricted to `M <= maxM`
so that it stays runnable as a cross-check.
"""
function lambda_hb_certified_denseinv(A::SparseMatrixCSC{Float64,Int},
                                      B::SparseMatrixCSC{Float64,Int};
                                      maxM::Integer = 900)
    M = size(A, 1); N = size(B, 1)
    M <= maxM || error("dense interval inverse restricted to M <= $maxM (got $M)")
    t0 = time()
    Ad = Matrix(A)
    R = inv(Ad)
    Aiv = interval.(Ad); Riv = interval.(R)
    E = interval.(Matrix(1.0I, M, M)) - Aiv * Riv
    nE = maximum(sum(mag.(E); dims = 2))
    nE < 1 || error("verified inverse failed: ||I - A*R||_inf = $nE >= 1")
    # Neumann tail: inv(A) - R - R*E  in  R*E^2*inv(I - E), bounded entrywise
    RE = Riv * E
    tailmag = (maximum(sum(mag.(RE); dims = 2)) * nE) / (1 - nE)
    Ainv = Riv + RE .+ interval(-tailmag, tailmag)
    rowptr, colidx, nzval = csr_rows(B)
    Gup = -Inf; Glo = -Inf; imax = 1
    for i in 1:N
        rng = rowptr[i]:(rowptr[i + 1] - 1)
        s = interval(0.0)
        for q in rng, r in rng
            s += interval(nzval[q]) * interval(nzval[r]) * Ainv[colidx[q], colidx[r]]
        end
        if sup(s) > Gup
            Gup = sup(s); imax = i
        end
        Glo = max(Glo, inf(s))
    end
    encl = interval(1.0) / interval(Glo, Gup)
    return encl, (t_total = time() - t0, Glo = Glo, Gup = Gup, imax = imax,
                  normE = nE, method = :dense_interval_inverse,
                  mem_bytes = 32 * M^2)
end

# ---------------------------------------------------------------------------
#  screen_sharp.jl -- a sharper rigorous screen for  max_i beta_i' A^{-1} beta_i
#
#  WHY
#     The screen in `lambda_hb_certified.jl` is
#         g_i = beta_i' A^{-1} beta_i <= ||beta_i||_2^2 / lambda_min(A),
#     which is tight only when `beta_i` is aligned with the lowest eigenvector
#     of `A`.  On the Fujino-Morley matrices it is not: `beta_i` has only 6
#     nonzeros and is therefore a highly localised vector with almost no overlap
#     with the smooth lowest eigenmode.  MEASURED at n = 64, theta = pi/2,
#     variant = :verbatim:
#         lmin (rigorous)          = 7.597382e-03
#         g_max                    = 1.729736e-01
#         ||beta_i||_2^2  median   = 1.0,  max = 1.5
#         crude bound  1/lmin      = 1.3e+02   -- overshoots g_max by ~760x
#     so the screen rejects 3 of 24576 rows (and those 3 only because their
#     rows are empty: all six of their columns are deleted corner DOFs).
#     The screen is ineffective HERE; that is a property of this problem, not a
#     bug in the certified driver, and it is reported as such.
#
#  THE SHARPER BOUND
#     `g_i` depends only on the 6x6 submatrix of `A^{-1}` on the support
#     `S_i = supp(beta_i)`.  For a positive semidefinite `M`,
#         beta' M beta = || M^{1/2} beta ||_2^2
#                      <= ( sum_{j in S} |beta_j| * || M^{1/2} e_j || )^2
#                       = ( sum_{j in S} |beta_j| * sqrt(M_jj) )^2
#     by the triangle inequality.  With `M = A^{-1}` this gives
#
#         g_i <= ( sum_{j in S_i} |beta_ij| * sqrt( (A^{-1})_jj ) )^2 ,        (S)
#
#     which uses the ACTUAL diagonal of `A^{-1}` on the row's own support
#     instead of the global spectral extreme.  Each `(A^{-1})_jj = e_j' A^{-1} e_j`
#     is itself of the certified form, so a rigorous UPPER bound for it comes
#     from the same identity (U) already used for the survivors, with
#     `beta = e_j`.  Cost: `M` solves once, then `O(nnz(beta_i))` per row --
#     versus `N` solves if every row must be certified.  Here `M = 8382` and
#     `N = 24576`, so the sharper screen pays for itself if it rejects more than
#     `M/N = 34%` of the rows.
#
#  Everything below is rigorous: all arithmetic on the bounds is interval
#  arithmetic with outward rounding.  Nothing is tuned.
# ---------------------------------------------------------------------------


"""
    certified_diag_Ainv_upper(A, F, lmin; refine_steps=2, verbose=false)
        -> (d_up, t)

Rigorous elementwise UPPER bounds `d_up[j] >= (A^{-1})_jj`, from identity (U)
of `lambda_hb_certified.jl` applied to `beta = e_j`:

    (A^{-1})_jj <= z'A z + 2 z'r + ||r||_2^2 / lmin,   r := e_j - A z,

with `z` a refined float solve of `A z = e_j`.  `lmin > 0` must be a rigorous
lower bound of `lambda_min(A)`.  Cost `M` sparse solves plus `M` rigorous
matrix-vector products.
"""
function certified_diag_Ainv_upper(A::SparseMatrixCSC{Float64,Int}, F,
                                   lmin::Float64; refine_steps::Integer = 2,
                                   verbose::Bool = false)
    lmin > 0 || error("lmin must be a positive rigorous lower bound")
    M = size(A, 1)
    d_up = Vector{Float64}(undef, M)
    e = zeros(Float64, M)
    t0 = time()
    idx1 = Vector{Int}(undef, 1)
    val1 = Vector{Float64}(undef, 1)
    for j in 1:M
        e[j] = 1.0
        z = refine_solve(A, F, e; steps = refine_steps)
        idx1[1] = j; val1[1] = 1.0
        ub, _ = certified_g_upper(A, idx1, val1, z, lmin)
        d_up[j] = sup(ub)
        e[j] = 0.0
        verbose && j % 2000 == 0 && println("  diag $j/$M")
    end
    return d_up, time() - t0
end

"""
    screen_sharp(B, d_up, Glo) -> (survivors, bound)

Apply the rigorous screen (S): row `i` survives iff

    ( sum_{j in S_i} |beta_ij| sqrt(d_up[j]) )^2  >=  Glo .

All arithmetic is interval arithmetic, so `bound[i]` is a rigorous upper bound
for `g_i` and a row is discarded only when it provably cannot attain the
maximum.  Returns the surviving row indices and the per-row bound.
"""
function screen_sharp(B::SparseMatrixCSC{Float64,Int}, d_up::Vector{Float64},
                      Glo::Float64)
    rowptr, colidx, nzval = csr_rows(B)
    N = size(B, 1)
    survivors = Int[]
    bound = Vector{Float64}(undef, N)
    sq = [sqrt(interval(d)) for d in d_up]      # outward-rounded sqrt
    for i in 1:N
        s = interval(0.0)
        for q in rowptr[i]:(rowptr[i + 1] - 1)
            s += abs(interval(nzval[q])) * sq[colidx[q]]
        end
        bi = sup(s * s)
        bound[i] = bi
        bi >= Glo && push!(survivors, i)
    end
    return survivors, bound
end

# ---------------------------------------------------------------------------
#  Corollary 3.1 -- from lambda_{h,B} to the interpolation constant C^L_ub
# ---------------------------------------------------------------------------

"""
    CL_ub(lambda, n::Integer) -> Float64

Corollary 3.1: with `h = 1/n`, `lambda >= lambda_{h,B}·(1 − h²)` and
`C^L_ub = lambda^{-1/2}`, i.e.

    C^L_ub = 1 / sqrt( lambda_{h,B} · (1 − (1/n)²) ).

Float version; use [`CL_ub_interval`](@ref) for the rigorous one.
"""
CL_ub(lambda, n::Integer) = 1 / sqrt(lambda * (1 - (1 / n)^2))

"""
    CL_ub_interval(lam_iv, n::Integer) -> Interval{Float64}

Rigorous version of [`CL_ub`](@ref): every operation is an interval operation
with outward rounding. `n` is exactly representable, so the only widening comes
from the division and the square root.

Applied to the certified enclosure of `lambda_{h,B}` at `theta = pi/2`,
`n = 64`, this reproduces `C^L_ub in [0.415951672629, 0.415951673250]` for the
`:verbatim` convention — 10.4x narrower than the published rigorous interval
`[0.4159516728, 0.4159516793]`, with the upper end (the direction in which the
lemma is used) 6.1e-9 sharper. The two intervals are NOT nested: our lower end
sits 1.7e-10 BELOW the published one. That discrepancy is unresolved and is
flagged for the authors rather than reconciled here; nothing is adjusted to
close it. The published endpoints are themselves quoted to 10 decimals, the same
order as the disagreement.
"""
function CL_ub_interval(lam_iv, n::Integer)
    one_iv = interval(1.0)
    ns = interval(Float64(n))
    fac = one_iv - one_iv / (ns * ns)
    return one_iv / sqrt(lam_iv * fac)
end
