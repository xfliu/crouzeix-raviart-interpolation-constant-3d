# src/core/eigensolve2d/sparse_chol_pattern.jl
#
# Symbolic Cholesky analysis and a left-looking numeric Cholesky that is generic
# in the element type, both operating on an explicitly prescribed sparsity
# pattern.
#
# Provenance: support code for the inverse-free replacement of Lemma 3.2 in
# J. Galindo, K. Ike, X. Liu (manuscript on L^inf Lagrange interpolation error
# constants). Used by `lambda_h_bernstein.jl`; nothing here is specific to the
# Fujino-Morley element, so it is a sibling file rather than part of it.
#
# WHY AN EXPLICIT SYMBOLIC PATTERN IS NEEDED
#   CHOLMOD returns `sparse(F.L)` with numerically zero entries PRUNED, so its
#   pattern is in general a strict subset of the symbolic fill pattern. Two
#   consumers break on a pruned pattern:
#     * selected inversion needs `I_j = {i > j : L[i,j] != 0}` to be a CLIQUE of
#       the lower pattern (the classical fill-path property), which holds for the
#       symbolic pattern but not for a pruned one;
#     * the interval Cholesky must allocate storage for every position that
#       COULD become nonzero for a nearby matrix.
#   Both therefore work on the pattern produced by `symbolic_cholesky`.
#
# Conventions: 1-based indices, `SparseMatrixCSC`, and the element type selected
# by the keyword `T::Type = Float64` -- passing `T = Interval{Float64}` turns
# `sparse_chol_shift` itself into a rigorous positive-definiteness test, which is
# this library's core idiom for switching between approximate and verified mode.

# ---------------------------------------------------------------------------
#  sparse_symbolic.jl -- symbolic Cholesky analysis shared by the fast path
#                        (`lambda_hb_fast.jl`) and the verified positive
#                        definiteness test (`lambda_min_lower_bound.jl`).
#
#  Provenance
#     Support code for the inverse-free replacement of Lemma 3.2 in
#     J. Galindo, K. Ike, X. Liu (manuscript on L^infty Lagrange interpolation
#     error constants).
#
#  Why the EXACT symbolic pattern matters here
#     CHOLMOD returns `sparse(F.L)` with numerically zero entries pruned, so its
#     pattern is in general a strict SUBSET of the symbolic fill pattern.  Two
#     consumers break on the pruned pattern:
#       * selected inversion needs `I_j = {i > j : L[i,j] != 0}` to be a CLIQUE
#         of the lower pattern (the classical fill-path property), which holds
#         for the symbolic pattern but not for a pruned one;
#       * the interval Cholesky must allocate storage for every position that
#         *could* become nonzero for a nearby matrix.
#     Both therefore work on the pattern produced here.
#
#  Conventions (VFEM.jl style)
#     * 1-based indices, SparseMatrixCSC, plain .jl file (no module wrapper).
# ---------------------------------------------------------------------------


"""
    etree_sym(A) -> parent

Elimination tree of the symmetric sparsity pattern of `A`, computed from the
strict upper triangle with path compression.  `parent[k] == 0` marks a root.
"""
function etree_sym(A::SparseMatrixCSC)
    n = A.n
    parent = zeros(Int, n)
    ancestor = zeros(Int, n)
    @inbounds for k in 1:n
        for p in A.colptr[k]:(A.colptr[k + 1] - 1)
            i = A.rowval[p]
            i >= k && continue
            while i != 0 && i < k
                inext = ancestor[i]
                ancestor[i] = k
                inext == 0 && (parent[i] = k)
                i = inext
            end
        end
    end
    return parent
end

"""
    symbolic_cholesky(A) -> (Lp, Li)

Exact nonzero pattern of the Cholesky factor `L` of the symmetric matrix `A`
(`A = L L'`), in CSC form with sorted row indices and an explicit diagonal in
the first position of every column.

The pattern of row `k` is obtained by the row-subtree traversal (`ereach`): for
every `j < k` with `A[k,j] != 0`, walk `j, parent[j], ...` up the elimination
tree until a node already marked for this row is met.  The result depends only
on the pattern of `A`, hence is valid for `A - sigma*I` and for any interval
matrix with that pattern.
"""
function symbolic_cholesky(A::SparseMatrixCSC)
    n = A.n
    parent = etree_sym(A)
    marked = zeros(Int, n)
    path = Vector{Int}(undef, n)
    rowlists = Vector{Vector{Int}}(undef, n)
    colcount = zeros(Int, n)
    @inbounds for k in 1:n
        marked[k] = k
        rk = Int[]
        for p in A.colptr[k]:(A.colptr[k + 1] - 1)
            i = A.rowval[p]
            i >= k && continue
            len = 0
            while marked[i] != k
                len += 1
                path[len] = i
                marked[i] = k
                i = parent[i]
                i == 0 && break
            end
            for q in 1:len
                push!(rk, path[q])
            end
        end
        rowlists[k] = rk
        colcount[k] += 1                # the diagonal
        for j in rk
            colcount[j] += 1
        end
    end
    Lp = Vector{Int}(undef, n + 1)
    Lp[1] = 1
    @inbounds for j in 1:n
        Lp[j + 1] = Lp[j] + colcount[j]
    end
    Li = Vector{Int}(undef, Lp[n + 1] - 1)
    fill_ptr = copy(Lp)
    @inbounds for k in 1:n
        Li[fill_ptr[k]] = k             # diagonal first
        fill_ptr[k] += 1
    end
    # visiting k in increasing order keeps every column's row indices sorted
    @inbounds for k in 1:n
        for j in rowlists[k]
            Li[fill_ptr[j]] = k
            fill_ptr[j] += 1
        end
    end
    return Lp, Li
end

"""
    find_in_column(Lp, Li, j, i) -> Int

Position of the entry `(i, j)` within column `j` of the pattern `(Lp, Li)`, by
binary search on the sorted row indices; `0` if absent.
"""
@inline function find_in_column(Lp::Vector{Int}, Li::Vector{Int}, j::Int, i::Int)
    lo = Lp[j]; hi = Lp[j + 1] - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >> 1
        r = Li[mid]
        if r == i
            return mid
        elseif r < i
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0
end

# ---------------------------------------------------------------------------
#  sparse_chol_generic.jl -- left-looking sparse Cholesky, generic in the
#                            element type, on a prescribed symbolic pattern.
#
#  Provenance
#     Support code for the inverse-free replacement of Lemma 3.2 in
#     J. Galindo, K. Ike, X. Liu (manuscript on L^infty Lagrange interpolation
#     error constants).
#
#  Conventions (VFEM.jl style)
#     * 1-based indices, SparseMatrixCSC, plain .jl file (no module wrapper).
#     * the element type is selected by the keyword `T::Type = Float64`; passing
#       `T = Interval{Float64}` turns the very same routine into the verified
#       positive-definiteness test.  This is the library's core idiom for
#       switching between the fast approximate mode and the rigorous mode.
# ---------------------------------------------------------------------------


"""
    sparse_chol_shift(A, sigma, Lp, Li; T::Type = Float64)
        -> (ok, Lx, minpivot)

Left-looking sparse Cholesky of `A - sigma*I` on the given symbolic pattern,
carried out in the arithmetic of `T`.

With `T = Float64` this is an ordinary factorization and `ok` reports only that
no nonpositive pivot was met.  With `T = Interval{Float64}` every operation is
an interval operation, `A - sigma*I` is enclosed rigorously, and `ok == true`
is a **proof** that `A - sigma*I` is positive definite: all pivots are then
intervals with positive infimum, which by Alefeld's criterion certifies the
existence of a Cholesky factorization for every symmetric matrix in the hull.

The column-by-column update uses a linked list `head[j]` of the columns `k < j`
whose next unconsumed entry lies in row `j`, so the work is exactly the
Cholesky flop count of the pattern, not `O(n)` per column.

Arguments
* `A`        : `n x n` symmetric `SparseMatrixCSC` (full storage, both triangles)
* `sigma`    : real shift
* `Lp, Li`   : pattern from [`symbolic_cholesky`](@ref)
* `T`        : `Float64` (fast mode) or `Interval{Float64}` (verified mode)

Returns `ok::Bool`, the factor values `Lx::Vector{T}`, and the smallest pivot
encountered (`Float64` lower bound in verified mode).
"""
function sparse_chol_shift(A::SparseMatrixCSC{Float64,Int}, sigma::Real,
                           Lp::Vector{Int}, Li::Vector{Int};
                           T::Type = Float64)
    n = A.n
    nz = Lp[n + 1] - 1
    Lx = zeros(T, nz)
    w = zeros(T, n)
    head = zeros(Int, n)
    nextc = zeros(Int, n)
    ptr = zeros(Int, n)
    ivl = T === Float64 ? (x -> Float64(x)) : (x -> interval(x))
    sig = ivl(sigma)
    minpiv = Inf
    @inbounds for j in 1:n
        # scatter the lower part of column j of A, shifted on the diagonal
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            i < j && continue
            w[i] = i == j ? ivl(A.nzval[p]) - sig : ivl(A.nzval[p])
        end
        # left-looking updates from every column k < j with L[j,k] != 0
        k = head[j]
        head[j] = 0
        while k != 0
            knext = nextc[k]
            pk = ptr[k]
            ljk = Lx[pk]
            for p in pk:(Lp[k + 1] - 1)
                w[Li[p]] -= Lx[p] * ljk
            end
            ptr[k] = pk + 1
            if ptr[k] <= Lp[k + 1] - 1
                r = Li[ptr[k]]
                nextc[k] = head[r]
                head[r] = k
            end
            k = knext
        end
        d = w[j]
        dlo = T === Float64 ? Float64(d) : inf(d)
        minpiv = min(minpiv, dlo)
        if !(dlo > 0)
            return false, Lx, minpiv
        end
        sd = sqrt(d)
        Lx[Lp[j]] = sd
        for p in (Lp[j] + 1):(Lp[j + 1] - 1)
            Lx[p] = w[Li[p]] / sd
        end
        # clear the accumulator on this column's pattern
        for p in Lp[j]:(Lp[j + 1] - 1)
            w[Li[p]] = zero(T)
        end
        ptr[j] = Lp[j] + 1
        if ptr[j] <= Lp[j + 1] - 1
            r = Li[ptr[j]]
            nextc[j] = head[r]
            head[r] = j
        end
    end
    return true, Lx, minpiv
end
