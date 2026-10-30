# src/core/eigensolve2d/verified_lg_lower_eig.jl
#
# Verified-mode building blocks for the 2D Lehmann–Goerisch lower-bound
# pipeline. Pairs with `lg_lower_eig_bound_laplace.jl` (Float64 path).
#
# Two pieces, both stand-alone:
#
#   1. `verified_cr_liu_lower(m, neig)` — fully rigorous CR Liu lower
#      bound. Builds CR mass and stiffness as interval matrices via the
#      `T = Interval{Float64}` path of `create_matrix_crouzeix_raviart`,
#      restricts to interior DOFs, calls `Veigs.veigs(A1, A0, neig+1, :sm)`
#      for verified eigenvalues, and applies the Liu shift
#      `λ_low = λ / (1 + λ · Ch²)` in interval arithmetic.
#
#   2. `verified_lg_transform(AL, BL, rho)` — verified LG transform.
#      Symmetrizes the input interval matrices via `Veigs.sym_hull`,
#      calls `Veigs.veig(AL, BL)` for verified eigenvalues `μ`, then
#      applies `λ_low = ρ − ρ / (1 − μ)` in interval arithmetic and
#      sorts ascending. The caller is responsible for building `AL`,
#      `BL` rigorously.
#
# RT verified solve (the missing piece for a fully rigorous LG pipeline)
# is an open research item — Veigs.jl provides `interval_ldl` for
# symmetric (incl. indefinite) factorization, but no Krawczyk-style
# verified solver for arbitrary linear systems. Tracked in the README roadmap.

using LinearAlgebra: Symmetric
using IntervalArithmetic: Interval, interval, inf, sup, mid, hull
import Veigs

# Liu's CR interpolation constant (2D), reused from the Float64 path.
const _LIU_C2D_CR_INT = interval(0.1893)

"""
    verified_cr_liu_lower(m::Mesh2D, neig::Integer)
        -> (eig_lower::Vector{Interval{Float64}}, Ch_cr::Interval{Float64})

Verified Crouzeix–Raviart Liu lower bounds for the first `neig + 1`
Dirichlet Laplace eigenvalues on `m`. Returns the bounds together with
the Liu CR constant `Ch_cr = 0.1893 · h_max` as an interval enclosure.

The result is rigorous: every concrete eigenvalue of the true
continuous Dirichlet Laplacian lies above `inf(eig_lower[k])` for the
`k`-th eigenvalue.
"""
function verified_cr_liu_lower(m::Mesh2D, neig::Integer)
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))

    A0_int, A1_int = create_matrix_crouzeix_raviart(m; T = Interval{Float64})
    int_edge = setdiff(1:m.ne, m.bd_edge_ids)
    # Densify before calling Veigs.veigs — Julia 1.12 dropped the
    # `eigvals(Symmetric(sparse))` overload that veigs depends on.
    A0_int_red = Matrix(A0_int[int_edge, int_edge])
    A1_int_red = Matrix(A1_int[int_edge, int_edge])

    # Verified smallest-magnitude eigenvalues. The `:sm` legacy alias
    # selects the smallest by absolute value, identical to MATLAB
    # `eigs(.., 'sm')`. Eigenvalues are positive in this regime so this
    # also gives the smallest by ascending order.
    nev = min(neig + 1, size(A0_int_red, 1) - 1)
    cr_eig_int, _ = Veigs.veigs(A1_int_red, A0_int_red, nev, :sm)

    # find_mesh_hmax expects a Matrix{<:Real}; promote nodes to interval
    # for a rigorous h_max enclosure (max over interval edge lengths).
    nodes_int = interval.(m.nodes)
    hmax_int  = find_mesh_hmax(nodes_int, m.edges)
    Ch_cr     = _LIU_C2D_CR_INT * hmax_int

    Ch2 = Ch_cr * Ch_cr
    eig_lower = Vector{Interval{Float64}}(undef, length(cr_eig_int))
    @inbounds for k in eachindex(cr_eig_int)
        λ = cr_eig_int[k]
        eig_lower[k] = λ / (interval(1.0) + λ * Ch2)
    end
    # Sort ascending by lower bound — verified eigenvalues are
    # already sorted ascending up to clusters.
    perm = sortperm(eig_lower; by = inf)
    return eig_lower[perm], Ch_cr
end

"""
    verified_lg_transform(AL::AbstractMatrix, BL::AbstractMatrix,
                          rho::Union{Real, Interval}) -> Vector{Interval{Float64}}

Apply the Lehmann–Goerisch lower-bound transform on the small
generalized eigenproblem `AL u = μ BL u`, returning verified lower
bounds `λ_low = ρ − ρ / (1 − μ)` for the corresponding eigenvalues.

Inputs may be `Float64` or `Interval{Float64}` matrices; they are
auto-promoted to intervals if needed. Both matrices are symmetrized
via `Veigs.sym_hull` before the eigensolve. Returns intervals sorted
ascending by lower bound. The output length matches `size(AL, 1)`
(or grows by one or two if the verified eigensolver detects a cluster
that crosses the requested boundary).

The caller is responsible for constructing `AL`, `BL`, and `ρ`
rigorously. With Float64 inputs the result is *not* a true verified
bound — it just routes the same arithmetic through interval enclosures.
"""
function verified_lg_transform(AL::AbstractMatrix, BL::AbstractMatrix,
                                rho::Union{Real, Interval})
    size(AL) == size(BL) ||
        throw(DimensionMismatch("AL and BL must have the same shape"))
    size(AL, 1) == size(AL, 2) ||
        throw(DimensionMismatch("AL must be square"))

    AL_int = AL isa AbstractMatrix{<:Interval} ? AL : interval.(Matrix(AL))
    BL_int = BL isa AbstractMatrix{<:Interval} ? BL : interval.(Matrix(BL))
    AL_sym = Veigs.sym_hull(AL_int)
    BL_sym = Veigs.sym_hull(BL_int)

    μ_arr, _ = Veigs.veig(AL_sym, BL_sym)

    rho_int = rho isa Interval ? rho : interval(float(rho))
    one_int = interval(1.0)

    eig_lower = Vector{Interval{Float64}}(undef, length(μ_arr))
    @inbounds for k in eachindex(μ_arr)
        # Apply on the descending-ordered μ to mirror MATLAB
        # `LG_eig_low = rho - rho ./ (1 - LG_eig_low(end:-1:1))`.
        μ = μ_arr[end - k + 1]
        eig_lower[k] = rho_int - rho_int / (one_int - μ)
    end
    perm = sortperm(eig_lower; by = inf)
    return eig_lower[perm]
end
