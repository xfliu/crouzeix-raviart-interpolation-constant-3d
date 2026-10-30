# src/core/eigensolve2d/lg_lower_eig_bound_laplace.jl
#
# Lehmann–Goerisch sharpening of Laplace eigenvalue lower bounds on a
# 2D mesh. Port of `vfem2d/example_lower_eig_bound_laplace.m`.
#
# Pipeline (Float64 / approximation mode):
#   1. CR Laplace eigenvalue computation (BC removed) gives `neig + 1`
#      eigenvalues. Apply Liu's CR shift  μ_k / (1 + μ_k · Ch_CR²)
#      with Ch_CR = 0.1893 · h_max (Liu 2015).
#   2. Take ρ = CR_eig_low[neig + 1] as the LG shift parameter.
#   3. Conforming Lagrange CG eigenpairs (degree `lagrange_order`)
#      via `laplace_eig_lagrange`. The result `(LA_eig, LA_eigf, LA_A,
#      LA_M)` is in the monomial Lagrange basis.
#   4. Build the LG generalized eigenproblem on the projected matrices
#        A_proj = LA_eigf' · LA_A · LA_eigf,
#        M_proj = LA_eigf' · LA_M · LA_eigf,
#        A_lg   = rt_hdiv_problem(m, RT_order, LA_eigf)   # Goerisch w-w
#      with
#        AL = A_proj − ρ · M_proj,
#        BL = A_proj − 2ρ · M_proj + ρ² · A_lg.
#      Solve `μ = sort(eig(AL, BL))` (real-valued in this regime).
#   5. Apply the LG transform  λ_lower = ρ − ρ / (1 − μ)  on the
#      eigenvalues sorted in DESCENDING order; the result, ascending,
#      gives the sharpened lower bounds.

using LinearAlgebra: eigvals
using Arpack: eigs

const _LIU_C2D_CR = 0.1893    # Liu's CR interpolation constant in 2D.

"""
    LGLaplaceLowerBound

Result struct returned by [`lg_lower_eig_bound_laplace`](@ref):
* `eig_lower` :: `Vector{Float64}` — Lehmann–Goerisch lower bounds for
  the first `neig` Dirichlet Laplace eigenvalues, sorted ascending.
* `eig_upper` :: `Vector{Float64}` — Lagrange CG eigenvalues (Galerkin
  upper bounds), sorted ascending.
* `cr_eig_lower` :: `Vector{Float64}` — Liu CR lower bounds, sorted
  ascending. The `(neig + 1)`-th entry is the LG shift parameter ρ.
* `rho`       :: `Float64` — the LG shift parameter.
* `Ch_cr`     :: `Float64` — Liu's CR constant `0.1893 · h_max` used in
  step 1.
"""
struct LGLaplaceLowerBound
    eig_lower::Vector{Float64}
    eig_upper::Vector{Float64}
    cr_eig_lower::Vector{Float64}
    rho::Float64
    Ch_cr::Float64
end

# Step 1: CR eigenvalue + Liu shift. Returns the first `neig + 1`
# Liu-corrected lower bounds (ascending).
function _cr_liu_lower_bounds(m::Mesh2D, neig::Integer)
    A0, A1 = create_matrix_crouzeix_raviart(m)
    int_edge = setdiff(1:m.ne, m.bd_edge_ids)
    A0_int = A0[int_edge, int_edge]
    A1_int = A1[int_edge, int_edge]

    nev = min(neig + 1, size(A0_int, 1) - 1)
    λ_arr, _ = eigs(A1_int, A0_int; nev = nev, which = :SM,
                    tol = 1e-10, maxiter = 500)
    cr_eig = sort(real.(λ_arr))

    hmax = find_mesh_hmax(m.nodes, m.edges)
    Ch_cr = _LIU_C2D_CR * hmax
    cr_eig_low = @. cr_eig / (1 + cr_eig * Ch_cr^2)
    return cr_eig_low, Ch_cr
end

"""
    lg_lower_eig_bound_laplace(m::Mesh2D, lagrange_order::Integer,
                               neig::Integer; RT_order::Integer = lagrange_order)
        -> LGLaplaceLowerBound

Compute Lehmann–Goerisch lower bounds (Float64 path) for the first
`neig` Dirichlet Laplace eigenvalues on `m`.

The CR-based Liu shift gives the spectral parameter ρ; the conforming
Lagrange CG eigenpairs of the same operator project everything down
to a small dense generalized eigenproblem; and the Goerisch w-w
bilinear form computed by `rt_hdiv_problem` provides the third
matrix needed to close the LG eigenproblem.

`RT_order` defaults to `lagrange_order` (matching the MATLAB driver).

Returns `LGLaplaceLowerBound(eig_lower, eig_upper, cr_eig_lower, rho,
Ch_cr)`. `eig_lower` and `eig_upper` form the validated lower / upper
bound pair on a per-eigenvalue basis (entry k is for the k-th
eigenvalue).

This is the Float64 path. The verified-mode wrapper that delegates the
LG generalized eigenproblem to Veigs.jl (with interval LDL and
cluster handling) lives in a follow-up phase.
"""
function lg_lower_eig_bound_laplace(m::Mesh2D, lagrange_order::Integer,
                                    neig::Integer;
                                    RT_order::Integer = lagrange_order)
    lagrange_order ≥ 1 ||
        throw(ArgumentError("Lagrange order must be ≥ 1 (got $lagrange_order)"))
    RT_order ≥ 0 ||
        throw(DomainError(RT_order, "RT_order must be ≥ 0"))
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))

    cr_eig_low, Ch_cr = _cr_liu_lower_bounds(m, neig)
    length(cr_eig_low) ≥ neig + 1 ||
        error("CR computation returned only $(length(cr_eig_low)) eigenvalues, " *
              "need at least $(neig + 1) for the LG shift")

    r_cg = laplace_eig_lagrange(m, lagrange_order, neig)
    eig_upper = r_cg.eig_value
    LA_eigf  = r_cg.eig_func
    LA_A     = r_cg.A
    LA_M     = r_cg.M

    A_proj = LA_eigf' * LA_A * LA_eigf
    M_proj = LA_eigf' * LA_M * LA_eigf
    A_lg   = rt_hdiv_problem(m, RT_order, LA_eigf)

    rho = cr_eig_low[neig + 1]
    AL  = A_proj .- rho .* M_proj
    BL  = A_proj .- 2 * rho .* M_proj .+ rho^2 .* A_lg

    # Symmetrize numerically before generalized eig (rounding can
    # introduce a 1e-14 skew that confuses some solvers).
    AL_sym = (AL + AL') / 2
    BL_sym = (BL + BL') / 2
    μ_arr = sort(real.(eigvals(AL_sym, BL_sym)))

    # LG transform: λ_lower = ρ − ρ / (1 − μ) on the eigenvalues sorted
    # in descending order. After the map, sort ascending again.
    eig_lower = similar(μ_arr)
    @inbounds for k in eachindex(μ_arr)
        eig_lower[k] = rho - rho / (1 - μ_arr[end - k + 1])
    end

    return LGLaplaceLowerBound(eig_lower, eig_upper, cr_eig_low, rho, Ch_cr)
end
