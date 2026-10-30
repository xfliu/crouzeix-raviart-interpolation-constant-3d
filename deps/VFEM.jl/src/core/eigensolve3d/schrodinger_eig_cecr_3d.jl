# src/core/eigensolve3d/schrodinger_eig_cecr_3d.jl
#
# Port of VFEM3D/lib/eigensolve/schrodinger_eig_cecr_3d.m.
#
# Pipeline:
#   1. Sample the reaction coefficient V at each element centroid (or
#      accept a precomputed `c_data` from the caller — useful for
#      Coulomb where the average requires Duffy quadrature).
#   2. Assemble CECR (A, M) on the 3D mesh via `create_matrix_cecr_3d`.
#   3. Apply Dirichlet BC by removing boundary face DOFs (cell DOFs
#      always interior).
#   4. Solve `A_int u = λ M_int u` for the smallest `neig` eigenvalues
#      via Arpack `which = :SM` (the same shift-invert quirk noted in
#      the 2D driver applies here).
#   5. Apply Liu's lower bound: `λ_lower = ν / (1 + Ch² · ν)` with
#      optional shift `γ_h` for sign-changing potentials. The 3D ECR
#      Liu constant is `Ch = h_max / √40 ≈ 0.1581 · h_max` (per
#      `lib/eigensolve/schrodinger_eig_cecr_3d.m`; see docs/decisions.md).

using Arpack: eigs
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: eigen, Symmetric

# Liu's 3D ECR constant. MATLAB `lib/eigensolve/schrodinger_eig_cecr_3d.m`
# uses the literal `0.1581`, a 4-decimal truncation of `1/√40 ≈
# 0.158113883`. We match MATLAB exactly so the cross-validation
# fixtures in `test/fixtures/` compare bit-for-bit.
#
# CAVEAT — this direction of rounding is NOT conservative. `λ_lower =
# ν / (1 + Ch²·ν)` is *decreasing* in `Ch`, so the truncated (smaller)
# constant returns a slightly *larger* lower bound than `1/√40` would.
# The excess is O(1e-3) absolute on the shipped fixtures (relative size
# ~1.8e-4 · Ch²ν/(1+Ch²ν)) — below discretization error, but it is an
# overshoot of the certified Liu bound rather than a margin on it.
# For results that must be rigorous, use `1/sqrt(40)` rounded *up*, or
# carry `Ch` as an interval. Tracked in docs/decisions.md.
const _LIU_C3D_INV_SQRT = 0.1581
const _DENSE_EIG_THRESHOLD_3D = 1000          # Below this, use dense eigen
                                              # for robust handling of
                                              # degenerate clusters that
                                              # Arpack `which=:SM` misses.

"""
    SchrodingerEig3D

Result struct returned by [`schrodinger_eig_cecr_3d`](@ref):
* `eig_lower` :: `Vector{Float64}` — Liu lower bounds.
* `eig_upper` :: `Vector{Float64}` — discrete CECR eigenvalues
  (numerical upper bounds for the truncated Dirichlet problem).
* `eig_h`     :: `Vector{Float64}` — raw CECR FE eigenvalues.
* `Ch`        :: `Float64` — Liu's 3D constant on the mesh.
* `gamma_h`   :: `Float64` — shift for sign-changing V.
"""
struct SchrodingerEig3D
    eig_lower::Vector{Float64}
    eig_upper::Vector{Float64}
    eig_h::Vector{Float64}
    Ch::Float64
    gamma_h::Float64
end

# Sample V at the centroid of each tetrahedron.
function _centroid_V(m::Mesh3D, V_func)
    c = Vector{Float64}(undef, m.NumElt)
    @inbounds for e in 1:m.NumElt
        v1 = m.ElementList[e, 1]; v2 = m.ElementList[e, 2]
        v3 = m.ElementList[e, 3]; v4 = m.ElementList[e, 4]
        xc = (m.NodeList[v1, 1] + m.NodeList[v2, 1] +
              m.NodeList[v3, 1] + m.NodeList[v4, 1]) / 4
        yc = (m.NodeList[v1, 2] + m.NodeList[v2, 2] +
              m.NodeList[v3, 2] + m.NodeList[v4, 2]) / 4
        zc = (m.NodeList[v1, 3] + m.NodeList[v2, 3] +
              m.NodeList[v3, 3] + m.NodeList[v4, 3]) / 4
        c[e] = Float64(V_func(xc, yc, zc))
    end
    return c
end

# Boundary face DOFs: facets that have only one parent element
# (Facet2Element[:, 2] == 0). The MATLAB `Inner_Facet_Idx = find(sum(Facet2Element' > 0) == 2)`
# is equivalent to "both columns nonzero".
function _interior_facet_dofs(m::Mesh3D)
    inner = Int[]
    sizehint!(inner, m.NumF)
    @inbounds for f in 1:m.NumF
        if m.Facet2Element[f, 2] != 0
            push!(inner, f)
        end
    end
    return inner
end

"""
    schrodinger_eig_cecr_3d(m::Mesh3D, V_input, neig::Integer;
                            bc::Symbol = :dirichlet,
                            gamma_h_override::Union{Nothing, Real} = nothing)
        -> SchrodingerEig3D

3D CECR Schrödinger eigenvalue lower bounds for `H = −Δ + V` on the
tetrahedral mesh `m`. `V_input` is either:
* a callable `V_func(x, y, z)` evaluated at element centroids, or
* an `AbstractVector{<:Real}` of length `NumElt` (precomputed
  per-element reaction coefficient, e.g. Duffy-averaged Coulomb).

`bc ∈ {:dirichlet, :neumann}`. For Dirichlet BC, boundary face DOFs
are removed; cell DOFs are always interior.

`gamma_h_override` lets callers supply an analytically justified
shift instead of the (often loose) `max(−c_data, 0)` default.
"""
function schrodinger_eig_cecr_3d(m::Mesh3D, V_input, neig::Integer;
                                  bc::Symbol = :dirichlet,
                                  gamma_h_override::Union{Nothing, Real} = nothing)
    bc in (:dirichlet, :neumann) ||
        throw(ArgumentError("bc must be :dirichlet or :neumann (got $bc)"))
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))

    c_data = if V_input isa AbstractVector
        length(V_input) == m.NumElt ||
            throw(DimensionMismatch("V_input vector must have length NumElt = $(m.NumElt)"))
        Vector{Float64}(V_input)
    else
        _centroid_V(m, V_input)
    end

    A, M, info = create_matrix_cecr_3d(m, c_data)

    interior_dofs = if bc == :dirichlet
        vcat(_interior_facet_dofs(m), collect((info.NumF + 1):info.ndof))
    else
        collect(1:info.ndof)
    end

    A_int = A[interior_dofs, interior_dofs]
    M_int = M[interior_dofs, interior_dofs]

    n_int = size(A_int, 1)
    k_eff = min(neig, n_int)
    eig_h = if n_int ≤ _DENSE_EIG_THRESHOLD_3D
        # Dense path: robust on degenerate clusters that Arpack `:SM` misses.
        # Solve `A v = λ M v` densely via `eigen(::Symmetric, ::Symmetric)`.
        F = eigen(Symmetric(Matrix(A_int)), Symmetric(Matrix(M_int)))
        sort(real.(F.values))[1:k_eff]
    else
        λ_arr, _ = eigs(A_int, M_int; nev = k_eff, which = :SM,
                        tol = 1e-10, maxiter = 500)
        sort(real.(λ_arr))
    end

    eig_upper = bc == :dirichlet ? copy(eig_h) : zeros(length(eig_h))

    h_max = find_mesh_hmax_3d(m)
    Ch    = _LIU_C3D_INV_SQRT * h_max
    Ch2   = Ch * Ch

    γ_h = if gamma_h_override === nothing
        γ_default = 0.0
        @inbounds for e in 1:m.NumElt
            if -c_data[e] > γ_default
                γ_default = -c_data[e]
            end
        end
        γ_default
    else
        Float64(gamma_h_override)
    end

    eig_lower = if γ_h < 1e-14
        @. eig_h / (1 + eig_h * Ch2)
    else
        ν = eig_h .+ γ_h
        @. ν / (1 + ν * Ch2) - γ_h
    end

    return SchrodingerEig3D(eig_lower, eig_upper, eig_h, Ch, γ_h)
end
