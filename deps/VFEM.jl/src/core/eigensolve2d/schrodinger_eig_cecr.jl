# src/core/eigensolve2d/schrodinger_eig_cecr.jl
#
# Port of vfem2d/lib_eigenvalue_bound/schrodinger_eig_cecr.m.
#
# Pipeline:
#   1. Assemble CECR (A, M) for −Δ + V on the mesh.
#   2. Apply Dirichlet BC by removing boundary-edge DOFs.
#   3. Solve A_int u = λ M_int u for the smallest neig eigenvalues.
#   4. Apply Liu's lower bound (with shift if V can be negative).
#
# Liu's 2D constant: C_h = 0.1490 · h_max.
# Standard formula (V ≥ 0):           λ_lower = λ_h / (1 + λ_h · C_h²)
# Shift formula (V can be negative):  ν = λ_h + γ_h
#                                     λ_lower = ν / (1 + ν · C_h²) − γ_h
# with γ_h = max over elements of max(−V_K, 0) — element-centroid c.
#
# The MATLAB code stuffs `c_K = V at centroid` into the CECR reaction
# coefficient. This Julia port follows the same convention.

using Arpack: eigs
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: Symmetric

"""
    SchrodingerEig

Result struct returned by [`schrodinger_eig_cecr`](@ref):
* `eig_lower` :: `Vector{Float64}` — Liu lower bounds for the true eigenvalues.
* `eig_upper` :: `Vector{Float64}` — discrete CECR eigenvalues (which are
  upper bounds for the truncated problem under Dirichlet BC); for Neumann
  this field is left at zeros.
* `eig_h`     :: `Vector{Float64}` — raw CECR FE eigenvalues.
* `Ch`        :: `Float64` — Liu's constant on the mesh.
* `gamma_h`   :: `Float64` — max negative part of V on element centroids
  (zero when `V ≥ 0`).
"""
struct SchrodingerEig
    eig_lower::Vector{Float64}
    eig_upper::Vector{Float64}
    eig_h::Vector{Float64}
    Ch::Float64
    gamma_h::Float64
end

const _LIU_C2D = 0.1490   # Liu (Xie–Liu 2018) ECR interpolation constant in 2D.

"""
    schrodinger_eig_cecr(m::Mesh2D, V_func, neig::Integer;
                         bc::Symbol = :dirichlet) -> SchrodingerEig

Compute Liu lower bounds for the first `neig` eigenvalues of
`H = −Δ + V` on `m`. The potential `V_func(x, y) -> Real` is sampled
at element centroids for the CECR reaction coefficient and at degree-4
Bernstein control points for the per-element negativity bound.

`bc` ∈ {`:dirichlet`, `:neumann`}. Default is Dirichlet.

For Dirichlet BC, `eig_h` is also a numerically computed upper bound
for the truncated problem.

This is the Float64 path. The verified-mode wrapper is added in a
later phase together with the Lehmann–Goerisch RT auxiliary.
"""
function schrodinger_eig_cecr(m::Mesh2D, V_func, neig::Integer;
                              bc::Symbol = :dirichlet)
    bc in (:dirichlet, :neumann) ||
        throw(ArgumentError("bc must be :dirichlet or :neumann (got $bc)"))
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))

    # CECR reaction coefficient = V at element centroid.
    centroids_V = Vector{Float64}(undef, m.nt)
    @inbounds for k in 1:m.nt
        xc = (m.nodes[m.elements[k, 1], 1] + m.nodes[m.elements[k, 2], 1]
            + m.nodes[m.elements[k, 3], 1]) / 3
        yc = (m.nodes[m.elements[k, 1], 2] + m.nodes[m.elements[k, 2], 2]
            + m.nodes[m.elements[k, 3], 2]) / 3
        centroids_V[k] = Float64(V_func(xc, yc))
    end

    A, M, dof_map = create_matrix_cecr(m, centroids_V)

    int_dof = bc == :dirichlet ? interior_ecr_dofs(m, dof_map) : collect(1:(m.ne + m.nt))
    A_int = A[int_dof, int_dof]
    M_int = M[int_dof, int_dof]

    k_eff = min(neig, size(A_int, 1) - 1)
    # Smallest eigenvalues. Use `which = :SM` rather than `sigma = 0` —
    # Arpack.jl's shift-invert path returns the eigenvalues of the
    # shift-inverted operator instead of the originals (a known wrapping
    # quirk in the Julia Arpack binding), which gives wrong values here.
    # `which = :SM` selects the same indices we want without that issue.
    λ_arr, _ = eigs(A_int, M_int; nev = k_eff, which = :SM,
                    tol = 1e-10, maxiter = 500)
    eig_h = sort(real.(λ_arr))

    eig_upper = bc == :dirichlet ? copy(eig_h) : zeros(length(eig_h))

    Ch = _LIU_C2D * find_mesh_hmax(m.nodes, m.edges)
    Ch2 = Ch * Ch

    γ_h = 0.0
    @inbounds for k in 1:m.nt
        if -centroids_V[k] > γ_h
            γ_h = -centroids_V[k]
        end
    end

    eig_lower = if γ_h < 1e-14
        @. eig_h / (1 + eig_h * Ch2)
    else
        ν = eig_h .+ γ_h
        @. ν / (1 + ν * Ch2) - γ_h
    end

    return SchrodingerEig(eig_lower, eig_upper, eig_h, Ch, γ_h)
end
