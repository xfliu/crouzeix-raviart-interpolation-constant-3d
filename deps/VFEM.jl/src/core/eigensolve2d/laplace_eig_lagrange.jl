# src/core/eigensolve2d/laplace_eig_lagrange.jl
#
# Port of vfem2d/lib_eigenvalue_bound/laplace_eig_lagrange.m for the
# Lagrange CG (P1, P2, …) Laplace eigenproblem on a 2D triangular mesh
# with homogeneous Dirichlet BC.
#
# Pipeline:
#   1. Assemble the CG stiffness `A = (∇φ, ∇ψ)` and mass `M = (φ, ψ)`
#      using the *monomial* (Bernstein-monomial) local basis
#      `φ_α = L^α` exactly as MATLAB does. This basis is the one
#      `rt_hdiv_problem` consumes for its `f` argument, so feeding the
#      eigenfunctions returned here directly into `rt_hdiv_problem`
#      keeps the Lehmann–Goerisch pipeline self-consistent.
#   2. Build the boundary-DOF set:
#        * boundary vertex DOFs = unique endpoints of `m.bd_edges`,
#        * for order ≥ 2: edge-interior DOFs of all boundary edges.
#   3. Restrict to interior DOFs and solve `eigs(A0, M0; nev = neig,
#      which = :SM)` with Arpack.
#   4. Pad eigenfunctions back to full DOF size with zeros on the
#      boundary, sort ascending by eigenvalue.
#
# IMPORTANT — basis convention (cross-reference for confused readers):
# `create_matrix_lagrange.jl` (the conforming-Lagrange assembly used by
# CECR) implements the *nodal* Lagrange basis (φ_v = L(2L−1), φ_e =
# 4L_jL_k). It mirrors `vfem2d/lib/fem_assembly/create_matrix_lagrange.m`.
# That basis is *not* the one needed by the LG pipeline. Eigenvalues
# are basis-invariant so the two implementations agree on those, but
# the matrix entries and eigenfunction coefficient stacks differ.

using Arpack: eigs
using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!
using LinearAlgebra: tr, det, eigen, Hermitian

"""
    LaplaceEigLagrange

Result struct returned by [`laplace_eig_lagrange`](@ref):
* `eig_value` :: `Vector{Float64}` — first `neig` Dirichlet Laplace
  eigenvalues, sorted ascending.
* `eig_func`  :: `Matrix{Float64}` — `ndof × neig` eigenfunctions in
  monomial-basis Lagrange-coefficient form (zeros on Dirichlet DOFs).
* `A`         :: `SparseMatrixCSC{Float64}` — full ndof × ndof stiffness
  matrix (no BC removal).
* `M`         :: `SparseMatrixCSC{Float64}` — full ndof × ndof mass
  matrix (no BC removal).
* `bd_dofs`   :: `Vector{Int}` — boundary DOF indices that were removed
  before the eigensolve.
"""
struct LaplaceEigLagrange
    eig_value::Vector{Float64}
    eig_func::Matrix{Float64}
    A::SparseMatrixCSC{Float64, Int}
    M::SparseMatrixCSC{Float64, Int}
    bd_dofs::Vector{Int}
end

# ---- Bernstein-basis coordinate vectors ------------------------------------

# Coordinate vector e of length nbasis(n) for the gradient of the idx-th
# Lagrange monomial basis function L_1^i L_2^j L_3^k expressed back as a
# degree-n monomial expansion (multiply by L1+L2+L3 ≡ 1 to lift).
# Returns an `nbasis × 2` matrix [dudx | dudy].
function _Lagrange_create_coord_basis_grad(basis::AbstractMatrix{<:Integer},
                                            idx::Integer,
                                            lag_order::Integer)
    n = lag_order
    len = _RT_lag_nbasis(n)
    dudx = zeros(Float64, len)
    dudy = zeros(Float64, len)

    i, j, k = basis[idx, 1], basis[idx, 2], basis[idx, 3]

    # ∂/∂x: dL1/dx = −1, dL2/dx = +1, dL3/dx = 0. Same expansion as MATLAB
    # `Lagrange_create_coord_basis_grad`.
    p = _RT_map_ijk_to_idx(i, j, k, n);     p > 0 && (dudx[p] += -i + j)
    p = _RT_map_ijk_to_idx(i-1, j+1, k, n); p > 0 && (dudx[p] += -i)
    p = _RT_map_ijk_to_idx(i-1, j, k+1, n); p > 0 && (dudx[p] += -i)
    p = _RT_map_ijk_to_idx(i+1, j-1, k, n); p > 0 && (dudx[p] +=  j)
    p = _RT_map_ijk_to_idx(i, j-1, k+1, n); p > 0 && (dudx[p] +=  j)

    # ∂/∂y: dL1/dy = −1, dL2/dy = 0, dL3/dy = +1.
    p = _RT_map_ijk_to_idx(i, j, k, n);     p > 0 && (dudy[p] += -i + k)
    p = _RT_map_ijk_to_idx(i-1, j+1, k, n); p > 0 && (dudy[p] += -i)
    p = _RT_map_ijk_to_idx(i-1, j, k+1, n); p > 0 && (dudy[p] += -i)
    p = _RT_map_ijk_to_idx(i+1, j, k-1, n); p > 0 && (dudy[p] +=  k)
    p = _RT_map_ijk_to_idx(i, j+1, k-1, n); p > 0 && (dudy[p] +=  k)

    return hcat(dudx, dudy)
end

# Build the per-element local stiffness and mass matrices in the monomial
# Lagrange basis, returning two `nbasis × nbasis` matrices ready for
# scattering by `_cg_lagrange_local_to_global`.
function _lagrange_laplace_local_blocks(lag_order::Integer)
    basis, nbasis = _Lagrange_basis(lag_order)
    M_ip_elem = _RT_inner_product_L1L2L3_all(lag_order)

    # Σ_e e_i' M_ip e_j is a 2×2 matrix (the four entries of the gradient
    # cross-product). Cache it for later use under the affine map.
    grad_blocks = Matrix{Matrix{Float64}}(undef, nbasis, nbasis)
    M_ref      = zeros(Float64, nbasis, nbasis)
    @inbounds for i in 1:nbasis
        ei_grad = _Lagrange_create_coord_basis_grad(basis, i, lag_order)
        ei_val  = _Lagrange_create_coord_basis(basis, i, lag_order)
        for j in i:nbasis
            ej_grad = _Lagrange_create_coord_basis_grad(basis, j, lag_order)
            ej_val  = _Lagrange_create_coord_basis(basis, j, lag_order)
            grad_blocks[i, j] = ei_grad' * M_ip_elem * ej_grad
            grad_blocks[j, i] = grad_blocks[i, j]'
            M_ref[i, j] = ei_val' * M_ip_elem * ej_val
            M_ref[j, i] = M_ref[i, j]
        end
    end
    return basis, nbasis, grad_blocks, M_ref
end

# ---- Boundary DOFs ----------------------------------------------------------

# Build the boundary DOF index set for arbitrary `lagrange_order`,
# matching the global DOF ordering of `_cg_lagrange_local_to_global`:
#   1..nv               : vertex DOFs
#   nv+1..nv+(p-1)·ne   : edge-interior DOFs (p-1 per edge)
#   …+1..ndof           : element-interior DOFs (none for p ≤ 2)
function _lagrange_boundary_dofs(m::Mesh2D, lagrange_order::Integer)
    seen = falses(m.nv)
    @inbounds for r in 1:m.nb
        seen[m.bd_edges[r, 1]] = true
        seen[m.bd_edges[r, 2]] = true
    end
    bd_v = findall(seen)
    p = lagrange_order
    n_per_edge = p - 1
    if n_per_edge ≤ 0
        return bd_v
    end
    bd_e = Vector{Int}(undef, length(m.bd_edge_ids) * n_per_edge)
    cur = 1
    @inbounds for eid in m.bd_edge_ids
        base = m.nv + n_per_edge * (eid - 1)
        for r in 1:n_per_edge
            bd_e[cur] = base + r
            cur += 1
        end
    end
    return vcat(bd_v, bd_e)
end

"""
    lagrange_laplace_matrices(m::Mesh2D, p::Integer; T = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T}, bd_dofs::Vector{Int})

Assemble the conforming-Lagrange stiffness `A = (∇φ, ∇ψ)` and mass
`M = (φ, ψ)` matrices for order `p ≥ 1` in the monomial Lagrange basis.
No boundary conditions are applied — the caller restricts to interior
DOFs via `bd_dofs`.

`T = Interval{Float64}` produces interval enclosures of the bilinear
form values, suitable for the verified Lehmann–Goerisch pipeline.
"""
function lagrange_laplace_matrices(m::Mesh2D, p::Integer; T::Type = Float64)
    p ≥ 1 || throw(ArgumentError("Lagrange order must be ≥ 1 (got $p)"))
    nbasis_local = _RT_lag_nbasis(p)

    n_per_edge = p - 1
    n_per_elem = max(0, (p - 1) * (p - 2) ÷ 2)
    ndof = m.nv + n_per_edge * m.ne + n_per_elem * m.nt

    _, _, grad_blocks_f64, M_ref_f64 = _lagrange_laplace_local_blocks(p)
    grad_blocks = T === Float64 ? grad_blocks_f64 :
        [convert(Matrix{T}, grad_blocks_f64[i, j])
         for i in 1:nbasis_local, j in 1:nbasis_local]
    M_ref = T === Float64 ? M_ref_f64 : convert(Matrix{T}, M_ref_f64)

    nodes_T = T === Float64 ? m.nodes : convert(Matrix{T}, m.nodes)

    # Global accumulation goes through COO triplets and a single `sparse`
    # call (which sums duplicates), NOT scatter-adds into a CSC matrix.
    # `A[i,j] += v` on a CSC has to insert into the sparse structure when the
    # entry is new -- an O(nnz) memmove -- so element-by-element scatter is
    # quadratic in problem size. At P3 on a 64x64 mesh that was >96% of the
    # assembly time (memmove dominated the profile). Triplets are exact:
    # the local blocks are unchanged, only the order of the global sum moves.
    nentry = m.nt * nbasis_local * nbasis_local
    Irow = Vector{Int}(undef, nentry)
    Jcol = Vector{Int}(undef, nentry)
    Aval = Vector{T}(undef, nentry)
    Mval = Vector{T}(undef, nentry)
    pos = 0

    @inbounds for k in 1:m.nt
        v1 = m.elements[k, 1]; v2 = m.elements[k, 2]; v3 = m.elements[k, 3]
        x1, y1 = nodes_T[v1, 1], nodes_T[v1, 2]
        x2, y2 = nodes_T[v2, 1], nodes_T[v2, 2]
        x3, y3 = nodes_T[v3, 1], nodes_T[v3, 2]
        Bmat = T[x2 - x1  x3 - x1;
                 y2 - y1  y3 - y1]
        det_B = Bmat[1, 1] * Bmat[2, 2] - Bmat[1, 2] * Bmat[2, 1]
        Binv = T[ y3 - y1   x1 - x3;
                  y1 - y2   x2 - x1] ./ det_B

        A_local = zeros(T, nbasis_local, nbasis_local)
        for i in 1:nbasis_local, j in 1:nbasis_local
            A_local[i, j] = tr(Binv' * grad_blocks[i, j] * Binv) * det_B
        end
        M_local = M_ref .* det_B

        g = _cg_lagrange_local_to_global(m, k, p)
        for i in 1:nbasis_local, j in 1:nbasis_local
            pos += 1
            Irow[pos] = g[i]
            Jcol[pos] = g[j]
            Aval[pos] = A_local[i, j]
            Mval[pos] = M_local[i, j]
        end
    end

    A = dropzeros!(sparse(Irow, Jcol, Aval, ndof, ndof))
    M = dropzeros!(sparse(Irow, Jcol, Mval, ndof, ndof))

    bd_dofs = _lagrange_boundary_dofs(m, p)
    return A, M, bd_dofs
end

"""
    laplace_eig_lagrange(m::Mesh2D, lagrange_order::Integer, neig::Integer)
        -> LaplaceEigLagrange

Compute the first `neig` Dirichlet Laplace eigenpairs on `m` with
conforming Lagrange `P_order` elements (`order` ≥ 1).

The local basis is the *monomial* Lagrange basis `φ_α = L^α`. This is
the basis required by `rt_hdiv_problem`, so the returned `eig_func`
columns can be passed straight through to the LG sharpening step.

Returns `LaplaceEigLagrange(eig_value, eig_func, A, M, bd_dofs)` —
`A`, `M` are the full (no BC removal) stiffness/mass matrices and
`eig_func` is padded with zeros at the Dirichlet DOFs so it has the
same number of rows as `A`.
"""
function laplace_eig_lagrange(m::Mesh2D, lagrange_order::Integer,
                              neig::Integer)
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))

    p = lagrange_order
    A, M, bd_dofs = lagrange_laplace_matrices(m, p; T = Float64)
    ndof = size(A, 1)
    is_bd = falses(ndof)
    @inbounds for d in bd_dofs
        is_bd[d] = true
    end
    int_dof = findall(!, is_bd)

    A0 = A[int_dof, int_dof]
    M0 = M[int_dof, int_dof]

    k_eff = min(neig, size(A0, 1) - 1)
    n_int = size(A0, 1)
    if n_int ≤ 6000
        # Dense path — robust on high-order / fine meshes where Arpack
        # struggles with the smallest-magnitude shift-invert.
        Ad = Matrix(A0); Md = Matrix(M0)
        F  = eigen(Hermitian((Ad + Ad') / 2), Hermitian((Md + Md') / 2))
        perm0 = sortperm(F.values)[1:k_eff]
        eig_value = F.values[perm0]
        V_sorted  = F.vectors[:, perm0]
    else
        # Shift-invert about 0 — smallest generalized eigenvalues. We
        # expand ncv well beyond the default to give Arnoldi room to
        # converge on higher-order / finer-mesh cases.
        ncv = min(n_int - 1, max(40, 4 * k_eff + 5))
        λ_arr, V_arr = eigs(A0, M0; nev = k_eff, sigma = 0.0,
                            which = :LM, ncv = ncv,
                            tol = 1e-10, maxiter = 1000)
        eig_int = real.(λ_arr)
        perm0 = sortperm(eig_int)
        eig_value = eig_int[perm0]
        V_sorted = real.(V_arr[:, perm0])
    end

    eig_func = zeros(Float64, ndof, k_eff)
    eig_func[int_dof, :] .= V_sorted

    return LaplaceEigLagrange(eig_value, eig_func, A, M, bd_dofs)
end
