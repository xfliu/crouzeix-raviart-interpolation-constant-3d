# src/core/assembly2d/create_matrix_enriched_crouzeix_raviart.jl
#
# Port of vfem2d/lib/fem_assembly/create_matrix_enriched_crouzeix_raviart.m.
#
# Exact (no-quadrature) ECR assembly via the degree-2 Bernstein basis
# on each physical triangle. This is the "trustable" assembly path —
# the local mass and stiffness matrices come from closed-form Gram
# matrices, not from a numerical quadrature rule.
#
# Bernstein-2 ordering: [L1², L2², L3², 2 L1 L2, 2 L2 L3, 2 L3 L1].
#
# Reference Gram matrices (from tex/main.tex):
#   G2 = [6 1 1 3 1 3; 1 6 1 3 3 1; 1 1 6 1 3 3;
#         3 3 1 4 2 2; 1 3 3 2 4 2; 3 1 3 2 2 4] / 90
#       = (1/|K|) * ∫_K B_i^{(2)} B_j^{(2)} dx
#   G1 = [2 1 1; 1 2 1; 1 1 2] / 12
#       = (1/|K|) * ∫_K L_i L_j dx
#
# DOF numbering follows `build_ecr_dof_ordering` so each element's
# three edge DOFs + one cell DOF are nearly contiguous in the global
# matrix.

using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!

# Element-independent constants (Bernstein-2 form). These are the
# `C_cr_data`, `G2_data`, `G1_data` from the MATLAB source.
const _ECR_C_CR = [
    -1  1  1;
     1 -1  1;
     1  1 -1;
     0  0  1;
     1  0  0;
     0  1  0
]

const _ECR_G2_NUM = [
    6 1 1 3 1 3;
    1 6 1 3 3 1;
    1 1 6 1 3 3;
    3 3 1 4 2 2;
    1 3 3 2 4 2;
    3 1 3 2 2 4
]

const _ECR_G1_NUM = [
    2 1 1;
    1 2 1;
    1 1 2
]

# Local Bernstein-2 coefficient matrix (6×4) of the ECR basis for one
# physical triangle: columns 1–3 are the edge basis functions, column
# 4 is the cell-average basis function.
function _local_ecr_bernstein_coefficients(P::AbstractMatrix{T},
                                           C_cr::AbstractMatrix{T}) where {T<:Real}
    # ‖P_i‖² and ⟨P_i, P_j⟩ for i ≠ j.
    p1_sq = P[1, 1] * P[1, 1] + P[1, 2] * P[1, 2]
    p2_sq = P[2, 1] * P[2, 1] + P[2, 2] * P[2, 2]
    p3_sq = P[3, 1] * P[3, 1] + P[3, 2] * P[3, 2]
    dot12 = P[1, 1] * P[2, 1] + P[1, 2] * P[2, 2]
    dot23 = P[2, 1] * P[3, 1] + P[2, 2] * P[3, 2]
    dot31 = P[3, 1] * P[1, 1] + P[3, 2] * P[1, 2]

    # Bernstein-2 coefficients of q(x) = x² + y² on the triangle.
    c_q = T[p1_sq, p2_sq, p3_sq, dot12, dot23, dot31]

    m_edge = T[(p2_sq + p3_sq + dot23) / T(3),
               (p3_sq + p1_sq + dot31) / T(3),
               (p1_sq + p2_sq + dot12) / T(3)]
    m_cell = (p1_sq + p2_sq + p3_sq + dot12 + dot23 + dot31) / T(6)
    β = m_cell - (m_edge[1] + m_edge[2] + m_edge[3]) / T(3)

    c_psi = (c_q .- C_cr * m_edge) ./ β

    # Edge basis cols: subtract (c_psi / 3) from each CR column to make
    # the cell-average vanish on edge DOFs.
    edge_cols = C_cr .- c_psi * (ones(T, 1, 3) / T(3))
    return hcat(edge_cols, c_psi)
end

# 3×2 matrix whose i-th row is grad(L_i) on the physical triangle.
function _local_grad_lambda(P::AbstractMatrix{T}, areaK::T) where {T<:Real}
    # edge_vec(i, :) = P[next] - P[prev], the edge opposite vertex i.
    edge_vec = T[P[3, 1] - P[2, 1] P[3, 2] - P[2, 2];
                 P[1, 1] - P[3, 1] P[1, 2] - P[3, 2];
                 P[2, 1] - P[1, 1] P[2, 2] - P[1, 2]]
    # grad L_i is the inward normal divided by 2|K|.
    return T[-edge_vec[1, 2]  edge_vec[1, 1];
             -edge_vec[2, 2]  edge_vec[2, 1];
             -edge_vec[3, 2]  edge_vec[3, 1]] ./ (2 * areaK)
end

# For Bernstein-2 coefficients c (length 6) of a polynomial p, return
# the 3×2 matrix `gc` such that grad(p) = L1·gc[1, :] + L2·gc[2, :] +
# L3·gc[3, :]. The closed form comes from differentiating
#   p = c1 L1² + c2 L2² + c3 L3² + c4·2·L1L2 + c5·2·L2L3 + c6·2·L3L1.
function _bernstein_quadratic_grad_coeffs(c::AbstractVector{T},
                                          grad_λ::AbstractMatrix{T}) where {T<:Real}
    gc = Matrix{T}(undef, 3, 2)
    @inbounds for d in 1:2
        gc[1, d] = 2 * (c[1] * grad_λ[1, d] + c[4] * grad_λ[2, d] + c[6] * grad_λ[3, d])
        gc[2, d] = 2 * (c[4] * grad_λ[1, d] + c[2] * grad_λ[2, d] + c[5] * grad_λ[3, d])
        gc[3, d] = 2 * (c[6] * grad_λ[1, d] + c[5] * grad_λ[2, d] + c[3] * grad_λ[3, d])
    end
    return gc
end

"""
    create_matrix_enriched_crouzeix_raviart(m::Mesh2D; T::Type = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T},
            dof_map::EcrDofOrdering)

Exact (no-quadrature) ECR assembly via the degree-2 Bernstein basis.
Returns the global stiffness `A`, mass `M`, and the DOF ordering
struct (caller's CECR layer needs it for the cell-DOF lookup).

Generic on `T<:Real`. Cross-validated against MATLAB element-wise on
UnitSquare8x8 — see `test_create_matrix_enriched_crouzeix_raviart.jl`.
"""
function create_matrix_enriched_crouzeix_raviart(m::Mesh2D; T::Type = Float64)
    nt = m.nt
    ne = m.ne
    ndof = ne + nt
    dof_map = build_ecr_dof_ordering(m.tri2edge, ne)

    C_cr    = T.(_ECR_C_CR)
    G2_core = T.(_ECR_G2_NUM) ./ T(90)
    G1_core = T.(_ECR_G1_NUM) ./ T(12)

    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = nt * 16
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Aval = Vector{T}(undef, nent);   Mval = Vector{T}(undef, nent)
    pos = 0

    @inbounds for k in 1:nt
        t = (m.elements[k, 1], m.elements[k, 2], m.elements[k, 3])
        P = T[m.nodes[t[1], 1] m.nodes[t[1], 2];
              m.nodes[t[2], 1] m.nodes[t[2], 2];
              m.nodes[t[3], 1] m.nodes[t[3], 2]]
        dof_idx = dof_map.local_dof[k, :]

        areaK = abs(T(1) / T(2) * ((P[2, 1] - P[1, 1]) * (P[3, 2] - P[1, 2])
                                  - (P[3, 1] - P[1, 1]) * (P[2, 2] - P[1, 2])))
        C_ecr = _local_ecr_bernstein_coefficients(P, C_cr)
        grad_λ = _local_grad_lambda(P, areaK)

        # Local mass: areaK · C_ecr' · G2_core · C_ecr.
        M_loc = areaK .* (C_ecr' * G2_core * C_ecr)

        # Local stiffness: integrate grad(phi_i) · grad(phi_j) using
        # the Bernstein gradient closed form.
        Gx = Matrix{T}(undef, 3, 4)
        Gy = Matrix{T}(undef, 3, 4)
        for j in 1:4
            gc = _bernstein_quadratic_grad_coeffs(C_ecr[:, j], grad_λ)
            Gx[:, j] = gc[:, 1]
            Gy[:, j] = gc[:, 2]
        end
        A_loc = areaK .* (Gx' * G1_core * Gx + Gy' * G1_core * Gy)

        for i in 1:4, j in 1:4
            pos += 1
            Irow[pos] = dof_idx[i]; Jcol[pos] = dof_idx[j]
            Aval[pos] = A_loc[i, j]
            Mval[pos] = M_loc[i, j]
        end
    end
    A = dropzeros!(sparse(Irow, Jcol, Aval, ndof, ndof))
    M = dropzeros!(sparse(Irow, Jcol, Mval, ndof, ndof))

    return A, M, dof_map
end
