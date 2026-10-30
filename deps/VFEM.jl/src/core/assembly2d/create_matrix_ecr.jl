# src/core/assembly2d/create_matrix_ecr.jl
#
# Port of vfem2d/lib/fem_assembly/create_matrix_ecr.m.
#
# ECR (Enriched Crouzeix–Raviart) on each triangle K uses
#     P_1(K) + span{ x²+y² }
# with 4 DOFs:
#   1–3: edge averages on the three edges (opposite vertices 1, 2, 3)
#   4  : cell average on K
#
# Global DOF ordering:
#   1:ne          edge-average DOFs
#   ne+1:ne+nt    element-average DOFs
#
# The MATLAB original is "trustable": it reconstructs the local basis
# from the DOF definitions (4×4 system) and integrates by quadrature.
# This port preserves that approach exactly so the assembly is faithful.
#
# Verification: cross-validated against MATLAB to ~1e-13 entry-wise on
# the UnitSquare8x8 fixture (208 edges, 128 elements ⇒ 336×336 matrix).

using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!
using LinearAlgebra: det

# Local 4×4 DOF matrix mapping polynomial coefficients (a, b, c, d) for
# `p(x, y) = a + b·x + c·y + d·(x² + y²)` to the 4 DOF values of `p`
# (3 edge averages + cell average).
function _ecr_local_dof_matrix(P::AbstractMatrix{T}) where {T<:Real}
    D = Matrix{T}(undef, 4, 4)
    edge_vertices = ((2, 3), (3, 1), (1, 2))
    @inbounds for i in 1:3
        a, b = edge_vertices[i]
        Ax, Ay = P[a, 1], P[a, 2]
        Bx, By = P[b, 1], P[b, 2]
        avg_x  = (Ax + Bx) / T(2)
        avg_y  = (Ay + By) / T(2)
        avg_r2 = (Ax * Ax + Ay * Ay
                + Ax * Bx + Ay * By
                + Bx * Bx + By * By) / T(3)
        D[i, 1] = one(T)
        D[i, 2] = avg_x
        D[i, 3] = avg_y
        D[i, 4] = avg_r2
    end
    xbar = (P[1, 1] + P[2, 1] + P[3, 1]) / T(3)
    ybar = (P[1, 2] + P[2, 2] + P[3, 2]) / T(3)
    Σ2 = (P[1, 1] * P[1, 1] + P[1, 2] * P[1, 2]
        + P[2, 1] * P[2, 1] + P[2, 2] * P[2, 2]
        + P[3, 1] * P[3, 1] + P[3, 2] * P[3, 2])
    Cross = (P[1, 1] * P[2, 1] + P[1, 2] * P[2, 2]
           + P[2, 1] * P[3, 1] + P[2, 2] * P[3, 2]
           + P[3, 1] * P[1, 1] + P[3, 2] * P[1, 2])
    D[4, 1] = one(T)
    D[4, 2] = xbar
    D[4, 3] = ybar
    D[4, 4] = (Σ2 + Cross) / T(6)
    return D
end

# Evaluate ECR basis values and gradients at point x = (x1, x2).
# `coeff[:, j]` = (a, b, c, d) for basis φ_j.
function _ecr_eval_basis(coeff::AbstractMatrix{T}, x1::T, x2::T) where {T<:Real}
    r2 = x1 * x1 + x2 * x2
    phi = Vector{T}(undef, 4)
    grad = Matrix{T}(undef, 4, 2)
    @inbounds for j in 1:4
        a = coeff[1, j]; b = coeff[2, j]; c = coeff[3, j]; d = coeff[4, j]
        phi[j] = a + b * x1 + c * x2 + d * r2
        grad[j, 1] = b + 2 * d * x1
        grad[j, 2] = c + 2 * d * x2
    end
    return phi, grad
end

"""
    create_matrix_ecr(m::Mesh2D; T::Type = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T})

Assemble the global ECR stiffness `A` and mass `M` matrices on the
2D mesh. ECR space is `P₁(K) ⊕ span{x²+y²}`. Global DOF size is
`ne + nt`: edge-average DOFs first (`1:ne`), then cell-average DOFs
(`ne+1:ne+nt`).

`T` controls the numeric type. Generic on `T<:Real`.
"""
function create_matrix_ecr(m::Mesh2D; T::Type = Float64)
    nt = m.nt
    ne = m.ne
    ndof = ne + nt
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = nt * 16
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Aval = Vector{T}(undef, nent);   Mval = Vector{T}(undef, nent)
    pos = 0

    λ_q, w_q = dunavant_rule_6()
    nq = length(w_q)

    @inbounds for k in 1:nt
        t = (m.elements[k, 1], m.elements[k, 2], m.elements[k, 3])
        P = T[m.nodes[t[1], 1] m.nodes[t[1], 2];
              m.nodes[t[2], 1] m.nodes[t[2], 2];
              m.nodes[t[3], 1] m.nodes[t[3], 2]]
        dof_idx = (m.tri2edge[k, 1], m.tri2edge[k, 2], m.tri2edge[k, 3], ne + k)

        areaK = abs(T(1) / T(2) * ((P[2, 1] - P[1, 1]) * (P[3, 2] - P[1, 2])
                                  - (P[3, 1] - P[1, 1]) * (P[2, 2] - P[1, 2])))

        D = _ecr_local_dof_matrix(P)
        coeff = D \ Matrix{T}(LinearAlgebra.I, 4, 4)

        A_loc = zeros(T, 4, 4)
        M_loc = zeros(T, 4, 4)
        for q in 1:nq
            lam = λ_q[q, :]
            xq1 = T(lam[1]) * P[1, 1] + T(lam[2]) * P[2, 1] + T(lam[3]) * P[3, 1]
            xq2 = T(lam[1]) * P[1, 2] + T(lam[2]) * P[2, 2] + T(lam[3]) * P[3, 2]
            phi, grad = _ecr_eval_basis(coeff, xq1, xq2)
            wq = T(w_q[q])
            for i in 1:4, j in 1:4
                M_loc[i, j] += phi[i] * phi[j] * wq
                A_loc[i, j] += (grad[i, 1] * grad[j, 1] + grad[i, 2] * grad[j, 2]) * wq
            end
        end
        # ∫_K f = 2|K| · Σ_q w_q f(x_q).
        scaling = 2 * areaK
        for i in 1:4, j in 1:4
            pos += 1
            Irow[pos] = dof_idx[i]; Jcol[pos] = dof_idx[j]
            Aval[pos] = scaling * A_loc[i, j]
            Mval[pos] = scaling * M_loc[i, j]
        end
    end
    A = dropzeros!(sparse(Irow, Jcol, Aval, ndof, ndof))
    M = dropzeros!(sparse(Irow, Jcol, Mval, ndof, ndof))

    return A, M
end
