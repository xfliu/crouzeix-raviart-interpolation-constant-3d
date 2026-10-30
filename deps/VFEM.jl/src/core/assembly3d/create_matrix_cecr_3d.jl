# src/core/assembly3d/create_matrix_cecr_3d.jl
#
# Port of VFEM3D/create_matrix_cecr_3d.m.
#
# CECR (Composite Enriched Crouzeix–Raviart) is the ECR element with an
# additional element-wise reaction term acting only on the cell-average
# DOF. Bilinear forms:
#   a_hat({u1, u2}, {v1, v2}) = (∇u1, ∇v1) + (c · u2, v2)
#   b_hat({u1, u2}, {v1, v2}) = (u1, v1)
# discretised on
#   V̂_h = { (u_h, Π_{0,h} u_h) : u_h ∈ V_h^{ECR} }
# where Π_{0,h} is element-wise L²-projection onto piecewise constants
# (= the cell-average DOF in the ECR basis).
#
# The reaction term reduces to a per-element diagonal entry on the
# cell DOF: A[NumF + e, NumF + e] += c_e · |K_e|.

using SparseArrays: spzeros, sparse, SparseMatrixCSC

# Per-element volume (1/6 |det(P₂−P₁, P₃−P₁, P₄−P₁)|).
function _tet_element_volume(m::Mesh3D, e::Integer, ::Type{T}) where {T<:Real}
    v1 = m.ElementList[e, 1]; v2 = m.ElementList[e, 2]
    v3 = m.ElementList[e, 3]; v4 = m.ElementList[e, 4]
    Jmat = T[ m.NodeList[v2, 1] - m.NodeList[v1, 1]   m.NodeList[v2, 2] - m.NodeList[v1, 2]   m.NodeList[v2, 3] - m.NodeList[v1, 3] ;
              m.NodeList[v3, 1] - m.NodeList[v1, 1]   m.NodeList[v3, 2] - m.NodeList[v1, 2]   m.NodeList[v3, 3] - m.NodeList[v1, 3] ;
              m.NodeList[v4, 1] - m.NodeList[v1, 1]   m.NodeList[v4, 2] - m.NodeList[v1, 2]   m.NodeList[v4, 3] - m.NodeList[v1, 3] ]
    return abs(_det3(Jmat)) / T(6)
end

"""
    create_matrix_cecr_3d(m::Mesh3D, c_data::AbstractVector;
                          T::Type = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T}, dof::EcrDof3D)

Assemble the global CECR stiffness `A` and mass `M` matrices on the
3D mesh `m`. `c_data` is the element-wise reaction coefficient
(typically `V_K`, the average potential on element K). The reaction
acts only on the cell-average DOF: `A[NumF + e, NumF + e] += c_e · |K_e|`.

DOF layout matches `create_matrix_ecr_3d`: facets first (`1..NumF`),
then cells (`NumF+1..NumF+NumElt`).
"""
function create_matrix_cecr_3d(m::Mesh3D, c_data::AbstractVector;
                                T::Type = Float64)
    length(c_data) == m.NumElt ||
        throw(DimensionMismatch("c_data must have length NumElt = $(m.NumElt), got $(length(c_data))"))

    A, M, info = create_matrix_ecr_3d(m; T = T)

    # Add reaction term to cell DOFs.
    @inbounds for e in 1:m.NumElt
        vol_e = _tet_element_volume(m, e, T)
        A[info.NumF + e, info.NumF + e] += T(c_data[e]) * vol_e
    end
    return A, M, info
end
