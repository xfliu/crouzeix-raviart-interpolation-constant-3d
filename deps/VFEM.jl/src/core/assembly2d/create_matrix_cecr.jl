# src/core/assembly2d/create_matrix_cecr.jl
#
# Port of vfem2d/lib/fem_assembly/create_matrix_cecr.m.
#
# CECR adds a reaction term `c · u_h^proj · v_h^proj` to the ECR
# stiffness, where the projection is the elementwise L²-projection
# onto piecewise constants. Since cell-average DOF *is* the
# projection, the reaction matrix is purely diagonal on cell DOFs:
#   A_extra[cell_dof_K, cell_dof_K] = c_K · |K|.

using SparseArrays: spzeros, sparse

"""
    create_matrix_cecr(m::Mesh2D, c_data; T::Type = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T},
            dof_map::EcrDofOrdering)

CECR matrices for the bilinear forms
    â({u₁,u₂},{v₁,v₂}) = (∇u₁, ∇v₁) + (c·u₂, v₂),
    b̂({u₁,u₂},{v₁,v₂}) = (u₁, v₁),
on the discrete space {(u_h, Π₀ u_h) : u_h ∈ V_h^{ECR}}, where Π₀
is the elementwise L²-projection onto piecewise constants.

`c_data` may be:
* a `Real` (scalar c, broadcast to every element),
* an `AbstractVector` of length `nt` (per-element constant c_K),
* a function `(x, y) -> Real` evaluated at element centroids.

Returns `(A, M, dof_map)`. `M` equals the ECR mass — `b̂` uses only
the first component.
"""
function create_matrix_cecr(m::Mesh2D, c_data; T::Type = Float64)
    A, M, dof_map = create_matrix_enriched_crouzeix_raviart(m; T = T)

    nt = m.nt
    c_elem = Vector{T}(undef, nt)
    if c_data isa Function
        @inbounds for k in 1:nt
            xc = (m.nodes[m.elements[k, 1], 1] + m.nodes[m.elements[k, 2], 1]
                + m.nodes[m.elements[k, 3], 1]) / T(3)
            yc = (m.nodes[m.elements[k, 1], 2] + m.nodes[m.elements[k, 2], 2]
                + m.nodes[m.elements[k, 3], 2]) / T(3)
            c_elem[k] = T(c_data(xc, yc))
        end
    elseif c_data isa Real
        fill!(c_elem, T(c_data))
    elseif c_data isa AbstractVector
        length(c_data) == nt ||
            throw(DimensionMismatch("c_data length $(length(c_data)) ≠ nt $nt"))
        @inbounds for k in 1:nt
            c_elem[k] = T(c_data[k])
        end
    else
        throw(ArgumentError("c_data must be a Real, an nt-vector, or a function"))
    end

    @inbounds for k in 1:nt
        v1 = (T(m.nodes[m.elements[k, 1], 1]), T(m.nodes[m.elements[k, 1], 2]))
        v2 = (T(m.nodes[m.elements[k, 2], 1]), T(m.nodes[m.elements[k, 2], 2]))
        v3 = (T(m.nodes[m.elements[k, 3], 1]), T(m.nodes[m.elements[k, 3], 2]))
        areaK = abs(T(1) / T(2) * ((v2[1] - v1[1]) * (v3[2] - v1[2])
                                  - (v3[1] - v1[1]) * (v2[2] - v1[2])))
        cell = dof_map.cell[k]
        A[cell, cell] += c_elem[k] * areaK
    end
    return A, M, dof_map
end
