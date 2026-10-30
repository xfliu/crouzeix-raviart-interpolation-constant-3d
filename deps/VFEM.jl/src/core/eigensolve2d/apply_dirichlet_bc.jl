# src/core/eigensolve2d/apply_dirichlet_bc.jl
#
# Generic Dirichlet-BC reducer. Given a mesh, the ECR DOF ordering,
# and any matrix on the full ECR DOF set, return the submatrix of
# interior DOFs (i.e. DOFs not associated with boundary edges).

using SparseArrays: SparseMatrixCSC

"""
    interior_ecr_dofs(m::Mesh2D, dof_map::EcrDofOrdering) -> Vector{Int}

Indices (in the ECR DOF numbering produced by `build_ecr_dof_ordering`)
of DOFs that are NOT on the Dirichlet boundary. Cell DOFs are always
interior. Edge DOFs are boundary iff their edge is in `m.bd_edge_ids`.
"""
function interior_ecr_dofs(m::Mesh2D, dof_map::EcrDofOrdering)
    ndof = m.ne + m.nt
    is_bd = falses(ndof)
    @inbounds for eid in m.bd_edge_ids
        is_bd[dof_map.edge[eid]] = true
    end
    return findall(!, is_bd)
end

"""
    restrict_to_interior(A, int_dof) -> A[int_dof, int_dof]

Convenience for slicing a global matrix down to its interior block.
Operates on any `AbstractMatrix` (the result is the same matrix type).
"""
restrict_to_interior(A::AbstractMatrix, int_dof::AbstractVector{<:Integer}) =
    A[int_dof, int_dof]
