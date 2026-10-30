# src/core/assembly2d/build_ecr_dof_ordering.jl
#
# Port of vfem2d/lib/fem_assembly/build_ecr_dof_ordering.m.
#
# Renumber the ECR/CECR DOFs (edges first, then cells) so that each
# element's three edge DOFs and one cell DOF are close in the global
# numbering. This produces a sparser fill-in pattern when factoring
# the assembled matrices.

"""
    EcrDofOrdering

Result struct of `build_ecr_dof_ordering`:
* `edge`        :: `Vector{Int}` of length `ne` — new global index for each edge DOF
* `cell`        :: `Vector{Int}` of length `nt` — new global index for each cell DOF
* `local`       :: `Matrix{Int}` of size `nt × 4` — local-to-global mapping
                   `[edge1 edge2 edge3 cell]` per element
* `old_to_new`  :: `Vector{Int}` of length `ne+nt` — permutation: legacy → new
* `new_to_old`  :: `Vector{Int}` of length `ne+nt` — inverse permutation
"""
struct EcrDofOrdering
    edge::Vector{Int}
    cell::Vector{Int}
    local_dof::Matrix{Int}
    old_to_new::Vector{Int}
    new_to_old::Vector{Int}
end

"""
    build_ecr_dof_ordering(tri2edge::AbstractMatrix{<:Integer}, ne::Integer)
        -> EcrDofOrdering

Walk elements in order, assigning global DOFs as we go: any unseen
edge DOFs of the current element first, then the cell DOF immediately
after them. Result: ECR DOFs of each element form a near-contiguous
block in the global numbering.
"""
function build_ecr_dof_ordering(tri2edge::AbstractMatrix{<:Integer}, ne::Integer)
    size(tri2edge, 2) == 3 ||
        throw(DimensionMismatch("tri2edge must have 3 columns"))
    nt = size(tri2edge, 1)
    ndof = ne + nt

    edge_dof = zeros(Int, ne)
    cell_dof = zeros(Int, nt)
    local_dof = zeros(Int, nt, 4)

    next = 1
    @inbounds for k in 1:nt
        for j in 1:3
            eid = Int(tri2edge[k, j])
            if edge_dof[eid] == 0
                edge_dof[eid] = next
                next += 1
            end
            local_dof[k, j] = edge_dof[eid]
        end
        cell_dof[k] = next
        local_dof[k, 4] = next
        next += 1
    end

    next - 1 == ndof ||
        error("build_ecr_dof_ordering: invalid numbering size (got $(next - 1), expected $ndof)")

    old_to_new = vcat(edge_dof, cell_dof)
    new_to_old = zeros(Int, ndof)
    @inbounds for i in 1:ndof
        new_to_old[old_to_new[i]] = i
    end
    return EcrDofOrdering(edge_dof, cell_dof, local_dof, old_to_new, new_to_old)
end
