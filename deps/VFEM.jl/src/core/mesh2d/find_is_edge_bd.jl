# src/core/mesh2d/find_is_edge_bd.jl
#
# Port of vfem2d/lib/fem_assembly/find_is_edge_bd.m. Returns the
# 0/1 indicator vector of which edges are on the boundary, via a
# hash join on sorted-pair encoding.

"""
    find_is_edge_bd(edges::AbstractMatrix{<:Integer},
                    bd_edges::AbstractMatrix{<:Integer}) -> Vector{Int}

Return a length-`size(edges, 1)` 0/1 vector. Entry r is 1 iff
edge row r matches some row of `bd_edges` (after sorting endpoints
ascending).
"""
function find_is_edge_bd(edges::AbstractMatrix{<:Integer},
                         bd_edges::AbstractMatrix{<:Integer})
    size(edges, 2) == 2 ||
        throw(DimensionMismatch("edges must have 2 columns"))
    size(bd_edges, 2) == 2 ||
        throw(DimensionMismatch("bd_edges must have 2 columns"))

    ne = size(edges, 1)
    nb = size(bd_edges, 1)

    bd_set = Set{NTuple{2, Int}}()
    sizehint!(bd_set, nb)
    @inbounds for r in 1:nb
        a, b = Int(bd_edges[r, 1]), Int(bd_edges[r, 2])
        push!(bd_set, a < b ? (a, b) : (b, a))
    end

    out = zeros(Int, ne)
    @inbounds for r in 1:ne
        a, b = Int(edges[r, 1]), Int(edges[r, 2])
        key = a < b ? (a, b) : (b, a)
        if key in bd_set
            out[r] = 1
        end
    end
    return out
end
