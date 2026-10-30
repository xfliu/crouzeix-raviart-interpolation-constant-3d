# src/core/mesh/get_edge_list.jl
#
# Port of VFEM3D/lib/mesh/mesh_get_EdgeList.m. The MATLAB version
# is O(NumElt × NumEdgeSoFar) due to a linear scan; we use a Set
# for O(NumElt × 6) lookups while keeping the same insertion order.

"""
    get_edge_list(ElementList::AbstractMatrix{<:Integer}) -> Matrix{Int}

Return the `NumEdge × 2` matrix of unique edges. Each row's nodes
are stored in the same order as the MATLAB original — the sorted
(rows-of-ElementList) pair `(e[i], e[j])` for i < j with
i, j ∈ {1, 2, 3, 4}, walked in that lexicographic order. Insertion
order matches MATLAB's `mesh_get_EdgeList.m`.
"""
function get_edge_list(ElementList::AbstractMatrix{<:Integer})
    size(ElementList, 2) == 4 ||
        throw(DimensionMismatch("ElementList must be NumElt × 4"))
    nelt = size(ElementList, 1)
    nelt == 0 && return Matrix{Int}(undef, 0, 2)

    M = maximum(ElementList)
    seen = Set{Int}()
    sizehint!(seen, 6 * nelt)
    edges = Matrix{Int}(undef, 6 * nelt, 2)
    cur = 1
    @inbounds for k in 1:nelt
        e = (Int(ElementList[k, 1]), Int(ElementList[k, 2]),
             Int(ElementList[k, 3]), Int(ElementList[k, 4]))
        # MATLAB's traversal: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
        local_edges = ((e[1], e[2]), (e[1], e[3]), (e[1], e[4]),
                       (e[2], e[3]), (e[2], e[4]), (e[3], e[4]))
        for ed in local_edges
            key = ed[1] * (M + 1) + ed[2]
            if key ∉ seen
                push!(seen, key)
                edges[cur, 1] = ed[1]
                edges[cur, 2] = ed[2]
                cur += 1
            end
        end
    end
    return edges[1:(cur - 1), :]
end
