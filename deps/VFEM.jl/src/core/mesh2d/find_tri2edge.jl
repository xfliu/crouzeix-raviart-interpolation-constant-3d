# src/core/mesh2d/find_tri2edge.jl
#
# Port of vfem2d/lib/fem_assembly/find_tri2edge.m.
#
# Convention: column k of the returned `tri2edge` matrix holds the
# global edge index of the local edge *opposite* vertex k of the
# triangle — local edge order [(2,3), (1,3), (1,2)]. Edges are
# matched by sorted endpoint pairs.

"""
    find_tri2edge(tri::AbstractMatrix{<:Integer},
                  edge::AbstractMatrix{<:Integer}) -> Matrix{Int}

Build the `nt × 3` triangle-to-edge map. `tri2edge[k, j]` is the
global edge index (row of `edge`) of the j-th local edge of triangle
`k`, where local edges are ordered as `(v2, v3)`, `(v1, v3)`,
`(v1, v2)` — opposite vertices 1, 2, 3 respectively.
"""
function find_tri2edge(tri::AbstractMatrix{<:Integer},
                       edge::AbstractMatrix{<:Integer})
    size(tri, 2) == 3 ||
        throw(DimensionMismatch("tri must have 3 columns"))
    size(edge, 2) == 2 ||
        throw(DimensionMismatch("edge must have 2 columns"))

    nt = size(tri, 1)
    ne = size(edge, 1)
    out = Matrix{Int}(undef, nt, 3)

    # Hash sorted-edge pair → global index.
    edge_to_idx = Dict{NTuple{2, Int}, Int}()
    sizehint!(edge_to_idx, ne)
    @inbounds for r in 1:ne
        a, b = Int(edge[r, 1]), Int(edge[r, 2])
        a, b = a < b ? (a, b) : (b, a)
        edge_to_idx[(a, b)] = r
    end

    @inbounds for k in 1:nt
        v1, v2, v3 = Int(tri[k, 1]), Int(tri[k, 2]), Int(tri[k, 3])
        # Local edge order from MATLAB: edge_local = [2 3; 1 3; 1 2]
        for (j, (a, b)) in enumerate(((v2, v3), (v1, v3), (v1, v2)))
            a_, b_ = a < b ? (a, b) : (b, a)
            out[k, j] = edge_to_idx[(a_, b_)]
        end
    end
    return out
end
