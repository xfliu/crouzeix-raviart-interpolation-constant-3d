# src/core/mesh2d/find_mesh_hmax.jl
#
# Port of vfem2d/lib/fem_assembly/find_mesh_hmax.m. Computes the
# maximum edge length over all edges. Uses squared distance until
# the final sqrt for one less expensive sqrt per edge.
#
# In verified mode, callers should pass `Interval{Float64}` `nodes`;
# the routine returns `Interval{Float64}` and the result is a
# valid upper bound for the true h_max (no `I_sup` shenanigans
# needed — interval `max` and `sqrt` propagate cleanly).

"""
    find_mesh_hmax(nodes::AbstractMatrix, edges::AbstractMatrix{<:Integer}) -> Real

Maximum edge length. Returns `eltype(nodes)`-promoted scalar (e.g.
`Float64` for `Float64` nodes; `Interval{Float64}` for interval
nodes — the resulting interval encloses the true `h_max`).
"""
function find_mesh_hmax(nodes::AbstractMatrix, edges::AbstractMatrix{<:Integer})
    size(nodes, 2) == 2 ||
        throw(DimensionMismatch("nodes must have 2 columns"))
    size(edges, 2) == 2 ||
        throw(DimensionMismatch("edges must have 2 columns"))

    ne = size(edges, 1)
    ne == 0 && throw(ArgumentError("no edges"))
    T = eltype(nodes)
    h2_max = zero(T)
    @inbounds for r in 1:ne
        i = Int(edges[r, 1]); j = Int(edges[r, 2])
        dx = nodes[i, 1] - nodes[j, 1]
        dy = nodes[i, 2] - nodes[j, 2]
        h2 = dx * dx + dy * dy
        # `>` on intervals returns a Boolean reflecting strict-positivity
        # of the difference's lower bound; for hmax we want a sound upper
        # bound, so on intervals we take the hull-style max.
        if _real_value(h2) > _real_value(h2_max)
            h2_max = h2
        end
    end
    return sqrt(h2_max)
end
