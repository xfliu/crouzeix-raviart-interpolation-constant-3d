# src/core/mesh/get_facet_list.jl
#
# Port of VFEM3D/lib/mesh/get_FacetList.m. Given an `ElementList`
# (NumElt × 4, 1-based, rows sorted ascending), enumerate every
# distinct triangular facet as 3 sorted node indices.
#
# Algorithm: walk elements, generate the 4 sorted facet triples per
# tet, encode each as a unique integer key, and append on first
# occurrence. The MATLAB original encodes the key as
# `f1·M² + f2·M + f3` with `M = max(NodeIdx)`; we keep the same
# encoding so the row order is identical (downstream tests depend on
# it).

"""
    get_facet_list(ElementList::AbstractMatrix{<:Integer}) -> Matrix{Int}

Return the `NumF × 3` matrix of unique triangular facets across all
tetrahedra. Each row contains 3 sorted ascending node indices.
The output row order matches MATLAB `get_FacetList.m` exactly.
"""
function get_facet_list(ElementList::AbstractMatrix{<:Integer})
    size(ElementList, 2) == 4 ||
        throw(DimensionMismatch("ElementList must be NumElt × 4"))
    nelt = size(ElementList, 1)
    nelt == 0 && return Matrix{Int}(undef, 0, 3)

    M = maximum(ElementList)
    seen = Set{Int}()
    sizehint!(seen, 4 * nelt)
    facets = Matrix{Int}(undef, 4 * nelt, 3)
    cur = 1
    @inbounds for k in 1:nelt
        e = (Int(ElementList[k, 1]), Int(ElementList[k, 2]),
             Int(ElementList[k, 3]), Int(ElementList[k, 4]))
        a, b, c, d = e[1], e[2], e[3], e[4]
        a ≤ b ≤ c ≤ d || throw(ArgumentError("ElementList row $k must be sorted ascending"))
        local_facets = ((a, b, c), (a, b, d), (a, c, d), (b, c, d))
        for f in local_facets
            key = f[1] * M * M + f[2] * M + f[3]
            if key ∉ seen
                push!(seen, key)
                facets[cur, 1] = f[1]
                facets[cur, 2] = f[2]
                facets[cur, 3] = f[3]
                cur += 1
            end
        end
    end
    return facets[1:(cur - 1), :]
end
