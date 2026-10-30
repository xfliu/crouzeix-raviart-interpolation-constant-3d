# src/core/mesh/facet_id_at_element.jl
#
# Port of VFEM3D/lib/mesh/get_facet_id_at_element.m. Given a facet's
# 3 sorted node indices and an element's 4 sorted node indices, find
# the local face id ∈ 1:4 (= the index of the vertex *missing* from
# the facet).
#
# MATLAB trick: `find(sum(elt) - sum(facet) == elt_entry)` selects
# the unique vertex k such that `sum(elt) - sum(facet) == elt[k]`,
# which is exactly the missing vertex.

"""
    facet_id_at_element(facet_nodes::AbstractVector{<:Integer},
                        element_nodes::AbstractVector{<:Integer}) -> Int

Local face index ∈ 1:4 of `facet_nodes` (length 3) within the
tetrahedron `element_nodes` (length 4, sorted ascending). Equals the
index k such that `element_nodes[k]` is the unique vertex absent
from `facet_nodes`.
"""
function facet_id_at_element(facet_nodes::AbstractVector{<:Integer},
                             element_nodes::AbstractVector{<:Integer})
    length(facet_nodes) == 3 ||
        throw(DimensionMismatch("facet_nodes must have length 3"))
    length(element_nodes) == 4 ||
        throw(DimensionMismatch("element_nodes must have length 4"))
    target = sum(element_nodes) - sum(facet_nodes)
    @inbounds for k in 1:4
        element_nodes[k] == target && return k
    end
    throw(ArgumentError("facet not found inside element"))
end
