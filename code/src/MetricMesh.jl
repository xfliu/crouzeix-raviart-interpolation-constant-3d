module MetricMesh

using LinearAlgebra
using VFEM: Mesh3D, get_facet_list, get_edge_list,
            facet_element_connectivity

export metric_red_refine_mesh_3d, metric_hmax2

"""
    metric_hmax2(mesh, H)

Return the largest squared edge length in `mesh` for the constant metric
`H`, i.e. `max (x_i-x_j)'H(x_i-x_j)`.  The node coordinates are reference
coordinates; consequently `H` is the vertex-edge Gram matrix of the affine
physical tetrahedron.
"""
function metric_hmax2(mesh::Mesh3D, H::AbstractMatrix{<:Real})
    metric = _checked_metric(H)
    h2 = 0.0
    @inbounds for edge in axes(mesh.EdgeList, 1)
        i, j = mesh.EdgeList[edge, 1], mesh.EdgeList[edge, 2]
        d1 = mesh.NodeList[i, 1] - mesh.NodeList[j, 1]
        d2 = mesh.NodeList[i, 2] - mesh.NodeList[j, 2]
        d3 = mesh.NodeList[i, 3] - mesh.NodeList[j, 3]
        q = _metric_length2(d1, d2, d3, metric)
        h2 = max(h2, q)
    end
    return h2
end

"""
    metric_red_refine_mesh_3d(mesh, H) -> Mesh3D

Uniformly red-refine a tetrahedral mesh, choosing the central-octahedron
diagonal separately in every parent tetrahedron.  Of the three possible
diagonals, the one shortest in the constant physical metric `H` is used.
The candidate order

```
(m12,m34), (m13,m24), (m14,m23)
```

also defines deterministic first-candidate tie breaking.

Only the triangulation of the octahedron in the interior of a parent is
changed.  Its four face subdivisions are therefore independent of this
choice, so adjacent parents remain conforming and each original boundary
triangle is divided into four equal-area triangles.  All `Mesh3D` derived
connectivity and count fields are rebuilt before returning.

`H` must be a finite, symmetric positive-definite `3 x 3` matrix of
midpoint values.  Interval uncertainty belongs in the later matrix and
diameter certification, not in this combinatorial diagonal decision.
"""
function metric_red_refine_mesh_3d(mesh::Mesh3D,
                                   H::AbstractMatrix{<:Real})
    metric = _checked_metric(H)
    size(mesh.NodeList, 2) == 3 ||
        throw(DimensionMismatch("Mesh3D nodes must have three coordinates"))

    # All parents sharing an edge must share its midpoint.  Enumerating the
    # already canonical global edge list also makes node numbering stable.
    edge_to_midpoint = Dict{Tuple{Int, Int}, Int}()
    nodes = Matrix{Float64}(undef, mesh.NumNode + mesh.NumEdge, 3)
    nodes[1:mesh.NumNode, :] .= mesh.NodeList
    @inbounds for edge_id in 1:mesh.NumEdge
        i, j = mesh.EdgeList[edge_id, 1], mesh.EdgeList[edge_id, 2]
        key = minmax(i, j)
        midpoint = mesh.NumNode + edge_id
        edge_to_midpoint[key] = midpoint
        nodes[midpoint, 1] = (mesh.NodeList[i, 1] + mesh.NodeList[j, 1]) / 2
        nodes[midpoint, 2] = (mesh.NodeList[i, 2] + mesh.NodeList[j, 2]) / 2
        nodes[midpoint, 3] = (mesh.NodeList[i, 3] + mesh.NodeList[j, 3]) / 2
    end

    elements = Matrix{Int}(undef, 8 * mesh.NumElt, 4)
    row = 0
    @inbounds for element_id in 1:mesh.NumElt
        v1, v2, v3, v4 = mesh.ElementList[element_id, :]
        m12 = edge_to_midpoint[minmax(v1, v2)]
        m13 = edge_to_midpoint[minmax(v1, v3)]
        m14 = edge_to_midpoint[minmax(v1, v4)]
        m23 = edge_to_midpoint[minmax(v2, v3)]
        m24 = edge_to_midpoint[minmax(v2, v4)]
        m34 = edge_to_midpoint[minmax(v3, v4)]

        # The four corner tetrahedra do not depend on the octahedron split.
        row = _put_child!(elements, row, v1, m12, m13, m14)
        row = _put_child!(elements, row, v2, m12, m23, m24)
        row = _put_child!(elements, row, v3, m13, m23, m34)
        row = _put_child!(elements, row, v4, m14, m24, m34)

        diagonals = ((m12, m34), (m13, m24), (m14, m23))
        lengths2 = ntuple(k -> begin
            i, j = diagonals[k]
            d1 = nodes[i, 1] - nodes[j, 1]
            d2 = nodes[i, 2] - nodes[j, 2]
            d3 = nodes[i, 3] - nodes[j, 3]
            _metric_length2(d1, d2, d3, metric)
        end, 3)

        # Strict comparison deliberately keeps the earliest candidate on an
        # exact tie.  This is deterministic on every Julia platform.
        choice = 1
        lengths2[2] < lengths2[choice] && (choice = 2)
        lengths2[3] < lengths2[choice] && (choice = 3)

        if choice == 1
            # diagonal m12--m34; cyclic equator m13,m14,m24,m23
            row = _put_child!(elements, row, m12, m34, m13, m14)
            row = _put_child!(elements, row, m12, m34, m14, m24)
            row = _put_child!(elements, row, m12, m34, m24, m23)
            row = _put_child!(elements, row, m12, m34, m23, m13)
        elseif choice == 2
            # diagonal m13--m24; cyclic equator m12,m14,m34,m23
            row = _put_child!(elements, row, m13, m24, m12, m14)
            row = _put_child!(elements, row, m13, m24, m14, m34)
            row = _put_child!(elements, row, m13, m24, m34, m23)
            row = _put_child!(elements, row, m13, m24, m23, m12)
        else
            # diagonal m14--m23; cyclic equator m12,m13,m34,m24
            row = _put_child!(elements, row, m14, m23, m12, m13)
            row = _put_child!(elements, row, m14, m23, m13, m34)
            row = _put_child!(elements, row, m14, m23, m34, m24)
            row = _put_child!(elements, row, m14, m23, m24, m12)
        end
    end
    row == size(elements, 1) || error("internal child-count mismatch")
    return _complete_mesh(nodes, elements)
end

@inline function _put_child!(elements::Matrix{Int}, row::Int,
                             a::Int, b::Int, c::Int, d::Int)
    row += 1
    child = sort((a, b, c, d))
    elements[row, 1] = child[1]
    elements[row, 2] = child[2]
    elements[row, 3] = child[3]
    elements[row, 4] = child[4]
    return row
end

@inline function _metric_length2(d1::Float64, d2::Float64, d3::Float64,
                                 H::Matrix{Float64})
    # This expanded symmetric form evaluates each off-diagonal contribution
    # once and is shared by diagonal selection and h_max diagnostics.
    return H[1, 1] * d1 * d1 + H[2, 2] * d2 * d2 + H[3, 3] * d3 * d3 +
           2H[1, 2] * d1 * d2 + 2H[1, 3] * d1 * d3 +
           2H[2, 3] * d2 * d3
end

function _checked_metric(H::AbstractMatrix{<:Real})
    size(H) == (3, 3) ||
        throw(DimensionMismatch("metric must be a 3 x 3 matrix"))
    metric = Matrix{Float64}(H)
    all(isfinite, metric) || throw(ArgumentError("metric entries must be finite"))
    issymmetric(metric) || throw(ArgumentError("metric must be symmetric"))
    isposdef(Symmetric(metric)) ||
        throw(ArgumentError("metric must be positive definite"))
    return metric
end

function _complete_mesh(nodes::Matrix{Float64}, elements::Matrix{Int})
    facets = get_facet_list(elements)
    edges = get_edge_list(elements)
    facet_to_element, element_to_facet =
        facet_element_connectivity(elements, facets)
    return Mesh3D(nodes, elements, facets, edges,
                  facet_to_element, element_to_facet,
                  size(nodes, 1), size(elements, 1),
                  size(facets, 1), size(edges, 1))
end

end # module MetricMesh
