# src/core/mesh/find_mesh_hmax_3d.jl
#
# Largest tetrahedron-edge length over the mesh. Port of the inline
# `compute_hmax_3d` helper inside `VFEM3D/schrodinger_eig_cecr_3d.m`.

"""
    find_mesh_hmax_3d(NodeList::AbstractMatrix, ElementList::AbstractMatrix)
        -> typeof(eltype(NodeList))

Maximum tetrahedron-edge length over `ElementList`. Iterates over all
six edges (i, j) with 1 ≤ i < j ≤ 4 of every element. Generic on the
node coordinate type so an `Interval{Float64}` `NodeList` produces an
interval enclosure of `h_max`.
"""
function find_mesh_hmax_3d(NodeList::AbstractMatrix, ElementList::AbstractMatrix)
    size(NodeList, 2) == 3 ||
        throw(DimensionMismatch("NodeList must be NumNode × 3"))
    size(ElementList, 2) == 4 ||
        throw(DimensionMismatch("ElementList must be NumElt × 4"))

    T = eltype(NodeList)
    h_max = zero(T)
    @inbounds for e in 1:size(ElementList, 1)
        v = (Int(ElementList[e, 1]), Int(ElementList[e, 2]),
             Int(ElementList[e, 3]), Int(ElementList[e, 4]))
        for i in 1:3, j in (i + 1):4
            dx = NodeList[v[i], 1] - NodeList[v[j], 1]
            dy = NodeList[v[i], 2] - NodeList[v[j], 2]
            dz = NodeList[v[i], 3] - NodeList[v[j], 3]
            h2 = dx * dx + dy * dy + dz * dz
            h  = sqrt(h2)
            if _hmax_compare_value(h) > _hmax_compare_value(h_max)
                h_max = h
            end
        end
    end
    return h_max
end

_hmax_compare_value(x::Interval) = sup(x)
_hmax_compare_value(x::Real) = x

"""
    find_mesh_hmax_3d(m::Mesh3D) -> Float64

Convenience overload that pulls `NodeList`, `ElementList` from the
`Mesh3D` struct.
"""
find_mesh_hmax_3d(m::Mesh3D) = find_mesh_hmax_3d(m.NodeList, m.ElementList)
