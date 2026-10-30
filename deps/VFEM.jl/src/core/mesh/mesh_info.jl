# src/core/mesh/mesh_info.jl
#
# Port of VFEM3D/lib/mesh/mesh_info.m. Returns mesh statistics
# relevant to the eigenvalue bounds (h_max for Liu's constant,
# volume range for assembly sanity).
#
# Differs from the MATLAB version in that it does not print —
# return the NamedTuple and let the caller format if needed.

using LinearAlgebra: norm, det

"""
    mesh_info(m::Mesh3D) -> NamedTuple

Compute and return mesh statistics:
* `h_max`, `h_min` — max/min edge length over all 6-edge tuples per element.
* `C_h` — Liu's 3D ECR constant `h_max / √10`.
* `vol_total`, `vol_min`, `vol_max` — total, min, max element volume.
* `domain_radius` — max norm of any node (useful for ball-domain pipelines).

The values match `mesh_info.m` exactly on a given mesh.
"""
function mesh_info(m::Mesh3D)
    NodeList = m.NodeList
    ElementList = m.ElementList

    h_max = 0.0
    h_min = Inf
    vol_total = 0.0
    vol_min = Inf
    vol_max = 0.0

    @inbounds for e in 1:m.NumElt
        v1 = (NodeList[ElementList[e, 1], 1],
              NodeList[ElementList[e, 1], 2],
              NodeList[ElementList[e, 1], 3])
        v2 = (NodeList[ElementList[e, 2], 1],
              NodeList[ElementList[e, 2], 2],
              NodeList[ElementList[e, 2], 3])
        v3 = (NodeList[ElementList[e, 3], 1],
              NodeList[ElementList[e, 3], 2],
              NodeList[ElementList[e, 3], 3])
        v4 = (NodeList[ElementList[e, 4], 1],
              NodeList[ElementList[e, 4], 2],
              NodeList[ElementList[e, 4], 3])

        verts = (v1, v2, v3, v4)
        for i in 1:4, j in (i + 1):4
            d = (verts[i][1] - verts[j][1],
                 verts[i][2] - verts[j][2],
                 verts[i][3] - verts[j][3])
            h = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
            h > h_max && (h_max = h)
            h < h_min && (h_min = h)
        end

        d1 = (v2[1] - v1[1], v2[2] - v1[2], v2[3] - v1[3])
        d2 = (v3[1] - v1[1], v3[2] - v1[2], v3[3] - v1[3])
        d3 = (v4[1] - v1[1], v4[2] - v1[2], v4[3] - v1[3])
        det_ = (d1[1] * (d2[2] * d3[3] - d2[3] * d3[2])
              - d1[2] * (d2[1] * d3[3] - d2[3] * d3[1])
              + d1[3] * (d2[1] * d3[2] - d2[2] * d3[1]))
        vol = abs(det_) / 6
        vol_total += vol
        vol < vol_min && (vol_min = vol)
        vol > vol_max && (vol_max = vol)
    end

    domain_radius = 0.0
    @inbounds for r in 1:m.NumNode
        rn = sqrt(NodeList[r, 1]^2 + NodeList[r, 2]^2 + NodeList[r, 3]^2)
        rn > domain_radius && (domain_radius = rn)
    end

    return (; h_max, h_min,
              C_h = h_max / sqrt(10.0),
              vol_total, vol_min, vol_max,
              domain_radius)
end
