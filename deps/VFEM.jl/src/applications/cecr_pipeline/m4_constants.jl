# src/applications/cecr_pipeline/m4_constants.jl
#
# Step m4: compute mesh-dependent constants C_h^PW, Γ_h, A_h.
# Port of Code_Sorted/modules/m4_compute_constants/m4_compute_constants.m.
#
# h_K = max edge length of element K.
# C_h^PW = h_max / π.
# Γ_h = max_K c_h^-(K) · (h_K/π)²  where c_h^-(K) = max(-c_h[K], 0).
# A_h = Γ_h / (1-ε).

using LinearAlgebra: norm

function _elem_diameters_2d(m::Mesh2D)
    h_K = Vector{Float64}(undef, m.nt)
    @inbounds for k in 1:m.nt
        v1, v2, v3 = m.elements[k, 1], m.elements[k, 2], m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        d12 = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        d23 = sqrt((x3 - x2)^2 + (y3 - y2)^2)
        d31 = sqrt((x1 - x3)^2 + (y1 - y3)^2)
        h_K[k] = max(d12, d23, d31)
    end
    return h_K
end

function _elem_diameters_3d(m::Mesh3D)
    h_K = Vector{Float64}(undef, m.NumElt)
    @inbounds for k in 1:m.NumElt
        v1, v2, v3, v4 = m.ElementList[k, 1], m.ElementList[k, 2],
                         m.ElementList[k, 3], m.ElementList[k, 4]
        verts = (m.NodeList[v1, :], m.NodeList[v2, :],
                 m.NodeList[v3, :], m.NodeList[v4, :])
        d_max = 0.0
        for i in 1:4, j in (i + 1):4
            d = sqrt(sum((verts[i][c] - verts[j][c])^2 for c in 1:3))
            d_max = max(d_max, d)
        end
        h_K[k] = d_max
    end
    return h_K
end

"""
    compute_mesh_constants(m::Mesh2D, c_h, epsilon) -> MeshConstants
    compute_mesh_constants(m::Mesh3D, c_h, epsilon) -> MeshConstants

Compute mesh-dependent constants C_h^PW, Γ_h, A_h from the per-element
reaction coefficients `c_h` and form-bound coefficient `epsilon`.
"""
function compute_mesh_constants(m::Mesh2D, c_h::AbstractVector{Float64},
                                epsilon::Float64)
    h_K = _elem_diameters_2d(m)
    h_max = maximum(h_K)
    Ch_PW = h_max / π
    Gamma_h = maximum(max(0.0, -c_h[k]) * (h_K[k] / π)^2 for k in 1:m.nt)
    A_h = Gamma_h / (1.0 - epsilon)
    return MeshConstants(Ch_PW, Gamma_h, A_h, h_max, h_K)
end

function compute_mesh_constants(m::Mesh3D, c_h::AbstractVector{Float64},
                                epsilon::Float64)
    h_K = _elem_diameters_3d(m)
    h_max = maximum(h_K)
    Ch_PW = h_max / π
    Gamma_h = maximum(max(0.0, -c_h[k]) * (h_K[k] / π)^2 for k in 1:m.NumElt)
    A_h = Gamma_h / (1.0 - epsilon)
    return MeshConstants(Ch_PW, Gamma_h, A_h, h_max, h_K)
end
