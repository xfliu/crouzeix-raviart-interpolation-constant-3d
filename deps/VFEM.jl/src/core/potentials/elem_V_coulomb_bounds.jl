# src/core/potentials/elem_V_coulomb_bounds.jl
#
# Port of VFEM3D/lib/eigensolve/potentials/elem_V_coulomb_bounds.m.
#
# Per-element min/max of the Coulomb potential V(x) = − Σ_c Z_c / |x − c_c|
# over a tetrahedron K. Since `−Z_c / |x − c_c|` is monotone in
# `|x − c_c|`, finding extrema of V on K reduces to finding the
# closest and farthest points of K from each center:
#   d_max = max over the 4 vertices (always at a vertex, since |·−c|
#           is convex and the farthest point of a convex polytope
#           from a fixed point is at a vertex).
#   d_min = closest distance from c to K, attained at a vertex, edge
#           interior, face interior, or 0 if c ∈ K.
#
# V_hat (least negative, "upper bound") = sum_c (−Z_c / d_max(c))
# V_bar (most negative, "lower bound")  = sum_c (−Z_c / d_min(c))
#                                       = −∞ if any singularity ∈ K.
#
# These bounds let the caller pick a Liu shift `γ_h` smaller than the
# trivial `max(−V_avg, 0)` while still guaranteeing positivity.

using LinearAlgebra: dot, cross, norm

# Test if `P` lies in the triangle `ABC` (assumed coplanar; the caller
# projects `c` onto the face plane first). Uses barycentric.
function _point_in_triangle(P::NTuple{3, Float64}, A::NTuple{3, Float64},
                             B::NTuple{3, Float64}, C::NTuple{3, Float64})
    v0 = (C[1] - A[1], C[2] - A[2], C[3] - A[3])
    v1 = (B[1] - A[1], B[2] - A[2], B[3] - A[3])
    v2 = (P[1] - A[1], P[2] - A[2], P[3] - A[3])
    d00 = v0[1] * v0[1] + v0[2] * v0[2] + v0[3] * v0[3]
    d01 = v0[1] * v1[1] + v0[2] * v1[2] + v0[3] * v1[3]
    d02 = v0[1] * v2[1] + v0[2] * v2[2] + v0[3] * v2[3]
    d11 = v1[1] * v1[1] + v1[2] * v1[2] + v1[3] * v1[3]
    d12 = v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3]
    denom = d00 * d11 - d01 * d01
    denom == 0 && return false
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return u ≥ -1e-12 && v ≥ -1e-12 && u + v ≤ 1 + 1e-12
end

# Test if `P` is inside the tetrahedron with `verts` (4×3) via barycentric.
function _point_in_tet(P::NTuple{3, Float64}, verts::AbstractMatrix{Float64})
    v0 = (verts[1, 1], verts[1, 2], verts[1, 3])
    Tmat = Float64[verts[2, 1] - v0[1]   verts[3, 1] - v0[1]   verts[4, 1] - v0[1] ;
                   verts[2, 2] - v0[2]   verts[3, 2] - v0[2]   verts[4, 2] - v0[2] ;
                   verts[2, 3] - v0[3]   verts[3, 3] - v0[3]   verts[4, 3] - v0[3]]
    rhs = Float64[P[1] - v0[1], P[2] - v0[2], P[3] - v0[3]]
    bary = Tmat \ rhs
    lam = (1 - sum(bary), bary[1], bary[2], bary[3])
    return all(λ -> λ ≥ -1e-12 && λ ≤ 1 + 1e-12, lam)
end

# Closest distance from a center `c` to the tetrahedron with given verts.
function _min_dist_to_tet(verts::AbstractMatrix{Float64}, c::NTuple{3, Float64})
    # Vertex distances.
    d_min = Inf
    @inbounds for v in 1:4
        dx = verts[v, 1] - c[1]
        dy = verts[v, 2] - c[2]
        dz = verts[v, 3] - c[3]
        d = sqrt(dx * dx + dy * dy + dz * dz)
        d < d_min && (d_min = d)
    end

    # Edge projections: the 6 edges of a tet.
    edges = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    @inbounds for (i, j) in edges
        P1 = (verts[i, 1], verts[i, 2], verts[i, 3])
        P2 = (verts[j, 1], verts[j, 2], verts[j, 3])
        d  = (P2[1] - P1[1], P2[2] - P1[2], P2[3] - P1[3])
        d2 = d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
        d2 > 0 || continue
        s = ((c[1] - P1[1]) * d[1] + (c[2] - P1[2]) * d[2] + (c[3] - P1[3]) * d[3]) / d2
        if s > 0 && s < 1
            Pf = (P1[1] + s * d[1], P1[2] + s * d[2], P1[3] + s * d[3])
            dist = sqrt((Pf[1] - c[1])^2 + (Pf[2] - c[2])^2 + (Pf[3] - c[3])^2)
            dist < d_min && (d_min = dist)
        end
    end

    # Face projections: the 4 faces of a tet.
    faces = ((2, 3, 4), (1, 3, 4), (1, 2, 4), (1, 2, 3))
    @inbounds for (i, j, k) in faces
        P1 = (verts[i, 1], verts[i, 2], verts[i, 3])
        P2 = (verts[j, 1], verts[j, 2], verts[j, 3])
        P3 = (verts[k, 1], verts[k, 2], verts[k, 3])
        a = (P2[1] - P1[1], P2[2] - P1[2], P2[3] - P1[3])
        b = (P3[1] - P1[1], P3[2] - P1[2], P3[3] - P1[3])
        n = (a[2] * b[3] - a[3] * b[2],
             a[3] * b[1] - a[1] * b[3],
             a[1] * b[2] - a[2] * b[1])
        n_len = sqrt(n[1] * n[1] + n[2] * n[2] + n[3] * n[3])
        n_len > 0 || continue
        nn = (n[1] / n_len, n[2] / n_len, n[3] / n_len)
        t = (c[1] - P1[1]) * nn[1] + (c[2] - P1[2]) * nn[2] + (c[3] - P1[3]) * nn[3]
        Pf = (c[1] - t * nn[1], c[2] - t * nn[2], c[3] - t * nn[3])
        if _point_in_triangle(Pf, P1, P2, P3)
            dist = abs(t)
            dist < d_min && (d_min = dist)
        end
    end

    # Interior: if c ∈ K, the distance is 0.
    if _point_in_tet(c, verts)
        d_min = 0.0
    end
    return d_min
end

"""
    elem_V_coulomb_bounds(m::Mesh3D, info::CoulombInfo)
        -> (V_bar::Vector{Float64}, V_hat::Vector{Float64})

Element-wise min/max of the Coulomb potential `V = − Σ_c Z_c / |x −
center_c|`. `V_bar[K]` (most negative) and `V_hat[K]` (least negative)
satisfy `V_bar[K] ≤ V(x) ≤ V_hat[K]` for all x ∈ K.

`V_bar` is `−Inf` for elements that contain a singularity (the closest
distance is 0 there). Callers needing a finite γ_h should pass an
override (e.g. the smallest finite V_bar over non-singular elements,
or an analytic estimate for the truncated domain).
"""
function elem_V_coulomb_bounds(m::Mesh3D, info::CoulombInfo)
    NumElt = m.NumElt
    V_bar = zeros(Float64, NumElt)
    V_hat = zeros(Float64, NumElt)

    @inbounds for ci in 1:size(info.centers, 1)
        c = (info.centers[ci, 1], info.centers[ci, 2], info.centers[ci, 3])
        Z = info.charges[ci]
        for e in 1:NumElt
            v1 = m.ElementList[e, 1]; v2 = m.ElementList[e, 2]
            v3 = m.ElementList[e, 3]; v4 = m.ElementList[e, 4]
            verts = Float64[m.NodeList[v1, 1] m.NodeList[v1, 2] m.NodeList[v1, 3];
                            m.NodeList[v2, 1] m.NodeList[v2, 2] m.NodeList[v2, 3];
                            m.NodeList[v3, 1] m.NodeList[v3, 2] m.NodeList[v3, 3];
                            m.NodeList[v4, 1] m.NodeList[v4, 2] m.NodeList[v4, 3]]
            d_max = 0.0
            for v in 1:4
                dx = verts[v, 1] - c[1]
                dy = verts[v, 2] - c[2]
                dz = verts[v, 3] - c[3]
                d = sqrt(dx * dx + dy * dy + dz * dz)
                d > d_max && (d_max = d)
            end
            d_min = _min_dist_to_tet(verts, c)

            V_hat[e] += -Z / d_max
            if d_min > 0
                V_bar[e] += -Z / d_min
            else
                V_bar[e] = -Inf
            end
        end
    end
    return V_bar, V_hat
end
