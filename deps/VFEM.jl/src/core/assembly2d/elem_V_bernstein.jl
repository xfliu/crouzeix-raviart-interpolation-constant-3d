# src/core/assembly2d/elem_V_bernstein.jl
#
# Port of vfem2d/lib/fem_assembly/elem_V_bernstein.m. Per-element
# degree-4 Bernstein control values of a 2D scalar potential V(x, y).
#
# Multi-index ordering on each triangle (a, b, c) with a + b + c = 4:
#   (4,0,0), (3,1,0), (3,0,1), (2,2,0), (2,1,1), (2,0,2),
#   (1,3,0), (1,2,1), (1,1,2), (1,0,3),
#   (0,4,0), (0,3,1), (0,2,2), (0,1,3), (0,0,4)
# — i.e. a descending, then b descending. The 15 control points are
# (a·P₁ + b·P₂ + c·P₃) / 4.

"""
    bernstein4_multiindices_2d() -> Matrix{Int}

Return the 15×3 matrix of degree-4 Bernstein multi-indices in
canonical order (`a` descending, then `b` descending). Used by
`elem_V_bernstein` and `create_matrix_lagrange`.
"""
function bernstein4_multiindices_2d()
    out = Matrix{Int}(undef, 15, 3)
    cur = 1
    @inbounds for a in 4:-1:0, b in (4 - a):-1:0
        out[cur, 1] = a
        out[cur, 2] = b
        out[cur, 3] = 4 - a - b
        cur += 1
    end
    return out
end

"""
    elem_V_bernstein(m::Mesh2D, V_func) -> Matrix{Float64}

Compute the `nt × 15` matrix of degree-4 Bernstein control values
of `V_func(x, y)` on each element of `m`. The Bernstein interpolant
is exact for polynomials of total degree ≤ 4.

`V_func` is called with `Float64` scalar arguments and must return
a real-coercible value.
"""
function elem_V_bernstein(m::Mesh2D, V_func)
    bern4 = bernstein4_multiindices_2d()
    out = Matrix{Float64}(undef, m.nt, 15)
    @inbounds for k in 1:m.nt
        v1 = (m.nodes[m.elements[k, 1], 1], m.nodes[m.elements[k, 1], 2])
        v2 = (m.nodes[m.elements[k, 2], 1], m.nodes[m.elements[k, 2], 2])
        v3 = (m.nodes[m.elements[k, 3], 1], m.nodes[m.elements[k, 3], 2])
        for r in 1:15
            a = bern4[r, 1]; b = bern4[r, 2]; c = bern4[r, 3]
            xq = (a * v1[1] + b * v2[1] + c * v3[1]) / 4
            yq = (a * v1[2] + b * v2[2] + c * v3[2]) / 4
            out[k, r] = Float64(V_func(xq, yq))
        end
    end
    return out
end
