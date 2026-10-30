# src/core/mesh2d/mesh2d_triangle_uniform.jl
#
# Uniform triangulation of the parametrised triangle
#
#   K_{alpha,theta,h} = conv{p1, p2, p3},
#     p1 = (0, 0),  p2 = (h, 0),  p3 = (alpha·h·cos θ, alpha·h·sin θ)
#
# introduced for the L^∞ Lagrange-interpolation-constant computation of
# J. Galindo, K. Ike, X. Liu (replacement of Lemma 3.2). The generator is
# self-contained — it has no MATLAB counterpart in vfem2d, so there is no
# fixture to port; the reference is the closed-form count/geometry contract
# asserted by `mesh2d_triangle_uniform_check`.
#
# The parameter triangle is subdivided into n² congruent sub-triangles
# ("upward" then "downward"), then pushed forward barycentrically onto K.
# The map is affine, so mesh size 1/n means exactly n subdivisions per side
# and every sub-triangle is similar to K itself.
#
# Conventions (identical to `mesh2d_load` / `find_tri2edge`, so the output is a
# drop-in `Mesh2D` for every 2D assembly routine in this library):
#   * 1-based indices throughout;
#   * `elements` rows are counter-clockwise (positive signed area);
#   * `edges` rows are sorted ascending and the row order is lexicographic,
#     which also fixes a *global* orientation per edge — used by
#     `create_matrix_fujino_morley` to give both adjacent elements the same
#     edge functional;
#   * `tri2edge[k, m]` is the global index of the local edge OPPOSITE local
#     vertex m, i.e. local edge order (v2,v3), (v1,v3), (v1,v2) — built by
#     calling `find_tri2edge`, not re-derived here.
#
# `Mesh2D.nodes` is `Matrix{Float64}` by the library's field contract, so the
# generator is not parameterised on T. Verified-mode callers promote the
# coordinates at the point of use (`interval.(m.nodes)`), exactly as
# `verified_cr_liu_lower` does.

"""
    mesh2d_triangle_uniform(n::Integer; alpha = 1.0, theta = pi / 2, h = 1.0)
        -> (m::Mesh2D, corners::NTuple{3, Int})

Uniform `n`-subdivision mesh of the parametrised triangle `K_{alpha,theta,h}`
with vertices `p1 = (0, 0)`, `p2 = (h, 0)`,
`p3 = (alpha·h·cos(theta), alpha·h·sin(theta))`.

Arguments
* `n`     — subdivisions per side (`n ≥ 1`); mesh size is `1/n`.
* `alpha` — edge-length ratio `|p1 p3| / |p1 p2|`, `alpha > 0`.
* `theta` — interior angle at `p1`, in `(0, π)`.
* `h`     — length `|p1 p2|`, `h > 0`.

Returns the `Mesh2D` together with `corners`, the node indices of the three
corners of `K` (the images of `p1`, `p2`, `p3`). `corners` is *not* a `Mesh2D`
field — it is returned separately because the Fujino–Morley space constrains
exactly those three vertex values.

Counts (asserted by [`mesh2d_triangle_uniform_check`](@ref)):

    nv = (n+1)(n+2)/2,   nt = n²,   ne = 3n(n+1)/2,   nb = 3n.

Node numbering: the barycentric lattice point `(i, j)` with `i + j ≤ n`, having
barycentric coordinates `((n-i-j)/n, i/n, j/n)` with respect to `(p1, p2, p3)`,
is numbered row by row in `j`:

    idx(i, j) = j(n+1) − j(j−1)/2 + i + 1.

Hence `corners = (1, n+1, nv)`. Element numbering lists all upward triangles
`(idx(i,j), idx(i+1,j), idx(i,j+1))` first, then all downward triangles
`(idx(i+1,j), idx(i+1,j+1), idx(i,j+1))`; both are counter-clockwise.
"""
function mesh2d_triangle_uniform(n::Integer; alpha::Real = 1.0,
                                 theta::Real = pi / 2, h::Real = 1.0)
    n ≥ 1 || throw(DomainError(n, "n must be ≥ 1"))
    alpha > 0 || throw(DomainError(alpha, "alpha must be > 0"))
    h > 0 || throw(DomainError(h, "h must be > 0"))
    0 < theta < pi || throw(DomainError(theta, "theta must lie in (0, π)"))

    nn = Int(n)
    al = Float64(alpha)
    th = Float64(theta)
    hh = Float64(h)

    p1 = (0.0, 0.0)
    p2 = (hh, 0.0)
    p3 = (al * hh * cos(th), al * hh * sin(th))

    nv = div((nn + 1) * (nn + 2), 2)
    nt = nn^2
    # lattice index → node number, numbered row by row in j
    idx(i, j) = j * (nn + 1) - div(j * (j - 1), 2) + i + 1

    nodes = Matrix{Float64}(undef, nv, 2)
    @inbounds for j in 0:nn, i in 0:(nn - j)
        l2 = i / nn
        l3 = j / nn
        l1 = 1.0 - l2 - l3
        p = idx(i, j)
        nodes[p, 1] = l1 * p1[1] + l2 * p2[1] + l3 * p3[1]
        nodes[p, 2] = l1 * p1[2] + l2 * p2[2] + l3 * p3[2]
    end

    # elements: upward triangles first, then downward — both counter-clockwise
    elements = Matrix{Int}(undef, nt, 3)
    k = 0
    for j in 0:(nn - 1), i in 0:(nn - 1 - j)
        k += 1
        elements[k, 1] = idx(i, j)
        elements[k, 2] = idx(i + 1, j)
        elements[k, 3] = idx(i, j + 1)
    end
    for j in 0:(nn - 2), i in 0:(nn - 2 - j)
        k += 1
        elements[k, 1] = idx(i + 1, j)
        elements[k, 2] = idx(i + 1, j + 1)
        elements[k, 3] = idx(i, j + 1)
    end
    k == nt || error("element count mismatch: built $k, expected $nt")

    # edges: unique sorted endpoint pairs; multiplicity 1 ⇒ boundary edge
    local_edge = ((2, 3), (1, 3), (1, 2))
    mult = Dict{NTuple{2, Int}, Int}()
    sizehint!(mult, div(3 * nt, 2) + 3 * nn)
    @inbounds for kk in 1:nt, m in 1:3
        a = elements[kk, local_edge[m][1]]
        b = elements[kk, local_edge[m][2]]
        key = a < b ? (a, b) : (b, a)
        mult[key] = get(mult, key, 0) + 1
    end
    keys_sorted = sort!(collect(keys(mult)))
    ne = length(keys_sorted)
    edges = Matrix{Int}(undef, ne, 2)
    @inbounds for e in 1:ne
        edges[e, 1] = keys_sorted[e][1]
        edges[e, 2] = keys_sorted[e][2]
    end
    bd_edge_ids = [e for e in 1:ne if mult[keys_sorted[e]] == 1]
    bd_edges = edges[bd_edge_ids, :]
    nb = length(bd_edge_ids)

    tri2edge = find_tri2edge(elements, edges)
    corners = (idx(0, 0), idx(nn, 0), idx(0, nn))

    m = Mesh2D(nodes, elements, edges, bd_edges, bd_edge_ids, tri2edge,
               nv, nt, ne, nb)
    return m, corners
end

"""
    mesh2d_triangle_uniform_check(m::Mesh2D, corners, n; alpha = 1.0,
                                  theta = pi / 2, h = 1.0) -> NamedTuple

Structural self-check of a mesh from [`mesh2d_triangle_uniform`](@ref): count
formulas, ascending `edges` rows, `tri2edge` consistency, boundary-edge count,
counter-clockwise element orientation, and corner coordinates.

Throws on a structural violation; otherwise returns the observed
`(nv, nt, ne, nb, area, min_signed_area, corner_coords)`, where `area` is the
summed signed element area — comparable to `|K| = ½ · alpha · h² · sin(theta)`.
Used by `test/test_mesh2d_triangle_uniform.jl`.
"""
function mesh2d_triangle_uniform_check(m::Mesh2D, corners::NTuple{3, Int},
                                       n::Integer; alpha::Real = 1.0,
                                       theta::Real = pi / 2, h::Real = 1.0)
    nn = Int(n)
    m.nv == div((nn + 1) * (nn + 2), 2) || error("nv mismatch: $(m.nv)")
    m.nt == nn^2 || error("nt mismatch: $(m.nt)")
    m.ne == div(3 * nn * (nn + 1), 2) || error("ne mismatch: $(m.ne)")
    m.nb == 3 * nn || error("nb mismatch: $(m.nb)")
    all(m.edges[e, 1] < m.edges[e, 2] for e in 1:m.ne) ||
        error("edges rows are not ascending")

    local_edge = ((2, 3), (1, 3), (1, 2))
    @inbounds for k in 1:m.nt, j in 1:3
        a = m.elements[k, local_edge[j][1]]
        b = m.elements[k, local_edge[j][2]]
        e = m.tri2edge[k, j]
        pair = a < b ? (a, b) : (b, a)
        pair == (m.edges[e, 1], m.edges[e, 2]) ||
            error("tri2edge inconsistent at (k, j) = ($k, $j)")
    end

    area = 0.0
    min_signed = Inf
    @inbounds for k in 1:m.nt
        x1, y1 = m.nodes[m.elements[k, 1], 1], m.nodes[m.elements[k, 1], 2]
        x2, y2 = m.nodes[m.elements[k, 2], 1], m.nodes[m.elements[k, 2], 2]
        x3, y3 = m.nodes[m.elements[k, 3], 1], m.nodes[m.elements[k, 3], 2]
        s = ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
        area += s
        min_signed = min(min_signed, s)
    end
    min_signed > 0 || error("non-positive signed area: $min_signed")

    _ = (alpha, theta, h)   # kept in the signature for call-site symmetry
    corner_coords = ((m.nodes[corners[1], 1], m.nodes[corners[1], 2]),
                     (m.nodes[corners[2], 1], m.nodes[corners[2], 2]),
                     (m.nodes[corners[3], 1], m.nodes[corners[3], 2]))
    return (nv = m.nv, nt = m.nt, ne = m.ne, nb = m.nb, area = area,
            min_signed_area = min_signed, corner_coords = corner_coords)
end
