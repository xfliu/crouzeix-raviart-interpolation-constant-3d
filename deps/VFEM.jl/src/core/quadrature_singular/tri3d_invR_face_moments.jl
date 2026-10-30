# src/core/quadrature_singular/tri3d_invR_face_moments.jl
#
# Port of VFEM3D/lib/quadrature/tri3d_invR_face_moments.m.
#
# Computes
#   F[b1+1, b2+1, b3+1] = ∫_Δ μ₁^{b1} μ₂^{b2} μ₃^{b3} / |μ₁·a + μ₂·b + μ₃·c| dμ
# for b1+b2+b3 ≤ max_degree, where (a, b, c) are the face triangle's
# vertices given relative to the singular tetrahedron vertex.
#
# Method (no quadrature):
#   1. Project the 3D triangle onto its own plane in 2D (verts2), with
#      the singular vertex at perpendicular distance h above the plane.
#   2. Compute J[p+1, q+1] = ∫_T x^p y^q / √(x²+y²+h²) dxdy in closed
#      form via planar potential identities and a divergence-theorem
#      recurrence.
#   3. Express each barycentric μ_i as an affine function of (x, y),
#      raise to the requested power, convolve, and combine with J.
#
# All helpers below are private to this file.

using LinearAlgebra: norm, dot, cross, det, I

# 1D coefficient power: (c0 + c1·t)^p as ascending coefficients.
function _poly_power_linear_local(c0::T, c1::T, p::Integer) where {T<:Real}
    coeff = Vector{T}(undef, p + 1)
    @inbounds for k in 0:p
        coeff[k + 1] = T(binomial(p, k)) * c0^(p - k) * c1^k
    end
    return coeff
end

# 2D polynomial convolution. P is a matrix; P[i+1, j+1] is the
# coefficient of x^i y^j.
function _poly2_conv(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:Real}
    Ax, Ay = size(A)
    Bx, By = size(B)
    C = zeros(T, Ax + Bx - 1, Ay + By - 1)
    @inbounds for ax in 1:Ax, ay in 1:Ay
        a = A[ax, ay]
        iszero(a) && continue
        for bx in 1:Bx, by in 1:By
            b = B[bx, by]
            iszero(b) && continue
            C[ax + bx - 1, ay + by - 1] += a * b
        end
    end
    return C
end

# Coefficients of (c0 + c1·x + c2·y)^p as a (p+1) × (p+1) lower-triangular
# matrix C with C[i+1, j+1] = coefficient of x^i y^j.
function _poly2_power_affine(c::AbstractVector{T}, p::Integer) where {T<:Real}
    P = ones(T, 1, 1)
    L = zeros(T, 2, 2)
    L[1, 1] = c[1]; L[2, 1] = c[2]; L[1, 2] = c[3]
    @inbounds for _ in 1:p
        P = _poly2_conv(P, L)
    end
    return P
end

# Antiderivative-based H(1) = ∫_0^1 √(at² + bt + c) dt.
# Recurrence for H(k+1) = ∫_0^1 t^k √(at²+bt+c) dt for k ≥ 1.
function _sqrt_quadratic_moments(a::T, b::T, c::T, max_degree::Integer) where {T<:Real}
    am = _real_value(a)
    am > 1e-14 || throw(DomainError(a, "Degenerate edge: leading coefficient too small"))

    H = Vector{T}(undef, max_degree + 1)
    Δ = 4 * a * c - b * b
    Δm = _real_value(Δ)
    Δm > 0 || throw(DomainError(Δ, "Expected positive quadratic discriminant"))

    sqrtΔ = sqrt(Δ)
    sqrt_a = sqrt(a)

    # Antiderivative at t (closed form)
    function antider(t::T)
        q = a * t * t + b * t + c
        return ((2 * a * t + b) * sqrt(q) / (4 * a)
                + Δ / (8 * sqrt_a^3) * asinh((2 * a * t + b) / sqrtΔ))
    end
    H[1] = antider(one(T)) - antider(zero(T))

    q0 = c
    q1 = a + b + c
    @inbounds for n in 1:max_degree
        if n == 1
            endpoint = q1 * sqrt(q1) - q0 * sqrt(q0)
            lower = zero(T)
        else
            endpoint = q1 * sqrt(q1)
            lower = c * T(n - 1) * H[n - 1]
        end
        H[n + 1] = (endpoint - b * (T(n) + T(1) / T(2)) * H[n] - lower) /
                   (a * T(n + 2))
    end
    return H
end

# Edge integral ∫_0^1 x(t)^px y(t)^py √(x(t)²+y(t)²+h²) dt
# where x(t) = A[1] + (B[1]-A[1])·t, y(t) = A[2] + (B[2]-A[2])·t.
function _edge_poly_R_integral(A::AbstractVector{T}, B::AbstractVector{T},
                               h::T, px::Integer, py::Integer) where {T<:Real}
    d = B .- A
    qa = dot(d, d)
    qb = 2 * dot(A, d)
    qc = dot(A, A) + h * h
    cx = _poly_power_linear_local(A[1], d[1], px)
    cy = _poly_power_linear_local(A[2], d[2], py)
    poly = _polymul_1d(cx, cy)
    H = _sqrt_quadratic_moments(qa, qb, qc, length(poly) - 1)
    s = zero(T)
    @inbounds for k in eachindex(poly)
        s += poly[k] * H[k]
    end
    return s
end

# 1D polynomial convolution (same as our top-level polymul, replicated
# here to keep this file self-contained for now).
function _polymul_1d(p::AbstractVector{T}, q::AbstractVector{T}) where {T<:Real}
    m = length(p) - 1
    n = length(q) - 1
    r = zeros(T, m + n + 1)
    @inbounds for i in 0:m, j in 0:n
        r[i + j + 1] += p[i + 1] * q[j + 1]
    end
    return r
end

# Boundary integral of x^px y^py R · n_component ds over the triangle's
# CCW edges. Component = 1 → n_x ds = e_y dt, Component = 2 → n_y ds = -e_x dt.
function _boundary_R_moment(verts::AbstractMatrix{T}, h::T,
                            px::Integer, py::Integer, component::Integer) where {T<:Real}
    val = zero(T)
    @inbounds for i in 1:3
        A = verts[i, :]
        B = verts[mod(i, 3) + 1, :]
        e = B .- A
        edge_val = _edge_poly_R_integral(A, B, h, px, py)
        if component == 1
            val += e[2] * edge_val
        else
            val -= e[1] * edge_val
        end
    end
    return val
end

# Solid angle subtended at the singular vertex by the planar triangle
# at height h, via the Van Oosterom–Strang formula.
function _solid_angle_projected_triangle(verts::AbstractMatrix{T}, h::T) where {T<:Real}
    A = (verts[1, 1], verts[1, 2], h)
    B = (verts[2, 1], verts[2, 2], h)
    C = (verts[3, 1], verts[3, 2], h)
    RA = sqrt(A[1]^2 + A[2]^2 + A[3]^2)
    RB = sqrt(B[1]^2 + B[2]^2 + B[3]^2)
    RC = sqrt(C[1]^2 + C[2]^2 + C[3]^2)
    numer = (A[1] * (B[2] * C[3] - B[3] * C[2])
           - A[2] * (B[1] * C[3] - B[3] * C[1])
           + A[3] * (B[1] * C[2] - B[2] * C[1]))
    denom = (RA * RB * RC
           + (A[1] * B[1] + A[2] * B[2] + A[3] * B[3]) * RC
           + (B[1] * C[1] + B[2] * C[2] + B[3] * C[3]) * RA
           + (C[1] * A[1] + C[2] * A[2] + C[3] * A[3]) * RB)
    return 2 * atan(numer, denom)
end

# Closed-form planar potential I00 = ∫_T 1/√(x²+y²+h²) dxdy for a triangle
# T in 2D and a singular point at perpendicular distance h.
function _tri2d_invR_I00(verts::AbstractMatrix{T}, h::T) where {T<:Real}
    S = zero(T)
    @inbounds for i in 1:3
        A = verts[i, :]
        B = verts[mod(i, 3) + 1, :]
        e = B .- A
        L = sqrt(e[1]^2 + e[2]^2)
        Lm = _real_value(L)
        Lm > 1e-14 || throw(DomainError(L, "Degenerate edge"))
        outward = (e[2] / L, -e[1] / L)
        signed_dist = A[1] * outward[1] + A[2] * outward[2]
        RA = sqrt(A[1]^2 + A[2]^2 + h * h)
        RB = sqrt(B[1]^2 + B[2]^2 + h * h)
        log_arg = (RA + RB + L) / (RA + RB - L)
        S += signed_dist * log(log_arg)
    end
    Ω = _solid_angle_projected_triangle(verts, h)
    return S - h * Ω
end

# Higher-degree planar moments J[p+1, q+1] = ∫_T x^p y^q / R dxdy via a
# linear system at each total degree d, derived from the divergence theorem.
function _tri2d_invR_monomial_moments(verts::AbstractMatrix{T}, h::T,
                                       max_degree::Integer) where {T<:Real}
    # Verify orientation: positive 2x area required.
    e1 = verts[2, :] .- verts[1, :]
    e2 = verts[3, :] .- verts[1, :]
    area2 = e1[1] * e2[2] - e1[2] * e2[1]
    abs(_real_value(area2)) > 1e-14 || throw(DomainError(area2, "Degenerate 2D triangle"))
    if _real_value(area2) < 0
        # Reorder rows so the triangle is CCW. Make a copy so we don't mutate
        # the caller's matrix.
        v = similar(verts)
        v[1, :] = verts[1, :]
        v[2, :] = verts[3, :]
        v[3, :] = verts[2, :]
        verts = v
    end

    J = zeros(T, max_degree + 1, max_degree + 1)
    J[1, 1] = _tri2d_invR_I00(verts, h)

    @inbounds for d in 1:max_degree
        neq = 2 * d
        Aeq = zeros(T, neq, d + 1)
        beq = zeros(T, neq)
        row = 0

        for p in 0:d
            q = d - p
            if p ≥ 1
                row += 1
                Aeq[row, p + 1] = T(p)
                boundary = _boundary_R_moment(verts, h, p - 1, q, 1)
                if p ≥ 2
                    Aeq[row, p - 1] = Aeq[row, p - 1] + T(p - 1)
                    beq[row] = boundary - T(p - 1) * h * h * J[p - 1, q + 1]
                else
                    beq[row] = boundary
                end
            end
        end

        for p in 0:d
            q = d - p
            if q ≥ 1
                row += 1
                Aeq[row, p + 1] = T(q)
                boundary = _boundary_R_moment(verts, h, p, q - 1, 2)
                if q ≥ 2
                    Aeq[row, p + 3] = Aeq[row, p + 3] + T(q - 1)
                    beq[row] = boundary - T(q - 1) * h * h * J[p + 1, q - 1]
                else
                    beq[row] = boundary
                end
            end
        end

        sol = Aeq \ beq

        for p in 0:d
            q = d - p
            J[p + 1, q + 1] = sol[p + 1]
        end
    end
    return J
end

"""
    tri3d_invR_face_moments(face_nodes::AbstractMatrix, max_degree::Integer)
        -> (F::Array{T,3}, info::NamedTuple)

Exact face moments for the shifted Coulomb kernel on a triangle in
ℝ³. `face_nodes` is the 3×3 matrix of face vertices measured relative
to the singular tetrahedron vertex P_s. Returns the array `F` with

    F[b1+1, b2+1, b3+1] = ∫_Δ μ₁^{b1} μ₂^{b2} μ₃^{b3} / |μ₁·a + μ₂·b + μ₃·c| dμ

for `b1 + b2 + b3 ≤ max_degree`. No quadrature.

`info` is a NamedTuple with fields `h`, `area_jac`, `verts2`,
`lambda_coeff` for diagnostics — same names as the MATLAB struct.
"""
function tri3d_invR_face_moments(face_nodes::AbstractMatrix{T},
                                 max_degree::Integer) where {T<:Real}
    size(face_nodes) == (3, 3) ||
        throw(DimensionMismatch("face_nodes must be 3×3 (got $(size(face_nodes)))"))
    max_degree ≥ 0 || throw(DomainError(max_degree, "max_degree must be ≥ 0"))

    a = face_nodes[1, :]
    b = face_nodes[2, :]
    c = face_nodes[3, :]

    n_raw = cross(b .- a, c .- a)
    area_jac = norm(n_raw)
    abs(_real_value(area_jac)) > 1e-14 ||
        throw(DomainError(area_jac, "Degenerate face: area too small"))

    n = n_raw ./ area_jac
    h = dot(a, n)
    if _real_value(h) < 0
        n = -n
        h = -h
    end
    abs(_real_value(h)) > 1e-14 ||
        throw(DomainError(h, "Degenerate tet: singular vertex lies in opposite face plane"))

    e1 = (b .- a) ./ norm(b .- a)
    e2 = cross(n, e1)
    projection = h .* n

    verts2 = Matrix{T}(undef, 3, 2)
    @inbounds for i in 1:3
        r = face_nodes[i, :] .- projection
        verts2[i, 1] = dot(r, e1)
        verts2[i, 2] = dot(r, e2)
    end

    J = _tri2d_invR_monomial_moments(verts2, h, max_degree)

    # Solve A · λ = I, where A = [1 x y] rows for each vertex. λ_coeff
    # columns are the (constant, x, y) coefficients of μ_i(x, y).
    A = [ones(T, 3) verts2]
    λ_coeff = A \ Matrix{T}(I, 3, 3)

    F = fill(T(NaN), max_degree + 1, max_degree + 1, max_degree + 1)
    @inbounds for b1 in 0:max_degree, b2 in 0:(max_degree - b1),
                  b3 in 0:(max_degree - b1 - b2)
        P1 = _poly2_power_affine(λ_coeff[:, 1], b1)
        P2 = _poly2_power_affine(λ_coeff[:, 2], b2)
        P3 = _poly2_power_affine(λ_coeff[:, 3], b3)
        P  = _poly2_conv(_poly2_conv(P1, P2), P3)

        val = zero(T)
        for px in 0:(size(P, 1) - 1), py in 0:(size(P, 2) - 1)
            (px + py) ≤ max_degree || continue
            val += P[px + 1, py + 1] * J[px + 1, py + 1]
        end
        F[b1 + 1, b2 + 1, b3 + 1] = val / area_jac
    end

    info = (; h, area_jac, verts2, lambda_coeff = λ_coeff)
    return F, info
end
