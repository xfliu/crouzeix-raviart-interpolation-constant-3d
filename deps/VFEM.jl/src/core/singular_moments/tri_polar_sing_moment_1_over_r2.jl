# src/core/singular_moments/tri_polar_sing_moment_1_over_r2.jl
#
# Closed-form value of  ∫_K (1/r²) x^m y^n dxdy  on a triangle K with
# vertices (0,0), (r1,t1), (r2,t2). Supports 1 ≤ m+n ≤ 4. The case
# m+n = 0 is rejected: ∫ 1/r² dA diverges when the origin is at a
# vertex of K. Port of matlab_lib/tri_polar_sing_moment_1_over_r2.m.
#
# Derivation:
#   With the same parameterisation as the 1/r case,
#       ∫_K (1/r²) x^m y^n dxdy
#     = det(v1,v2)/(m+n) · ∫_0^1 wx(τ)^m wy(τ)^n / q(τ) dτ
#   for m+n ≥ 1. The 1D integrals K_k = ∫_0^1 τ^k / q(τ) dτ satisfy
#       K_0 = 2/√Δ · [atan((2a+b)/√Δ) − atan(b/√Δ)]
#       K_1 = (1/a) ln(r2/r1) − (b/(2a)) K_0
#       K_k = 1/(a(k−1)) − (b/a) K_{k−1} − (c/a) K_{k−2},  k ≥ 2.

"""
    tri_polar_sing_moment_1_over_r2(r1, t1, r2, t2, m::Integer, n::Integer)

Closed-form value of `∫_K (1/r²) x^m y^n dxdy` on the triangle with
vertices `(0,0)`, `(r1·cos t1, r1·sin t1)`, `(r2·cos t2, r2·sin t2)`.
Requires `r1 > 0`, `r2 > 0`, `t2 > t1`, `t2 - t1 < π`, and
`1 ≤ m+n ≤ 4`. The `m+n = 0` case diverges and throws `DomainError`.

Generic on `T<:Real`. With `Interval{Float64}` inputs the result is a
verified enclosure.
"""
function tri_polar_sing_moment_1_over_r2(r1::T, t1::T, r2::T, t2::T,
                                         m::Integer, n::Integer) where {T<:Real}
    _check_polar_inputs(r1, r2, t1, t2, m, n)
    (m + n) > 4 && throw(DomainError((m, n), "m+n must be ≤ 4"))
    (m + n) == 0 && throw(DomainError((m, n),
        "m+n = 0 case diverges (∫ 1/r² dA = ∞ when origin is at a vertex)"))

    x1 = r1 * cos(t1); y1 = r1 * sin(t1)
    x2 = r2 * cos(t2); y2 = r2 * sin(t2)
    dx = x2 - x1;      dy = y2 - y1
    det12 = x1 * y2 - y1 * x2

    a = dx * dx + dy * dy
    b = 2 * (x1 * dx + y1 * dy)
    c = x1 * x1 + y1 * y1
    Δ = 4 * a * c - b * b
    _check_positive_discriminant(Δ)

    px = poly_power_linear(x1, dx, m)
    py = poly_power_linear(y1, dy, n)
    p  = polymul(px, py)

    maxdeg = m + n
    K = Vector{T}(undef, maxdeg + 1)
    sqrtΔ = sqrt(Δ)
    K[1] = (T(2) / sqrtΔ) *
           (atan((2 * a + b) / sqrtΔ) - atan(b / sqrtΔ))

    if maxdeg ≥ 1
        K[2] = (T(1) / a) * log(r2 / r1) - (b / (2 * a)) * K[1]
    end
    @inbounds for k in 2:maxdeg
        K[k + 1] = T(1) / (a * T(k - 1)) - (b / a) * K[k] - (c / a) * K[k - 1]
    end

    s = zero(T)
    @inbounds for k in 0:maxdeg
        s += p[k + 1] * K[k + 1]
    end
    return det12 / T(maxdeg) * s
end

function tri_polar_sing_moment_1_over_r2(r1::Real, t1::Real, r2::Real, t2::Real,
                                         m::Integer, n::Integer)
    T = promote_type(typeof(r1), typeof(t1), typeof(r2), typeof(t2))
    return tri_polar_sing_moment_1_over_r2(T(r1), T(t1), T(r2), T(t2), m, n)
end
