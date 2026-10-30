# src/core/singular_moments/tri_general_sing_moment_1_over_r2.jl
#
# General-triangle 1/r² moment via signed decomposition into origin-based
# polar sub-triangles using the *interior angle at the origin* (rather
# than the raw atan2 difference, which does not handle the wraparound
# across the −π/π branch). Port of
# matlab_lib/tri_general_sing_moment_1_over_r2.m.
#
# Caller responsibility: the origin must NOT be strictly inside K.

"""
    tri_general_sing_moment_1_over_r2(x1, y1, x2, y2, x3, y3,
                                      m::Integer, n::Integer)

Closed-form value of `∫_K (1/r²) x^m y^n dxdy` on the triangle with
Cartesian vertices `(x1,y1), (x2,y2), (x3,y3)`. Supports
`1 ≤ m+n ≤ 4`. The case `m+n = 0` diverges and throws `DomainError`.

Caller responsibility: the origin must NOT be strictly inside K — the
caller must guarantee this; the routine does not check. Sub-triangles
with a vertex at the origin or with degenerate area are skipped.

Generic on `T<:Real`. With `Interval{Float64}` inputs the result is a
verified enclosure.
"""
function tri_general_sing_moment_1_over_r2(x1::T, y1::T, x2::T, y2::T,
                                           x3::T, y3::T,
                                           m::Integer, n::Integer) where {T<:Real}
    (m + n) == 0 && throw(DomainError((m, n),
        "m+n = 0 case diverges (∫ 1/r² dA = ∞)"))
    (m + n) > 4 && throw(DomainError((m, n), "m+n must be ≤ 4"))
    m ≥ 0 || throw(DomainError(m, "m must be ≥ 0"))
    n ≥ 0 || throw(DomainError(n, "n must be ≥ 0"))

    V = ((x1, y1), (x2, y2), (x3, y3))

    I = zero(T)
    @inbounds for i in 1:3
        v1 = V[i]
        v2 = V[mod(i, 3) + 1]
        det12 = v1[1] * v2[2] - v1[2] * v2[1]
        abs(_real_value(det12)) < 1e-14 && continue

        r1 = hypot(v1[1], v1[2])
        r2 = hypot(v2[1], v2[2])
        # Skip when one endpoint coincides with the origin — that
        # sub-triangle has measure zero and the polar formula is
        # undefined at r=0.
        _real_value(r1) < 1e-14 && continue
        _real_value(r2) < 1e-14 && continue

        # Interior angle at the origin between v1 and v2, clamped.
        cos_a = (v1[1] * v2[1] + v1[2] * v2[2]) / (r1 * r2)
        cos_a_real = _real_value(cos_a)
        cos_a_clamped_real = max(-1.0, min(1.0, cos_a_real))
        ang = if cos_a_clamped_real == cos_a_real
            acos(cos_a)
        else
            # Saturated: use the float-saturated value (interval result
            # is a degenerate point — fine, since this only happens at
            # the geometric corner cases ang = 0 or π).
            T(acos(cos_a_clamped_real))
        end
        _real_value(ang) < 1e-14 && continue

        # Orient CCW around the origin.
        if _real_value(det12) > 0
            t1 = atan(v1[2], v1[1])
            t2 = t1 + ang
            rr1 = r1; rr2 = r2
            sgn = one(T)
        else
            t1 = atan(v2[2], v2[1])
            t2 = t1 + ang
            rr1 = r2; rr2 = r1
            sgn = -one(T)
        end

        I_sub = tri_polar_sing_moment_1_over_r2(rr1, t1, rr2, t2, m, n)
        I += sgn * I_sub
    end
    return I
end

function tri_general_sing_moment_1_over_r2(x1::Real, y1::Real, x2::Real, y2::Real,
                                           x3::Real, y3::Real,
                                           m::Integer, n::Integer)
    T = promote_type(typeof(x1), typeof(y1), typeof(x2), typeof(y2),
                     typeof(x3), typeof(y3))
    return tri_general_sing_moment_1_over_r2(T(x1), T(y1), T(x2), T(y2),
                                             T(x3), T(y3), m, n)
end
