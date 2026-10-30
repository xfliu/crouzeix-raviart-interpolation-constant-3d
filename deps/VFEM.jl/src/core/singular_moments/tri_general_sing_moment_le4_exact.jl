# src/core/singular_moments/tri_general_sing_moment_le4_exact.jl
#
# Decompose a general triangle into three signed origin-based polar
# sub-triangles and sum the closed-form 1/r moments.
# Port of matlab_lib/tri_general_sing_moment_le4_exact.m.

"""
    tri_general_sing_moment_le4_exact(x1, y1, x2, y2, x3, y3,
                                      m::Integer, n::Integer)

Closed-form value of `∫_K (1/r) x^m y^n dxdy` on the triangle with
Cartesian vertices `(x1,y1), (x2,y2), (x3,y3)`. Supports `m+n ≤ 4`.

The triangle is split via the shoelace identity into three signed
sub-triangles `(0, V_i, V_{i+1})`, each handled by the polar
closed-form `tri_polar_sing_moment_le4_exact`. Sub-triangles with
near-zero oriented area are skipped.

Generic on `T = promote_type(...)`. With `Interval{Float64}` inputs the
result is a verified enclosure.
"""
function tri_general_sing_moment_le4_exact(x1::T, y1::T, x2::T, y2::T,
                                           x3::T, y3::T,
                                           m::Integer, n::Integer) where {T<:Real}
    (m + n) > 4 && throw(DomainError((m, n), "m+n must be ≤ 4"))
    m ≥ 0 || throw(DomainError(m, "m must be ≥ 0"))
    n ≥ 0 || throw(DomainError(n, "n must be ≥ 0"))

    V = ((x1, y1), (x2, y2), (x3, y3))

    I = zero(T)
    @inbounds for i in 1:3
        v1 = V[i]
        v2 = V[mod(i, 3) + 1]
        det12 = v1[1] * v2[2] - v1[2] * v2[1]
        # Skip sub-triangles whose oriented area is numerically zero.
        # Use the float midpoint for the test (interval-safe).
        abs(_real_value(det12)) < 1e-14 && continue

        # Capture the ORIGINAL orientation sign before any swap. The
        # shoelace decomposition is
        #     ∫_K f dx = Σ_i sign(det(V_i, V_{i+1})) · ∫_{W_i} f dx
        # where W_i = unsigned wedge (origin, V_i, V_{i+1}). We compute
        # ∫_{W_i} f dx via the polar routine (which expects t2 > t1, so
        # we may need to swap below), and then multiply by the *original*
        # det12 sign — not the post-swap sign.
        #
        # Note: the MATLAB source matlab_lib/tri_general_sing_moment_le4_exact.m
        # has a latent sign bug here — it negates det12 inside the swap
        # and then takes sign() of the negated value, which flips the
        # contribution sign for swapped edges. There is no MATLAB test
        # for the 1/r general routine (only for 1/r²), so the bug never
        # surfaced upstream. Verified by independent Gauss quadrature.
        sgn = sign(_real_value(det12))

        r1 = hypot(v1[1], v1[2])
        r2 = hypot(v2[1], v2[2])
        t1 = atan(v1[2], v1[1])
        t2 = atan(v2[2], v2[1])

        # Force t2 > t1 so the polar routine's preconditions hold. The
        # polar routine returns the integral over the unsigned wedge.
        if _real_value(t2) < _real_value(t1)
            t1, t2 = t2, t1
            r1, r2 = r2, r1
        end

        # Handle the ±π branch-cut case: when the two vertices are on
        # nearly opposite sides of the negative x-axis, atan2 values
        # straddle ±π and the raw span t2-t1 ≈ 2π is spuriously large.
        # The true angular extent of the wedge (signed by det12) is the
        # short arc 2π-(t2-t1) < π.  Fix: keep sgn from det12 but
        # re-express the wedge as the short CCW arc from the high-angle
        # vertex around the branch cut: t1_new=t2, t2_new=t1+2π.
        if _real_value(t2) - _real_value(t1) >= T(π)
            t1, t2 = t2, t1 + T(2π)
            r1, r2 = r2, r1
        end

        I_loc = tri_polar_sing_moment_le4_exact(r1, t1, r2, t2, m, n)
        I += sgn * I_loc
    end
    return I
end

function tri_general_sing_moment_le4_exact(x1::Real, y1::Real, x2::Real, y2::Real,
                                           x3::Real, y3::Real,
                                           m::Integer, n::Integer)
    T = promote_type(typeof(x1), typeof(y1), typeof(x2), typeof(y2),
                     typeof(x3), typeof(y3))
    return tri_general_sing_moment_le4_exact(T(x1), T(y1), T(x2), T(y2),
                                             T(x3), T(y3), m, n)
end
