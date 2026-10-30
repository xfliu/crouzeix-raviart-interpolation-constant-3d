# src/core/singular_moments/tri_polar_sing_moment_le4_exact.jl
#
# Closed-form value of  ∫_K (1/r) x^m y^n dxdy  on a triangle K with
# vertices (0,0), (r1,t1), (r2,t2) (latter two in polar coords).
# Supports m+n ≤ 4. Port of matlab_lib/tri_polar_sing_moment_le4_exact.m.
#
# Derivation (mirrors the MATLAB header):
#   Parameterize K by (x,y) = s·((1-τ)v1 + τ v2),  s∈[0,1], τ∈[0,1].
#   Then dxdy = |det(v1,v2)| · s ds dτ and r = s·|(1-τ)v1+τv2|, so
#       ∫_K (1/r) x^m y^n dxdy
#     = det(v1,v2)/(m+n+1) · ∫_0^1 wx(τ)^m wy(τ)^n / sqrt(q(τ)) dτ,
#   with q(τ) = a τ² + b τ + c. The remaining 1D integral reduces to a
#   linear combination of J_k = ∫_0^1 τ^k / sqrt(q(τ)) dτ for k=0..m+n,
#   and J_k satisfies a 2-term recurrence.

"""
    tri_polar_sing_moment_le4_exact(r1, t1, r2, t2, m::Integer, n::Integer)

Closed-form value of `∫_K (1/r) x^m y^n dxdy` over the triangle with
vertices `(0,0)`, `(r1·cos t1, r1·sin t1)`, `(r2·cos t2, r2·sin t2)`.
Requires `r1 > 0`, `r2 > 0`, `t2 > t1`, `t2 - t1 < π`, and `m+n ≤ 4`.

Generic on `T = promote_type(typeof(r1), typeof(t1), …)`. With
`Interval{Float64}` inputs the result is a verified enclosure.

Throws:
* `DomainError` for any violated input precondition.
"""
function tri_polar_sing_moment_le4_exact(r1::T, t1::T, r2::T, t2::T,
                                         m::Integer, n::Integer) where {T<:Real}
    _check_polar_inputs(r1, r2, t1, t2, m, n)
    (m + n) > 4 && throw(DomainError((m, n), "m+n must be ≤ 4"))

    x1 = r1 * cos(t1); y1 = r1 * sin(t1)
    x2 = r2 * cos(t2); y2 = r2 * sin(t2)
    dx = x2 - x1;      dy = y2 - y1
    det12 = x1 * y2 - y1 * x2

    # q(τ) = a τ² + b τ + c
    a = dx * dx + dy * dy
    b = 2 * (x1 * dx + y1 * dy)
    c = x1 * x1 + y1 * y1            # = r1²
    Δ = 4 * a * c - b * b            # 4·det²
    _check_positive_discriminant(Δ)

    px = poly_power_linear(x1, dx, m)
    py = poly_power_linear(y1, dy, n)
    p  = polymul(px, py)

    maxdeg = m + n
    J = Vector{T}(undef, maxdeg + 1)
    sqrtΔ = sqrt(Δ)
    sqrt_a = sqrt(a)
    J[1] = (asinh((2 * a + b) / sqrtΔ) - asinh(b / sqrtΔ)) / sqrt_a

    if maxdeg ≥ 1
        B1 = r2 - r1
        J[2] = (B1 - (b / 2) * J[1]) / a
    end
    @inbounds for k in 2:maxdeg
        Bk = r2
        # k - 1/2 written this way to keep T-typed arithmetic exact.
        J[k + 1] = (Bk - b * (T(k) - T(1) / T(2)) * J[k] -
                    c * (T(k) - 1) * J[k - 1]) / (a * T(k))
    end

    s = zero(T)
    @inbounds for k in 0:maxdeg
        s += p[k + 1] * J[k + 1]
    end
    return det12 / T(maxdeg + 1) * s
end

function tri_polar_sing_moment_le4_exact(r1::Real, t1::Real, r2::Real, t2::Real,
                                         m::Integer, n::Integer)
    T = promote_type(typeof(r1), typeof(t1), typeof(r2), typeof(t2))
    return tri_polar_sing_moment_le4_exact(T(r1), T(t1), T(r2), T(t2), m, n)
end

# ---- Shared input checks (also used by the 1/r² variant) ------------------
function _check_polar_inputs(r1::T, r2::T, t1::T, t2::T,
                             m::Integer, n::Integer) where {T<:Real}
    # Comparisons on Interval inputs use IntervalArithmetic semantics; we
    # use the float midpoint for the precondition checks. The precondition
    # is a property of the inputs the caller provides, not of the
    # enclosure, so checking the midpoint is the right thing to do — it
    # mirrors how the MATLAB version treats `intval` inputs.
    r1m = _real_value(r1); r2m = _real_value(r2)
    t1m = _real_value(t1); t2m = _real_value(t2)

    r1m > 0 || throw(DomainError(r1, "r1 must be > 0"))
    r2m > 0 || throw(DomainError(r2, "r2 must be > 0"))
    t2m > t1m || throw(DomainError((t1, t2), "require t2 > t1"))
    (t2m - t1m) < π || throw(DomainError(t2m - t1m, "require t2 - t1 < π"))
    m ≥ 0 || throw(DomainError(m, "m must be ≥ 0"))
    n ≥ 0 || throw(DomainError(n, "n must be ≥ 0"))
    return nothing
end

function _check_positive_discriminant(Δ::T) where {T<:Real}
    Δm = _real_value(Δ)
    Δm > 0 || throw(DomainError(Δ,
        "degenerate triangle (Δ = 4ac − b² ≤ 0)"))
    return nothing
end

# Float midpoint extractor that works for both Float64 and Interval{Float64}.
_real_value(x::Real) = float(x)
_real_value(x::Interval) = mid(x)
