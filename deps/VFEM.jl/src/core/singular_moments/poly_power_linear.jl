# src/core/singular_moments/poly_power_linear.jl
#
# Coefficients (ascending powers) of (c0 + c1*t)^p.
#
# Port of matlab_lib/poly_power_linear.m — internal helper used by the
# closed-form triangle singular-moment routines to expand wx(τ)^m * wy(τ)^n
# as a polynomial in τ.

"""
    poly_power_linear(c0::T, c1::T, p::Integer) -> Vector{T}

Return the (p+1)-vector of coefficients of `(c0 + c1*t)^p`, listed in
ascending powers of `t`. Coefficients are computed exactly via the
binomial theorem `binomial(p, k) * c0^(p-k) * c1^k`, so the operation
preserves interval enclosures: each `binomial(p, k)` is exact, and only
the floating-point multiplications introduce widening.

Throws `DomainError` if `p < 0`.

# Examples
```julia
julia> poly_power_linear(2.0, 3.0, 2)   # 4 + 12 t + 9 t^2
3-element Vector{Float64}:
  4.0
 12.0
  9.0
```
"""
function poly_power_linear(c0::T, c1::T, p::Integer) where {T<:Real}
    p < 0 && throw(DomainError(p, "poly_power_linear: p must be ≥ 0"))
    coeff = Vector{T}(undef, p + 1)
    @inbounds for k in 0:p
        coeff[k + 1] = T(binomial(p, k)) * c0^(p - k) * c1^k
    end
    return coeff
end

# Promotion entry point: callers may pass (Float64, Int, Int).
function poly_power_linear(c0::Real, c1::Real, p::Integer)
    T = promote_type(typeof(c0), typeof(c1))
    return poly_power_linear(T(c0), T(c1), p)
end
