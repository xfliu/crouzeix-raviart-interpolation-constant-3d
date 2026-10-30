# src/core/singular_moments/polymul.jl
#
# Convolution of two coefficient vectors. Replaces MATLAB `conv(px, py)`
# inside the closed-form moment routines. Local helper, not exported.

"""
    polymul(p::AbstractVector, q::AbstractVector) -> Vector

Polynomial product in coefficient form (ascending powers). For
`p = [a0, a1, ..., am]` and `q = [b0, b1, ..., bn]` returns the
length-`(m+n+1)` vector of coefficients of `(Σ aᵢ tⁱ)·(Σ bⱼ tʲ)`.

Both empty inputs are not supported (caller's responsibility — the only
caller is the closed-form singular-moment family, which always passes
non-empty polynomials).
"""
function polymul(p::AbstractVector, q::AbstractVector)
    m = length(p) - 1
    n = length(q) - 1
    T = promote_type(eltype(p), eltype(q))
    r = zeros(T, m + n + 1)
    @inbounds for i in 0:m, j in 0:n
        r[i + j + 1] += p[i + 1] * q[j + 1]
    end
    return r
end
