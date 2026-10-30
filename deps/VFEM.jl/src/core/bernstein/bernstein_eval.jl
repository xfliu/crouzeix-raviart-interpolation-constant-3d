# src/core/bernstein/bernstein_eval.jl
#
# Evaluation of 1D and tetrahedral Bernstein polynomials.
# Ports of VFEM3D/lib/quadrature/Bernstein.m and Bernstein3D.m.
#
# 1D form:        B_i^n(x) = C(n,i) x^i (1-x)^{n-i}
# Tetrahedral:    B_α^N(L) = C(N;α) L₁^{α₁} L₂^{α₂} L₃^{α₃} L₄^{α₄}
#
# We do NOT include `Bernstein3D(M, T, C, x)` — the cartesian-to-
# barycentric step in MATLAB depends on `xyz2uvwt`, which lives in
# fem_assembly. The barycentric-input form `bernstein3d_eval_bary`
# below is the kernel; once the mesh layer ports `xyz2uvwt`, the
# cartesian wrapper is a one-liner.

"""
    bernstein_eval(n::Integer, i::Integer, x) -> typeof(x)

1D Bernstein polynomial `B_i^n(x) = C(n,i) x^i (1-x)^{n-i}` for
`0 ≤ i ≤ n`. Evaluates at scalar or array `x`.
"""
function bernstein_eval(n::Integer, i::Integer, x)
    (n ≥ 0 && 0 ≤ i ≤ n) ||
        throw(DomainError((n, i), "require n ≥ 0 and 0 ≤ i ≤ n"))
    return binomial(n, i) .* x .^ i .* (1 .- x) .^ (n - i)
end

"""
    bernstein3d_eval_bary(M::Integer, c::AbstractVector, L) -> Vector

Evaluate the polynomial `Σ_α c_α B_α^M(L)` at one or more barycentric
points. `L` may be a length-4 vector (one point) or an `m × 4` matrix
(`m` points, one per row). `c` is the coefficient vector ordered by
`ijkl_list(M)`.

# Returns
A scalar if `L` is a single point (length-4 vector), else a length-`m`
column vector.
"""
function bernstein3d_eval_bary(M::Integer, c::AbstractVector,
                               L::AbstractMatrix)
    size(L, 2) == 4 || throw(DimensionMismatch("L must have 4 columns"))
    list = ijkl_list(M)
    length(c) == size(list, 1) ||
        throw(DimensionMismatch("c length $(length(c)) ≠ DOF $(size(list, 1))"))
    multi = bernstein_multinomial_3d(M, list)
    T = promote_type(eltype(c), eltype(L), Int)
    out = zeros(T, size(L, 1))
    @inbounds for k in 1:size(list, 1)
        a1, a2, a3, a4 = list[k, 1], list[k, 2], list[k, 3], list[k, 4]
        coeff = c[k] * multi[k]
        for r in axes(L, 1)
            out[r] += coeff * L[r, 1]^a1 * L[r, 2]^a2 *
                              L[r, 3]^a3 * L[r, 4]^a4
        end
    end
    return out
end

# Single-point convenience.
function bernstein3d_eval_bary(M::Integer, c::AbstractVector,
                               L::AbstractVector)
    length(L) == 4 || throw(DimensionMismatch("L must have length 4"))
    return bernstein3d_eval_bary(M, c, reshape(collect(L), 1, 4))[1]
end
