# src/core/bernstein/bernstein_product.jl
#
# Exact tetrahedral Bernstein-polynomial product. Port of
# VFEM3D/lib/quadrature/BernsteinProductExact3D.m.
#
#   B_α^N B_β^M = (C(N;α) C(M;β) / C(N+M; α+β)) · B_{α+β}^{N+M}
#
# So if p = Σ_α c1_α B_α^N and q = Σ_β c2_β B_β^M, then
# p·q = Σ_γ d_γ B_γ^{N+M}, with
#
#   d_γ = Σ_{α+β=γ} c1_α c2_β · C(N;α) C(M;β) / C(N+M;γ)
#
# **WHY THIS MATTERS** — Validation/RESULTS.md documents a real bug
# from dropping these multinomial factors. The FEM Gram, differentiation,
# and face/cell-average matrices all flow through this routine. The
# adversarial test for this routine intentionally constructs a polynomial
# whose coefficients only line up if the C(·;·) ratio is correct.

"""
    bernstein_product_exact(N::Integer, M::Integer,
                            c1::AbstractVector, c2::AbstractVector)
        -> (new_c::Vector, ijkl_new::Matrix{Int})

Exact product of two tetrahedral Bernstein polynomials in degree
`N + M`. The Bernstein convention `B_α^N = C(N;α) L^α` means the
multinomial-ratio factor `C(N;α)·C(M;β)/C(N+M;γ)` is required;
this is the source of the bug logged in `Validation/RESULTS.md`.

# Arguments
- `N`, `M` : degrees of `c1`, `c2`.
- `c1`, `c2` : Bernstein coefficients ordered by `ijkl_list(N)`,
  `ijkl_list(M)`.

# Returns
- `new_c` : Bernstein coefficients of `p·q` ordered by `ijkl_list(N+M)`.
- `ijkl_new` : the matching multi-index list (returned for the caller's
  convenience; identical to `ijkl_list(N+M)`).
"""
function bernstein_product_exact(N::Integer, M::Integer,
                                 c1::AbstractVector, c2::AbstractVector)
    N ≥ 0 && M ≥ 0 ||
        throw(DomainError((N, M), "N and M must be ≥ 0"))

    ijkl1   = ijkl_list(N)
    ijkl2   = ijkl_list(M)
    ijklnew = ijkl_list(N + M)

    length(c1) == size(ijkl1, 1) ||
        throw(DimensionMismatch("c1 length $(length(c1)) ≠ DOF $(size(ijkl1, 1))"))
    length(c2) == size(ijkl2, 1) ||
        throw(DimensionMismatch("c2 length $(length(c2)) ≠ DOF $(size(ijkl2, 1))"))

    C1 = bernstein_multinomial_3d(N, ijkl1)
    C2 = bernstein_multinomial_3d(M, ijkl2)
    idx_map = ijkl_index_map(N + M)

    T = promote_type(eltype(c1), eltype(c2), Float64)
    new_c = zeros(T, size(ijklnew, 1))

    @inbounds for i in axes(ijkl1, 1)
        ci = c1[i]
        iszero(ci) && continue
        αi = (ijkl1[i, 1], ijkl1[i, 2], ijkl1[i, 3], ijkl1[i, 4])
        for j in axes(ijkl2, 1)
            cj = c2[j]
            iszero(cj) && continue
            βj = (ijkl2[j, 1], ijkl2[j, 2], ijkl2[j, 3], ijkl2[j, 4])
            γ  = (αi[1] + βj[1], αi[2] + βj[2], αi[3] + βj[3], αi[4] + βj[4])
            idx = idx_map[γ]
            Cγ  = bernstein_multinomial_3d(N + M, γ)
            new_c[idx] += ci * cj * (C1[i] * C2[j]) / Cγ
        end
    end
    return new_c, ijklnew
end

# ---------------------------------------------------------------------------
# Monomial-form product (no multinomial ratio). Port of BernsteinProduct.m.
# Used when the coefficients represent the polynomial as Σ_α c_α L^α
# directly (no C(N;α) factor) — e.g. for the gradient-on-Bernstein and
# RT auxiliary computations that work with monomial-form coefficients.
# Keep this routine separate from `bernstein_product_exact` to avoid the
# multinomial bug class.
# ---------------------------------------------------------------------------

"""
    bernstein_product_monomial(N::Integer, M::Integer,
                               c1::AbstractVector, c2::AbstractVector)
        -> Vector

Product of two polynomials given in *monomial* tetrahedral form
`p = Σ_α c1_α L^α`, `q = Σ_β c2_β L^β`. No multinomial ratio is
applied — this is `BernsteinProduct.m`, used for monomial-form data
inside the assembly. Use `bernstein_product_exact` for true Bernstein
basis-form coefficients.
"""
function bernstein_product_monomial(N::Integer, M::Integer,
                                    c1::AbstractVector, c2::AbstractVector)
    N ≥ 0 && M ≥ 0 ||
        throw(DomainError((N, M), "N and M must be ≥ 0"))

    ijkl1   = ijkl_list(N)
    ijkl2   = ijkl_list(M)
    ijklnew = ijkl_list(N + M)

    length(c1) == size(ijkl1, 1) ||
        throw(DimensionMismatch("c1 length $(length(c1)) ≠ DOF $(size(ijkl1, 1))"))
    length(c2) == size(ijkl2, 1) ||
        throw(DimensionMismatch("c2 length $(length(c2)) ≠ DOF $(size(ijkl2, 1))"))

    idx_map = ijkl_index_map(N + M)
    T = promote_type(eltype(c1), eltype(c2))
    new_c = zeros(T, size(ijklnew, 1))

    @inbounds for i in axes(ijkl1, 1), j in axes(ijkl2, 1)
        γ = (ijkl1[i, 1] + ijkl2[j, 1],
             ijkl1[i, 2] + ijkl2[j, 2],
             ijkl1[i, 3] + ijkl2[j, 3],
             ijkl1[i, 4] + ijkl2[j, 4])
        new_c[idx_map[γ]] += c1[i] * c2[j]
    end
    return new_c
end
