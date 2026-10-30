# src/core/bernstein/bernstein_multinomial.jl
#
# The multinomial factor C(N;α) = N! / (α₁! α₂! α₃! α₄!) for tetrahedral
# Bernstein basis functions B_α^N = C(N;α) · L^α.
#
# This factor is the source of the bug fix logged in
# Validation/RESULTS.md: in MATLAB's `create_matrix_ecr_3d.m`, an
# earlier version dropped C(N;α) from the Gram and differentiation
# matrices, causing a Bernstein-vs-monomial coefficient mismatch.
# The Julia port and its tests retain the factor explicitly.

"""
    bernstein_multinomial_3d(N::Integer, α::NTuple{4, <:Integer}) -> Int
    bernstein_multinomial_3d(N::Integer, α::AbstractVector{<:Integer}) -> Int

Tetrahedral Bernstein multinomial coefficient
`C(N; α) = N! / (α₁! α₂! α₃! α₄!)` with `|α| = N`.

Throws `DomainError` if any entry of `α` is negative or `sum(α) ≠ N`.

# Vectorised form

    bernstein_multinomial_3d(N, αs::AbstractMatrix) -> Vector{Int}

Apply the scalar form to each row of `αs` (an `m × 4` matrix). Same
preconditions checked per-row.
"""
function bernstein_multinomial_3d(N::Integer, α)
    a = (Int(α[1]), Int(α[2]), Int(α[3]), Int(α[4]))
    (a[1] ≥ 0 && a[2] ≥ 0 && a[3] ≥ 0 && a[4] ≥ 0) ||
        throw(DomainError(α, "α entries must be ≥ 0"))
    sum(a) == N ||
        throw(DomainError((N, α), "sum(α) must equal N (got $(sum(a)) vs $N)"))
    # Use Int factorial — degrees up to N=12 fit in Int64. For larger N
    # this would overflow; the FEM kernel here is degree-N ≤ 5 in
    # practice, so we keep it simple and fast.
    return factorial(N) ÷ (factorial(a[1]) * factorial(a[2]) *
                            factorial(a[3]) * factorial(a[4]))
end

function bernstein_multinomial_3d(N::Integer, αs::AbstractMatrix{<:Integer})
    size(αs, 2) == 4 || throw(DimensionMismatch("αs must have 4 columns"))
    out = Vector{Int}(undef, size(αs, 1))
    @inbounds for r in axes(αs, 1)
        out[r] = bernstein_multinomial_3d(N, (αs[r, 1], αs[r, 2], αs[r, 3], αs[r, 4]))
    end
    return out
end
