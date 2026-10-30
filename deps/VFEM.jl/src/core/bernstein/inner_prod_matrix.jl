# src/core/bernstein/inner_prod_matrix.jl
#
# Reference and physical inner-product matrices for tetrahedral
# polynomials given in *monomial* form (basis L^α).
# Ports of getInnerProdMatrix.m, getInnerProdMatrix_Reference.m,
# and getInnerProdMatrix_Reference3D.m.
#
# Underlying identity (Eisenberg-Lass):
#     ∫_K L₁^{a₁} L₂^{a₂} L₃^{a₃} L₄^{a₄} dV = 6|K| · a₁! a₂! a₃! a₄! / (|a|+3)!
#
# So for monomial-form pairs:
#     A[α, β] = ∫_K L^α L^β dV = 6|K| · (α+β)! / (|α|+|β|+3)!
# where (α+β)! is the product of factorials of the entries of α+β.

"""
    inner_prod_matrix(deg1::Integer, deg2::Integer, K_vol) -> Matrix

Pairwise integral matrix
`A[α, β] = ∫_K L^α · L^β dV = 6 · K_vol · (α+β)! / (|α|+|β|+3)!`
on a tetrahedron of volume `K_vol`. Rows/columns are ordered by
`ijkl_list(deg1)` and `ijkl_list(deg2)` respectively. The element
type follows `K_vol` (so passing an `Interval` returns an interval
matrix).
"""
function inner_prod_matrix(deg1::Integer, deg2::Integer, K_vol::T) where {T<:Real}
    L1 = ijkl_list(deg1)
    L2 = ijkl_list(deg2)
    m, n = size(L1, 1), size(L2, 1)
    A = Matrix{T}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        val = T(6) * K_vol
        for q in 1:4
            for r in 2:(L1[i, q] + L2[j, q])
                val *= T(r)
            end
        end
        for r in 2:(deg1 + deg2 + 3)
            val /= T(r)
        end
        A[i, j] = val
    end
    return A
end

"""
    inner_prod_matrix_reference(deg1::Integer, deg2::Integer; T = Float64) -> Matrix{T}

Specialisation of `inner_prod_matrix` to the reference tetrahedron
`K_vol = 1`. Element type controlled by the keyword `T`.
"""
function inner_prod_matrix_reference(deg1::Integer, deg2::Integer; T::Type = Float64)
    return inner_prod_matrix(deg1, deg2, one(T))
end

"""
    inner_prod_matrix_reference3d(deg1::Integer, deg2::Integer, deg3::Integer;
                                  T = Float64) -> Array{T,3}

Triple-monomial reference inner product on the unit-volume reference
tetrahedron:
    A[α, β, γ] = 6 · (α+β+γ)! / (|α|+|β|+|γ|+3)!

Used by RT/curl assembly that needs three-way inner products.
"""
function inner_prod_matrix_reference3d(deg1::Integer, deg2::Integer, deg3::Integer;
                                       T::Type = Float64)
    L1 = ijkl_list(deg1)
    L2 = ijkl_list(deg2)
    L3 = ijkl_list(deg3)
    m, n, l = size(L1, 1), size(L2, 1), size(L3, 1)
    A = Array{T, 3}(undef, m, n, l)
    @inbounds for i in 1:m, j in 1:n, k in 1:l
        val = T(6)
        for q in 1:4
            for r in 2:(L1[i, q] + L2[j, q] + L3[k, q])
                val *= T(r)
            end
        end
        for r in 2:(deg1 + deg2 + deg3 + 3)
            val /= T(r)
        end
        A[i, j, k] = val
    end
    return A
end
