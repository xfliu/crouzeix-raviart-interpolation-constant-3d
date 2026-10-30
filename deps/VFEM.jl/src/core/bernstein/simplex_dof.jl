# src/core/bernstein/simplex_dof.jl
#
# Number of multi-indices α ∈ ℤ^{n+1}_{≥0} with |α| = M, equivalently
# the dimension of the space of polynomials of total degree ≤ M on an
# n-simplex. Port of VFEM3D/lib/fem_assembly/get_DOF.m.

"""
    simplex_dof(n::Integer, M::Integer) -> Int

Dimension of the space of polynomials of total degree `M` on an
`n`-simplex, equal to `binomial(n + M, n)`. For tetrahedra (n=3),
this is `(M+1)(M+2)(M+3)/6` — the number of barycentric multi-indices
`(i,j,k,l)` with `i+j+k+l = M`.

Throws `DomainError` if `n < 0` or `M < 0`.
"""
function simplex_dof(n::Integer, M::Integer)
    n ≥ 0 || throw(DomainError(n, "n must be ≥ 0"))
    M ≥ 0 || throw(DomainError(M, "M must be ≥ 0"))
    return binomial(n + M, n)
end
