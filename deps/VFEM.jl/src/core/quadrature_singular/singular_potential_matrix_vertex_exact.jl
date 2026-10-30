# src/core/quadrature_singular/singular_potential_matrix_vertex_exact.jl
#
# Port of VFEM3D/lib/quadrature/singular_potential_matrix_vertex_exact.m.
#
# Returns the local Coulomb-kernel matrix
#   T[i, j] = ∫_K B_i^N(x) B_j^N(x) / |x - P_s| dx
# where B^N is the degree-N tetrahedral Bernstein basis ordered by
# ijkl_list(N) and P_s is one tet vertex. For physical potential
# V = -Z/|x - P_s|, multiply T by -Z.

"""
    singular_potential_matrix_vertex_exact(LocalNodes::AbstractMatrix,
                                           singular_vertex::Integer,
                                           N::Integer)
        -> (T::Matrix, info::NamedTuple)

Local matrix `T[i, j] = ∫_K B_i^N B_j^N / |x - P_s| dx`. Element
type follows `eltype(LocalNodes)` so interval inputs return an
interval matrix. `info` carries `degree`, `ijkl`, `moment_degree`.
"""
function singular_potential_matrix_vertex_exact(LocalNodes::AbstractMatrix{T},
                                                singular_vertex::Integer,
                                                N::Integer) where {T<:Real}
    N ≥ 0 || throw(DomainError(N, "N must be ≥ 0"))

    list_N  = ijkl_list(N)
    list_2N = ijkl_list(2 * N)
    moments, _ = tet_vertex_sing_bernstein_moments(LocalNodes,
                                                   singular_vertex, 2 * N)

    multi_N = bernstein_multinomial_3d(N, list_N)
    idx_2N  = ijkl_index_map(2 * N)
    DegK = size(list_N, 1)
    Tmat = Matrix{T}(undef, DegK, DegK)

    @inbounds for i in 1:DegK, j in 1:DegK
        γ = (list_N[i, 1] + list_N[j, 1],
             list_N[i, 2] + list_N[j, 2],
             list_N[i, 3] + list_N[j, 3],
             list_N[i, 4] + list_N[j, 4])
        Cγ = bernstein_multinomial_3d(2 * N, γ)
        factor = T(multi_N[i]) * T(multi_N[j]) / T(Cγ)
        Tmat[i, j] = factor * moments[idx_2N[γ]]
    end

    info = (; degree = N, ijkl = list_N, moment_degree = 2 * N)
    return Tmat, info
end
