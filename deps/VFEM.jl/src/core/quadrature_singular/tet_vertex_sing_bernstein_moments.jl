# src/core/quadrature_singular/tet_vertex_sing_bernstein_moments.jl
#
# Port of VFEM3D/lib/quadrature/tet_vertex_sing_bernstein_moments.m.
#
# Computes
#   S(γ) = ∫_K B_γ^N(x) / |x - P_s| dx
# for every degree-N tetrahedral Bernstein basis function, where
# K is the tetrahedron with vertices LocalNodes (4×3) and the
# singularity is at LocalNodes[singular_vertex, :].
#
# Method: the Duffy radial integral is taken analytically (a Beta-
# function value), and the remaining 2D opposite-face moment is
# evaluated by `tri3d_invR_face_moments`.

using LinearAlgebra: det

"""
    tet_vertex_sing_bernstein_moments(LocalNodes::AbstractMatrix,
                                      singular_vertex::Integer,
                                      N::Integer)
        -> (S::Vector, info::NamedTuple)

Exact vertex-singular Bernstein moments
`S[k] = ∫_K B_{γ_k}^N(x) / |x - P_s| dx` for `γ_k = ijkl_list(N)[k, :]`,
where the singularity is at `LocalNodes[singular_vertex, :]`.

`info` has the same diagnostic fields as the MATLAB version: degree,
ijkl, singular_vertex, face_idx, det6, volume, face.
"""
function tet_vertex_sing_bernstein_moments(LocalNodes::AbstractMatrix{T},
                                           singular_vertex::Integer,
                                           N::Integer) where {T<:Real}
    size(LocalNodes) == (4, 3) ||
        throw(DimensionMismatch("LocalNodes must be 4×3"))
    1 ≤ singular_vertex ≤ 4 ||
        throw(DomainError(singular_vertex, "singular_vertex must be in 1:4"))
    N ≥ 0 || throw(DomainError(N, "N must be ≥ 0"))

    P0 = LocalNodes[singular_vertex, :]
    face_idx = filter(!=(singular_vertex), 1:4)
    face_nodes = Matrix{T}(undef, 3, 3)
    @inbounds for r in 1:3
        face_nodes[r, :] = LocalNodes[face_idx[r], :] .- P0
    end

    det6 = abs(det(face_nodes))
    abs(_real_value(det6)) > 1e-14 ||
        throw(DomainError(det6, "Degenerate tetrahedron"))

    F, face_info = tri3d_invR_face_moments(face_nodes, N)

    list = ijkl_list(N)
    multi = bernstein_multinomial_3d(N, list)
    S = Vector{T}(undef, size(list, 1))

    @inbounds for k in axes(list, 1)
        γ = (list[k, 1], list[k, 2], list[k, 3], list[k, 4])
        β = (γ[face_idx[1]], γ[face_idx[2]], γ[face_idx[3]])
        β_sum = β[1] + β[2] + β[3]
        # Radial integral ∫_0^1 r^(β_sum + γ_s) (1-r)^... dr handled
        # by Beta function:
        radial = T(factorial(β_sum + 1)) * T(factorial(γ[singular_vertex])) /
                 T(factorial(N + 2))
        face_val = F[β[1] + 1, β[2] + 1, β[3] + 1]
        S[k] = det6 * T(multi[k]) * radial * face_val
    end

    info = (; degree = N,
              ijkl = list,
              singular_vertex,
              face_idx,
              det6,
              volume = det6 / 6,
              face = face_info)
    return S, info
end
