# src/core/eigensolve3d/one_piece_bubble_laplace_3d.jl
#
# One-piece polynomial Rayleigh-Ritz solve for the 3D Dirichlet Laplacian on
# a single tetrahedron. The basis is the barycentric bubble space
#   lambda_1 lambda_2 lambda_3 lambda_4 P_{degree-4},
# so no mesh or boundary-DOF elimination is needed.

using LinearAlgebra: Symmetric, det, eigen, inv

struct OnePieceBubbleLaplace3D
    degree::Int
    eig_value::Vector{Float64}
    eig_func::Matrix{Float64}
    A::Matrix{Float64}
    M::Matrix{Float64}
    exponents::Matrix{Int}
    vertices::Matrix{Float64}
end

function _tet_volume_and_grad_lambda(vertices::AbstractMatrix{<:Real})
    size(vertices) == (4, 3) ||
        throw(DimensionMismatch("vertices must be a 4 x 3 matrix"))
    P = Matrix{Float64}(vertices)
    J = [P[2, 1] - P[1, 1]  P[2, 2] - P[1, 2]  P[2, 3] - P[1, 3];
         P[3, 1] - P[1, 1]  P[3, 2] - P[1, 2]  P[3, 3] - P[1, 3];
         P[4, 1] - P[1, 1]  P[4, 2] - P[1, 2]  P[4, 3] - P[1, 3]]
    volume = abs(det(J)) / 6.0
    volume > 0 || throw(DomainError(volume, "degenerate tetrahedron"))
    Ji = inv(J)
    grad = zeros(Float64, 4, 3)
    for k in 1:3
        grad[2, k] = Ji[k, 1]
        grad[3, k] = Ji[k, 2]
        grad[4, k] = Ji[k, 3]
    end
    grad[1, :] .= .-(grad[2, :] .+ grad[3, :] .+ grad[4, :])
    return volume, grad
end

function _barycentric_monomial_integral(exp::NTuple{4, Int}, volume::Float64)
    total = exp[1] + exp[2] + exp[3] + exp[4]
    val = 6.0 * volume
    for k in 1:4
        for n in 2:exp[k]
            val *= n
        end
    end
    for n in 2:(total + 3)
        val /= n
    end
    return val
end

function _bubble_exponents_3d(degree::Integer)
    degree ≥ 4 || throw(DomainError(degree, "degree must be at least 4"))
    α = ijkl_list(Int(degree) - 4)
    β = similar(α)
    @inbounds for i in axes(α, 1), j in 1:4
        β[i, j] = α[i, j] + 1
    end
    return β
end

function one_piece_bubble_laplace_matrices_3d(vertices::AbstractMatrix{<:Real},
                                              degree::Integer)
    p = Int(degree)
    β = _bubble_exponents_3d(p)
    volume, grad = _tet_volume_and_grad_lambda(vertices)
    n = size(β, 1)
    M = zeros(Float64, n, n)
    A = zeros(Float64, n, n)

    @inbounds for j in 1:n, i in 1:j
        e_mass = (β[i, 1] + β[j, 1], β[i, 2] + β[j, 2],
                  β[i, 3] + β[j, 3], β[i, 4] + β[j, 4])
        Mij = _barycentric_monomial_integral(e_mass, volume)
        Aij = 0.0
        for a in 1:4, b in 1:4
            β[i, a] == 0 && continue
            β[j, b] == 0 && continue
            e = (β[i, 1] + β[j, 1] - (a == 1) - (b == 1),
                 β[i, 2] + β[j, 2] - (a == 2) - (b == 2),
                 β[i, 3] + β[j, 3] - (a == 3) - (b == 3),
                 β[i, 4] + β[j, 4] - (a == 4) - (b == 4))
            Aij += β[i, a] * β[j, b] * dot(grad[a, :], grad[b, :]) *
                   _barycentric_monomial_integral(e, volume)
        end
        M[i, j] = Mij
        A[i, j] = Aij
        if i != j
            M[j, i] = Mij
            A[j, i] = Aij
        end
    end
    return A, M, β
end

function one_piece_bubble_laplace_3d(vertices::AbstractMatrix{<:Real},
                                     degree::Integer,
                                     neig::Integer)
    neig ≥ 1 || throw(DomainError(neig, "neig must be at least 1"))
    A, M, β = one_piece_bubble_laplace_matrices_3d(vertices, degree)
    n = size(A, 1)
    k = min(Int(neig), n)
    F = eigen(Symmetric(A), Symmetric(M))
    vals = F.values[1:k]
    vecs = F.vectors[:, 1:k]
    return OnePieceBubbleLaplace3D(Int(degree), vals, vecs, A, M, β,
                                   Matrix{Float64}(vertices))
end

function special_tetrahedron_vertices()
    return [ 0.0   0.0   0.0;
             0.0   0.0   1.0;
             0.5   0.5   0.5;
            -0.5   0.5   0.5 ]
end
