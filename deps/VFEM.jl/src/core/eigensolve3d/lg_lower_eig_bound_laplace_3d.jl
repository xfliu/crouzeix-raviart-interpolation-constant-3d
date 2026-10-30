# src/core/eigensolve3d/lg_lower_eig_bound_laplace_3d.jl
#
# Lehmann-Goerisch lower bounds for the 3D Dirichlet Laplacian, using
# conforming CG/Lagrange eigenvectors and the scalar RT/DG auxiliary
# matrices ported from VFEM3D.

using IntervalArithmetic: Interval, interval, inf
using LinearAlgebra: Symmetric, dot, eigen, eigvals, inv, norm

const _LIU_C3D_CR = 1 / sqrt(10.0)

struct LGLaplaceLowerBound3D
    eig_lower::Vector{Float64}
    eig_upper::Vector{Float64}
    cr_eig_lower::Vector{Float64}
    rho::Float64
    A2::Matrix{Float64}
    rt_data::RtData3D
end

struct VerifiedLGLaplaceLowerBound3D
    eig_lower::Vector{Interval{Float64}}
    eig_upper::Vector{Interval{Float64}}
    cr_eig_lower::Vector{Interval{Float64}}
    rho::Interval{Float64}
    A2::Matrix{Interval{Float64}}
end

function cr_liu_lower_bounds_3d(m::Mesh3D, neig::Integer)
    Mcr, Acr, info = create_matrix_crouzeix_raviart_3d(m)
    int = info.interior_dofs
    n_int = length(int)
    n_int ≥ neig || throw(ArgumentError("not enough CR interior DOFs"))
    vals = eigen(Symmetric(Matrix(Acr[int, int])), Symmetric(Matrix(Mcr[int, int]))).values
    vals = sort(real(vals))[1:min(neig, length(vals))]
    Ch = _LIU_C3D_CR * find_mesh_hmax_3d(m)
    lows = vals ./ (1 .+ vals .* Ch^2)
    return lows, Ch, vals
end

function verified_cr_liu_lower_3d(m::Mesh3D, neig::Integer)
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))
    r = verified_cr_laplace_3d(m, neig)
    hmax_int = find_mesh_hmax_3d(interval.(m.NodeList), m.ElementList)
    Ch = interval(_LIU_C3D_CR) * hmax_int
    Ch2 = Ch * Ch
    lows = [λ / (interval(1.0) + λ * Ch2) for λ in r.eig_value]
    return lows, Ch, r.eig_value
end

function _cg_to_dg_3d(v_cg::AbstractVector, L2G_CG::AbstractMatrix{<:Integer},
                      rt::RtData3D)
    NumElt = size(L2G_CG, 1)
    v_dg = zeros(Float64, rt.DimDG)
    if rt.DegK == 1
        p = 0
        for N in 0:12
            if simplex_dof(3, N) == size(L2G_CG, 2)
                p = N
                break
            end
        end
        α = ijkl_list(p)
        w = [6.0 * prod(factorial.(α[i, :])) / factorial(p + 3) for i in axes(α, 1)]
        for e in 1:NumElt
            v_dg[e] = dot(v_cg[L2G_CG[e, :]], w)
        end
    else
        for e in 1:NumElt
            v_dg[(e - 1) * rt.DegK .+ (1:rt.DegK)] .= v_cg[L2G_CG[e, 1:rt.DegK]]
        end
    end
    return v_dg
end

function rt_hdiv_problem_3d(m::Mesh3D, degree_rt::Integer,
                            eig_func::AbstractMatrix,
                            L2G_CG::AbstractMatrix{<:Integer};
                            rt_data::Union{Nothing, RtData3D} = nothing)
    degree_rt ≥ 1 || throw(DomainError(degree_rt, "degree_rt must be ≥ 1"))
    rt = rt_data === nothing ? create_matrix_rt_3d(m, degree_rt) : rt_data
    # Direct dense solves are fine for current debug-scale meshes. The
    # matrix path mirrors MATLAB's A_rt inverse in the Schur operator.
    Ainv = inv(Matrix(rt.A_rt))
    S = Matrix(rt.B_rt) * Ainv * Matrix(rt.B_rt)'
    A2 = zeros(Float64, size(eig_func, 2), size(eig_func, 2))
    W = zeros(Float64, rt.DimRT, size(eig_func, 2))
    for k in axes(eig_func, 2)
        v_dg = _cg_to_dg_3d(@view(eig_func[:, k]), L2G_CG, rt)
        rhs = Matrix(rt.M_dg) * v_dg
        pvec = S \ rhs
        W[:, k] .= -Ainv * Matrix(rt.B_rt)' * pvec
        res = norm(Matrix(rt.B_rt) * W[:, k] + rhs) / max(norm(rhs), eps())
        res < 1e-8 || @warn "3D RT auxiliary divergence residual is large" k res
    end
    A2 .= W' * Matrix(rt.A_rt) * W
    return A2, rt, W
end

function _apply_dense_inverse_3d(R::Matrix{Float64},
                                 rhs::AbstractVector{<:Interval})
    n = size(R, 1)
    out = Vector{eltype(rhs)}(undef, n)
    @inbounds for i in 1:n
        acc = zero(eltype(rhs))
        for j in 1:n
            acc += R[i, j] * rhs[j]
        end
        out[i] = acc
    end
    return out
end

function verified_rt_hdiv_problem_3d(m::Mesh3D, degree_rt::Integer,
                                     eig_func::AbstractMatrix,
                                     L2G_CG::AbstractMatrix{<:Integer};
                                     rt_data::Union{Nothing, RtData3D} = nothing)
    degree_rt ≥ 1 || throw(DomainError(degree_rt, "degree_rt must be ≥ 1"))
    rt = rt_data === nothing ? create_matrix_rt_3d(m, degree_rt) : rt_data
    T = Interval{Float64}
    ncols = size(eig_func, 2)
    A_int = interval.(Matrix(rt.A_rt))
    B_int = interval.(Matrix(rt.B_rt))
    M_int = interval.(Matrix(rt.M_dg))

    K_int = [A_int             transpose(B_int);
             B_int             zeros(T, rt.DimDG, rt.DimDG)]
    K_mid = map(inf, K_int)
    R = inv(K_mid)
    W_int = Matrix{T}(undef, rt.DimRT, ncols)

    for k in axes(eig_func, 2)
        v_dg = _cg_to_dg_3d(@view(eig_func[:, k]), L2G_CG, rt)
        rhs_int = vcat(zeros(T, rt.DimRT), -(M_int * v_dg))
        rhs_mid = map(inf, rhs_int)
        x_a = K_mid \ rhs_mid
        res_int = rhs_int .- K_int * x_a
        dx_int = _apply_dense_inverse_3d(R, res_int)
        x_int = interval.(x_a) .+ dx_int
        W_int[:, k] .= x_int[1:rt.DimRT]
    end

    A2_int = W_int' * A_int * W_int
    return A2_int, rt, W_int
end

function lg_lower_eig_bound_laplace_3d(m::Mesh3D, degree::Integer,
                                       neig::Integer;
                                       RT_order::Integer = degree,
                                       rho_index::Integer = neig + 1,
                                       rho::Union{Nothing, Real} = nothing)
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))
    cr_low = if rho === nothing
        vals, _, _ = cr_liu_lower_bounds_3d(m, rho_index)
        length(vals) ≥ rho_index || error("not enough CR lower bounds for rho")
        vals
    else
        Float64[]
    end
    rho_val = rho === nothing ? cr_low[rho_index] : Float64(rho)
    A, M, info, L2G = create_matrix_lagrange_3d(m, degree)
    int = info.interior_dofs
    F = eigen(Symmetric(Matrix(A[int, int])), Symmetric(Matrix(M[int, int])))
    vals = F.values[1:neig]
    eig_func = zeros(Float64, info.DimCG, neig)
    eig_func[int, :] .= F.vectors[:, 1:neig]
    A2, rt, _ = rt_hdiv_problem_3d(m, RT_order, eig_func, L2G)
    A0 = eig_func' * A * eig_func
    A1 = eig_func' * M * eig_func
    AL = A0 .- rho_val .* A1
    BL = A0 .- 2rho_val .* A1 .+ rho_val^2 .* A2
    μ = sort(real.(eigvals((AL + AL') / 2, (BL + BL') / 2)))
    lows = similar(μ)
    for k in eachindex(μ)
        ν = μ[end - k + 1]
        lows[k] = ν < 0 ? rho_val - rho_val / (1 - ν) : -Inf
    end
    return LGLaplaceLowerBound3D(lows, collect(vals), cr_low, rho_val, A2, rt)
end

function verified_lg_lower_eig_bound_laplace_3d(m::Mesh3D, degree::Integer,
                                                neig::Integer;
                                                RT_order::Integer = degree,
                                                rho_index::Integer = neig + 1,
                                                rho::Union{Nothing, Real, Interval{Float64}} = nothing)
    r = lg_lower_eig_bound_laplace_3d(m, degree, neig;
                                      RT_order = RT_order,
                                      rho_index = rho_index,
                                      rho = rho)
    cr_low_int = if rho === nothing
        first(verified_cr_liu_lower_3d(m, rho_index))
    else
        interval.(r.cr_eig_lower)
    end
    rho_int = rho === nothing ? cr_low_int[rho_index] : interval(rho)
    A, M, info, L2G = create_matrix_lagrange_3d(m, degree; T = Interval{Float64})
    # Use the approximate eigenvectors from the Float64 driver as the
    # projection basis, but assemble projected CG matrices in intervals.
    r_cg = laplace_eig_lagrange_3d(m, degree, neig)
    V = r_cg.eig_func
    A0 = V' * A * V
    A1 = V' * M * V
    A2, _, _ = verified_rt_hdiv_problem_3d(m, RT_order, V, L2G; rt_data = r.rt_data)
    AL = A0 .- rho_int .* A1
    BL = A0 .- interval(2.0) .* rho_int .* A1 .+ rho_int^2 .* A2
    lows = verified_lg_transform(AL, BL, rho_int)
    upper_int = interval.(r.eig_upper)
    return VerifiedLGLaplaceLowerBound3D(lows[1:neig], upper_int,
                                         cr_low_int, rho_int, A2)
end
