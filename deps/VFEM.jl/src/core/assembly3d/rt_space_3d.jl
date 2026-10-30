# src/core/assembly3d/rt_space_3d.jl
#
# Scalar Raviart-Thomas mixed matrices on tetrahedra, ported from
# VFEM3D/build_scalar_rt_matrices.m. This Float64 path is the parity
# target for the MATLAB-vs-Julia migration; interval wrapping is a later
# layer after the local RT transformation is cross-checked.

using LinearAlgebra: Diagonal, I, Symmetric, dot, eigvals, inv, norm
using SparseArrays: sparse, nnz, SparseMatrixCSC

struct RtData3D
    A_rt::SparseMatrixCSC{Float64, Int}
    B_rt::SparseMatrixCSC{Float64, Int}
    M_dg::SparseMatrixCSC{Float64, Int}
    DimRT::Int
    DimDG::Int
    DegK::Int
    DegF::Int
    DegRTInner::Int
    DegRTElt::Int
end

_rt3d_degf(M::Integer) = simplex_dof(2, M)
_rt3d_degk(M::Integer) = simplex_dof(3, M)
_rt3d_inner(M::Integer) = M == 0 ? 0 : 3 * simplex_dof(3, M - 1)
_rt3d_elt(M::Integer) = 4 * _rt3d_degf(M) + _rt3d_inner(M)

@inline _cross3(a, b) = (a[2] * b[3] - a[3] * b[2],
                         a[3] * b[1] - a[1] * b[3],
                         a[1] * b[2] - a[2] * b[1])

function _rt3d_outward_normals(P::AbstractMatrix{Float64})
    local_faces = ((2, 3, 4), (1, 3, 4), (1, 2, 4), (1, 2, 3))
    N = zeros(Float64, 4, 3)
    for f in 1:4
        F = local_faces[f]
        a = P[F[2], :] .- P[F[1], :]
        b = P[F[3], :] .- P[F[1], :]
        last = 10 - sum(F)
        eout = P[last, :] .- P[F[1], :]
        n = collect(_cross3(a, b))
        dot(eout, n) > 0 && (n .*= -1)
        N[f, :] .= n ./ norm(n)
    end
    return N
end

function _mat_degree_up_3d(M::Integer)
    M ≥ 0 || throw(DomainError(M, "M must be ≥ 0"))
    α = ijkl_list(M)
    βidx = ijkl_index_map(M + 1)
    out = zeros(Float64, simplex_dof(3, M + 1), simplex_dof(3, M))
    for k in axes(α, 1)
        a = (α[k, 1], α[k, 2], α[k, 3], α[k, 4])
        for li in 1:4
            key = (a[1] + (li == 1), a[2] + (li == 2),
                   a[3] + (li == 3), a[4] + (li == 4))
            out[βidx[key], k] = 1.0
        end
    end
    return out
end

function _mat_degree_up_by_x_3d(M::Integer, P::AbstractMatrix{Float64})
    M ≥ 0 || throw(DomainError(M, "M must be ≥ 0"))
    α = ijkl_list(M)
    βidx = ijkl_index_map(M + 1)
    out = zeros(Float64, simplex_dof(3, M + 1), simplex_dof(3, M), 3)
    for k in axes(α, 1)
        a = (α[k, 1], α[k, 2], α[k, 3], α[k, 4])
        for li in 1:4
            key = (a[1] + (li == 1), a[2] + (li == 2),
                   a[3] + (li == 3), a[4] + (li == 4))
            row = βidx[key]
            for d in 1:3
                out[row, k, d] = P[li, d]
            end
        end
    end
    return out
end

function _grad_lambda_rt3d(P::AbstractMatrix{Float64})
    J = [P[2, 1] - P[1, 1]  P[2, 2] - P[1, 2]  P[2, 3] - P[1, 3];
         P[3, 1] - P[1, 1]  P[3, 2] - P[1, 2]  P[3, 3] - P[1, 3];
         P[4, 1] - P[1, 1]  P[4, 2] - P[1, 2]  P[4, 3] - P[1, 3]]
    Ji = inv(J)
    G = zeros(Float64, 4, 3)
    for k in 1:3
        G[2, k] = Ji[k, 1]
        G[3, k] = Ji[k, 2]
        G[4, k] = Ji[k, 3]
    end
    G[1, :] .= .-(G[2, :] .+ G[3, :] .+ G[4, :])
    return G
end

function _grad_mat_3d(M::Integer, P::AbstractMatrix{Float64})
    M ≥ 1 || throw(DomainError(M, "M must be ≥ 1"))
    return _lagrange3d_grad_coeffs(M, _grad_lambda_rt3d(P))
end

function _div_mat_rt3d(M::Integer, P::AbstractMatrix{Float64})
    M ≥ 1 || throw(DomainError(M, "M must be ≥ 1"))
    DegK = simplex_dof(3, M)
    DegF = simplex_dof(2, M)
    DegRTElt = _rt3d_elt(M)
    out = zeros(Float64, DegK, DegRTElt)
    MatDegreeUp = _mat_degree_up_3d(M - 1)
    Grad = _grad_mat_3d(M, P)
    out[:, 1:DegK] .= MatDegreeUp * Grad[1]
    out[:, DegK .+ (1:DegK)] .= MatDegreeUp * Grad[2]
    out[:, 2 * DegK .+ (1:DegK)] .= MatDegreeUp * Grad[3]
    sub = dof_on_facet(M, 4)
    dcols = 3 * DegK .+ (1:DegF)
    out[sub, dcols] .+= 3.0 .* Matrix{Float64}(I, DegF, DegF)
    Xup = _mat_degree_up_by_x_3d(M - 1, P)
    for k in 1:3
        out[:, dcols] .+= Xup[:, :, k] * Grad[k][:, sub]
    end
    return out
end

function _element_trans_mat_rt3d(P::AbstractMatrix{Float64}, vol::Float64, M::Integer)
    M ≥ 1 || throw(DomainError(M, "RT degree M must be ≥ 1 in this port"))
    DegRTElt = _rt3d_elt(M)
    DegF = simplex_dof(2, M)
    DegK = simplex_dof(3, M)
    NormVec = _rt3d_outward_normals(P)

    S = ntuple(i -> dof_on_facet(M, i), 4)
    F = zeros(Float64, DegRTElt, DegRTElt)

    # Face normal moment functionals.
    for face in 1:4
        rows = (face - 1) * DegF .+ (1:DegF)
        F[rows, S[face]] .= NormVec[face, 1] .* Matrix{Float64}(I, DegF, DegF)
        F[rows, DegK .+ S[face]] .= NormVec[face, 2] .* Matrix{Float64}(I, DegF, DegF)
        F[rows, 2 * DegK .+ S[face]] .= NormVec[face, 3] .* Matrix{Float64}(I, DegF, DegF)
    end

    s1, s4 = common_dof_on_facets(M, 1, 4)
    F[s1, 3 * DegK .+ s4] .= dot(P[2, :], NormVec[1, :]) .* Matrix{Float64}(I, M + 1, M + 1)
    s2, s4 = common_dof_on_facets(M, 2, 4)
    F[DegF .+ s2, 3 * DegK .+ s4] .= dot(P[3, :], NormVec[2, :]) .* Matrix{Float64}(I, M + 1, M + 1)
    s3, s4 = common_dof_on_facets(M, 3, 4)
    F[2 * DegF .+ s3, 3 * DegK .+ s4] .= dot(P[4, :], NormVec[3, :]) .* Matrix{Float64}(I, M + 1, M + 1)
    F[3 * DegF .+ (1:DegF), 3 * DegK .+ (1:DegF)] .= dot(P[1, :], NormVec[4, :]) .* Matrix{Float64}(I, DegF, DegF)

    # Interior moments.
    DegMinus = simplex_dof(3, M - 1)
    rows1 = 4 * DegF .+ (1:DegMinus)
    rows2 = 4 * DegF + DegMinus .+ (1:DegMinus)
    rows3 = 4 * DegF + 2 * DegMinus .+ (1:DegMinus)
    A_m1_m = inner_prod_matrix(M - 1, M, vol)
    A_m1_mp1 = inner_prod_matrix(M - 1, M + 1, vol)
    Xup = _mat_degree_up_by_x_3d(M, P)
    Xh = Xup[:, S[4], :]
    dcols = 3 * DegK .+ (1:DegF)
    F[rows1, 1:DegK] .= A_m1_m
    F[rows1, dcols] .= A_m1_mp1 * Xh[:, :, 1]
    F[rows2, DegK .+ (1:DegK)] .= A_m1_m
    F[rows2, dcols] .= A_m1_mp1 * Xh[:, :, 2]
    F[rows3, 2 * DegK .+ (1:DegK)] .= A_m1_m
    F[rows3, dcols] .= A_m1_mp1 * Xh[:, :, 3]

    row_scale = vec(maximum(abs.(F), dims = 2))
    any(iszero, row_scale) && throw(ArgumentError("Singular RT functional matrix"))
    Fs = Diagonal(1.0 ./ row_scale) * F
    return inv(Fs)
end

function create_matrix_rt_3d(m::Mesh3D, degree::Integer)
    degree ≥ 1 || throw(DomainError(degree, "RT degree must be ≥ 1"))
    Mdeg = Int(degree)
    _, _, ESign = facet_element_connectivity_with_sign(m.ElementList, m.FacetList)
    DegK = simplex_dof(3, Mdeg)
    DegF = simplex_dof(2, Mdeg)
    DegRTInner = _rt3d_inner(Mdeg)
    DegRTElt = _rt3d_elt(Mdeg)
    DOF_Facet = m.NumF * DegF
    DimRT = DOF_Facet + m.NumElt * DegRTInner
    DimDG = m.NumElt * DegK

    nnzA = m.NumElt * DegRTElt^2
    nnzB = m.NumElt * DegK * DegRTElt
    nnzC = m.NumElt * DegK^2
    iiA = Vector{Int}(undef, nnzA); jjA = similar(iiA); vvA = Vector{Float64}(undef, nnzA)
    iiB = Vector{Int}(undef, nnzB); jjB = similar(iiB); vvB = Vector{Float64}(undef, nnzB)
    iiC = Vector{Int}(undef, nnzC); jjC = similar(iiC); vvC = Vector{Float64}(undef, nnzC)
    pA = 0; pB = 0; pC = 0

    A_MM_ref = inner_prod_matrix_reference(Mdeg, Mdeg)
    A_M_MP_ref = inner_prod_matrix_reference(Mdeg, Mdeg + 1)
    A_MP_MP_ref = inner_prod_matrix_reference(Mdeg + 1, Mdeg + 1)
    sub_homo = dof_on_facet(Mdeg, 4)

    for e in 1:m.NumElt
        P = Matrix{Float64}(m.NodeList[m.ElementList[e, :], :])
        vol = Float64(_tet_element_volume(m, e, Float64))
        A_MM = A_MM_ref .* vol
        A_M_MP = A_M_MP_ref .* vol
        A_MP_MP = A_MP_MP_ref .* vol
        Xup = _mat_degree_up_by_x_3d(Mdeg, P)
        Xh = Xup[:, sub_homo, :]

        L2G = zeros(Int, DegRTElt)
        neg = falses(4 * DegF)
        for f in 1:4
            fdofs = (f - 1) * DegF .+ (1:DegF)
            gf = m.Element2Facet[e, f]
            L2G[fdofs] .= (gf - 1) * DegF .+ (1:DegF)
            ESign[e, f] < 0 && (neg[fdofs] .= true)
        end
        if DegRTInner > 0
            L2G[4 * DegF .+ (1:DegRTInner)] .= DOF_Facet + (e - 1) * DegRTInner .+ (1:DegRTInner)
        end
        neg_idx = findall(neg)
        dg = (e - 1) * DegK .+ (1:DegK)

        LocalA_poly = zeros(Float64, DegRTElt, DegRTElt)
        ia = 1:DegK; ib = DegK .+ (1:DegK); ic = 2 * DegK .+ (1:DegK)
        id = 3 * DegK .+ (1:DegF)
        LocalA_poly[ia, ia] .= A_MM
        LocalA_poly[ib, ib] .= A_MM
        LocalA_poly[ic, ic] .= A_MM
        Dblock = zeros(Float64, DegF, DegF)
        for k in 1:3
            Dblock .+= Xh[:, :, k]' * A_MP_MP * Xh[:, :, k]
        end
        LocalA_poly[id, id] .= Dblock
        LocalA_poly[ia, id] .= A_M_MP * Xh[:, :, 1]
        LocalA_poly[ib, id] .= A_M_MP * Xh[:, :, 2]
        LocalA_poly[ic, id] .= A_M_MP * Xh[:, :, 3]
        LocalA_poly[id, ia] .= LocalA_poly[ia, id]'
        LocalA_poly[id, ib] .= LocalA_poly[ib, id]'
        LocalA_poly[id, ic] .= LocalA_poly[ic, id]'

        Tmat = _element_trans_mat_rt3d(P, vol, Mdeg)
        LocalA = Tmat' * LocalA_poly * Tmat
        LocalA[neg_idx, :] .*= -1
        LocalA[:, neg_idx] .*= -1

        for j in 1:DegRTElt, i in 1:DegRTElt
            pA += 1
            iiA[pA] = L2G[i]; jjA[pA] = L2G[j]; vvA[pA] = LocalA[i, j]
        end

        LocalDiv = Tmat' * _div_mat_rt3d(Mdeg, P)' * A_MM
        LocalDiv[neg_idx, :] .*= -1
        LocalB = LocalDiv'
        for j in 1:DegRTElt, i in 1:DegK
            pB += 1
            iiB[pB] = dg[i]; jjB[pB] = L2G[j]; vvB[pB] = LocalB[i, j]
        end

        for j in 1:DegK, i in 1:DegK
            pC += 1
            iiC[pC] = dg[i]; jjC[pC] = dg[j]; vvC[pC] = A_MM[i, j]
        end
    end

    return RtData3D(
        sparse(iiA[1:pA], jjA[1:pA], vvA[1:pA], DimRT, DimRT),
        sparse(iiB[1:pB], jjB[1:pB], vvB[1:pB], DimDG, DimRT),
        sparse(iiC[1:pC], jjC[1:pC], vvC[1:pC], DimDG, DimDG),
        DimRT, DimDG, DegK, DegF, DegRTInner, DegRTElt)
end

function debug_rt_3d(m::Mesh3D, degree::Integer)
    rt = create_matrix_rt_3d(m, degree)
    Amid = Matrix(rt.A_rt)
    Mdg = Matrix(rt.M_dg)
    return (degree = Int(degree),
            DimRT = rt.DimRT,
            DimDG = rt.DimDG,
            DegK = rt.DegK,
            DegF = rt.DegF,
            DegRTInner = rt.DegRTInner,
            DegRTElt = rt.DegRTElt,
            nnz_A = nnz(rt.A_rt),
            nnz_B = nnz(rt.B_rt),
            nnz_Mdg = nnz(rt.M_dg),
            symmetric_A = rt.A_rt ≈ rt.A_rt',
            symmetric_Mdg = rt.M_dg ≈ rt.M_dg',
            min_eig_A = minimum(eigvals(Symmetric(Amid))),
            min_eig_Mdg = minimum(eigvals(Symmetric(Mdg))))
end
