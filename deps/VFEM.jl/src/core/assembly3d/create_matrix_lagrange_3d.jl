# src/core/assembly3d/create_matrix_lagrange_3d.jl
#
# Conforming scalar Lagrange space on tetrahedra for arbitrary degree p >= 1.
# Local basis is the monomial barycentric basis L^alpha ordered by
# `ijkl_list(p)`, matching the MATLAB VFEM3D polynomial convention.

using Arpack: eigs
using LinearAlgebra: Symmetric, dot, eigen, eigvals
using SparseArrays: spzeros, sparse, SparseMatrixCSC, nnz, dropzeros!

"""
    LagrangeDof3D

DOF metadata for 3D conforming Lagrange space.
"""
struct LagrangeDof3D
    degree::Int
    DegK::Int
    DimCG::Int
    bd_dofs::Vector{Int}
    interior_dofs::Vector{Int}
    vertex_dofs::UnitRange{Int}
    edge_dofs::UnitRange{Int}
    face_dofs::UnitRange{Int}
    cell_dofs::UnitRange{Int}
end

function _edge_lookup_3d(m::Mesh3D)
    d = Dict{Tuple{Int, Int}, Int}()
    sizehint!(d, m.NumEdge)
    @inbounds for e in 1:m.NumEdge
        d[(m.EdgeList[e, 1], m.EdgeList[e, 2])] = e
    end
    return d
end

function _face_lookup_3d(m::Mesh3D)
    d = Dict{NTuple{3, Int}, Int}()
    sizehint!(d, m.NumF)
    @inbounds for f in 1:m.NumF
        d[(m.FacetList[f, 1], m.FacetList[f, 2], m.FacetList[f, 3])] = f
    end
    return d
end

"""
    lagrange_l2g_3d(m::Mesh3D, degree::Integer) -> (L2G, info)

Build local-to-global DOF mapping for conforming tetrahedral Lagrange
degree `degree`. Shared vertex, edge, and face DOFs are canonicalized by
global mesh entity; tetrahedron-interior DOFs are element-local.
"""
function lagrange_l2g_3d(m::Mesh3D, degree::Integer)
    degree ≥ 1 || throw(ArgumentError("Lagrange degree must be ≥ 1"))
    p = Int(degree)
    α = ijkl_list(p)
    DegK = size(α, 1)
    L2G = zeros(Int, m.NumElt, DegK)

    next_dof = m.NumNode + 1
    vertex_range = 1:m.NumNode

    edge_lookup = _edge_lookup_3d(m)
    edge_dof = Dict{Tuple{Int, Int}, Int}()
    for ed in 1:m.NumEdge, s in 1:(p - 1)
        edge_dof[(ed, s)] = next_dof
        next_dof += 1
    end
    edge_range = (m.NumNode + 1):(next_dof - 1)

    face_start = next_dof
    face_dof = Dict{Tuple{Int, NTuple{3, Int}}, Int}()
    face_lookup = _face_lookup_3d(m)
    @inbounds for f in 1:m.NumF
        for a in 1:(p - 1), b in 1:(p - a - 1)
            c = p - a - b
            c ≥ 1 || continue
            face_dof[(f, (a, b, c))] = next_dof
            next_dof += 1
        end
    end
    face_range = face_start:(next_dof - 1)

    cell_start = next_dof
    bd_flag = falses(10)  # resized below after final dimension is known.

    @inbounds for e in 1:m.NumElt
        nodes = (m.ElementList[e, 1], m.ElementList[e, 2],
                 m.ElementList[e, 3], m.ElementList[e, 4])
        for d in 1:DegK
            aa = (α[d, 1], α[d, 2], α[d, 3], α[d, 4])
            nz = Int[]
            for i in 1:4
                aa[i] > 0 && push!(nz, i)
            end
            if length(nz) == 1
                L2G[e, d] = nodes[nz[1]]
            elseif length(nz) == 2
                i, j = nz[1], nz[2]
                vi, vj = nodes[i], nodes[j]
                mn, mx = minmax(vi, vj)
                ed = edge_lookup[(mn, mx)]
                frac_num = vi < vj ? aa[j] : aa[i]
                L2G[e, d] = edge_dof[(ed, frac_num)]
            elseif length(nz) == 3
                face_nodes = sort([nodes[nz[1]], nodes[nz[2]], nodes[nz[3]]])
                f = face_lookup[(face_nodes[1], face_nodes[2], face_nodes[3])]
                coeff_by_node = Dict(nodes[i] => aa[i] for i in nz)
                key = (coeff_by_node[face_nodes[1]],
                       coeff_by_node[face_nodes[2]],
                       coeff_by_node[face_nodes[3]])
                L2G[e, d] = face_dof[(f, key)]
            else
                L2G[e, d] = next_dof
                next_dof += 1
            end
        end
    end

    DimCG = next_dof - 1
    cell_range = cell_start:DimCG
    bd_flag = falses(DimCG)
    @inbounds for f in 1:m.NumF
        m.Facet2Element[f, 2] == 0 || continue
        e = m.Facet2Element[f, 1]
        loc = facet_id_at_element(@view(m.FacetList[f, :]), @view(m.ElementList[e, :]))
        for d in dof_on_facet(p, loc)
            bd_flag[L2G[e, d]] = true
        end
    end
    bd = findall(bd_flag)
    interior = findall(!, bd_flag)

    return L2G, LagrangeDof3D(p, DegK, DimCG, bd, interior, vertex_range,
                              edge_range, face_range, cell_range)
end

function _lagrange3d_grad_coeffs(degree::Integer, grad_L::AbstractMatrix{T}) where {T<:Real}
    p = Int(degree)
    α = ijkl_list(p)
    β = ijkl_list(p - 1)
    β_idx = ijkl_index_map(p - 1)
    G = ntuple(_ -> zeros(T, size(β, 1), size(α, 1)), 3)
    @inbounds for aidx in axes(α, 1)
        aa = (α[aidx, 1], α[aidx, 2], α[aidx, 3], α[aidx, 4])
        for ell in 1:4
            aa[ell] == 0 && continue
            key = (aa[1] - (ell == 1), aa[2] - (ell == 2),
                   aa[3] - (ell == 3), aa[4] - (ell == 4))
            bidx = β_idx[key]
            for dim in 1:3
                G[dim][bidx, aidx] += T(aa[ell]) * grad_L[ell, dim]
            end
        end
    end
    return G
end

"""
    create_matrix_lagrange_3d(m, degree; T = Float64) -> (A, M, info, L2G)

Assemble conforming 3D Lagrange stiffness and mass matrices before
Dirichlet restriction.
"""
function create_matrix_lagrange_3d(m::Mesh3D, degree::Integer; T::Type = Float64)
    L2G, info = lagrange_l2g_3d(m, degree)
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = m.NumElt * info.DegK * info.DegK
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Aval = Vector{T}(undef, nent);   Mval = Vector{T}(undef, nent)
    pos = 0
    Mref = inner_prod_matrix_reference(degree, degree; T = T)
    Gref = inner_prod_matrix_reference(degree - 1, degree - 1; T = T)

    @inbounds for e in 1:m.NumElt
        v1 = m.ElementList[e, 1]; v2 = m.ElementList[e, 2]
        v3 = m.ElementList[e, 3]; v4 = m.ElementList[e, 4]
        P = T[m.NodeList[v1, 1] m.NodeList[v1, 2] m.NodeList[v1, 3];
              m.NodeList[v2, 1] m.NodeList[v2, 2] m.NodeList[v2, 3];
              m.NodeList[v3, 1] m.NodeList[v3, 2] m.NodeList[v3, 3];
              m.NodeList[v4, 1] m.NodeList[v4, 2] m.NodeList[v4, 3]]
        Jmat = T[P[2, 1] - P[1, 1]  P[2, 2] - P[1, 2]  P[2, 3] - P[1, 3];
                 P[3, 1] - P[1, 1]  P[3, 2] - P[1, 2]  P[3, 3] - P[1, 3];
                 P[4, 1] - P[1, 1]  P[4, 2] - P[1, 2]  P[4, 3] - P[1, 3]]
        vol = abs(_det3(Jmat)) / T(6)
        vol == zero(T) && throw(DomainError(vol, "Degenerate tetrahedron"))

        Jinv = _inv3(Jmat)
        grad_L = Matrix{T}(undef, 4, 3)
        for k in 1:3
            grad_L[2, k] = Jinv[k, 1]
            grad_L[3, k] = Jinv[k, 2]
            grad_L[4, k] = Jinv[k, 3]
        end
        for k in 1:3
            grad_L[1, k] = -(grad_L[2, k] + grad_L[3, k] + grad_L[4, k])
        end

        G = _lagrange3d_grad_coeffs(degree, grad_L)
        Kloc = vol .* (G[1]' * Gref * G[1] +
                       G[2]' * Gref * G[2] +
                       G[3]' * Gref * G[3])
        Mloc = vol .* Mref
        dofs = @view L2G[e, :]
        for j in 1:info.DegK, i in 1:info.DegK
            pos += 1
            Irow[pos] = dofs[i]; Jcol[pos] = dofs[j]
            Aval[pos] = Kloc[i, j]
            Mval[pos] = Mloc[i, j]
        end
    end
    A = dropzeros!(sparse(Irow, Jcol, Aval, info.DimCG, info.DimCG))
    M = dropzeros!(sparse(Irow, Jcol, Mval, info.DimCG, info.DimCG))
    return A, M, info, L2G
end

struct LaplaceEigLagrange3D
    eig_value::Vector{Float64}
    eig_func::Matrix{Float64}
    A::SparseMatrixCSC{Float64, Int}
    M::SparseMatrixCSC{Float64, Int}
    bd_dofs::Vector{Int}
    dof::LagrangeDof3D
end

function laplace_eig_lagrange_3d(m::Mesh3D, degree::Integer, neig::Integer)
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))
    A, M, info, _ = create_matrix_lagrange_3d(m, degree)
    int_dofs = info.interior_dofs
    n_int = length(int_dofs)
    n_int ≥ 1 || throw(ArgumentError("Dirichlet CG space has no interior DOFs"))
    k = min(Int(neig), n_int)
    Ared = Matrix(A[int_dofs, int_dofs])
    Mred = Matrix(M[int_dofs, int_dofs])
    vals, vecs = if n_int ≤ 1000 || k == n_int
        F = eigen(Symmetric(Ared), Symmetric(Mred))
        F.values[1:k], F.vectors[:, 1:k]
    else
        λ, V = eigs(A[int_dofs, int_dofs], M[int_dofs, int_dofs];
                   nev = k, which = :SM, tol = 1e-10, maxiter = 1000)
        perm = sortperm(real.(λ))
        real.(λ[perm]), real.(V[:, perm])
    end
    eig_func = zeros(Float64, info.DimCG, k)
    eig_func[int_dofs, :] .= vecs
    return LaplaceEigLagrange3D(collect(vals), eig_func, A, M, info.bd_dofs, info)
end

function debug_lagrange_3d(m::Mesh3D, degree::Integer)
    A, M, info, L2G = create_matrix_lagrange_3d(m, degree)
    c = zeros(Float64, info.DimCG)
    local_one = Float64.(bernstein_multinomial_3d(degree, ijkl_list(degree)))
    @inbounds for e in 1:m.NumElt
        c[L2G[e, :]] .= local_one
    end
    Aint = Matrix(A[info.interior_dofs, info.interior_dofs])
    Mint = Matrix(M[info.interior_dofs, info.interior_dofs])
    return (degree = info.degree,
            DegK = info.DegK,
            DimCG = info.DimCG,
            n_boundary = length(info.bd_dofs),
            n_interior = length(info.interior_dofs),
            nnz_A = nnz(A),
            nnz_M = nnz(M),
            symmetric_A = Matrix(A) ≈ Matrix(A)',
            symmetric_M = Matrix(M) ≈ Matrix(M)',
            constant_stiffness_norm = maximum(abs, A * c),
            constant_mass = dot(c, M * c),
            min_eig_A_int = isempty(Aint) ? NaN : minimum(eigvals(Symmetric(Aint))),
            min_eig_M_int = isempty(Mint) ? NaN : minimum(eigvals(Symmetric(Mint))))
end
