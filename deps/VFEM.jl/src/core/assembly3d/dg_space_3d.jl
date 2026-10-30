# src/core/assembly3d/dg_space_3d.jl
#
# Discontinuous scalar P_M space on tetrahedra. The local basis is the
# monomial barycentric basis ordered by `ijkl_list(M)`, matching VFEM3D
# MATLAB's `get_IJKL` convention.

using LinearAlgebra: Symmetric, dot, eigvals
using SparseArrays: spzeros, sparse, SparseMatrixCSC, nnz, dropzeros!

"""
    DgDof3D

DOF info for the 3D discontinuous scalar space:
* `degree` :: polynomial degree `M`.
* `DegK`   :: local DOFs per tetrahedron, `binomial(M+3, 3)`.
* `DimDG`  :: global DG dimension, `m.NumElt * DegK`.
"""
struct DgDof3D
    degree::Int
    DegK::Int
    DimDG::Int
end

_dg_mid_value(x::Real) = Float64(x)
_dg_mid_value(x::Interval) = Float64(mid(x))

"""
    dg_l2g_3d(m::Mesh3D, degree::Integer) -> (L2G, info)

Element-local to global mapping for scalar DG degree `degree`.
Each element owns its complete local polynomial block:
`L2G[e, :] = (e-1)*DegK .+ (1:DegK)`.
"""
function dg_l2g_3d(m::Mesh3D, degree::Integer)
    degree ≥ 0 || throw(DomainError(degree, "degree must be ≥ 0"))
    DegK = simplex_dof(3, degree)
    DimDG = m.NumElt * DegK
    L2G = Matrix{Int}(undef, m.NumElt, DegK)
    @inbounds for e in 1:m.NumElt
        base = (e - 1) * DegK
        for d in 1:DegK
            L2G[e, d] = base + d
        end
    end
    return L2G, DgDof3D(Int(degree), DegK, DimDG)
end

"""
    create_matrix_dg_3d(m::Mesh3D, degree::Integer; T::Type = Float64)
        -> (M::SparseMatrixCSC{T}, info::DgDof3D)

Assemble the scalar DG mass matrix `M_ij = ∫ q_i q_j dx`. The result is
block diagonal, one full `DegK × DegK` block per tetrahedron. Passing
`T = Interval{Float64}` produces interval matrix entries.
"""
function create_matrix_dg_3d(m::Mesh3D, degree::Integer; T::Type = Float64)
    L2G, info = dg_l2g_3d(m, degree)
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = m.NumElt * info.DegK * info.DegK
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Mval = Vector{T}(undef, nent); pos = 0
    @inbounds for e in 1:m.NumElt
        vol = _tet_element_volume(m, e, T)
        vol == zero(T) && throw(DomainError(vol, "Degenerate tetrahedron"))
        Mloc = inner_prod_matrix(degree, degree, vol)
        dofs = @view L2G[e, :]
        for j in 1:info.DegK, i in 1:info.DegK
            pos += 1
            Irow[pos] = dofs[i]; Jcol[pos] = dofs[j]; Mval[pos] = Mloc[i, j]
        end
    end
    M = dropzeros!(sparse(Irow, Jcol, Mval, info.DimDG, info.DimDG))
    return M, info
end

"""
    debug_dg_3d(m::Mesh3D, degree::Integer; T::Type = Float64) -> NamedTuple

Run cheap consistency checks for the scalar 3D DG space. This is meant as
a migration/debug helper and returns measured quantities instead of using
`@test`, so it can be called from scripts and MATLAB cross-check drivers.
"""
function debug_dg_3d(m::Mesh3D, degree::Integer; T::Type = Float64)
    M, info = create_matrix_dg_3d(m, degree; T = T)
    denseM = Matrix(M)
    c = Vector{T}(undef, info.DimDG)
    local_one = T.(bernstein_multinomial_3d(degree, ijkl_list(degree)))
    @inbounds for e in 1:m.NumElt
        c[((e - 1) * info.DegK + 1):(e * info.DegK)] .= local_one
    end
    domain_vol = zero(T)
    @inbounds for e in 1:m.NumElt
        domain_vol += _tet_element_volume(m, e, T)
    end
    denseM_mid = Matrix{Float64}(undef, size(denseM)...)
    @inbounds for j in axes(denseM, 2), i in axes(denseM, 1)
        denseM_mid[i, j] = _dg_mid_value(denseM[i, j])
    end
    min_eig_mid = minimum(eigvals(Symmetric(denseM_mid)))
    return (degree = info.degree,
            DegK = info.DegK,
            DimDG = info.DimDG,
            expected_DimDG = m.NumElt * simplex_dof(3, degree),
            nnz_mass = nnz(M),
            symmetric = denseM == denseM',
            constant_mass = dot(c, M * c),
            domain_volume = domain_vol,
            min_eig_mid = min_eig_mid)
end
