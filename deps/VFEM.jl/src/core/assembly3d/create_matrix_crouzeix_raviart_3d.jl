# src/core/assembly3d/create_matrix_crouzeix_raviart_3d.jl
#
# Crouzeix-Raviart P1 element on tetrahedra. CR DOFs live on triangular
# facets; one DOF per facet in `m.FacetList` order. The local basis on a
# tetrahedron is
#
#     phi_i = 1 - 3 L_i,  i = 1..4,
#
# where L_i is the barycentric coordinate of vertex i. Its average on the
# face opposite vertex i is 1 and its averages on the other faces vanish.
#
# The routine is generic in the scalar type T. With T = Interval{Float64}
# and exactly represented mesh coordinates, the returned sparse matrices
# are interval enclosures of the exact CR mass and stiffness matrices.

using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!

"""
    CrDof3D

DOF info for [`create_matrix_crouzeix_raviart_3d`](@ref):
* `ndof`          :: `Int`       - total facet DOF count.
* `NumF`          :: `Int`       - number of facets.
* `boundary_dofs` :: `Vector{Int}` - facets with only one adjacent element.
* `interior_dofs` :: `Vector{Int}` - facets with two adjacent elements.
"""
struct CrDof3D
    ndof::Int
    NumF::Int
    boundary_dofs::Vector{Int}
    interior_dofs::Vector{Int}
end

function _cr3d_dof_info(m::Mesh3D)
    boundary = findall(==(0), @view m.Facet2Element[:, 2])
    interior = setdiff(collect(1:m.NumF), boundary)
    return CrDof3D(m.NumF, m.NumF, boundary, interior)
end

"""
    create_matrix_crouzeix_raviart_3d(m::Mesh3D; T::Type = Float64)
        -> (M::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, info::CrDof3D)

Assemble the 3D CR mass `M = (phi_i, phi_j)` and stiffness
`A = (grad phi_i, grad phi_j)` matrices on the tetrahedral mesh `m`.
The global DOFs are facets ordered as `m.FacetList`.

Dirichlet boundary conditions are not applied in this assembler. Use
`info.interior_dofs` to restrict the pencil before solving.
"""
function create_matrix_crouzeix_raviart_3d(m::Mesh3D; T::Type = Float64)
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = m.NumElt * 16
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Aval = Vector{T}(undef, nent);   Mval = Vector{T}(undef, nent)
    pos = 0

    # For phi_i = 1 - 3 L_i:
    #   int_K phi_i^2     = 2|K|/5
    #   int_K phi_i phi_j = -|K|/20, i != j.
    mass_ref = Matrix{T}(undef, 4, 4)
    @inbounds for i in 1:4, j in 1:4
        mass_ref[i, j] = i == j ? T(2) / T(5) : -T(1) / T(20)
    end

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

        dof = (m.Element2Facet[e, 1], m.Element2Facet[e, 2],
               m.Element2Facet[e, 3], m.Element2Facet[e, 4])

        for i in 1:4, j in 1:4
            gdot = grad_L[i, 1] * grad_L[j, 1] +
                   grad_L[i, 2] * grad_L[j, 2] +
                   grad_L[i, 3] * grad_L[j, 3]
            pos += 1
            Irow[pos] = dof[i]; Jcol[pos] = dof[j]
            Mval[pos] = vol * mass_ref[i, j]
            Aval[pos] = T(9) * vol * gdot
        end
    end

    M = dropzeros!(sparse(Irow, Jcol, Mval, m.NumF, m.NumF))
    A = dropzeros!(sparse(Irow, Jcol, Aval, m.NumF, m.NumF))

    return M, A, _cr3d_dof_info(m)
end
