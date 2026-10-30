# src/core/eigensolve3d/verified_cr_laplace_3d.jl
#
# Verified discrete eigenvalue solve for the 3D Crouzeix-Raviart
# Dirichlet Laplacian. This verifies the finite-dimensional CR pencil
# A u = lambda M u; it does not apply a Liu post-processing lower bound
# for the continuous eigenvalue.

using IntervalArithmetic: Interval, interval, inf
import Veigs

"""
    CrLaplaceEig3D

Result from [`verified_cr_laplace_3d`](@ref):
* `eig_value` :: `Vector{Interval{Float64}}` - verified discrete CR eigenvalues.
* `M`         :: interval CR mass matrix before boundary restriction.
* `A`         :: interval CR stiffness matrix before boundary restriction.
* `dof`       :: CR DOF information.
"""
struct CrLaplaceEig3D
    eig_value::Vector{Interval{Float64}}
    M::SparseMatrixCSC{Interval{Float64}, Int}
    A::SparseMatrixCSC{Interval{Float64}, Int}
    dof::CrDof3D
end

"""
    verified_cr_laplace_3d(m::Mesh3D, neig::Integer) -> CrLaplaceEig3D

Assemble the 3D CR Dirichlet Laplace generalized eigenproblem in
interval arithmetic and compute verified enclosures for the first
`neig` discrete eigenvalues using `Veigs.veigs`.

Boundary facets are removed by restricting to `dof.interior_dofs`.
The mesh coordinates are treated as exactly given; coordinates such as
`0.5` and `0.25` are exactly representable in `Float64` and become
degenerate intervals under `interval.(...)`.
"""
function verified_cr_laplace_3d(m::Mesh3D, neig::Integer)
    neig ≥ 1 || throw(DomainError(neig, "neig must be ≥ 1"))

    M, A, dof = create_matrix_crouzeix_raviart_3d(m; T = Interval{Float64})
    int_dofs = dof.interior_dofs
    n_int = length(int_dofs)
    n_int ≥ 1 || throw(ArgumentError("Dirichlet CR space has no interior facet DOFs"))

    nev = min(Int(neig), n_int)
    A_red = Matrix(A[int_dofs, int_dofs])
    M_red = Matrix(M[int_dofs, int_dofs])
    eig_int, _ = if nev == n_int
        Veigs.veig(Veigs.sym_hull(A_red), Veigs.sym_hull(M_red))
    else
        Veigs.veigs(A_red, M_red, nev, :sm)
    end

    perm = sortperm(eig_int; by = inf)
    return CrLaplaceEig3D(eig_int[perm], M, A, dof)
end

"""
    special_tetrahedron_mesh() -> Mesh3D

Return a centroid split of the fundamental tetrahedron `T_F` described in
`docs/notes/special_tetrahedron.md`:

    (0,0,0), (0,0,1), (1/2,1/2,1/2), (-1/2,1/2,1/2).

A single tetrahedron has no interior CR facet DOFs under Dirichlet
boundary conditions, so this helper splits `T_F` into four tetrahedra by
adding the exact centroid `(0, 1/4, 1/2)`.
"""
function special_tetrahedron_mesh()
    nodes = [ 0.0   0.0   0.0;
              0.0   0.0   1.0;
              0.5   0.5   0.5;
             -0.5   0.5   0.5;
              0.0   0.25  0.5 ]
    elements = [2 3 4 5;
                1 3 4 5;
                1 2 4 5;
                1 2 3 5]
    facets = get_facet_list(elements)
    edges = get_edge_list(elements)
    f2e, e2f = facet_element_connectivity(elements, facets)
    return Mesh3D(nodes, elements, facets, edges, f2e, e2f,
                  size(nodes, 1), size(elements, 1), size(facets, 1),
                  size(edges, 1))
end

function _complete_mesh3d(nodes::Matrix{Float64}, elements::Matrix{Int})
    elements = sort(elements, dims = 2)
    facets = get_facet_list(elements)
    edges = get_edge_list(elements)
    f2e, e2f = facet_element_connectivity(elements, facets)
    return Mesh3D(nodes, elements, facets, edges, f2e, e2f,
                  size(nodes, 1), size(elements, 1), size(facets, 1),
                  size(edges, 1))
end

"""
    red_refine_mesh_3d(m::Mesh3D) -> Mesh3D

Uniform red refinement for tetrahedral meshes. Each tetrahedron is split
into eight subtetrahedra by adding midpoints on its six edges. The
connectivity matches the MATLAB `refine_tet_mesh` / `fast_refine_tet_mesh`
helpers used by the VFEM3D Lehmann-Goerisch scripts.
"""
function red_refine_mesh_3d(m::Mesh3D)
    edge_pairs = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    edge_set = Set{Tuple{Int, Int}}()
    for e in 1:m.NumElt
        v = m.ElementList[e, :]
        for (a, b) in edge_pairs
            push!(edge_set, minmax(Int(v[a]), Int(v[b])))
        end
    end

    edges = sort!(collect(edge_set))
    edge_to_mid = Dict(edge => m.NumNode + i for (i, edge) in enumerate(edges))
    nodes = Matrix{Float64}(undef, m.NumNode + length(edges), 3)
    nodes[1:m.NumNode, :] .= m.NodeList
    for (i, edge) in enumerate(edges)
        nodes[m.NumNode + i, :] .= 0.5 .* (m.NodeList[edge[1], :] .+ m.NodeList[edge[2], :])
    end

    elements = Matrix{Int}(undef, 8 * m.NumElt, 4)
    row = 0
    for e in 1:m.NumElt
        v1, v2, v3, v4 = Int.(m.ElementList[e, :])
        m12 = edge_to_mid[minmax(v1, v2)]
        m13 = edge_to_mid[minmax(v1, v3)]
        m14 = edge_to_mid[minmax(v1, v4)]
        m23 = edge_to_mid[minmax(v2, v3)]
        m24 = edge_to_mid[minmax(v2, v4)]
        m34 = edge_to_mid[minmax(v3, v4)]
        children = ((v1, m12, m13, m14),
                    (v2, m12, m23, m24),
                    (v3, m13, m23, m34),
                    (v4, m14, m24, m34),
                    (m12, m13, m14, m24),
                    (m12, m13, m23, m24),
                    (m13, m14, m24, m34),
                    (m13, m23, m24, m34))
        for child in children
            row += 1
            elements[row, :] .= sort(collect(child))
        end
    end
    return _complete_mesh3d(nodes, elements)
end

"""
    special_tetrahedron_red_mesh(level::Integer) -> Mesh3D

Build the fundamental tetrahedron from `docs/notes/special_tetrahedron.md`, then
apply `level` uniform red refinements. This is the Julia counterpart of
MATLAB `special_tet_make_mesh(level)`.
"""
function special_tetrahedron_red_mesh(level::Integer)
    level ≥ 0 || throw(DomainError(level, "level must be ≥ 0"))
    nodes = [ 0.0   0.0   0.0;
              0.0   0.0   1.0;
              0.5   0.5   0.5;
             -0.5   0.5   0.5 ]
    m = _complete_mesh3d(nodes, reshape([1, 2, 3, 4], 1, 4))
    for _ in 1:Int(level)
        m = red_refine_mesh_3d(m)
    end
    return m
end
