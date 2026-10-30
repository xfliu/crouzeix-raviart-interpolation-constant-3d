# src/applications/cecr_pipeline/m3_p1_ub.jl
#
# Step m3: P1 Lagrange upper bound (conforming Galerkin with Dirichlet BC).
# Port of Code_Sorted/modules/m3_p1_ub/m3_p1_upper_bound.m.
#
# Uses _shift_invert_eigs (defined in pipeline_types.jl) with shift=-3.0
# so the k smallest P1 eigenvalues are found efficiently.
# MATLAB uses eigs(A, M, neig, -3.0) which is also shift-invert at -3.0.

using SparseArrays: spzeros, sparse, dropzeros!
using LinearAlgebra: det, dot

# ----- 2D helpers -----------------------------------------------------------

function _interior_vertex_dofs_2d(m::Mesh2D)
    bd = Set{Int}()
    for i in 1:m.nb
        push!(bd, m.bd_edges[i, 1])
        push!(bd, m.bd_edges[i, 2])
    end
    return [v for v in 1:m.nv if v ∉ bd]
end

"""
    p1_upper_bound(m::Mesh2D, cfg::CaseConfig) -> Vector{Float64}

P1 Dirichlet upper bound for the first `cfg.neig` eigenvalues.
Assembles A = K_h + D_h where K_h is the pure kinetic stiffness and D_h
is the exact Coulomb potential matrix from `_assemble_Dh_2d` (exact 1/r
singular moments via tri_general_sing_moment_le4_exact, no vertex
sampling).  Dirichlet BC enforced by restricting to interior vertices.
"""
function p1_upper_bound(m::Mesh2D, cfg::CaseConfig)
    # Pure kinetic stiffness + mass (zero potential Bernstein coeffs → A = K_h)
    K_h_full, M_full = create_matrix_lagrange(m, 1, zeros(Float64, m.nt, 15))
    # Exact Coulomb potential matrix: D_h[i,j] = Σ_c Z_c ∫(1/|x-a_c|)φ_iφ_j dx
    # Potential V = -Z/r (attractive), so A = K_h + ∫Vφ_iφ_j = K_h - D_h
    D_full = _assemble_Dh_2d(m, cfg.centers, cfg.charges)
    A_full = K_h_full - D_full
    int_dofs = _interior_vertex_dofs_2d(m)
    A_int = A_full[int_dofs, int_dofs]
    M_int = M_full[int_dofs, int_dofs]
    k_eff = min(cfg.neig, size(A_int, 1) - 1)
    return _shift_invert_eigs(A_int, M_int, k_eff, -3.0)
end

# ----- 3D helpers -----------------------------------------------------------

# Interior node IDs: nodes NOT appearing in any boundary facet.
function _interior_node_dofs_3d(m::Mesh3D)
    bd = Set{Int}()
    @inbounds for f in 1:m.NumF
        m.Facet2Element[f, 2] == 0 || continue   # boundary iff elem2 == 0
        push!(bd, m.FacetList[f, 1])
        push!(bd, m.FacetList[f, 2])
        push!(bd, m.FacetList[f, 3])
    end
    return [v for v in 1:m.NumNode if v ∉ bd]
end

# Standard P1 Lagrange on tetrahedra with Dirichlet BC.
# Uses c_h as piecewise-constant potential (cell averages).
function _p1_ub_3d_assembly(m::Mesh3D, c_h::AbstractVector{Float64}, neig::Int)
    int_nodes = _interior_node_dofs_3d(m)
    node_to_int = zeros(Int, m.NumNode)
    for (i, v) in enumerate(int_nodes)
        node_to_int[v] = i
    end
    n_int = length(int_nodes)

    # COO triplets + one `sparse` call rather than scatter-adds into a CSC
    # (`X[i,j] +=` inserts on a new entry: an O(nnz) memmove, so element-by-
    # element assembly is quadratic). `dropzeros!` reproduces the original
    # sparsity, since `+= 0.0` on a non-stored entry is a no-op in Julia.
    cap = m.NumElt * 16
    Irow = Int[]; Jcol = Int[]; Aval = Float64[]; Mval = Float64[]
    sizehint!(Irow, cap); sizehint!(Jcol, cap)
    sizehint!(Aval, cap); sizehint!(Mval, cap)

    @inbounds for k in 1:m.NumElt
        v1 = m.ElementList[k, 1]; v2 = m.ElementList[k, 2]
        v3 = m.ElementList[k, 3]; v4 = m.ElementList[k, 4]
        p1 = @view m.NodeList[v1, :]; p2 = @view m.NodeList[v2, :]
        p3 = @view m.NodeList[v3, :]; p4 = @view m.NodeList[v4, :]

        J11 = p2[1]-p1[1]; J21 = p2[2]-p1[2]; J31 = p2[3]-p1[3]
        J12 = p3[1]-p1[1]; J22 = p3[2]-p1[2]; J32 = p3[3]-p1[3]
        J13 = p4[1]-p1[1]; J23 = p4[2]-p1[2]; J33 = p4[3]-p1[3]

        detJ = J11*(J22*J33 - J23*J32) - J12*(J21*J33 - J23*J31) + J13*(J21*J32 - J22*J31)
        vol = abs(detJ) / 6.0

        idet = 1.0 / detJ
        g2x = idet * (J22*J33 - J23*J32); g2y = idet * (J13*J32 - J12*J33); g2z = idet * (J12*J23 - J13*J22)
        g3x = idet * (J23*J31 - J21*J33); g3y = idet * (J11*J33 - J13*J31); g3z = idet * (J13*J21 - J11*J23)
        g4x = idet * (J21*J32 - J22*J31); g4y = idet * (J12*J31 - J11*J32); g4z = idet * (J11*J22 - J12*J21)
        g1x = -(g2x + g3x + g4x); g1y = -(g2y + g3y + g4y); g1z = -(g2z + g3z + g4z)

        gx = (g1x, g2x, g3x, g4x)
        gy = (g1y, g2y, g3y, g4y)
        gz = (g1z, g2z, g3z, g4z)

        vids = (v1, v2, v3, v4)
        ck = c_h[k]

        for i in 1:4, j in 1:4
            ii = node_to_int[vids[i]]
            jj = node_to_int[vids[j]]
            (ii == 0 || jj == 0) && continue

            k_val = vol * (gx[i]*gx[j] + gy[i]*gy[j] + gz[i]*gz[j])
            m_val = vol * (i == j ? 1.0/10.0 : 1.0/20.0)
            push!(Irow, ii); push!(Jcol, jj)
            push!(Aval, k_val + ck * m_val); push!(Mval, m_val)
        end
    end

    A = dropzeros!(sparse(Irow, Jcol, Aval, n_int, n_int))
    M = dropzeros!(sparse(Irow, Jcol, Mval, n_int, n_int))

    k_eff = min(neig, n_int - 1)
    return _shift_invert_eigs(A, M, k_eff, -3.0)
end

"""
    p1_upper_bound(m::Mesh3D, c_h, cfg::CaseConfig) -> Vector{Float64}

P1 Dirichlet upper bound for the 3D problem.
"""
function p1_upper_bound(m::Mesh3D, c_h::AbstractVector{Float64}, cfg::CaseConfig)
    return _p1_ub_3d_assembly(m, c_h, cfg.neig)
end
