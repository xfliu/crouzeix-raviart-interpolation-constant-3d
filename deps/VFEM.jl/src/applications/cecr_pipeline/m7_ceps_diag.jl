# src/applications/cecr_pipeline/m7_ceps_diag.jl
#
# Step m7: C_ε diagnostic via the regularised P1 eigenproblem.
# Port of Code_Sorted/modules/m7_ceps_diag/m7_ceps_diagnostic.m.
#
# Solves  (D_h − ε K_h) x = λ (M_h + τ K_h) x  for the largest λ.
# λ_max^{P1} ≤ C_ε^{cert} (certified form-bound constant).
#
# D_h[i,j] = Σ_c Z_c * ∫_Ω φ_i φ_j / |x − a_c| dx
#
# For 2D: uses tri_general_sing_moment_le4_exact for exact 1/r moments
# (all elements, handles singularity at vertices correctly).
#
# For 3D: uses singular_potential_matrix_vertex_exact.
#
# Regularization: B_reg = M_h + τ K_h (τ = 1e-3) prevents ill-conditioning
# of M_h on graded meshes with h_min ~ 1e-6. λ_reg ≤ λ_true because
# B_reg ≥ M_h in the PSD sense.

using Arpack: eigs
using SparseArrays: spzeros, sparse, dropzeros!

const _M7_TAU = 1.0e-3   # regularization coefficient for B_reg = M_h + τ K_h

# ---- 2D D_h assembly via exact singular moments ---------------------------

# Barycentric coordinate coefficients for P1 on a translated triangle.
# λ_i(u, v) = a[i] + b[i]*u + c[i]*v  (i = 1, 2, 3)
# (u, v) are coordinates translated so the singularity is at the origin.
function _p1_bary_coeffs_2d(u1, v1, u2, v2, u3, v3)
    area2 = (u2 - u1) * (v3 - v1) - (u3 - u1) * (v2 - v1)  # signed 2*area
    inv_a2 = 1.0 / area2

    b1 = (v2 - v3) * inv_a2;   c1 = (u3 - u2) * inv_a2
    a1 = (u2 * v3 - u3 * v2) * inv_a2

    b2 = (v3 - v1) * inv_a2;   c2 = (u1 - u3) * inv_a2
    a2 = (u3 * v1 - u1 * v3) * inv_a2

    b3 = (v1 - v2) * inv_a2;   c3 = (u2 - u1) * inv_a2
    a3 = (u1 * v2 - u2 * v1) * inv_a2

    return (a1, b1, c1), (a2, b2, c2), (a3, b3, c3)
end

# Exact 3×3 local D_h matrix for one element and one nucleus at origin.
# Vertices (u1,v1t), (u2,v2t), (u3,v3t) are already translated so nucleus=origin.
# Uses tri_general_sing_moment_le4_exact for exact 1/r×polynomial integration.
function _Dh_local_2d_exact(u1, v1t, u2, v2t, u3, v3t)
    (a1,b1,c1), (a2,b2,c2), (a3,b3,c3) = _p1_bary_coeffs_2d(u1,v1t,u2,v2t,u3,v3t)
    av = (a1, a2, a3)
    bv = (b1, b2, b3)
    cv = (c1, c2, c3)
    I00 = tri_general_sing_moment_le4_exact(u1,v1t,u2,v2t,u3,v3t, 0, 0)
    I10 = tri_general_sing_moment_le4_exact(u1,v1t,u2,v2t,u3,v3t, 1, 0)
    I01 = tri_general_sing_moment_le4_exact(u1,v1t,u2,v2t,u3,v3t, 0, 1)
    I20 = tri_general_sing_moment_le4_exact(u1,v1t,u2,v2t,u3,v3t, 2, 0)
    I11 = tri_general_sing_moment_le4_exact(u1,v1t,u2,v2t,u3,v3t, 1, 1)
    I02 = tri_general_sing_moment_le4_exact(u1,v1t,u2,v2t,u3,v3t, 0, 2)
    D_loc = zeros(Float64, 3, 3)
    for i in 1:3, j in 1:3
        ai,bi,ci = av[i], bv[i], cv[i]
        aj,bj,cj = av[j], bv[j], cv[j]
        D_loc[i,j] = ai*aj*I00 + (ai*bj+aj*bi)*I10 + (ai*cj+aj*ci)*I01 +
                     bi*bj*I20 + (bi*cj+ci*bj)*I11 + ci*cj*I02
    end
    # For CW elements (area2 < 0), both the barycentric coefficients and the
    # singular moments are negated relative to CCW, so D_loc is negative. Multiply
    # by sign(area2) to restore the correct positive sign (mirrors MATLAB's
    # sgn = sign(D) convention in local_sing_matrix_2d).
    area2 = (u2-u1)*(v3t-v1t) - (u3-u1)*(v2t-v1t)
    sgn = area2 > 0 ? 1.0 : -1.0
    return D_loc .* sgn
end

# Per-node mesh diameter: h_node[v] = max edge length of all adjacent elements.
# Used by MATLAB m7 to filter fine-mesh nodes near the singularity.
function _per_node_h_2d(m::Mesh2D)
    h_node = zeros(Float64, m.nv)
    @inbounds for k in 1:m.nt
        v1, v2, v3 = m.elements[k, 1], m.elements[k, 2], m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        e12 = sqrt((x2-x1)^2 + (y2-y1)^2)
        e13 = sqrt((x3-x1)^2 + (y3-y1)^2)
        e23 = sqrt((x3-x2)^2 + (y3-y2)^2)
        hK  = max(e12, e13, e23)
        h_node[v1] = max(h_node[v1], hK)
        h_node[v2] = max(h_node[v2], hK)
        h_node[v3] = max(h_node[v3], hK)
    end
    return h_node
end

function _assemble_Dh_2d(m::Mesh2D, centers::Matrix{Float64},
                          charges::Vector{Float64})
    nv = m.nv
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC
    # (`X[i,j] +=` inserts on a new entry: an O(nnz) memmove, so element-by-
    # element assembly is quadratic). `dropzeros!` reproduces the original
    # sparsity, since `+= 0.0` on a non-stored entry is a no-op in Julia.
    Irow = Int[]; Jcol = Int[]; Dval = Float64[]
    cap = m.nt * size(centers, 1) * 9
    sizehint!(Irow, cap); sizehint!(Jcol, cap); sizehint!(Dval, cap)

    @inbounds for k in 1:m.nt
        v1, v2, v3 = m.elements[k, 1], m.elements[k, 2], m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        vids = (v1, v2, v3)

        for ci in 1:size(centers, 1)
            ax, ay = centers[ci, 1], centers[ci, 2]
            Z = charges[ci]
            u1, w1 = x1 - ax, y1 - ay
            u2, w2 = x2 - ax, y2 - ay
            u3, w3 = x3 - ax, y3 - ay
            D_loc = _Dh_local_2d_exact(u1, w1, u2, w2, u3, w3)
            for i in 1:3, j in 1:3
                push!(Irow, vids[i]); push!(Jcol, vids[j])
                push!(Dval, Z * D_loc[i, j])
            end
        end
    end
    return dropzeros!(sparse(Irow, Jcol, Dval, nv, nv))
end

"""
    ceps_diagnostic_2d(m, centers, charges, epsilon) -> Float64

Certify C_ε via the 2D P1 eigenproblem
    (D_h − ε K_h) x = λ (M_h + τ K_h) x,
and return λ_max as a certified upper bound C_ε^{cert}.

`D_h` is assembled using exact 1/r moments from `tri_general_sing_moment_le4_exact`.
Dirichlet BC applied (interior nodes only). τ = 1e-3 regularises M_h.
"""
function ceps_diagnostic_2d(m::Mesh2D, centers::Matrix{Float64},
                             charges::Vector{Float64}, epsilon::Float64)
    # P1 matrices (zero potential → A = pure stiffness K_h)
    K_full, M_full = create_matrix_lagrange(m, 1, zeros(Float64, m.nt, 15))

    # Dirichlet: restrict to interior vertices
    bd = Set{Int}()
    for i in 1:m.nb; push!(bd, m.bd_edges[i,1]); push!(bd, m.bd_edges[i,2]); end
    int_dofs_all = [v for v in 1:m.nv if v ∉ bd]

    # MATLAB node filter (m7_ceps_diagnostic.m lines 108-131):
    # exclude interior nodes whose per-node diameter h_node < 1e-3 × h_max.
    # This removes the innermost rings of fine elements near the singularity.
    h_node     = _per_node_h_2d(m)
    h_max_mesh = maximum(h_node)
    threshold  = 1e-3 * h_max_mesh
    int_dofs   = [v for v in int_dofs_all if h_node[v] >= threshold]
    @info "  [m7-2d] n_int=$(length(int_dofs_all))  n_reg=$(length(int_dofs))  threshold=$(round(threshold, sigdigits=3))"

    K_h = K_full[int_dofs, int_dofs]
    M_h = M_full[int_dofs, int_dofs]

    # Assemble D_h and restrict
    D_full = _assemble_Dh_2d(m, centers, charges)
    D_h    = D_full[int_dofs, int_dofs]

    # Regularised eigenproblem
    A_m7  = D_h - epsilon * K_h
    B_reg = M_h + _M7_TAU * K_h

    n = size(A_m7, 1)
    p_sub = min(60, max(20, n))
    v0 = ones(Float64, n) ./ sqrt(Float64(n))
    λ_arr, _ = eigs(A_m7, B_reg; nev = 1, which = :LR,
                    tol = 1e-8, maxiter = 5000,
                    v0 = v0, ncv = p_sub)
    return real(λ_arr[1])
end

# Per-node mesh diameter for 3D: h_node[v] = max edge length of all adjacent tets.
function _per_node_h_3d(m::Mesh3D)
    h_node = zeros(Float64, m.NumNode)
    @inbounds for k in 1:m.NumElt
        v1 = m.ElementList[k,1]; v2 = m.ElementList[k,2]
        v3 = m.ElementList[k,3]; v4 = m.ElementList[k,4]
        x1,y1,z1 = m.NodeList[v1,1],m.NodeList[v1,2],m.NodeList[v1,3]
        x2,y2,z2 = m.NodeList[v2,1],m.NodeList[v2,2],m.NodeList[v2,3]
        x3,y3,z3 = m.NodeList[v3,1],m.NodeList[v3,2],m.NodeList[v3,3]
        x4,y4,z4 = m.NodeList[v4,1],m.NodeList[v4,2],m.NodeList[v4,3]
        e12 = sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)
        e13 = sqrt((x3-x1)^2+(y3-y1)^2+(z3-z1)^2)
        e14 = sqrt((x4-x1)^2+(y4-y1)^2+(z4-z1)^2)
        e23 = sqrt((x3-x2)^2+(y3-y2)^2+(z3-z2)^2)
        e24 = sqrt((x4-x2)^2+(y4-y2)^2+(z4-z2)^2)
        e34 = sqrt((x4-x3)^2+(y4-y3)^2+(z4-z3)^2)
        hK  = max(e12, e13, e14, e23, e24, e34)
        h_node[v1] = max(h_node[v1], hK)
        h_node[v2] = max(h_node[v2], hK)
        h_node[v3] = max(h_node[v3], hK)
        h_node[v4] = max(h_node[v4], hK)
    end
    return h_node
end

# ---- 3D D_h assembly using vertex singular moments -----------------------

# For 3D: D_h[i,j] = Σ_c Z_c * ∫_Ω ψ_i ψ_j / |x-a_c| dx  (P1 basis ψ_i)
# Uses singular_potential_matrix_vertex_exact for elements with a
# nucleus at one vertex, and a 4-point Gauss-Legendre rule elsewhere.
function _assemble_Dh_3d(m::Mesh3D, centers::Matrix{Float64},
                          charges::Vector{Float64})
    nv = m.NumNode
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC
    # (`X[i,j] +=` inserts on a new entry: an O(nnz) memmove, so element-by-
    # element assembly is quadratic). `dropzeros!` reproduces the original
    # sparsity, since `+= 0.0` on a non-stored entry is a no-op in Julia.
    Irow = Int[]; Jcol = Int[]; Dval = Float64[]
    sizehint!(Irow, m.NumElt * 16); sizehint!(Jcol, m.NumElt * 16)
    sizehint!(Dval, m.NumElt * 16)

    # 4-point Gauss-Legendre on [0,1].
    xi_gl, wi_gl = _gauss_legendre_01(4)

    @inbounds for k in 1:m.NumElt
        v1 = m.ElementList[k, 1]; v2 = m.ElementList[k, 2]
        v3 = m.ElementList[k, 3]; v4 = m.ElementList[k, 4]
        vids = (v1, v2, v3, v4)
        # 4×3 local nodes matrix for singular_potential_matrix_vertex_exact
        LocalNodes = Float64[m.NodeList[v1,1] m.NodeList[v1,2] m.NodeList[v1,3];
                             m.NodeList[v2,1] m.NodeList[v2,2] m.NodeList[v2,3];
                             m.NodeList[v3,1] m.NodeList[v3,2] m.NodeList[v3,3];
                             m.NodeList[v4,1] m.NodeList[v4,2] m.NodeList[v4,3]]

        d11=LocalNodes[2,1]-LocalNodes[1,1]; d21=LocalNodes[2,2]-LocalNodes[1,2]; d31=LocalNodes[2,3]-LocalNodes[1,3]
        d12=LocalNodes[3,1]-LocalNodes[1,1]; d22=LocalNodes[3,2]-LocalNodes[1,2]; d32=LocalNodes[3,3]-LocalNodes[1,3]
        d13=LocalNodes[4,1]-LocalNodes[1,1]; d23=LocalNodes[4,2]-LocalNodes[1,2]; d33=LocalNodes[4,3]-LocalNodes[1,3]
        detJ = d11*(d22*d33-d23*d32) - d12*(d21*d33-d23*d31) + d13*(d21*d32-d22*d31)
        vol = abs(detJ) / 6.0

        for ci in 1:size(centers, 1)
            Z = charges[ci]
            ac = @view centers[ci, :]

            # Find closest vertex to nucleus
            min_d = Inf; sv = 1
            for v in 1:4
                dx = LocalNodes[v,1] - ac[1]
                dy = LocalNodes[v,2] - ac[2]
                dz = LocalNodes[v,3] - ac[3]
                d = sqrt(dx*dx + dy*dy + dz*dz)
                if d < min_d; min_d = d; sv = v; end
            end

            if min_d < 1e-12
                # Nucleus at vertex sv: exact singular moments (N=1 = P1 Bernstein)
                local_D, _ = singular_potential_matrix_vertex_exact(LocalNodes, sv, 1)
                for i in 1:4, j in 1:4
                    push!(Irow, vids[i]); push!(Jcol, vids[j])
                    push!(Dval, Z * local_D[i, j])
                end
            else
                # Far element: conical-product Gauss-Legendre, 4 pts.
                _add_Dh_3d_gauss!(Irow, Jcol, Dval, vids, m, ac, Z, vol, xi_gl, wi_gl)
            end
        end
    end
    return dropzeros!(sparse(Irow, Jcol, Dval, nv, nv))
end

# Conical-product GL quadrature contribution to D_h on a far element.
function _add_Dh_3d_gauss!(Irow, Jcol, Dval, vids, m::Mesh3D, ac, Z::Float64,
                             vol::Float64,
                             xi::Vector{Float64}, wi::Vector{Float64})
    n = length(xi)
    @inbounds for i1 in 1:n, i2 in 1:n, i3 in 1:n
        r_q = xi[i1]; s_q = xi[i2]; t_q = xi[i3]
        lam4 = r_q
        lam3 = s_q * (1 - r_q)
        lam2 = t_q * (1 - r_q) * (1 - s_q)
        lam1 = 1 - lam2 - lam3 - lam4
        jac = (1 - r_q)^2 * (1 - s_q)
        lams = (lam1, lam2, lam3, lam4)

        x = m.NodeList[vids[1], 1]*lam1 + m.NodeList[vids[2], 1]*lam2 +
            m.NodeList[vids[3], 1]*lam3 + m.NodeList[vids[4], 1]*lam4
        y = m.NodeList[vids[1], 2]*lam1 + m.NodeList[vids[2], 2]*lam2 +
            m.NodeList[vids[3], 2]*lam3 + m.NodeList[vids[4], 2]*lam4
        z = m.NodeList[vids[1], 3]*lam1 + m.NodeList[vids[2], 3]*lam2 +
            m.NodeList[vids[3], 3]*lam3 + m.NodeList[vids[4], 3]*lam4
        dx = x - ac[1]; dy = y - ac[2]; dz = z - ac[3]
        r = sqrt(dx*dx + dy*dy + dz*dz)
        w = wi[i1] * wi[i2] * wi[i3] * jac * 6 * vol

        for i in 1:4, j in 1:4
            push!(Irow, vids[i]); push!(Jcol, vids[j])
            push!(Dval, Z * lams[i] * lams[j] / max(r, 1e-15) * w)
        end
    end
end

"""
    ceps_diagnostic_3d(m, centers, charges, epsilon) -> Float64

Certify C_ε via the 3D P1 eigenproblem (same structure as 2D).
"""
function ceps_diagnostic_3d(m::Mesh3D, centers::Matrix{Float64},
                             charges::Vector{Float64}, epsilon::Float64)
    int_nodes_all = _interior_node_dofs_3d(m)

    # MATLAB node filter (m7_ceps_diagnostic.m 3D section):
    # exclude interior nodes with per-node diameter h_node < 1e-3 × h_max.
    # Prevents ill-conditioning from ultra-fine origin-patch nodes (h_min ~ 1e-4).
    h_node     = _per_node_h_3d(m)
    h_max_mesh = maximum(h_node)
    threshold  = 1e-3 * h_max_mesh
    int_nodes  = [v for v in int_nodes_all if h_node[v] >= threshold]
    @info "  [m7-3d] n_int=$(length(int_nodes_all))  n_reg=$(length(int_nodes))  threshold=$(round(threshold, sigdigits=3))"

    node_map = zeros(Int, m.NumNode)
    for (i, v) in enumerate(int_nodes); node_map[v] = i; end
    n_int = length(int_nodes)

    # P1 stiffness and mass (no potential).
    # COO triplets + one `sparse` call rather than scatter-adds into a CSC
    # (`X[i,j] +=` inserts on a new entry: an O(nnz) memmove, so element-by-
    # element assembly is quadratic). `dropzeros!` reproduces the original
    # sparsity, since `+= 0.0` on a non-stored entry is a no-op in Julia.
    Ki = Int[]; Kj = Int[]; Kv = Float64[]; Mv = Float64[]
    cap = m.NumElt * 16
    sizehint!(Ki, cap); sizehint!(Kj, cap); sizehint!(Kv, cap); sizehint!(Mv, cap)

    @inbounds for k in 1:m.NumElt
        v1 = m.ElementList[k,1]; v2 = m.ElementList[k,2]
        v3 = m.ElementList[k,3]; v4 = m.ElementList[k,4]
        p1r = m.NodeList[v1,:]; p2r = m.NodeList[v2,:]
        p3r = m.NodeList[v3,:]; p4r = m.NodeList[v4,:]
        J11 = p2r[1]-p1r[1]; J21 = p2r[2]-p1r[2]; J31 = p2r[3]-p1r[3]
        J12 = p3r[1]-p1r[1]; J22 = p3r[2]-p1r[2]; J32 = p3r[3]-p1r[3]
        J13 = p4r[1]-p1r[1]; J23 = p4r[2]-p1r[2]; J33 = p4r[3]-p1r[3]
        detJ = J11*(J22*J33-J23*J32) - J12*(J21*J33-J23*J31) + J13*(J21*J32-J22*J31)
        vol = abs(detJ) / 6.0
        idet = 1.0 / detJ
        g2x=idet*(J22*J33-J23*J32); g2y=idet*(J13*J32-J12*J33); g2z=idet*(J12*J23-J13*J22)
        g3x=idet*(J23*J31-J21*J33); g3y=idet*(J11*J33-J13*J31); g3z=idet*(J13*J21-J11*J23)
        g4x=idet*(J21*J32-J22*J31); g4y=idet*(J12*J31-J11*J32); g4z=idet*(J11*J22-J12*J21)
        g1x=-(g2x+g3x+g4x); g1y=-(g2y+g3y+g4y); g1z=-(g2z+g3z+g4z)
        gx=(g1x,g2x,g3x,g4x); gy=(g1y,g2y,g3y,g4y); gz=(g1z,g2z,g3z,g4z)
        vids=(v1,v2,v3,v4)
        for i in 1:4, j in 1:4
            ii=node_map[vids[i]]; jj=node_map[vids[j]]
            (ii==0||jj==0) && continue
            push!(Ki, ii); push!(Kj, jj)
            push!(Kv, vol*(gx[i]*gx[j]+gy[i]*gy[j]+gz[i]*gz[j]))
            push!(Mv, vol*(i==j ? 1.0/10.0 : 1.0/20.0))
        end
    end

    K_h = dropzeros!(sparse(Ki, Kj, Kv, n_int, n_int))
    M_h = dropzeros!(sparse(Ki, Kj, Mv, n_int, n_int))

    # D_h (restrict to interior nodes)
    D_full = _assemble_Dh_3d(m, centers, charges)
    D_h    = D_full[int_nodes, int_nodes]

    A_m7  = D_h - epsilon * K_h
    B_reg = M_h + _M7_TAU * K_h

    n = size(A_m7, 1)
    p_sub = min(60, max(20, n))
    v0 = ones(Float64, n) ./ sqrt(Float64(n))
    λ_arr, _ = eigs(A_m7, B_reg; nev = 1, which = :LR,
                    tol = 1e-8, maxiter = 5000,
                    v0 = v0, ncv = p_sub)
    return real(λ_arr[1])
end

"""
    ceps_diagnostic(m::Mesh2D, cfg::CaseConfig) -> Float64
    ceps_diagnostic(m::Mesh3D, cfg::CaseConfig) -> Float64

Dispatch to the appropriate 2D or 3D C_ε diagnostic.
"""
ceps_diagnostic(m::Mesh2D, cfg::CaseConfig) =
    ceps_diagnostic_2d(m, cfg.centers, cfg.charges, cfg.epsilon)

ceps_diagnostic(m::Mesh3D, cfg::CaseConfig) =
    ceps_diagnostic_3d(m, cfg.centers, cfg.charges, cfg.epsilon)
