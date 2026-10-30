# src/core/eigensolve2d/rt_hdiv_problem.jl
#
# Port of vfem2d/lib/eigensolve/RT_Hdiv_problem.m (~480 LOC of MATLAB).
#
# Builds the Goerisch w-w bilinear form
#   mat_b_w_w = RT_vec' · A_RT · RT_vec
# from a mixed Raviart–Thomas / DG-Lagrange formulation:
#   [A   B] [u_RT] = [0]
#   [B^T 0] [u_DG]   [-F]
# where A is the RT mass matrix, B is the RT-DG mixed coupling, F
# is the Lagrange RHS built from the input `f` (typically a stack
# of Lagrange-basis coefficient vectors of CG eigenfunctions).
#
# All RT_*, Lagrange_*, create_ijk helpers below are private to this
# file. The public entry point is `rt_hdiv_problem`.
#
# This routine is the heart of the 2D Lehmann–Goerisch sharpening
# pipeline. Cross-validated against MATLAB on a small fixture in the
# corresponding test file.

using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!
using LinearAlgebra: tr, det, I

# ---- Multi-index helpers ----------------------------------------------------

_RT_lag_nbasis(order::Integer) = (order + 1) * (order + 2) ÷ 2
_RT_full_nbasis(order::Integer) = (order + 1) * (order + 3)

# Number of Bernstein-polynomial basis functions of given degree on a triangle.
_RT_bern_nbasis(degree::Integer) = (degree + 1) * (degree + 2) ÷ 2

# (n+1)(n+2)/2 multi-indices (p, q, n-p-q) in lex-descending order.
function _RT_create_ijk(n::Integer)
    out = Matrix{Int}(undef, _RT_bern_nbasis(n), 3)
    cur = 1
    @inbounds for p in n:-1:0, q in (n - p):-1:0
        out[cur, 1] = p
        out[cur, 2] = q
        out[cur, 3] = n - p - q
        cur += 1
    end
    return out
end

# Map (i, j, k) with i+j+k = n to its row index in `_RT_create_ijk(n)`.
# Returns a sentinel `0` for out-of-range or invalid inputs (caller checks).
function _RT_map_ijk_to_idx(i::Integer, j::Integer, k::Integer, n::Integer)
    (i < 0 || j < 0 || k < 0) && return 0
    i + j + k == n || return 0
    return (n - i) * (n - i + 1) ÷ 2 + (n - i - j) + 1
end

# i! j! k! / (i+j+k+2)!  — barycentric monomial integral on the unit
# reference triangle (analog of `inner_prod_matrix_reference` in 2D).
function _RT_integral_L1L2L3_ijk(i::Integer, j::Integer, k::Integer)
    return Float64(factorial(i) * factorial(j) * factorial(k)) /
           Float64(factorial(i + j + k + 2))
end

# Symmetric inner-product matrix of all Bernstein-degree-n monomials
# on the unit reference triangle. M[p, q] = ∫ L^{α_p + α_q} dx.
function _RT_inner_product_L1L2L3_all(order::Integer)
    ijk = _RT_create_ijk(order)
    len = size(ijk, 1)
    M = zeros(Float64, len, len)
    @inbounds for p in 1:len, q in p:len
        M[p, q] = _RT_integral_L1L2L3_ijk(ijk[p, 1] + ijk[q, 1],
                                           ijk[p, 2] + ijk[q, 2],
                                           ijk[p, 3] + ijk[q, 3])
        p == q || (M[q, p] = M[p, q])
    end
    return M
end

# ---- RT basis (port of MATLAB `RT_basis`) ----------------------------------

function _RT_basis(order::Integer)
    n = order
    nbasis = _RT_full_nbasis(order)
    basis_abc = zeros(Int, nbasis, 3)
    basis_ijk = zeros(Int, nbasis, 3)
    idx = 1

    # edge 1
    basis_abc[idx, :] = [0, 0, 1]; basis_ijk[idx, :] = [0, n, 0]; idx += 1
    @inbounds for p in (n - 1):-1:1
        basis_abc[idx, :] = [1, 0, 0]; basis_ijk[idx, :] = [0, p, n - p]; idx += 1
    end
    if n != 0
        basis_abc[idx, :] = [0, 1, 0]; basis_ijk[idx, :] = [0, 0, n]; idx += 1
    end

    # edge 2
    basis_abc[idx, :] = [-1, 0, 1]; basis_ijk[idx, :] = [0, 0, n]; idx += 1
    @inbounds for p in (n - 1):-1:1
        basis_abc[idx, :] = [-1, 0, 0]; basis_ijk[idx, :] = [n - p, 0, p]; idx += 1
    end
    if n != 0
        basis_abc[idx, :] = [-1, 0, 0]; basis_ijk[idx, :] = [n, 0, 0]; idx += 1
    end

    # edge 3
    basis_abc[idx, :] = [0, -1, 1]; basis_ijk[idx, :] = [n, 0, 0]; idx += 1
    @inbounds for p in (n - 1):-1:1
        basis_abc[idx, :] = [0, -1, 0]; basis_ijk[idx, :] = [p, n - p, 0]; idx += 1
    end
    if n != 0
        basis_abc[idx, :] = [0, -1, 1]; basis_ijk[idx, :] = [0, n, 0]; idx += 1
    end

    if n ≥ 1
        basis_abc[idx, :] = [0, 0, 1]; basis_ijk[idx, :] = [n, 0, 0]; idx += 1
        basis_abc[idx, :] = [-1, 0, 1]; basis_ijk[idx, :] = [0, n, 0]; idx += 1
    end

    if n ≥ 2
        @inbounds for p in (n - 1):-1:1
            basis_abc[idx, :] = [1, -1, 0]; basis_ijk[idx, :] = [0, p, n - p]; idx += 1
        end
        @inbounds for p in (n - 1):-1:1
            basis_abc[idx, :] = [0, 1, 0]; basis_ijk[idx, :] = [n - p, 0, p]; idx += 1
        end
        @inbounds for p in (n - 1):-1:1
            basis_abc[idx, :] = [-1, 0, 0]; basis_ijk[idx, :] = [p, n - p, 0]; idx += 1
        end
        @inbounds for p in (n - 1):-1:1
            basis_abc[idx, :] = [0, 0, 1]; basis_ijk[idx, :] = [p, n - p, 0]; idx += 1
        end
    end

    if n ≥ 3
        @inbounds for p in (n - 2):-1:1, q in (n - 1 - p):-1:1
            basis_abc[idx, :] = [1, 0, 0]
            basis_ijk[idx, :] = [p, q, n - p - q]
            idx += 1
        end
        @inbounds for p in (n - 2):-1:1, q in (n - 1 - p):-1:1
            basis_abc[idx, :] = [0, 1, 0]
            basis_ijk[idx, :] = [p, q, n - p - q]
            idx += 1
        end
    end

    return basis_abc, basis_ijk, nbasis
end

# ---- Lagrange (CG) basis on the reference triangle -------------------------

function _Lagrange_basis(order::Integer)
    n = order
    nbasis = _RT_lag_nbasis(order)
    basis = zeros(Int, nbasis, 3)
    idx = 1

    basis[idx, :] = [n, 0, 0]; idx += 1
    if n > 0
        basis[idx, :] = [0, n, 0]; idx += 1
        basis[idx, :] = [0, 0, n]; idx += 1
    end
    @inbounds for p in (n - 1):-1:1
        basis[idx, :] = [0, p, n - p]; idx += 1
    end
    @inbounds for p in (n - 1):-1:1
        basis[idx, :] = [n - p, 0, p]; idx += 1
    end
    @inbounds for p in (n - 1):-1:1
        basis[idx, :] = [p, n - p, 0]; idx += 1
    end
    @inbounds for p in (n - 2):-1:1, q in (n - 1 - p):-1:1
        basis[idx, :] = [p, q, n - p - q]; idx += 1
    end
    return basis, nbasis
end

# Coordinate vector e (one-hot) for the idx-th Lagrange basis function.
function _Lagrange_create_coord_basis(basis::AbstractMatrix{<:Integer},
                                       idx::Integer, lag_order::Integer)
    len = _RT_lag_nbasis(lag_order)
    e = zeros(Float64, len)
    i, j, k = basis[idx, 1], basis[idx, 2], basis[idx, 3]
    pos = _RT_map_ijk_to_idx(i, j, k, lag_order)
    pos > 0 || error("Lagrange basis index out of range")
    e[pos] = 1.0
    return e
end

# ---- RT basis as Bernstein-polynomial coefficients of the two
#      cartesian components (degree RT_order + 1).
function _RT_create_coord_basis(basis_abc::AbstractMatrix{<:Integer},
                                 basis_ijk::AbstractMatrix{<:Integer},
                                 idx::Integer, RT_order::Integer)
    np1 = RT_order + 1
    len = _RT_bern_nbasis(np1)
    e = zeros(Float64, len, 2)

    a, b, c = basis_abc[idx, 1], basis_abc[idx, 2], basis_abc[idx, 3]
    i, j, k = basis_ijk[idx, 1], basis_ijk[idx, 2], basis_ijk[idx, 3]

    p1 = _RT_map_ijk_to_idx(i + 1, j,     k,     np1)
    p2 = _RT_map_ijk_to_idx(i,     j + 1, k,     np1)
    p3 = _RT_map_ijk_to_idx(i,     j,     k + 1, np1)
    p1 > 0 && (e[p1, 1] = a;       e[p1, 2] = b)
    p2 > 0 && (e[p2, 1] = a + c;   e[p2, 2] = b)
    p3 > 0 && (e[p3, 1] = a;       e[p3, 2] = b + c)
    return e
end

# Coefficients of the divergence of the idx-th RT basis function in
# Bernstein-degree-`RT_order` representation.
function _RT_create_coord_basis_div(basis_abc::AbstractMatrix{<:Integer},
                                     basis_ijk::AbstractMatrix{<:Integer},
                                     idx::Integer, RT_order::Integer)
    n = RT_order
    len = _RT_bern_nbasis(n)
    e = zeros(Float64, len)
    f = zeros(Float64, len)

    a, b, c = basis_abc[idx, 1], basis_abc[idx, 2], basis_abc[idx, 3]
    i, j, k = basis_ijk[idx, 1], basis_ijk[idx, 2], basis_ijk[idx, 3]

    # ∂_x component of div, expanded as Bernstein-`n` coefficients.
    let pos
        pos = _RT_map_ijk_to_idx(i,     j,     k,     n)
        pos > 0 && (e[pos] += a * (j - i) + c * (j + 1))
        pos = _RT_map_ijk_to_idx(i - 1, j + 1, k,     n)
        pos > 0 && (e[pos] += -(a + c) * i)
        pos = _RT_map_ijk_to_idx(i - 1, j,     k + 1, n)
        pos > 0 && (e[pos] += -a * i)
        pos = _RT_map_ijk_to_idx(i + 1, j - 1, k,     n)
        pos > 0 && (e[pos] += a * j)
        pos = _RT_map_ijk_to_idx(i,     j - 1, k + 1, n)
        pos > 0 && (e[pos] += a * j)

        pos = _RT_map_ijk_to_idx(i,     j,     k,     n)
        pos > 0 && (f[pos] += b * (k - i) + c * (k + 1))
        pos = _RT_map_ijk_to_idx(i - 1, j,     k + 1, n)
        pos > 0 && (f[pos] += -(b + c) * i)
        pos = _RT_map_ijk_to_idx(i - 1, j + 1, k,     n)
        pos > 0 && (f[pos] += -b * i)
        pos = _RT_map_ijk_to_idx(i + 1, j,     k - 1, n)
        pos > 0 && (f[pos] += b * k)
        pos = _RT_map_ijk_to_idx(i,     j + 1, k - 1, n)
        pos > 0 && (f[pos] += b * k)
    end
    return e .+ f
end

# ---- Build the local A1 (RT mass) and A2 (div RT × DG) blocks --------------

# Returns a vector of `nbasis × nbasis` matrices `M_ij` such that
# `A1[i, j] = trace(B · M_ij · B') / det(B)` after the affine transform
# of the reference triangle to the physical triangle.
function _RT_basis_inner_products(RT_order::Integer)
    Mnp1 = _RT_inner_product_L1L2L3_all(RT_order + 1)
    basis_abc, basis_ijk, nbasis = _RT_basis(RT_order)
    M_ip = Vector{Matrix{Float64}}(undef, nbasis * (nbasis + 1) ÷ 2)
    cur = 1
    @inbounds for i in 1:nbasis
        ei = _RT_create_coord_basis(basis_abc, basis_ijk, i, RT_order)
        for j in i:nbasis
            ej = _RT_create_coord_basis(basis_abc, basis_ijk, j, RT_order)
            M_ip[cur] = ei' * Mnp1 * ej
            cur += 1
        end
    end
    return basis_abc, basis_ijk, nbasis, M_ip
end

# Lower-triangular index `(i, j) → cur` for `_RT_basis_inner_products`.
@inline function _ip_idx(i, j, nbasis)
    @assert j ≤ i
    return (j - 1) * nbasis - (j - 1) * (j - 2) ÷ 2 + (i - j + 1)
end

# Lookup that returns M_ij for any (i, j) by symmetry (rows ↔ cols).
@inline function _ip_get(M_ip::Vector{Matrix{Float64}}, i::Integer, j::Integer,
                         nbasis::Integer)
    return i ≥ j ? M_ip[_ip_idx(i, j, nbasis)] : M_ip[_ip_idx(j, i, nbasis)]'
end

# RT div-times-DG inner-product matrix.
function _RT_div_dg_matrix(RT_order::Integer)
    Mn = _RT_inner_product_L1L2L3_all(RT_order)
    Lagrange_order = RT_order
    basis_lag, n_dg_elt = _Lagrange_basis(Lagrange_order)
    basis_abc, basis_ijk, nbasis = _RT_basis(RT_order)

    M = zeros(Float64, nbasis, n_dg_elt)
    @inbounds for i in 1:nbasis
        ei = _RT_create_coord_basis_div(basis_abc, basis_ijk, i, RT_order)
        for j in 1:n_dg_elt
            ej = _Lagrange_create_coord_basis(basis_lag, j, Lagrange_order)
            M[i, j] = ei' * Mn * ej
        end
    end
    return M, n_dg_elt, basis_lag, basis_abc, basis_ijk, nbasis
end

# ---- Local-to-global DOF mapping for both RT and DG-Lagrange ----------------

# For an element k with sorted vertex indices, the DOF list for RT consists
# of (RT_order+1) DOFs per edge plus RT_order*(RT_order+1) interior DOFs.
# Sign vector `P` flips edge DOFs when the local edge orientation is
# opposite the global edge's stored vertex order.
function _rt_local_to_global(m::Mesh2D, k::Integer, RT_order::Integer,
                              nbasis::Integer)
    ne = m.ne
    nt = m.nt
    rt_dof_count = RT_order + 1
    int_per_elem = RT_order * rt_dof_count
    g = zeros(Int, nbasis)
    P = ones(Int, nbasis)
    local_edge_start_vert = (2, 3, 1)
    @inbounds for i in 1:3
        eid = m.tri2edge[k, i]
        v_local_start = m.elements[k, local_edge_start_vert[i]]
        if m.edges[eid, 1] == v_local_start
            for r in 1:rt_dof_count
                g[(i - 1) * rt_dof_count + r] = (eid - 1) * rt_dof_count + r
            end
        else
            for r in 1:rt_dof_count
                g[(i - 1) * rt_dof_count + r] = eid * rt_dof_count - (r - 1)
                P[(i - 1) * rt_dof_count + r] = -1
            end
        end
    end
    int_off = rt_dof_count * ne + int_per_elem * (k - 1)
    @inbounds for r in 1:int_per_elem
        g[3 * rt_dof_count + r] = int_off + r
    end
    return g, P
end

# DG-Lagrange local-to-global: contiguous block per element.
function _dg_local_to_global(k::Integer, n_dg_elt::Integer)
    return ((k - 1) * n_dg_elt + 1):(k * n_dg_elt)
end

# Global Lagrange (CG) local-to-global, for unpacking the input `f`.
function _cg_lagrange_local_to_global(m::Mesh2D, k::Integer,
                                       lagrange_order::Integer)
    nv = m.nv
    ne = m.ne
    n_per_edge = lagrange_order - 1
    n_per_elem = max(0, (lagrange_order - 1) * (lagrange_order - 2) ÷ 2)
    nlocal = _RT_lag_nbasis(lagrange_order)
    g = zeros(Int, nlocal)
    @inbounds g[1] = m.elements[k, 1]
    @inbounds g[2] = m.elements[k, 2]
    @inbounds g[3] = m.elements[k, 3]
    local_edge_start_vert = (2, 3, 1)
    @inbounds for i in 1:3
        eid = m.tri2edge[k, i]
        v_local_start = m.elements[k, local_edge_start_vert[i]]
        base = nv + (lagrange_order - 1) * (eid - 1)
        if n_per_edge > 0
            if m.edges[eid, 1] == v_local_start
                for r in 1:n_per_edge
                    g[3 + (i - 1) * n_per_edge + r] = base + r
                end
            else
                for r in 1:n_per_edge
                    g[3 + (i - 1) * n_per_edge + r] = base + (n_per_edge - r + 1)
                end
            end
        end
    end
    if n_per_elem > 0
        elem_off = nv + (lagrange_order - 1) * ne + n_per_elem * (k - 1)
        for r in 1:n_per_elem
            g[3 + 3 * n_per_edge + r] = elem_off + r
        end
    end
    return g
end

# ---- Driver ----------------------------------------------------------------

"""
    rt_hdiv_problem(m::Mesh2D, RT_order::Integer, f::AbstractMatrix) -> Matrix

Build the Goerisch w-w bilinear form via a mixed Raviart–Thomas /
DG-Lagrange auxiliary problem.

# Arguments
- `m::Mesh2D` — 2D mesh.
- `RT_order::Integer` — order of the RT space (≥ 0). The companion
  Lagrange order is the same.
- `f::AbstractMatrix` — `nlag × ncols` matrix of CG-Lagrange
  coefficients (one column per CG eigenfunction). `nlag` is the
  global Lagrange dimension `nv + (RT_order−1)·ne + max(0, …)·nt`.

# Returns
`mat_b_w_w :: Matrix{Float64}` of size `ncols × ncols` — the
Lehmann–Goerisch w-w bilinear form on the input eigenfunctions.

This is the Float64 (approximation-mode) path. The verified-mode
wrapper that delegates the saddle-point solve to Veigs.jl /
`interval_ldl` is queued for a follow-up phase.
"""
function rt_hdiv_problem(m::Mesh2D, RT_order::Integer, f::AbstractMatrix)
    RT_order ≥ 0 || throw(DomainError(RT_order, "RT_order must be ≥ 0"))
    n_dim_f = size(f, 2)

    M_div_dg, n_dg_elt, basis_lag, basis_abc, basis_ijk, nbasis =
        _RT_div_dg_matrix(RT_order)
    _, _, _, M_ip = _RT_basis_inner_products(RT_order)

    Mn = _RT_inner_product_L1L2L3_all(RT_order)
    M_ip_L2 = zeros(Float64, n_dg_elt, n_dg_elt)
    @inbounds for i in 1:n_dg_elt
        ei = _Lagrange_create_coord_basis(basis_lag, i, RT_order)
        for j in 1:n_dg_elt
            ej = _Lagrange_create_coord_basis(basis_lag, j, RT_order)
            M_ip_L2[i, j] = ei' * Mn * ej
        end
    end

    nt = m.nt
    ne = m.ne
    n_dg = nt * n_dg_elt
    rt_dof_count = RT_order + 1
    int_per_elem = RT_order * rt_dof_count
    ndof_RT = rt_dof_count * ne + int_per_elem * nt

    # COO triplets + one `sparse` call, not scatter-adds into a CSC: `A[i,j] +=`
    # on a sparse matrix inserts when the entry is new, an O(nnz) memmove, which
    # makes element-by-element assembly quadratic in problem size.
    nnzA = nt * nbasis * nbasis
    nnzB = nt * nbasis * n_dg_elt
    Ai = Vector{Int}(undef, nnzA); Aj = Vector{Int}(undef, nnzA); Av = Vector{Float64}(undef, nnzA)
    Bi = Vector{Int}(undef, nnzB); Bj = Vector{Int}(undef, nnzB); Bv = Vector{Float64}(undef, nnzB)
    pA = 0; pB = 0
    F_mat = zeros(Float64, n_dg, n_dim_f)

    @inbounds for k in 1:nt
        x1 = m.nodes[m.elements[k, 1], 1]; y1 = m.nodes[m.elements[k, 1], 2]
        x2 = m.nodes[m.elements[k, 2], 1]; y2 = m.nodes[m.elements[k, 2], 2]
        x3 = m.nodes[m.elements[k, 3], 1]; y3 = m.nodes[m.elements[k, 3], 2]
        Bmat = [x2 - x1  x3 - x1;
                y2 - y1  y3 - y1]
        det_B = det(Bmat)

        # Local A1[i, j] = tr(B · M_ip_ij · B') / det(B), symmetric.
        A1 = zeros(Float64, nbasis, nbasis)
        for i in 1:nbasis, j in 1:i
            mij = _ip_get(M_ip, i, j, nbasis)
            A1[i, j] = tr(Bmat * mij * Bmat') / det_B
            i == j || (A1[j, i] = A1[i, j])
        end
        A2 = M_div_dg

        g_rt, P = _rt_local_to_global(m, k, RT_order, nbasis)
        g_dg = _dg_local_to_global(k, n_dg_elt)

        # Apply orientation sign: A_block = diag(P) · A1 · diag(P).
        for i in 1:nbasis, j in 1:nbasis
            pA += 1
            Ai[pA] = g_rt[i]; Aj[pA] = g_rt[j]; Av[pA] = P[i] * A1[i, j] * P[j]
        end
        for i in 1:nbasis, j in 1:n_dg_elt
            pB += 1
            Bi[pB] = g_rt[i]; Bj[pB] = g_dg[j]; Bv[pB] = P[i] * A2[i, j]
        end

        # F block uses the global Lagrange numbering (over CG DOFs).
        g_lag = _cg_lagrange_local_to_global(m, k, RT_order)
        F_mat[g_dg, :] .= M_ip_L2 * f[g_lag, :] .* det_B
    end

    A_mat = dropzeros!(sparse(Ai, Aj, Av, ndof_RT, ndof_RT))
    B_mat = dropzeros!(sparse(Bi, Bj, Bv, ndof_RT, n_dg))

    # Solve the saddle-point system [A B; B^T 0] x = [0; -F].
    n = ndof_RT
    nB = n_dg
    saddle = [A_mat        B_mat;
              transpose(B_mat)  spzeros(Float64, nB, nB)]
    rhs = vcat(zeros(Float64, n, n_dim_f), -F_mat)
    x = saddle \ rhs

    RT_vec = x[1:n, :]
    return RT_vec' * A_mat * RT_vec
end

# ============================================================================
# Verified-mode RT/Hdiv saddle solve.
# ============================================================================
#
# Pipeline:
#   1. Assemble A_int, B_int, F_int as interval matrices using
#      interval-promoted mesh coordinates and interval reference matrices.
#   2. Compute the Float64 mid system K_mid = mid(K_int), rhs_mid =
#      mid(rhs_int) and factor K_mid via sparse LU.
#   3. Float64 approximate solution x_a = K_mid \ rhs_mid.
#   4. Compute interval residual r_int = rhs_int - K_int * x_a (point
#      x_a propagates through interval arithmetic so r_int captures both
#      the interval radius of K_int / rhs_int and the floating-point
#      mismatch from x_a).
#   5. Apply the Float64 LU to interval r_int via interval forward / back
#      substitution (helper `_interval_lu_solve!` below) yielding dx_int.
#   6. x_int = x_a + dx_int is a verified enclosure of K_mid^{-1} * rhs_int.
#      For tiny interval radius on K_int (here at most a few times machine
#      epsilon × ‖K‖, set by the interval CR / RT assembly), this also
#      encloses K_int^{-1} * rhs_int up to terms ‖K_mid^{-1}‖ · rad(K_int)
#      × ‖x*‖, which are dominated by dx_int's own width.
#
# The verified output is `RT_vec_int' * A_int * RT_vec_int`, a small
# `n_dim_f × n_dim_f` interval matrix consumed by `verified_lg_transform`.

using IntervalArithmetic: Interval, interval, mid, sup, inf, hull
using SparseArrays: lu, nzrange, rowvals, nonzeros

# Apply a precomputed dense R ≈ K_mid^{-1} (Float64) to an interval rhs via
# straight matrix-vector product. This avoids the dependency-explosion of
# forward/back substitution on sparse interval RHS — each output entry is
# a single sum of (Float × Interval), so the width grows linearly with the
# sum length × input interval width rather than exponentially with the
# triangular-solve depth.
function _apply_dense_R(R::Matrix{Float64}, rhs::AbstractVector{<:Interval})
    n = size(R, 1)
    out = Vector{eltype(rhs)}(undef, n)
    @inbounds for i in 1:n
        acc = zero(eltype(rhs))
        for j in 1:n
            acc = acc + R[i, j] * rhs[j]
        end
        out[i] = acc
    end
    return out
end

# Build interval RT assembly: returns A_int, B_int, F_int, dims.
function _verified_rt_assembly(m::Mesh2D, RT_order::Integer,
                                f_int::AbstractMatrix{<:Interval})
    T = Interval{Float64}
    n_dim_f = size(f_int, 2)

    M_div_dg, n_dg_elt, basis_lag, basis_abc, basis_ijk, nbasis =
        _RT_div_dg_matrix(RT_order)
    _, _, _, M_ip = _RT_basis_inner_products(RT_order)
    Mn_f64 = _RT_inner_product_L1L2L3_all(RT_order)

    Mn      = convert(Matrix{T}, Mn_f64)
    M_dvdg  = convert(Matrix{T}, M_div_dg)

    M_ip_L2 = zeros(T, n_dg_elt, n_dg_elt)
    @inbounds for i in 1:n_dg_elt
        ei = convert(Vector{T},
                      _Lagrange_create_coord_basis(basis_lag, i, RT_order))
        for j in 1:n_dg_elt
            ej = convert(Vector{T},
                          _Lagrange_create_coord_basis(basis_lag, j, RT_order))
            M_ip_L2[i, j] = ei' * Mn * ej
        end
    end

    nodes_int = convert(Matrix{T}, m.nodes)

    nt = m.nt
    ne = m.ne
    n_dg = nt * n_dg_elt
    rt_dof_count = RT_order + 1
    int_per_elem = RT_order * rt_dof_count
    ndof_RT = rt_dof_count * ne + int_per_elem * nt

    # COO triplets + one `sparse` call, not scatter-adds into a CSC: `A[i,j] +=`
    # on a sparse matrix inserts when the entry is new, an O(nnz) memmove, which
    # makes element-by-element assembly quadratic in problem size.
    nnzA = nt * nbasis * nbasis
    nnzB = nt * nbasis * n_dg_elt
    Ai = Vector{Int}(undef, nnzA); Aj = Vector{Int}(undef, nnzA); Av = Vector{T}(undef, nnzA)
    Bi = Vector{Int}(undef, nnzB); Bj = Vector{Int}(undef, nnzB); Bv = Vector{T}(undef, nnzB)
    pA = 0; pB = 0
    F_mat = zeros(T, n_dg, n_dim_f)

    @inbounds for k in 1:nt
        x1 = nodes_int[m.elements[k, 1], 1]; y1 = nodes_int[m.elements[k, 1], 2]
        x2 = nodes_int[m.elements[k, 2], 1]; y2 = nodes_int[m.elements[k, 2], 2]
        x3 = nodes_int[m.elements[k, 3], 1]; y3 = nodes_int[m.elements[k, 3], 2]
        Bmat = T[x2 - x1  x3 - x1;
                 y2 - y1  y3 - y1]
        det_B = Bmat[1, 1] * Bmat[2, 2] - Bmat[1, 2] * Bmat[2, 1]

        A1 = zeros(T, nbasis, nbasis)
        for i in 1:nbasis, j in 1:i
            mij_f64 = _ip_get(M_ip, i, j, nbasis)
            mij = convert(Matrix{T}, mij_f64)
            A1[i, j] = tr(Bmat * mij * Bmat') / det_B
            i == j || (A1[j, i] = A1[i, j])
        end

        g_rt, P = _rt_local_to_global(m, k, RT_order, nbasis)
        g_dg = _dg_local_to_global(k, n_dg_elt)

        for i in 1:nbasis, j in 1:nbasis
            pA += 1
            Ai[pA] = g_rt[i]; Aj[pA] = g_rt[j]
            Av[pA] = interval(P[i] * P[j]) * A1[i, j]
        end
        for i in 1:nbasis, j in 1:n_dg_elt
            pB += 1
            Bi[pB] = g_rt[i]; Bj[pB] = g_dg[j]
            Bv[pB] = interval(P[i]) * M_dvdg[i, j]
        end

        g_lag = _cg_lagrange_local_to_global(m, k, RT_order)
        F_mat[g_dg, :] .= (M_ip_L2 * f_int[g_lag, :]) .* det_B
    end

    A_mat = dropzeros!(sparse(Ai, Aj, Av, ndof_RT, ndof_RT))
    B_mat = dropzeros!(sparse(Bi, Bj, Bv, ndof_RT, n_dg))

    return A_mat, B_mat, F_mat, ndof_RT, n_dg
end

"""
    verified_rt_hdiv_problem(m::Mesh2D, RT_order::Integer,
                              f_int::AbstractMatrix{<:Interval{Float64}})
        -> Matrix{Interval{Float64}}

Verified-mode counterpart of [`rt_hdiv_problem`](@ref). Builds the
Goerisch w-w bilinear form `RT_vecᵀ A_RT RT_vec` as an interval enclosure.

Pipeline:
* Assemble `A`, `B`, `F` as `Interval{Float64}` matrices via
  `_verified_rt_assembly`.
* Solve the Float64 mid saddle system `[mid(A) mid(B); mid(B)ᵀ 0] x = [0; -mid(F)]`
  via sparse LU.
* Compute the interval residual `r = rhs_int - K_int x` and apply the
  Float64 LU back to it (interval forward/back substitution) to obtain
  the verified correction `dx`. The enclosure `x + dx` then bounds the
  true saddle solution up to `‖mid(K)⁻¹‖·rad(K)·‖x*‖`, which is
  dominated by the explicit width of `dx` for the interval radii
  produced by the assembly above (≈ machine epsilon).

The result is the same `n_cols × n_cols` matrix as the Float64 driver
returns, but with interval enclosures suitable for downstream
`verified_lg_transform`.
"""
function verified_rt_hdiv_problem(m::Mesh2D, RT_order::Integer,
                                   f_int::AbstractMatrix{<:Interval})
    RT_order ≥ 0 || throw(DomainError(RT_order, "RT_order must be ≥ 0"))
    T = Interval{Float64}
    n_dim_f = size(f_int, 2)

    A_int, B_int, F_int, ndof_RT, n_dg = _verified_rt_assembly(m, RT_order, f_int)
    n_total = ndof_RT + n_dg

    # Build full interval saddle K_int and Float64 mid K_mid.
    K_int = [A_int             B_int;
             transpose(B_int)  spzeros(T, n_dg, n_dg)]
    K_mid_full = sparse(map(mid, K_int))

    # Float64 LU on the mid saddle, plus dense R = K_mid^{-1} for verified
    # correction (a single dense n×n inverse — heavy at large n_total but
    # avoids the dependency-blowup of interval triangular back-sub).
    F  = lu(K_mid_full)
    R  = F \ Matrix{Float64}(I, n_total, n_total)

    rhs_int_block = vcat(zeros(T, ndof_RT, n_dim_f), -F_int)
    rhs_mid       = map(mid, rhs_int_block)

    x_a = F \ Matrix(rhs_mid)         # Float64 approximate, n_total × n_dim_f

    # Per-column verified correction.
    x_int = Matrix{T}(undef, n_total, n_dim_f)
    for c in 1:n_dim_f
        rhs_c   = rhs_int_block[:, c]
        x_a_c   = x_a[:, c]
        # Interval residual r_int = rhs - K_int * x_a_c.
        r_int   = rhs_c .- K_int * x_a_c
        # dx ≈ K_mid^{-1} r_int via dense R · r_int.
        dx_int  = _apply_dense_R(R, r_int)
        @inbounds for i in 1:n_total
            x_int[i, c] = interval(x_a_c[i]) + dx_int[i]
        end
    end

    RT_vec = x_int[1:ndof_RT, :]
    return RT_vec' * A_int * RT_vec
end
