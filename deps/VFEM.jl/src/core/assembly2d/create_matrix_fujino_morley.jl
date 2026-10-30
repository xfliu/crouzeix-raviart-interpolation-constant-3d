# src/core/assembly2d/create_matrix_fujino_morley.jl
#
# Fujino–Morley (FM) P2 element on a triangulation: local DOF matrix Σ, the
# local FM → degree-2 Bernstein transform Σ⁻¹, the exact local stiffness of the
# discretised H² seminorm, and the global sparse pair (A, B).
#
# Provenance: written for the inverse-free replacement of Lemma 3.2 in
# J. Galindo, K. Ike, X. Liu (manuscript on L^∞ Lagrange interpolation error
# constants), and cross-validated entry-by-entry against the authors' Octave
# implementation. There is no vfem2d MATLAB counterpart — FM is the element
# this library did not previously have.
#
# The space discretised is the interpolation-error space on K
#
#   V^FM_h(K) = { v : v|_Kh ∈ P2, v continuous at the mesh nodes,
#                 edge-average of the jump of ∂v/∂n zero across interior edges,
#                 v(p_i) = 0 at the THREE CORNERS of K },
#
# so there is NO boundary condition on the sides of K beyond the three corner
# values: the DOFs are every mesh vertex except the three corners, plus EVERY
# mesh edge (interior and boundary alike),
#
#   M = (nv − 3) + ne,      N = 6·nt.
#
# Matrices
#   A (M × M) : a_ij = Σ_Kh ∫_Kh (u_xx v_xx + 2 u_xy v_xy + u_yy v_yy)
#   B (N × M) : FM coefficient vector ↦ stacked degree-2 Bernstein
#               coefficients, 6 per element
#
# Local DOFs (σ₁..σ₆) on an element with counter-clockwise vertices q1, q2, q3:
#   σ_a(v)     = v(q_a),                             a = 1, 2, 3
#   σ_{3+m}(v) = (1/|e_m|) ∫_{e_m} ∂v/∂n_m ds,       m = 1, 2, 3
# with e_m the edge OPPOSITE vertex m — matching `tri2edge` column m and the
# local edge order (v2,v3), (v1,v3), (v1,v2) used everywhere in this library.
#
# Degree-2 Bernstein basis, in the authors' ordering:
#   J₁ = l₁², J₂ = l₂², J₃ = l₃²,  J_{3+m} = 2·l_a·l_b, {a,b} = {1,2,3} \ {m},
# so the mixed function J_{3+m} is again attached to the edge opposite vertex m.
# The three vertex-value rows of Σ are therefore [I 0]: Σ = [I 0; C D] is block
# lower triangular, and Σ⁻¹ = [I 0; −D⁻¹C  D⁻¹] needs a single 3×3 cofactor
# inverse — no pivoting, which keeps `Interval{Float64}` enclosures tight.
#
# Because v|_Kh is P2, its Hessian is CONSTANT on each element: the local
# stiffness is |Kh| times a contraction of constant Hessians, EXACTLY, with no
# quadrature. All entries are rational in the vertex coordinates.
#
# ---------------------------------------------------------------------------
# EDGE-NORMAL ORIENTATION — the `convention` keyword
# ---------------------------------------------------------------------------
# With E_m the cyclic edge vector opposite vertex m (E₁ = q3−q2, E₂ = q1−q3,
# E₃ = q2−q1) and P(v) = (−v₂, v₁), one has grad l_m = P(E_m)/(2|Kh|), so
# n^out_m := −P(E_m) is the OUTWARD normal of that element on that edge, of
# magnitude |e_m|. Two elements sharing an edge have OPPOSITE outward normals,
# so the outward choice does not define a single global functional per edge.
#
# `:consistent` (default, mathematically correct) — one absolute normal
#   direction per edge, taken from the edge's REFERENCE element: the
#   lowest-numbered of its adjacent elements. Both adjacent elements then
#   evaluate the identical functional, the FM inter-element continuity
#   constraint is the intended one, and the assembled space really is V^FM_h.
#
# `:verbatim` — reproduces the authors' published table. Their code conjugates
#   the local stiffness on downward-pointing elements by
#   tempT = diag(1,1,1,−1,−1,−1) but does NOT apply it to the local
#   FM → Bernstein block written into B. In geometric terms: A uses the
#   consistent normal (both conventions give the SAME A), while B's local block
#   uses the per-element OUTWARD normal. The result is that on interior edges
#   the two adjacent elements disagree about the sign of the edge DOF, and the
#   reconstructed function is not in the FM space.
#
#   Evidence, reproduced by `test/test_create_matrix_fujino_morley.jl`: the
#   edge-average of the normal-derivative jump across interior edges vanishes
#   for `:consistent` and equals exactly 2× the one-sided average for
#   `:verbatim`; and q = l₁·l₂ on K — a global quadratic vanishing at all three
#   corners, hence in V^FM_h — is reproduced to ~1e-15 by `:consistent` but with
#   O(1) relative error by `:verbatim` (measured 1.25 at n = 4, theta = pi/2 and
#   2.05 at n = 3, theta = 3pi/4 — the reconstruction is simply not the intended
#   function, not a mildly perturbed one).
#
#   `:verbatim` is retained because it is the bit-level oracle for the published
#   numbers, and BOTH conventions are carried side by side, never rescaled:
#   `:consistent` yields a SHARPER interpolation constant. Note that λ_{h,B} is
#   invariant under any sign change of the per-edge reference direction for
#   `:consistent` (a diagonal rescaling of the DOF basis rescales A and B
#   together) but NOT for `:verbatim` — that non-invariance is itself a symptom
#   of the inconsistency, and it is why `:verbatim` pins the reference direction
#   to the lowest-numbered adjacent element, which on the meshes of
#   `mesh2d_triangle_uniform` is the upward-pointing triangle, exactly the
#   authors' reference element.
#
# Routines are generic in the element type via the `T::Type = Float64` keyword
# (the library's mode-switch idiom); pass `T = Interval{Float64}` for the
# rigorous mode.

using LinearAlgebra: dot, SymTridiagonal, eigen, transpose
using SparseArrays: sparse, SparseMatrixCSC

# 90° counter-clockwise rotation
@inline _fm_rot90(v1, v2) = (-v2, v1)

# Largest entry magnitude, as a plain Float64, for either element type. Needed
# because `==` and `maximum` on `Interval` operands are INCONCLUSIVE (they raise
# `InconclusiveBooleanOperation`), so the symmetry diagnostics below cannot use
# them directly. For an interval entry the magnitude bound `sup(mag(x))` is zero
# exactly when the entry is the zero-width interval [0, 0], so
# `_fm_absmax(A - A') == 0` still means "A is exactly symmetric" in both modes.
@inline _fm_mag1(x::Real) = abs(float(x))
@inline _fm_mag1(x::Interval) = sup(mag(x))
_fm_absmax(A) = isempty(A) ? 0.0 : maximum(_fm_mag1, A)

# cyclic successor pair of the local index m: the two vertices of the edge
# opposite vertex m, in the order (v_{m+1}, v_{m+2})
@inline _fm_cyc2(m::Int) = (mod(m, 3) + 1, mod(m + 1, 3) + 1)

"""
    fm_inv3(Mm::AbstractMatrix{T}) -> Matrix{T}

Cofactor (adjugate / determinant) inverse of a 3×3 matrix. Pivot-free, so it is
both safe and tight for `T = Interval{Float64}`, where `inv` would branch on
comparisons of intervals.
"""
function fm_inv3(Mm::AbstractMatrix{T}) where {T}
    a, b, c = Mm[1, 1], Mm[1, 2], Mm[1, 3]
    d, e, f = Mm[2, 1], Mm[2, 2], Mm[2, 3]
    g, h, i = Mm[3, 1], Mm[3, 2], Mm[3, 3]
    A11 = e * i - f * h; A12 = c * h - b * i; A13 = b * f - c * e
    A21 = f * g - d * i; A22 = a * i - c * g; A23 = c * d - a * f
    A31 = d * h - e * g; A32 = b * g - a * h; A33 = a * e - b * d
    det = a * A11 + b * A21 + c * A31
    return [A11/det A12/det A13/det
            A21/det A22/det A23/det
            A31/det A32/det A33/det]
end

"""
    fm_element_geometry(q::AbstractMatrix; T::Type = Float64)
        -> (area, g, hess, nout)

Geometric primitives of one triangle.

* `q`    — `3 × 2` vertex coordinates, counter-clockwise.
* `area` — `|Kh|`, positive for counter-clockwise `q`.
* `g`    — `3 × 2`; row `m` is `grad l_m = P(E_m) / (2|Kh|)`.
* `hess` — `6 × 3`; row `m` is the CONSTANT Hessian `(v_xx, v_xy, v_yy)` of the
           `m`-th degree-2 Bernstein basis function.
* `nout` — `3 × 2`; row `m` is the OUTWARD normal `−P(E_m)` of the edge opposite
           vertex `m`, of magnitude `|e_m|` (unnormalised, as the DOF requires).
"""
function fm_element_geometry(q::AbstractMatrix; T::Type = Float64)
    qq = T.(q)
    E = Matrix{T}(undef, 3, 2)
    @inbounds for m in 1:3
        (a, b) = _fm_cyc2(m)
        E[m, 1] = qq[b, 1] - qq[a, 1]
        E[m, 2] = qq[b, 2] - qq[a, 2]
    end
    two = convert(T, 2)
    # |Kh| = ((q2−q1) × (q3−q1)) / 2, with q2−q1 = E₃ and q3−q1 = −E₂
    area = (E[3, 1] * (-E[2, 2]) - E[3, 2] * (-E[2, 1])) / two
    g = Matrix{T}(undef, 3, 2)
    nout = Matrix{T}(undef, 3, 2)
    @inbounds for m in 1:3
        (r1, r2) = _fm_rot90(E[m, 1], E[m, 2])
        g[m, 1] = r1 / (two * area)
        g[m, 2] = r2 / (two * area)
        nout[m, 1] = -r1
        nout[m, 2] = -r2
    end
    # constant Hessians of the Bernstein basis, as (xx, xy, yy):
    #   J_m     = l_m²      → H = 2 g_m ⊗ g_m
    #   J_{3+m} = 2 l_a l_b → H = 2 (g_a ⊗ g_b + g_b ⊗ g_a)
    hess = Matrix{T}(undef, 6, 3)
    @inbounds for m in 1:3
        hess[m, 1] = two * g[m, 1] * g[m, 1]
        hess[m, 2] = two * g[m, 1] * g[m, 2]
        hess[m, 3] = two * g[m, 2] * g[m, 2]
    end
    @inbounds for m in 1:3
        (a, b) = _fm_cyc2(m)
        hess[3+m, 1] = two * (two * g[a, 1] * g[b, 1])
        hess[3+m, 2] = two * (g[a, 1] * g[b, 2] + g[b, 1] * g[a, 2])
        hess[3+m, 3] = two * (two * g[a, 2] * g[b, 2])
    end
    return area, g, hess, nout
end

"""
    fm_sigma(q::AbstractMatrix, nrm::AbstractMatrix; T::Type = Float64) -> Matrix{T}

The `6 × 6` FM DOF matrix `Σ[a, m] = σ_a(J_m)` of the element with
counter-clockwise vertices `q` (`3 × 2`) and edge-DOF normals `nrm` (`3 × 2`,
row `m` for the edge opposite vertex `m`, unnormalised of magnitude `|e_m|`).

Closed forms (from `grad l_m = P(E_m)/(2|Kh|)` and the edge averages
`avg_{e_i} l_j = 1/2` for `j ≠ i`, `avg_{e_i} l_i = 0`):

    Σ[1:3, 1:3] = I,   Σ[1:3, 4:6] = 0
    Σ[3+i, j]   = (i == j) ? 0     : grad l_j · n_i
    Σ[3+i, 3+j] = (i == j) ? −(grad l_i · n_i) : grad l_i · n_i

which reproduce the authors' `C`, `D` blocks exactly when `n_i` is the outward
normal `−P(E_i)`.
"""
function fm_sigma(q::AbstractMatrix, nrm::AbstractMatrix; T::Type = Float64)
    _, g, _, _ = fm_element_geometry(q; T = T)
    nn = T.(nrm)
    Sig = zeros(T, 6, 6)
    @inbounds for a in 1:3
        Sig[a, a] = one(T)
    end
    @inbounds for i in 1:3
        gini = g[i, 1] * nn[i, 1] + g[i, 2] * nn[i, 2]      # grad l_i · n_i
        for j in 1:3
            if i != j
                Sig[3+i, j] = g[j, 1] * nn[i, 1] + g[j, 2] * nn[i, 2]
            end
            Sig[3+i, 3+j] = (i == j) ? -gini : gini
        end
    end
    return Sig
end

"""
    fm_local_matrices(q::AbstractMatrix, nrm::AbstractMatrix; T::Type = Float64)
        -> (A_loc, B_loc, Sigma, A_bern, area)

Local FM matrices of one element with counter-clockwise vertices `q` (`3 × 2`)
and edge-DOF normals `nrm` (`3 × 2`).

* `Sigma`  — `6 × 6` DOF matrix, `Σ[a, m] = σ_a(J_m)`; see [`fm_sigma`](@ref).
* `B_loc`  — `Σ⁻¹`, the FM-coefficient → Bernstein-coefficient map (`d = B_loc x`),
             equivalently `φ_i = Σ_m B_loc[m, i] J_m`. Built from the block
             structure `Σ = [I 0; C D]` as `[I 0; −D⁻¹C  D⁻¹]`, so only one 3×3
             cofactor inverse is needed (no pivoting; interval-safe).
* `A_bern` — `6 × 6` Gram matrix of the H² seminorm in the Bernstein basis,
             `|Kh| · (H_m : H_m')` with the contraction
             `u_xx v_xx + 2 u_xy v_xy + u_yy v_yy`. Exact, no quadrature.
* `A_loc`  — `B_loc' · A_bern · B_loc`, the local stiffness in the FM basis.
             Symmetric positive semidefinite with kernel exactly P1 restricted
             to the element, i.e. of dimension 3.
* `area`   — `|Kh|`.
"""
function fm_local_matrices(q::AbstractMatrix, nrm::AbstractMatrix;
                           T::Type = Float64)
    area, g, hess, _ = fm_element_geometry(q; T = T)
    nn = T.(nrm)
    two = convert(T, 2)
    A_bern = Matrix{T}(undef, 6, 6)
    @inbounds for m in 1:6, mp in 1:6
        A_bern[m, mp] = area * (hess[m, 1] * hess[mp, 1] +
                                two * hess[m, 2] * hess[mp, 2] +
                                hess[m, 3] * hess[mp, 3])
    end
    Sig = zeros(T, 6, 6)
    @inbounds for a in 1:3
        Sig[a, a] = one(T)
    end
    Cb = zeros(T, 3, 3)
    Db = Matrix{T}(undef, 3, 3)
    @inbounds for i in 1:3
        gini = g[i, 1] * nn[i, 1] + g[i, 2] * nn[i, 2]
        for j in 1:3
            if i != j
                Cb[i, j] = g[j, 1] * nn[i, 1] + g[j, 2] * nn[i, 2]
            end
            Db[i, j] = (i == j) ? -gini : gini
        end
    end
    Sig[4:6, 1:3] = Cb
    Sig[4:6, 4:6] = Db
    Dinv = fm_inv3(Db)
    B_loc = zeros(T, 6, 6)
    @inbounds for a in 1:3
        B_loc[a, a] = one(T)
    end
    B_loc[4:6, 1:3] = -Dinv * Cb
    B_loc[4:6, 4:6] = Dinv
    A_loc = transpose(B_loc) * A_bern * B_loc
    return A_loc, B_loc, Sig, A_bern, area
end

"""
    fm_reference_elements(m::Mesh2D) -> Vector{Int}

For each edge, the index of its REFERENCE element: the lowest-numbered element
adjacent to it. Two elements sharing an edge have opposite outward normals, so
this single choice fixes one absolute normal direction per edge and hence one
globally consistent edge functional.

On the meshes of [`mesh2d_triangle_uniform`](@ref) upward-pointing triangles are
numbered first, so the reference element of every edge is its upward neighbour —
the authors' reference element.
"""
function fm_reference_elements(m::Mesh2D)
    ref = zeros(Int, m.ne)
    @inbounds for k in 1:m.nt, j in 1:3
        e = m.tri2edge[k, j]
        if ref[e] == 0
            ref[e] = k
        end
    end
    all(>(0), ref) || error("edge with no adjacent element in tri2edge")
    return ref
end

"""
    fm_edge_signs(m::Mesh2D, ref::Vector{Int}, k::Integer) -> NTuple{3, Int}

Signs relating element `k`'s OUTWARD normals to the per-edge reference
directions of [`fm_reference_elements`](@ref): `+1` on edges for which `k` is
itself the reference element, `−1` otherwise (the two elements sharing an edge
have opposite outward normals).
"""
function fm_edge_signs(m::Mesh2D, ref::Vector{Int}, k::Integer)
    e1, e2, e3 = m.tri2edge[k, 1], m.tri2edge[k, 2], m.tri2edge[k, 3]
    return (ref[e1] == k ? 1 : -1,
            ref[e2] == k ? 1 : -1,
            ref[e3] == k ? 1 : -1)
end

"""
    create_matrix_fujino_morley(m::Mesh2D, corners::NTuple{3, Int};
                                convention::Symbol = :consistent,
                                symmetrize::Bool = true,
                                T::Type = Float64) -> NamedTuple

Global Fujino–Morley assembly on the mesh `m`, with the three vertex values at
`corners` constrained to zero. `corners` is the tuple returned alongside the
mesh by [`mesh2d_triangle_uniform`](@ref).

Keyword arguments
* `convention` — `:consistent` (default, mathematically correct) or `:verbatim`
  (reproduces the authors' published table). `A` is IDENTICAL for both; they
  differ only in the local block written into `B`. See the file header for the
  full discussion and for the two tests that separate them.
* `symmetrize` — replace each local block by `(A_loc + A_loc')/2` before
  scattering. The 6×6 floating-point triple product `B_loc' A_bern B_loc` is
  asymmetric at the 1e-16..1e-15 relative level even though `A_bern` is exactly
  symmetric; a dense LU never notices, but CHOLMOD's `cholesky` tests symmetry
  EXACTLY and throws `ArgumentError: sparse matrix is not symmetric/Hermitian`.
  Symmetrizing the local block once, at the source, makes the assembled `A`
  exactly symmetric (`issymmetric(A) == true`), because the same symmetric block
  is scattered for both `(i,j)` and `(j,i)`. Measured effect on λ_{h,B}:
  ≤ 1.8e-12 relative. Pass `false` to reproduce the raw published matrix — then
  `cholesky` is unusable.
* `T` — element type; `Float64` (approximation) or `Interval{Float64}` (verified).

DOF numbering: the uncondensed index of vertex `i` is `i` and of edge `e` is
`nv + e`; the three corner indices are then deleted, preserving the order of the
survivors. Hence

    M = (nv − 3) + ne,      N = 6·nt

(`M = 8382`, `N = 24576` at `n = 64`, the paper's production size).

Returns a NamedTuple
* `A::SparseMatrixCSC{T}` — `M × M`, symmetric, positive definite on the
  constrained space.
* `B::SparseMatrixCSC{T}` — `N × M`; rows `6(k−1)+1 : 6k` hold element `k`'s
  degree-2 Bernstein coefficients. Each row has AT MOST 6 nonzeros — exactly 6
  before the corner columns are deleted — which is the structural fact the
  inverse-free solver [`lambda_hb_fast`](@ref) exploits.
* `M`, `N` — dimensions.
* `dofmap::Vector{Int}` — length `nv + ne`, uncondensed index → condensed DOF
  (`0` marks a deleted corner).
* `Aloc`, `Bloc` — per-element `6 × 6` local matrices, for testing.
* `convention`, `asym`, `issym` — the convention used, the observed
  `max|A − A'|`, and `issymmetric(A)`.
"""
function create_matrix_fujino_morley(m::Mesh2D, corners::NTuple{3, Int};
                                     convention::Symbol = :consistent,
                                     symmetrize::Bool = true,
                                     T::Type = Float64)
    convention in (:consistent, :verbatim) ||
        throw(ArgumentError("convention must be :consistent or :verbatim, " *
                            "got :$convention"))
    all(c -> 1 ≤ c ≤ m.nv, corners) ||
        throw(ArgumentError("corners must be valid vertex indices"))
    length(unique(corners)) == 3 ||
        throw(ArgumentError("corners must be three distinct vertices"))

    nv, nt, ne = m.nv, m.nt, m.ne
    ndof_raw = nv + ne
    dofmap = zeros(Int, ndof_raw)
    is_corner = falses(ndof_raw)
    for c in corners
        is_corner[c] = true
    end
    cnt = 0
    for i in 1:ndof_raw
        if !is_corner[i]
            cnt += 1
            dofmap[i] = cnt
        end
    end
    Mdim = cnt
    Mdim == (nv - 3) + ne || error("condensed dimension mismatch: $Mdim")
    Ndim = 6 * nt

    ref = fm_reference_elements(m)

    Ia = Int[]; Ja = Int[]; Va = T[]
    Ib = Int[]; Jb = Int[]; Vb = T[]
    sizehint!(Ia, 36 * nt); sizehint!(Ja, 36 * nt); sizehint!(Va, 36 * nt)
    sizehint!(Ib, 36 * nt); sizehint!(Jb, 36 * nt); sizehint!(Vb, 36 * nt)

    Aloc = Vector{Matrix{T}}(undef, nt)
    Bloc = Vector{Matrix{T}}(undef, nt)
    gl = Vector{Int}(undef, 6)

    for k in 1:nt
        v = @view m.elements[k, :]
        q = m.nodes[v, :]
        sg = fm_edge_signs(m, ref, k)
        _, _, _, nout = fm_element_geometry(q; T = T)
        # consistent (reference-element) normals: used by A in BOTH conventions
        nrm = Matrix{T}(undef, 3, 2)
        @inbounds for j in 1:3
            s = convert(T, sg[j])
            nrm[j, 1] = s * nout[j, 1]
            nrm[j, 2] = s * nout[j, 2]
        end
        A_raw, B_con, _, _, _ = fm_local_matrices(q, nrm; T = T)
        A_k = symmetrize ? (A_raw + transpose(A_raw)) / convert(T, 2) : A_raw
        # B's local block: outward normals under :verbatim, which is
        # B_con · diag(1, 1, 1, s₁, s₂, s₃)
        if convention === :verbatim
            B_k = copy(B_con)
            @inbounds for j in 1:3
                if sg[j] == -1
                    for r in 1:6
                        B_k[r, 3+j] = -B_k[r, 3+j]
                    end
                end
            end
        else
            B_k = B_con
        end
        Aloc[k] = A_k
        Bloc[k] = B_k

        @inbounds for a in 1:3
            gl[a]   = dofmap[v[a]]
            gl[3+a] = dofmap[nv + m.tri2edge[k, a]]
        end
        @inbounds for a in 1:6
            ga = gl[a]
            ga == 0 && continue
            for b in 1:6
                gb = gl[b]
                gb == 0 && continue
                push!(Ia, ga); push!(Ja, gb); push!(Va, A_k[a, b])
            end
        end
        row0 = 6 * (k - 1)
        @inbounds for mm in 1:6, a in 1:6
            ga = gl[a]
            ga == 0 && continue
            push!(Ib, row0 + mm); push!(Jb, ga); push!(Vb, B_k[mm, a])
        end
    end

    A = sparse(Ia, Ja, Va, Mdim, Mdim)
    B = sparse(Ib, Jb, Vb, Ndim, Mdim)
    asym = _fm_absmax(A - transpose(A))
    return (A = A, B = B, M = Mdim, N = Ndim, dofmap = dofmap,
            Aloc = Aloc, Bloc = Bloc, convention = convention,
            asym = asym, issym = asym == 0.0)
end

# ---------------------------------------------------------------------------
# Evaluation helpers — used by the unit tests to check the DOF duality, the P1
# reproduction and the interior-edge jump INDEPENDENTLY of `fm_sigma`.
# ---------------------------------------------------------------------------

"""
    fm_bary(q::AbstractMatrix, x, y) -> (l1, l2, l3)

Barycentric coordinates of `(x, y)` in the triangle `q` (`3 × 2`).
"""
function fm_bary(q::AbstractMatrix, x, y)
    d = (q[2,1]-q[1,1])*(q[3,2]-q[1,2]) - (q[3,1]-q[1,1])*(q[2,2]-q[1,2])
    l2 = ((x-q[1,1])*(q[3,2]-q[1,2]) - (q[3,1]-q[1,1])*(y-q[1,2])) / d
    l3 = ((q[2,1]-q[1,1])*(y-q[1,2]) - (x-q[1,1])*(q[2,2]-q[1,2])) / d
    return (1 - l2 - l3, l2, l3)
end

"""
    fm_bernstein_vals(l) -> Vector

The six degree-2 Bernstein basis values at barycentric coordinates `l`, in the
ordering `l1², l2², l3², 2·l2·l3, 2·l1·l3, 2·l1·l2`.
"""
fm_bernstein_vals(l) = [l[1]^2, l[2]^2, l[3]^2,
                        2*l[2]*l[3], 2*l[1]*l[3], 2*l[1]*l[2]]

"""
    fm_p2_eval(q::AbstractMatrix, d::AbstractVector, x, y)

Value at `(x, y)` of the P2 function on the element `q` with degree-2 Bernstein
coefficient vector `d`.
"""
fm_p2_eval(q::AbstractMatrix, d::AbstractVector, x, y) =
    dot(d, fm_bernstein_vals(fm_bary(q, x, y)))

"""
    fm_p2_grad(q::AbstractMatrix, d::AbstractVector, x, y) -> (px, py)

Gradient at `(x, y)` of the P2 function with Bernstein coefficients `d`.
"""
function fm_p2_grad(q::AbstractMatrix, d::AbstractVector, x, y)
    l = fm_bary(q, x, y)
    ar = ((q[2,1]-q[1,1])*(q[3,2]-q[1,2]) - (q[3,1]-q[1,1])*(q[2,2]-q[1,2])) / 2
    E = ((q[3,1]-q[2,1], q[3,2]-q[2,2]),
         (q[1,1]-q[3,1], q[1,2]-q[3,2]),
         (q[2,1]-q[1,1], q[2,2]-q[1,2]))
    g = ntuple(mm -> (-E[mm][2] / (2*ar), E[mm][1] / (2*ar)), 3)
    dx = 2*d[1]*l[1]*g[1][1] + 2*d[2]*l[2]*g[2][1] + 2*d[3]*l[3]*g[3][1] +
         2*d[4]*(l[2]*g[3][1] + l[3]*g[2][1]) +
         2*d[5]*(l[1]*g[3][1] + l[3]*g[1][1]) +
         2*d[6]*(l[1]*g[2][1] + l[2]*g[1][1])
    dy = 2*d[1]*l[1]*g[1][2] + 2*d[2]*l[2]*g[2][2] + 2*d[3]*l[3]*g[3][2] +
         2*d[4]*(l[2]*g[3][2] + l[3]*g[2][2]) +
         2*d[5]*(l[1]*g[3][2] + l[3]*g[1][2]) +
         2*d[6]*(l[1]*g[2][2] + l[2]*g[1][2])
    return (dx, dy)
end

"""
    fm_dof_values(q::AbstractMatrix, nrm::AbstractMatrix, d::AbstractVector;
                  ngauss::Integer = 6) -> Vector{Float64}

The six FM functionals `σ_a` applied to the P2 function with Bernstein
coefficients `d`, evaluated INDEPENDENTLY of [`fm_sigma`](@ref): vertex values
by direct evaluation, edge terms by `ngauss`-point Gauss–Legendre quadrature of
`(1/|e_m|) ∫_{e_m} grad v · n_m ds`. The quadrature weights are normalised to
sum to one, so the edge entries are averages, matching the DOF definition.

Used to test `σ_a(φ_i) = δ_ai`, and — evaluated from both sides of an interior
edge with the SAME normal — to measure the normal-derivative jump that separates
the two `convention` settings.
"""
function fm_dof_values(q::AbstractMatrix, nrm::AbstractMatrix,
                       d::AbstractVector; ngauss::Integer = 6)
    out = zeros(Float64, 6)
    for a in 1:3
        out[a] = fm_p2_eval(q, d, q[a, 1], q[a, 2])
    end
    # Gauss–Legendre nodes/weights on [0, 1] via the Golub–Welsch companion matrix
    kk = 1:(ngauss - 1)
    beta = kk ./ sqrt.(4 .* kk.^2 .- 1)
    F = eigen(SymTridiagonal(zeros(ngauss), collect(beta)))
    xs = (F.values .+ 1) ./ 2
    ws = F.vectors[1, :].^2                      # Σ ws = 1 on [0, 1]
    for mm in 1:3
        (ia, ib) = _fm_cyc2(mm)
        acc = 0.0
        for (xi, wi) in zip(xs, ws)
            px = q[ia, 1] + xi * (q[ib, 1] - q[ia, 1])
            py = q[ia, 2] + xi * (q[ib, 2] - q[ia, 2])
            gx, gy = fm_p2_grad(q, d, px, py)
            acc += wi * (gx * nrm[mm, 1] + gy * nrm[mm, 2])
        end
        out[3+mm] = acc
    end
    return out
end
