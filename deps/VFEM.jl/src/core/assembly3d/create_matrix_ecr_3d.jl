# src/core/assembly3d/create_matrix_ecr_3d.jl
#
# Port of VFEM3D/create_matrix_ecr_3d.m.
#
# Enriched Crouzeix–Raviart element on tetrahedra:
#   * 4 face-midpoint DOFs (CR basis  φ_i = 1 − 3·L_i  on a 3-simplex)
#   * 1 cell-average DOF (enrichment via  q(x) = |x|²  in degree-2
#     Bernstein form)
#   * Total: 5 DOFs per element.
#
# All integrals are exact (Bernstein closed form):
#   ∫_K B_α^(2) B_β^(2) dV = C(2;α) C(2;β) · 6 |K| · (α+β)! / (|α|+|β|+3)!
# with multinomial factor `C(N;α) = N!/α!`.
#
# Global DOF layout:
#   1..NumF              : facet (face) DOFs in `m.FacetList` order
#   NumF+1..NumF+NumElt  : element (cell) DOFs in `m.ElementList` order
#
# Returns sparse `A`, `M`, plus an `EcrDof3D` info struct.

using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!

# Explicit 3×3 determinant (avoids `LinearAlgebra.det` falling back to
# `eigvals` for `Interval{Float64}` 3×3 matrices, which is wrong here).
@inline function _det3(M::AbstractMatrix{T}) where {T}
    return M[1, 1] * (M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]) -
           M[1, 2] * (M[2, 1] * M[3, 3] - M[2, 3] * M[3, 1]) +
           M[1, 3] * (M[2, 1] * M[3, 2] - M[2, 2] * M[3, 1])
end

# Explicit 3×3 inverse via cofactors (same reason — Julia's `inv` would
# go through `lu`, which interval-promoted matrices don't support).
function _inv3(M::AbstractMatrix{T}) where {T}
    d = _det3(M)
    inv = Matrix{T}(undef, 3, 3)
    inv[1, 1] = (M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]) / d
    inv[1, 2] = (M[1, 3] * M[3, 2] - M[1, 2] * M[3, 3]) / d
    inv[1, 3] = (M[1, 2] * M[2, 3] - M[1, 3] * M[2, 2]) / d
    inv[2, 1] = (M[2, 3] * M[3, 1] - M[2, 1] * M[3, 3]) / d
    inv[2, 2] = (M[1, 1] * M[3, 3] - M[1, 3] * M[3, 1]) / d
    inv[2, 3] = (M[1, 3] * M[2, 1] - M[1, 1] * M[2, 3]) / d
    inv[3, 1] = (M[2, 1] * M[3, 2] - M[2, 2] * M[3, 1]) / d
    inv[3, 2] = (M[1, 2] * M[3, 1] - M[1, 1] * M[3, 2]) / d
    inv[3, 3] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) / d
    return inv
end

"""
    EcrDof3D

DOF info for [`create_matrix_ecr_3d`](@ref):
* `ndof`      :: `Int`       — total DOF count (`NumF + NumElt`).
* `NumF`      :: `Int`       — number of facets.
* `NumElt`    :: `Int`       — number of elements.
* `face_dofs` :: `UnitRange` — `1:NumF`.
* `cell_dofs` :: `UnitRange` — `NumF+1:NumF+NumElt`.
"""
struct EcrDof3D
    ndof::Int
    NumF::Int
    NumElt::Int
    face_dofs::UnitRange{Int}
    cell_dofs::UnitRange{Int}
end

# Bernstein-form Gram matrix on the unit-volume reference tetrahedron:
# G[α, β] = ∫_{K_ref} B_α^(N) B_β^(N) dV / |K_ref| computed once per call.
# Multiply by |K_phys| at the call site for the physical Gram.
function _bernstein_gram3d_ref(N::Integer, ::Type{T}) where {T<:Real}
    α_list = ijkl_list(N)
    Cs = bernstein_multinomial_3d(N, α_list)
    n = size(α_list, 1)
    G = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        f = factorial(α_list[i, 1] + α_list[j, 1]) *
            factorial(α_list[i, 2] + α_list[j, 2]) *
            factorial(α_list[i, 3] + α_list[j, 3]) *
            factorial(α_list[i, 4] + α_list[j, 4])
        denom = factorial(2 * N + 3)
        G[i, j] = T(Cs[i] * Cs[j] * 6) * (T(f) / T(denom))
    end
    return G
end

# Differentiation matrix D_i mapping degree-N → degree-(N−1) Bernstein
# coefficients via  ∂/∂L_i [B_α^(N)] = N · B_{α−e_i}^{(N−1)}  (when α_i ≥ 1).
# `D_i[β, α] = N` if `β = α − e_i` and `α_i ≥ 1`, else 0.
function _bernstein_deriv_matrices3d(N::Integer, ::Type{T}) where {T<:Real}
    α_list = ijkl_list(N)
    β_list = ijkl_list(N - 1)
    β_idx  = ijkl_index_map(N - 1)
    n_α    = size(α_list, 1)
    n_β    = size(β_list, 1)
    D = ntuple(_ -> zeros(T, n_β, n_α), 4)
    @inbounds for α in 1:n_α
        a = (α_list[α, 1], α_list[α, 2], α_list[α, 3], α_list[α, 4])
        for i in 1:4
            ai = a[i]
            if ai ≥ 1
                key = (a[1] - (i == 1), a[2] - (i == 2),
                       a[3] - (i == 3), a[4] - (i == 4))
                β = β_idx[key]
                D[i][β, α] = T(N)
            end
        end
    end
    return D
end

# Build C_cr[m, i] (degree-2 Bernstein coefficients of CR basis function
# φ_i = 1 − 3 L_i, for i = 1..4). The constant 1 lifted to degree 2 has
# all-ones Bernstein coefficients; L_i lifted to degree 2 has coefficient
# `α_i / 2` at multi-index α.
function _ecr3d_cr_coeffs(::Type{T}) where {T<:Real}
    α_list = ijkl_list(2)
    n = size(α_list, 1)
    C = ones(T, n, 4)
    @inbounds for i in 1:4, m in 1:n
        C[m, i] -= T(3) * (T(α_list[m, i]) / T(2))
    end
    return C
end

# Bernstein-2 coefficients of q(x) = |x|² on element with vertices P (4×3).
# q = Σ_i |P_i|² L_i² + Σ_{i<j} (P_i·P_j) (2 L_i L_j) where the (P_i·P_j)
# enters at multi-index α with α_i = α_j = 1 and the Bernstein coefficient
# `c_q[α]` is exactly P_i·P_j (because B_{e_i+e_j}^(2) = 2 L_i L_j and
# we want q = Σ c_q[α] B_α^{(2)}, so the L_i² term has coefficient |P_i|²
# at α=2e_i since B_{2e_i}^{(2)} = L_i²). MATLAB lines 134–151.
function _ecr3d_q_coeffs(P::AbstractMatrix{T}) where {T<:Real}
    α_list = ijkl_list(2)
    α_idx  = ijkl_index_map(2)
    n = size(α_list, 1)
    cq = zeros(T, n)
    @inbounds for i in 1:4, j in i:4
        key = (Int(i == 1) + Int(j == 1), Int(i == 2) + Int(j == 2),
               Int(i == 3) + Int(j == 3), Int(i == 4) + Int(j == 4))
        row = α_idx[key]
        s = P[i, 1] * P[j, 1] + P[i, 2] * P[j, 2] + P[i, 3] * P[j, 3]
        cq[row] = s
    end
    return cq
end

# Face-average of q on each face (face i opposite vertex i): for terms
# with α_i > 0 the face integral vanishes; the remaining contribute
# C(2;α) · 2 · α!_face / (|α|_face + 2)! where α_face = α with i-entry dropped.
# Returns a 4-vector face-avg of q.
function _ecr3d_q_face_avgs(cq::AbstractVector{T}) where {T<:Real}
    α_list = ijkl_list(2)
    Cs = bernstein_multinomial_3d(2, α_list)
    n = size(α_list, 1)
    m_face = zeros(T, 4)
    @inbounds for fi in 1:4
        for m in 1:n
            α_list[m, fi] == 0 || continue
            face_idx = (α_list[m, 1], α_list[m, 2], α_list[m, 3], α_list[m, 4])
            # Drop i-th component; sum over remaining is |α| since α_i = 0.
            tot = face_idx[1] + face_idx[2] + face_idx[3] + face_idx[4]
            f = factorial(face_idx[1]) * factorial(face_idx[2]) *
                factorial(face_idx[3]) * factorial(face_idx[4])
            face_int = T(Cs[m] * 2) * (T(f) / T(factorial(tot + 2)))
            m_face[fi] += cq[m] * face_int
        end
    end
    return m_face
end

# Cell-average of q over reference unit-volume tetrahedron.
function _ecr3d_q_cell_avg(cq::AbstractVector{T}) where {T<:Real}
    α_list = ijkl_list(2)
    Cs = bernstein_multinomial_3d(2, α_list)
    n = size(α_list, 1)
    m_cell = zero(T)
    @inbounds for m in 1:n
        a = (α_list[m, 1], α_list[m, 2], α_list[m, 3], α_list[m, 4])
        f = factorial(a[1]) * factorial(a[2]) * factorial(a[3]) * factorial(a[4])
        m_cell += cq[m] * T(Cs[m] * 6) * (T(f) / T(factorial(2 + 3)))
    end
    return m_cell
end

"""
    create_matrix_ecr_3d(m::Mesh3D; T::Type = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T}, dof::EcrDof3D)

Assemble the global ECR stiffness `A = (∇φ, ∇ψ)` and mass `M = (φ, ψ)`
matrices for the 3D mesh `m`. DOFs: 4 face-midpoint per element + 1
cell DOF per element, layout `[face_1..face_NumF, cell_1..cell_NumElt]`.

The element type `T` controls precision: `Float64` for approximation,
`Interval{Float64}` for verified.
"""
function create_matrix_ecr_3d(m::Mesh3D; T::Type = Float64)
    NumF   = m.NumF
    NumElt = m.NumElt
    ndof   = NumF + NumElt

    G2 = _bernstein_gram3d_ref(2, T)
    G1 = _bernstein_gram3d_ref(1, T)
    D  = _bernstein_deriv_matrices3d(2, T)
    C_cr_ref = _ecr3d_cr_coeffs(T)

    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = m.NumElt * 25
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Aval = Vector{T}(undef, nent);   Mval = Vector{T}(undef, nent)
    pos = 0

    @inbounds for e in 1:NumElt
        v1 = m.ElementList[e, 1]; v2 = m.ElementList[e, 2]
        v3 = m.ElementList[e, 3]; v4 = m.ElementList[e, 4]
        P = T[m.NodeList[v1, 1] m.NodeList[v1, 2] m.NodeList[v1, 3];
              m.NodeList[v2, 1] m.NodeList[v2, 2] m.NodeList[v2, 3];
              m.NodeList[v3, 1] m.NodeList[v3, 2] m.NodeList[v3, 3];
              m.NodeList[v4, 1] m.NodeList[v4, 2] m.NodeList[v4, 3]]

        d1 = (P[2, 1] - P[1, 1], P[2, 2] - P[1, 2], P[2, 3] - P[1, 3])
        d2 = (P[3, 1] - P[1, 1], P[3, 2] - P[1, 2], P[3, 3] - P[1, 3])
        d3 = (P[4, 1] - P[1, 1], P[4, 2] - P[1, 2], P[4, 3] - P[1, 3])
        Jmat = T[d1[1] d1[2] d1[3];
                 d2[1] d2[2] d2[3];
                 d3[1] d3[2] d3[3]]
        vol = abs(_det3(Jmat)) / T(6)

        # Bernstein coeffs of q = |x|² on this element.
        cq     = _ecr3d_q_coeffs(P)
        m_face = _ecr3d_q_face_avgs(cq)
        m_cell = _ecr3d_q_cell_avg(cq)
        β      = m_cell - sum(m_face) / T(4)

        # ψ Bernstein coeffs: c_q − Σ_i m_face[i] · φ_i^CR, normalized by β.
        c_psi = (cq .- C_cr_ref * m_face) ./ β

        # ECR basis [φ_1..φ_4, ψ] in degree-2 Bernstein form.
        # Adjusted CR: φ_i^ECR = φ_i^CR − ψ / 4 to enforce zero-face-avg of ψ.
        C_ecr = Matrix{T}(undef, size(C_cr_ref, 1), 5)
        for j in 1:4
            for r in 1:size(C_cr_ref, 1)
                C_ecr[r, j] = C_cr_ref[r, j] - c_psi[r] / T(4)
            end
        end
        for r in 1:size(C_cr_ref, 1)
            C_ecr[r, 5] = c_psi[r]
        end

        M_local = vol .* (C_ecr' * G2 * C_ecr)

        # ∇L_i for the physical element: ∇L_2 = J⁻¹·e₁, ∇L_3 = J⁻¹·e₂,
        # ∇L_4 = J⁻¹·e₃, ∇L_1 = −(∇L_2+∇L_3+∇L_4). Use the fact that
        # `Jmat' * grad_L_phys[2..4]'` is the identity columns.
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

        # Gradient Bernstein-degree-1 coefficients of each ECR basis fn.
        n1 = size(G1, 1)
        Gx = zeros(T, n1, 5)
        Gy = zeros(T, n1, 5)
        Gz = zeros(T, n1, 5)
        for j in 1:5, i in 1:4
            Di_cj = D[i] * @view C_ecr[:, j]
            for r in 1:n1
                Gx[r, j] += Di_cj[r] * grad_L[i, 1]
                Gy[r, j] += Di_cj[r] * grad_L[i, 2]
                Gz[r, j] += Di_cj[r] * grad_L[i, 3]
            end
        end

        A_local = vol .* (Gx' * G1 * Gx + Gy' * G1 * Gy + Gz' * G1 * Gz)

        # Scatter to global. dof_local = [face_1..4, cell].
        dof_local = (m.Element2Facet[e, 1], m.Element2Facet[e, 2],
                     m.Element2Facet[e, 3], m.Element2Facet[e, 4],
                     NumF + e)
        for i in 1:5, j in 1:5
            pos += 1
            Irow[pos] = dof_local[i]; Jcol[pos] = dof_local[j]
            Aval[pos] = A_local[i, j]
            Mval[pos] = M_local[i, j]
        end
    end

    info = EcrDof3D(ndof, NumF, NumElt, 1:NumF, (NumF + 1):(NumF + NumElt))
    A = dropzeros!(sparse(Irow, Jcol, Aval, ndof, ndof))
    M = dropzeros!(sparse(Irow, Jcol, Mval, ndof, ndof))

    return A, M, info
end
