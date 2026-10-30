# src/core/assembly2d/create_matrix_lagrange.jl
#
# Port of vfem2d/lib/fem_assembly/create_matrix_lagrange.m.
#
# Conforming Lagrange (P1 or P2) assembly of
#   A(i, j) = ∫_Ω (∇φ_i · ∇φ_j + V φ_i φ_j) dx
#   M(i, j) = ∫_Ω φ_i φ_j dx
# using exact Bernstein integration of the potential V (represented
# as 15 per-element degree-4 control values from `elem_V_bernstein`).
#
# The Laplacian and pure-mass parts use closed-form Bernstein-2 Gram
# matrices identical to the ECR assembly. The potential mass uses a
# precomputed weight tensor `W_pot` of size 15 × nbasis² that
# encodes:
#   ∫_K V φ_i φ_j = 2|K| · Σ_m V_bern[k, m] · W_pot[m, (j-1)·nbasis+i]
#
# DOF ordering:
#   degree=1 (P1): vertex DOFs only, 1..nv.
#   degree=2 (P2): vertex DOFs 1..nv, then edge-midpoint DOFs nv+1..nv+ne
#                  (ordered by the row order of `m.edges`).

using SparseArrays: spzeros, sparse, SparseMatrixCSC, dropzeros!

# 6 × nbasis matrix of degree-2 Bernstein coefficients of each Lagrange
# basis function. Same as MATLAB `local_bernstein_coeff_matrix(degree)`.
function _lagrange_bernstein2_coeffs(degree::Integer, ::Type{T}) where {T<:Real}
    if degree == 1
        return T[1   0   0;
                 0   1   0;
                 0   0   1;
                 1//2 1//2 0;
                 0   1//2 1//2;
                 1//2 0   1//2]
    elseif degree == 2
        return T[ 1     0    0    0  0  0;
                  0     1    0    0  0  0;
                  0     0    1    0  0  0;
                 -1//2 -1//2 0    0  0  2;
                  0    -1//2 -1//2 2  0  0;
                 -1//2  0   -1//2 0  2  0]
    else
        throw(ArgumentError("Lagrange degree must be 1 or 2 (got $degree)"))
    end
end

# Each basis function as a list of monomial terms in barycentric coords.
# Returns a Matrix indexed by (i, j) of `Vector{NTuple{4, Int}}`-like data,
# but it's simpler to keep a 2D Vector-of-Vectors with rows [a, b, c, coeff].
function _lagrange_phi_monomials(degree::Integer)
    if degree == 1
        # phi_i = L_i: a single monomial e_i.
        # phi_i * phi_j = L_i * L_j: a single term with exponent vector e_i+e_j.
        out = Matrix{Vector{NTuple{4, Int}}}(undef, 3, 3)
        @inbounds for i in 1:3, j in 1:3
            ev = [0, 0, 0]; ev[i] += 1; ev[j] += 1
            out[i, j] = [(ev[1], ev[2], ev[3], 1)]
        end
        return out
    elseif degree == 2
        # phi_1 = 2 L1² - L1, phi_2 = 2 L2² - L2, phi_3 = 2 L3² - L3,
        # phi_4 = 4 L2 L3, phi_5 = 4 L1 L3, phi_6 = 4 L1 L2.
        basis = (
            [(2, 0, 0,  2), (1, 0, 0, -1)],
            [(0, 2, 0,  2), (0, 1, 0, -1)],
            [(0, 0, 2,  2), (0, 0, 1, -1)],
            [(0, 1, 1,  4)],
            [(1, 0, 1,  4)],
            [(1, 1, 0,  4)],
        )
        out = Matrix{Vector{NTuple{4, Int}}}(undef, 6, 6)
        @inbounds for i in 1:6, j in 1:6
            terms = NTuple{4, Int}[]
            for p in basis[i], q in basis[j]
                push!(terms, (p[1] + q[1], p[2] + q[2], p[3] + q[3],
                              p[4] * q[4]))
            end
            # Combine like terms.
            d = Dict{NTuple{3, Int}, Int}()
            for t in terms
                key = (t[1], t[2], t[3])
                d[key] = get(d, key, 0) + t[4]
            end
            out[i, j] = [(k[1], k[2], k[3], v) for (k, v) in d if v != 0]
        end
        return out
    else
        throw(ArgumentError("Lagrange degree must be 1 or 2"))
    end
end

# Precompute the 15 × nbasis² weight tensor that encodes the potential
# integration. Indexing convention (column-major flat):
#   flat_ij = (j - 1) * nbasis + i
# so that `A_pot(i, j) = 2|K| · Σ_m V_bern[k, m] · W_pot[m, flat_ij]`.
function _precompute_W_pot(degree::Integer, ::Type{T}) where {T<:Real}
    bern4 = bernstein4_multiindices_2d()
    nbasis = degree == 1 ? 3 : 6
    phi_mono = _lagrange_phi_monomials(degree)
    fac = T[1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]   # 0!..10!

    # binomial-4 weights for each Bernstein-4 control value.
    binom4 = Vector{T}(undef, 15)
    @inbounds for m in 1:15
        a = bern4[m, 1]; b = bern4[m, 2]; c = bern4[m, 3]
        binom4[m] = fac[5] / (fac[a + 1] * fac[b + 1] * fac[c + 1])    # 4!/(a! b! c!)
    end

    W = zeros(T, 15, nbasis * nbasis)
    @inbounds for m in 1:15
        a_m = bern4[m, 1]; b_m = bern4[m, 2]; c_m = bern4[m, 3]
        bc = binom4[m]
        for j in 1:nbasis, i in 1:nbasis
            flat_ij = (j - 1) * nbasis + i
            wval = zero(T)
            for term in phi_mono[i, j]
                p, q, r, coeff = term
                a_t = a_m + p; b_t = b_m + q; c_t = c_m + r
                tot = a_t + b_t + c_t
                num = fac[a_t + 1] * fac[b_t + 1] * fac[c_t + 1]
                den = fac[tot + 3]                                  # = (tot+2)!
                wval += bc * T(coeff) * num / den
            end
            W[m, flat_ij] = wval
        end
    end
    return W
end

"""
    create_matrix_lagrange(m::Mesh2D, degree::Integer, V_bern::AbstractMatrix;
                           T::Type = Float64)
        -> (A::SparseMatrixCSC{T}, M::SparseMatrixCSC{T})

Conforming Lagrange (P1 if `degree == 1`, P2 if `degree == 2`)
assembly of `A = (∇φ, ∇ψ) + (V φ, ψ)` and `M = (φ, ψ)`. `V_bern`
is the `nt × 15` per-element degree-4 Bernstein control values of
the potential, typically obtained from `elem_V_bernstein`.

DOF order:
* P1: vertex DOFs 1..`nv`.
* P2: vertex DOFs 1..`nv`, then edge-midpoint DOFs `nv+1..nv+ne`
  in the row order of `m.edges`.
"""
function create_matrix_lagrange(m::Mesh2D, degree::Integer,
                                V_bern::AbstractMatrix;
                                T::Type = Float64)
    (degree == 1 || degree == 2) ||
        throw(ArgumentError("Lagrange degree must be 1 or 2"))
    nbasis = degree == 1 ? 3 : 6
    ndof = degree == 1 ? m.nv : m.nv + m.ne
    size(V_bern) == (m.nt, 15) ||
        throw(DimensionMismatch("V_bern must be nt×15 (got $(size(V_bern)))"))

    C_loc   = _lagrange_bernstein2_coeffs(degree, T)
    G2_core = T.(_ECR_G2_NUM) ./ T(90)
    G1_core = T.(_ECR_G1_NUM) ./ T(12)
    W_pot   = _precompute_W_pot(degree, T)

    # COO triplets + one `sparse` call rather than scatter-adds into a CSC:
    # `X[i,j] +=` on a sparse matrix inserts when the entry is new (an O(nnz)
    # memmove), which makes element-by-element assembly quadratic in size.
    nent = m.nt * nbasis * nbasis
    Irow = Vector{Int}(undef, nent); Jcol = Vector{Int}(undef, nent)
    Aval = Vector{T}(undef, nent);   Mval = Vector{T}(undef, nent)
    pos = 0

    @inbounds for k in 1:m.nt
        t = (m.elements[k, 1], m.elements[k, 2], m.elements[k, 3])
        P = T[m.nodes[t[1], 1] m.nodes[t[1], 2];
              m.nodes[t[2], 1] m.nodes[t[2], 2];
              m.nodes[t[3], 1] m.nodes[t[3], 2]]
        areaK = abs(T(1) / T(2) * ((P[2, 1] - P[1, 1]) * (P[3, 2] - P[1, 2])
                                  - (P[3, 1] - P[1, 1]) * (P[2, 2] - P[1, 2])))
        grad_λ = _local_grad_lambda(P, areaK)

        # Laplacian Gram via Bernstein gradient closed form.
        Gx = Matrix{T}(undef, 3, nbasis)
        Gy = Matrix{T}(undef, 3, nbasis)
        for j in 1:nbasis
            gc = _bernstein_quadratic_grad_coeffs(C_loc[:, j], grad_λ)
            Gx[:, j] = gc[:, 1]
            Gy[:, j] = gc[:, 2]
        end
        A_lap = areaK .* (Gx' * G1_core * Gx + Gy' * G1_core * Gy)
        M_loc = areaK .* (C_loc' * G2_core * C_loc)

        # Potential part: A_pot[i, j] = 2|K| · Σ_m V_bern[k, m] · W_pot[m, (j-1)·nbasis+i].
        v_row = T.(@view V_bern[k, :])
        flat = v_row' * W_pot
        # `flat` is 1×nbasis², reshape (column-major) into (nbasis, nbasis).
        A_pot_vec = reshape(flat, nbasis, nbasis)
        A_pot = T(2) * areaK .* A_pot_vec
        A_local = A_lap .+ A_pot

        # Global DOF list: degree=1 → vertices; degree=2 → vertices + edge mids.
        if degree == 1
            g_dofs = (t[1], t[2], t[3])
        else
            g_dofs = (t[1], t[2], t[3],
                      m.nv + m.tri2edge[k, 1],
                      m.nv + m.tri2edge[k, 2],
                      m.nv + m.tri2edge[k, 3])
        end

        for i in 1:nbasis, j in 1:nbasis
            pos += 1
            Irow[pos] = g_dofs[i]; Jcol[pos] = g_dofs[j]
            Aval[pos] = A_local[i, j]
            Mval[pos] = M_loc[i, j]
        end
    end
    A = dropzeros!(sparse(Irow, Jcol, Aval, ndof, ndof))
    M = dropzeros!(sparse(Irow, Jcol, Mval, ndof, ndof))

    return A, M
end
