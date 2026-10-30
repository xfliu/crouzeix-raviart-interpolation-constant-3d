# src/core/bernstein/ijkl_list.jl
#
# Enumerate all tetrahedral barycentric multi-indices (i,j,k,l) with
# i+j+k+l = M. Port of VFEM3D/lib/quadrature/get_IJKL.m. The MATLAB
# routine returns a (DOF × 4) matrix; we return the same as a
# `Matrix{Int}` so existing index lookups carry over directly.
#
# CRITICAL: the row order matches MATLAB exactly (outer i descending,
# then j descending, then k descending, l fixed by sum constraint).
# Downstream caches in `pre_ijkl_list_*.txt` and any persisted indices
# depend on this ordering — do not change without an explicit migration.

"""
    ijkl_list(M::Integer) -> Matrix{Int}

Return the `(simplex_dof(3, M), 4)` matrix whose rows enumerate every
tetrahedral barycentric multi-index `(i,j,k,l)` with `i+j+k+l = M`,
in the lex-descending order

    for i = M:-1:0, j = M-i:-1:0, k = M-i-j:-1:0
        emit (i, j, k, M-i-j-k)
    end

This row order is the canonical MATLAB ordering used everywhere in
VFEM3D — it is the contract that `ijkl_index_map` and the cached
index files in `lib/quadrature/*.txt` rely on.
"""
function ijkl_list(M::Integer)
    M ≥ 0 || throw(DomainError(M, "M must be ≥ 0"))
    n = simplex_dof(3, M)
    out = Matrix{Int}(undef, n, 4)
    idx = 1
    @inbounds for i in M:-1:0
        for j in (M - i):-1:0
            for k in (M - i - j):-1:0
                out[idx, 1] = i
                out[idx, 2] = j
                out[idx, 3] = k
                out[idx, 4] = M - i - j - k
                idx += 1
            end
        end
    end
    return out
end
