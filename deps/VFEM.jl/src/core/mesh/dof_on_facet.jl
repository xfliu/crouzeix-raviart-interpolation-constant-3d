# src/core/mesh/dof_on_facet.jl
#
# Port of VFEM3D/lib/mesh/get_dof_on_facet.m and get_common_dof_on_facets.m.
#
# DOFs are enumerated in the canonical `ijkl_list(M)` row order. A DOF
# (multi-index γ = (i, j, k, l)) lies on local facet f if and only if
# `γ[f] == 0`, since facet f is opposite vertex f.

"""
    dof_on_facet(M::Integer, FacetIdx::Integer) -> Vector{Int}

Return the 1-based DOF indices (into `ijkl_list(M)`) lying on local
facet `FacetIdx ∈ 1:4` of a tetrahedron with degree-`M` Bernstein
DOFs. The facet is opposite vertex `FacetIdx`, so a DOF γ lies on
the facet iff `γ[FacetIdx] == 0`. Output length is `simplex_dof(2, M)`.
"""
function dof_on_facet(M::Integer, FacetIdx::Integer)
    M ≥ 0 || throw(DomainError(M, "M must be ≥ 0"))
    1 ≤ FacetIdx ≤ 4 || throw(DomainError(FacetIdx, "FacetIdx must be in 1:4"))

    list = ijkl_list(M)
    out = Vector{Int}(undef, simplex_dof(2, M))
    cur = 1
    @inbounds for r in axes(list, 1)
        if list[r, FacetIdx] == 0
            out[cur] = r
            cur += 1
        end
    end
    return out
end

"""
    common_dof_on_facets(M::Integer, FacetIdx1::Integer, FacetIdx2::Integer)
        -> (dof_on_facet1::Vector{Int}, dof_on_facet2::Vector{Int})

Return positions, within `dof_on_facet(M, FacetIdx1)` and
`dof_on_facet(M, FacetIdx2)` respectively, of the DOFs common to
both facets (i.e. DOFs whose multi-index has `γ[f1] == γ[f2] == 0`).
Used by inter-element DOF matching for facet-shared DOFs.

The two output vectors have length `M + 1` (one DOF per index along
the shared edge).
"""
function common_dof_on_facets(M::Integer, FacetIdx1::Integer, FacetIdx2::Integer)
    M ≥ 0 || throw(DomainError(M, "M must be ≥ 0"))
    1 ≤ FacetIdx1 ≤ 4 || throw(DomainError(FacetIdx1, "FacetIdx1 must be in 1:4"))
    1 ≤ FacetIdx2 ≤ 4 || throw(DomainError(FacetIdx2, "FacetIdx2 must be in 1:4"))
    FacetIdx1 == FacetIdx2 &&
        throw(ArgumentError("FacetIdx1 and FacetIdx2 must differ"))

    list = ijkl_list(M)
    n = M + 1
    dof1 = Vector{Int}(undef, n)
    dof2 = Vector{Int}(undef, n)
    idx1 = 0; idx2 = 0
    cur = 0
    @inbounds for r in axes(list, 1)
        on1 = list[r, FacetIdx1] == 0
        on2 = list[r, FacetIdx2] == 0
        on1 && (idx1 += 1)
        on2 && (idx2 += 1)
        if on1 && on2
            cur += 1
            dof1[cur] = idx1
            dof2[cur] = idx2
        end
    end
    return dof1, dof2
end
