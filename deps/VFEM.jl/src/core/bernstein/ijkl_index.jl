# src/core/bernstein/ijkl_index.jl
#
# Lookup: given the canonical `ijkl_list(M)` matrix and a target
# multi-index, return the row index. Port of get_ijkl_idx.m.
#
# MATLAB's implementation packs (i,j,k) into `i + j·base + k·base²`
# (with `base = max(ijkl_list(:)) + 1 = M + 1`) and finds the matching
# row by linear scan. For a one-shot lookup that's fine. For repeated
# lookups (BernsteinProduct caches, singular potential assembly) we
# ALSO expose `ijkl_index_map(M)` returning a `Dict` so the lookup is
# O(1).
#
# Both forms produce the same answer for valid inputs.

"""
    ijkl_index(list::AbstractMatrix{<:Integer}, ijkl) -> Int

Return the 1-based row index of the multi-index `ijkl` in the
canonical `ijkl_list` matrix `list`. Throws `KeyError` if not found.
"""
function ijkl_index(list::AbstractMatrix{<:Integer},
                    ijkl::Union{NTuple{4, <:Integer}, AbstractVector{<:Integer}})
    n = size(list, 1)
    @inbounds for r in 1:n
        if list[r, 1] == ijkl[1] && list[r, 2] == ijkl[2] &&
           list[r, 3] == ijkl[3] && list[r, 4] == ijkl[4]
            return r
        end
    end
    throw(KeyError(ijkl))
end

"""
    ijkl_index_map(M::Integer) -> Dict{NTuple{4,Int}, Int}

Return a hashmap from multi-index `(i,j,k,l)` to its 1-based row
index in `ijkl_list(M)`. Use this when the same lookup is performed
many times — it amortises to O(1) per call, while a linear scan on
`ijkl_index` is O(n).
"""
function ijkl_index_map(M::Integer)
    list = ijkl_list(M)
    n = size(list, 1)
    map = Dict{NTuple{4, Int}, Int}()
    sizehint!(map, n)
    @inbounds for r in 1:n
        map[(list[r, 1], list[r, 2], list[r, 3], list[r, 4])] = r
    end
    return map
end
