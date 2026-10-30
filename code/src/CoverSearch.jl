module CoverSearch

using IntervalArithmetic: interval, inf, mid, sup
using LinearAlgebra: Symmetric, eigvals

using ..BoxCover

export BASIS_PERMUTATIONS, ReferenceOrigin, CandidateChoice,
       SearchCell, AdaptiveSearch, SearchProgress, CoverRecord,
       expanded_reference_catalog, select_reference,
       start_search, advance_search!, search_complete,
       validate_search_state, materialize_cover, replay_record,
       save_reference_catalog, load_reference_catalog,
       save_cover_record, load_cover_record,
       save_checkpoint, load_checkpoint

"""
The six relabellings of the three edge vectors based at one tetrahedron
vertex.  Congruently permuting a reference Gram matrix does not change the
reference tetrahedron or its certified constant; it only changes the basis in
which the comparison with a cover cell is made.
"""
const BASIS_PERMUTATIONS = ((1, 2, 3), (1, 3, 2), (2, 1, 3),
                            (2, 3, 1), (3, 1, 2), (3, 2, 1))

struct ReferenceOrigin
    base_id::String
    permutation::NTuple{3, Int}
end

struct CandidateChoice
    reference_id::String
    origin::ReferenceOrigin
    bound2_estimate::Float64
    margin::Float64
    result::LeafTestResult
end

struct SearchCell
    path::String
    box::DyadicBox
    depth::Int
end

"""
Mutable, resumable state for a deterministic depth-first adaptive search.

`references` is the expanded reference catalog used by independent replay.
`origins` records which input certificate and edge-basis permutation produced
each expanded reference.  A path is `.` at the root and dot-separated,
zero-based child indices below it.
"""
mutable struct AdaptiveSearch
    family::Symbol
    root::DyadicBox
    target::Rational{BigInt}
    references::Dict{String, ReferenceBound}
    origins::Dict{String, ReferenceOrigin}
    pending::Vector{SearchCell}
    splits::Dict{String, Tuple{Vararg{Int}}}
    leaves::Dict{String, CoverLeaf}
    processed::Int
end

struct SearchProgress
    processed_now::Int
    processed_total::Int
    pending::Int
    splits::Int
    retained::Int
    excluded::Int
    depth_limited::Int
    complete::Bool
end

struct CoverRecord
    family::Symbol
    root::DyadicBox
    target::Rational{BigInt}
    tree::AbstractCoverNode
end

const _REFERENCE_MAGIC = "KAPPA3_REFERENCE_CATALOG_V1"
const _COVER_MAGIC = "KAPPA3_COVER_RECORD_V1"
const _CHECKPOINT_MAGIC = "KAPPA3_SEARCH_CHECKPOINT_V1"

# ---------------------------------------------------------------------------
# Reference variants and deterministic candidate selection
# ---------------------------------------------------------------------------

_perm_tag(p::NTuple{3, Int}) = string(p[1], p[2], p[3])
_variant_id(id::AbstractString, p::NTuple{3, Int}) =
    string(id, "@basis-", _perm_tag(p))

function _input_references(references)
    refs = references isa AbstractDict ? collect(values(references)) : collect(references)
    all(r -> r isa ReferenceBound, refs) ||
        throw(ArgumentError("the reference catalog must contain ReferenceBound values"))
    sort!(refs; by = r -> r.id)
    ids = [r.id for r in refs]
    length(unique(ids)) == length(ids) ||
        throw(ArgumentError("reference ids must be unique"))
    if references isa AbstractDict
        for (key, ref) in references
            String(key) == ref.id ||
                throw(ArgumentError("reference dictionary key '$key' disagrees with id '$(ref.id)'"))
        end
    end
    return ReferenceBound[refs...]
end

"""
    expanded_reference_catalog(references; permutations=true)

Build the replay catalog searched by the adaptive algorithm.  By default all
six congruent edge-basis permutations are included.  The return value is
`(bounds, origins)`, with both dictionaries keyed by the expanded id.
"""
function expanded_reference_catalog(references; permutations::Bool = true)
    bounds = Dict{String, ReferenceBound}()
    origins = Dict{String, ReferenceOrigin}()
    perms = permutations ? BASIS_PERMUTATIONS : ((1, 2, 3),)
    for ref in _input_references(references)
        for p in perms
            id = permutations ? _variant_id(ref.id, p) : ref.id
            haskey(bounds, id) && throw(ArgumentError("expanded reference id collision: '$id'"))
            inds = collect(p)
            gram = ref.gram[inds, inds]
            bounds[id] = ReferenceBound(id, ref.family, gram, ref.u2_upper)
            origins[id] = ReferenceOrigin(ref.id, p)
        end
    end
    _validate_reference_catalog(bounds, origins)
    return bounds, origins
end

function _same_interval_matrix(A, B)
    size(A) == size(B) || return false
    return all(isequal(inf(A[i, j]), inf(B[i, j])) &&
               isequal(sup(A[i, j]), sup(B[i, j]))
               for i in axes(A, 1), j in axes(A, 2))
end

"""
Check that every serialized variant is exactly a congruent permutation of the
identity variant carrying the original point bound.  Thus the human-readable
catalogue does not ask replay to trust its `origin` metadata.
"""
function _validate_reference_catalog(references, origins)
    Set(keys(references)) == Set(keys(origins)) ||
        throw(ArgumentError("reference origins and bounds have different keys"))
    base_ids = sort!(unique(origin.base_id for origin in values(origins)))
    for base_id in base_ids
        group = sort!([id for (id, origin) in origins if origin.base_id == base_id])
        identity_ids = [id for id in group
                        if origins[id].permutation == (1, 2, 3)]
        isempty(identity_ids) &&
            throw(ArgumentError("reference group '$base_id' has no identity representation"))
        base_ref = references[first(identity_ids)]
        for id in group
            ref = references[id]
            origin = origins[id]
            expected_id = id == base_id ? base_id :
                          _variant_id(base_id, origin.permutation)
            id == expected_id ||
                throw(ArgumentError("reference '$id' disagrees with its origin metadata"))
            ref.family == base_ref.family ||
                throw(ArgumentError("reference group '$base_id' mixes families"))
            isequal(ref.u2_upper, base_ref.u2_upper) ||
                throw(ArgumentError("reference group '$base_id' mixes point bounds"))
            inds = collect(origin.permutation)
            _same_interval_matrix(ref.gram, base_ref.gram[inds, inds]) ||
                throw(ArgumentError("reference '$id' is not the recorded Gram permutation"))
        end
    end
    return nothing
end

function _candidate_margin(result::LeafTestResult)
    result.vacuous && return Inf
    isempty(result.corners) && return -Inf
    return minimum(minimum(c.pd.leading_minor_lower) for c in result.corners)
end

function _heuristic_bound2(family::Symbol, box::CoefficientBox,
                           ref::ReferenceBound)
    G0 = Symmetric(mid.(ref.gram))
    q = -Inf
    for corner in coefficient_corners(box)
        G = family == :star ? star_gram(corner) : path_vertex_gram(corner)
        q = max(q, maximum(eigvals(Symmetric(mid.(G)), G0)))
    end
    return max(0.0, q) * ref.u2_upper
end

"""
Test every same-family reference variant.  Among passing candidates, choose
the smallest midpoint estimate of the transported squared bound, breaking
ties by the largest rigorous interval Sylvester-minor margin and then by
expanded reference id.  This ranking is only a floating-point search
heuristic; every chosen leaf is recomputed by `BoxCover.replay_cover`.
"""
function select_reference(family::Symbol, box::DyadicBox,
                          references::AbstractDict{String, <:ReferenceBound},
                          origins::AbstractDict{String, ReferenceOrigin};
                          target::Rational = TARGET_KAPPA)
    family in (:star, :path) || throw(ArgumentError("family must be :star or :path"))
    coeff = family == :star ? star_coefficient_enclosure(box) :
                              path_coefficient_enclosure(box)
    choices = CandidateChoice[]
    for id in sort!(collect(keys(references)))
        ref = references[id]
        ref.family == family || continue
        haskey(origins, id) || throw(ArgumentError("missing origin for reference '$id'"))
        # The angular-to-coefficient enclosure is independent of the
        # reference.  Construct it once even for catalogues with hundreds of
        # point certificates.
        result = isnothing(coeff) ?
            certify_leaf(family, box, ref.gram, ref.u2_upper; target) :
            certify_coefficient_leaf(family, coeff, ref.gram,
                                     ref.u2_upper; target)
        result.passed || continue
        estimate = isnothing(coeff) ? -Inf :
                   _heuristic_bound2(family, coeff, ref)
        push!(choices, CandidateChoice(id, origins[id], estimate,
                                       _candidate_margin(result), result))
    end
    isempty(choices) && return nothing
    # Sorting avoids dependence on dictionary iteration.  `isless` on finite
    # Float64 values is deterministic; margins are computed only from outward
    # interval endpoints.
    sort!(choices; by = c -> (c.bound2_estimate, -c.margin, c.reference_id))
    return first(choices)
end

# ---------------------------------------------------------------------------
# Adaptive search
# ---------------------------------------------------------------------------

function _default_root(family::Symbol)
    family == :star && return star_root_box()
    family == :path && return path_root_box()
    throw(ArgumentError("family must be :star or :path"))
end

function start_search(family::Symbol, references;
                      root::DyadicBox = _default_root(family),
                      permutations::Bool = true,
                      target::Rational = TARGET_KAPPA)
    family in (:star, :path) || throw(ArgumentError("family must be :star or :path"))
    bounds, origins = expanded_reference_catalog(references; permutations)
    exact_target = BigInt(numerator(target)) // BigInt(denominator(target))
    exact_target > 0 || throw(DomainError(target, "target must be positive"))
    return AdaptiveSearch(family, root, exact_target, bounds, origins,
                          SearchCell[SearchCell(".", root, 0)],
                          Dict{String, Tuple{Vararg{Int}}}(),
                          Dict{String, CoverLeaf}(), 0)
end

_child_path(path::String, child::Int) = path == "." ? string(child) : string(path, '.', child)

function _width_as_rational(b::DyadicInterval)
    w = b.hi - b.lo
    return w.numerator // (BigInt(1) << w.exponent)
end

"""Choose the path split axis by exact normalized width, with low-axis ties."""
function _path_split_axis(box::DyadicBox)
    maxima = (60, 60, 90)
    best_axis = 0
    best_width = BigInt(-1) // BigInt(1)
    for axis in 1:3
        width = _width_as_rational(box.bounds[axis]) / maxima[axis]
        if width > best_width
            best_axis = axis
            best_width = width
        end
    end
    best_width > 0 || throw(ArgumentError("cannot split a zero-width path box"))
    return best_axis
end

function _split_axes(family::Symbol, box::DyadicBox)
    if family == :star
        all(_width_as_rational(b) > 0 for b in box.bounds) ||
            throw(ArgumentError("cannot octree-split a star box with a zero-width axis"))
        return (1, 2, 3)
    end
    return (_path_split_axis(box),)
end

_excluded(family::Symbol, box::DyadicBox) =
    family == :star ? star_box_excluded(box) : path_box_excluded(box)

search_complete(state::AdaptiveSearch) = isempty(state.pending)

"""
Advance a search by at most `max_new_nodes` non-depth-limited cells.  Star
cells use an eight-child octree split.  Path cells bisect the coordinate of
largest exact normalized angular width.  Children are pushed in reverse mask
order so deterministic depth-first traversal visits child zero first.

Cells at `max_depth` remain pending, allowing a later call (or a loaded
checkpoint) to continue with a larger depth.  If `checkpoint_path` is given,
an atomic checkpoint is written after each `checkpoint_every` processed cells
and once more before returning.
"""
function advance_search!(state::AdaptiveSearch;
                         max_new_nodes::Int = typemax(Int),
                         max_depth::Int = typemax(Int),
                         checkpoint_path::Union{Nothing, AbstractString} = nothing,
                         checkpoint_every::Int = 1000)
    max_new_nodes >= 0 || throw(DomainError(max_new_nodes, "max_new_nodes must be nonnegative"))
    max_depth >= 0 || throw(DomainError(max_depth, "max_depth must be nonnegative"))
    checkpoint_every > 0 || throw(DomainError(checkpoint_every, "checkpoint_every must be positive"))
    processed_now = 0
    deferred = SearchCell[]
    while !isempty(state.pending) && processed_now < max_new_nodes
        cell = pop!(state.pending)
        if cell.depth >= max_depth
            push!(deferred, cell)
            continue
        end

        if _excluded(state.family, cell.box)
            state.leaves[cell.path] = CoverLeaf(cell.box, :excluded)
        else
            choice = select_reference(state.family, cell.box,
                                      state.references, state.origins;
                                      target = state.target)
            if !isnothing(choice)
                state.leaves[cell.path] =
                    CoverLeaf(cell.box, :retained, choice.reference_id)
            else
                axes = _split_axes(state.family, cell.box)
                child_boxes = split_boxes(cell.box, axes)
                state.splits[cell.path] = axes
                for child in reverse(eachindex(child_boxes))
                    child_index = child - 1
                    push!(state.pending,
                          SearchCell(_child_path(cell.path, child_index),
                                     child_boxes[child], cell.depth + 1))
                end
            end
        end
        processed_now += 1
        state.processed += 1
        if !isnothing(checkpoint_path) && state.processed % checkpoint_every == 0
            # Preserve deferred cells in the same stack order in the on-disk
            # state without perturbing the live traversal.
            saved_pending = state.pending
            state.pending = vcat(saved_pending, reverse(deferred))
            try
                save_checkpoint(checkpoint_path, state)
            finally
                state.pending = saved_pending
            end
        end
    end

    # Deferred cells occupied the top of the old stack in pop order.  Reverse
    # them and append them so the next pop reproduces the old DFS order.
    state.pending = vcat(state.pending, reverse(deferred))
    !isnothing(checkpoint_path) && save_checkpoint(checkpoint_path, state)
    retained = count(leaf -> leaf.status == :retained, values(state.leaves))
    excluded = length(state.leaves) - retained
    return SearchProgress(processed_now, state.processed, length(state.pending),
                          length(state.splits), retained, excluded,
                          length(deferred), search_complete(state))
end

function _path_components(path::String)
    path == "." && return Int[]
    isempty(path) && throw(ArgumentError("an empty cell path is invalid"))
    parts = split(path, '.')
    all(p -> occursin(r"^[0-9]+$", p), parts) ||
        throw(ArgumentError("invalid cell path '$path'"))
    return parse.(Int, parts)
end

_path_sort_key(path::String) = Tuple(_path_components(path))

function _expected_box(state::AdaptiveSearch, path::String)
    box = state.root
    prefix = "."
    for child_index in _path_components(path)
        haskey(state.splits, prefix) ||
            throw(ArgumentError("cell '$path' has no recorded split at ancestor '$prefix'"))
        children = split_boxes(box, state.splits[prefix])
        0 <= child_index < length(children) ||
            throw(ArgumentError("child index $child_index is invalid below '$prefix'"))
        box = children[child_index + 1]
        prefix = _child_path(prefix, child_index)
    end
    return box
end

"""Validate checkpoint topology, exact cell boxes, and reference links."""
function validate_search_state(state::AdaptiveSearch)
    state.family in (:star, :path) || throw(ArgumentError("unknown search family $(state.family)"))
    state.target > 0 || throw(ArgumentError("search target must be positive"))
    leaf_paths = Set(keys(state.leaves))
    split_paths = Set(keys(state.splits))
    pending_paths = Set(cell.path for cell in state.pending)
    length(pending_paths) == length(state.pending) ||
        throw(ArgumentError("checkpoint contains duplicate pending cells"))
    isempty(intersect(leaf_paths, split_paths)) ||
        throw(ArgumentError("a path is recorded as both a leaf and a split"))
    isempty(intersect(leaf_paths, pending_paths)) ||
        throw(ArgumentError("a path is recorded as both a leaf and pending"))
    isempty(intersect(split_paths, pending_paths)) ||
        throw(ArgumentError("a path is recorded as both a split and pending"))

    all_paths = union(leaf_paths, split_paths, pending_paths)
    "." in all_paths || throw(ArgumentError("checkpoint does not contain the root cell"))
    for (path, axes) in state.splits
        box = _expected_box(state, path)
        expected_axes = _split_axes(state.family, box)
        axes == expected_axes ||
            throw(ArgumentError("split axes at '$path' do not match the deterministic policy"))
        for child in 0:(2^length(axes) - 1)
            child_path = _child_path(path, child)
            child_path in all_paths ||
                throw(ArgumentError("split '$path' is missing direct child $child"))
        end
    end
    for (path, leaf) in state.leaves
        leaf.box == _expected_box(state, path) ||
            throw(ArgumentError("leaf '$path' has the wrong exact box"))
        if leaf.status == :retained
            haskey(state.references, leaf.reference_id::String) ||
                throw(ArgumentError("leaf '$path' names an absent reference"))
        elseif !_excluded(state.family, leaf.box)
            throw(ArgumentError("excluded leaf '$path' has no rigorous infeasibility proof"))
        end
    end
    for cell in state.pending
        cell.box == _expected_box(state, cell.path) ||
            throw(ArgumentError("pending cell '$(cell.path)' has the wrong exact box"))
        cell.depth == length(_path_components(cell.path)) ||
            throw(ArgumentError("pending cell '$(cell.path)' has the wrong depth"))
    end
    for (id, ref) in state.references
        id == ref.id || throw(ArgumentError("reference key '$id' disagrees with its id"))
        haskey(state.origins, id) || throw(ArgumentError("reference '$id' has no origin"))
    end
    Set(keys(state.origins)) == Set(keys(state.references)) ||
        throw(ArgumentError("reference origins and bounds have different keys"))
    _validate_reference_catalog(state.references, state.origins)
    return state
end

function _materialize(state::AdaptiveSearch, path::String, box::DyadicBox)
    if haskey(state.leaves, path)
        leaf = state.leaves[path]
        leaf.box == box || error("internal leaf box mismatch at '$path'")
        return leaf
    end
    haskey(state.splits, path) ||
        throw(ArgumentError("search is incomplete at cell '$path'"))
    axes = state.splits[path]
    boxes = split_boxes(box, axes)
    children = AbstractCoverNode[
        _materialize(state, _child_path(path, child - 1), boxes[child])
        for child in eachindex(boxes)
    ]
    return CoverSplit(box, axes, children)
end

"""Materialize and audit the final replay tree; incomplete searches throw."""
function materialize_cover(state::AdaptiveSearch)
    validate_search_state(state)
    search_complete(state) ||
        throw(ArgumentError("search has $(length(state.pending)) pending cells"))
    tree = _materialize(state, ".", state.root)
    validate_tree(tree, state.root; family = state.family)
    return tree
end

"""Run the independent rigorous replay encoded by a loaded cover record."""
replay_record(record::CoverRecord,
              references::AbstractDict{String, <:ReferenceBound}) =
    replay_cover(record.tree, record.root, references;
                 family = record.family, target = record.target)

# ---------------------------------------------------------------------------
# Stable text encoding
# ---------------------------------------------------------------------------

_dyadic_text(x::Dyadic) = string(x.numerator, "/2^", x.exponent)

function _parse_dyadic(text::AbstractString)
    m = match(r"^(-?[0-9]+)/2\^([0-9]+)$", text)
    isnothing(m) && throw(ArgumentError("invalid dyadic '$text'"))
    return Dyadic(parse(BigInt, m.captures[1]), parse(Int, m.captures[2]))
end

function _box_text(box::DyadicBox)
    return join((string(_dyadic_text(b.lo), "..", _dyadic_text(b.hi))
                 for b in box.bounds), ',')
end

function _parse_box(text::AbstractString)
    coords = split(text, ',')
    length(coords) == 3 || throw(ArgumentError("invalid box '$text'"))
    bounds = map(coords) do coord
        ends = split(coord, "..")
        length(ends) == 2 || throw(ArgumentError("invalid interval '$coord'"))
        DyadicInterval(_parse_dyadic(ends[1]), _parse_dyadic(ends[2]))
    end
    return DyadicBox((bounds[1], bounds[2], bounds[3]))
end

_float_text(x::Float64) = string("0x", string(reinterpret(UInt64, x); base = 16, pad = 16))

function _parse_float(text::AbstractString)
    startswith(text, "0x") || throw(ArgumentError("invalid Float64 encoding '$text'"))
    length(text) == 18 || throw(ArgumentError("invalid Float64 encoding '$text'"))
    return reinterpret(Float64, parse(UInt64, text[3:end]; base = 16))
end

function _escape_id(text::AbstractString)
    io = IOBuffer()
    for byte in codeunits(String(text))
        if (UInt8('a') <= byte <= UInt8('z')) ||
           (UInt8('A') <= byte <= UInt8('Z')) ||
           (UInt8('0') <= byte <= UInt8('9')) || byte in codeunits("_.:@+=-")
            write(io, byte)
        else
            print(io, '%', uppercase(string(byte; base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

function _unescape_id(text::AbstractString)
    bytes = UInt8[]
    i = firstindex(text)
    while i <= lastindex(text)
        if text[i] == '%'
            i1 = nextind(text, i)
            i2 = nextind(text, i1)
            i2 <= lastindex(text) || throw(ArgumentError("truncated id escape '$text'"))
            push!(bytes, parse(UInt8, String(text[i1:i2]); base = 16))
            i = nextind(text, i2)
        else
            c = text[i]
            isascii(c) || throw(ArgumentError("unescaped non-ASCII byte in id '$text'"))
            push!(bytes, UInt8(c))
            i = nextind(text, i)
        end
    end
    return String(bytes)
end

_axes_text(axes) = join(axes, ',')

_rational_text(x::Rational) = string(numerator(x), '/', denominator(x))

function _parse_rational(text::AbstractString)
    fields = split(text, '/')
    length(fields) == 2 || throw(ArgumentError("invalid rational '$text'"))
    den = parse(BigInt, fields[2])
    den > 0 || throw(ArgumentError("a rational denominator must be positive"))
    return parse(BigInt, fields[1]) // den
end

function _parse_axes(text::AbstractString)
    isempty(text) && throw(ArgumentError("empty split-axis record"))
    return Tuple(parse.(Int, split(text, ',')))
end

function _reference_fields(ref::ReferenceBound, origin::ReferenceOrigin)
    endpoints = String[]
    for j in 1:3, i in 1:3
        push!(endpoints, string(_float_text(inf(ref.gram[i, j])), ':',
                                _float_text(sup(ref.gram[i, j]))))
    end
    return ("R", _escape_id(ref.id), String(ref.family),
            _float_text(ref.u2_upper), _escape_id(origin.base_id),
            _perm_tag(origin.permutation), join(endpoints, ','))
end

function _parse_reference(fields)
    length(fields) == 7 && fields[1] == "R" ||
        throw(ArgumentError("malformed reference record"))
    id = _unescape_id(fields[2])
    family = Symbol(fields[3])
    u2 = _parse_float(fields[4])
    base_id = _unescape_id(fields[5])
    ptext = fields[6]
    length(ptext) == 3 && all(isdigit, ptext) ||
        throw(ArgumentError("invalid reference permutation '$ptext'"))
    p = Tuple(parse(Int, string(c)) for c in ptext)
    p in BASIS_PERMUTATIONS || throw(ArgumentError("invalid reference permutation '$ptext'"))
    cells = split(fields[7], ',')
    length(cells) == 9 || throw(ArgumentError("a reference Gram matrix needs nine entries"))
    gram = Matrix{typeof(interval(1.0))}(undef, 3, 3)
    k = 0
    for j in 1:3, i in 1:3
        k += 1
        ends = split(cells[k], ':')
        length(ends) == 2 || throw(ArgumentError("malformed interval endpoint pair"))
        gram[i, j] = interval(_parse_float(ends[1]), _parse_float(ends[2]))
    end
    return ReferenceBound(id, family, gram, u2), ReferenceOrigin(base_id, p)
end

function _write_references(io::IO, references, origins)
    ids = sort!(collect(keys(references)))
    println(io, "references\t", length(ids))
    for id in ids
        haskey(origins, id) || throw(ArgumentError("missing origin for reference '$id'"))
        println(io, join(_reference_fields(references[id], origins[id]), '\t'))
    end
end

function _read_references(lines::Vector{String}, index::Base.RefValue{Int})
    header = split(lines[index[]], '\t'; keepempty = true)
    index[] += 1
    length(header) == 2 && header[1] == "references" ||
        throw(ArgumentError("missing reference-count record"))
    count = parse(Int, header[2])
    refs = Dict{String, ReferenceBound}()
    origins = Dict{String, ReferenceOrigin}()
    for _ in 1:count
        index[] <= length(lines) || throw(ArgumentError("truncated reference catalog"))
        ref, origin = _parse_reference(split(lines[index[]], '\t'; keepempty = true))
        index[] += 1
        haskey(refs, ref.id) && throw(ArgumentError("duplicate reference id '$(ref.id)'"))
        refs[ref.id] = ref
        origins[ref.id] = origin
    end
    return refs, origins
end

function _atomic_write(writer, path::AbstractString)
    target = abspath(path)
    mkpath(dirname(target))
    temp = string(target, ".tmp")
    open(temp, "w") do io
        writer(io)
        flush(io)
    end
    mv(temp, target; force = true)
    return target
end

"""Save expanded replay references with bit-exact Float64 endpoints."""
function save_reference_catalog(path::AbstractString,
                                references::AbstractDict{String, <:ReferenceBound},
                                origins::AbstractDict{String, ReferenceOrigin})
    _validate_reference_catalog(references, origins)
    return _atomic_write(path) do io
        println(io, _REFERENCE_MAGIC)
        _write_references(io, references, origins)
        println(io, "END")
    end
end

function load_reference_catalog(path::AbstractString)
    lines = readlines(path)
    !isempty(lines) && lines[1] == _REFERENCE_MAGIC ||
        throw(ArgumentError("not a Kappa3 reference catalog"))
    index = Ref(2)
    refs, origins = _read_references(lines, index)
    index[] <= length(lines) && lines[index[]] == "END" ||
        throw(ArgumentError("reference catalog has no END marker"))
    index[] == length(lines) || throw(ArgumentError("trailing records after reference catalog"))
    _validate_reference_catalog(refs, origins)
    return refs, origins
end

function _write_tree(io::IO, node::AbstractCoverNode)
    if node isa CoverLeaf
        leaf = node::CoverLeaf
        ref = leaf.status == :retained ? _escape_id(leaf.reference_id::String) : "-"
        println(io, join(("L", String(leaf.status), ref, _box_text(leaf.box)), '\t'))
    else
        split = node::CoverSplit
        println(io, join(("S", _axes_text(split.axes), _box_text(split.box)), '\t'))
        for child in split.children
            _write_tree(io, child)
        end
    end
end

function _read_tree(lines::Vector{String}, index::Base.RefValue{Int})
    index[] <= length(lines) || throw(ArgumentError("truncated cover tree"))
    fields = split(lines[index[]], '\t'; keepempty = true)
    index[] += 1
    if fields[1] == "L"
        length(fields) == 4 || throw(ArgumentError("malformed leaf record"))
        status = Symbol(fields[2])
        box = _parse_box(fields[4])
        return status == :retained ? CoverLeaf(box, status, _unescape_id(fields[3])) :
                                     CoverLeaf(box, status)
    elseif fields[1] == "S"
        length(fields) == 3 || throw(ArgumentError("malformed split record"))
        axes = _parse_axes(fields[2])
        box = _parse_box(fields[3])
        children = AbstractCoverNode[_read_tree(lines, index)
                                     for _ in 1:(2^length(axes))]
        return CoverSplit(box, axes, children)
    end
    throw(ArgumentError("unknown cover-tree record '$(fields[1])'"))
end

function save_cover_record(path::AbstractString, record::CoverRecord)
    validate_tree(record.tree, record.root; family = record.family)
    return _atomic_write(path) do io
        println(io, _COVER_MAGIC)
        println(io, "family\t", record.family)
        println(io, "root\t", _box_text(record.root))
        println(io, "target\t", _rational_text(record.target))
        _write_tree(io, record.tree)
        println(io, "END")
    end
end

save_cover_record(path::AbstractString, state::AdaptiveSearch) =
    save_cover_record(path, CoverRecord(state.family, state.root, state.target,
                                        materialize_cover(state)))

function load_cover_record(path::AbstractString)
    lines = readlines(path)
    !isempty(lines) && lines[1] == _COVER_MAGIC ||
        throw(ArgumentError("not a Kappa3 cover record"))
    length(lines) >= 6 || throw(ArgumentError("truncated cover record"))
    family_fields = split(lines[2], '\t'; keepempty = true)
    root_fields = split(lines[3], '\t'; keepempty = true)
    target_fields = split(lines[4], '\t'; keepempty = true)
    length(family_fields) == 2 && family_fields[1] == "family" ||
        throw(ArgumentError("missing cover family"))
    length(root_fields) == 2 && root_fields[1] == "root" ||
        throw(ArgumentError("missing cover root"))
    length(target_fields) == 2 && target_fields[1] == "target" ||
        throw(ArgumentError("missing cover target"))
    family = Symbol(family_fields[2])
    root = _parse_box(root_fields[2])
    target = _parse_rational(target_fields[2])
    target > 0 || throw(ArgumentError("cover target must be positive"))
    index = Ref(5)
    tree = _read_tree(lines, index)
    index[] <= length(lines) && lines[index[]] == "END" ||
        throw(ArgumentError("cover record has no END marker"))
    index[] == length(lines) || throw(ArgumentError("trailing records after cover tree"))
    validate_tree(tree, root; family)
    return CoverRecord(family, root, target, tree)
end

function save_checkpoint(path::AbstractString, state::AdaptiveSearch)
    validate_search_state(state)
    return _atomic_write(path) do io
        println(io, _CHECKPOINT_MAGIC)
        println(io, "family\t", state.family)
        println(io, "root\t", _box_text(state.root))
        println(io, "target\t", _rational_text(state.target))
        println(io, "processed\t", state.processed)
        _write_references(io, state.references, state.origins)

        split_paths = sort!(collect(keys(state.splits)); by = _path_sort_key)
        println(io, "splits\t", length(split_paths))
        for pathkey in split_paths
            box = _expected_box(state, pathkey)
            println(io, join(("S", pathkey, _axes_text(state.splits[pathkey]),
                              _box_text(box)), '\t'))
        end

        leaf_paths = sort!(collect(keys(state.leaves)); by = _path_sort_key)
        println(io, "leaves\t", length(leaf_paths))
        for pathkey in leaf_paths
            leaf = state.leaves[pathkey]
            ref = leaf.status == :retained ? _escape_id(leaf.reference_id::String) : "-"
            println(io, join(("L", pathkey, String(leaf.status), ref,
                              _box_text(leaf.box)), '\t'))
        end

        # Vector order is the DFS stack order and therefore part of the
        # deterministic resume state; do not sort it.
        println(io, "pending\t", length(state.pending))
        for cell in state.pending
            println(io, join(("P", cell.path, string(cell.depth),
                              _box_text(cell.box)), '\t'))
        end
        println(io, "END")
    end
end

function _count_header(lines, index, name)
    index[] <= length(lines) || throw(ArgumentError("missing '$name' section"))
    fields = split(lines[index[]], '\t'; keepempty = true)
    index[] += 1
    length(fields) == 2 && fields[1] == name ||
        throw(ArgumentError("missing '$name' section"))
    n = parse(Int, fields[2])
    n >= 0 || throw(ArgumentError("negative '$name' count"))
    return n
end

function load_checkpoint(path::AbstractString)
    lines = readlines(path)
    !isempty(lines) && lines[1] == _CHECKPOINT_MAGIC ||
        throw(ArgumentError("not a Kappa3 search checkpoint"))
    length(lines) >= 9 || throw(ArgumentError("truncated search checkpoint"))
    family_fields = split(lines[2], '\t'; keepempty = true)
    root_fields = split(lines[3], '\t'; keepempty = true)
    target_fields = split(lines[4], '\t'; keepempty = true)
    processed_fields = split(lines[5], '\t'; keepempty = true)
    length(family_fields) == 2 && family_fields[1] == "family" ||
        throw(ArgumentError("missing checkpoint family"))
    length(root_fields) == 2 && root_fields[1] == "root" ||
        throw(ArgumentError("missing checkpoint root"))
    length(target_fields) == 2 && target_fields[1] == "target" ||
        throw(ArgumentError("missing checkpoint target"))
    length(processed_fields) == 2 && processed_fields[1] == "processed" ||
        throw(ArgumentError("missing checkpoint processed count"))
    family = Symbol(family_fields[2])
    root = _parse_box(root_fields[2])
    target = _parse_rational(target_fields[2])
    target > 0 || throw(ArgumentError("checkpoint target must be positive"))
    processed = parse(Int, processed_fields[2])
    processed >= 0 || throw(ArgumentError("negative processed count"))
    index = Ref(6)
    refs, origins = _read_references(lines, index)

    splits = Dict{String, Tuple{Vararg{Int}}}()
    saved_split_boxes = Dict{String, DyadicBox}()
    for _ in 1:_count_header(lines, index, "splits")
        fields = split(lines[index[]], '\t'; keepempty = true)
        index[] += 1
        length(fields) == 4 && fields[1] == "S" ||
            throw(ArgumentError("malformed checkpoint split"))
        pathkey, axes, box = fields[2], _parse_axes(fields[3]), _parse_box(fields[4])
        haskey(splits, pathkey) && throw(ArgumentError("duplicate split path '$pathkey'"))
        splits[pathkey] = axes
        saved_split_boxes[pathkey] = box
    end

    leaves = Dict{String, CoverLeaf}()
    for _ in 1:_count_header(lines, index, "leaves")
        fields = split(lines[index[]], '\t'; keepempty = true)
        index[] += 1
        length(fields) == 5 && fields[1] == "L" ||
            throw(ArgumentError("malformed checkpoint leaf"))
        pathkey, status, box = fields[2], Symbol(fields[3]), _parse_box(fields[5])
        haskey(leaves, pathkey) && throw(ArgumentError("duplicate leaf path '$pathkey'"))
        leaves[pathkey] = status == :retained ?
            CoverLeaf(box, status, _unescape_id(fields[4])) : CoverLeaf(box, status)
    end

    pending = SearchCell[]
    for _ in 1:_count_header(lines, index, "pending")
        fields = split(lines[index[]], '\t'; keepempty = true)
        index[] += 1
        length(fields) == 4 && fields[1] == "P" ||
            throw(ArgumentError("malformed checkpoint pending cell"))
        push!(pending, SearchCell(fields[2], _parse_box(fields[4]), parse(Int, fields[3])))
    end
    index[] <= length(lines) && lines[index[]] == "END" ||
        throw(ArgumentError("checkpoint has no END marker"))
    index[] == length(lines) || throw(ArgumentError("trailing records after checkpoint"))

    state = AdaptiveSearch(family, root, target, refs, origins, pending,
                           splits, leaves, processed)
    validate_search_state(state)
    for (pathkey, box) in saved_split_boxes
        box == _expected_box(state, pathkey) ||
            throw(ArgumentError("split '$pathkey' has the wrong exact box"))
    end
    return state
end

end # module CoverSearch
