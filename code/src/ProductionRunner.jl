module ProductionRunner

using Printf
using IntervalArithmetic: inf, sup
import Kappa3Certificate

using ..BoxCover: family_target, ReferenceBound, DyadicBox,
                  star_root_box, path_root_box
using ..CoverSearch: AdaptiveSearch, ReferenceOrigin, SearchCell,
                     start_search, advance_search!, search_complete,
                     save_checkpoint, load_checkpoint, save_cover_record,
                     load_cover_record, replay_record
using ..ReferencePlan: ReferenceSeed, CatalogPlan, full_reference_plan,
                       expanded_representations, enrichment_seed,
                       promote_seed, needs_promotion
using ..ProofRecords: CANONICAL_SCRATCH_ROOT, ScratchLayout,
                      prepare_scratch_layout, collect_standard_fingerprints,
                      save_fingerprints, load_fingerprints,
                      verify_standard_fingerprints, capture_environment,
                      save_environment, load_environment, sha256_file,
                      load_point_record, point_proof_record,
                      save_point_record, replay_point,
                      reference_proof_record, ReferenceProofRecord, ReferenceIndex,
                      save_reference_index, replay_reference_index,
                      ArtifactLink, ProofBundle, save_proof_bundle,
                      load_proof_bundle, verify_bundle_artifacts

export ProductionConfig, RunLimits, ProductionOutcome,
       save_seed_plan, load_seed_plan, validate_seed_plan,
       pending_cells, pending_promotions, pending_enrichments,
       same_search_catalog, inspect_status, initialize_production,
       run_production, parse_cli, cli_main

const _SEED_MAGIC = "KAPPA3_PRODUCTION_SEEDS_V1"
const _IDENTITY_TRANSFORM = (1, 0, 0,
                             0, 1, 0,
                             0, 0, 1)
const _PATH_REVERSAL_TRANSFORM = (0, -1, 1,
                                  0, -1, 0,
                                  1, -1, 0)
const _POINT_FILE_RE = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"

"""Immutable locations and numerical choices for one production campaign."""
Base.@kwdef struct ProductionConfig
    code_root::String
    scratch_root::String = CANONICAL_SCRATCH_ROOT
    vfem_root::String = "/home/xfliu/Workspace/Julia_LIB/Lib/VFEM.jl"
    veigs_root::String = "/home/xfliu/Workspace/Julia_LIB/Lib/Veigs.jl"
    basis::Symbol = :haar
    mesh_strategy::Symbol = :metric
    initial_backoff::Float64 = 1e-4
    max_spd_attempts::Int = 8
end

"""
Hard resource bounds for a resumable invocation.  Re-running the same command
continues from atomic point records and search checkpoints.  The bounds are
deliberately modest: the launcher is the only supported entry point for the
heavy `run` and `replay` modes.
"""
Base.@kwdef struct RunLimits
    max_point_solves::Int = 8
    max_search_nodes::Int = 200_000
    max_enrichments::Int = 8
    max_cycles::Int = 2
    checkpoint_every::Int = 1_000
    star_depths::Vector{Int} = [4, 5, 6, 7, 8]
    path_depths::Vector{Int} = [12, 18, 24, 30]
end

struct ProductionOutcome
    complete::Bool
    point_solves::Int
    search_nodes::Int
    new_seeds::Int
    message::String
end

_bits(x::Float64) = reinterpret(UInt64, x)
_float_token(x::Float64) = string("0x", string(_bits(x); base = 16, pad = 16))

function _parse_float_token(text::AbstractString)
    occursin(r"^0x[0-9a-f]{16}$", text) ||
        throw(ArgumentError("invalid Float64 seed token '$text'"))
    value = reinterpret(Float64, parse(UInt64, text[3:end]; base = 16))
    isfinite(value) || throw(ArgumentError("nonfinite seed parameter"))
    _float_token(value) == text || throw(ArgumentError("noncanonical Float64 seed token"))
    return value
end

function _atomic_write(writer, path::AbstractString)
    target = abspath(path)
    mkpath(dirname(target))
    temporary = string(target, ".tmp.", getpid())
    ispath(temporary) && throw(ArgumentError("temporary file already exists: '$temporary'"))
    try
        open(temporary, "w") do io
            writer(io)
            flush(io)
        end
        mv(temporary, target; force = true)
    finally
        ispath(temporary) && rm(temporary; force = true)
    end
    return target
end

function _same_parameters(a, b)
    return all(_bits(Float64(a[i])) == _bits(Float64(b[i])) for i in 1:3)
end

function _same_seed(a::ReferenceSeed, b::ReferenceSeed)
    return a.id == b.id && a.family == b.family &&
           _same_parameters(a.parameters_deg, b.parameters_deg) &&
           a.primary_level == b.primary_level &&
           a.fallback_level == b.fallback_level &&
           a.priority == b.priority && a.role == b.role
end

_seed_order(seed::ReferenceSeed) = (seed.priority, String(seed.family), seed.id)

function _canonical_seeds(seeds)
    result = sort!(ReferenceSeed[seeds...]; by = _seed_order)
    ids = getfield.(result, :id)
    length(unique(ids)) == length(ids) || throw(ArgumentError("duplicate production seed id"))
    keys = [(seed.family, ntuple(i -> _bits(seed.parameters_deg[i]), 3),
             seed.primary_level) for seed in result]
    length(unique(keys)) == length(keys) ||
        throw(ArgumentError("duplicate production seed family/point/level"))
    return result
end

"""Write the deterministic requested-solve manifest atomically."""
function save_seed_plan(path::AbstractString, seeds)
    ordered = _canonical_seeds(seeds)
    return _atomic_write(path) do io
        println(io, _SEED_MAGIC)
        println(io, "seeds\t", length(ordered))
        for seed in ordered
            fallback = isnothing(seed.fallback_level) ? "-" : string(seed.fallback_level)
            params = join(_float_token.(seed.parameters_deg), ',')
            println(io, join(("S", seed.id, String(seed.family), params,
                              string(seed.primary_level), fallback,
                              string(seed.priority), String(seed.role)), '\t'))
        end
        println(io, "END")
    end
end

"""Load a strict, canonical requested-solve manifest."""
function load_seed_plan(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 3 && lines[1] == _SEED_MAGIC ||
        throw(ArgumentError("not a Kappa3 production seed plan"))
    header = split(lines[2], '\t'; keepempty = true)
    length(header) == 2 && header[1] == "seeds" ||
        throw(ArgumentError("missing seed count"))
    count = parse(Int, header[2])
    count >= 0 || throw(ArgumentError("negative seed count"))
    length(lines) == count + 3 || throw(ArgumentError("seed count mismatch"))
    lines[end] == "END" || throw(ArgumentError("seed plan has no END marker"))
    seeds = ReferenceSeed[]
    for line in lines[3:end-1]
        fields = split(line, '\t'; keepempty = true)
        length(fields) == 8 && fields[1] == "S" ||
            throw(ArgumentError("malformed production seed entry"))
        occursin(_POINT_FILE_RE, fields[2]) ||
            throw(ArgumentError("unsafe production seed id '$(fields[2])'"))
        pfields = split(fields[4], ','; keepempty = true)
        length(pfields) == 3 || throw(ArgumentError("seed entry needs three angles"))
        parameters = Tuple(_parse_float_token.(pfields))
        fallback = fields[6] == "-" ? nothing : parse(Int, fields[6])
        push!(seeds, ReferenceSeed(fields[2], Symbol(fields[3]), parameters,
                                   parse(Int, fields[5]), fallback,
                                   parse(Int, fields[7]), Symbol(fields[8])))
    end
    ordered = _canonical_seeds(seeds)
    all(_same_seed(a, b) for (a, b) in zip(seeds, ordered)) ||
        throw(ArgumentError("production seed plan is not canonically sorted"))
    return ordered
end

function _parameter_key(seed::ReferenceSeed)
    return (seed.family, ntuple(i -> _bits(seed.parameters_deg[i]), 3))
end

"""
Require the immutable `full_reference_plan()` and accept only deterministic
level-5 promotions or deterministic adaptive-enrichment ids after it.
"""
function validate_seed_plan(seeds)
    ordered = _canonical_seeds(seeds)
    by_id = Dict(seed.id => seed for seed in ordered)
    baseline = full_reference_plan().seeds
    for expected in baseline
        haskey(by_id, expected.id) ||
            throw(ArgumentError("production seed plan omits baseline seed '$(expected.id)'"))
        _same_seed(by_id[expected.id], expected) ||
            throw(ArgumentError("baseline seed '$(expected.id)' was changed"))
    end

    baseline_ids = Set(getfield.(baseline, :id))
    for seed in ordered
        seed.id in baseline_ids && continue
        if seed.role == :adaptive_enrichment
            pattern = seed.family == :star ? r"^SE[0-9]{5}-L[45]$" :
                                             r"^PE[0-9]{5}-L[45]$"
            occursin(pattern, seed.id) ||
                throw(ArgumentError("noncanonical enrichment id '$(seed.id)'"))
        elseif seed.role == :level5_promotion
            sources = [source for source in ordered
                       if source.family == seed.family &&
                          source.primary_level == 4 &&
                          source.fallback_level == 5 &&
                          _same_parameters(source.parameters_deg, seed.parameters_deg)]
            length(sources) == 1 ||
                throw(ArgumentError("promotion '$(seed.id)' has no unique level-4 source"))
            _same_seed(seed, promote_seed(only(sources))) ||
                throw(ArgumentError("promotion '$(seed.id)' is not canonical"))
        else
            throw(ArgumentError("unrecognized added seed '$(seed.id)'"))
        end
    end
    return ordered
end

function _validate_limits(limits::RunLimits)
    0 <= limits.max_point_solves <= 32 ||
        throw(ArgumentError("max_point_solves must lie in 0:32"))
    0 <= limits.max_search_nodes <= 2_000_000 ||
        throw(ArgumentError("max_search_nodes must lie in 0:2000000"))
    0 <= limits.max_enrichments <= 64 ||
        throw(ArgumentError("max_enrichments must lie in 0:64"))
    1 <= limits.max_cycles <= 8 || throw(ArgumentError("max_cycles must lie in 1:8"))
    1 <= limits.checkpoint_every <= 100_000 ||
        throw(ArgumentError("checkpoint_every must lie in 1:100000"))
    for (family, depths, maximum) in ((:star, limits.star_depths, 12),
                                       (:path, limits.path_depths, 60))
        !isempty(depths) || throw(ArgumentError("$family depth schedule is empty"))
        issorted(depths) && length(unique(depths)) == length(depths) ||
            throw(ArgumentError("$family depths must be strictly increasing"))
        first(depths) >= 1 && last(depths) <= maximum ||
            throw(ArgumentError("$family depth schedule is outside 1:$maximum"))
    end
    return limits
end

_seed_plan_path(layout::ScratchLayout) = joinpath(layout.checkpoints, "reference-seeds.txt")
_fingerprint_path(layout::ScratchLayout) = joinpath(layout.proof, "source-fingerprints.txt")
_environment_path(layout::ScratchLayout) = joinpath(layout.proof, "environment.txt")
# The index may only name descendants of its own directory.  Keeping it at
# the campaign root makes `points/*.point` safe without accepting `..`.
_reference_path(layout::ScratchLayout) = joinpath(layout.root, "references.txt")
_cover_path(layout::ScratchLayout, family::Symbol) = joinpath(layout.proof, "$(family)-cover.txt")
_checkpoint_path(layout::ScratchLayout, family::Symbol) = joinpath(layout.checkpoints, "$(family)-search.txt")
_bundle_path(layout::ScratchLayout) = joinpath(layout.root, "proof-bundle.txt")

function _point_path(layout::ScratchLayout, id::AbstractString)
    occursin(_POINT_FILE_RE, id) || throw(ArgumentError("unsafe point id '$id'"))
    return joinpath(layout.points, string(id, ".point"))
end

function _has_campaign_state(layout::ScratchLayout)
    !isempty(readdir(layout.points)) && return true
    any(isfile, (_seed_plan_path(layout), _reference_path(layout),
                 _checkpoint_path(layout, :star), _checkpoint_path(layout, :path),
                 _cover_path(layout, :star), _cover_path(layout, :path),
                 _bundle_path(layout)))
end

function _same_environment(a, b)
    return a.entries == b.entries
end

"""
Create immutable provenance and the baseline seed plan.  Existing files are
verified, never silently replaced.  This makes resuming after interruption
safe and makes a source edit require a fresh campaign directory.
"""
function initialize_production(config::ProductionConfig)
    Kappa3Certificate.RigorousSPD.ieee_preflight()
    layout = prepare_scratch_layout(config.scratch_root)
    fpath, epath = _fingerprint_path(layout), _environment_path(layout)
    if isfile(fpath)
        recorded = load_fingerprints(fpath)
        verify_standard_fingerprints(recorded, config.code_root;
                                     vfem_root = config.vfem_root,
                                     veigs_root = config.veigs_root)
    else
        _has_campaign_state(layout) &&
            throw(ArgumentError("campaign state exists without source fingerprints"))
        save_fingerprints(fpath, collect_standard_fingerprints(
            config.code_root; vfem_root = config.vfem_root,
            veigs_root = config.veigs_root))
    end
    if isfile(epath)
        recorded = load_environment(epath)
        _same_environment(recorded, capture_environment()) ||
            throw(ArgumentError("current host/runtime environment differs from the campaign snapshot"))
    else
        !isempty(readdir(layout.points)) &&
            throw(ArgumentError("point records exist without an environment snapshot"))
        save_environment(epath, capture_environment())
    end

    seed_path = _seed_plan_path(layout)
    seeds = if isfile(seed_path)
        validate_seed_plan(load_seed_plan(seed_path))
    else
        _has_campaign_state(layout) &&
            throw(ArgumentError("derived campaign state exists without a seed plan"))
        initial = validate_seed_plan(full_reference_plan().seeds)
        save_seed_plan(seed_path, initial)
        initial
    end
    return layout, seeds, sha256_file(fpath), sha256_file(epath)
end

function _validate_point_metadata(record, seed::ReferenceSeed,
                                  fingerprints_hash::String,
                                  environment_hash::String,
                                  config::ProductionConfig)
    record.id == seed.id || throw(ArgumentError("point file id differs from seed '$(seed.id)'"))
    record.family == seed.family || throw(ArgumentError("point family differs for '$(seed.id)'"))
    _same_parameters(record.parameters_deg, seed.parameters_deg) ||
        throw(ArgumentError("point angles differ for '$(seed.id)'"))
    record.refinement_level == seed.primary_level ||
        throw(ArgumentError("point level differs for '$(seed.id)'"))
    record.basis == config.basis || throw(ArgumentError("point basis differs for '$(seed.id)'"))
    record.mesh_strategy == config.mesh_strategy ||
        throw(ArgumentError("point mesh strategy differs for '$(seed.id)'"))
    record.spd_method == :residual ||
        throw(ArgumentError("point '$(seed.id)' does not use residual SPD"))
    record.spd_max_attempts == config.max_spd_attempts ||
        throw(ArgumentError("point SPD attempt limit differs for '$(seed.id)'"))
    record.fingerprints_sha256 == fingerprints_hash ||
        throw(ArgumentError("point '$(seed.id)' is bound to different source fingerprints"))
    record.environment_sha256 == environment_hash ||
        throw(ArgumentError("point '$(seed.id)' is bound to a different environment"))
    return record
end

function _load_point(layout, seed, fhash, ehash, config)
    path = _point_path(layout, seed.id)
    isfile(path) || return nothing
    return _validate_point_metadata(load_point_record(path), seed, fhash, ehash, config)
end

function _has_level5(seeds, source::ReferenceSeed)
    return any(seed -> seed.family == source.family && seed.primary_level == 5 &&
                       _same_parameters(seed.parameters_deg, source.parameters_deg), seeds)
end

function _append_seeds!(layout, seeds::Vector{ReferenceSeed}, additions)
    isempty(additions) && return 0
    ids = Set(getfield.(seeds, :id))
    keys = Set((seed.family, ntuple(i -> _bits(seed.parameters_deg[i]), 3),
                seed.primary_level) for seed in seeds)
    count = 0
    for seed in additions
        seed.id in ids && continue
        key = (seed.family, ntuple(i -> _bits(seed.parameters_deg[i]), 3),
               seed.primary_level)
        key in keys && continue
        push!(seeds, seed)
        push!(ids, seed.id); push!(keys, key)
        count += 1
    end
    if count > 0
        seeds[:] = validate_seed_plan(seeds)
        save_seed_plan(_seed_plan_path(layout), seeds)
    end
    return count
end

function _automatic_promotions(layout, seeds, records)
    result = ReferenceSeed[]
    for seed in seeds
        haskey(records, seed.id) || continue
        needs_promotion(seed, records[seed.id].u2_upper) || continue
        _has_level5(seeds, seed) && continue
        push!(result, promote_seed(seed))
    end
    sort!(result; by = _seed_order)
    return result
end

function _solve_one_point(layout, seed, fhash, ehash, config)
    @printf("POINT START id=%s family=%s angles=%s level=%d\n",
            seed.id, String(seed.family), repr(seed.parameters_deg), seed.primary_level)
    flush(stdout)
    certificate = Kappa3Certificate.point_certificate(
        seed.family, seed.parameters_deg;
        level = seed.primary_level, basis = config.basis,
        initial_backoff = config.initial_backoff,
        mesh_strategy = config.mesh_strategy,
        max_spd_attempts = config.max_spd_attempts)
    record = point_proof_record(
        seed.id, certificate; fingerprints_sha256 = fhash,
        environment_sha256 = ehash, spd_initial_rho = 0.0,
        spd_max_attempts = config.max_spd_attempts)
    path = _point_path(layout, seed.id)
    save_point_record(path, record)
    loaded = _validate_point_metadata(load_point_record(path), seed, fhash, ehash, config)
    replay_point(loaded) # a newly emitted point must pass before progress is reported
    @printf("POINT PASSED id=%s C_upper=%.17g sha256=%s\n",
            seed.id, loaded.C_upper, sha256_file(path))
    flush(stdout)
    return loaded
end

function _ensure_points!(layout, seeds, fhash, ehash, config, solve_budget::Int)
    records = Dict{String, Any}()
    for seed in seeds
        record = _load_point(layout, seed, fhash, ehash, config)
        isnothing(record) || (records[seed.id] = record)
    end
    additions = _automatic_promotions(layout, seeds, records)
    added = _append_seeds!(layout, seeds, additions)

    solved = 0
    while solved < solve_budget
        missing = [seed for seed in seeds if !haskey(records, seed.id)]
        isempty(missing) && break
        seed = first(sort!(missing; by = _seed_order))
        record = _solve_one_point(layout, seed, fhash, ehash, config)
        records[seed.id] = record
        solved += 1
        if needs_promotion(seed, record.u2_upper) && !_has_level5(seeds, seed)
            added += _append_seeds!(layout, seeds, [promote_seed(seed)])
        end
    end

    missing = [seed.id for seed in seeds if !haskey(records, seed.id)]
    sort!(missing)
    return records, missing, solved, added
end

function _build_reference_index(layout, seeds)
    records = ReferenceProofRecord[]
    for seed in sort!(copy(seeds); by = _seed_order)
        point_file = _point_path(layout, seed.id)
        isfile(point_file) || throw(ArgumentError("missing point record for '$(seed.id)'"))
        relative = relpath(point_file, dirname(_reference_path(layout)))
        push!(records, reference_proof_record(seed.id, relative, point_file;
                                              transform = _IDENTITY_TRANSFORM))
        if seed.family == :path && seed.parameters_deg[1] != seed.parameters_deg[2]
            push!(records, reference_proof_record(
                string(seed.id, "@reverse"), relative, point_file;
                transform = _PATH_REVERSAL_TRANSFORM))
        end
    end
    index = ReferenceIndex(records)
    path = _reference_path(layout)
    save_reference_index(path, index)
    return path
end

function _reference_bounds(index_path, fhash, ehash)
    replayed = replay_reference_index(index_path;
                                      fingerprints_sha256 = fhash,
                                      environment_sha256 = ehash)
    return Dict(id => ReferenceBound(id, point.family, point.gram, point.u2_upper)
                for (id, point) in replayed)
end

function _same_float(a::Float64, b::Float64)
    return _bits(a) == _bits(b)
end

function _same_gram(A, B)
    size(A) == size(B) == (3, 3) || return false
    return all(_same_float(inf(A[i, j]), inf(B[i, j])) &&
               _same_float(sup(A[i, j]), sup(B[i, j]))
               for i in 1:3, j in 1:3)
end

"""Bit-exact comparison used to decide whether a checkpoint may continue."""
function same_search_catalog(old::AdaptiveSearch, fresh::AdaptiveSearch)
    old.family == fresh.family && old.root == fresh.root && old.target == fresh.target ||
        return false
    Set(keys(old.references)) == Set(keys(fresh.references)) || return false
    Set(keys(old.origins)) == Set(keys(fresh.origins)) || return false
    for id in keys(fresh.references)
        a, b = old.references[id], fresh.references[id]
        a.id == b.id && a.family == b.family && _same_float(a.u2_upper, b.u2_upper) &&
            _same_gram(a.gram, b.gram) || return false
        oa, ob = old.origins[id], fresh.origins[id]
        oa.base_id == ob.base_id && oa.permutation == ob.permutation || return false
    end
    return true
end

function _validated_search_state(layout, family::Symbol, references)
    family_refs = Dict(id => ref for (id, ref) in references if ref.family == family)
    isempty(family_refs) && throw(ArgumentError("no $family references are available"))
    fresh = start_search(family, family_refs; target = family_target(family))
    path = _checkpoint_path(layout, family)
    if !isfile(path)
        save_checkpoint(path, fresh)
        return fresh, true
    end
    saved = load_checkpoint(path)
    saved.family == family || throw(ArgumentError("$family checkpoint has wrong family"))
    saved.root == (family == :star ? star_root_box() : path_root_box()) ||
        throw(ArgumentError("$family checkpoint has a noncanonical root"))
    saved.target == family_target(family) ||
        throw(ArgumentError("$family checkpoint has a noncanonical target"))
    if same_search_catalog(saved, fresh)
        return saved, false
    end

    # A source edit cannot reach here because provenance verification runs
    # first.  Thus a changed catalogue must be the append-only result of a
    # recorded promotion/enrichment.  Refuse deletion or mutation, then
    # restart so every coarser cell can benefit from the new references.
    issubset(Set(keys(saved.references)), Set(keys(fresh.references))) ||
        throw(ArgumentError("$family checkpoint references are not an append-only subset"))
    for id in keys(saved.references)
        a, b = saved.references[id], fresh.references[id]
        _same_float(a.u2_upper, b.u2_upper) && _same_gram(a.gram, b.gram) ||
            throw(ArgumentError("$family checkpoint reference '$id' changed"))
        saved.origins[id].base_id == fresh.origins[id].base_id &&
        saved.origins[id].permutation == fresh.origins[id].permutation ||
            throw(ArgumentError("$family checkpoint origin '$id' changed"))
    end
    save_checkpoint(path, fresh)
    return fresh, true
end

"""Return pending cells in a stable human/audit order, not stack order."""
function pending_cells(state::AdaptiveSearch)
    return sort!(copy(state.pending); by = cell -> (cell.depth, cell.path))
end

function _dyadic_float(x)
    return Float64(x.numerator // (BigInt(1) << x.exponent))
end

function _contains(box::DyadicBox, parameters)
    for i in 1:3
        lo = _dyadic_float(box.bounds[i].lo)
        hi = _dyadic_float(box.bounds[i].hi)
        lo <= parameters[i] <= hi || return false
    end
    return true
end

function _seed_images(seed::ReferenceSeed)
    return getfield.(expanded_representations(CatalogPlan([seed])),
                     :parameters_deg)
end

"""
Promote completed level-4 points lying in a depth-limited failed cell before
requesting a new angular reference.  The ordering and cap are deterministic.
"""
function pending_promotions(family::Symbol, state::AdaptiveSearch, seeds,
                            completed_ids; limit::Int)
    limit >= 0 || throw(DomainError(limit, "promotion limit must be nonnegative"))
    boxes = getfield.(pending_cells(state), :box)
    completed = Set(String.(completed_ids))
    result = ReferenceSeed[]
    for seed in sort!([s for s in seeds if s.family == family]; by = _seed_order)
        seed.id in completed || continue
        seed.primary_level == 4 && seed.fallback_level == 5 || continue
        _has_level5(seeds, seed) && continue
        # Search includes all star angle permutations and path reversal at no
        # new point-solve cost.  A failed cell occupied by any such image is a
        # reason to promote the one underlying level-4 point.
        images = _seed_images(seed)
        any(parameters -> any(box -> _contains(box, parameters), boxes),
            images) || continue
        push!(result, promote_seed(seed))
        length(result) == limit && break
    end
    return result
end

function _next_enrichment_serial(family::Symbol, seeds)
    prefix = family == :star ? "SE" : "PE"
    maximum_seen = 0
    pattern = Regex("^" * prefix * "([0-9]{5})-L[45]" * raw"$")
    for seed in seeds
        match_result = match(pattern, seed.id)
        isnothing(match_result) && continue
        maximum_seen = max(maximum_seen, parse(Int, only(match_result.captures)))
    end
    return maximum_seen + 1
end

"""Choose distinct deterministic enrichment requests from pending cells."""
function pending_enrichments(family::Symbol, state::AdaptiveSearch, seeds;
                             limit::Int)
    limit >= 0 || throw(DomainError(limit, "enrichment limit must be nonnegative"))
    limit == 0 && return ReferenceSeed[]
    seen = Set(_parameter_key(seed) for seed in seeds)
    serial = _next_enrichment_serial(family, seeds)
    result = ReferenceSeed[]
    for cell in pending_cells(state)
        candidate = enrichment_seed(family, cell.box, serial)
        key = _parameter_key(candidate)
        key in seen && continue
        push!(seen, key)
        push!(result, candidate)
        serial += 1
        length(result) == limit && break
    end
    return result
end

function _cover_is_valid(path, family, expanded_references)
    isfile(path) || return false
    record = load_cover_record(path)
    record.family == family || throw(ArgumentError("$family cover has wrong family"))
    record.root == (family == :star ? star_root_box() : path_root_box()) ||
        throw(ArgumentError("$family cover has noncanonical root"))
    record.target == family_target(family) ||
        throw(ArgumentError("$family cover has noncanonical target"))
    result = replay_record(record, expanded_references)
    result.passed || throw(ArgumentError("stored $family cover failed replay: " *
                                         join(result.failures, "; ")))
    return true
end

function _run_search!(layout, family, references, depths, node_budget,
                      checkpoint_every)
    state, restarted = _validated_search_state(layout, family, references)
    expanded = state.references
    cover_path = _cover_path(layout, family)
    if _cover_is_valid(cover_path, family, expanded)
        return (complete = true, nodes = 0, state = state,
                depth_limited = false, restarted = restarted)
    end

    remaining = node_budget
    processed = 0
    checkpoint = _checkpoint_path(layout, family)
    for depth in depths
        remaining == 0 && break
        # CoverSearch defers a cell before testing it when cell.depth equals
        # max_depth.  Pass depth+1 so the advertised depth is actually tested.
        progress = advance_search!(state; max_new_nodes = remaining,
                                   max_depth = depth + 1,
                                   checkpoint_path = checkpoint,
                                   checkpoint_every)
        processed += progress.processed_now
        remaining -= progress.processed_now
        @printf("SEARCH family=%s depth=%d processed=%d total=%d pending=%d retained=%d excluded=%d\n",
                String(family), depth, progress.processed_now,
                progress.processed_total, progress.pending,
                progress.retained, progress.excluded)
        flush(stdout)
        search_complete(state) && break
    end
    if search_complete(state)
        save_cover_record(cover_path, state)
        _cover_is_valid(cover_path, family, expanded) ||
            error("internal error: completed $family cover was not saved")
        return (complete = true, nodes = processed, state = state,
                depth_limited = false, restarted = restarted)
    end
    limited = !isempty(state.pending) &&
              all(cell -> cell.depth >= last(depths) + 1, state.pending)
    return (complete = false, nodes = processed, state = state,
            depth_limited = limited, restarted = restarted)
end

function _bundle(layout)
    paths = Dict(
        "environment" => _environment_path(layout),
        "fingerprints" => _fingerprint_path(layout),
        "path_cover" => _cover_path(layout, :path),
        "references" => _reference_path(layout),
        "star_cover" => _cover_path(layout, :star),
    )
    all(isfile, values(paths)) || throw(ArgumentError("cannot bundle incomplete proof artifacts"))
    root = layout.root
    links = [ArtifactLink(role, relpath(path, root), sha256_file(path))
             for (role, path) in paths]
    bundle = ProofBundle(links)
    path = _bundle_path(layout)
    save_proof_bundle(path, bundle)
    verify_bundle_artifacts(path, load_proof_bundle(path))
    return path
end

function _run_independent_replay(config, bundle_path)
    script = joinpath(config.code_root, "scripts", "replay_proof.jl")
    isfile(script) || throw(ArgumentError("independent replay script is absent"))
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(config.code_root) $script $bundle_path`
    println("FINAL REPLAY command=", command)
    flush(stdout)
    run(command)
    return true
end

function _require_launcher_guard()
    get(ENV, "KAPPA3_LAUNCHER_GUARD", "") == "passed" ||
        throw(ArgumentError("heavy mode refused: use scripts/launch_production.sh so blocker and resource checks run"))
end

"""
Execute one bounded, resumable production invocation.  A return value with
`complete=false` is not a proof claim; rerun with the same immutable source
tree and campaign directory.  Only an independently replayed bundle returns
`complete=true`.
"""
function run_production(config::ProductionConfig, limits::RunLimits = RunLimits();
                        require_launcher::Bool = true)
    require_launcher && _require_launcher_guard()
    _validate_limits(limits)
    layout, seeds, fhash, ehash = initialize_production(config)
    total_solves = 0
    total_nodes = 0
    total_added = 0

    for cycle in 1:limits.max_cycles
        println("CYCLE ", cycle, "/", limits.max_cycles)
        remaining_solves = limits.max_point_solves - total_solves
        records, missing, solved, added = _ensure_points!(
            layout, seeds, fhash, ehash, config, max(0, remaining_solves))
        total_solves += solved
        total_added += added
        if !isempty(missing)
            message = "point budget exhausted; $(length(missing)) requested point records remain"
            println("INCOMPLETE: ", message)
            println("  next point ids: ", join(first(missing, min(8, length(missing))), ", "))
            return ProductionOutcome(false, total_solves, total_nodes, total_added, message)
        end

        index_path = _build_reference_index(layout, seeds)
        references = _reference_bounds(index_path, fhash, ehash)
        completed = Set(keys(records))
        requests = ReferenceSeed[]
        family_complete = Dict{Symbol, Bool}()
        for (family, depths) in ((:star, limits.star_depths),
                                 (:path, limits.path_depths))
            node_allowance = max(0, limits.max_search_nodes - total_nodes)
            result = _run_search!(layout, family, references, depths,
                                  node_allowance,
                                  limits.checkpoint_every)
            total_nodes += result.nodes
            family_complete[family] = result.complete
            result.complete && continue
            request_allowance = max(0, limits.max_enrichments - length(requests))
            if result.depth_limited && request_allowance > 0
                promotions = pending_promotions(family, result.state, seeds,
                                                completed;
                                                limit = request_allowance)
                if !isempty(promotions)
                    append!(requests, promotions)
                else
                    append!(requests, pending_enrichments(
                        family, result.state, seeds;
                        limit = request_allowance))
                end
            end
        end

        if get(family_complete, :star, false) && get(family_complete, :path, false)
            bundle_path = _bundle(layout)
            _run_independent_replay(config, bundle_path)
            message = "canonical star and path covers passed independent replay"
            println("COMPLETE: ", message)
            return ProductionOutcome(true, total_solves, total_nodes, total_added, message)
        end

        added_now = _append_seeds!(layout, seeds, requests)
        total_added += added_now
        if added_now == 0
            message = total_nodes >= limits.max_search_nodes ?
                "search-node budget exhausted; checkpoint saved" :
                "depth-limited cells remain but no new deterministic reference was requested"
            println("INCOMPLETE: ", message)
            return ProductionOutcome(false, total_solves, total_nodes, total_added, message)
        end
        println("ENRICHMENT requested=", added_now,
                " (point records will be solved before search restarts)")
    end

    message = "cycle budget exhausted; campaign state is resumable"
    println("INCOMPLETE: ", message)
    return ProductionOutcome(false, total_solves, total_nodes, total_added, message)
end

function inspect_status(config::ProductionConfig; io::IO = stdout)
    root = abspath(config.scratch_root)
    if !isdir(root)
        println(io, "STATUS uninitialized root=", root)
        return (initialized = false, points = 0, seeds = 0, bundle = false)
    end
    seed_path = joinpath(root, "checkpoints", "reference-seeds.txt")
    seeds = isfile(seed_path) ? load_seed_plan(seed_path) : ReferenceSeed[]
    point_count = count(seed -> isfile(joinpath(root, "points", seed.id * ".point")), seeds)
    println(io, "STATUS root=", root)
    println(io, "  seeds=", length(seeds), " point_records=", point_count,
            " missing=", length(seeds) - point_count)
    for family in (:star, :path)
        checkpoint = joinpath(root, "checkpoints", "$(family)-search.txt")
        cover = joinpath(root, "proof", "$(family)-cover.txt")
        if isfile(checkpoint)
            try
                state = load_checkpoint(checkpoint)
                println(io, "  ", family, " checkpoint: processed=", state.processed,
                        " pending=", length(state.pending), " cover=", isfile(cover))
            catch error
                println(io, "  ", family, " checkpoint: INVALID (", sprint(showerror, error), ")")
            end
        else
            println(io, "  ", family, " checkpoint: absent cover=", isfile(cover))
        end
    end
    bundle = isfile(joinpath(root, "proof-bundle.txt"))
    println(io, "  proof_bundle=", bundle)
    return (initialized = isfile(seed_path), points = point_count,
            seeds = length(seeds), bundle = bundle)
end

function _parse_nonnegative(name, text, maximum)
    occursin(r"^(0|[1-9][0-9]*)$", text) ||
        throw(ArgumentError("$name must be a nonnegative integer"))
    value = parse(Int, text)
    value <= maximum || throw(ArgumentError("$name exceeds the hard cap $maximum"))
    return value
end

function _parse_depths(name, text, maximum)
    fields = split(text, ','; keepempty = true)
    isempty(fields) && throw(ArgumentError("$name must not be empty"))
    values = [_parse_nonnegative(name, field, maximum) for field in fields]
    all(>(0), values) || throw(ArgumentError("$name depths must be positive"))
    return values
end

"""Strict CLI parser.  A mode is mandatory; unknown options are rejected."""
function parse_cli(args)
    isempty(args) && throw(ArgumentError("mode required: status, initialize, run, or replay"))
    mode = Symbol(first(args))
    mode in (:status, :initialize, :run, :replay) ||
        throw(ArgumentError("unknown mode '$(first(args))'"))
    limits = RunLimits()
    values = Dict{Symbol, Any}(
        :max_point_solves => limits.max_point_solves,
        :max_search_nodes => limits.max_search_nodes,
        :max_enrichments => limits.max_enrichments,
        :max_cycles => limits.max_cycles,
        :checkpoint_every => limits.checkpoint_every,
        :star_depths => limits.star_depths,
        :path_depths => limits.path_depths,
    )
    for argument in args[2:end]
        fields = split(argument, '='; limit = 2)
        length(fields) == 2 || throw(ArgumentError("option must have --name=value form: '$argument'"))
        name, value = fields
        if name == "--max-point-solves"
            values[:max_point_solves] = _parse_nonnegative(name, value, 32)
        elseif name == "--max-search-nodes"
            values[:max_search_nodes] = _parse_nonnegative(name, value, 2_000_000)
        elseif name == "--max-enrichments"
            values[:max_enrichments] = _parse_nonnegative(name, value, 64)
        elseif name == "--max-cycles"
            values[:max_cycles] = _parse_nonnegative(name, value, 8)
        elseif name == "--checkpoint-every"
            values[:checkpoint_every] = _parse_nonnegative(name, value, 100_000)
        elseif name == "--star-depths"
            values[:star_depths] = _parse_depths(name, value, 12)
        elseif name == "--path-depths"
            values[:path_depths] = _parse_depths(name, value, 60)
        else
            throw(ArgumentError("unknown option '$name'"))
        end
    end
    mode != :run && length(args) > 1 &&
        throw(ArgumentError("mode '$mode' takes no options"))
    parsed = RunLimits(; (name => values[name] for name in keys(values))...)
    _validate_limits(parsed)
    return mode, parsed
end

function cli_main(args; code_root::AbstractString)
    mode, limits = parse_cli(args)
    config = ProductionConfig(
        code_root = abspath(code_root),
        scratch_root = get(ENV, "KAPPA3_SCRATCH_ROOT", CANONICAL_SCRATCH_ROOT),
        vfem_root = get(ENV, "KAPPA3_VFEM_ROOT",
                        "/home/xfliu/Workspace/Julia_LIB/Lib/VFEM.jl"),
        veigs_root = get(ENV, "KAPPA3_VEIGS_ROOT",
                         "/home/xfliu/Workspace/Julia_LIB/Lib/Veigs.jl"),
    )
    if mode == :status
        inspect_status(config)
        return 0
    elseif mode == :initialize
        _require_launcher_guard()
        layout, seeds, fhash, ehash = initialize_production(config)
        println("INITIALIZED root=", layout.root, " seeds=", length(seeds))
        println("  fingerprints_sha256=", fhash)
        println("  environment_sha256=", ehash)
        return 0
    elseif mode == :replay
        _require_launcher_guard()
        bundle = _bundle_path(prepare_scratch_layout(config.scratch_root))
        isfile(bundle) || throw(ArgumentError("proof bundle is absent: '$bundle'"))
        _run_independent_replay(config, bundle)
        return 0
    end
    outcome = run_production(config, limits)
    return outcome.complete ? 0 : 75
end

end # module ProductionRunner
