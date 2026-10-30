module ReferencePlan

using Printf
import Kappa3Certificate

using ..BoxCover: Dyadic, DyadicBox, ReferenceBound, TARGET_KAPPA,
                  star_box_excluded, path_box_excluded

export ReferenceSeed, ReferenceImage, CatalogPlan, PlanSummary,
       star_reference_plan, path_reference_plan, full_reference_plan,
       plan_summary, summary_text, expanded_representations,
       transformed_gram, search_reference_catalog,
       enrichment_seed, enrichment_seeds, promote_seed,
       needs_promotion

const _EDGE_PERMUTATIONS = ((1, 2, 3), (1, 3, 2), (2, 1, 3),
                            (2, 3, 1), (3, 1, 2), (3, 2, 1))

"""
One requested point solve.  A smaller `priority` is scheduled earlier.
`primary_level` is the first red-refinement level to try; `fallback_level`
is deliberately only a request, not a claim that the coarser solve failed.

All catalogue points are physical, nondegenerate, unit-diameter shapes.
Angle permutations and path reversal are represented separately and never
increase the number of point eigenproblems.
"""
struct ReferenceSeed
    id::String
    family::Symbol
    parameters_deg::NTuple{3, Float64}
    primary_level::Int
    fallback_level::Union{Nothing, Int}
    priority::Int
    role::Symbol

    function ReferenceSeed(id::AbstractString, family::Symbol,
                           parameters_deg::NTuple{3, <:Real},
                           primary_level::Integer,
                           fallback_level::Union{Nothing, Integer},
                           priority::Integer, role::Symbol)
        family in (:star, :path) ||
            throw(ArgumentError("family must be :star or :path"))
        p = ntuple(i -> Float64(parameters_deg[i]), 3)
        all(isfinite, p) || throw(ArgumentError("reference angles must be finite"))
        if family == :star
            all(0.0 < x <= 60.0 for x in p) ||
                throw(ArgumentError("star angles must lie in (0,60] degrees"))
            all(p[i] < p[j] + p[k]
                for (i, j, k) in ((1, 2, 3), (2, 1, 3), (3, 1, 2))) ||
                throw(ArgumentError("star angles must satisfy strict triangle inequalities"))
        else
            0.0 < p[1] <= 60.0 && 0.0 < p[2] <= 60.0 && 0.0 < p[3] < 90.0 ||
                throw(ArgumentError("path angles must lie in (0,60]x(0,60]x(0,90)"))
            Kappa3Certificate.path_unit_diameter_certified(p) ||
                throw(ArgumentError("path dihedral angle exceeds, or cannot rigorously certify, the unit-diameter cap"))
        end
        primary_level in (4, 5) ||
            throw(ArgumentError("the production plan uses refinement level 4 or 5"))
        if !isnothing(fallback_level)
            Int(fallback_level) in (4, 5) ||
                throw(ArgumentError("fallback level must be 4 or 5"))
            Int(fallback_level) > Int(primary_level) ||
                throw(ArgumentError("fallback level must exceed the primary level"))
        end
        priority > 0 || throw(ArgumentError("priority must be positive"))
        isempty(id) && throw(ArgumentError("a reference id must not be empty"))
        new(String(id), family, p, Int(primary_level),
            isnothing(fallback_level) ? nothing : Int(fallback_level),
            Int(priority), role)
    end
end

"""A no-new-solve representation obtained by a vertex relabelling."""
struct ReferenceImage
    id::String
    base_id::String
    family::Symbol
    parameters_deg::NTuple{3, Float64}
    relabelling::Symbol
    # Row-major integer T for G_image = T*G_base*T'.
    transform::NTuple{9, Int}
end

struct CatalogPlan
    seeds::Vector{ReferenceSeed}
    function CatalogPlan(seeds::AbstractVector{ReferenceSeed})
        ids = [seed.id for seed in seeds]
        length(ids) == length(unique(ids)) ||
            throw(ArgumentError("reference seed ids must be unique"))
        new(sort(collect(seeds); by = seed ->
                 (seed.priority, String(seed.family), seed.id)))
    end
end

Base.length(plan::CatalogPlan) = length(plan.seeds)
Base.iterate(plan::CatalogPlan, state...) = iterate(plan.seeds, state...)
Base.getindex(plan::CatalogPlan, i::Integer) = plan.seeds[i]

struct PlanSummary
    base_solves::Int
    star_solves::Int
    path_solves::Int
    level4_first::Int
    level5_first::Int
    level5_fallbacks::Int
    symmetry_images::Int
    search_base_bounds::Int
    coversearch_variants::Int
    roles::Vector{Pair{Symbol, Int}}
end

_parameter_key(p::NTuple{3, Float64}) = ntuple(i -> reinterpret(UInt64, p[i]), 3)

function _path_cap(theta1::Real, theta2::Real)
    argument = tand(Float64(theta1) / 2) * tand(Float64(theta2) / 2)
    return acosd(clamp(argument, -1.0, 1.0))
end

function _spec!(specs, seen, parameters, level, fallback, priority, role)
    p = ntuple(i -> Float64(parameters[i]), 3)
    key = _parameter_key(p)
    key in seen && return specs
    push!(seen, key)
    push!(specs, (parameters = p, level = level, fallback = fallback,
                  priority = priority, role = role))
    return specs
end

function _numbered_seeds(family::Symbol, specs)
    prefix = family == :star ? "S" : "P"
    return [ReferenceSeed(string(prefix, lpad(i, 3, '0')), family,
                          spec.parameters, spec.level, spec.fallback,
                          spec.priority, spec.role)
            for (i, spec) in enumerate(specs)]
end

"""
    star_reference_plan()

Return a symmetry-reduced, deterministic star catalogue.  It has three
level-5 anchors at 5, 7.5, and 10 degrees; every other point starts at level
4 and is promoted only if its point bound or a failed cover leaf requires it.

The two Cartesian grids align with early octree scales.  The normalized
ratio points resolve distinct directions into the fully degenerate corner,
and the boundary points approach `gamma = alpha + beta` without placing a
reference on the singular face.
"""
function star_reference_plan()
    specs = NamedTuple[]
    seen = Set{NTuple{3, UInt64}}()

    for angle in (5.0, 7.5, 10.0)
        _spec!(specs, seen, (angle, angle, angle), 5, nothing, 1,
               :star_small_anchor)
    end

    for values in ((15.0, 30.0, 45.0, 60.0),
                   (7.5, 22.5, 37.5, 52.5))
        role = first(values) == 15.0 ? :star_core : :star_staggered
        priority = first(values) == 15.0 ? 2 : 3
        for i in eachindex(values), j in i:lastindex(values),
            k in j:lastindex(values)
            p = (values[i], values[j], values[k])
            p[1] + p[2] > p[3] || continue
            _spec!(specs, seen, p, 4, 5, priority, role)
        end
    end

    # c is normalized to one.  The first five are a coarse ratio net; the
    # last three lie only 1/16 beyond the planar triangle boundary.
    ratios = ((1/4, 1.0, 1.0),
              (1/2, 3/4, 1.0),
              (1/2, 1.0, 1.0),
              (3/4, 3/4, 1.0),
              (3/4, 1.0, 1.0),
              (1/4, 13/16, 1.0),
              (3/8, 11/16, 1.0),
              (1/2, 9/16, 1.0))
    for ratio in ratios
        _spec!(specs, seen, ntuple(i -> 10.0 * ratio[i], 3),
               4, 5, 4, :star_ratio)
    end

    for alpha in 10.0:10.0:60.0, beta in alpha:10.0:60.0
        gamma = alpha + beta - 2.5
        beta <= gamma <= 60.0 || continue
        _spec!(specs, seen, (alpha, beta, gamma),
               4, 5, 4, :star_planar_boundary)
    end
    return CatalogPlan(_numbered_seeds(:star, specs))
end

function _path_grid!(specs, seen, values, fractions, priority, role)
    for i in eachindex(values), j in i:lastindex(values)
        theta1, theta2 = Float64(values[i]), Float64(values[j])
        cap = _path_cap(theta1, theta2)
        for fraction in fractions
            phi = Float64(fraction) * cap
            _spec!(specs, seen, (theta1, theta2, phi),
                   4, 5, priority, role)
        end
    end
    return specs
end

"""
    path_reference_plan()

Return a chain-reversal-reduced path catalogue.  The path family has ample
margin below 4/25 in the existing computations, so all points start at level
4.  Five dihedral fractions on the coarse face-angle grid resolve both planar
ends and the `|AD|=1` cap; a staggered grid and small-face-angle strip are a
second tranche.  Level 5 is available only as an adaptive fallback.
"""
function path_reference_plan()
    specs = NamedTuple[]
    seen = Set{NTuple{3, UInt64}}()
    _path_grid!(specs, seen, (15.0, 30.0, 45.0, 60.0),
                (1/16, 1/4, 1/2, 3/4, 15/16),
                2, :path_core)
    _path_grid!(specs, seen, (7.5, 22.5, 37.5, 52.5),
                (1/8, 1/2, 7/8),
                3, :path_staggered)
    small_pairs = ((5.0, 5.0), (5.0, 15.0), (5.0, 30.0),
                   (5.0, 45.0), (5.0, 60.0), (10.0, 10.0),
                   (10.0, 30.0), (10.0, 60.0))
    for (theta1, theta2) in small_pairs
        cap = _path_cap(theta1, theta2)
        for fraction in (1/8, 1/2, 7/8)
            _spec!(specs, seen, (theta1, theta2, Float64(fraction) * cap),
                   4, 5, 4, :path_small_angle)
        end
    end
    return CatalogPlan(_numbered_seeds(:path, specs))
end

function full_reference_plan()
    star = star_reference_plan()
    path = path_reference_plan()
    return CatalogPlan(vcat(star.seeds, path.seeds))
end

_rowmajor(A::AbstractMatrix{<:Integer}) =
    ntuple(k -> Int(A[(k - 1) ÷ 3 + 1, (k - 1) % 3 + 1]), 9)

function _matrix(transform::NTuple{9, Int})
    return [transform[3(i - 1) + j] for i in 1:3, j in 1:3]
end

function _permutation_transform(p::NTuple{3, Int})
    P = zeros(Int, 3, 3)
    for i in 1:3
        P[i, p[i]] = 1
    end
    return _rowmajor(P)
end

function _star_pair_angle(parameters, i::Int, j::Int)
    i > j && return _star_pair_angle(parameters, j, i)
    (i, j) == (1, 2) && return parameters[1]
    (i, j) == (1, 3) && return parameters[2]
    (i, j) == (2, 3) && return parameters[3]
    error("invalid star edge pair")
end

function _star_image(seed::ReferenceSeed, p::NTuple{3, Int})
    a = (_star_pair_angle(seed.parameters_deg, p[1], p[2]),
         _star_pair_angle(seed.parameters_deg, p[1], p[3]),
         _star_pair_angle(seed.parameters_deg, p[2], p[3]))
    tag = string(p[1], p[2], p[3])
    return ReferenceImage(string(seed.id, "@star-", tag), seed.id, :star,
                          a, Symbol("basis_", tag),
                          _permutation_transform(p))
end

const _IDENTITY_TRANSFORM = (1, 0, 0, 0, 1, 0, 0, 0, 1)
const _PATH_REVERSAL_TRANSFORM = (0, -1, 1,
                                  0, -1, 0,
                                  1, -1, 0)

function _images(seed::ReferenceSeed)
    if seed.family == :star
        images = ReferenceImage[]
        seen = Set{NTuple{3, UInt64}}()
        for p in _EDGE_PERMUTATIONS
            image = _star_image(seed, p)
            key = _parameter_key(image.parameters_deg)
            key in seen && continue
            push!(seen, key)
            push!(images, image)
        end
        return images
    end

    forward = ReferenceImage(string(seed.id, "@path-forward"), seed.id,
                             :path, seed.parameters_deg, :forward,
                             _IDENTITY_TRANSFORM)
    seed.parameters_deg[1] == seed.parameters_deg[2] && return [forward]
    reversed = (seed.parameters_deg[2], seed.parameters_deg[1],
                seed.parameters_deg[3])
    reverse_image = ReferenceImage(string(seed.id, "@path-reverse"),
                                   seed.id, :path, reversed, :chain_reversal,
                                   _PATH_REVERSAL_TRANSFORM)
    return [forward, reverse_image]
end

"""Expand star permutations and path reversal without requesting new solves."""
function expanded_representations(plan::CatalogPlan)
    images = ReferenceImage[]
    for seed in plan.seeds
        append!(images, _images(seed))
    end
    return images
end

"""Apply the exact integer congruence attached to a representation."""
function transformed_gram(gram::AbstractMatrix, image::ReferenceImage)
    size(gram) == (3, 3) || throw(DimensionMismatch("Gram matrix must be 3 by 3"))
    T = _matrix(image.transform)
    # Generic matrix multiplication mixes bare integer zeros into decorated
    # intervals in IntervalArithmetic.jl and can mark an otherwise rigorous
    # result as non-guaranteed.  Every relabelling coefficient here is
    # exactly -1, 0, or 1, so form the congruence with signed additions only.
    result = Matrix{eltype(gram)}(undef, 3, 3)
    for i in 1:3, j in 1:3
        value = zero(gram[1, 1])
        for a in 1:3, b in 1:3
            sign = T[i, a] * T[j, b]
            iszero(sign) && continue
            value += sign == 1 ? gram[a, b] : -gram[a, b]
        end
        result[i, j] = value
    end
    return result
end

"""
    search_reference_catalog(plan, base_bounds)

Prepare the minimal base catalogue for `CoverSearch.start_search(...;
permutations=true)`.  Star permutations are left to `CoverSearch`; a path
chain reversal changes the base vertex and is added here by an exact integer
congruence.  Neither operation performs a new point solve.
"""
function search_reference_catalog(plan::CatalogPlan,
                                  base_bounds::AbstractDict{String, <:ReferenceBound})
    result = Dict{String, ReferenceBound}()
    for seed in plan.seeds
        haskey(base_bounds, seed.id) ||
            throw(ArgumentError("missing point certificate for seed '$(seed.id)'"))
        ref = base_bounds[seed.id]
        ref.id == seed.id ||
            throw(ArgumentError("reference key '$(seed.id)' disagrees with its id"))
        ref.family == seed.family ||
            throw(ArgumentError("reference '$(seed.id)' has the wrong family"))
        result[ref.id] = ref
        if seed.family == :path && seed.parameters_deg[1] != seed.parameters_deg[2]
            reverse_image = last(_images(seed))
            id = string(seed.id, "@reverse")
            result[id] = ReferenceBound(id, :path,
                                        transformed_gram(ref.gram, reverse_image),
                                        ref.u2_upper)
        end
    end
    return result
end

function plan_summary(plan::CatalogPlan)
    star = count(seed -> seed.family == :star, plan.seeds)
    path = length(plan) - star
    level4 = count(seed -> seed.primary_level == 4, plan.seeds)
    level5 = length(plan) - level4
    fallbacks = count(seed -> seed.fallback_level == 5, plan.seeds)
    images = expanded_representations(plan)
    search_bases = sum(seed.family == :path &&
                       seed.parameters_deg[1] != seed.parameters_deg[2] ? 2 : 1
                       for seed in plan.seeds)
    roles = Pair{Symbol, Int}[]
    for role in sort!(unique(seed.role for seed in plan.seeds); by = String)
        push!(roles, role => count(seed -> seed.role == role, plan.seeds))
    end
    return PlanSummary(length(plan), star, path, level4, level5, fallbacks,
                       length(images), search_bases, 6search_bases, roles)
end

function summary_text(plan::CatalogPlan)
    s = plan_summary(plan)
    io = IOBuffer()
    println(io, "base point solves: ", s.base_solves,
            " (star ", s.star_solves, ", path ", s.path_solves, ")")
    println(io, "first-pass levels: L4 ", s.level4_first,
            ", L5 ", s.level5_first,
            "; conditional L5 promotions ", s.level5_fallbacks)
    println(io, "unique angular symmetry images: ", s.symmetry_images)
    println(io, "search base bounds after path reversal: ", s.search_base_bounds)
    println(io, "raw CoverSearch basis variants: ", s.coversearch_variants)
    println(io, "roles:")
    for (role, count) in s.roles
        println(io, "  ", role, ": ", count)
    end
    return String(take!(io))
end

_dyadic_float(x::Dyadic) = Float64(x.numerator // (BigInt(1) << x.exponent))

function _bounds(box::DyadicBox)
    lo = ntuple(i -> _dyadic_float(box.bounds[i].lo), 3)
    hi = ntuple(i -> _dyadic_float(box.bounds[i].hi), 3)
    return lo, hi
end

_star_feasible(p) = all(p[i] > 0.0 &&
                         p[i] < p[j] + p[k]
                         for (i, j, k) in ((1, 2, 3),
                                          (2, 1, 3),
                                          (3, 1, 2)))

function _star_enrichment_point(box::DyadicBox)
    star_box_excluded(box) &&
        throw(ArgumentError("cannot enrich an exactly excluded star cell"))
    lo, hi = _bounds(box)
    midpoint = ntuple(i -> (lo[i] + hi[i]) / 2, 3)
    _star_feasible(midpoint) && return midpoint

    candidate = collect(hi)
    for (i, j, k) in ((1, 2, 3), (2, 1, 3), (3, 1, 2))
        if candidate[i] >= candidate[j] + candidate[k]
            lo[i] < candidate[j] + candidate[k] ||
                throw(ArgumentError("star cell has no strict feasible interior point"))
            candidate[i] = (lo[i] + candidate[j] + candidate[k]) / 2
            break
        end
    end
    p = (candidate[1], candidate[2], candidate[3])
    _star_feasible(p) ||
        throw(ArgumentError("failed to construct an interior star reference"))
    return p
end

function _path_enrichment_point(box::DyadicBox)
    path_box_excluded(box) &&
        throw(ArgumentError("cannot enrich an interval-excluded path cell"))
    lo, hi = _bounds(box)
    hi[1] > 0.0 && hi[2] > 0.0 && hi[3] > 0.0 ||
        throw(ArgumentError("path cell has no nondegenerate interior point"))

    # Move toward the lower face angles until their diameter cap intersects
    # the requested dihedral interval.  This is deterministic and exploits
    # monotonicity of phi_max in both face angles.
    fractions = (1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128,
                 1/256, 1/512, 1/1024, 1/2048, 1/4096)
    for fraction in fractions
        theta1 = lo[1] + Float64(fraction) * (hi[1] - lo[1])
        theta2 = lo[2] + Float64(fraction) * (hi[2] - lo[2])
        theta1 > 0.0 && theta2 > 0.0 || continue
        cap = _path_cap(theta1, theta2)
        upper = min(hi[3], prevfloat(cap))
        if lo[3] < upper && upper > 0.0
            phi = (max(lo[3], 0.0) + upper) / 2
            candidate = (theta1, theta2, phi)
            phi > 0.0 &&
            Kappa3Certificate.path_unit_diameter_certified(candidate; strict = true) &&
                return candidate
        end
    end

    # A cell may meet the feasible domain only on |AD|=1 at its lower face
    # angles.  References need not lie in the target cell, so retreat by a
    # controlled amount in phi and insist on a rigorous strict-diameter test.
    # This avoids creating a nominal boundary point on the wrong side because
    # of a libm-rounded `acosd` cap.
    theta1, theta2 = lo[1], lo[2]
    if theta1 > 0.0 && theta2 > 0.0
        cap = _path_cap(theta1, theta2)
        upper = min(hi[3], cap)
        if 0.0 < lo[3] <= upper
            boundary_phi = min(lo[3], cap)
            for retreat in (1 - 2.0^-24, 1 - 2.0^-20, 1 - 2.0^-16,
                            1 - 2.0^-12, 1 - 2.0^-8, 0.5)
                phi = boundary_phi * retreat
                candidate = (theta1, theta2, phi)
                phi > 0.0 &&
                Kappa3Certificate.path_unit_diameter_certified(candidate; strict = true) &&
                    return candidate
            end
        end
    end
    throw(ArgumentError("failed to construct an admissible path reference"))
end

"""
    enrichment_seed(family, failed_box, serial; force_level5=false)

Choose a deterministic admissible point from a failed cover cell.  A balanced
small-angle star is sent directly to level 5; all other new references start
at level 4.  Calling `promote_seed` after a failed level-4 self/leaf test is
the only other route to level 5.
"""
function enrichment_seed(family::Symbol, box::DyadicBox, serial::Integer;
                         force_level5::Bool = false)
    serial > 0 || throw(ArgumentError("enrichment serial must be positive"))
    p = if family == :star
        _star_enrichment_point(box)
    elseif family == :path
        _path_enrichment_point(box)
    else
        throw(ArgumentError("family must be :star or :path"))
    end
    balanced_small_star = family == :star && maximum(p) <= 12.5 &&
                          minimum(p) / maximum(p) >= 0.45
    level = force_level5 || balanced_small_star ? 5 : 4
    fallback = level == 4 ? 5 : nothing
    prefix = family == :star ? "SE" : "PE"
    id = string(prefix, lpad(serial, 5, '0'), "-L", level)
    return ReferenceSeed(id, family, p, level, fallback, 5, :adaptive_enrichment)
end

"""Generate and bit-exactly deduplicate enrichment requests."""
function enrichment_seeds(family::Symbol, boxes;
                          first_serial::Int = 1,
                          force_level5::Bool = false,
                          existing_parameters = NTuple{3, Float64}[])
    first_serial > 0 || throw(ArgumentError("first_serial must be positive"))
    seen = Set{NTuple{3, UInt64}}(_parameter_key(ntuple(i -> Float64(p[i]), 3))
                                  for p in existing_parameters)
    result = ReferenceSeed[]
    serial = first_serial
    for box in boxes
        seed = enrichment_seed(family, box, serial; force_level5)
        key = _parameter_key(seed.parameters_deg)
        key in seen && continue
        push!(seen, key)
        push!(result, seed)
        serial += 1
    end
    return result
end

function promote_seed(seed::ReferenceSeed)
    seed.primary_level == 4 && seed.fallback_level == 5 ||
        throw(ArgumentError("seed '$(seed.id)' has no level-5 promotion"))
    id = replace(seed.id, r"-L4$" => "-L5")
    id == seed.id && (id = string(seed.id, "-L5"))
    return ReferenceSeed(id, seed.family, seed.parameters_deg,
                         5, nothing, seed.priority, :level5_promotion)
end

"""
Return true when a level-4 point certificate has too little local margin to
be useful.  `relative_reserve=0.002` keeps a 0.2% reserve for Gram comparison;
a failed leaf may always be promoted explicitly even if this test is false.
"""
function needs_promotion(seed::ReferenceSeed, u2_upper::Real;
                         target::Rational = TARGET_KAPPA,
                         relative_reserve::Real = 0.002)
    seed.primary_level == 4 && seed.fallback_level == 5 || return false
    0.0 <= relative_reserve < 1.0 ||
        throw(ArgumentError("relative_reserve must lie in [0,1)"))
    u2_upper > 0 || throw(DomainError(u2_upper, "u2_upper must be positive"))
    threshold = Float64(target) * (1 - Float64(relative_reserve))
    return sqrt(Float64(u2_upper)) >= threshold
end

end # module ReferencePlan
