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

# The catalogues below are the references used by the 2026-10-29 level-5
# pilot covers (LowerBound/scripts/pilot_cover.jl): the full star domain at
# STAR_TARGET and the full path domain at PATH_TARGET.  The pilot's leaf
# tests are those of this campaign, so the same references are expected to
# close both covers; adaptive enrichment remains available if a recomputed
# point bound differs in its last bits.
const _STAR_CATALOGUE = (
    (2.5, 2.5, 2.5), (3, 3, 4), (4, 7, 7), (7, 7, 8),
    (7, 9, 9), (9, 9, 9), (8, 9, 12), (8, 13, 13),
    (10, 10, 12), (10, 13, 13), (13, 13, 13), (9, 17.5, 17.5),
    (10, 17.5, 20), (11, 12, 17.5), (11, 13, 17.5), (11, 14, 20),
    (11, 17.5, 17.5), (12, 14, 17.5), (12, 17.5, 20), (14, 14, 17.5),
    (14, 14, 20), (14, 17.5, 17.5), (14, 17.5, 25), (15, 17.5, 20),
    (15, 25, 25), (17.5, 17.5, 17.5), (17.5, 17.5, 20), (17.5, 17.5, 25),
    (17.5, 20, 25), (17.5, 35, 35), (20, 25, 25), (20, 27.5, 35),
    (20, 35, 35), (20, 35, 40), (20, 40, 40), (20, 45, 45),
    (22.5, 27.5, 35), (22.5, 27.5, 40), (22.5, 35, 40), (22.5, 40, 40),
    (22.5, 40, 45), (22.5, 45, 45), (25, 25, 25), (25, 25, 30),
    (25, 35, 45), (25, 40, 45), (27.5, 27.5, 35), (27.5, 27.5, 40),
    (27.5, 27.5, 45), (27.5, 35, 35), (27.5, 35, 40), (27.5, 40, 40),
    (30, 35, 45), (30, 40, 45), (30, 50, 60), (35, 35, 35),
    (35, 35, 40), (35, 35, 45), (35, 40, 40), (35, 45, 50),
    (35, 60, 60), (40, 40, 50), (40, 40, 60), (40, 45, 50),
    (40, 50, 60), (40, 60, 60), (50, 50, 50), (50, 50, 60),
    (50, 60, 60), (55, 55, 55), (55, 55, 60), (55, 60, 60),
    (60, 60, 60),
)

const _PATH_CATALOGUE = (
    (15, 45, 81.444357729049571), (15, 50, 81.075343495547315),
    (15, 60, 82.964492056536628), (20, 30, 81.836209642130228),
    (20, 35, 75.961338901512235), (20, 35, 81.387148823048832),
    (20, 40, 75.530311149682092), (20, 45, 80.448334838645636),
    (20, 50, 79.953437269102508), (20, 50, 82.618551844739258),
    (20, 55, 79.437568313862144), (20, 55, 82.085487257657547),
    (20, 60, 81.527121004237188), (25, 25, 76.285004329661191),
    (25, 30, 81.182305632502803), (25, 30, 83.888382486919554),
    (25, 35, 75.24278392868149), (25, 35, 80.617268495015892),
    (25, 35, 83.304510778183086), (25, 40, 74.700287466048209),
    (25, 40, 80.036022285051644), (25, 45, 74.139770958290569),
    (25, 45, 79.435468883882763), (25, 45, 82.083317846678852),
    (25, 50, 78.812137904402064), (25, 50, 81.439209167882126),
    (25, 55, 78.162096486990492), (25, 55, 80.767499703223507),
    (25, 60, 72.315448021136959), (25, 60, 77.480837165503885),
    (25, 60, 80.063531737687356), (30, 30, 80.515130371649832),
    (30, 30, 83.198968050704835), (30, 35, 79.831540566796704),
    (30, 35, 82.492591919023269), (30, 40, 79.128096355417966),
    (30, 40, 81.76569956726523), (30, 45, 78.400984229644607),
    (30, 45, 81.014350370632769), (30, 50, 77.645923804887687),
    (30, 50, 80.234121265050618), (30, 55, 61.486442507782208),
    (30, 55, 71.734182925745912), (30, 55, 76.858053134727754),
    (30, 55, 79.419988239218682), (30, 60, 60.825428340141571),
    (30, 60, 70.962999730165166), (30, 60, 76.031785425176963),
    (30, 60, 78.566178272682862), (35, 35, 79.026198556198395),
    (35, 35, 81.660405174738344), (35, 40, 72.983971976950571),
    (35, 40, 78.197112832447033), (35, 40, 80.803683260195271),
    (35, 45, 77.339695149743406), (35, 45, 79.917684988068189),
    (35, 50, 76.448782581897973), (35, 50, 78.997075334627908),
    (35, 55, 60.414795150902336), (35, 55, 70.483927676052716),
    (35, 55, 75.518493938627913), (35, 55, 78.035777069915511),
    (35, 60, 49.694699491828743), (35, 60, 59.633639390194496),
    (35, 60, 69.572579288560249), (35, 60, 74.542049237743115),
    (35, 60, 77.026784212334562), (40, 40, 77.238189413409671),
    (40, 40, 79.812795727189993), (40, 45, 76.24588706788218),
    (40, 45, 78.787416636811585), (40, 50, 70.199795572623174),
    (40, 50, 75.214066684953409), (40, 50, 77.721202241118519),
    (40, 55, 49.423806995296829), (40, 55, 59.3085683943562),
    (40, 55, 69.193329793415558), (40, 55, 74.13571049294525),
    (40, 55, 76.60690084271009), (40, 60, 48.66846384713962),
    (40, 60, 58.402156616567538), (40, 60, 68.135849385995471),
    (40, 60, 73.002695770709423), (40, 60, 75.436118963066406),
    (45, 45, 70.105628296131187), (45, 45, 75.113173174426279),
    (45, 45, 77.616945613573819), (45, 50, 69.005363171079594),
    (45, 50, 73.934317683299568), (45, 50, 76.398794939409555),
    (45, 55, 48.467342520373691), (45, 55, 58.160811024448435),
    (45, 55, 67.854279528523165), (45, 55, 72.701013780560544),
    (45, 55, 75.124380906579219), (45, 60, 38.081919760371761),
    (45, 60, 47.602399700464701), (45, 60, 57.122879640557642),
    (45, 60, 66.643359580650582), (45, 60, 71.403599550697052),
    (45, 60, 73.78371953572028), (50, 50, 58.080838145612702),
    (50, 50, 67.76097783654815), (50, 50, 72.601047682015874),
    (50, 50, 75.021082604749736), (50, 55, 37.975712998870172),
    (50, 55, 47.469641248587713), (50, 55, 56.963569498305262),
    (50, 55, 66.457497748022803), (50, 55, 71.204461872881566),
    (50, 55, 73.577943935310955), (50, 60, 37.190986370986757),
    (50, 60, 46.488732963733447), (50, 60, 55.786479556480131),
    (50, 60, 65.084226149226822), (50, 60, 69.733099445600175),
    (50, 60, 72.057536093786837), (55, 55, 37.13840533697693),
    (55, 55, 46.42300667122116), (55, 55, 55.707608005465396),
    (55, 55, 64.992209339709632), (55, 55, 69.634510006831746),
    (55, 55, 71.955660340392797), (55, 60, 36.254694022180182),
    (55, 60, 45.318367527725229), (55, 60, 54.382041033270269),
    (55, 60, 63.445714538815317), (55, 60, 67.977551291587844),
    (55, 60, 70.243469667974097), (60, 60, 17.632194841377327),
    (60, 60, 26.448292262065991), (60, 60, 35.264389682754654),
    (60, 60, 44.080487103443318), (60, 60, 52.896584524131981),
    (60, 60, 61.712681944820645), (60, 60, 66.120730655164976),
    (60, 60, 68.324755010337142),
)

"""
    star_reference_plan()

Return the deterministic star catalogue: every point is a level-5 reference.
Level-4 references exceed the target near the degenerate corner, and level 6
gives weaker certified bounds, so neither is requested.
"""
function star_reference_plan()
    specs = NamedTuple[]
    seen = Set{NTuple{3, UInt64}}()
    for p in _STAR_CATALOGUE
        _spec!(specs, seen, p, 5, nothing, 1, :star_level5)
    end
    return CatalogPlan(_numbered_seeds(:star, specs))
end

"""
    path_reference_plan()

Return the deterministic chain-reversal-reduced path catalogue, all level 5.
Each non-symmetric point also serves through its exact chain reversal.
"""
function path_reference_plan()
    specs = NamedTuple[]
    seen = Set{NTuple{3, UInt64}}()
    for p in _PATH_CATALOGUE
        _spec!(specs, seen, p, 5, nothing, 2, :path_level5)
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
    # Exact T*G*T' is symmetric: compute the upper triangle once and mirror,
    # so differing summation orders cannot break interval symmetry.
    for i in 1:3, j in i:3
        value = zero(gram[1, 1])
        for a in 1:3, b in 1:3
            sign = T[i, a] * T[j, b]
            iszero(sign) && continue
            value += sign == 1 ? gram[a, b] : -gram[a, b]
        end
        result[i, j] = value
        result[j, i] = value
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

Choose a deterministic admissible point from a failed cover cell.  Every new
reference is a level-5 solve, matching the level-5 catalogues;
`force_level5` is kept for call compatibility.
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
    prefix = family == :star ? "SE" : "PE"
    id = string(prefix, lpad(serial, 5, '0'), "-L5")
    return ReferenceSeed(id, family, p, 5, nothing, 5, :adaptive_enrichment)
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
