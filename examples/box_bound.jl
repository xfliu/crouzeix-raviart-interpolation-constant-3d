# Upper bound for C(K) on every shape in one parameter box, using the stored
# reference bounds of the proof (records/campaign/points).
#
#   julia --project=code examples/box_bound.jl FAMILY lo1 hi1 lo2 hi2 lo3 hi3
#
# FAMILY is star or path; the six numbers are the box endpoints in degrees
# (star: alpha, beta, gamma; path: theta1, theta2, phi).  Endpoints must be
# dyadic rationals such as 7.5, 3.75 or 1.875.  The script finds, to six
# decimals, the smallest bound t such that one stored reference passes the
# eight interval corner tests of the box at target t.
#
# Optional environment variable: BOX_REFERENCE=S001 restricts the search to
# one reference (and its relabellings).
#
# The reference bounds are read from the stored point records; their
# eigenvalue certificates are not recomputed here (scripts/replay_proof.jl
# does that).  The corner tests themselves are rigorous.

using Printf
using Kappa3Certificate

const ROOT = normpath(joinpath(@__DIR__, ".."))
code_root = joinpath(ROOT, "code")
include(joinpath(code_root, "src", "BoxCover.jl"))
include(joinpath(code_root, "src", "CoverSearch.jl"))
include(joinpath(code_root, "src", "ReferencePlan.jl"))
using .BoxCover
using .CoverSearch
using .ReferencePlan

length(ARGS) == 7 || error("usage: box_bound.jl FAMILY lo1 hi1 lo2 hi2 lo3 hi3")
family = Symbol(ARGS[1])
family in (:star, :path) || error("FAMILY must be star or path")
ends = [Rational{BigInt}(parse(Float64, a)) for a in ARGS[2:7]]
box = dyadic_box((ends[1], ends[3], ends[5]), (ends[2], ends[4], ends[6]))
only_ref = get(ENV, "BOX_REFERENCE", "")

hexfloat(text) = reinterpret(Float64, parse(UInt64, split(text, '@')[2][3:end]; base = 16))

# Read the stored reference shapes of this family.
seeds = ReferenceSeed[]
base = Dict{String, ReferenceBound}()
pointdir = joinpath(ROOT, "records", "campaign", "points")
for file in sort(readdir(pointdir))
    endswith(file, ".point") || continue
    fields = Dict{String, String}()
    for line in eachline(joinpath(pointdir, file))
        parts = split(line, '\t'; limit = 2)
        length(parts) == 2 && (fields[parts[1]] = parts[2])
    end
    Symbol(fields["family"]) == family || continue
    id = fields["id"]
    isempty(only_ref) || id == only_ref || continue
    p = Tuple(hexfloat.(split(fields["parameters_deg"], ',')))
    u2 = hexfloat(fields["u2_upper"])
    gram = family == :star ? Kappa3Certificate.star_vertex_gram(p) :
                             Kappa3Certificate.path_vertex_gram(p)
    push!(seeds, ReferenceSeed(id, family, p, 5, nothing, 1, :stored))
    base[id] = ReferenceBound(id, family, gram, u2)
end
isempty(seeds) && error("no stored $family reference matches")
refs, origins = expanded_reference_catalog(
    search_reference_catalog(CatalogPlan(seeds), base))
@printf("%s box %s\n", family, string(box))
println("stored references: ", length(seeds), " (with relabellings: ", length(refs), ")")

excluded = family == :star ? star_box_excluded(box) : path_box_excluded(box)
if excluded
    println("The box contains no nondegenerate shape of this family.")
    exit(0)
end

passes(t) = select_reference(family, box, refs, origins; target = t)

# Bisection on the target over the grid k/10^6.
scale = 10^6
lo, hi = 100_000, 200_000            # targets 0.1 and 0.2
choice = passes(BigInt(hi) // scale)
if isnothing(choice)
    println("No stored reference certifies a bound below 0.2 on this box; split it.")
    exit(0)
end
isnothing(passes(BigInt(lo) // scale)) ||
    error("unexpected: the box passes already at 0.1")
while hi - lo > 1
    mid = (lo + hi) ÷ 2
    c = passes(BigInt(mid) // scale)
    if isnothing(c)
        global lo = mid
    else
        global hi = mid
        global choice = c
    end
end
origin = choice.origin
@printf("C(K) <= %.6f for every feasible shape in the box\n", hi / scale)
@printf("certified with reference %s (relabelling %s)\n",
        origin.base_id, join(origin.permutation, ""))
