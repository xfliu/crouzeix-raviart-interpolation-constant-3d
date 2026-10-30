# Pilot: adaptive star or path cover at a chosen target, using reference
# bounds from pilot_points.jl output files.  This is a feasibility
# diagnostic, not a proof record.
#
#   julia --project=Code pilot_cover.jl HI MAXDEPTH MAXNODES REFS.tsv...
#
# PILOT_FAMILY=star (default): root [0,HI]^3; references with all angles
# <= PILOT_REFMAX (default 2*HI+5).  PILOT_FAMILY=path: the full root
# [0,60]x[0,60]x[0,90] (HI ignored); each reference is added together with
# its exact chain reversal.  Only references with C_upper below the target
# (PILOT_TARGET, e.g. "1598/10000"; default 4/25) are used.

using Printf
using Kappa3Certificate

code_root = normpath(joinpath(dirname(pathof(Kappa3Certificate)), ".."))
include(joinpath(code_root, "src", "BoxCover.jl"))
include(joinpath(code_root, "src", "CoverSearch.jl"))
include(joinpath(code_root, "src", "ReferencePlan.jl"))
using .BoxCover
using .CoverSearch
using .ReferencePlan

family = Symbol(get(ENV, "PILOT_FAMILY", "star"))
hi = parse(Int, ARGS[1])
max_depth = parse(Int, ARGS[2])
max_nodes = parse(Int, ARGS[3])
refmax = parse(Float64, get(ENV, "PILOT_REFMAX", string(2hi + 5)))
target = let t = split(get(ENV, "PILOT_TARGET", "4/25"), '/')
    parse(Int, t[1]) // parse(Int, t[2])
end
println("family: ", family, "  target: ", target, " = ", Float64(target))

seeds = ReferenceSeed[]
base = Dict{String, ReferenceBound}()
for file in ARGS[4:end], line in eachline(file)
    (isempty(line) || startswith(line, "#")) && continue
    f = split(line, '\t')
    f[9] == "ok" || continue
    fam = length(f) >= 10 ? Symbol(f[10]) : :star
    fam == family || continue
    parse(Float64, f[7]) < target || continue
    p = (parse(Float64, f[2]), parse(Float64, f[3]), parse(Float64, f[4]))
    family == :star && maximum(p) > refmax && continue
    u2 = reinterpret(Float64, parse(UInt64, f[6][3:end]; base = 16))
    id = string(f[1], "-L", f[5])
    gram = family == :star ? Kappa3Certificate.star_vertex_gram(p) :
                             Kappa3Certificate.path_vertex_gram(p)
    push!(seeds, ReferenceSeed(id, family, p, parse(Int, f[5]), nothing, 1, :pilot))
    base[id] = ReferenceBound(id, family, gram, u2)
end
refs = search_reference_catalog(CatalogPlan(seeds), base)
println("references used: ", length(seeds), " (search bases ", length(refs), ")")
flush(stdout)

root = family == :star ? dyadic_box((0, 0, 0), (hi, hi, hi)) : path_root_box()
state = start_search(family, refs; root, target)
t0 = time()
for depth in 1:max_depth
    p = advance_search!(state; max_new_nodes = max_nodes - state.processed,
                        max_depth = depth)
    @printf("depth<%d processed=%d pending=%d retained=%d excluded=%d t=%.0fs\n",
            depth, p.processed_total, p.pending, p.retained, p.excluded, time() - t0)
    flush(stdout)
    (search_complete(state) || state.processed >= max_nodes) && break
end

println(search_complete(state) ? "PILOT COVER COMPLETE" : "PILOT COVER INCOMPLETE")
widths = Dict{Float64, Int}()
for leaf in values(state.leaves)
    leaf.status == :retained || continue
    b = leaf.box.bounds[1]
    w = Float64((b.hi.numerator // (BigInt(1) << b.hi.exponent)) -
                (b.lo.numerator // (BigInt(1) << b.lo.exponent)))
    widths[w] = get(widths, w, 0) + 1
end
println("retained leaves by side (deg): ", sort(collect(widths); rev = true))
used = sort!(unique(state.origins[leaf.reference_id].base_id
                    for leaf in values(state.leaves) if leaf.status == :retained))
println("distinct base references used: ", length(used))
println("used: ", join(used, " "))
if !search_complete(state)
    println("sample pending boxes:")
    for cell in Iterators.take(reverse(state.pending), 12)
        println("  ", cell.box)
    end
end
