using Kappa3Certificate

code_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(code_root, "src", "BoxCover.jl"))
include(joinpath(code_root, "src", "CoverSearch.jl"))
include(joinpath(code_root, "src", "ReferencePlan.jl"))

using .BoxCover
using .CoverSearch
using .ReferencePlan

family = Symbol(get(ENV, "KAPPA3_DIAG_FAMILY", "star"))
family in (:star, :path) || error("KAPPA3_DIAG_FAMILY must be star or path")
u2 = parse(Float64, get(ENV, "KAPPA3_DIAG_U2", "0.02545"))
max_depth = parse(Int, get(ENV, "KAPPA3_DIAG_MAX_DEPTH",
                           family == :star ? "7" : "24"))
max_nodes = parse(Int, get(ENV, "KAPPA3_DIAG_MAX_NODES", "200000"))

full = full_reference_plan()
plan = CatalogPlan([seed for seed in full.seeds if seed.family == family])
base = Dict{String, ReferenceBound}()
for seed in plan.seeds
    gram = seed.family == :star ?
           star_vertex_gram(seed.parameters_deg) :
           Kappa3Certificate.path_vertex_gram(seed.parameters_deg)
    base[seed.id] = ReferenceBound(seed.id, seed.family, gram, u2)
end

search_base = search_reference_catalog(plan, base)
state = start_search(family, search_base)
progress = advance_search!(state; max_new_nodes = max_nodes,
                            max_depth = max_depth)
@show family u2 max_depth max_nodes
@show progress

if !search_complete(state)
    depths = [cell.depth for cell in state.pending]
    println("pending depth range: ", extrema(depths))
    println("first pending boxes:")
    for cell in Iterators.take(reverse(state.pending), 10)
        println("  ", cell.path, " ", cell.box)
    end
end
