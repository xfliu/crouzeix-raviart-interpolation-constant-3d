using Test
using IntervalArithmetic: interval
import Kappa3Certificate

if !isdefined(@__MODULE__, :BoxCover)
    include(joinpath(@__DIR__, "..", "src", "BoxCover.jl"))
end
if !isdefined(@__MODULE__, :CoverSearch)
    include(joinpath(@__DIR__, "..", "src", "CoverSearch.jl"))
end
if !isdefined(@__MODULE__, :ReferencePlan)
    include(joinpath(@__DIR__, "..", "src", "ReferencePlan.jl"))
end
if !isdefined(@__MODULE__, :ProofRecords)
    include(joinpath(@__DIR__, "..", "src", "ProofRecords.jl"))
end
include(joinpath(@__DIR__, "..", "src", "ProductionRunner.jl"))
using .ProductionRunner

@testset "production runner CLI is explicit and bounded" begin
    @test Kappa3Certificate.RigorousSPD.ieee_preflight()
    @test_throws ArgumentError parse_cli(String[])
    @test_throws ArgumentError parse_cli(["unknown"])
    @test_throws ArgumentError parse_cli(["status", "--max-point-solves=1"])
    @test_throws ArgumentError parse_cli(["run", "--unknown=1"])
    @test_throws ArgumentError parse_cli(["run", "--max-point-solves=33"])
    @test_throws ArgumentError parse_cli(["run", "--max-cycles=0"])
    mode, limits = parse_cli(["run", "--max-point-solves=2",
                              "--max-search-nodes=99", "--star-depths=3,5",
                              "--path-depths=8,16"])
    @test mode == :run
    @test limits.max_point_solves == 2
    @test limits.max_search_nodes == 99
    @test limits.star_depths == [3, 5]
    @test limits.path_depths == [8, 16]

    # Completion is a proof claim, so the production API must not expose a
    # keyword that can bypass its mandatory independent replay.
    config = ProductionConfig(code_root = "/does/not/matter")
    @test_throws MethodError run_production(
        config, RunLimits(); require_launcher = false,
        independent_replay = false)
end

@testset "production seed plan is strict and round trips" begin
    baseline = Main.ReferencePlan.full_reference_plan().seeds
    @test length(validate_seed_plan(baseline)) == length(baseline)
    mktempdir() do directory
        path = joinpath(directory, "seeds.txt")
        save_seed_plan(path, baseline)
        loaded = load_seed_plan(path)
        @test length(loaded) == length(baseline)
        @test read(path, String) == begin
            copy_path = joinpath(directory, "copy.txt")
            save_seed_plan(copy_path, loaded)
            read(copy_path, String)
        end
        tampered = replace(read(path, String), "KAPPA3_PRODUCTION_SEEDS_V1" =>
                                               "KAPPA3_PRODUCTION_SEEDS_V0")
        write(joinpath(directory, "bad.txt"), tampered)
        @test_throws ArgumentError load_seed_plan(joinpath(directory, "bad.txt"))
    end
end

@testset "catalog changes force deterministic restart" begin
    ref = Main.BoxCover.ReferenceBound("r", :star,
                                       [interval(1.0) interval(0.5) interval(0.5);
                                        interval(0.5) interval(1.0) interval(0.5);
                                        interval(0.5) interval(0.5) interval(1.0)],
                                       0.02)
    first_state = Main.CoverSearch.start_search(:star, [ref])
    same_state = Main.CoverSearch.start_search(:star, [ref])
    @test same_search_catalog(first_state, same_state)
    ref2 = Main.BoxCover.ReferenceBound("s", :star, ref.gram, 0.021)
    expanded = Main.CoverSearch.start_search(:star, [ref, ref2])
    @test !same_search_catalog(first_state, expanded)
end

@testset "pending scheduling is deterministic" begin
    # With no useful references, a one-level star pass leaves eight exact
    # pending cells.  Enrichment order must not depend on dictionary order.
    weak = Main.BoxCover.ReferenceBound("weak", :star,
                                        [interval(1.0) interval(0.0) interval(0.0);
                                         interval(0.0) interval(1.0) interval(0.0);
                                         interval(0.0) interval(0.0) interval(1.0)],
                                        1.0)
    state = Main.CoverSearch.start_search(:star, [weak])
    Main.CoverSearch.advance_search!(state; max_depth = 1,
                                     max_new_nodes = 100)
    ordered = pending_cells(state)
    @test issorted([(cell.depth, cell.path) for cell in ordered])
    seeds = Main.ReferencePlan.full_reference_plan().seeds
    enriched1 = pending_enrichments(:star, state, seeds; limit = 2)
    enriched2 = pending_enrichments(:star, state, reverse(seeds); limit = 2)
    @test getfield.(enriched1, :id) == getfield.(enriched2, :id)
    @test getfield.(enriched1, :parameters_deg) ==
          getfield.(enriched2, :parameters_deg)

    # A point solve also supplies its star permutations or path reversal.
    # Promotion scheduling must therefore notice failed cells containing an
    # image even when the seed's stored parameter ordering is elsewhere.
    # The catalogue is all level 5, so promotion is exercised on explicit
    # level-4 seeds appended to it.
    star_seed = Main.ReferencePlan.ReferenceSeed(
        "S-L4", :star, (15.0, 30.0, 30.0), 4, 5, 9, :test)
    path_seed = Main.ReferencePlan.ReferenceSeed(
        "P-L4", :path, (15.0, 30.0, 40.0), 4, 5, 9, :test)
    seeds = vcat(seeds, [star_seed, path_seed])
    state.pending = Main.CoverSearch.SearchCell[
        Main.CoverSearch.SearchCell(
            "image", Main.BoxCover.dyadic_box((29, 29, 14), (31, 31, 16)), 1)
    ]
    star_promotions = pending_promotions(
        :star, state, seeds, [star_seed.id]; limit = 1)
    @test length(star_promotions) == 1
    @test star_promotions[1].parameters_deg == star_seed.parameters_deg

    state.family = :path
    state.pending = Main.CoverSearch.SearchCell[
        Main.CoverSearch.SearchCell(
            "reverse", Main.BoxCover.dyadic_box((29, 14, 0), (31, 16, 90)), 1)
    ]
    path_promotions = pending_promotions(
        :path, state, seeds, [path_seed.id]; limit = 1)
    @test length(path_promotions) == 1
    @test path_promotions[1].parameters_deg == path_seed.parameters_deg
end
