using Test
using IntervalArithmetic: interval, inf, sup

if !isdefined(@__MODULE__, :BoxCover)
    include(joinpath(@__DIR__, "..", "src", "BoxCover.jl"))
end
include(joinpath(@__DIR__, "..", "src", "CoverSearch.jl"))
using .BoxCover
using .CoverSearch

function _interval_matrix_endpoints(A)
    return [(inf(A[i, j]), sup(A[i, j])) for i in axes(A, 1), j in axes(A, 2)]
end

@testset "reference permutations and deterministic selection" begin
    target_box = dyadic_box((20, 40, 50), (20, 40, 50))
    target_coeff = star_coefficient_enclosure(target_box)
    target_gram = star_gram(coefficient_corners(target_coeff)[1])
    swap = [2, 1, 3]
    base_gram = target_gram[swap, swap]

    # q is only slightly larger than one.  The matching permuted basis passes,
    # while the deliberately mismatched identity basis does not.
    q = 1.0001
    u2 = Float64(TARGET_KAPPA_SQUARED) / q
    base = ReferenceBound("asymmetric ref", :star, base_gram, u2)
    identity = certify_leaf(:star, target_box, base.gram, base.u2_upper)
    @test !identity.passed

    refs, origins = expanded_reference_catalog(Dict(base.id => base))
    @test length(refs) == 6
    choice = select_reference(:star, target_box, refs, origins)
    @test !isnothing(choice)
    @test choice.result.passed
    @test choice.origin.base_id == base.id
    @test choice.reference_id == "asymmetric ref@basis-213"

    # Reversed dictionary insertion cannot affect the selected certificate.
    reverse_refs = Dict(reverse(collect(refs))...)
    reverse_origins = Dict(reverse(collect(origins))...)
    choice2 = select_reference(:star, target_box, reverse_refs, reverse_origins)
    @test choice2.reference_id == choice.reference_id
end

@testset "star octree and path binary policies" begin
    empty_refs = Dict{String, ReferenceBound}()

    star_root = dyadic_box((10, 10, 10), (20, 20, 20))
    star_state = start_search(:star, empty_refs; root = star_root)
    progress = advance_search!(star_state; max_depth = 1)
    @test progress.processed_now == 1
    @test progress.depth_limited == 8
    @test length(star_state.splits) == 1
    @test star_state.splits["."] == (1, 2, 3)
    @test length(star_state.pending) == 8
    @test_throws ArgumentError materialize_cover(star_state)
    @test validate_search_state(star_state) === star_state

    path_root = dyadic_box((10, 10, 10), (20, 20, 25))
    path_state = start_search(:path, empty_refs; root = path_root)
    progress = advance_search!(path_state; max_depth = 1)
    @test progress.processed_now == 1
    @test progress.depth_limited == 2
    # Normalized widths are 1/6, 1/6, 1/6, so the deterministic tie is axis 1.
    @test path_state.splits["."] == (1,)
    @test length(path_state.pending) == 2

    excluded = dyadic_box((40, 0, 0), (60, 10, 20))
    excluded_state = start_search(:star, empty_refs; root = excluded)
    result = advance_search!(excluded_state; max_depth = 1)
    @test result.complete
    @test excluded_state.leaves["."].status == :excluded
    tree = materialize_cover(excluded_state)
    @test audit_tree(tree, excluded; family = :star).passed
end

@testset "checkpoint and auditable record round trips" begin
    point = dyadic_box((20, 40, 50), (20, 40, 50))
    coeff = star_coefficient_enclosure(point)
    G = star_gram(coefficient_corners(coeff)[1])
    ref = ReferenceBound("unicode-α and space", :star, G, 0.01)

    complete = start_search(:star, Dict(ref.id => ref); root = point,
                            permutations = false)
    progress = advance_search!(complete; max_depth = 1)
    @test progress.complete
    tree = materialize_cover(complete)
    replay = replay_cover(tree, point, complete.references; family = :star)
    @test replay.passed

    mktempdir() do directory
        refs1 = joinpath(directory, "references.txt")
        refs2 = joinpath(directory, "references-copy.txt")
        save_reference_catalog(refs1, complete.references, complete.origins)
        loaded_refs, loaded_origins = load_reference_catalog(refs1)
        save_reference_catalog(refs2, loaded_refs, loaded_origins)
        @test read(refs1, String) == read(refs2, String)
        @test collect(keys(loaded_refs)) == collect(keys(complete.references))
        for id in keys(loaded_refs)
            @test loaded_refs[id].u2_upper === complete.references[id].u2_upper
            @test _interval_matrix_endpoints(loaded_refs[id].gram) ==
                  _interval_matrix_endpoints(complete.references[id].gram)
        end

        cover1 = joinpath(directory, "cover.txt")
        cover2 = joinpath(directory, "cover-copy.txt")
        save_cover_record(cover1, complete)
        loaded_cover = load_cover_record(cover1)
        save_cover_record(cover2, loaded_cover)
        @test read(cover1, String) == read(cover2, String)
        @test loaded_cover.target == TARGET_KAPPA
        @test replay_record(loaded_cover, loaded_refs).passed

        # Exercise an incomplete, resumable checkpoint with exact dyadic boxes.
        partial = start_search(:star, Dict{String, ReferenceBound}();
                               root = dyadic_box((10, 10, 10), (20, 20, 20)))
        advance_search!(partial; max_depth = 1)
        checkpoint1 = joinpath(directory, "checkpoint.txt")
        checkpoint2 = joinpath(directory, "checkpoint-copy.txt")
        save_checkpoint(checkpoint1, partial)
        loaded = load_checkpoint(checkpoint1)
        save_checkpoint(checkpoint2, loaded)
        @test read(checkpoint1, String) == read(checkpoint2, String)
        @test loaded.processed == partial.processed
        @test loaded.target == TARGET_KAPPA
        @test length(loaded.pending) == 8
        @test validate_search_state(loaded) === loaded

        resumed = advance_search!(loaded; max_depth = 2)
        @test resumed.processed_now == 8
        @test resumed.depth_limited == 64
        @test length(loaded.splits) == 9
        @test length(loaded.pending) == 64
    end
end
