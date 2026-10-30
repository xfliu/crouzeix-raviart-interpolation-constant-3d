using Test
using IntervalArithmetic: interval, mid
using Kappa3Certificate

if !isdefined(@__MODULE__, :BoxCover)
    include(joinpath(@__DIR__, "..", "src", "BoxCover.jl"))
end
include(joinpath(@__DIR__, "..", "src", "ReferencePlan.jl"))
using .BoxCover
using .ReferencePlan

function _path_point_gram(parameters)
    theta1, theta2, phi = parameters
    x, y = cosd(theta1), cosd(theta2)
    z = x * y - sind(theta1) * sind(theta2) * cosd(phi)
    return BoxCover.path_vertex_gram((x, y, z))
end

@testset "deterministic reference plans" begin
    star = star_reference_plan()
    path = path_reference_plan()
    full = full_reference_plan()
    @test length(star) == 73
    @test length(path) == 137
    @test length(full) == 210

    summary = plan_summary(full)
    @test summary.star_solves == 73
    @test summary.path_solves == 137
    @test summary.level4_first == 0
    @test summary.level5_first == 210
    @test summary.level5_fallbacks == 0
    @test summary.symmetry_images == 522
    @test summary.search_base_bounds == 319
    @test summary.coversearch_variants == 1914
    @test occursin("base point solves: 210", summary_text(full))

    star_again = star_reference_plan()
    @test [s.id for s in star] == [s.id for s in star_again]
    @test [s.parameters_deg for s in star] ==
          [s.parameters_deg for s in star_again]

    @test all(s.primary_level == 5 && isnothing(s.fallback_level)
              for s in full)
    @test first(star).parameters_deg == (2.5, 2.5, 2.5)
    @test all(s.parameters_deg[1] <= s.parameters_deg[2] for s in path)
    @test all(Kappa3Certificate.path_unit_diameter_certified(s.parameters_deg)
              for s in path)
    @test all(Kappa3Certificate.path_unit_diameter_certified(
                  s.parameters_deg; strict = true) for s in path)
    @test count(s -> s.role == :star_level5, star) == 73
    @test count(s -> s.role == :path_level5, path) == 137
end

@testset "no-solve relabellings" begin
    star_seed = ReferenceSeed("star", :star, (20.0, 35.0, 45.0),
                              4, 5, 1, :test)
    star_images = expanded_representations(CatalogPlan([star_seed]))
    @test length(star_images) == 6
    Gs = star_gram((cosd(20.0), cosd(35.0), cosd(45.0)))
    for image in star_images
        transformed = transformed_gram(Gs, image)
        expected = star_gram(cosd.(image.parameters_deg))
        @test all(isapprox(mid(transformed[i, j]), mid(expected[i, j]);
                           atol = 2e-15, rtol = 2e-15)
                  for i in 1:3, j in 1:3)
    end

    path_seed = ReferenceSeed("path", :path, (20.0, 50.0, 30.0),
                              4, 5, 1, :test)
    path_images = expanded_representations(CatalogPlan([path_seed]))
    @test length(path_images) == 2
    H = _path_point_gram(path_seed.parameters_deg)
    reverse_image = only(image for image in path_images
                         if image.relabelling == :chain_reversal)
    Hreverse = _path_point_gram((50.0, 20.0, 30.0))
    transformed = transformed_gram(H, reverse_image)
    @test all(isapprox(mid(transformed[i, j]), mid(Hreverse[i, j]);
                       atol = 4e-15, rtol = 4e-15)
              for i in 1:3, j in 1:3)

    symmetric = ReferenceSeed("symmetric path", :path, (30.0, 30.0, 20.0),
                              4, 5, 1, :test)
    @test length(expanded_representations(CatalogPlan([symmetric]))) == 1

    base = Dict("star" => ReferenceBound("star", :star, Gs, 0.01),
                "path" => ReferenceBound("path", :path, H, 0.01))
    search = search_reference_catalog(CatalogPlan([star_seed, path_seed]), base)
    @test Set(keys(search)) == Set(["star", "path", "path@reverse"])
    @test search["path@reverse"].u2_upper == search["path"].u2_upper
end

@testset "adaptive enrichment and promotion" begin
    central = dyadic_box((0, 0, 0), (15, 15, 15))
    seed = enrichment_seed(:star, central, 1)
    @test seed.primary_level == 5
    @test seed.parameters_deg == (7.5, 7.5, 7.5)

    boundary = dyadic_box((40, 10, 10), (60, 30, 30))
    boundary_seed = enrichment_seed(:star, boundary, 2)
    a, b, c = boundary_seed.parameters_deg
    @test a < b + c && b < a + c && c < a + b
    @test boundary_seed.primary_level == 5
    @test boundary_seed.id == "SE00002-L5"

    path_seed = enrichment_seed(:path, path_root_box(), 1)
    @test path_seed.primary_level == 5
    @test isnothing(path_seed.fallback_level)
    theta1, theta2, phi = path_seed.parameters_deg
    @test 0 < theta1 <= 60 && 0 < theta2 <= 60 && 0 < phi < 90
    @test Kappa3Certificate.path_unit_diameter_certified(
        path_seed.parameters_deg; strict = true)

    @test_throws ArgumentError ReferenceSeed(
        "invalid path", :path, (60.0, 60.0, 90.0), 4, 5, 1, :test)

    # Promotion remains available for explicitly level-4 seeds.
    level4 = ReferenceSeed("P-L4", :path, path_seed.parameters_deg,
                           4, 5, 1, :test)
    promoted = promote_seed(level4)
    @test promoted.primary_level == 5
    @test promoted.parameters_deg == level4.parameters_deg
    @test promoted.role == :level5_promotion
    @test needs_promotion(level4, (0.1597)^2)
    @test !needs_promotion(level4, (0.150)^2)
    @test !needs_promotion(promoted, (0.1597)^2)
    @test !needs_promotion(path_seed, (0.1597)^2)

    duplicates = enrichment_seeds(:star, [central, central])
    @test length(duplicates) == 1
    @test_throws ArgumentError enrichment_seed(
        :star, dyadic_box((40, 0, 0), (60, 10, 20)), 3)
    @test_throws ArgumentError enrichment_seed(
        :path, dyadic_box((50, 50, 80), (60, 60, 90)), 3)
end

@testset "relabelled Gram enclosures are exactly symmetric" begin
    # Regression: the 2026-10-29 campaign stopped because T*G*T' summed
    # (i,j) and (j,i) in different orders, giving one-ulp asymmetry that
    # BoxCover.strict_pd3 rejects.
    plan = full_reference_plan()
    byid = Dict(s.id => s for s in plan.seeds)
    for image in expanded_representations(plan)
        seed = byid[image.base_id]
        G = seed.family == :star ?
            Kappa3Certificate.star_vertex_gram(seed.parameters_deg) :
            Kappa3Certificate.path_vertex_gram(seed.parameters_deg)
        H = transformed_gram(G, image)
        for i in 1:3, j in (i + 1):3
            @test isequal(BoxCover.inf(H[i, j]), BoxCover.inf(H[j, i])) &&
                  isequal(BoxCover.sup(H[i, j]), BoxCover.sup(H[j, i]))
        end
        @test BoxCover.strict_pd3(H)
    end
end
