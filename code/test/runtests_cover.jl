using Test
using IntervalArithmetic: interval, inf, sup

include(joinpath(@__DIR__, "..", "src", "BoxCover.jl"))
using .BoxCover

@testset "exact dyadic geometry" begin
    @test TARGET_KAPPA == 799 // 5000
    @test TARGET_KAPPA_SQUARED == TARGET_KAPPA^2
    @test STAR_TARGET == TARGET_KAPPA
    @test PATH_TARGET == 31 // 200
    @test family_target(:star) == STAR_TARGET
    @test family_target(:path) == PATH_TARGET
    @test_throws ArgumentError family_target(:other)
    @test Dyadic(2, 2) == Dyadic(1, 1)
    @test Dyadic(3, 1) + Dyadic(1, 1) == Dyadic(2)
    @test midpoint(Dyadic(0), Dyadic(60)) == Dyadic(30)
    @test_throws ArgumentError Dyadic(1 // 3)

    root = dyadic_box((0, 0, 0), (60, 60, 60))
    lo, hi = split_box(root, 1)
    @test lo.bounds[1] == DyadicInterval(0, 30)
    @test hi.bounds[1] == DyadicInterval(30, 60)
    octants = split_boxes(root, (1, 2, 3))
    @test length(octants) == 8
    @test octants[1] == dyadic_box((0, 0, 0), (30, 30, 30))
    @test octants[8] == dyadic_box((30, 30, 30), (60, 60, 60))
end

@testset "star enclosure and exclusion" begin
    box = dyadic_box((30, 40, 50), (40, 50, 60))
    coeff = star_coefficient_enclosure(box)
    for theta in ((30.0, 40.0, 50.0), (35.0, 45.0, 55.0), (40.0, 50.0, 60.0))
        c = cosd.(theta)
        @test all(coeff.lo[i] <= c[i] <= coeff.hi[i] for i in 1:3)
    end
    @test length(coefficient_corners(coeff)) == 8

    @test star_box_excluded(dyadic_box((40, 0, 0), (60, 10, 20)))
    @test !star_box_excluded(dyadic_box((20, 20, 20), (30, 30, 30)))
    # Equality is correctly excluded because the feasible triangle inequality
    # is strict.
    @test star_box_excluded(dyadic_box((40, 10, 10), (50, 20, 20)))
end

@testset "path enclosure and exclusion" begin
    box = dyadic_box((40, 40, 10), (50, 50, 30))
    coeff = path_coefficient_enclosure(box)
    @test !isnothing(coeff)
    for theta in ((40.0, 40.0, 10.0), (45.0, 45.0, 20.0), (50.0, 50.0, 30.0))
        x, y = cosd(theta[1]), cosd(theta[2])
        z = x * y - sind(theta[1]) * sind(theta[2]) * cosd(theta[3])
        if z <= x + y - 1 # only feasible images are claimed by the enclosure
            @test coeff.lo[1] <= x <= coeff.hi[1]
            @test coeff.lo[2] <= y <= coeff.hi[2]
            @test coeff.lo[3] <= z <= coeff.hi[3]
        end
    end

    @test path_box_excluded(dyadic_box((50, 50, 80), (60, 60, 90)))
    @test !path_box_excluded(path_root_box())
end

@testset "strict 3 by 3 interval PD" begin
    I3 = [interval(i == j ? 1.0 : 0.0) for i in 1:3, j in 1:3]
    result = strict_pd3_result(I3)
    @test result.passed
    @test all(>(0.0), result.leading_minor_lower)

    uncertain = copy(I3)
    uncertain[3, 3] = interval(-0.1, 1.0)
    @test !strict_pd3(uncertain)

    nonsymmetric = copy(I3)
    nonsymmetric[1, 2] = interval(0.1)
    @test_throws ArgumentError strict_pd3(nonsymmetric)

    @test_throws ArgumentError ReferenceBound("bad", :star, uncertain, 0.01)
end

@testset "exact-target corner certificate" begin
    point = dyadic_box((60, 60, 60), (60, 60, 60))
    coeff = star_coefficient_enclosure(point)
    c = coefficient_corners(coeff)[1]
    G0 = star_gram(c)

    pass = certify_leaf(:star, point, G0, 0.01)
    @test pass.passed
    @test !pass.vacuous
    @test length(pass.corners) == 8
    @test pass.q_allow * 0.01 <= Float64(TARGET_KAPPA_SQUARED)

    fail = certify_leaf(:star, point, G0, 0.03)
    @test !fail.passed
end

@testset "tree validation and replay" begin
    root = dyadic_box((60, 60, 60), (60, 60, 60))
    coeff = star_coefficient_enclosure(root)
    G0 = star_gram(coefficient_corners(coeff)[1])
    leaf = CoverLeaf(root, :retained, "regular")
    audit = validate_tree(leaf, root; family = :star)
    @test audit.retained == 1

    refs = Dict("regular" => ReferenceBound("regular", :star, G0, 0.01))
    replay = replay_cover(leaf, root, refs; family = :star)
    @test replay.passed
    @test replay.checked_leaves == 1
    @test replay.checked_corners == 8

    splitroot = dyadic_box((0, 0, 0), (2, 2, 2))
    cells = split_boxes(splitroot, (1, 2, 3))
    children = AbstractCoverNode[CoverLeaf(cell, :retained, "r") for cell in cells]
    tree = CoverSplit(splitroot, (1, 2, 3), children)
    @test audit_tree(tree, splitroot).passed

    wrong = copy(children)
    wrong[1] = CoverLeaf(cells[2], :retained, "r")
    @test !audit_tree(CoverSplit(splitroot, (1, 2, 3), wrong), splitroot).passed

    excluded_star = dyadic_box((40, 0, 0), (60, 10, 20))
    @test audit_tree(CoverLeaf(excluded_star, :excluded), excluded_star;
                     family = :star).passed
    feasible_star = dyadic_box((20, 20, 20), (30, 30, 30))
    @test !audit_tree(CoverLeaf(feasible_star, :excluded), feasible_star;
                      family = :star).passed
end
