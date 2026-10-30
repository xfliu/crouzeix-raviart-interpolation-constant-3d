using Test
using LinearAlgebra
using Kappa3Certificate
using VFEM: red_refine_mesh_3d

include(joinpath(@__DIR__, "..", "src", "MetricMesh.jl"))
using .MetricMesh: metric_red_refine_mesh_3d, metric_hmax2

function _star_midpoint_metric(parameters_deg)
    alpha, beta, gamma = deg2rad.(parameters_deg)
    return [1.0 cos(alpha) cos(beta);
            cos(alpha) 1.0 cos(gamma);
            cos(beta) cos(gamma) 1.0]
end

function _tetra_volume(nodes, element)
    p1, p2, p3, p4 = eachrow(nodes[element, :])
    return abs(det(hcat(p2-p1, p3-p1, p4-p1))) / 6
end

function _triangle_area(nodes, facet)
    p1, p2, p3 = eachrow(nodes[facet, :])
    return norm(cross(p2-p1, p3-p1)) / 2
end

function _node_at(mesh, point)
    matches = findall(i -> mesh.NodeList[i, :] == point,
                      axes(mesh.NodeList, 1))
    @test length(matches) == 1
    return only(matches)
end

@testset "metric red refinement: deterministic tie" begin
    mesh = Kappa3Certificate.reference_mesh(0)
    refined = metric_red_refine_mesh_3d(mesh, Matrix{Float64}(I, 3, 3))

    m12 = _node_at(refined, [0.5, 0.0, 0.0])
    m13 = _node_at(refined, [0.0, 0.5, 0.0])
    m14 = _node_at(refined, [0.0, 0.0, 0.5])
    m23 = _node_at(refined, [0.5, 0.5, 0.0])
    m24 = _node_at(refined, [0.5, 0.0, 0.5])
    m34 = _node_at(refined, [0.0, 0.5, 0.5])
    edge_set = Set(Tuple(refined.EdgeList[e, :])
                   for e in axes(refined.EdgeList, 1))

    # All three central diagonals have length sqrt(3)/2 in the identity
    # metric.  The documented first-candidate rule chooses m12--m34.
    @test minmax(m12, m34) in edge_set
    @test minmax(m13, m24) ∉ edge_set
    @test minmax(m14, m23) ∉ edge_set
end

@testset "metric red refinement: conformity, volume, and metadata" begin
    metric = _star_midpoint_metric((5.0, 7.5, 10.0))
    mesh = Kappa3Certificate.reference_mesh(0)
    for level in 1:3
        parent_volume = sum(_tetra_volume(mesh.NodeList, mesh.ElementList[e, :])
                            for e in axes(mesh.ElementList, 1))
        mesh = metric_red_refine_mesh_3d(mesh, metric)
        n = 2^level

        @test mesh.NumNode == binomial(n + 3, 3)
        @test mesh.NumElt == n^3
        @test mesh.NumF == 2n^3 + 2n^2
        @test mesh.NumEdge == size(mesh.EdgeList, 1)
        @test size(mesh.Facet2Element) == (mesh.NumF, 2)
        @test size(mesh.Element2Facet) == (mesh.NumElt, 4)

        volumes = [_tetra_volume(mesh.NodeList, mesh.ElementList[e, :])
                   for e in axes(mesh.ElementList, 1)]
        @test all(isapprox(v, 1 / (6n^3); rtol = 0, atol = 16eps(Float64))
                  for v in volumes)
        @test isapprox(sum(volumes), parent_volume; rtol = 0,
                       atol = 64eps(Float64))

        # An independently counted face incidence verifies that there are no
        # cracks, overlaps, or faces shared by more than two tetrahedra.
        incidence = Dict{NTuple{3, Int}, Int}()
        for element in eachrow(mesh.ElementList), omitted in 1:4
            face = Tuple(sort(element[[i for i in 1:4 if i != omitted]]))
            incidence[face] = get(incidence, face, 0) + 1
        end
        @test length(incidence) == mesh.NumF
        @test all(count in (1, 2) for count in values(incidence))
        @test count(==(1), values(incidence)) == 4n^2
        @test count(==(2), values(incidence)) == mesh.NumF - 4n^2

        # The four macro faces retain their midpoint subdivision.  Every
        # boundary facet belongs to exactly one macro face and has 1/n^2 of
        # that face's reference area.
        macro_areas = (0.5, 0.5, 0.5, sqrt(3) / 2)
        face_areas = [Float64[] for _ in 1:4]
        for facet_id in findall(==(0), @view mesh.Facet2Element[:, 2])
            facet = mesh.FacetList[facet_id, :]
            P = mesh.NodeList[facet, :]
            group = if all(P[:, 1] .== 0.0)
                1
            elseif all(P[:, 2] .== 0.0)
                2
            elseif all(P[:, 3] .== 0.0)
                3
            elseif all(vec(sum(P; dims = 2)) .== 1.0)
                4
            else
                0
            end
            @test group != 0
            push!(face_areas[group], _triangle_area(mesh.NodeList, facet))
        end
        @test all(length(areas) == n^2 for areas in face_areas)
        for group in 1:4
            @test all(isapprox(area, macro_areas[group] / n^2;
                               rtol = 0, atol = 16eps(Float64))
                      for area in face_areas[group])
        end
    end
end

@testset "metric red refinement: slender-star h contraction" begin
    # This admissible asymmetric star is close to the planar boundary.  A
    # fixed octahedron diagonal stalls, whereas the shortest-diagonal choice
    # preserves the ideal h^2 factor 1/4 at every tested level.
    metric = _star_midpoint_metric((5.0, 7.5, 10.0))
    fixed = Kappa3Certificate.reference_mesh(0)
    adapted = Kappa3Certificate.reference_mesh(0)
    previous_h2 = metric_hmax2(adapted, metric)
    for _ in 1:4
        fixed = red_refine_mesh_3d(fixed)
        adapted = metric_red_refine_mesh_3d(adapted, metric)
        h2 = metric_hmax2(adapted, metric)
        @test h2 <= previous_h2 / 4 + 32eps(Float64)
        previous_h2 = h2
    end

    fixed_h2 = metric_hmax2(fixed, metric)
    adapted_h2 = metric_hmax2(adapted, metric)
    @test isapprox(adapted_h2, 1 / 4^4; rtol = 0, atol = 64eps(Float64))
    @test fixed_h2 > 3.9adapted_h2

    # The CR correction contributes (951/2500)^2 h_max^2 to U^2, so this
    # directly checks the point-bound ingredient improved by the new mesh.
    correction_factor2 = (951 / 2500)^2
    @test correction_factor2 * adapted_h2 <
          correction_factor2 * fixed_h2 / 3.9
end

@testset "metric red refinement: input validation" begin
    mesh = Kappa3Certificate.reference_mesh(0)
    @test_throws DimensionMismatch metric_red_refine_mesh_3d(
        mesh, Matrix{Float64}(I, 2, 2))
    @test_throws ArgumentError metric_red_refine_mesh_3d(
        mesh, [1.0 0.1 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
    @test_throws ArgumentError metric_red_refine_mesh_3d(
        mesh, [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0])
end
