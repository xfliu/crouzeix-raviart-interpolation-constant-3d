using Test
using Kappa3Certificate
using IntervalArithmetic: inf, mid, sup
using LinearAlgebra

@testset "reference mesh and constraints" begin
    for level in 0:3
        mesh = reference_mesh(level)
        n = 2^level
        @test mesh.NumElt == n^3
        @test mesh.NumF == 2n^3 + 2n^2
        groups = boundary_face_groups(mesh)
        @test all(length(g) == n^2 for g in groups)
        for basis in (:chain, :haar)
            Z = constraint_basis(mesh; basis)
            @test size(Z) == (mesh.NumF, mesh.NumF - 4)
            for group in groups
                residual = vec(sum(Z[group, :]; dims = 1))
                @test maximum(abs, residual; init = 0.0) <= 16eps(Float64)
            end
        end
    end
end

@testset "smallest generalized eigenvalue regression" begin
    mesh = reference_mesh(1)
    H = star_vertex_gram((60.0, 60.0, 60.0))
    A, M = assemble_cr_metric(mesh, H)
    Z = constraint_basis(mesh; basis = :haar)
    Am = Matrix(mid.(transpose(Z) * A * Z))
    Mm = Matrix(mid.(transpose(Z) * M * Z))
    dense_value = first(eigvals(Symmetric((Am + Am') / 2),
                                Symmetric((Mm + Mm') / 2)))
    arpack_value = approximate_smallest_eigenvalue(A, M, Z)
    @test isapprox(arpack_value, dense_value; rtol = 1e-10)
    @test arpack_value < 100.0 # catches Arpack.jl's wrong-end transform
end

@testset "Gram matrices" begin
    G = star_vertex_gram((60.0, 60.0, 60.0))
    @test all(inf(G[i, i]) == 1.0 == sup(G[i, i]) for i in 1:3)
    @test all(inf(G[i, j]) <= 0.5 <= sup(G[i, j])
              for i in 1:3 for j in 1:3 if i != j)

    phi_regular = acosd(1 / 3)
    H = path_vertex_gram((60.0, 60.0, phi_regular))
    @test all(inf(H[i, j]) <= 0.5 <= sup(H[i, j])
              for i in 1:3 for j in 1:3 if i != j)
    @test all(inf(H[i, i]) <= 1.0 <= sup(H[i, i]) for i in 1:3)
end

@testset "path unit-diameter validation" begin
    @test path_unit_diameter_certified((60.0, 60.0, 60.0); strict = true)
    @test !path_unit_diameter_certified((60.0, 60.0, 90.0))
    @test_throws ArgumentError path_vertex_gram((60.0, 60.0, 90.0))
    # Rejection happens in geometry validation, before a mesh or eigensolve is
    # constructed; level 1 merely supplies an otherwise valid point request.
    @test_throws ArgumentError point_certificate(:path, (60.0, 60.0, 90.0);
                                                 level = 1)
end


include("runtests_spd.jl")
include("runtests_cover.jl")
include("runtests_search.jl")
include("runtests_metric_mesh.jl")
include("runtests_assembly_audit.jl")
include("runtests_records.jl")
include("runtests_reference_plan.jl")
include("runtests_production_runner.jl")

@testset "coarse point certificate" begin
    cert = point_certificate(:star, (60.0, 60.0, 60.0);
                             level = 2, basis = :haar,
                             initial_backoff = 1e-6)
    @test cert.lambda_lower > 0
    @test cert.C_upper < 0.20
    @test cert.dofs_after_constraints == cert.dofs_before_constraints - 4
end
