using Test
using Printf
include(joinpath(@__DIR__, "..", "src", "BernsteinLB.jl"))
using .BernsteinLB
const B = BernsteinLB

Cfl(lam) = 1 / sqrt(lam)

@testset "embedding and exact constraints" begin
    for M in 2:6
        free, vertex, P = B.constraint_embedding(M)
        I = B.multi_indices(M)
        @test length(I) == binomial(M + 3, 3)
        @test length(free) == length(I) - 4
        y = randn(length(free))
        c = B.exact_coefficients(M, y)
        @test all(iszero, B.face_moments_exact(M, c))
        @test maximum(abs, Float64.(c) - P * y) < 1e-12
    end
end

@testset "regular tetrahedron: star and path agree" begin
    phi_reg = acosd(1 / 3)
    star = B.ShapeSpec(:star, (60.0, 60.0, 60.0))
    path = B.ShapeSpec(:path, (60.0, 60.0, phi_reg))
    for M in (2, 4, 6)
        ls, _ = B.ritz(star, M)
        lp, _ = B.ritz(path, M)
        @test abs(ls - lp) / ls < 1e-10
    end
end

@testset "permutation invariance" begin
    a = B.ritz(B.ShapeSpec(:star, (20.0, 30.0, 40.0)), 6)[1]
    b = B.ritz(B.ShapeSpec(:star, (40.0, 20.0, 30.0)), 6)[1]
    c = B.ritz(B.ShapeSpec(:path, (20.0, 45.0, 30.0)), 6)[1]
    d = B.ritz(B.ShapeSpec(:path, (45.0, 20.0, 30.0)), 6)[1]
    @test abs(a - b) / a < 1e-10
    @test abs(c - d) / c < 1e-10
end

@testset "certified bound sits below the Ritz value" begin
    for (fam, p, M) in ((:star, (60.0, 60.0, 60.0), 6),
                        (:star, (10.0, 10.0, 10.0), 6),
                        (:path, (30.0, 60.0, 16.220114), 6))
        s = B.ShapeSpec(fam, p)
        r = B.point_lower(s, M)
        @test r.face_moments_zero
        @test r.C_lower <= Cfl(r.ritz_lambda_float) * (1 + 1e-9)
        @test r.C_lower > Cfl(r.ritz_lambda_float) * (1 - 1e-8)
        @printf("%s %s M=%d  ritz C=%.12f  certified C>=%.12f\n",
                fam, repr(p), M, Cfl(r.ritz_lambda_float), r.C_lower)
    end
end

@testset "agreement with the paper's independent Ritz values" begin
    # Table exp-smallpoints (M=10) and Table paper-scan.
    r = B.point_lower(B.ShapeSpec(:star, (1.0, 1.0, 1.0)), 10)
    @printf("star (1,1,1) M=10: ritz %.12f certified %.12f (paper ritz 0.159148882986)\n",
            Cfl(r.ritz_lambda_float), r.C_lower)
    @test abs(Cfl(r.ritz_lambda_float) - 0.159148882986) < 2e-9
    r = B.point_lower(B.ShapeSpec(:star, (5.0, 5.0, 5.0)), 10)
    @printf("star (5,5,5) M=10: ritz %.12f certified %.12f (paper ritz 0.159003478196)\n",
            Cfl(r.ritz_lambda_float), r.C_lower)
    @test abs(Cfl(r.ritz_lambda_float) - 0.159003478196) < 2e-9
    r = B.point_lower(B.ShapeSpec(:path, (30.0, 60.0, 16.220114)), 10)
    @printf("path (30,60,16.220114) M=10: ritz %.12f certified %.12f (paper 0.148735623)\n",
            Cfl(r.ritz_lambda_float), r.C_lower)
    @test abs(Cfl(r.ritz_lambda_float) - 0.148735623) < 2e-9
end
