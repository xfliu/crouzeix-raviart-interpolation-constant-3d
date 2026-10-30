using Test
using IntervalArithmetic: Interval, inf, interval, sup
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "RigorousSPD.jl"))
using .RigorousSPD

const IA = Interval{Float64}

@testset "directed scalar reductions" begin
    samples = ((1.0, eps(Float64)),
               (floatmax(Float64) / 4, floatmax(Float64) / 8),
               (floatmin(Float64), nextfloat(0.0)),
               (1.0e-200, 1.0e-180))
    setprecision(BigFloat, 256) do
        for (a, b) in samples
            @test BigFloat(RigorousSPD._add_up(a, b)) >= BigFloat(a) + BigFloat(b)
            @test BigFloat(RigorousSPD._mul_up(a, b)) >= BigFloat(a) * BigFloat(b)
        end
        for (a, b) in ((1.0, eps(Float64)), (1.0e100, 1.0e-100))
            @test BigFloat(RigorousSPD._div_up(a, b)) >= BigFloat(a) / BigFloat(b)
        end
        for a in (nextfloat(0.0), 2.0, 1.0e200)
            @test BigFloat(RigorousSPD._sqrt_up(a)) >= sqrt(BigFloat(a))
        end
        for (a, b) in ((1.0, eps(Float64)), (1.0e100, 1.0))
            @test BigFloat(RigorousSPD._sub_down(a, b)) <= BigFloat(a) - BigFloat(b)
        end
    end
end

@testset "Rump gamma denominator is included" begin
    for k in (2, 17, 67_581)
        q = RigorousSPD._gamma_ratio_up(k)
        exact = setprecision(BigFloat, 256) do
            u = BigFloat(2)^(-53)
            (BigFloat(k) * u) / (1 - 2 * BigFloat(k) * u)
        end
        @test BigFloat(q) >= exact
    end
end

@testset "a posteriori factor product norm" begin
    L = sparse([2.0 0.0 0.0;
                -1.0 3.0 0.0;
                0.5 -2.0 1.0])
    got = RigorousSPD._factor_product_inf_norm_up(L)
    exact = setprecision(BigFloat, 256) do
        LB = BigFloat.(abs.(Matrix(L)))
        maximum(sum(LB * transpose(LB); dims = 2))
    end
    @test BigFloat(got) >= exact

    gamma = RigorousSPD._gamma_up(67_581)
    exact_gamma = setprecision(BigFloat, 256) do
        ku = BigFloat(67_581) * BigFloat(2)^(-53)
        ku / (1 - ku)
    end
    @test BigFloat(gamma) >= exact_gamma
end

@testset "small point and interval matrices" begin
    I5 = sparse(interval.(Matrix{Float64}(I, 5, 5)))
    cert = certify_spd(I5; scaling = false)
    @test cert.certified
    @test cert.reason == :certified
    @test cert.total_shift >= cert.roundoff_shift + cert.radius_bound

    A = sparse(interval.([2.0 -0.4 0.0;
                          -0.4 1.5 -0.2;
                          0.0 -0.2 1.0]))
    @test is_spd_certified(A)
    @test certify_spd(sparse([2.0 -1.0; -1.0 2.0]);
                      scaling = false, ordering = :natural,
                      roundoff_bound = :dense).certified

    bad = sparse(interval.([1.0 2.0; 2.0 1.0]))
    @test !certify_spd(bad).certified

    singular = sparse(interval.([1.0 1.0; 1.0 1.0]))
    @test !certify_spd(singular).certified

    nonpositive_diagonal = sparse(interval.([0.0 0.0; 0.0 1.0]))
    @test certify_spd(nonpositive_diagonal).reason == :nonpositive_interval_diagonal
end

@testset "a posteriori residual certificate" begin
    A = sparse(interval.([2.0 -0.4 0.0;
                          -0.4 1.5 -0.2;
                          0.0 -0.2 1.0]))
    cert = certify_spd_residual(A; scaling = false, ordering = :natural)
    @test cert.certified
    @test cert.cholesky_completed
    @test cert.permutation_matched
    @test cert.iterations >= 2
    @test cert.rho > cert.total_error_bound
    @test cert.strict_margin_lower > 0.0
    @test cert.roundoff_bound >= cert.gamma_bound * cert.factor_product_norm_bound

    AI = spzeros(IA, 3, 3)
    AI[1, 1] = interval(2.0, 2.1)
    AI[2, 2] = interval(2.5, 2.6)
    AI[3, 3] = interval(3.0, 3.1)
    AI[1, 2] = AI[2, 1] = interval(-0.21, -0.19)
    AI[2, 3] = AI[3, 2] = interval(0.09, 0.11)
    certI = certify_spd_residual(AI)
    @test certI.certified
    @test certI.input_radius_bound > 0.0
    @test certI.rho > certI.input_radius_bound

    bad = sparse(interval.([1.0 2.0; 2.0 1.0]))
    @test !certify_spd_residual(bad).certified

    wide = spzeros(IA, 2, 2)
    wide[1, 1] = wide[2, 2] = interval(1.0)
    wide[1, 2] = wide[2, 1] = interval(-2.0, 2.0)
    @test !certify_spd_residual(wide).certified
end

@testset "interval family and symmetric hull" begin
    # This whole symmetric interval family is strictly diagonally dominant.
    A = spzeros(IA, 3, 3)
    A[1, 1] = interval(2.0, 2.1)
    A[2, 2] = interval(2.5, 2.6)
    A[3, 3] = interval(3.0, 3.1)
    A[1, 2] = A[2, 1] = interval(-0.21, -0.19)
    A[2, 3] = A[3, 2] = interval(0.09, 0.11)
    cert = certify_spd(A)
    @test cert.certified
    @test cert.radius_bound > 0.0

    # One triangle alone looks harmless, but the other allows |a12|=1.9.
    # Taking a symmetric hull must prevent a false certificate, independent
    # of which triangle CHOLMOD reads.
    B = spzeros(IA, 2, 2)
    B[1, 1] = B[2, 2] = interval(1.0)
    B[1, 2] = interval(0.2, 0.3)
    B[2, 1] = interval(0.2, 1.9)
    @test !certify_spd(B; scaling = false).certified
    @test !certify_spd(transpose(B); scaling = false).certified

    # Enumerate all symmetric corners of a certified 3x3 box as an
    # independent finite sanity check.
    entries = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
    C = [interval(2.0, 2.02) interval(-0.11, -0.09) interval(0.04, 0.06);
         interval(-0.11, -0.09) interval(1.7, 1.72) interval(-0.08, -0.06);
         interval(0.04, 0.06) interval(-0.08, -0.06) interval(1.3, 1.32)]
    @test certify_spd(sparse(C)).certified
    for mask in 0:(2^length(entries) - 1)
        X = zeros(3, 3)
        for (bit, (i, j)) in enumerate(entries)
            x = iszero(mask & (1 << (bit - 1))) ? inf(C[i, j]) : sup(C[i, j])
            X[i, j] = X[j, i] = x
        end
        @test isposdef(Symmetric(X))
    end
end

@testset "fail closed on an interval containing an indefinite matrix" begin
    A = spzeros(IA, 2, 2)
    A[1, 1] = A[2, 2] = interval(1.0)
    A[1, 2] = A[2, 1] = interval(-2.0, 2.0)
    cert = certify_spd(A)
    @test !cert.certified
    @test cert.reason in (:shifted_diagonal_nonpositive, :cholesky_failed)
end

@testset "sparse scaling path" begin
    # A 2000-by-2000 tridiagonal SPD matrix exercises the sparse path while
    # remaining light enough for a one-thread regression test.
    n = 2_000
    T = spdiagm(-1 => fill(-1.0, n - 1),
                 0 => fill(4.0, n),
                 1 => fill(-1.0, n - 1))
    cert = certify_spd(interval.(T); scaling = true,
                       ordering = :amd, roundoff_bound = :envelope)
    @test cert.certified
    @test cert.n == n
    @test cert.nnz == 3n - 2
    @test cert.max_envelope >= 0

    residual_cert = certify_spd_residual(interval.(T); scaling = true,
                                         ordering = :amd)
    @test residual_cert.certified
    @test residual_cert.factor_nnz >= n
end
