using Test
using Kappa3Certificate

# This is intentionally separate from the quick unit-test suite.  It exercises
# the publication-sized n=32 (five red refinements) constrained CR pencil and
# the exact metric-mesh, symmetric-hull, automatic-backoff path used by the
# production runner.  It should be run with one Julia/BLAS thread on the HPC
# host.  The small-angle star is the catalogue's tight local certificate.
family = Symbol(get(ENV, "KAPPA3_TEST_FAMILY", "star"))
angle = parse(Float64, get(ENV, "KAPPA3_TEST_ANGLE", "5.0"))
backoff = parse(Float64, get(ENV, "KAPPA3_TEST_BACKOFF", "1e-4"))
level = parse(Int, get(ENV, "KAPPA3_TEST_LEVEL", "5"))
params = (angle, angle, angle)

@testset "publication-scale CR residual SPD certificate" begin
    cert = point_certificate(family, params; level, basis = :haar,
                             mesh_strategy = :metric,
                             initial_backoff = backoff,
                             max_spd_attempts = 8)
    spd = cert.spd_certificate
    record = (; family, params, requested_backoff = backoff, level,
              lambda_approx = cert.lambda_approx,
              lambda_lower = cert.lambda_lower,
              certified_backoff = cert.relative_backoff,
              hmax2_upper = cert.hmax2_upper,
              u2_upper = cert.u2_upper,
              C_upper = cert.C_upper,
              matrix_size = (cert.dofs_after_constraints,
                             cert.dofs_after_constraints),
              certificate = spd)
    @info "CR residual certificate" record
    n = 2^level
    @test cert.dofs_after_constraints == 2n^3 + 2n^2 - 4
    @test spd.certified
    @test spd.rho > spd.total_error_bound
    @test spd.strict_margin_lower > 0.0
    @test cert.u2_upper < 16 // 625
    @test cert.C_upper < 4 // 25
end
