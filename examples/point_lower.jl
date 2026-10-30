# Certified lower bound C(K) >= L_M for one shape, from an exact-rational
# Bernstein trial polynomial of degree M (Appendix D.1 of the paper).
#
#   julia --project=lowerbound examples/point_lower.jl FAMILY a1 a2 a3 [M]
#
# FAMILY is star or path; the angles are in degrees; M defaults to 10.

include(joinpath(@__DIR__, "..", "lowerbound", "src", "BernsteinLB.jl"))
using .BernsteinLB
using Printf

length(ARGS) in (4, 5) || error("usage: point_lower.jl FAMILY a1 a2 a3 [M]")
family = Symbol(ARGS[1])
p = (parse(Float64, ARGS[2]), parse(Float64, ARGS[3]), parse(Float64, ARGS[4]))
M = length(ARGS) == 5 ? parse(Int, ARGS[5]) : 10
t = @elapsed r = point_lower(ShapeSpec(family, p), M)
@printf("%s %s, degree %d\n", family, repr(p), M)
@printf("Ritz approximation   C ~ %.12f\n", 1 / sqrt(r.ritz_lambda_float))
@printf("certified lower bound C(K) >= %.12f   (%.1f s)\n", r.C_lower, t)
