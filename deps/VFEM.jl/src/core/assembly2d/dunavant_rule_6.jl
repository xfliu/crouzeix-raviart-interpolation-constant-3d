# src/core/assembly2d/dunavant_rule_6.jl
#
# 6-point Dunavant quadrature rule, exact for polynomials of total
# degree ≤ 4 on a triangle. Port of the local helper in
# vfem2d/lib/fem_assembly/create_matrix_ecr.m.
#
# Returns (lambda, w) with `lambda[q, :]` = barycentric coords and
# `w[q]` = weight, both summing to 1/2 (reference triangle area).
# So ∫_K f dx = 2|K| · Σ_q w_q f(x_q).

"""
    dunavant_rule_6() -> (lambda::Matrix{Float64}, w::Vector{Float64})

6-point Dunavant rule on the reference triangle, degree-4 exact.
Weights sum to 1/2 (= area of the reference triangle); apply the
factor `2|K|` outside to get a physical-triangle integral.
"""
function dunavant_rule_6()
    A1 = 0.445948490915965
    B1 = 0.108103018168070
    w1 = 0.111690794839005

    A2 = 0.091576213509771
    B2 = 0.816847572980459
    w2 = 0.054975871827661

    lambda = [
        A1 A1 B1;
        A1 B1 A1;
        B1 A1 A1;
        A2 A2 B2;
        A2 B2 A2;
        B2 A2 A2
    ]
    w = [w1, w1, w1, w2, w2, w2]
    return lambda, w
end
