using Kappa3Certificate
using Kappa3Certificate.RigorousSPD
using IntervalArithmetic: interval, mid
using LinearAlgebra
using SparseArrays

level = parse(Int, get(ENV, "KAPPA3_LEVEL", "5"))
rel = parse(Float64, get(ENV, "KAPPA3_BACKOFF", "1e-4"))
params = (5.0, 5.0, 5.0)
H, Hinv = Kappa3Certificate._reference_geometry(:star, params)
mesh = reference_mesh(level; metric = mid.(H))
A, M = assemble_cr_metric(mesh, H; inverse_metric = Hinv)
Z = constraint_basis(mesh; basis = :haar)
lambda = approximate_smallest_eigenvalue(A, M, Z)
candidate = prevfloat(lambda * (1-rel))
AZ, MZ = A * Z, M * Z
Braw = sparse(transpose(Z) * (AZ - interval(candidate) * MZ))
B = sparse((Braw + transpose(Braw)) / interval(2))

centered, status, _ = RigorousSPD._center_radius(B; scaling = true)
status === :ok || error(status)
C = centered.C
p = RigorousSPD._ordering_permutation(C, :amd)
F = cholesky(Symmetric(C, :L); perm = p, check = false)
issuccess(F) || error("midpoint factorization failed")
L = sparse(F.L)

colsum = zeros(Float64, size(L, 2))
rowsum = zeros(Float64, size(L, 1))
for j in 1:size(L, 2)
    for k in nzrange(L, j)
        colsum[j] = RigorousSPD._add_up(colsum[j], abs(nonzeros(L)[k]))
    end
end
for j in 1:size(L, 2)
    for k in nzrange(L, j)
        i = rowvals(L)[k]
        term = RigorousSPD._mul_up(abs(nonzeros(L)[k]), colsum[j])
        rowsum[i] = RigorousSPD._add_up(rowsum[i], term)
    end
end
product_norm = maximum(rowsum)
n = size(C, 1)
ku = RigorousSPD._mul_up(Float64(n+1), eps(Float64)/2)
gamma = RigorousSPD._div_up(ku, RigorousSPD._sub_down(1.0, ku))
chol_error = RigorousSPD._mul_up(gamma, product_norm)

@show level rel n nnz(C) nnz(L) lambda candidate
@show centered.radius_bound product_norm gamma chol_error
println("suggested shift = ", RigorousSPD._add_up(centered.radius_bound, chol_error))
