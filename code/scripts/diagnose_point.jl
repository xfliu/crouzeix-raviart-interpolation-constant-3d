using Kappa3Certificate
using Arpack
using IntervalArithmetic: interval, mid, radius
using LinearAlgebra
using SparseArrays
using Veigs

level = parse(Int, get(ENV, "LEVEL", "2"))
mesh = reference_mesh(level)
H = star_vertex_gram((60.0, 60.0, 60.0))
A, M = assemble_cr_metric(mesh, H)
Z = constraint_basis(mesh; basis = :haar)
AZ = A * Z
MZ = M * Z
B0 = sparse(transpose(Z) * AZ)
N0 = sparse(transpose(Z) * MZ)
Am = Matrix(mid.(B0)); Am = (Am + Am') / 2
Mm = Matrix(mid.(N0)); Mm = (Mm + Mm') / 2
vals = eigvals(Symmetric(Am), Symmetric(Mm))
println("level=", level, " size=", size(Am, 1))
println("dense first eigenvalues=", vals[1:min(5, length(vals))])
lambda_arpack = approximate_smallest_eigenvalue(A, M, Z)
println("ARPACK lambda=", lambda_arpack)
for kwargs in ((; nev=5, which=:SM),
               (; nev=5, sigma=0.0, which=:LM),
               (; nev=5, sigma=0.0, which=:SM))
    av, _, nc, ni, nm, _ = Arpack.eigs(sparse(Am), sparse(Mm);
                                        tol=1e-10, maxiter=10000, kwargs...)
    println("ARPACK ", kwargs, " values=", sort(real.(av)),
            " nconv=", nc, " niter=", ni, " nmult=", nm)
end

for rel in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    candidate = prevfloat(lambda_arpack * (1-rel))
    B = sparse(transpose(Z) * (AZ - interval(candidate) * MZ))
    Bm = sparse(mid.(B)); Br = sparse(radius.(B))
    chol = cholesky(Symmetric(Bm, :L); check = false)
    r1 = maximum(sum(Br; dims = 1))
    ri = maximum(sum(Br; dims = 2))
    println("rel=", rel,
            " float_chol=", issuccess(chol),
            " r1=", r1, " rinf=", ri,
            " rump=", Veigs.verified_isspd(B; method=:rump2006))
end
