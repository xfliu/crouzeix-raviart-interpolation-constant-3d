# Ritz approximation of C(K) (degree M) for tetrahedra given by vertices.
# Input lines: step t x0 y0 z0 x1 ... z3.  Output: same with C appended.
include(joinpath(ARGS[1], "src", "BernsteinLB.jl"))
using .BernsteinLB, LinearAlgebra
const B = BernsteinLB
M = 10
_, _, Pemb = B.constraint_embedding(M)
for line in eachline(ARGS[2])
    f = parse.(Float64, split(line))
    V = reshape(f[3:end], 3, 4)          # columns are vertices
    E = V[:, 2:4] .- V[:, 1]
    Hinv = inv(E' * E)
    g = zeros(4, 4)
    g[2:4, 2:4] = Hinv
    for j in 1:3
        g[1, j + 1] = -sum(Hinv[:, j]); g[j + 1, 1] = g[1, j + 1]
    end
    g[1, 1] = -sum(g[1, 2:4])
    Mass, K = B.float_matrices(M, g)
    lam = minimum(eigvals(Symmetric(Pemb' * K * Pemb), Symmetric(Pemb' * Mass * Pemb)))
    println(line, " ", 1 / sqrt(lam))
end
