module BernsteinLB

# Certified lower bounds for the tetrahedral Crouzeix--Raviart constant
#   C(K) = sup { ||v||_0 / |v|_1 : v in H^1(K), four face means zero }
# from Bernstein trial polynomials of degree M.
#
# All Bernstein mass/stiffness kernels are exact rationals, the four
# face-mean constraints are geometry independent integer equations, and the
# constrained trial coefficients are exact rationals.  The only interval
# quantity is the 4x4 matrix g_ij = grad(lambda_i).grad(lambda_j), enclosed at
# 256 bits from the angle parameters.  The volume |K| cancels in the
# Rayleigh quotient.

using LinearAlgebra
using Printf
using IntervalArithmetic: Interval, interval, inf, sup, cospi, sinpi

export multi_indices, gradient_metric, ritz, certify_lower, ShapeSpec,
       face_moments_exact, PointLower

const IB = Interval{BigFloat}
const GEOMETRY_PRECISION = 256

# ---------------------------------------------------------------- indices --

"""Degree-`n` barycentric multi-indices (a0,a1,a2,a3), a0 belonging to P0."""
function multi_indices(n::Int)
    out = NTuple{4, Int}[]
    for a1 in 0:n, a2 in 0:(n - a1), a3 in 0:(n - a1 - a2)
        push!(out, (n - a1 - a2 - a3, a1, a2, a3))
    end
    return out
end

index_map(I) = Dict(a => k for (k, a) in enumerate(I))

multinom(n::Int, a::NTuple{4, Int}) =
    factorial(big(n)) ÷ (factorial(big(a[1])) * factorial(big(a[2])) *
                         factorial(big(a[3])) * factorial(big(a[4])))

addidx(a, b) = (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])
unit(i) = ntuple(k -> k == i ? 1 : 0, 4)
subidx(a, i) = ntuple(k -> k == i ? a[k] - 1 : a[k], 4)

# ------------------------------------------------------------- geometry ----

struct ShapeSpec
    family::Symbol                 # :star or :path
    parameters_deg::NTuple{3, Float64}
end

_deg(x) = interval(BigFloat, BigFloat(x)) / interval(BigFloat, BigFloat(180))
_cosd(x) = cospi(_deg(x))
_sind(x) = sinpi(_deg(x))

function _det3(A)
    return A[1, 1] * (A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2]) -
           A[1, 2] * (A[2, 1] * A[3, 3] - A[2, 3] * A[3, 1]) +
           A[1, 3] * (A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1])
end

function _inv3(A)
    d = _det3(A)
    inf(d) > 0 || throw(DomainError(d, "Gram determinant is not certified positive"))
    B = similar(A)
    B[1, 1] = (A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2]) / d
    B[1, 2] = (A[1, 3] * A[3, 2] - A[1, 2] * A[3, 3]) / d
    B[1, 3] = (A[1, 2] * A[2, 3] - A[1, 3] * A[2, 2]) / d
    B[2, 1] = (A[2, 3] * A[3, 1] - A[2, 1] * A[3, 3]) / d
    B[2, 2] = (A[1, 1] * A[3, 3] - A[1, 3] * A[3, 1]) / d
    B[2, 3] = (A[1, 3] * A[2, 1] - A[1, 1] * A[2, 3]) / d
    B[3, 1] = (A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1]) / d
    B[3, 2] = (A[1, 2] * A[3, 1] - A[1, 1] * A[3, 2]) / d
    B[3, 3] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) / d
    return B
end

"""
Vertex Gram matrix H = E'E of the three edges leaving the base vertex P0, as
256-bit intervals.  Star: P0=A, (P1,P2,P3)=(B,C,D), angles (BAC,BAD,CAD).
Path A-B-C-D: P0=B, (P1,P2,P3)=(A,C,D), parameters (theta1,theta2,phi).
These are the conventions of Kappa3Certificate._reference_geometry.
"""
function vertex_gram(s::ShapeSpec)
    p = s.parameters_deg
    o = interval(BigFloat, BigFloat(1))
    two = interval(BigFloat, BigFloat(2))
    if s.family == :star
        a, b, c = _cosd(p[1]), _cosd(p[2]), _cosd(p[3])
        return [o a b; a o c; b c o]
    elseif s.family == :path
        x, y = _cosd(p[1]), _cosd(p[2])
        z = x * y - _sind(p[1]) * _sind(p[2]) * _cosd(p[3])
        return [o x x-z; x o o-y; x-z o-y two-two*y]
    end
    throw(ArgumentError("family must be :star or :path"))
end

"""
4x4 interval matrix g[i,j] = grad(lambda_{i-1}).grad(lambda_{j-1}).
With E = [P1-P0 P2-P0 P3-P0], grad(lambda_k) (k=1:3) are the rows of
inv(E), so the 3x3 block is inv(H); lambda_0 = 1 - lambda_1 - lambda_2 -
lambda_3 gives the remaining entries.
"""
function gradient_metric(s::ShapeSpec)
    setprecision(BigFloat, GEOMETRY_PRECISION) do
        H = vertex_gram(s)
        Minv = _inv3(H)
        g = Matrix{IB}(undef, 4, 4)
        for i in 1:3, j in 1:3
            g[i + 1, j + 1] = Minv[i, j]
        end
        for j in 1:3
            g[1, j + 1] = -(Minv[1, j] + Minv[2, j] + Minv[3, j])
            g[j + 1, 1] = g[1, j + 1]
        end
        g[1, 1] = -(g[1, 2] + g[1, 3] + g[1, 4])
        return g
    end
end

_mid(x::IB) = Float64((inf(x) + sup(x)) / 2)

# ------------------------------------------------- constraint embedding ----

"""
Face k is {lambda_k = 0}; its constraint is sum_{alpha_k = 0} c_alpha = 0
(every degree-M Bernstein polynomial on a face has the same face integral).
The vertex coefficients c_{M e_i} are eliminated:
    c_{M e_i} = r_i - (r_1+r_2+r_3+r_4)/3,   r_k = sum of the free
coefficients with alpha_k = 0.
Returns (free indices, vertex indices, P) with c = P*y for float use.
"""
function constraint_embedding(M::Int)
    I = multi_indices(M)
    vertex = [findfirst(==(ntuple(k -> k == i ? M : 0, 4)), I) for i in 1:4]
    free = [k for k in eachindex(I) if !(k in vertex)]
    P = zeros(Float64, length(I), length(free))
    for (col, k) in enumerate(free)
        a = I[k]
        P[k, col] = 1.0
        z = count(==(0), a)
        for i in 1:4
            P[vertex[i], col] = (a[i] == 0 ? 1.0 : 0.0) - z / 3
        end
    end
    return free, vertex, P
end

"""Exact coefficients c (Rational{BigInt}) from free values y."""
function exact_coefficients(M::Int, y::AbstractVector{<:Real})
    I = multi_indices(M)
    free, vertex, _ = constraint_embedding(M)
    length(y) == length(free) || throw(DimensionMismatch("free length"))
    c = zeros(Rational{BigInt}, length(I))
    r = zeros(Rational{BigInt}, 4)
    for (col, k) in enumerate(free)
        v = Rational{BigInt}(y[col])
        c[k] = v
        for i in 1:4
            I[k][i] == 0 && (r[i] += v)
        end
    end
    s = sum(r)
    for i in 1:4
        c[vertex[i]] = r[i] - s // 3
    end
    return c
end

"""Exact face sums sum_{alpha_k=0} c_alpha (must all be zero)."""
function face_moments_exact(M::Int, c::Vector{Rational{BigInt}})
    I = multi_indices(M)
    return [sum(c[k] for k in eachindex(I) if I[k][i] == 0) for i in 1:4]
end

# -------------------------------------------------- float Ritz problem ----

"""Float64 mass and stiffness matrices (divided by |K|) for degree M."""
function float_matrices(M::Int, g::Matrix{Float64})
    I = multi_indices(M)
    N = length(I)
    cM = [Float64(multinom(M, a)) for a in I]
    C2 = Dict(a => Float64(multinom(2M, a)) for a in multi_indices(2M))
    n2 = Float64(binomial(2M + 3, 3))
    Mass = Matrix{Float64}(undef, N, N)
    for p in 1:N, q in 1:N
        Mass[p, q] = cM[p] * cM[q] / C2[addidx(I[p], I[q])] / n2
    end
    J = multi_indices(M - 1)
    jmap = index_map(J)
    cJ = [Float64(multinom(M - 1, a)) for a in J]
    C2J = Dict(a => Float64(multinom(2M - 2, a)) for a in multi_indices(2M - 2))
    n2J = Float64(binomial(2M + 1, 3))
    # derivative lists: (row of J, i) for every alpha with alpha_i > 0
    der = [[(jmap[subidx(I[p], i)], i) for i in 1:4 if I[p][i] > 0] for p in 1:N]
    K = zeros(Float64, N, N)
    for p in 1:N, q in p:N
        s = 0.0
        for (dp, i) in der[p], (dq, j) in der[q]
            s += g[i, j] * cJ[dp] * cJ[dq] / C2J[addidx(J[dp], J[dq])]
        end
        K[p, q] = M^2 * s / n2J
        K[q, p] = K[p, q]
    end
    return Mass, K
end

"""
Smallest Ritz eigenpair of P'KP y = lambda P'MP y in binary64.
Returns (lambda_float, y) with y the free coefficient vector.
"""
function ritz(s::ShapeSpec, M::Int)
    M >= 2 || throw(ArgumentError("P_M intersect V_F is trivial for M < 2"))
    g = map(_mid, gradient_metric(s))
    Mass, K = float_matrices(M, g)
    _, _, P = constraint_embedding(M)
    A = Symmetric(P' * K * P)
    B = Symmetric(P' * Mass * P)
    F = eigen(A, B)
    k = argmin(F.values)
    y = F.vectors[:, k]
    y ./= maximum(abs, y)
    return F.values[k], y
end

# --------------------------------------------- exact quadratic forms ------

"""Integer-scaled coefficients: returns (ctilde::Vector{BigInt}, L) with c = ctilde/L."""
function scaled_integers(c::Vector{Rational{BigInt}})
    L = reduce(lcm, (denominator(x) for x in c); init = big(1))
    return [numerator(x) * (L ÷ denominator(x)) for x in c], L
end

"""
Exact value of sum_{p,q} u_p v_q * int_K B^n_p B^n_q / |K|
  = sum_gamma W_gamma / C(2n,gamma) / binom(2n+3,3),
W_gamma = sum_{p+q=gamma} u_p v_q C(n,p) C(n,q).
"""
function bernstein_bilinear(n::Int, u::Vector{BigInt}, v::Vector{BigInt})
    I = multi_indices(n)
    cn = [multinom(n, a) for a in I]
    uu = u .* cn
    vv = v .* cn
    W = Dict{NTuple{4, Int}, BigInt}()
    for p in eachindex(I)
        iszero(uu[p]) && continue
        for q in eachindex(I)
            iszero(vv[q]) && continue
            gidx = addidx(I[p], I[q])
            W[gidx] = get(W, gidx, big(0)) + uu[p] * vv[q]
        end
    end
    total = zero(Rational{BigInt})
    for (gidx, w) in W
        total += w // multinom(2n, gidx)
    end
    return total // binomial(big(2n + 3), 3)
end

"""Directed enclosure of an exact rational as a 256+ bit interval."""
function enclose(q::Rational{BigInt})
    n, d = numerator(q), denominator(q)
    sgn = sign(n)
    an = abs(n)
    lo = setrounding(BigFloat, RoundDown) do
        a = BigFloat(an)
        b = setrounding(BigFloat, RoundUp) do
            BigFloat(d)
        end
        a / b
    end
    hi = setrounding(BigFloat, RoundUp) do
        a = BigFloat(an)
        b = setrounding(BigFloat, RoundDown) do
            BigFloat(d)
        end
        a / b
    end
    if sgn >= 0
        return interval(BigFloat, lo, hi)
    else
        return interval(BigFloat, -hi, -lo)
    end
end

struct PointLower
    spec::ShapeSpec
    degree::Int
    ritz_lambda_float::Float64
    lambda_upper::BigFloat          # certified upper bound of the Rayleigh quotient
    C_lower::Float64                # certified, rounded down
    mass_form::Rational{BigInt}     # ||v||^2/|K| exactly
    face_moments_zero::Bool
end

"""
Certified lower bound for C(K) from the free coefficient vector y.
The trial polynomial is exactly in P_M intersect V_F; its mass is an exact
rational and its energy an interval combination of exact rationals.
"""
function certify_lower(s::ShapeSpec, M::Int, y::AbstractVector{<:Real};
                       ritz_lambda::Float64 = NaN)
    c = exact_coefficients(M, y)
    fm = face_moments_exact(M, c)
    zero_faces = all(iszero, fm)
    zero_faces || error("exact face moments do not vanish")
    ct, L = scaled_integers(c)
    I = multi_indices(M)
    J = multi_indices(M - 1)
    jmap = index_map(J)
    mass = bernstein_bilinear(M, ct, ct) // (L^2)
    d = [zeros(BigInt, length(J)) for _ in 1:4]
    for p in eachindex(I), i in 1:4
        I[p][i] > 0 || continue
        d[i][jmap[subidx(I[p], i)]] = ct[p]
    end
    g = gradient_metric(s)
    setprecision(BigFloat, max(GEOMETRY_PRECISION,
                               8 * ndigits(numerator(mass)) + 64)) do
        energy = interval(BigFloat, BigFloat(0))
        for i in 1:4, j in i:4
            Sij = bernstein_bilinear(M - 1, d[i], d[j]) * M^2 // (L^2)
            term = g[i, j] * enclose(Sij)
            energy += (i == j) ? term : interval(BigFloat, BigFloat(2)) * term
        end
        massI = enclose(mass)
        inf(energy) > 0 || error("energy enclosure is not positive")
        quotient = energy / massI                 # Rayleigh quotient >= lambda_1
        Csq_low = massI / energy
        C_low = setrounding(BigFloat, RoundDown) do
            sqrt(inf(Csq_low))
        end
        Clow64 = Float64(C_low, RoundDown)
        return PointLower(s, M, ritz_lambda, sup(quotient), Clow64, mass,
                          zero_faces)
    end
end

"""Ritz solve followed by certification."""
function point_lower(s::ShapeSpec, M::Int)
    lam, y = ritz(s, M)
    return certify_lower(s, M, y; ritz_lambda = lam)
end

export point_lower

end # module
