module RigorousSPD

using IntervalArithmetic: Interval, hull, inf, interval, sup
using LinearAlgebra
using SparseArrays

export SPDCertificate, SPDResidualCertificate,
       certify_spd, certify_spd_residual, is_spd_certified,
       ieee_preflight

"""
    SPDCertificate

Audit record returned by [`certify_spd`](@ref).  `certified == true` has the
one-sided meaning that every symmetric real matrix in the supplied interval
matrix is positive definite.  A false result only means that this sufficient
test did not prove positive definiteness.

The two shifts have distinct mathematical roles:

* `roundoff_shift` bounds the backward error of the floating-point Cholesky
  factorization (Rump 2006, Theorem 2.3 and Corollary 2.4).
* `radius_bound` bounds the spectral norm of the centered interval radius
  matrix (Rump 2006, Corollary 2.7).

`total_shift` is an outward-rounded upper bound on their sum.  The diagonal
of the matrix actually passed to CHOLMOD is rounded *downward*, as required by
the corollary.
"""
struct SPDCertificate
    certified::Bool
    reason::Symbol
    n::Int
    nnz::Int
    cholesky_completed::Bool
    roundoff_shift::Float64
    radius_bound::Float64
    total_shift::Float64
    roundoff_bound::Symbol
    ordering::Symbol
    max_envelope::Int
    scaled::Bool
    scale_exponent_min::Int
    scale_exponent_max::Int
end

"""
    SPDResidualCertificate

Audit record for [`certify_spd_residual`](@ref), the a posteriori
Cholesky-backward-error certificate used in the manuscript.  On success,
`strict_margin_lower` is a directed lower bound for

    rho - (input_radius_bound + roundoff_bound + underflow_bound).

The certificate is valid only when this number is strictly positive and the
shifted Cholesky factorization completed successfully.
"""
struct SPDResidualCertificate
    certified::Bool
    reason::Symbol
    n::Int
    nnz::Int
    cholesky_completed::Bool
    iterations::Int
    rho::Float64
    input_radius_bound::Float64
    factor_product_norm_bound::Float64
    gamma_bound::Float64
    roundoff_bound::Float64
    underflow_bound::Float64
    total_error_bound::Float64
    strict_margin_lower::Float64
    ordering::Symbol
    permutation_matched::Bool
    factor_nnz::Int
    scaled::Bool
    scale_exponent_min::Int
    scale_exponent_max::Int
end

const _U = eps(Float64) / 2                 # binary64 unit roundoff, 2^-53
const _ETA = nextfloat(0.0)                 # binary64 underflow unit, 2^-1074

"""
Fail unless the current process satisfies the binary64 and BLAS assumptions
used by the a posteriori residual-SPD certificate.  Production and every
standalone replay must call this in their own process.
"""
function ieee_preflight()
    sizeof(Float64) == 8 || throw(ArgumentError("Float64 is not binary64-sized"))
    eps(Float64) == 2.0^-52 || throw(ArgumentError("unexpected Float64 precision"))
    tiny = Ref(floatmin(Float64))[]
    half_tiny = tiny * Ref(0.5)[]
    half_tiny > 0.0 && half_tiny < tiny ||
        throw(ArgumentError("gradual underflow is disabled (flush-to-zero detected)"))
    eta = Ref(nextfloat(0.0))[]
    eta > 0.0 && eta + eta == nextfloat(eta) ||
        throw(ArgumentError("binary64 subnormal arithmetic check failed"))
    one = Ref(1.0)[]
    unit = Ref(eps(Float64))[]
    one + unit / 2 == one ||
        throw(ArgumentError("binary64 arithmetic is not round-to-nearest ties-to-even"))
    one + 3unit / 4 == nextfloat(one) ||
        throw(ArgumentError("binary64 directed-rounding mode is active"))

    # Exercise the linked BLAS rather than inferring its floating-point mode
    # from scalar Julia operations.  The exact result is the least subnormal.
    LinearAlgebra.BLAS.get_num_threads() == 1 ||
        throw(ArgumentError("BLAS must use exactly one thread for the residual SPD proof"))
    x = Float64[tiny]
    y = Float64[unit]
    LinearAlgebra.BLAS.dot(1, x, 1, y, 1) == eta ||
        throw(ArgumentError("BLAS ddot flushes or changes a subnormal result"))
    product = zeros(Float64, 1, 1)
    LinearAlgebra.BLAS.gemm!('N', 'N', 1.0, reshape(x, 1, 1),
                             reshape(y, 1, 1), 0.0, product)
    product[1, 1] == eta ||
        throw(ArgumentError("BLAS dgemm flushes or changes a subnormal result"))
    return true
end

# ---------------------------------------------------------------------------
# Directed scalar operations
# ---------------------------------------------------------------------------
#
# All operands below are finite nonnegative binary64 numbers.  A correctly
# rounded-to-nearest elementary operation differs from its exact value by less
# than one ulp.  Moving one representable number toward +Inf therefore gives a
# rigorous upper bound.  Analogously, one step toward -Inf bounds a
# subtraction from below.  Special cases that are exact avoid needless growth
# by one subnormal.

@inline function _add_up(a::Float64, b::Float64)
    a >= 0.0 && b >= 0.0 || throw(DomainError((a, b), "_add_up expects nonnegative operands"))
    a == 0.0 && return b
    b == 0.0 && return a
    z = a + b
    return isfinite(z) ? nextfloat(z) : Inf
end

@inline function _mul_up(a::Float64, b::Float64)
    a >= 0.0 && b >= 0.0 || throw(DomainError((a, b), "_mul_up expects nonnegative operands"))
    (a == 0.0 || b == 0.0) && return 0.0
    z = a * b
    return isfinite(z) ? nextfloat(z) : Inf
end

@inline function _div_up(a::Float64, b::Float64)
    a >= 0.0 && b > 0.0 || throw(DomainError((a, b), "_div_up expects a >= 0 and b > 0"))
    a == 0.0 && return 0.0
    z = a / b
    return isfinite(z) ? nextfloat(z) : Inf
end

@inline function _sqrt_up(a::Float64)
    a >= 0.0 || throw(DomainError(a, "_sqrt_up expects a nonnegative operand"))
    a == 0.0 && return 0.0
    z = sqrt(a)
    return isfinite(z) ? nextfloat(z) : Inf
end

@inline function _sub_up(a::Float64, b::Float64)
    a >= b || throw(DomainError((a, b), "_sub_up expects a >= b"))
    a == b && return 0.0
    z = a - b
    return isfinite(z) ? nextfloat(z) : Inf
end

@inline function _sub_down(a::Float64, b::Float64)
    b >= 0.0 || throw(DomainError((a, b), "_sub_down expects a nonnegative subtrahend"))
    b == 0.0 && return a
    z = a - b
    return isfinite(z) ? prevfloat(z) : -Inf
end

@inline function _midpoint(lo::Float64, hi::Float64)
    lo <= hi || throw(DomainError((lo, hi), "invalid interval endpoints"))
    # Halving first avoids overflow when lo and hi have the same large sign.
    m = 0.5 * lo + 0.5 * hi
    isfinite(m) || return NaN
    return clamp(m, lo, hi)
end

@inline function _radius_up(lo::Float64, hi::Float64, center::Float64)
    return max(_sub_up(center, lo), _sub_up(hi, center))
end

function _failure(reason::Symbol, n::Int, nz::Int;
                  roundoff_bound::Symbol = :envelope,
                  ordering::Symbol = :amd,
                  roundoff_shift::Float64 = Inf,
                  radius_bound::Float64 = Inf,
                  total_shift::Float64 = Inf,
                  max_envelope::Int = 0,
                  scaled::Bool = false,
                  scale_exponent_min::Int = 0,
                  scale_exponent_max::Int = 0,
                  cholesky_completed::Bool = false)
    return SPDCertificate(false, reason, n, nz, cholesky_completed,
                          roundoff_shift, radius_bound, total_shift,
                          roundoff_bound, ordering, max_envelope, scaled,
                          scale_exponent_min, scale_exponent_max)
end

function _residual_failure(reason::Symbol, n::Int, nz::Int;
                           iterations::Int = 0,
                           rho::Float64 = 0.0,
                           input_radius_bound::Float64 = Inf,
                           factor_product_norm_bound::Float64 = Inf,
                           gamma_bound::Float64 = Inf,
                           roundoff_bound::Float64 = Inf,
                           underflow_bound::Float64 = Inf,
                           total_error_bound::Float64 = Inf,
                           strict_margin_lower::Float64 = -Inf,
                           ordering::Symbol = :amd,
                           permutation_matched::Bool = false,
                           factor_nnz::Int = 0,
                           scaled::Bool = false,
                           scale_exponent_min::Int = 0,
                           scale_exponent_max::Int = 0,
                           cholesky_completed::Bool = false)
    return SPDResidualCertificate(false, reason, n, nz,
                                  cholesky_completed, iterations, rho,
                                  input_radius_bound,
                                  factor_product_norm_bound, gamma_bound,
                                  roundoff_bound, underflow_bound,
                                  total_error_bound, strict_margin_lower,
                                  ordering, permutation_matched, factor_nnz,
                                  scaled, scale_exponent_min,
                                  scale_exponent_max)
end

# ---------------------------------------------------------------------------
# Symmetric interval enclosure and optional exact power-of-two congruence
# ---------------------------------------------------------------------------

function _interval_sparse(M::AbstractMatrix)
    if eltype(M) <: Interval{Float64}
        return sparse(M)
    elseif eltype(M) === Float64
        return sparse(interval.(M))
    elseif eltype(M) <: Interval
        throw(ArgumentError("certify_spd currently requires Interval{Float64} input"))
    end
    throw(ArgumentError("certify_spd currently requires Float64 or Interval{Float64} input"))
end

"""Choose a harmless power-of-two diagonal scaling exponent."""
function _scale_exponent(x::Float64)
    x > 0.0 && isfinite(x) || return 0
    # Scaling is only a conditioning aid; its choice need not itself be
    # verified.  Congruence by whatever positive powers of two are selected
    # preserves inertia exactly.  The clamp keeps pairwise scale factors
    # normal binary64 numbers.
    q = round(Int, -0.5 * log2(x))
    return clamp(q, -500, 500)
end

"""
Build `C` and a rigorous infinity-norm bound for a symmetric nonnegative
radius matrix `R` such that every symmetric matrix represented by `M` lies in
`C + [-R,R]`.

The symmetric hull is intentional.  It prevents a sparse factorizer's choice
of one triangle from silently discarding a wider enclosure in the other
triangle.  Power-of-two diagonal scaling is performed in interval arithmetic,
so certification of `D*M*D` implies certification of `M` by congruence.
"""
function _center_radius(M::AbstractMatrix; scaling::Bool)
    Mi = _interval_sparse(M)
    n, m = size(Mi)
    n == m || throw(DimensionMismatch("certify_spd: matrix must be square"))

    # hull.(sparse, transpose(sparse)) preserves sparsity and produces
    # bit-for-bit equal enclosures in symmetric positions.
    H = sparse(hull.(Mi, transpose(Mi)))
    nz = nnz(H)

    exponents = zeros(Int, n)
    @inbounds for j in 1:n
        x = H[j, j]
        lo, hi = Float64(inf(x)), Float64(sup(x))
        if !(isfinite(lo) && isfinite(hi) && lo <= hi)
            return nothing, :nonfinite_interval, nz
        end
        # Every SPD member must have a positive diagonal.  Requiring the full
        # diagonal interval to be positive is a cheap necessary condition and
        # makes the Rump trace bound applicable.
        lo > 0.0 || return nothing, :nonpositive_interval_diagonal, nz
        c = _midpoint(lo, hi)
        isfinite(c) || return nothing, :nonfinite_midpoint, nz
        exponents[j] = scaling ? _scale_exponent(c) : 0
    end

    colptr = copy(SparseArrays.getcolptr(H))
    rowval = copy(SparseArrays.rowvals(H))
    hvals = SparseArrays.nonzeros(H)
    cvals = Vector{Float64}(undef, length(hvals))
    row_sums = zeros(Float64, n)
    col_sums = zeros(Float64, n)
    diagpos = zeros(Int, n)

    @inbounds for j in 1:n
        for k in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[k]
            q = exponents[i] + exponents[j]
            scale = ldexp(1.0, q)
            isfinite(scale) && scale > 0.0 ||
                return nothing, :nonfinite_scaling, nz

            # The scale is a power of two.  IntervalArithmetic still performs
            # this product outward, covering subnormal/overflow edge cases.
            y = interval(scale) * hvals[k]
            lo, hi = Float64(inf(y)), Float64(sup(y))
            if !(isfinite(lo) && isfinite(hi) && lo <= hi)
                return nothing, :nonfinite_scaled_interval, nz
            end
            c = _midpoint(lo, hi)
            isfinite(c) || return nothing, :nonfinite_midpoint, nz
            r = _radius_up(lo, hi, c)
            isfinite(r) || return nothing, :nonfinite_radius, nz
            cvals[k] = c
            row_sums[i] = _add_up(row_sums[i], r)
            col_sums[j] = _add_up(col_sums[j], r)
            i == j && (diagpos[j] = k)
        end
    end

    all(!iszero, diagpos) || return nothing, :missing_diagonal, nz
    C = SparseMatrixCSC(n, n, colptr, rowval, cvals)
    issymmetric(C) || return nothing, :midpoint_not_symmetric, nz
    # For any real perturbation E with |E| <= R entrywise,
    # ||E||_2 <= ||R||_2 <= sqrt(||R||_1 ||R||_inf).  Computing both norm
    # reductions also keeps the certificate sound even if a future sparse
    # interval implementation returns the symmetric hull with tiny
    # representation-level asymmetries.
    maxrow = isempty(row_sums) ? 0.0 : maximum(row_sums)
    maxcol = isempty(col_sums) ? 0.0 : maximum(col_sums)
    radius_bound = _sqrt_up(_mul_up(maxrow, maxcol))
    data = (; C, radius_bound, diagpos, exponents)
    return data, :ok, nz
end

# ---------------------------------------------------------------------------
# Rump 2006 Cholesky roundoff bound
# ---------------------------------------------------------------------------

"""
Return an outward-rounded upper bound on

    gamma_k / (1 - gamma_k),  gamma_k = k*u/(1-k*u).

Algebraically this is `(k*u)/(1-2*k*u)`.  This second denominator is
load-bearing: using only `gamma_k` (as the former Veigs implementation did)
is a slight underestimate of Rump's bound I/II.
"""
function _gamma_ratio_up(k::Int)
    k >= 0 || throw(DomainError(k, "k must be nonnegative"))
    ku = _mul_up(Float64(k), _U)
    twice = _mul_up(2.0, ku)
    twice < 1.0 || return Inf
    den = _sub_down(1.0, twice)
    den > 0.0 || return Inf
    return _div_up(ku, den)
end

"""Outward upper bound on `gamma_k = k*u/(1-k*u)`."""
function _gamma_up(k::Int)
    k >= 0 || throw(DomainError(k, "k must be nonnegative"))
    ku = _mul_up(Float64(k), _U)
    ku < 1.0 || return Inf
    den = _sub_down(1.0, ku)
    den > 0.0 || return Inf
    return _div_up(ku, den)
end

function _symbolic_amd_permutation(C::SparseMatrixCSC{Float64, Int})
    # Julia 1.12 exposes CHOLMOD's symbolic analysis inside the stdlib module.
    # It performs no numerical factorization.  We subsequently pass the
    # returned permutation explicitly to cholesky, so the envelope bound and
    # the actual factorization use exactly the same ordering.
    S = SparseArrays.CHOLMOD.Sparse(Symmetric(C, :L))
    F = SparseArrays.CHOLMOD.symbolic(S)
    return collect(Int, F.p)
end

function _ordering_permutation(C::SparseMatrixCSC{Float64, Int}, ordering::Symbol)
    n = size(C, 1)
    if ordering === :natural
        return collect(1:n)
    elseif ordering === :amd
        return _symbolic_amd_permutation(C)
    end
    throw(ArgumentError("ordering must be :amd or :natural"))
end

"""Envelope widths `t_j` in Rump (2006), equation (3.2)."""
function _envelope_widths(C::SparseMatrixCSC{Float64, Int}, p::Vector{Int})
    n = size(C, 1)
    length(p) == n || throw(DimensionMismatch("invalid ordering length"))
    sort(p) == collect(1:n) || throw(ArgumentError("ordering is not a permutation"))
    ip = invperm(p)
    firstrow = collect(1:n)
    rows = SparseArrays.rowvals(C)
    colptr = SparseArrays.getcolptr(C)
    @inbounds for oldj in 1:n
        newj = ip[oldj]
        for k in colptr[oldj]:(colptr[oldj + 1] - 1)
            oldi = rows[k]
            newi = ip[oldi]
            a, b = minmax(newi, newj)
            firstrow[b] = min(firstrow[b], a)
        end
    end
    return [j - firstrow[j] for j in 1:n]
end

function _underflow_term_up(n::Int, maxdiag::Float64)
    two_n = _mul_up(2.0, Float64(n))
    M = _mul_up(3.0, _add_up(two_n, maxdiag))
    return _mul_up(Float64(n), _mul_up(M, _ETA))
end

function _roundoff_shift(C::SparseMatrixCSC{Float64, Int}, p::Vector{Int},
                         method::Symbol)
    n = size(C, 1)
    diagC = diag(C)
    all(x -> isfinite(x) && x > 0.0, diagC) ||
        return Inf, 0, :nonpositive_midpoint_diagonal
    maxdiag = maximum(diagC; init = 0.0)
    acc = 0.0
    maxenv = 0

    if method === :dense
        q = _gamma_ratio_up(n + 1)
        isfinite(q) || return Inf, n - 1, :roundoff_bound_overflow
        trace_up = 0.0
        @inbounds for a in diagC
            trace_up = _add_up(trace_up, a)
        end
        acc = _mul_up(q, trace_up)
        maxenv = max(n - 1, 0)
    elseif method === :envelope
        t = _envelope_widths(C, p)
        maxenv = maximum(t; init = 0)
        # Bound II in Rump (2006), equations (3.2)--(3.3):
        # sum_i [gamma_(t_i+2)/(1-gamma_(t_i+2))] * a_ii.
        @inbounds for newi in 1:n
            q = _gamma_ratio_up(t[newi] + 2)
            isfinite(q) || return Inf, maxenv, :roundoff_bound_overflow
            acc = _add_up(acc, _mul_up(q, diagC[p[newi]]))
        end
    else
        throw(ArgumentError("roundoff_bound must be :envelope or :dense"))
    end

    c = _add_up(acc, _underflow_term_up(n, maxdiag))
    return c, maxenv, isfinite(c) ? :ok : :roundoff_bound_overflow
end

function _shift_diagonal_down(C::SparseMatrixCSC{Float64, Int},
                              diagpos::Vector{Int}, shift::Float64)
    vals = copy(SparseArrays.nonzeros(C))
    min_diag = Inf
    @inbounds for j in eachindex(diagpos)
        k = diagpos[j]
        vals[k] = _sub_down(vals[k], shift)
        min_diag = min(min_diag, vals[k])
    end
    Atilde = SparseMatrixCSC(size(C, 1), size(C, 2),
                             copy(SparseArrays.getcolptr(C)),
                             copy(SparseArrays.rowvals(C)), vals)
    return Atilde, min_diag
end

"""
Compute an outward upper bound for `norm(abs(L)*abs(L)', Inf)` without
forming the generally much denser product.

For the nonnegative matrix `Q = abs(L)*abs(L)'`,

    sum_j Q[i,j] = sum_k abs(L[i,k]) * sum_j abs(L[j,k]).

The first pass obtains outward-rounded column sums and the second performs
the corresponding outward-rounded row contractions.  Work and storage are
`O(nnz(L) + n)`.
"""
function _factor_product_inf_norm_up(L::SparseMatrixCSC{Float64, Int})
    n, m = size(L)
    n == m || throw(DimensionMismatch("Cholesky factor must be square"))
    colsum = zeros(Float64, n)
    rowsum = zeros(Float64, n)
    rows = SparseArrays.rowvals(L)
    vals = SparseArrays.nonzeros(L)
    colptr = SparseArrays.getcolptr(L)

    @inbounds for j in 1:n
        s = 0.0
        for k in colptr[j]:(colptr[j + 1] - 1)
            a = abs(vals[k])
            isfinite(a) || return Inf
            s = _add_up(s, a)
        end
        colsum[j] = s
    end

    @inbounds for j in 1:n
        sj = colsum[j]
        for k in colptr[j]:(colptr[j + 1] - 1)
            i = rows[k]
            rowsum[i] = _add_up(rowsum[i], _mul_up(abs(vals[k]), sj))
        end
    end
    return isempty(rowsum) ? 0.0 : maximum(rowsum)
end

"""
    certify_spd(M; scaling=true, ordering=:amd,
                roundoff_bound=:envelope) -> SPDCertificate

Prove positive definiteness of every symmetric matrix in a real interval
matrix `M` using the one-Cholesky sufficient criterion of S. M. Rump,
"Verification of Positive Definiteness", BIT 46 (2006), Theorem 2.3 and
Corollaries 2.4/2.7.

The implementation is intended for sparse finite-element matrices:

1. take the symmetric interval hull and optionally apply a positive diagonal
   power-of-two congruence;
2. split it into a binary64 midpoint `C` and radius matrix `R`, with every
   scalar reduction rounded outward;
3. bound `norm(R,2)` by the outward-rounded maximum row sum;
4. compute Rump's sparse envelope bound II for Cholesky roundoff; and
5. run one CHOLMOD Cholesky after a rigorously downward-rounded diagonal
   shift.

`certified == true` is a proof.  `false` is inconclusive.  `:dense` selects
Rump's more conservative bound I and is useful as an independent fallback.
`ordering=:amd` obtains a symbolic CHOLMOD ordering and then forces that exact
ordering in the numerical factorization, so the envelope calculation matches
the operation being certified.
"""
function certify_spd(M::AbstractMatrix;
                     scaling::Bool = true,
                     ordering::Symbol = :amd,
                     roundoff_bound::Symbol = :envelope)
    n, m = size(M)
    n == m || throw(DimensionMismatch("certify_spd: matrix must be square"))
    ordering in (:amd, :natural) ||
        throw(ArgumentError("ordering must be :amd or :natural"))
    roundoff_bound in (:envelope, :dense) ||
        throw(ArgumentError("roundoff_bound must be :envelope or :dense"))

    if n == 0
        return SPDCertificate(true, :vacuous, 0, 0, true, 0.0, 0.0, 0.0,
                              roundoff_bound, ordering, 0, scaling, 0, 0)
    end

    centered, status, nz = _center_radius(M; scaling)
    status === :ok || return _failure(status, n, nz;
                                      roundoff_bound, ordering, scaled = scaling)
    C = centered.C
    exponents = centered.exponents
    emin, emax = extrema(exponents)

    p = try
        _ordering_permutation(C, ordering)
    catch err
        err isa InterruptException && rethrow()
        return _failure(:symbolic_ordering_failed, n, nz;
                        roundoff_bound, ordering,
                        radius_bound = centered.radius_bound,
                        scaled = scaling,
                        scale_exponent_min = emin,
                        scale_exponent_max = emax)
    end

    c, maxenv, rstatus = _roundoff_shift(C, p, roundoff_bound)
    rstatus === :ok || return _failure(rstatus, n, nz;
                                       roundoff_bound, ordering,
                                       roundoff_shift = c,
                                       radius_bound = centered.radius_bound,
                                       max_envelope = maxenv,
                                       scaled = scaling,
                                       scale_exponent_min = emin,
                                       scale_exponent_max = emax)
    shift = _add_up(c, centered.radius_bound)
    isfinite(shift) || return _failure(:total_shift_overflow, n, nz;
                                       roundoff_bound, ordering,
                                       roundoff_shift = c,
                                       radius_bound = centered.radius_bound,
                                       total_shift = shift,
                                       max_envelope = maxenv,
                                       scaled = scaling,
                                       scale_exponent_min = emin,
                                       scale_exponent_max = emax)

    Atilde, min_diag = _shift_diagonal_down(C, centered.diagpos, shift)
    if !(isfinite(min_diag) && min_diag >= 0.0)
        return _failure(:shifted_diagonal_nonpositive, n, nz;
                        roundoff_bound, ordering,
                        roundoff_shift = c,
                        radius_bound = centered.radius_bound,
                        total_shift = shift,
                        max_envelope = maxenv,
                        scaled = scaling,
                        scale_exponent_min = emin,
                        scale_exponent_max = emax)
    end

    completed = try
        F = cholesky(Symmetric(Atilde, :L); perm = p, check = false)
        issuccess(F)
    catch err
        err isa InterruptException && rethrow()
        false
    end

    reason = completed ? :certified : :cholesky_failed
    return SPDCertificate(completed, reason, n, nz, completed, c,
                          centered.radius_bound, shift, roundoff_bound,
                          ordering, maxenv, scaling, emin, emax)
end

"""
    certify_spd_residual(M; scaling=true, ordering=:amd,
                         initial_rho=0.0, max_attempts=8)
        -> SPDResidualCertificate

Prove positive definiteness with an a posteriori bound on the backward error
of the *computed* sparse Cholesky factor.  This is the certificate described
by equation `exp-cholesky-model` in the manuscript.

After the same rigorous symmetric-hull, optional power-of-two congruence, and
midpoint/radius split used by [`certify_spd`](@ref), the routine repeats:

1. form `Atilde` with diagonal entries rounded downward so that
   `Atilde <= C - rho*I` in the diagonal Loewner order;
2. compute CHOLMOD's binary64 factor `L` of the explicitly ordered matrix;
3. bound the componentwise Cholesky residual by

       gamma_(n+1) * abs(L)*abs(L)' + M*eta*ones(n,n),

   where `M = 3*(2n + max(diag(C)))`, and reduce it outward to an operator
   norm bound; and
4. accept only if the directed lower bound for

       rho - (norm(R,2) + roundoff_error + underflow_error)

   is strictly positive.

If the inequality is not yet strict, `rho` is moved two binary64 values above
the required outward bound and the factorization is repeated.  A successful
return proves every symmetric matrix in `M` SPD.  Failure is inconclusive.

The proof uses the standard componentwise backward-error theorem for a
completed real Cholesky factorization (equivalently Rump 2006, Theorem 2.3,
with `alpha_ij <= gamma_(n+1)`) and Weyl's inequality.  It assumes the pinned
CHOLMOD/BLAS stack performs ordinary IEEE-754 binary64 operations with gradual
underflow.  `F.L` is the lower factor of `Atilde[p,p]`; the code checks that
CHOLMOD reports exactly the requested permutation before using it.  The
infinity norm and the interval-radius norm are invariant under that symmetric
permutation.
"""
function certify_spd_residual(M::AbstractMatrix;
                              scaling::Bool = true,
                              ordering::Symbol = :amd,
                              initial_rho::Float64 = 0.0,
                              max_attempts::Int = 8)
    n, m = size(M)
    n == m || throw(DimensionMismatch("certify_spd_residual: matrix must be square"))
    ordering in (:amd, :natural) ||
        throw(ArgumentError("ordering must be :amd or :natural"))
    isfinite(initial_rho) && initial_rho >= 0.0 ||
        throw(ArgumentError("initial_rho must be finite and nonnegative"))
    max_attempts >= 1 || throw(ArgumentError("max_attempts must be positive"))

    if n == 0
        return SPDResidualCertificate(true, :vacuous, 0, 0, true, 0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, Inf, ordering, true, 0,
                                      scaling, 0, 0)
    end

    centered, status, nz = _center_radius(M; scaling)
    status === :ok || return _residual_failure(status, n, nz;
                                                ordering, scaled = scaling)
    C = centered.C
    exponents = centered.exponents
    emin, emax = extrema(exponents)

    p = try
        _ordering_permutation(C, ordering)
    catch err
        err isa InterruptException && rethrow()
        return _residual_failure(:symbolic_ordering_failed, n, nz;
                                 ordering,
                                 input_radius_bound = centered.radius_bound,
                                 scaled = scaling,
                                 scale_exponent_min = emin,
                                 scale_exponent_max = emax)
    end

    gamma = _gamma_up(n + 1)
    (isfinite(gamma) && gamma < 1.0) ||
        return _residual_failure(:gamma_hypothesis_failed, n, nz;
                                 ordering,
                                 input_radius_bound = centered.radius_bound,
                                 gamma_bound = gamma,
                                 scaled = scaling,
                                 scale_exponent_min = emin,
                                 scale_exponent_max = emax)
    maxdiag = maximum(diag(C); init = 0.0)
    underflow = _underflow_term_up(n, maxdiag)
    isfinite(underflow) || return _residual_failure(:underflow_bound_overflow, n, nz;
                                                    ordering,
                                                    input_radius_bound = centered.radius_bound,
                                                    gamma_bound = gamma,
                                                    underflow_bound = underflow,
                                                    scaled = scaling,
                                                    scale_exponent_min = emin,
                                                    scale_exponent_max = emax)

    rho = initial_rho
    last_qnorm = Inf
    last_roundoff = Inf
    last_total = Inf
    last_margin = -Inf
    last_factor_nnz = 0
    last_completed = false

    for attempt in 1:max_attempts
        Atilde, min_diag = _shift_diagonal_down(C, centered.diagpos, rho)
        if !(isfinite(min_diag) && min_diag > 0.0)
            return _residual_failure(:shifted_diagonal_nonpositive, n, nz;
                                     iterations = attempt, rho,
                                     input_radius_bound = centered.radius_bound,
                                     factor_product_norm_bound = last_qnorm,
                                     gamma_bound = gamma,
                                     roundoff_bound = last_roundoff,
                                     underflow_bound = underflow,
                                     total_error_bound = last_total,
                                     strict_margin_lower = last_margin,
                                     ordering,
                                     permutation_matched = true,
                                     factor_nnz = last_factor_nnz,
                                     scaled = scaling,
                                     scale_exponent_min = emin,
                                     scale_exponent_max = emax,
                                     cholesky_completed = last_completed)
        end

        F = try
            cholesky(Symmetric(Atilde, :L); perm = p, check = false)
        catch err
            err isa InterruptException && rethrow()
            return _residual_failure(:cholesky_exception, n, nz;
                                     iterations = attempt, rho,
                                     input_radius_bound = centered.radius_bound,
                                     factor_product_norm_bound = last_qnorm,
                                     gamma_bound = gamma,
                                     roundoff_bound = last_roundoff,
                                     underflow_bound = underflow,
                                     total_error_bound = last_total,
                                     strict_margin_lower = last_margin,
                                     ordering,
                                     factor_nnz = last_factor_nnz,
                                     scaled = scaling,
                                     scale_exponent_min = emin,
                                     scale_exponent_max = emax)
        end
        last_completed = issuccess(F)
        if !last_completed
            return _residual_failure(:cholesky_failed, n, nz;
                                     iterations = attempt, rho,
                                     input_radius_bound = centered.radius_bound,
                                     factor_product_norm_bound = last_qnorm,
                                     gamma_bound = gamma,
                                     roundoff_bound = last_roundoff,
                                     underflow_bound = underflow,
                                     total_error_bound = last_total,
                                     strict_margin_lower = last_margin,
                                     ordering,
                                     permutation_matched = collect(Int, F.p) == p,
                                     factor_nnz = last_factor_nnz,
                                     scaled = scaling,
                                     scale_exponent_min = emin,
                                     scale_exponent_max = emax)
        end

        permutation_matched = collect(Int, F.p) == p
        permutation_matched || return _residual_failure(:permutation_mismatch, n, nz;
                                                         iterations = attempt, rho,
                                                         input_radius_bound = centered.radius_bound,
                                                         gamma_bound = gamma,
                                                         underflow_bound = underflow,
                                                         ordering,
                                                         permutation_matched = false,
                                                         scaled = scaling,
                                                         scale_exponent_min = emin,
                                                         scale_exponent_max = emax,
                                                         cholesky_completed = true)

        L = try
            sparse(F.L)
        catch err
            err isa InterruptException && rethrow()
            return _residual_failure(:factor_extraction_failed, n, nz;
                                     iterations = attempt, rho,
                                     input_radius_bound = centered.radius_bound,
                                     gamma_bound = gamma,
                                     underflow_bound = underflow,
                                     ordering,
                                     permutation_matched = true,
                                     scaled = scaling,
                                     scale_exponent_min = emin,
                                     scale_exponent_max = emax,
                                     cholesky_completed = true)
        end
        last_factor_nnz = nnz(L)
        if !(istril(L) && all(isfinite, SparseArrays.nonzeros(L)) &&
             all(x -> isfinite(x) && x > 0.0, diag(L)))
            return _residual_failure(:invalid_cholesky_factor, n, nz;
                                     iterations = attempt, rho,
                                     input_radius_bound = centered.radius_bound,
                                     gamma_bound = gamma,
                                     underflow_bound = underflow,
                                     ordering,
                                     permutation_matched = true,
                                     factor_nnz = last_factor_nnz,
                                     scaled = scaling,
                                     scale_exponent_min = emin,
                                     scale_exponent_max = emax,
                                     cholesky_completed = true)
        end
        last_qnorm = _factor_product_inf_norm_up(L)
        isfinite(last_qnorm) || return _residual_failure(:factor_norm_overflow, n, nz;
                                                         iterations = attempt, rho,
                                                         input_radius_bound = centered.radius_bound,
                                                         factor_product_norm_bound = last_qnorm,
                                                         gamma_bound = gamma,
                                                         underflow_bound = underflow,
                                                         ordering,
                                                         permutation_matched = true,
                                                         factor_nnz = last_factor_nnz,
                                                         scaled = scaling,
                                                         scale_exponent_min = emin,
                                                         scale_exponent_max = emax,
                                                         cholesky_completed = true)
        last_roundoff = _mul_up(gamma, last_qnorm)
        last_total = _add_up(centered.radius_bound,
                             _add_up(last_roundoff, underflow))
        isfinite(last_total) || return _residual_failure(:total_error_overflow, n, nz;
                                                         iterations = attempt, rho,
                                                         input_radius_bound = centered.radius_bound,
                                                         factor_product_norm_bound = last_qnorm,
                                                         gamma_bound = gamma,
                                                         roundoff_bound = last_roundoff,
                                                         underflow_bound = underflow,
                                                         total_error_bound = last_total,
                                                         ordering,
                                                         permutation_matched = true,
                                                         factor_nnz = last_factor_nnz,
                                                         scaled = scaling,
                                                         scale_exponent_min = emin,
                                                         scale_exponent_max = emax,
                                                         cholesky_completed = true)

        last_margin = _sub_down(rho, last_total)
        if rho > last_total && last_margin > 0.0
            return SPDResidualCertificate(true, :certified, n, nz, true,
                                          attempt, rho,
                                          centered.radius_bound, last_qnorm,
                                          gamma, last_roundoff, underflow,
                                          last_total, last_margin, ordering,
                                          true, last_factor_nnz, scaling,
                                          emin, emax)
        end

        # Two representable steps give a strictly positive directed margin
        # even after `_sub_down` moves the computed difference down one step.
        rho_new = nextfloat(nextfloat(last_total))
        if !(isfinite(rho_new) && rho_new > rho)
            return _residual_failure(:rho_stagnation, n, nz;
                                     iterations = attempt, rho,
                                     input_radius_bound = centered.radius_bound,
                                     factor_product_norm_bound = last_qnorm,
                                     gamma_bound = gamma,
                                     roundoff_bound = last_roundoff,
                                     underflow_bound = underflow,
                                     total_error_bound = last_total,
                                     strict_margin_lower = last_margin,
                                     ordering,
                                     permutation_matched = true,
                                     factor_nnz = last_factor_nnz,
                                     scaled = scaling,
                                     scale_exponent_min = emin,
                                     scale_exponent_max = emax,
                                     cholesky_completed = true)
        end
        rho = rho_new
    end

    return _residual_failure(:max_attempts, n, nz;
                             iterations = max_attempts, rho,
                             input_radius_bound = centered.radius_bound,
                             factor_product_norm_bound = last_qnorm,
                             gamma_bound = gamma,
                             roundoff_bound = last_roundoff,
                             underflow_bound = underflow,
                             total_error_bound = last_total,
                             strict_margin_lower = last_margin,
                             ordering,
                             permutation_matched = true,
                             factor_nnz = last_factor_nnz,
                             scaled = scaling,
                             scale_exponent_min = emin,
                             scale_exponent_max = emax,
                             cholesky_completed = last_completed)
end

"""Boolean convenience wrapper around [`certify_spd`](@ref)."""
is_spd_certified(M::AbstractMatrix; kwargs...) = certify_spd(M; kwargs...).certified

end # module RigorousSPD
