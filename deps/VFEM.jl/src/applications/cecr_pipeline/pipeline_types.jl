# src/applications/cecr_pipeline/pipeline_types.jl
#
# Data types for the CECR certification pipeline (m4–m7 + drivers).

"""
    CaseConfig

Physical and numerical parameters for one CECR certification run.

* `dim`       :: 2 or 3
* `centers`   :: `Nc × dim` nuclear positions
* `charges`   :: `Nc`-vector of charges Z_c
* `sigma`     :: admissible shift (σ, must satisfy σ > C_ε)
* `epsilon`   :: form-bound coefficient ε ∈ (0, 1)
* `C_eps`     :: conservative form-bound constant (C_ε)
* `kappa_sig` :: min(1-ε, σ-C_ε) — certifiability margin
* `S_q0`      :: Sobolev embedding constant S_q₀ (H¹(Ω) → L^{2p₀})
* `neig`      :: number of eigenvalues to certify
"""
struct CaseConfig
    dim::Int
    centers::Matrix{Float64}
    charges::Vector{Float64}
    sigma::Float64
    epsilon::Float64
    C_eps::Float64
    kappa_sig::Float64
    S_q0::Float64
    neig::Int
    function CaseConfig(dim, centers, charges, sigma, epsilon, C_eps, kappa_sig, S_q0, neig)
        dim in (2, 3) || throw(ArgumentError("dim must be 2 or 3"))
        size(centers, 2) == dim || throw(DimensionMismatch(
            "centers must be Nc×$dim (got $(size(centers)))"))
        length(charges) == size(centers, 1) || throw(DimensionMismatch(
            "charges length must match centers rows"))
        new(Int(dim), Matrix{Float64}(centers), Vector{Float64}(charges),
            Float64(sigma), Float64(epsilon), Float64(C_eps),
            Float64(kappa_sig), Float64(S_q0), Int(neig))
    end
end

"""
    MeshConstants

Mesh-dependent constants from step m4 of the CECR pipeline.

* `Ch_PW`    :: h_max / π  (Poincaré–Wirtinger constant)
* `Gamma_h`  :: max_K c_h^-(K) * (h_K/π)²
* `A_h`      :: Gamma_h / (1 - ε)
* `h_max`    :: maximum element diameter
* `h_K`      :: per-element diameters (max edge length)
"""
struct MeshConstants
    Ch_PW::Float64
    Gamma_h::Float64
    A_h::Float64
    h_max::Float64
    h_K::Vector{Float64}
end

"""
    update_sigma(cfg::CaseConfig, sigma_new, C_eps_new) -> CaseConfig

Return a copy of `cfg` with updated σ and C_ε values.
"""
function update_sigma(cfg::CaseConfig, sigma_new::Float64, C_eps_new::Float64)
    kappa_new = min(1.0 - cfg.epsilon, sigma_new - C_eps_new)
    return CaseConfig(cfg.dim, cfg.centers, cfg.charges,
                      sigma_new, cfg.epsilon, C_eps_new, kappa_new,
                      cfg.S_q0, cfg.neig)
end

# ---- Shift-invert operator for ArnoldiMethod --------------------------------
#
# ShiftInvOp wraps (A - shift*M)^{-1} * M as a matrix-like object for use
# with ArnoldiMethod.partialschur. Its eigenvalues ν = 1/(λ - shift), so
# back-transform: λ = 1/ν + shift.
#
# Arpack.jl v0.5.4 ignores the sigma keyword — this struct is the correct
# workaround.

struct ShiftInvOp{TF, TM}
    F::TF           # LU factorization of (A - shift*M)
    M::TM           # sparse mass matrix
    n::Int
    shift::Float64
end

Base.size(op::ShiftInvOp, ::Int) = op.n
Base.size(op::ShiftInvOp) = (op.n, op.n)
Base.eltype(::ShiftInvOp) = Float64

function LinearAlgebra.mul!(y::AbstractVector, op::ShiftInvOp, x::AbstractVector)
    mul!(y, op.M, x)     # y = M*x
    ldiv!(op.F, y)       # y = F \ y  (= (A-shift*M)^{-1} * M * x)
    return y
end

"""
    _shift_invert_eigs(A, M, k, shift; tol, restarts) -> Vector{Float64}

Find the `k` eigenvalues of `A x = λ M x` nearest to `shift` using
ArnoldiMethod with manual shift-invert.  Returns λ sorted in ascending order.

`shift` should be below the target eigenvalues so that the shift-inverted
operator's largest eigenvalues (1/(λ-shift)) correspond to the smallest λ.
"""
function _shift_invert_eigs(A::AbstractMatrix, M::AbstractMatrix, k::Int,
                             shift::Float64;
                             tol::Float64 = 1e-10, restarts::Int = 5000)
    n = size(A, 1)
    op = ShiftInvOp(lu(A - shift * M), M, n, shift)
    mindim = min(max(3*k + 10, 40), n)
    maxdim = min(max(5*k + 20, 60), n)
    decomp, hist = partialschur(op; nev=k, tol=tol, which=:LM,
                                mindim=mindim, maxdim=maxdim,
                                restarts=restarts)
    if !hist.converged
        @warn "shift-invert Arnoldi: only $(hist.nconverged)/$k eigenvalues converged " *
              "($(hist.mvproducts) mv-products). Results may be inaccurate."
    end
    nu = real.(decomp.eigenvalues)
    return sort(1.0 ./ nu .+ shift)
end
