# src/core/eigensolve3d/compute_truncation_correction.jl
#
# Port of VFEM3D/lib/eigensolve/compute_truncation_correction.m.
#
# Agmon-style exponential truncation error for the Schrödinger
# eigenvalue enclosure on ℝ^d when the FE problem is truncated to
# B(R + 1).
#
# trunc_err(R) = C_tr · exp(−μ · R)
#
# satisfies   λ_lower − trunc_err ≤ λ_∞ ≤ λ_upper
#
# where λ_∞ is the eigenvalue of the operator on ℝ^d and λ_{lower,upper}
# are the FE bounds on the truncated domain. Inputs:
#   Λ  — rigorous upper bound for the target eigenvalue (Λ < 0).
#   C_V — Coulomb-like decay constant: V(x) ≥ −C_V / |x|^α for |x| ≥ R0.
#   α   — decay exponent (= 1 for Coulomb).
#   R0  — radius beyond which the tail bound holds.
#   R   — truncation radius (FE domain is B(R + 1)).

"""
    TruncationParams

Struct returned by [`compute_truncation_correction`](@ref):
* `R1`       :: `Int`     — Agmon outer radius (smallest int ≥ R0 with δ > 0).
* `delta_R1` :: `Float64` — Agmon energy gap at R1.
* `mu`       :: `Float64` — exponential decay rate.
* `C`        :: `Float64` — exponential growth constant `exp(μ(R1+1))`.
* `C_tr`     :: `Float64` — overall truncation constant.
* `gamma`    :: `Float64` — Agmon shift (here always 0, "conservative").
* `Lambda`   :: `Float64` — input Λ for traceability.
"""
struct TruncationParams
    R1::Int
    delta_R1::Float64
    mu::Float64
    C::Float64
    C_tr::Float64
    gamma::Float64
    Lambda::Float64
end

"""
    compute_truncation_correction(Lambda::Real, C_V::Real, alpha::Real,
                                   R0::Real, R::Real)
        -> (trunc_err::Float64, params::TruncationParams)

Compute the truncation error `C_tr · exp(−μ · R)` for the Schrödinger
eigenvalue enclosure on ℝ^d. See file header for the input semantics.
The Agmon shift `γ` is fixed at 0 (the conservative choice that
maximizes μ for a given Λ).

Throws:
* `DomainError` if `Λ ≥ 0` (must be a bound state).
* `DomainError` if `−Λ − γ² ≤ 0` (tail bound trivializes).
"""
function compute_truncation_correction(Lambda::Real, C_V::Real, alpha::Real,
                                        R0::Real, R::Real)
    Lambda < 0 || throw(DomainError(Lambda, "Λ must be negative (bound state)"))
    γ = 0.0

    threshold = -Lambda - γ^2
    threshold > 0 ||
        throw(DomainError(Lambda, "Need −Λ > γ²; got Λ = $Lambda, γ = $γ"))

    R1_min = (C_V / threshold)^(1 / alpha)
    R1 = max(ceil(Int, R1_min) + 1, ceil(Int, R0))

    delta_R1 = -Lambda - γ^2 - C_V / R1^alpha
    # Width-3/2 cutoff function gives mu = log(1 + 4 δ / 9) (paper Cor.).
    μ = log(1 + 4 * delta_R1 / 9)
    C = exp(μ * (R1 + 1))
    C_tr = (9 / 2) * (1 + 2 * exp(μ / 2)) * C

    trunc_err = C_tr * exp(-μ * R)

    params = TruncationParams(R1, Float64(delta_R1), Float64(μ),
                               Float64(C), Float64(C_tr), Float64(γ),
                               Float64(Lambda))
    return trunc_err, params
end
