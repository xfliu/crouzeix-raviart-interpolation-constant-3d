# src/applications/cecr_pipeline/m6_lower_bound.jl
#
# Step m6: corrected lower bound from Theorem 3.7 of the paper.
#
# Formula:  L_k = (1 - ε_h) * (L_k^{μ,σ} + σ) - σ
#
# Inputs:
#   Lk_mu_sig :: Vector — L_k^{μ,σ} values from m2
#   eps_h     :: scalar — ε_h from m5
#   sigma     :: scalar — admissible shift (for reference; not used in formula)

"""
    corrected_lower_bound(Lk_mu_sig, eps_h, sigma) -> Vector{Float64}

Certified lower bound L_k via Theorem 3.7:
    L_k = (1 - ε_h) * (L_k^{μ,σ} + σ) - σ.

`eps_h` must satisfy `eps_h < κ_σ = min(1-ε, σ - C_ε)` for the bound
to be non-trivial (L_k > -σ). The `sigma` parameter is the admissible
shift used in step m2.
"""
function corrected_lower_bound(Lk_mu_sig::AbstractVector{Float64},
                                eps_h::Float64, sigma::Float64)
    return @. (1.0 - eps_h) * (Lk_mu_sig + sigma) - sigma
end
