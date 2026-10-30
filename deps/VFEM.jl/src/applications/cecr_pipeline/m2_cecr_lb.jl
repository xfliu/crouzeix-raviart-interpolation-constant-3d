# src/applications/cecr_pipeline/m2_cecr_lb.jl
#
# Step m2: CECR lower bound using the paper's Theorem 3.6 formula.
# Port of Code_Sorted/modules/m2_cecr_lb/m2_cecr_lower_bound.m.
#
# The CECR eigenproblem for H_σ = −Δ + V + σ uses NEUMANN BC (all DOFs kept).
# Raw eigenvalues μ_k^σ are post-processed via:
#
#   L_k^{μ,σ} = μ_k^σ / [(1 + A_h)(1 + (h_max/π)² μ_k^σ)] − σ
#
# where A_h comes from m4. The σ-shift ensures μ_k^σ > 0 for reasonable k.
#
# Eigensolver: ArnoldiMethod with manual shift-invert at -(σ+0.5).
# Arpack.jl v0.5.4 has a bug where the sigma keyword is silently ignored
# (it returns largest-magnitude eigenvalues instead of shift-inverted ones).
# We work around this via ShiftInvOp + _shift_invert_eigs (pipeline_types.jl).

"""
    cecr_lower_bound(m::Mesh2D, c_h, cfg::CaseConfig, cnst::MeshConstants)
        -> (mu_h, Lk_mu_sig)

CECR lower bound step for the 2D problem. Returns raw CECR eigenvalues
`mu_h` (= μ_k^σ) and corrected values `Lk_mu_sig` (= L_k^{μ,σ}).

`c_h` is the per-element Coulomb potential values (cell averages).
The CECR eigenproblem is assembled with reaction coefficient `c_h .+ cfg.sigma`
and solved with all DOFs (Neumann BC).
"""
function cecr_lower_bound(m::Mesh2D, c_h::AbstractVector{Float64},
                           cfg::CaseConfig, cnst::MeshConstants)
    c_shifted = c_h .+ cfg.sigma
    A, M, _ = create_matrix_cecr(m, c_shifted)
    k_eff = min(cfg.neig, size(A, 1) - 1)
    # Shift below spectrum: CECR eigenvalues ≈ λ_k + σ > 0, so -(σ+0.5) < 0.
    shift = -(cfg.sigma + 0.5)
    mu_h = _shift_invert_eigs(A, M, k_eff, shift)

    Ch2 = cnst.Ch_PW^2
    Lk_mu_sig = @. mu_h / ((1.0 + cnst.A_h) * (1.0 + Ch2 * mu_h)) - cfg.sigma
    return mu_h, Lk_mu_sig
end

"""
    cecr_lower_bound(m::Mesh3D, c_h, cfg::CaseConfig, cnst::MeshConstants)
        -> (mu_h, Lk_mu_sig)

CECR lower bound step for the 3D problem.
"""
function cecr_lower_bound(m::Mesh3D, c_h::AbstractVector{Float64},
                           cfg::CaseConfig, cnst::MeshConstants)
    c_shifted = c_h .+ cfg.sigma
    A, M, _ = create_matrix_cecr_3d(m, c_shifted)
    k_eff = min(cfg.neig, size(A, 1) - 1)
    shift = -(cfg.sigma + 0.1)
    mu_h = _shift_invert_eigs(A, M, k_eff, shift)

    Ch2 = cnst.Ch_PW^2
    Lk_mu_sig = @. mu_h / ((1.0 + cnst.A_h) * (1.0 + Ch2 * mu_h)) - cfg.sigma
    return mu_h, Lk_mu_sig
end
