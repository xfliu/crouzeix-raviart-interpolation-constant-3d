# src/applications/cecr_pipeline/run_pipeline.jl
#
# Full CECR certification pipeline: m4 → m2 → m3 → m5 → m6 → m7 → opt-σ.
# Mirrors the logic in Code_Sorted/cases/case{7,8}_*/run_cecr_*.m.

# ---- 2D Coulomb cell average via Dunavant-6 quadrature --------------------
#
# c_h[k] = (1/|K|) ∫_K V(x) dx ≈ Σ_q w_q V(x_q) * 2 (Dunavant-6, 6-pt)
# r clamped to r_min = 1e-12 to handle elements touching the nucleus.
#
const _R_MIN_2D = 1e-12

function coulomb_average_2d(m::Mesh2D, centers::Matrix{Float64},
                             charges::Vector{Float64})
    lam_q, w_q = dunavant_rule_6()
    c_h = zeros(Float64, m.nt)
    @inbounds for k in 1:m.nt
        v1, v2, v3 = m.elements[k, 1], m.elements[k, 2], m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        V_avg = 0.0
        for q in 1:6
            l1, l2, l3 = lam_q[q, 1], lam_q[q, 2], lam_q[q, 3]
            xq = l1*x1 + l2*x2 + l3*x3
            yq = l1*y1 + l2*y2 + l3*y3
            Vq = 0.0
            for ci in 1:size(centers, 1)
                dx = xq - centers[ci, 1]; dy = yq - centers[ci, 2]
                r = sqrt(dx*dx + dy*dy)
                Vq -= charges[ci] / max(r, _R_MIN_2D)
            end
            # w_q sums to 1/2 for Dunavant-6; multiply by 2 to normalise.
            V_avg += w_q[q] * Vq * 2.0
        end
        c_h[k] = V_avg
    end
    return c_h
end

# ---- Pipeline entry point --------------------------------------------------

"""
    run_cecr_pipeline(mesh, cfg; dh_om0 = NaN, verbose = true)
        -> NamedTuple

Run the full CECR certification pipeline (m4 → m2 → m3 → m5 → m6 → m7 →
opt-σ) on a pre-loaded mesh for the problem described by `cfg`.

Arguments:
* `mesh`   : `Mesh2D` or `Mesh3D`
* `cfg`    : `CaseConfig` with physical parameters and initial σ
* `dh_om0` : optimal-patch ε_h precomputed for the singularity domain Ω₀
  (used in the opt-σ step as `eps_for_opt = min(eps_h, dh_om0/κ_σ)`).
  Pass `NaN` to skip.
* `verbose` : print progress to stdout.

Returns a NamedTuple with fields:
  cfg, cnst, mu_h, Lk_mu_sig, lambda_ub, eps_h, certified,
  Lk_cert,   (m6 with rough σ)
  C_eps_cert, sigma_opt, mu_h_opt, Lk_mu_sig_opt, eps_h_opt, Lk_opt
  (m7 + opt-σ fields, or NaN/nothing if m7 did not converge).
"""
function run_cecr_pipeline(mesh, cfg::CaseConfig;
                            dh_om0::Float64 = NaN, verbose::Bool = true)
    # ------------------------------------------------------------------ m4
    verbose && print("[m4] Mesh constants... ")
    c_h = if cfg.dim == 2
        coulomb_average_2d(mesh, cfg.centers, cfg.charges)
    else
        elem_V_coulomb_average(mesh, CoulombInfo(cfg.centers, cfg.charges))
    end
    cnst = compute_mesh_constants(mesh, c_h, cfg.epsilon)
    verbose && println("Ch_PW=$(round(cnst.Ch_PW, sigdigits=4))  A_h=$(round(cnst.A_h, sigdigits=4))")

    # ------------------------------------------------------------------ m2
    verbose && print("[m2] CECR lower bound... ")
    mu_h, Lk_mu_sig = cecr_lower_bound(mesh, c_h, cfg, cnst)
    verbose && println("Lk_mu_sig = $(round.(Lk_mu_sig, sigdigits=6))")

    # ------------------------------------------------------------------ m3
    verbose && print("[m3] P1 upper bound... ")
    lambda_ub = if cfg.dim == 2
        p1_upper_bound(mesh, cfg)
    else
        p1_upper_bound(mesh, c_h, cfg)
    end
    verbose && println("lambda_ub = $(round.(lambda_ub, sigdigits=6))")

    # ------------------------------------------------------------------ m5
    verbose && print("[m5] eps_h computation... ")
    eps_h = compute_eps_h(mesh, c_h, cfg)
    certified = eps_h < cfg.kappa_sig
    verbose && println("eps_h=$(round(eps_h, sigdigits=5))  certified=$(certified)")

    # ------------------------------------------------------------------ m6 (rough σ)
    Lk_cert = corrected_lower_bound(Lk_mu_sig, eps_h, cfg.sigma)
    if verbose
        for k in 1:length(Lk_cert)
            println("  lambda_$(k) in [$(round(Lk_cert[k], sigdigits=6)), $(round(lambda_ub[k], sigdigits=6))]")
        end
    end

    # ------------------------------------------------------------------ m7
    verbose && print("[m7] C_eps diagnostic... ")
    C_eps_cert = NaN
    sigma_opt  = NaN
    mu_h_opt   = Float64[]
    Lk_mu_sig_opt = Float64[]
    eps_h_opt  = NaN
    Lk_opt     = Float64[]

    try
        C_eps_cert = ceps_diagnostic(mesh, cfg)
        sigma_opt  = C_eps_cert + cfg.kappa_sig
        verbose && println("C_eps_cert=$(round(C_eps_cert, sigdigits=5))  sigma_opt=$(round(sigma_opt, sigdigits=6))")

        if sigma_opt < cfg.sigma
            cfg_opt = update_sigma(cfg, sigma_opt, C_eps_cert)
            cnst_opt = compute_mesh_constants(mesh, c_h, cfg_opt.epsilon)
            mu_h_opt, Lk_mu_sig_opt = cecr_lower_bound(mesh, c_h, cfg_opt, cnst_opt)
            eps_h_opt = compute_eps_h(mesh, c_h, cfg_opt)
            if !isnan(dh_om0)
                eps_h_opt = min(eps_h_opt, dh_om0 / cfg_opt.kappa_sig)
            end
            Lk_opt = corrected_lower_bound(Lk_mu_sig_opt, eps_h_opt, sigma_opt)
            if verbose
                println("[m6-opt] opt-sigma bounds (eps_h=$(round(eps_h_opt, sigdigits=5))):")
                for k in 1:length(Lk_opt)
                    println("  lambda_$(k) in [$(round(Lk_opt[k], sigdigits=6)), $(round(lambda_ub[k], sigdigits=6))]")
                end
            end
        end
    catch e
        verbose && println("m7 failed: $e")
    end

    return (cfg = cfg, cnst = cnst,
            c_h = c_h,
            mu_h = mu_h, Lk_mu_sig = Lk_mu_sig,
            lambda_ub = lambda_ub,
            eps_h = eps_h, certified = certified,
            Lk_cert = Lk_cert,
            C_eps_cert = C_eps_cert, sigma_opt = sigma_opt,
            mu_h_opt = mu_h_opt, Lk_mu_sig_opt = Lk_mu_sig_opt,
            eps_h_opt = eps_h_opt, Lk_opt = Lk_opt)
end
