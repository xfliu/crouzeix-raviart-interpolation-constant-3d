# src/applications/cecr_pipeline/m5_eps_h.jl
#
# Step m5: ε_h computation via L^{p₀} potential-approximation error.
# Port of Code_Sorted/modules/m5_compute_eps_h/m5_compute_eps_h.m.
#
# 2D method A (global):
#   ε_h^A = S_q0² * (∫_Ω |V(x) - c_h(x)|^{4/3} dx)^{3/4} / κ_σ
#
# Quadrature: 6-point Dunavant rule (degree-4 exact) for far elements,
# 7-point rule for near-singular elements. Here we use 6-point for all
# (matches the MATLAB `dunavant_rule_6` used in m5).
#
# 3D method A:
#   ε_h^A = S_q0² * (∫_Ω |V(x) - c_h(x)|^{3/2} dx)^{2/3} / κ_σ
# Uses the existing `elem_V_coulomb_Lp_integral_3d`.

# Evaluate Coulomb potential at (x, y) in 2D.
@inline function _coulomb_2d(x::Float64, y::Float64,
                              centers::Matrix{Float64},
                              charges::Vector{Float64})
    V = 0.0
    @inbounds for c in 1:size(centers, 1)
        dx = x - centers[c, 1]
        dy = y - centers[c, 2]
        r = sqrt(dx*dx + dy*dy)
        V -= charges[c] / max(r, 1e-15)
    end
    return V
end

"""
    eps_h_method_A_2d(m, c_h, centers, charges, S_q0, kappa_sig) -> Float64

2D ε_h via global Method A: L^{4/3} norm of (V - c_h) normalised by κ_σ.

Uses the 6-point Dunavant rule (the existing `dunavant_rule_6()`).
"""
function eps_h_method_A_2d(m::Mesh2D, c_h::AbstractVector{Float64},
                            centers::Matrix{Float64}, charges::Vector{Float64},
                            S_q0::Float64, kappa_sig::Float64)
    lam_q, w_q = dunavant_rule_6()
    n_q = length(w_q)
    Lp_sum = 0.0

    @inbounds for k in 1:m.nt
        v1, v2, v3 = m.elements[k, 1], m.elements[k, 2], m.elements[k, 3]
        x1, y1 = m.nodes[v1, 1], m.nodes[v1, 2]
        x2, y2 = m.nodes[v2, 1], m.nodes[v2, 2]
        x3, y3 = m.nodes[v3, 1], m.nodes[v3, 2]
        area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))   # = 2 * area
        ch_K = c_h[k]

        for q in 1:n_q
            l1, l2, l3 = lam_q[q, 1], lam_q[q, 2], lam_q[q, 3]
            xq = l1*x1 + l2*x2 + l3*x3
            yq = l1*y1 + l2*y2 + l3*y3
            Vq = _coulomb_2d(xq, yq, centers, charges)
            Lp_sum += w_q[q] * abs(Vq - ch_K)^(4.0/3.0) * area2
            # factor area2 because dunavant_rule_6 sums to 1/2 and
            # physical integral = 2*area * sum(w_q * f(x_q)).
        end
    end
    return S_q0^2 * Lp_sum^(3.0/4.0) / kappa_sig
end

"""
    eps_h_method_A_3d(m, c_h, info, S_q0, kappa_sig) -> Float64

3D ε_h via global Method A: L^{3/2} norm of (V - c_h) normalised by κ_σ.
Reuses `elem_V_coulomb_Lp_integral_3d` with p₀ = 3/2.
"""
function eps_h_method_A_3d(m::Mesh3D, c_h::AbstractVector{Float64},
                            info::CoulombInfo,
                            S_q0::Float64, kappa_sig::Float64)
    Lp_elems = elem_V_coulomb_Lp_integral_3d(m, info, c_h, 3.0/2.0)
    Lp_total = sum(Lp_elems)
    return S_q0^2 * Lp_total^(2.0/3.0) / kappa_sig
end

"""
    compute_eps_h(m::Mesh2D, c_h, cfg::CaseConfig) -> Float64
    compute_eps_h(m::Mesh3D, c_h, cfg::CaseConfig) -> Float64

Dispatch to the appropriate ε_h method A for 2D or 3D.
"""
function compute_eps_h(m::Mesh2D, c_h::AbstractVector{Float64}, cfg::CaseConfig)
    return eps_h_method_A_2d(m, c_h, cfg.centers, cfg.charges,
                              cfg.S_q0, cfg.kappa_sig)
end

function compute_eps_h(m::Mesh3D, c_h::AbstractVector{Float64}, cfg::CaseConfig)
    info = CoulombInfo(cfg.centers, cfg.charges)   # cfg.centers is Nc×3 for dim=3
    return eps_h_method_A_3d(m, c_h, info, cfg.S_q0, cfg.kappa_sig)
end
