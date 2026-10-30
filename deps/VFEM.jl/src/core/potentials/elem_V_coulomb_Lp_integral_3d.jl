# src/core/potentials/elem_V_coulomb_Lp_integral_3d.jl
#
# Port of VFEM3D/lib/eigensolve/potentials/elem_V_coulomb_Lp_integral_3d.m.
#
# Per-element Lp integral
#   I[K] = ∫_K |V(x) − c_K|^{p₀} dx
# for the Coulomb potential V(x) = − Σ_c Z_c / |x − c_c|. Used for
# assessing the potential-approximation error (V_avg vs true V) on
# each element.
#
# The Duffy transform absorbs the 1/r singularity for `p₀ < 3` since
# `r^{−p} · η₁²` integrates as `∫ η₁^{2−p} dη₁` which is finite. For
# this reason the 3D Coulomb application typically uses `p₀ = 2` (so
# the kernel is L²) and the Duffy rule remains well-conditioned.

# Reuse `_gauss_legendre_01` from elem_V_coulomb_average.jl. (Both files
# live in the same module, so the `_gauss_legendre_01` defined there is
# visible here.)

# Compute V(x) = − Σ_c Z_c / |x − c_c| at one physical point.
@inline function _coulomb_V_at(x1::Float64, x2::Float64, x3::Float64,
                                centers::AbstractMatrix{Float64},
                                charges::AbstractVector{Float64})
    V = 0.0
    @inbounds for cc in 1:size(centers, 1)
        dx = x1 - centers[cc, 1]
        dy = x2 - centers[cc, 2]
        dz = x3 - centers[cc, 3]
        r = sqrt(dx * dx + dy * dy + dz * dz)
        V -= charges[cc] / max(r, 1e-15)
    end
    return V
end

# Duffy Lp quadrature on a tet with singularity vertex `sv`.
function _duffy_Lp(verts::AbstractMatrix{Float64},
                    centers::AbstractMatrix{Float64},
                    charges::AbstractVector{Float64},
                    cK_e::Float64, vol::Float64, p0::Float64, sv::Int,
                    xi::AbstractVector{Float64}, wi::AbstractVector{Float64})
    n = length(xi)
    val = 0.0
    others = Tuple(k for k in 1:4 if k != sv)
    @inbounds for i1 in 1:n, i2 in 1:n, i3 in 1:n
        η1 = xi[i1]; η2 = xi[i2]; η3 = xi[i3]
        lam_d = (1 - η1, η1 * (1 - η2), η1 * η2 * (1 - η3), η1 * η2 * η3)
        # Place lam_d[1] at sv; lam_d[2..4] at the others in order.
        lam1 = 0.0; lam2 = 0.0; lam3 = 0.0; lam4 = 0.0
        for assign in 1:4
            target = assign == 1 ? sv : others[assign - 1]
            v = lam_d[assign]
            if target == 1
                lam1 = v
            elseif target == 2
                lam2 = v
            elseif target == 3
                lam3 = v
            else
                lam4 = v
            end
        end
        jac = η1 * η1 * η2
        x1 = lam1 * verts[1, 1] + lam2 * verts[2, 1] + lam3 * verts[3, 1] + lam4 * verts[4, 1]
        x2 = lam1 * verts[1, 2] + lam2 * verts[2, 2] + lam3 * verts[3, 2] + lam4 * verts[4, 2]
        x3 = lam1 * verts[1, 3] + lam2 * verts[2, 3] + lam3 * verts[3, 3] + lam4 * verts[4, 3]
        V_val = _coulomb_V_at(x1, x2, x3, centers, charges)
        integrand = abs(V_val - cK_e) ^ p0
        w = wi[i1] * wi[i2] * wi[i3] * jac
        val += w * integrand * 6 * vol
    end
    return val
end

# Standard conical-product Lp quadrature (no singularity).
function _standard_Lp(verts::AbstractMatrix{Float64},
                       centers::AbstractMatrix{Float64},
                       charges::AbstractVector{Float64},
                       cK_e::Float64, vol::Float64, p0::Float64,
                       xi::AbstractVector{Float64}, wi::AbstractVector{Float64})
    n = length(xi)
    val = 0.0
    @inbounds for i1 in 1:n, i2 in 1:n, i3 in 1:n
        r_q = xi[i1]; s_q = xi[i2]; t_q = xi[i3]
        lam4 = r_q
        lam3 = s_q * (1 - r_q)
        lam2 = t_q * (1 - r_q) * (1 - s_q)
        lam1 = 1 - lam2 - lam3 - lam4
        jac = (1 - r_q)^2 * (1 - s_q)
        x1 = lam1 * verts[1, 1] + lam2 * verts[2, 1] + lam3 * verts[3, 1] + lam4 * verts[4, 1]
        x2 = lam1 * verts[1, 2] + lam2 * verts[2, 2] + lam3 * verts[3, 2] + lam4 * verts[4, 2]
        x3 = lam1 * verts[1, 3] + lam2 * verts[2, 3] + lam3 * verts[3, 3] + lam4 * verts[4, 3]
        V_val = _coulomb_V_at(x1, x2, x3, centers, charges)
        integrand = abs(V_val - cK_e) ^ p0
        w = wi[i1] * wi[i2] * wi[i3] * jac
        val += w * integrand * 6 * vol
    end
    return val
end

"""
    elem_V_coulomb_Lp_integral_3d(m::Mesh3D, info::CoulombInfo,
                                   cK::AbstractVector, p0::Real;
                                   n_duffy::Integer = 10, n_std::Integer = 5)
        -> Vector{Float64}

Per-element integral `∫_K |V(x) − cK[K]|^{p₀} dx` over the tetrahedral
mesh `m`, with `V` the Coulomb potential described by `info` and `cK`
the per-element approximation (typically `cK = elem_V_coulomb_average`).

Used to bound the FE potential-approximation error in Liu/LG
arguments. Global Lp norm: `sum(I)^(1/p₀)`.

Defaults `n_duffy = 10`, `n_std = 5` match
`VFEM3D/lib/eigensolve/potentials/elem_V_coulomb_Lp_integral_3d.m`.
"""
function elem_V_coulomb_Lp_integral_3d(m::Mesh3D, info::CoulombInfo,
                                        cK::AbstractVector, p0::Real;
                                        n_duffy::Integer = 10,
                                        n_std::Integer = 5)
    length(cK) == m.NumElt ||
        throw(DimensionMismatch("cK must have length NumElt = $(m.NumElt)"))
    p0_f = Float64(p0)

    NumElt = m.NumElt
    I = zeros(Float64, NumElt)

    xi_d, wi_d = _gauss_legendre_01(n_duffy)
    xi_s, wi_s = _gauss_legendre_01(n_std)

    @inbounds for e in 1:NumElt
        v1 = m.ElementList[e, 1]; v2 = m.ElementList[e, 2]
        v3 = m.ElementList[e, 3]; v4 = m.ElementList[e, 4]
        verts = Float64[m.NodeList[v1, 1] m.NodeList[v1, 2] m.NodeList[v1, 3];
                        m.NodeList[v2, 1] m.NodeList[v2, 2] m.NodeList[v2, 3];
                        m.NodeList[v3, 1] m.NodeList[v3, 2] m.NodeList[v3, 3];
                        m.NodeList[v4, 1] m.NodeList[v4, 2] m.NodeList[v4, 3]]
        d1 = (verts[2, 1] - verts[1, 1], verts[2, 2] - verts[1, 2], verts[2, 3] - verts[1, 3])
        d2 = (verts[3, 1] - verts[1, 1], verts[3, 2] - verts[1, 2], verts[3, 3] - verts[1, 3])
        d3 = (verts[4, 1] - verts[1, 1], verts[4, 2] - verts[1, 2], verts[4, 3] - verts[1, 3])
        det_J = d1[1] * (d2[2] * d3[3] - d2[3] * d3[2]) -
                d1[2] * (d2[1] * d3[3] - d2[3] * d3[1]) +
                d1[3] * (d2[1] * d3[2] - d2[2] * d3[1])
        vol = abs(det_J) / 6.0

        # Centroid for "near singularity" test (same as
        # elem_V_coulomb_average — find the closest center, mark
        # singularity vertex if min vertex distance < 0.5 · h_e).
        cx = (verts[1, 1] + verts[2, 1] + verts[3, 1] + verts[4, 1]) / 4
        cy = (verts[1, 2] + verts[2, 2] + verts[3, 2] + verts[4, 2]) / 4
        cz = (verts[1, 3] + verts[2, 3] + verts[3, 3] + verts[4, 3]) / 4
        h_e = 0.0
        for v in 1:4
            δx = verts[v, 1] - cx; δy = verts[v, 2] - cy; δz = verts[v, 3] - cz
            d = sqrt(δx * δx + δy * δy + δz * δz)
            d > h_e && (h_e = d)
        end

        sing_vertex = 0
        for ci in 1:size(info.centers, 1)
            min_dist = Inf
            local_sv = 1
            for v in 1:4
                δx = verts[v, 1] - info.centers[ci, 1]
                δy = verts[v, 2] - info.centers[ci, 2]
                δz = verts[v, 3] - info.centers[ci, 3]
                d = sqrt(δx * δx + δy * δy + δz * δz)
                if d < min_dist
                    min_dist = d
                    local_sv = v
                end
            end
            if min_dist < 0.5 * h_e
                sing_vertex = local_sv
                break
            end
        end

        I[e] = if sing_vertex > 0
            _duffy_Lp(verts, info.centers, info.charges, Float64(cK[e]),
                      vol, p0_f, sing_vertex, xi_d, wi_d)
        else
            _standard_Lp(verts, info.centers, info.charges, Float64(cK[e]),
                         vol, p0_f, xi_s, wi_s)
        end
    end
    return I
end
