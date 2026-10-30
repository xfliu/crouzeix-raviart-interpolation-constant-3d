# src/core/potentials/elem_V_coulomb_average.jl
#
# Port of VFEM3D/lib/eigensolve/potentials/elem_V_coulomb_average.m.
#
# Per-element average of the Coulomb potential
#   V(x) = − Σ_c Z_c / |x − c_c|
# over a tetrahedron K:
#   V_avg(K) = (1/|K|) ∫_K V(x) dx.
#
# Even when an element contains a singularity, the integral is finite
# (1/r ∈ L¹(ℝ³)). We use the Duffy transform on elements near each
# nucleus and a conical-product Gauss-Legendre rule on far elements.
# The Duffy Jacobian η₁² · η₂ exactly cancels the 1/r singularity at
# the chosen vertex, leaving a smooth integrand.
#
# Returned vector is suitable as the `c_data` input to
# `schrodinger_eig_cecr_3d` for Hydrogen / H₂⁺ / general Coulomb cases.

using LinearAlgebra: eigen, SymTridiagonal

"""
    CoulombInfo(centers::Matrix{Float64}, charges::Vector{Float64})

Coulomb potential descriptor:
* `centers` :: `Nc × 3` nuclear positions.
* `charges` :: `Nc`-vector of charges Z_c.

Convention: `V(x) = − Σ_c Z_c / |x − centers[c, :]|` (negative for the
attractive electron-nucleus potential).
"""
struct CoulombInfo
    centers::Matrix{Float64}
    charges::Vector{Float64}
    function CoulombInfo(centers::AbstractMatrix, charges::AbstractVector)
        size(centers, 2) == 3 ||
            throw(DimensionMismatch("centers must be Nc × 3 (got $(size(centers)))"))
        length(charges) == size(centers, 1) ||
            throw(DimensionMismatch("charges length must match centers rows"))
        return new(Matrix{Float64}(centers), Vector{Float64}(charges))
    end
end

# Gauss-Legendre nodes/weights on [0, 1] via Golub-Welsch.
function _gauss_legendre_01(n::Integer)
    n ≥ 1 || throw(DomainError(n, "n must be ≥ 1"))
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    T = SymTridiagonal(zeros(n), β)
    F = eigen(T)
    x0 = F.values
    w0 = 2 .* F.vectors[1, :] .^ 2
    x = 0.5 .* (x0 .+ 1)
    w = 0.5 .* w0
    return x, w
end

# Duffy quadrature for ∫_K (−Z/|x − c|) dx on tetrahedron K with the
# singularity-closest vertex `sing_vertex` (1..4). Returns the *signed
# total integral over the element* (i.e. NOT divided by |K|).
function _duffy_integral_1r(verts::AbstractMatrix{Float64}, center::AbstractVector{Float64},
                             Z::Float64, vol::Float64, sing_vertex::Int,
                             xi::AbstractVector{Float64}, wi::AbstractVector{Float64})
    n = length(xi)
    val = 0.0
    others_data = Tuple(k for k in 1:4 if k != sing_vertex)
    @inbounds for i1 in 1:n, i2 in 1:n, i3 in 1:n
        η1 = xi[i1]; η2 = xi[i2]; η3 = xi[i3]
        lam_duffy = (1 - η1, η1 * (1 - η2), η1 * η2 * (1 - η3), η1 * η2 * η3)
        lam1 = 0.0; lam2 = 0.0; lam3 = 0.0; lam4 = 0.0
        # Place lam_duffy[1] at sing_vertex, then 2/3/4 in the order of `others_data`.
        for v in 1:4
            if v == sing_vertex
                if v == 1; lam1 = lam_duffy[1]
                elseif v == 2; lam2 = lam_duffy[1]
                elseif v == 3; lam3 = lam_duffy[1]
                else;          lam4 = lam_duffy[1]
                end
            end
        end
        for k in 1:3
            other = others_data[k]
            val_lam = lam_duffy[k + 1]
            if other == 1; lam1 = val_lam
            elseif other == 2; lam2 = val_lam
            elseif other == 3; lam3 = val_lam
            else;              lam4 = val_lam
            end
        end
        jac = η1 * η1 * η2
        x = (lam1 * verts[1, 1] + lam2 * verts[2, 1] + lam3 * verts[3, 1] + lam4 * verts[4, 1],
             lam1 * verts[1, 2] + lam2 * verts[2, 2] + lam3 * verts[3, 2] + lam4 * verts[4, 2],
             lam1 * verts[1, 3] + lam2 * verts[2, 3] + lam3 * verts[3, 3] + lam4 * verts[4, 3])
        dx = x[1] - center[1]
        dy = x[2] - center[2]
        dz = x[3] - center[3]
        r = sqrt(dx * dx + dy * dy + dz * dz)
        V_val = -Z / max(r, 1e-15)
        w = wi[i1] * wi[i2] * wi[i3] * jac
        val += w * V_val * 6 * vol
    end
    return val
end

# Standard conical-product Gauss-Legendre quadrature over a tet.
function _standard_integral_1r(verts::AbstractMatrix{Float64}, center::AbstractVector{Float64},
                                Z::Float64, vol::Float64,
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
        x = (lam1 * verts[1, 1] + lam2 * verts[2, 1] + lam3 * verts[3, 1] + lam4 * verts[4, 1],
             lam1 * verts[1, 2] + lam2 * verts[2, 2] + lam3 * verts[3, 2] + lam4 * verts[4, 2],
             lam1 * verts[1, 3] + lam2 * verts[2, 3] + lam3 * verts[3, 3] + lam4 * verts[4, 3])
        dx = x[1] - center[1]
        dy = x[2] - center[2]
        dz = x[3] - center[3]
        r = sqrt(dx * dx + dy * dy + dz * dz)
        V_val = -Z / r
        w = wi[i1] * wi[i2] * wi[i3] * jac
        val += w * V_val * 6 * vol
    end
    return val
end

"""
    elem_V_coulomb_average(m::Mesh3D, info::CoulombInfo;
                            n_duffy::Integer = 10, n_std::Integer = 4)
        -> Vector{Float64}

Per-element average `(1/|K|) ∫_K V(x) dx` of the Coulomb potential
described by `info`. Uses Duffy quadrature with `n_duffy^3` points on
elements whose closest vertex to a center is within half the element
size, and `n_std^3`-point conical-product Gauss-Legendre elsewhere.

Defaults match `VFEM3D/lib/eigensolve/potentials/elem_V_coulomb_average.m`.
"""
function elem_V_coulomb_average(m::Mesh3D, info::CoulombInfo;
                                 n_duffy::Integer = 10, n_std::Integer = 4)
    NumElt = m.NumElt
    V_avg = zeros(Float64, NumElt)

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

        # Centroid for "near singularity" test.
        cx = (verts[1, 1] + verts[2, 1] + verts[3, 1] + verts[4, 1]) / 4
        cy = (verts[1, 2] + verts[2, 2] + verts[3, 2] + verts[4, 2]) / 4
        cz = (verts[1, 3] + verts[2, 3] + verts[3, 3] + verts[4, 3]) / 4

        h_e = 0.0
        for v in 1:4
            δx = verts[v, 1] - cx; δy = verts[v, 2] - cy; δz = verts[v, 3] - cz
            d = sqrt(δx * δx + δy * δy + δz * δz)
            d > h_e && (h_e = d)
        end

        for ci in 1:size(info.centers, 1)
            center = @view info.centers[ci, :]
            Z = info.charges[ci]
            min_dist = Inf
            closest_v = 1
            for v in 1:4
                δx = verts[v, 1] - center[1]
                δy = verts[v, 2] - center[2]
                δz = verts[v, 3] - center[3]
                d = sqrt(δx * δx + δy * δy + δz * δz)
                if d < min_dist
                    min_dist = d
                    closest_v = v
                end
            end

            val = if min_dist < 0.5 * h_e
                _duffy_integral_1r(verts, center, Z, vol, closest_v, xi_d, wi_d)
            else
                _standard_integral_1r(verts, center, Z, vol, xi_s, wi_s)
            end
            V_avg[e] += val / vol
        end
    end
    return V_avg
end
