# src/core/quadrature_singular/tet_vertex_sing_poly_integral.jl
#
# Port of VFEM3D/lib/quadrature/tet_vertex_sing_poly_integral.m.
# Wrapper combining bernstein_product_exact + tet_vertex_sing_bernstein_moments.

"""
    tet_vertex_sing_poly_integral(LocalNodes::AbstractMatrix,
                                  singular_vertex::Integer,
                                  degree_p::Integer, coeff_p::AbstractVector,
                                  degree_q::Integer, coeff_q::AbstractVector)
        -> Real

Exact closed-form value of `∫_K p(x) q(x) / |x - P_s| dx`, where
`p` and `q` are tetrahedral Bernstein polynomials of degrees
`degree_p`, `degree_q` with coefficients ordered by
`ijkl_list(degree_p)`, `ijkl_list(degree_q)`. The singular point
`P_s = LocalNodes[singular_vertex, :]` must coincide with a tet
vertex; the routine does not check off-vertex placement.

Uses `bernstein_product_exact` (NOT the monomial product) — the
multinomial-ratio factor is required.
"""
function tet_vertex_sing_poly_integral(LocalNodes::AbstractMatrix,
                                       singular_vertex::Integer,
                                       degree_p::Integer, coeff_p::AbstractVector,
                                       degree_q::Integer, coeff_q::AbstractVector)
    prod_coeff, _ = bernstein_product_exact(degree_p, degree_q, coeff_p, coeff_q)
    moments, _ = tet_vertex_sing_bernstein_moments(
        LocalNodes, singular_vertex, degree_p + degree_q)
    s = zero(eltype(prod_coeff))
    @inbounds for k in eachindex(prod_coeff, moments)
        s += prod_coeff[k] * moments[k]
    end
    return s
end
