module Kappa3Certificate

using Arpack
using IntervalArithmetic: Interval, interval, inf, sup, mid
using LinearAlgebra
using SparseArrays
using VFEM: Mesh3D, get_facet_list, get_edge_list,
            facet_element_connectivity, red_refine_mesh_3d

include("RigorousSPD.jl")
using .RigorousSPD: SPDResidualCertificate, certify_spd_residual
include("MetricMesh.jl")
using .MetricMesh: metric_red_refine_mesh_3d

const IA = Interval{Float64}
const CR_COEFFICIENT = interval(951) / interval(2500)

struct PointCertificate
    family::Symbol
    parameters_deg::NTuple{3, Float64}
    refinement_level::Int
    elements::Int
    dofs_before_constraints::Int
    dofs_after_constraints::Int
    lambda_approx::Float64
    lambda_lower::Float64
    relative_backoff::Float64
    hmax2_upper::Float64
    u2_upper::Float64
    C_upper::Float64
    basis::Symbol
    mesh_strategy::Symbol
    spd_certificate::SPDResidualCertificate
end

"""
Build the standard tetrahedron and apply `level` red refinements.  If a
positive-definite midpoint `metric` is supplied, each central octahedron uses
its shortest metric diagonal; otherwise VFEM's fixed diagonal is retained.
"""
function reference_mesh(level::Integer; metric = nothing)
    level >= 0 || throw(DomainError(level, "level must be nonnegative"))
    nodes = [0.0 0.0 0.0;
             1.0 0.0 0.0;
             0.0 1.0 0.0;
             0.0 0.0 1.0]
    elements = reshape(Int[1, 2, 3, 4], 1, 4)
    facets = get_facet_list(elements)
    edges = get_edge_list(elements)
    f2e, e2f = facet_element_connectivity(elements, facets)
    mesh = Mesh3D(nodes, elements, facets, edges, f2e, e2f,
                  size(nodes, 1), size(elements, 1), size(facets, 1),
                  size(edges, 1))
    for _ in 1:Int(level)
        mesh = isnothing(metric) ? red_refine_mesh_3d(mesh) :
               metric_red_refine_mesh_3d(mesh, metric)
    end
    return mesh
end

@inline function _det3(A)
    return A[1, 1] * (A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2]) -
           A[1, 2] * (A[2, 1] * A[3, 3] - A[2, 3] * A[3, 1]) +
           A[1, 3] * (A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1])
end

function _inv3(A::AbstractMatrix)
    d = _det3(A)
    inf(d) <= 0 <= sup(d) &&
        throw(DomainError(d, "Gram matrix determinant contains zero"))
    B = Matrix{eltype(A)}(undef, 3, 3)
    B[1, 1] = (A[2, 2] * A[3, 3] - A[2, 3] * A[3, 2]) / d
    B[1, 2] = (A[1, 3] * A[3, 2] - A[1, 2] * A[3, 3]) / d
    B[1, 3] = (A[1, 2] * A[2, 3] - A[1, 3] * A[2, 2]) / d
    B[2, 1] = (A[2, 3] * A[3, 1] - A[2, 1] * A[3, 3]) / d
    B[2, 2] = (A[1, 1] * A[3, 3] - A[1, 3] * A[3, 1]) / d
    B[2, 3] = (A[1, 3] * A[2, 1] - A[1, 1] * A[2, 3]) / d
    B[3, 1] = (A[2, 1] * A[3, 2] - A[2, 2] * A[3, 1]) / d
    B[3, 2] = (A[1, 2] * A[3, 1] - A[1, 1] * A[3, 2]) / d
    B[3, 3] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) / d
    return B
end

@inline _deg_interval_big(x::Real) =
    interval(BigFloat, BigFloat(x)) / interval(BigFloat, BigFloat(180))
@inline _cosd_big(x::Real) = cospi(_deg_interval_big(x))
@inline _sind_big(x::Real) = sinpi(_deg_interval_big(x))

@inline _float_interval(x::Interval{BigFloat}) =
    interval(Float64, inf(x), sup(x))

const _PATH_DIAMETER_PRECISIONS = (256, 512, 1024, 2048)

"""
    path_unit_diameter_certified(parameters_deg; strict=false)

Return `true` only after interval arithmetic proves that the remaining path
edge `AD` has length at most one (strictly less than one when `strict=true`).
The three chain edges have unit length, while the angle restrictions checked
by [`point_certificate`](@ref) make `AC` and `BD` no longer than one.  Thus
this is the only additional diameter condition.

Increasing precisions make the predicate robust for binary64 inputs very near
the cap.  An unresolved equality is deliberately rejected: callers needing an
interior reference should move the dihedral angle slightly below the cap.
"""
function path_unit_diameter_certified(parameters_deg::NTuple{3, <:Real};
                                      strict::Bool = false)
    all(isfinite, parameters_deg) || return false
    for precision in _PATH_DIAMETER_PRECISIONS
        ad2 = setprecision(BigFloat, precision) do
            theta1, theta2, phi = parameters_deg
            x = _cosd_big(theta1)
            y = _cosd_big(theta2)
            z = x * y - _sind_big(theta1) * _sind_big(theta2) * _cosd_big(phi)
            one = interval(BigFloat, BigFloat(1))
            two = interval(BigFloat, BigFloat(2))
            interval(BigFloat, BigFloat(3)) - two * x - two * y + two * z
        end
        if strict
            sup(ad2) < 1 && return true
            inf(ad2) >= 1 && return false
        else
            sup(ad2) <= 1 && return true
            inf(ad2) > 1 && return false
        end
    end
    return false
end

"""
Construct a point Gram matrix and its inverse at 256-bit interval precision,
then outward-round the final coefficients to binary64 intervals.  Inverting a
nearly singular small-angle Gram matrix directly in binary64 interval
arithmetic creates avoidable dependency inflation; the high-precision stage
keeps the reference eigenvalue pencil narrow while retaining a rigorous
binary64 enclosure for sparse assembly.
"""
function _reference_geometry(family::Symbol,
                             parameters_deg::NTuple{3, <:Real})
    family in (:star, :path) || throw(ArgumentError("family must be :star or :path"))
    if family == :path
        path_unit_diameter_certified(parameters_deg) ||
            throw(ArgumentError("path reference violates, or cannot rigorously certify, the unit-diameter condition |AD| <= 1"))
    end
    return setprecision(BigFloat, 256) do
        Hbig = if family == :star
            a, b, c = _cosd_big(parameters_deg[1]),
                      _cosd_big(parameters_deg[2]),
                      _cosd_big(parameters_deg[3])
            o = interval(BigFloat, BigFloat(1))
            [o a b; a o c; b c o]
        else
            theta1, theta2, phi = parameters_deg
            x = _cosd_big(theta1)
            y = _cosd_big(theta2)
            z = x * y - _sind_big(theta1) * _sind_big(theta2) * _cosd_big(phi)
            o = interval(BigFloat, BigFloat(1))
            [o x x-z;
             x o o-y;
             x-z o-y interval(BigFloat, BigFloat(2))-interval(BigFloat, BigFloat(2))*y]
        end
        metric_big = _inv3(Hbig)
        H = map(_float_interval, Hbig)
        metric = map(_float_interval, metric_big)
        return H, metric
    end
end

"""Vertex-edge Gram matrix for a star reference tetrahedron."""
function star_vertex_gram(parameters_deg::NTuple{3, <:Real})
    return first(_reference_geometry(:star, parameters_deg))
end

"""Vertex Gram matrix based at B for a path reference tetrahedron."""
function path_vertex_gram(parameters_deg::NTuple{3, <:Real})
    return first(_reference_geometry(:path, parameters_deg))
end

function _inv3_unimodular(Q::Matrix{Int}, grid_resolution::Int)
    d = _det3(Q)
    abs(d) == 1 ||
        throw(ArgumentError("reference sub-tetrahedron is not a unimodular grid cell"))
    B = Matrix{Float64}(undef, 3, 3)
    cofactors = (
        Q[2, 2] * Q[3, 3] - Q[2, 3] * Q[3, 2],
        Q[1, 3] * Q[3, 2] - Q[1, 2] * Q[3, 3],
        Q[1, 2] * Q[2, 3] - Q[1, 3] * Q[2, 2],
        Q[2, 3] * Q[3, 1] - Q[2, 1] * Q[3, 3],
        Q[1, 1] * Q[3, 3] - Q[1, 3] * Q[3, 1],
        Q[1, 3] * Q[2, 1] - Q[1, 1] * Q[2, 3],
        Q[2, 1] * Q[3, 2] - Q[2, 2] * Q[3, 1],
        Q[1, 2] * Q[3, 1] - Q[1, 1] * Q[3, 2],
        Q[1, 1] * Q[2, 2] - Q[1, 2] * Q[2, 1],
    )
    @inbounds for k in 1:9
        value = grid_resolution * cofactors[k] * d
        abs(value) <= maxintfloat(Float64) ||
            throw(ArgumentError("reference gradient is not exactly representable in binary64"))
        B[(k - 1) ÷ 3 + 1, (k - 1) % 3 + 1] = Float64(value)
    end
    return B
end

"""
Validate the exact mesh assumptions used by the interval assembly and return
the integer grid coordinates.  A level-`l` red mesh has resolution
`n=2^l`, exactly `n^3` cells, nodes on the `1/n` lattice of the standard
simplex, and a unimodular integer edge matrix for every cell.  Consequently
all cells have volume `1/(6n^3)` and their barycentric gradients are exactly
representable binary64 integers.  Rejecting any other `Mesh3D` prevents the
hard-coded equal-volume formula from being used outside its proof domain.
"""
function _validated_dyadic_reference_mesh(mesh::Mesh3D)
    mesh.NumElt == size(mesh.ElementList, 1) && mesh.NumElt > 0 ||
        throw(ArgumentError("mesh element metadata is inconsistent"))
    mesh.NumNode == size(mesh.NodeList, 1) ||
        throw(ArgumentError("mesh node metadata is inconsistent"))
    size(mesh.NodeList, 2) == 3 && size(mesh.ElementList, 2) == 4 ||
        throw(ArgumentError("expected a three-dimensional tetrahedral mesh"))
    size(mesh.Element2Facet) == (mesh.NumElt, 4) ||
        throw(ArgumentError("mesh element-to-facet metadata is inconsistent"))

    n = 1
    cells = 1
    while cells < mesh.NumElt
        cells <= typemax(Int) ÷ 8 ||
            throw(ArgumentError("mesh is too large for exact validation"))
        cells *= 8
        n *= 2
    end
    cells == mesh.NumElt ||
        throw(ArgumentError("mesh cell count is not n^3 for a dyadic resolution n"))
    n <= 65_536 ||
        throw(ArgumentError("dyadic resolution is too large for exact binary64 gradients"))

    integer_nodes = Matrix{Int}(undef, mesh.NumNode, 3)
    @inbounds for i in 1:mesh.NumNode
        coordinate_sum = 0
        for j in 1:3
            x = mesh.NodeList[i, j]
            isfinite(x) && 0.0 <= x <= 1.0 ||
                throw(ArgumentError("mesh node is outside the standard reference simplex"))
            scaled = x * n
            isinteger(scaled) ||
                throw(ArgumentError("mesh node is not on the certified dyadic 1/$n grid"))
            q = round(Int, scaled)
            Float64(q) / n == x ||
                throw(ArgumentError("mesh node does not have an exact dyadic representation"))
            integer_nodes[i, j] = q
            coordinate_sum += q
        end
        coordinate_sum <= n ||
            throw(ArgumentError("mesh node is outside the standard reference simplex"))
    end

    @inbounds for elem in 1:mesh.NumElt
        vertices = mesh.ElementList[elem, :]
        all(1 <= v <= mesh.NumNode for v in vertices) &&
        length(unique(vertices)) == 4 ||
            throw(ArgumentError("mesh element has invalid or repeated vertices"))
        Q = Matrix{Int}(undef, 3, 3)
        for row in 1:3, column in 1:3
            Q[row, column] = integer_nodes[vertices[row + 1], column] -
                             integer_nodes[vertices[1], column]
        end
        abs(_det3(Q)) == 1 ||
            throw(ArgumentError("mesh contains a cell with nonuniform reference volume"))
    end
    return n, integer_nodes
end

"""
Assemble the CR mass and stiffness matrices on an affine image whose
vertex-edge Gram matrix is `H`. The common physical Jacobian determinant is
omitted from both matrices and hence cancels from the generalized eigenproblem.
The mesh is checked to be a uniform dyadic red refinement of the standard
tetrahedron before the equal-volume formula or exact binary64 gradients are
used.
"""
function assemble_cr_metric(mesh::Mesh3D, H::Matrix{IA};
                            inverse_metric::Union{Nothing, Matrix{IA}} = nothing)
    grid_resolution, integer_nodes = _validated_dyadic_reference_mesh(mesh)
    metric = isnothing(inverse_metric) ? _inv3(H) : inverse_metric
    nent = 16 * mesh.NumElt
    rows = Vector{Int}(undef, nent)
    cols = Vector{Int}(undef, nent)
    avals = Vector{IA}(undef, nent)
    mvals = Vector{IA}(undef, nent)

    # Red refinement gives equal-volume sub-tetrahedra.
    volume = interval(1) / interval(6 * mesh.NumElt)
    mass_diag = interval(2) / interval(5)
    mass_off = -interval(1) / interval(20)
    pos = 0

    @inbounds for elem in 1:mesh.NumElt
        v1, v2, v3, v4 = mesh.ElementList[elem, :]
        Q = [integer_nodes[v2, 1]-integer_nodes[v1, 1] integer_nodes[v2, 2]-integer_nodes[v1, 2] integer_nodes[v2, 3]-integer_nodes[v1, 3];
             integer_nodes[v3, 1]-integer_nodes[v1, 1] integer_nodes[v3, 2]-integer_nodes[v1, 2] integer_nodes[v3, 3]-integer_nodes[v1, 3];
             integer_nodes[v4, 1]-integer_nodes[v1, 1] integer_nodes[v4, 2]-integer_nodes[v1, 2] integer_nodes[v4, 3]-integer_nodes[v1, 3]]
        Jinv = _inv3_unimodular(Q, grid_resolution)
        grad = Matrix{Float64}(undef, 4, 3)
        for k in 1:3
            grad[2, k] = Jinv[k, 1]
            grad[3, k] = Jinv[k, 2]
            grad[4, k] = Jinv[k, 3]
            grad[1, k] = -(grad[2, k] + grad[3, k] + grad[4, k])
        end
        dofs = mesh.Element2Facet[elem, :]
        for i in 1:4, j in 1:4
            gdot = zero(IA)
            for k in 1:3, l in 1:3
                gdot += grad[i, k] * metric[k, l] * grad[j, l]
            end
            pos += 1
            rows[pos] = dofs[i]
            cols[pos] = dofs[j]
            avals[pos] = interval(9) * volume * gdot
            mvals[pos] = volume * (i == j ? mass_diag : mass_off)
        end
    end
    A = sparse(rows, cols, avals, mesh.NumF, mesh.NumF)
    M = sparse(rows, cols, mvals, mesh.NumF, mesh.NumF)
    return A, M
end

function boundary_face_groups(mesh::Mesh3D)
    groups = [Int[] for _ in 1:4]
    boundary = findall(==(0), @view mesh.Facet2Element[:, 2])
    for fid in boundary
        nodes = mesh.FacetList[fid, :]
        P = mesh.NodeList[nodes, :]
        if all(P[:, 1] .== 0.0)
            push!(groups[1], fid)
        elseif all(P[:, 2] .== 0.0)
            push!(groups[2], fid)
        elseif all(P[:, 3] .== 0.0)
            push!(groups[3], fid)
        elseif all(vec(sum(P; dims = 2)) .== 1.0)
            push!(groups[4], fid)
        else
            error("boundary facet $fid was not assigned to an original face")
        end
    end
    reduce(vcat, groups) |> unique |> length == length(boundary) ||
        error("boundary facet partition is not disjoint and complete")
    return groups
end

function _spatially_sorted_group(mesh::Mesh3D, group::Vector{Int})
    return sort(group; by = fid -> begin
        nodes = mesh.FacetList[fid, :]
        c = vec(sum(mesh.NodeList[nodes, :]; dims = 1)) ./ 3
        (c[1], c[2], c[3])
    end)
end

"""
Return a sparse full-rank basis for the four face-mean constraints. `:chain`
uses nearest columns `e_i-e_{i+1}`. `:haar` uses spatially sorted, normalized
Haar contrasts and has much better conditioning on fine meshes.
"""
function constraint_basis(mesh::Mesh3D; basis::Symbol = :haar)
    basis in (:chain, :haar) || throw(ArgumentError("basis must be :chain or :haar"))
    groups = boundary_face_groups(mesh)
    boundary_set = Set(reduce(vcat, groups))
    interior = [i for i in 1:mesh.NumF if i ∉ boundary_set]
    ncols = mesh.NumF - 4
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, basis == :chain ? 2ncols : 8ncols)
    sizehint!(cols, length(rows))
    sizehint!(vals, length(rows))

    col = 0
    for dof in interior
        col += 1
        push!(rows, dof); push!(cols, col); push!(vals, 1.0)
    end
    for raw_group in groups
        group = _spatially_sorted_group(mesh, raw_group)
        m = length(group)
        if basis == :chain
            for j in 1:(m - 1)
                col += 1
                push!(rows, group[j]); push!(cols, col); push!(vals, 1.0)
                push!(rows, group[j+1]); push!(cols, col); push!(vals, -1.0)
            end
        else
            ispow2(m) || throw(ArgumentError("Haar basis requires a power-of-two face count, got $m"))
            block = 2
            while block <= m
                half = block ÷ 2
                scale = inv(sqrt(Float64(block)))
                for first in 1:block:m
                    col += 1
                    for j in first:(first + half - 1)
                        push!(rows, group[j]); push!(cols, col); push!(vals, scale)
                    end
                    for j in (first + half):(first + block - 1)
                        push!(rows, group[j]); push!(cols, col); push!(vals, -scale)
                    end
                end
                block *= 2
            end
        end
    end
    col == ncols || error("constraint basis has $col columns; expected $ncols")
    return sparse(rows, cols, vals, mesh.NumF, ncols)
end

function _symmetric_midpoint(A)
    B = mid.(A)
    return sparse((B + transpose(B)) / 2)
end

function approximate_smallest_eigenvalue(A, M, Z)
    Am = _symmetric_midpoint(transpose(Z) * A * Z)
    Mm = _symmetric_midpoint(transpose(Z) * M * Z)
    # Arpack.jl 0.5.4's default explicit generalized transformation flips
    # the selection and returns the largest eigenvalue here.  Forcing the
    # implicit shift-invert operator targets the eigenvalue nearest zero.
    values, _, nconv, _, _, _ = Arpack.eigs(Am, Mm; nev = 1, sigma = 0.0,
                                             which = :LM,
                                             explicittransform = :none,
                                             tol = 1e-10,
                                             maxiter = 10_000)
    nconv == 1 || error("ARPACK did not converge")
    lambda = real(values[1])
    lambda > 0 || error("computed nonpositive approximate eigenvalue $lambda")
    return lambda
end

"""Prove a lower bound for the first constrained discrete eigenvalue."""
function verified_lambda_lower(A, M, Z, lambda_approx::Real;
                               initial_backoff::Float64 = 1e-4,
                               max_backoff::Float64 = 1e-1,
                               max_spd_attempts::Int = 8)
    0.0 < initial_backoff <= max_backoff < 1.0 ||
        throw(ArgumentError("require 0 < initial_backoff <= max_backoff < 1"))
    max_spd_attempts >= 1 ||
        throw(ArgumentError("max_spd_attempts must be positive"))
    AZ = A * Z
    MZ = M * Z
    rel = initial_backoff
    while rel <= max_backoff
        candidate = prevfloat(Float64(lambda_approx) * (1 - rel))
        Braw = sparse(transpose(Z) * (AZ - interval(candidate) * MZ))
        # Element assembly can give the two orientations interval endpoints
        # differing by an ulp.  The exact pencil is symmetric, so averaging
        # the two interval enclosures is a valid sparse symmetric enclosure.
        B = sparse((Braw + transpose(Braw)) / interval(2))
        cert = certify_spd_residual(B; scaling = true, ordering = :amd,
                                    max_attempts = max_spd_attempts)
        if cert.certified
            return candidate, rel, cert
        end
        rel *= 10
    end
    error("could not verify a discrete eigenvalue lower bound with relative backoff <= $max_backoff")
end

function hmax2_upper(mesh::Mesh3D, H::Matrix{IA})
    result = 0.0
    for edge in eachrow(mesh.EdgeList)
        d = mesh.NodeList[edge[2], :] - mesh.NodeList[edge[1], :]
        q = zero(IA)
        for i in 1:3, j in 1:3
            q += d[i] * H[i, j] * d[j]
        end
        result = max(result, sup(q))
    end
    return nextfloat(result)
end

function point_certificate(family::Symbol, parameters_deg::NTuple{3, <:Real};
                           level::Int = 3, basis::Symbol = :haar,
                           initial_backoff::Float64 = 1e-4,
                           mesh_strategy::Symbol = :metric,
                           max_spd_attempts::Int = 8)
    family in (:star, :path) || throw(ArgumentError("family must be :star or :path"))
    params = Float64.(parameters_deg)
    ptuple = (params[1], params[2], params[3])
    all(isfinite, ptuple) || throw(ArgumentError("reference angles must be finite"))
    if family == :star
        all(0.0 < x <= 60.0 for x in ptuple) ||
            throw(ArgumentError("star reference angles must lie in (0,60] degrees"))
        all(ptuple[i] < ptuple[j] + ptuple[k]
            for (i, j, k) in ((1, 2, 3), (2, 1, 3), (3, 1, 2))) ||
            throw(ArgumentError("star reference angles must satisfy the strict triangle inequalities"))
    else
        0.0 < ptuple[1] <= 60.0 && 0.0 < ptuple[2] <= 60.0 &&
        0.0 < ptuple[3] <= 90.0 ||
            throw(ArgumentError("path reference angles must lie in (0,60]x(0,60]x(0,90] degrees"))
    end
    H, inverse_metric = _reference_geometry(family, ptuple)
    inf(_det3(H)) > 0.0 ||
        throw(ArgumentError("reference angles do not define a nondegenerate tetrahedron"))
    mesh_strategy in (:metric, :fixed) ||
        throw(ArgumentError("mesh_strategy must be :metric or :fixed"))
    mesh = mesh_strategy == :metric ? reference_mesh(level; metric = mid.(H)) :
                                      reference_mesh(level)
    A, M = assemble_cr_metric(mesh, H; inverse_metric)
    Z = constraint_basis(mesh; basis)
    lambda_approx = approximate_smallest_eigenvalue(A, M, Z)
    lambda_lower, rel, spd_cert = verified_lambda_lower(
        A, M, Z, lambda_approx; initial_backoff,
        max_spd_attempts)
    h2 = hmax2_upper(mesh, H)
    u2_interval = inv(interval(lambda_lower)) +
                  CR_COEFFICIENT^2 * interval(0.0, h2)
    u2_upper = sup(u2_interval)
    C_interval = sqrt(u2_interval)
    C_upper = sup(C_interval)
    return PointCertificate(family, ptuple, level, mesh.NumElt, mesh.NumF,
                            size(Z, 2), lambda_approx, lambda_lower, rel,
                            h2, u2_upper, C_upper, basis, mesh_strategy,
                            spd_cert)
end

export PointCertificate, reference_mesh, star_vertex_gram, path_vertex_gram,
       assemble_cr_metric, boundary_face_groups, constraint_basis,
       approximate_smallest_eigenvalue, verified_lambda_lower, hmax2_upper,
       point_certificate, path_unit_diameter_certified

end
