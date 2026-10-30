using Test
using LinearAlgebra
using SparseArrays
using IntervalArithmetic: interval, inf, mid, sup
using Kappa3Certificate
using VFEM: Mesh3D, create_matrix_crouzeix_raviart_3d

# This file deliberately rechecks the assembly through physical coordinates.
# It is separate from runtests.jl so that the implementation under audit is
# not used to manufacture all of its own expected values.

function affine_mesh(mesh::Mesh3D, L::Matrix{Float64})
    nodes = mesh.NodeList * L
    return Mesh3D(nodes, copy(mesh.ElementList), copy(mesh.FacetList),
                  copy(mesh.EdgeList), copy(mesh.Facet2Element),
                  copy(mesh.Element2Facet), mesh.NumNode, mesh.NumElt,
                  mesh.NumF, mesh.NumEdge)
end

function physical_face_area(mesh::Mesh3D, fid::Int)
    i, j, k = mesh.FacetList[fid, :]
    p, q, r = mesh.NodeList[i, :], mesh.NodeList[j, :], mesh.NodeList[k, :]
    return norm(cross(q - p, r - p)) / 2
end

function dense_smallest(A, M, Z)
    AZ = Matrix(transpose(Z) * A * Z)
    MZ = Matrix(transpose(Z) * M * Z)
    return eigen(Symmetric((AZ + AZ') / 2),
                 Symmetric((MZ + MZ') / 2)).values[1]
end

function mesh_with_nodes(mesh::Mesh3D, nodes::Matrix{Float64})
    return Mesh3D(nodes, copy(mesh.ElementList), copy(mesh.FacetList),
                  copy(mesh.EdgeList), copy(mesh.Facet2Element),
                  copy(mesh.Element2Facet), mesh.NumNode, mesh.NumElt,
                  mesh.NumF, mesh.NumEdge)
end

@testset "independent CR assembly audit" begin
    # A nonsymmetric row factor makes a metric-transpose error visible.
    # Its entries are dyadic, and det(L)=3/8.
    L = [1.0     0.0    0.0;
         0.25    0.75   0.0;
        -0.125   0.375  0.5]
    H = interval.(L * transpose(L))
    scale = abs(det(L))

    for level in 0:2
        mesh = reference_mesh(level)
        Ametric, Mmetric = assemble_cr_metric(mesh, H)
        physical = affine_mesh(mesh, L)
        Mphysical, Aphysical, _ = create_matrix_crouzeix_raviart_3d(physical)

        # The metric assembly omits the common affine volume factor det(L).
        @test Matrix(mid.(Ametric)) * scale ≈ Matrix(Aphysical) rtol=2e-13 atol=2e-13
        @test Matrix(mid.(Mmetric)) * scale ≈ Matrix(Mphysical) rtol=2e-13 atol=2e-13

        # Local facet i must be the face opposite local vertex i.
        for e in 1:mesh.NumElt, i in 1:4
            expected = sort(mesh.ElementList[e, setdiff(1:4, i)])
            actual = vec(mesh.FacetList[mesh.Element2Facet[e, i], :])
            @test actual == expected
        end

        # All red-refined children have the hard-coded reference volume.
        vols = Float64[]
        for e in 1:mesh.NumElt
            P = mesh.NodeList[mesh.ElementList[e, :], :]
            J = [transpose(P[2, :] - P[1, :]);
                 transpose(P[3, :] - P[1, :]);
                 transpose(P[4, :] - P[1, :])]
            push!(vols, abs(det(J)) / 6)
        end
        @test all(==(1 / (6 * mesh.NumElt)), vols)

        # Equal boundary subface areas justify unweighted macro-face sums.
        groups = boundary_face_groups(mesh)
        for group in groups
            areas = physical_face_area.(Ref(physical), group)
            @test maximum(areas) - minimum(areas) <= 20eps(maximum(areas))
        end

        # The unrestricted vector of all facet averages represents u=1.
        one = ones(mesh.NumF)
        Ar = Ametric * one
        @test all(inf(x) <= 0.0 <= sup(x) for x in Ar)
        total_mass = dot(one, Mmetric * one)
        @test inf(total_mass) <= 1 / 6 <= sup(total_mass)

        # Quantify sparse-accumulation asymmetry.  The exact matrix is
        # symmetric, but independently accumulated directed entries can
        # differ by an ulp.  The verified SPD input must be enclosed in an
        # explicitly symmetric interval matrix downstream.
        Ad = Matrix(Ametric)
        Md = Matrix(Mmetric)
        asym_A = max(maximum(abs.(inf.(Ad) - transpose(inf.(Ad)))),
                     maximum(abs.(sup.(Ad) - transpose(sup.(Ad)))))
        asym_M = max(maximum(abs.(inf.(Md) - transpose(inf.(Md)))),
                     maximum(abs.(sup.(Md) - transpose(sup.(Md)))))
        @test asym_A <= 8eps(maximum(abs, mid.(Ad)))
        @test asym_M <= 8eps(maximum(abs, mid.(Md)))

        Asym = Matrix((Ametric + transpose(Ametric)) / interval(2))
        Msym = Matrix((Mmetric + transpose(Mmetric)) / interval(2))
        @test inf.(Asym) == transpose(inf.(Asym))
        @test sup.(Asym) == transpose(sup.(Asym))
        @test inf.(Msym) == transpose(inf.(Msym))
        @test sup.(Msym) == transpose(sup.(Msym))
    end
end

@testset "assembly rejects meshes outside the exact dyadic-volume contract" begin
    mesh = reference_mesh(1)
    H = interval.(Matrix{Float64}(I, 3, 3))

    offgrid_nodes = copy(mesh.NodeList)
    midpoint = only(findall(i -> mesh.NodeList[i, :] == [0.5, 0.0, 0.0],
                            axes(mesh.NodeList, 1)))
    offgrid_nodes[midpoint, 1] = 0.25
    @test_throws ArgumentError assemble_cr_metric(
        mesh_with_nodes(mesh, offgrid_nodes), H)

    nonuniform_nodes = copy(mesh.NodeList)
    nonuniform_nodes[midpoint, :] .= 0.0
    @test_throws ArgumentError assemble_cr_metric(
        mesh_with_nodes(mesh, nonuniform_nodes), H)
end

@testset "four macro-face constraints and Haar null space" begin
    for level in 1:2
        mesh = reference_mesh(level)
        groups = boundary_face_groups(mesh)
        C = zeros(4, mesh.NumF)
        for (i, group) in enumerate(groups)
            C[i, group] .= 1.0
        end
        for basis in (:chain, :haar)
            Z = constraint_basis(mesh; basis)
            @test size(Z) == (mesh.NumF, mesh.NumF - 4)
            @test C * Matrix(Z) == zeros(4, mesh.NumF - 4)
            @test rank(Matrix(Z)) == mesh.NumF - 4
            @test C * ones(mesh.NumF) == fill((2^level)^2, 4)
            if basis == :haar
                @test Matrix(transpose(Z) * Z) ≈ I atol=5e-15 rtol=5e-15
            end
        end
    end
end

@testset "star/path geometry conventions" begin
    # Star: H is the Gram matrix of AB, AC, AD, all based at A.
    a, b, c = 37.0, 49.0, 58.0
    Hs = star_vertex_gram((a, b, c))
    @test mid(Hs[1, 2]) ≈ cosd(a)
    @test mid(Hs[1, 3]) ≈ cosd(b)
    @test mid(Hs[2, 3]) ≈ cosd(c)

    # Path: the assembly helper intentionally returns the vertex Gram based
    # at B (BA, BC, BD), not the path-edge Gram (AB, BC, CD) used for cover
    # comparisons in the manuscript. Check both conventions explicitly.
    t1, t2, phi = 40.0, 53.0, 37.0
    A = [cosd(t1), sind(t1), 0.0]
    B = zeros(3)
    Cpt = [1.0, 0.0, 0.0]
    D = [1 - cosd(t2), sind(t2) * cosd(phi), sind(t2) * sind(phi)]
    V = [transpose(A - B); transpose(Cpt - B); transpose(D - B)]
    Hv = V * transpose(V)
    Hpath = Kappa3Certificate.path_vertex_gram((t1, t2, phi))
    @test Matrix(mid.(Hpath)) ≈ Hv rtol=3e-15 atol=3e-15

    E = [transpose(B - A); transpose(Cpt - B); transpose(D - Cpt)]
    Gedge = E * transpose(E)
    x, y = cosd(t1), cosd(t2)
    z = x*y - sind(t1)*sind(t2)*cosd(phi)
    Gpaper = [1.0 -x z; -x 1.0 -y; z -y 1.0]
    @test Gedge ≈ Gpaper rtol=3e-15 atol=3e-15

    # The regular tetrahedron is represented correctly in both families.
    preg = acosd(1 / 3)
    @test Matrix(mid.(star_vertex_gram((60.0, 60.0, 60.0)))) ≈
          [1.0 .5 .5; .5 1.0 .5; .5 .5 1.0]
    @test Matrix(mid.(Kappa3Certificate.path_vertex_gram((60.0, 60.0, preg)))) ≈
          [1.0 .5 .5; .5 1.0 .5; .5 .5 1.0] atol=5e-16
end

@testset "diameter and constrained eigenvalue cross-check" begin
    L = [1.0     0.0    0.0;
         0.25    0.75   0.0;
        -0.125   0.375  0.5]
    H = interval.(L * transpose(L))
    mesh = reference_mesh(1)
    physical = affine_mesh(mesh, L)
    Mphysical, Aphysical, _ = create_matrix_crouzeix_raviart_3d(physical)
    Ametric, Mmetric = assemble_cr_metric(mesh, H)
    Z = constraint_basis(mesh; basis=:haar)

    lambda_metric = dense_smallest(Matrix(mid.(Ametric)), Matrix(mid.(Mmetric)), Z)
    lambda_physical = dense_smallest(Matrix(Aphysical), Matrix(Mphysical), Z)
    lambda_arpack = approximate_smallest_eigenvalue(Ametric, Mmetric, Z)
    @test lambda_metric ≈ lambda_physical rtol=2e-13
    @test lambda_arpack ≈ lambda_metric rtol=2e-10
    @test lambda_metric > 0

    lambda_lower, _, spd_cert = verified_lambda_lower(
        Ametric, Mmetric, Z, lambda_arpack; initial_backoff=1e-6)
    @test spd_cert.certified
    @test 0 < lambda_lower < lambda_metric

    direct_h2 = maximum(sum(abs2, physical.NodeList[e[2], :] -
                                  physical.NodeList[e[1], :])
                        for e in eachrow(physical.EdgeList))
    certified_h2 = hmax2_upper(mesh, H)
    @test certified_h2 >= direct_h2
    @test certified_h2 ≈ direct_h2 rtol=5e-15
end
