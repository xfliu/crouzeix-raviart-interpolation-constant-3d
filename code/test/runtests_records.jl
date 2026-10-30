using Test
using IntervalArithmetic: interval, mid
using SparseArrays: sparse

include(joinpath(@__DIR__, "..", "src", "ProofRecords.jl"))
using .ProofRecords

@testset "proof-record provenance formats" begin
    mktempdir() do directory
        layout = prepare_scratch_layout(joinpath(directory, "scratch");
                                        allow_noncanonical = true)
        @test all(isdir, (layout.checkpoints, layout.logs, layout.points,
                          layout.proof, layout.temporary))

        environment1 = joinpath(directory, "environment.txt")
        environment2 = joinpath(directory, "environment-copy.txt")
        snapshot = capture_environment()
        save_environment(environment1, snapshot)
        loaded_snapshot = load_environment(environment1)
        save_environment(environment2, loaded_snapshot)
        @test read(environment1, String) == read(environment2, String)

        code = joinpath(directory, "code")
        vfem = joinpath(directory, "VFEM")
        veigs = joinpath(directory, "Veigs")
        for root in (code, vfem, veigs)
            mkpath(joinpath(root, "src"))
            write(joinpath(root, "Project.toml"), "name = \"$(basename(root))\"\n")
            write(joinpath(root, "src", "module.jl"), "module Example\nend\n")
        end
        write(joinpath(code, "Manifest.toml"), "julia_version = \"1.12.6\"\n")
        fingerprints = collect_standard_fingerprints(code; vfem_root = vfem,
                                                       veigs_root = veigs)
        fingerprint1 = joinpath(directory, "fingerprints.txt")
        fingerprint2 = joinpath(directory, "fingerprints-copy.txt")
        save_fingerprints(fingerprint1, fingerprints)
        loaded_fingerprints = load_fingerprints(fingerprint1)
        save_fingerprints(fingerprint2, loaded_fingerprints)
        @test read(fingerprint1, String) == read(fingerprint2, String)
        @test verify_standard_fingerprints(loaded_fingerprints, code;
                                           vfem_root = vfem, veigs_root = veigs)
        write(joinpath(vfem, "src", "module.jl"), "module Changed\nend\n")
        @test_throws ArgumentError verify_standard_fingerprints(
            loaded_fingerprints, code; vfem_root = vfem, veigs_root = veigs)
    end
end

@testset "point proof round trip and ARPACK-free replay" begin
    certificate = point_certificate(:star, (60.0, 60.0, 60.0);
                                    level = 1, basis = :haar,
                                    initial_backoff = 1e-3)
    H, inverse_metric = Kappa3Certificate._reference_geometry(
        certificate.family, certificate.parameters_deg)
    mesh = reference_mesh(certificate.refinement_level; metric = mid.(H))
    A, M = assemble_cr_metric(mesh, H; inverse_metric)
    Z = constraint_basis(mesh; basis = certificate.basis)
    AZ, MZ = A * Z, M * Z
    Braw = sparse(transpose(Z) *
                  (AZ - interval(certificate.lambda_lower) * MZ))
    B = sparse((Braw + transpose(Braw)) / interval(2.0))
    residual = certificate.spd_certificate
    legacy = Kappa3Certificate.RigorousSPD.certify_spd(B)
    @test residual.certified
    provenance_hash = repeat("1", 64)
    environment_hash = repeat("2", 64)
    @test_throws ArgumentError point_proof_record(
        "legacy must fail", certificate;
        fingerprints_sha256 = provenance_hash,
        environment_sha256 = environment_hash,
        spd_certificate = legacy)
    point = point_proof_record("regular level 1", certificate;
                               fingerprints_sha256 = provenance_hash,
                               environment_sha256 = environment_hash)

    mktempdir() do directory
        mkpath(joinpath(directory, "points"))
        point1 = joinpath(directory, "points", "regular.point")
        point2 = joinpath(directory, "points", "regular-copy.point")
        save_point_record(point1, point)
        loaded = load_point_record(point1)
        save_point_record(point2, loaded)
        @test read(point1, String) == read(point2, String)

        replayed = replay_point(loaded)
        @test replayed.id == point.id
        @test replayed.family == :star
        @test reinterpret(UInt64, replayed.u2_upper) ==
              reinterpret(UInt64, point.u2_upper)

        reference = reference_proof_record(point.id, "points/regular.point", point1)
        path_reversal = (0, -1, 1,
                         0, -1, 0,
                         1, -1, 0)
        reversed = reference_proof_record("regular via path reversal",
                                          "points/regular.point", point1;
                                          transform = path_reversal)
        @test reversed.source_point_id == point.id
        @test reversed.transform == path_reversal
        @test_throws ArgumentError ReferenceProofRecord(
            "singular transform", point.id, :star, "points/regular.point",
            sha256_file(point1), ntuple(_ -> 0, 9), point.gram, point.u2_upper)
        shear = (1, 1, 0,
                 0, 1, 0,
                 0, 0, 1)
        @test_throws ArgumentError ReferenceProofRecord(
            "generic shear", point.id, :star, "points/regular.point",
            sha256_file(point1), shear, point.gram, point.u2_upper)
        index1 = joinpath(directory, "references.txt")
        index2 = joinpath(directory, "references-copy.txt")
        save_reference_index(index1, ReferenceIndex([reference, reversed]))
        loaded_index = load_reference_index(index1)
        save_reference_index(index2, loaded_index)
        @test read(index1, String) == read(index2, String)
        points = replay_reference_index(index1;
                                        fingerprints_sha256 = provenance_hash,
                                        environment_sha256 = environment_hash)
        @test Set(keys(points)) == Set([point.id, reversed.id])
        @test points[reversed.id].u2_upper == points[point.id].u2_upper

        # A generic unimodular shear does not relabel tetrahedron vertices and
        # is rejected even though its determinant is one.
        tampered_index = joinpath(directory, "references-tampered.txt")
        identity_text = "1,0,0,0,1,0,0,0,1"
        shear_text = "1,1,0,0,1,0,0,0,1"
        write(tampered_index,
              replace(read(index1, String), identity_text => shear_text; count = 1))
        @test_throws ArgumentError replay_reference_index(
            tampered_index; fingerprints_sha256 = provenance_hash,
            environment_sha256 = environment_hash)

        # Strict readers reject extra records and changed point bytes.
        open(point2, "a") do io
            println(io, "trailing")
        end
        @test_throws ArgumentError load_point_record(point2)
        open(point1, "a") do io
            println(io)
        end
        @test_throws ArgumentError replay_reference_index(
            index1; fingerprints_sha256 = provenance_hash,
            environment_sha256 = environment_hash)
    end
end

@testset "proof bundle hashes" begin
    mktempdir() do directory
        paths = Dict{String, String}()
        for role in ("environment", "fingerprints", "path_cover",
                     "references", "star_cover")
            path = joinpath(directory, string(role, ".txt"))
            write(path, string(role, '\n'))
            paths[role] = path
        end
        links = [ArtifactLink(role, basename(paths[role]), sha256_file(paths[role]))
                 for role in keys(paths)]
        bundle1 = joinpath(directory, "bundle.txt")
        bundle2 = joinpath(directory, "bundle-copy.txt")
        save_proof_bundle(bundle1, ProofBundle(links))
        loaded = load_proof_bundle(bundle1)
        save_proof_bundle(bundle2, loaded)
        @test read(bundle1, String) == read(bundle2, String)
        @test length(verify_bundle_artifacts(bundle1, loaded)) == 5
        write(paths["star_cover"], "tampered\n")
        @test_throws ArgumentError verify_bundle_artifacts(bundle1, loaded)
    end
end
