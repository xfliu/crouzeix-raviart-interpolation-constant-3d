using Kappa3Certificate

code_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(code_root, "src", "ProofRecords.jl"))
include(joinpath(code_root, "src", "BoxCover.jl"))
include(joinpath(code_root, "src", "CoverSearch.jl"))

using .ProofRecords

# The residual-SPD proof assumes round-to-nearest binary64 arithmetic,
# gradual underflow, and a single-threaded BLAS.  Recheck those properties in
# this independent process rather than trusting the historical snapshot.
Kappa3Certificate.RigorousSPD.ieee_preflight()

bundle_path = isempty(ARGS) ?
    joinpath(CANONICAL_SCRATCH_ROOT, "proof-bundle.txt") : abspath(only(ARGS))
vfem_root = get(ENV, "KAPPA3_VFEM_ROOT", dependency_root("VFEM"))
veigs_root = get(ENV, "KAPPA3_VEIGS_ROOT", dependency_root("Veigs"))

bundle = load_proof_bundle(bundle_path)
artifacts = verify_bundle_artifacts(bundle_path, bundle)

fingerprints = load_fingerprints(artifacts["fingerprints"])
verify_standard_fingerprints(fingerprints, code_root; vfem_root, veigs_root)
load_environment(artifacts["environment"]) # strict parse; historical, not host equality

star_record = Main.CoverSearch.load_cover_record(artifacts["star_cover"])
path_record = Main.CoverSearch.load_cover_record(artifacts["path_cover"])
star_target = Main.BoxCover.STAR_TARGET
path_target = Main.BoxCover.PATH_TARGET

# These checks are deliberately ahead of the expensive point replays.  A
# partial domain, a swapped family, or a relaxed target is not this theorem.
star_record.family == :star || error("star cover has the wrong family")
star_record.root == Main.BoxCover.star_root_box() ||
    error("star cover does not have the canonical [0,60]^3 root")
star_record.target == star_target == 799 // 5000 ||
    error("star cover target is not exactly 799/5000")
path_record.family == :path || error("path cover has the wrong family")
path_record.root == Main.BoxCover.path_root_box() ||
    error("path cover does not have the canonical [0,60]^2 x [0,90] root")
path_record.target == path_target == 31 // 200 ||
    error("path cover target is not exactly 31/200")

fingerprints_hash = sha256_file(artifacts["fingerprints"])
environment_hash = sha256_file(artifacts["environment"])
points = replay_reference_index(artifacts["references"];
                                fingerprints_sha256 = fingerprints_hash,
                                environment_sha256 = environment_hash)

base_references = Dict(
    id => Main.BoxCover.ReferenceBound(id, point.family,
                                       point.gram, point.u2_upper)
    for (id, point) in points
)
references, _ = Main.CoverSearch.expanded_reference_catalog(base_references)

star_result = Main.CoverSearch.replay_record(star_record, references)
star_result.passed || error("star cover replay failed:\n" * join(star_result.failures, '\n'))
path_result = Main.CoverSearch.replay_record(path_record, references)
path_result.passed || error("path cover replay failed:\n" * join(path_result.failures, '\n'))

println("KAPPA3 PROOF REPLAY PASSED")
println("  exact targets: star 799/5000, path 31/200")
println("  replayed point proofs: ", length(points))
println("  star retained leaves/corners: ", star_result.checked_leaves,
        "/", star_result.checked_corners)
println("  path retained leaves/corners: ", path_result.checked_leaves,
        "/", path_result.checked_corners)
