using Kappa3Certificate

code_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(code_root, "src", "ProofRecords.jl"))
using .ProofRecords

scratch_root = get(ENV, "KAPPA3_SCRATCH_ROOT", CANONICAL_SCRATCH_ROOT)
vfem_root = get(ENV, "KAPPA3_VFEM_ROOT", dependency_root("VFEM"))
veigs_root = get(ENV, "KAPPA3_VEIGS_ROOT", dependency_root("Veigs"))

layout = prepare_scratch_layout(scratch_root)
fingerprint_path = joinpath(layout.proof, "source-fingerprints.txt")
environment_path = joinpath(layout.proof, "environment.txt")

fingerprints = collect_standard_fingerprints(code_root; vfem_root, veigs_root)
save_fingerprints(fingerprint_path, fingerprints)
save_environment(environment_path, capture_environment())

println("Prepared the non-computational proof provenance files:")
println("  scratch root:       ", layout.root)
println("  source fingerprints: ", fingerprint_path)
println("  fingerprints SHA-256: ", sha256_file(fingerprint_path))
println("  environment:         ", environment_path)
println("  environment SHA-256: ", sha256_file(environment_path))

