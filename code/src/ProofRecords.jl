module ProofRecords

using SHA: sha256
using IntervalArithmetic: Interval, interval, inf, isguaranteed, mid, sup
using LinearAlgebra
using SparseArrays
import Kappa3Certificate

export dependency_root,
       CANONICAL_SCRATCH_ROOT, ScratchLayout, prepare_scratch_layout,
       FileDigest, FingerprintRecord, collect_standard_fingerprints,
       save_fingerprints, load_fingerprints, verify_standard_fingerprints,
       EnvironmentSnapshot, capture_environment, save_environment,
       load_environment, PointProofRecord, point_proof_record,
       save_point_record, load_point_record, ReplayedPoint, replay_point,
       ReferenceProofRecord, ReferenceIndex, reference_proof_record,
       save_reference_index, load_reference_index, replay_reference_index,
       ArtifactLink, ProofBundle, save_proof_bundle, load_proof_bundle,
       verify_bundle_artifacts, sha256_file

const IA = Interval{Float64}
const CANONICAL_SCRATCH_ROOT = "/mnt/ssd2T-2/xfliu/kappa3_cr_constant_3d"
const _POINT_MAGIC = "KAPPA3_POINT_PROOF_V1"
const _REFERENCE_MAGIC = "KAPPA3_REFERENCE_INDEX_V1"
const _FINGERPRINT_MAGIC = "KAPPA3_SOURCE_FINGERPRINTS_V1"
const _ENVIRONMENT_MAGIC = "KAPPA3_ENVIRONMENT_SNAPSHOT_V1"
const _BUNDLE_MAGIC = "KAPPA3_PROOF_BUNDLE_V1"
const _BUNDLE_ROLES = ("environment", "fingerprints", "path_cover",
                       "references", "star_cover")

# ---------------------------------------------------------------------------
# Strict scalar and text encoding
# ---------------------------------------------------------------------------

"""Return the lowercase SHA-256 digest of a regular file."""
function sha256_file(path::AbstractString)
    isfile(path) || throw(ArgumentError("not a regular file: '$path'"))
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

_valid_sha256(text::AbstractString) = occursin(r"^[0-9a-f]{64}$", text)

function _require_sha256(text::AbstractString, field::AbstractString)
    value = String(text)
    _valid_sha256(value) || throw(ArgumentError("$field is not a lowercase SHA-256 digest"))
    return value
end

function _atomic_write(writer, path::AbstractString)
    target = abspath(path)
    mkpath(dirname(target))
    temp = string(target, ".tmp.", getpid())
    ispath(temp) && throw(ArgumentError("temporary output already exists: '$temp'"))
    try
        open(temp, "w") do io
            writer(io)
            flush(io)
        end
        mv(temp, target; force = true)
    finally
        ispath(temp) && rm(temp; force = true)
    end
    return target
end

# Percent encoding keeps records readable while making tabs, newlines,
# non-ASCII bytes, and '%' unambiguous.  Loading requires canonical encoding.
function _escape(text::AbstractString)
    io = IOBuffer()
    for byte in codeunits(String(text))
        if UInt8(0x20) <= byte <= UInt8(0x7e) && byte != UInt8('%') &&
           byte != UInt8('\t')
            write(io, byte)
        else
            print(io, '%', uppercase(string(byte; base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

function _unescape(text::AbstractString)
    bytes = UInt8[]
    i = firstindex(text)
    while i <= lastindex(text)
        if text[i] == '%'
            i1 = nextind(text, i)
            i2 = nextind(text, i1)
            i2 <= lastindex(text) || throw(ArgumentError("truncated percent escape '$text'"))
            token = String(text[i1:i2])
            occursin(r"^[0-9A-F]{2}$", token) ||
                throw(ArgumentError("noncanonical percent escape in '$text'"))
            push!(bytes, parse(UInt8, token; base = 16))
            i = nextind(text, i2)
        else
            c = text[i]
            isascii(c) && isprint(c) && c != '\t' ||
                throw(ArgumentError("unescaped control or non-ASCII character in '$text'"))
            push!(bytes, UInt8(c))
            i = nextind(text, i)
        end
    end
    value = try
        String(bytes)
    catch
        throw(ArgumentError("invalid UTF-8 in escaped text"))
    end
    _escape(value) == text || throw(ArgumentError("noncanonical escaped text '$text'"))
    return value
end

function _safe_relative_path(text::AbstractString)
    path = String(text)
    isempty(path) && throw(ArgumentError("an artifact path must not be empty"))
    isabspath(path) && throw(ArgumentError("artifact path must be relative: '$path'"))
    occursin('\\', path) && throw(ArgumentError("backslashes are not allowed in artifact paths"))
    normpath(path) == path || throw(ArgumentError("artifact path is not normalized: '$path'"))
    any(part -> part in ("", ".", ".."), split(path, '/')) &&
        throw(ArgumentError("unsafe artifact path '$path'"))
    return path
end

function _resolve_relative(base::AbstractString, relative::AbstractString)
    rel = _safe_relative_path(relative)
    root = abspath(base)
    path = abspath(joinpath(root, rel))
    (path == root || startswith(path, string(root, '/'))) ||
        throw(ArgumentError("artifact escapes its bundle directory: '$relative'"))
    return path
end

function _float_text(x::Float64)
    isfinite(x) || throw(ArgumentError("proof records require finite Float64 values"))
    bits = string(reinterpret(UInt64, x); base = 16, pad = 16)
    return string(repr(x), "@0x", bits)
end

function _parse_float(text::AbstractString)
    fields = split(text, '@'; keepempty = true)
    length(fields) == 2 || throw(ArgumentError("malformed Float64 token '$text'"))
    occursin(r"^0x[0-9a-f]{16}$", fields[2]) ||
        throw(ArgumentError("malformed Float64 bit pattern in '$text'"))
    bits = parse(UInt64, fields[2][3:end]; base = 16)
    value = reinterpret(Float64, bits)
    isfinite(value) || throw(ArgumentError("nonfinite Float64 in proof record"))
    parsed = try
        parse(Float64, fields[1])
    catch
        throw(ArgumentError("malformed decimal Float64 in '$text'"))
    end
    reinterpret(UInt64, parsed) == bits ||
        throw(ArgumentError("decimal and bit encodings disagree in '$text'"))
    _float_text(value) == text || throw(ArgumentError("noncanonical Float64 token '$text'"))
    return value
end

function _int_text(x::Integer)
    return string(x)
end

function _parse_int(text::AbstractString)
    occursin(r"^(0|-?[1-9][0-9]*)$", text) ||
        throw(ArgumentError("noncanonical integer '$text'"))
    return parse(Int, text)
end

_bool_text(x::Bool) = x ? "true" : "false"
function _parse_bool(text::AbstractString)
    text == "true" && return true
    text == "false" && return false
    throw(ArgumentError("invalid Boolean '$text'"))
end

function _symbol_text(x::Symbol)
    text = String(x)
    occursin(r"^[a-z][a-z0-9_]*$", text) ||
        throw(ArgumentError("noncanonical proof-record symbol '$x'"))
    return text
end

function _parse_symbol(text::AbstractString)
    occursin(r"^[a-z][a-z0-9_]*$", text) ||
        throw(ArgumentError("invalid proof-record symbol '$text'"))
    return Symbol(text)
end

function _tuple3_text(x::NTuple{3, Float64})
    return join(_float_text.(x), ',')
end

function _parse_tuple3(text::AbstractString)
    fields = split(text, ','; keepempty = true)
    length(fields) == 3 || throw(ArgumentError("expected three Float64 values"))
    values = _parse_float.(fields)
    return (values[1], values[2], values[3])
end

function _interval_matrix(A::AbstractMatrix)
    size(A) == (3, 3) || throw(DimensionMismatch("a proof Gram matrix must be 3 by 3"))
    G = Matrix{IA}(undef, 3, 3)
    for j in 1:3, i in 1:3
        x = A[i, j]
        if x isa Interval
            isguaranteed(x) || throw(ArgumentError("Gram entries must be guaranteed intervals"))
            G[i, j] = interval(Float64, inf(x), sup(x))
        else
            G[i, j] = interval(Float64, x)
        end
        isfinite(inf(G[i, j])) && isfinite(sup(G[i, j])) ||
            throw(ArgumentError("Gram endpoints must be finite"))
    end
    return G
end

function _gram_text(G::Matrix{IA})
    cells = String[]
    for j in 1:3, i in 1:3
        push!(cells, string(_float_text(inf(G[i, j])), "..",
                            _float_text(sup(G[i, j]))))
    end
    return join(cells, ',')
end

function _parse_gram(text::AbstractString)
    cells = split(text, ','; keepempty = true)
    length(cells) == 9 || throw(ArgumentError("a Gram record needs nine entries"))
    G = Matrix{IA}(undef, 3, 3)
    k = 0
    for j in 1:3, i in 1:3
        k += 1
        endpoints = split(cells[k], ".."; keepempty = true)
        length(endpoints) == 2 || throw(ArgumentError("malformed Gram interval"))
        lo, hi = _parse_float(endpoints[1]), _parse_float(endpoints[2])
        lo <= hi || throw(ArgumentError("reversed Gram interval"))
        G[i, j] = interval(lo, hi)
    end
    return G
end

function _same_float(a::Float64, b::Float64)
    return reinterpret(UInt64, a) == reinterpret(UInt64, b)
end

function _same_gram(A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) == (3, 3) || return false
    return all(_same_float(inf(A[i, j]), inf(B[i, j])) &&
               _same_float(sup(A[i, j]), sup(B[i, j]))
               for i in 1:3, j in 1:3)
end

function _strict_key_values(lines::Vector{String}, magic::String,
                            keys::Vector{String})
    length(lines) == length(keys) + 2 || throw(ArgumentError("wrong record length for $magic"))
    lines[1] == magic || throw(ArgumentError("not a $magic record"))
    lines[end] == "END" || throw(ArgumentError("record has no END marker"))
    values = Dict{String, String}()
    for (line, expected) in zip(lines[2:end-1], keys)
        fields = split(line, '\t'; keepempty = true)
        length(fields) == 2 || throw(ArgumentError("malformed '$expected' record"))
        fields[1] == expected || throw(ArgumentError("expected '$expected', found '$(fields[1])'"))
        values[expected] = fields[2]
    end
    return values
end

# ---------------------------------------------------------------------------
# Source fingerprints and environment
# ---------------------------------------------------------------------------

struct FileDigest
    label::String
    sha256::String
    bytes::Int
    function FileDigest(label::AbstractString, digest::AbstractString, bytes::Integer)
        name = _safe_relative_path(label)
        hash = _require_sha256(digest, "file digest")
        bytes >= 0 || throw(ArgumentError("file size must be nonnegative"))
        new(name, hash, Int(bytes))
    end
end

struct FingerprintRecord
    files::Vector{FileDigest}
    function FingerprintRecord(files::AbstractVector{FileDigest})
        entries = sort!(collect(files); by = x -> x.label)
        labels = getfield.(entries, :label)
        length(unique(labels)) == length(labels) ||
            throw(ArgumentError("duplicate source-fingerprint label"))
        new(entries)
    end
end

function _tree_files(root::AbstractString, relative_roots::Tuple)
    result = String[]
    for rel in relative_roots
        start = joinpath(root, rel)
        ispath(start) || continue
        if isfile(start)
            push!(result, rel)
            continue
        end
        for (directory, dirs, files) in walkdir(start; follow_symlinks = false)
            sort!(dirs); sort!(files)
            for name in files
                path = joinpath(directory, name)
                islink(path) && throw(ArgumentError("source fingerprint refuses symlink '$path'"))
                extension = lowercase(splitext(name)[2])
                extension in (".jl", ".toml", ".sh") || continue
                push!(result, relpath(path, root))
            end
        end
    end
    return sort!(unique(result))
end

function _digest_tree!(entries, prefix::String, root::AbstractString,
                       relative_roots::Tuple)
    isdir(root) || throw(ArgumentError("source root is absent: '$root'"))
    for rel in _tree_files(root, relative_roots)
        path = joinpath(root, rel)
        push!(entries, FileDigest(joinpath(prefix, rel), sha256_file(path), filesize(path)))
    end
end

"""
    dependency_root(name)

Installation directory of the package `name` in the active environment, for
example the checkout of VFEM or Veigs that `Pkg` installed from GitHub.
"""
function dependency_root(name::AbstractString)
    id = Base.identify_package(name)
    isnothing(id) && throw(ArgumentError("package $name is not in the active environment"))
    entry = Base.locate_package(id)
    isnothing(entry) && throw(ArgumentError("package $name is not installed"))
    return dirname(dirname(entry))
end

"""
Hash all verifier sources, the exact Project/Manifest, and the Julia sources
of the locally developed VFEM and Veigs packages.  Labels are portable and
sorted; absolute installation paths are deliberately not serialized.
"""
function collect_standard_fingerprints(code_root::AbstractString;
                                       vfem_root::AbstractString,
                                       veigs_root::AbstractString)
    entries = FileDigest[]
    _digest_tree!(entries, "code", code_root,
                  ("Project.toml", "Manifest.toml", "setup.jl", "src", "scripts", "test"))
    _digest_tree!(entries, "VFEM", vfem_root,
                  ("Project.toml", "Manifest.toml", "src"))
    _digest_tree!(entries, "Veigs", veigs_root,
                  ("Project.toml", "Manifest.toml", "src"))
    return FingerprintRecord(entries)
end

function save_fingerprints(path::AbstractString, record::FingerprintRecord)
    return _atomic_write(path) do io
        println(io, _FINGERPRINT_MAGIC)
        println(io, "files\t", length(record.files))
        for file in record.files
            println(io, join(("F", _escape(file.label), file.sha256,
                              _int_text(file.bytes)), '\t'))
        end
        println(io, "END")
    end
end

function load_fingerprints(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 3 && lines[1] == _FINGERPRINT_MAGIC ||
        throw(ArgumentError("not a Kappa3 source-fingerprint record"))
    header = split(lines[2], '\t'; keepempty = true)
    length(header) == 2 && header[1] == "files" ||
        throw(ArgumentError("missing source-fingerprint count"))
    count = _parse_int(header[2])
    count >= 0 || throw(ArgumentError("negative source-fingerprint count"))
    length(lines) == count + 3 || throw(ArgumentError("source-fingerprint count mismatch"))
    lines[end] == "END" || throw(ArgumentError("source-fingerprint record has no END marker"))
    entries = FileDigest[]
    for line in lines[3:end-1]
        fields = split(line, '\t'; keepempty = true)
        length(fields) == 4 && fields[1] == "F" ||
            throw(ArgumentError("malformed source-fingerprint entry"))
        push!(entries, FileDigest(_unescape(fields[2]), fields[3], _parse_int(fields[4])))
    end
    labels = getfield.(entries, :label)
    issorted(labels) || throw(ArgumentError("source fingerprints are not canonically sorted"))
    return FingerprintRecord(entries)
end

function verify_standard_fingerprints(expected::FingerprintRecord,
                                      code_root::AbstractString;
                                      vfem_root::AbstractString,
                                      veigs_root::AbstractString)
    actual = collect_standard_fingerprints(code_root; vfem_root, veigs_root)
    length(actual.files) == length(expected.files) ||
        throw(ArgumentError("source fingerprint file set differs from the recorded set"))
    for (want, got) in zip(expected.files, actual.files)
        want.label == got.label || throw(ArgumentError("source file set differs at '$(want.label)'"))
        want.bytes == got.bytes || throw(ArgumentError("source size mismatch for '$(want.label)'"))
        want.sha256 == got.sha256 || throw(ArgumentError("source SHA-256 mismatch for '$(want.label)'"))
    end
    return true
end

struct EnvironmentSnapshot
    entries::Vector{Pair{String, String}}
    function EnvironmentSnapshot(entries)
        pairs = Pair{String, String}[String(first(p)) => String(last(p)) for p in entries]
        sort!(pairs; by = first)
        keys = first.(pairs)
        all(k -> occursin(r"^[a-z][a-z0-9_]*$", k), keys) ||
            throw(ArgumentError("environment keys must be lowercase identifiers"))
        length(unique(keys)) == length(keys) || throw(ArgumentError("duplicate environment key"))
        new(pairs)
    end
end

function capture_environment()
    cpu_model = isempty(Sys.cpu_info()) ? "unknown" : String(first(Sys.cpu_info()).model)
    active_project = something(Base.active_project(), "none")
    entries = [
        "active_project" => active_project,
        "blas_config" => sprint(show, LinearAlgebra.BLAS.get_config()),
        "cpu_model" => cpu_model,
        "cpu_threads" => string(Sys.CPU_THREADS),
        "hostname" => gethostname(),
        "julia_bindir" => Sys.BINDIR,
        "julia_threads" => string(Threads.nthreads()),
        "julia_version" => string(VERSION),
        "machine" => String(Sys.MACHINE),
        "omp_num_threads" => get(ENV, "OMP_NUM_THREADS", "unset"),
        "openblas_num_threads" => get(ENV, "OPENBLAS_NUM_THREADS", "unset"),
        "word_size" => string(Sys.WORD_SIZE),
    ]
    return EnvironmentSnapshot(entries)
end

function save_environment(path::AbstractString, snapshot::EnvironmentSnapshot)
    return _atomic_write(path) do io
        println(io, _ENVIRONMENT_MAGIC)
        println(io, "entries\t", length(snapshot.entries))
        for (key, value) in snapshot.entries
            println(io, join(("E", key, _escape(value)), '\t'))
        end
        println(io, "END")
    end
end

function load_environment(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 3 && lines[1] == _ENVIRONMENT_MAGIC ||
        throw(ArgumentError("not a Kappa3 environment snapshot"))
    header = split(lines[2], '\t'; keepempty = true)
    length(header) == 2 && header[1] == "entries" ||
        throw(ArgumentError("missing environment entry count"))
    count = _parse_int(header[2])
    count >= 0 || throw(ArgumentError("negative environment entry count"))
    length(lines) == count + 3 || throw(ArgumentError("environment entry count mismatch"))
    lines[end] == "END" || throw(ArgumentError("environment snapshot has no END marker"))
    entries = Pair{String, String}[]
    for line in lines[3:end-1]
        fields = split(line, '\t'; keepempty = true)
        length(fields) == 3 && fields[1] == "E" ||
            throw(ArgumentError("malformed environment entry"))
        push!(entries, fields[2] => _unescape(fields[3]))
    end
    issorted(first.(entries)) || throw(ArgumentError("environment entries are not sorted"))
    return EnvironmentSnapshot(entries)
end

# ---------------------------------------------------------------------------
# Point records and an ARPACK-free replay
# ---------------------------------------------------------------------------

Base.@kwdef struct PointProofRecord
    id::String
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
    spd_method::Symbol
    spd_initial_rho::Float64
    spd_max_attempts::Int
    spd_reason::Symbol
    spd_n::Int
    spd_nnz::Int
    spd_cholesky_completed::Bool
    spd_iterations::Int
    spd_rho::Float64
    spd_input_radius_bound::Float64
    spd_factor_product_norm_bound::Float64
    spd_gamma_bound::Float64
    spd_roundoff_bound::Float64
    spd_underflow_bound::Float64
    spd_total_error_bound::Float64
    spd_strict_margin_lower::Float64
    spd_ordering::Symbol
    spd_permutation_matched::Bool
    spd_factor_nnz::Int
    spd_scaled::Bool
    spd_scale_exponent_min::Int
    spd_scale_exponent_max::Int
    gram::Matrix{IA}
    fingerprints_sha256::String
    environment_sha256::String
end

const _POINT_KEYS = [
    "id", "family", "parameters_deg", "refinement_level", "elements",
    "dofs_before_constraints", "dofs_after_constraints", "lambda_approx",
    "lambda_lower", "relative_backoff", "hmax2_upper", "u2_upper", "C_upper",
    "basis", "mesh_strategy", "spd_method", "spd_initial_rho",
    "spd_max_attempts", "spd_reason", "spd_n", "spd_nnz",
    "spd_cholesky_completed", "spd_iterations", "spd_rho",
    "spd_input_radius_bound", "spd_factor_product_norm_bound", "spd_gamma_bound",
    "spd_roundoff_bound", "spd_underflow_bound", "spd_total_error_bound",
    "spd_strict_margin_lower", "spd_ordering", "spd_permutation_matched",
    "spd_factor_nnz", "spd_scaled", "spd_scale_exponent_min",
    "spd_scale_exponent_max", "gram",
    "fingerprints_sha256", "environment_sha256",
]

function _validate_point(record::PointProofRecord)
    isempty(record.id) && throw(ArgumentError("point id must not be empty"))
    record.family in (:star, :path) || throw(ArgumentError("point family must be star or path"))
    all(isfinite, record.parameters_deg) || throw(ArgumentError("point angles must be finite"))
    if record.family == :star
        all(0.0 < x <= 60.0 for x in record.parameters_deg) ||
            throw(ArgumentError("star point angles must lie in (0,60]"))
        all(record.parameters_deg[i] < record.parameters_deg[j] + record.parameters_deg[k]
            for (i, j, k) in ((1, 2, 3), (2, 1, 3), (3, 1, 2))) ||
            throw(ArgumentError("star point angles violate a strict triangle inequality"))
    else
        0.0 < record.parameters_deg[1] <= 60.0 &&
        0.0 < record.parameters_deg[2] <= 60.0 &&
        0.0 < record.parameters_deg[3] <= 90.0 ||
            throw(ArgumentError("path point angles are outside the canonical domain"))
    end
    record.refinement_level >= 0 || throw(ArgumentError("negative refinement level"))
    record.elements > 0 || throw(ArgumentError("point record has no elements"))
    record.dofs_before_constraints > 4 || throw(ArgumentError("too few unconstrained DOFs"))
    record.dofs_after_constraints == record.dofs_before_constraints - 4 ||
        throw(ArgumentError("point record does not impose exactly four constraints"))
    record.lambda_approx > 0 && record.lambda_lower > 0 ||
        throw(ArgumentError("point eigenvalues must be positive"))
    record.lambda_lower <= record.lambda_approx ||
        throw(ArgumentError("lambda_lower exceeds the approximate eigenvalue"))
    record.relative_backoff > 0 || throw(ArgumentError("relative backoff must be positive"))
    all(x -> isfinite(x) && x > 0,
        (record.hmax2_upper, record.u2_upper, record.C_upper)) ||
        throw(ArgumentError("point upper bounds must be finite and positive"))
    record.basis in (:chain, :haar) || throw(ArgumentError("unknown constraint basis"))
    record.mesh_strategy in (:metric, :fixed) || throw(ArgumentError("unknown mesh strategy"))
    record.spd_method == :residual ||
        throw(ArgumentError("final point records require the residual SPD method"))
    isfinite(record.spd_initial_rho) && record.spd_initial_rho >= 0 ||
        throw(ArgumentError("SPD initial rho must be finite and nonnegative"))
    record.spd_max_attempts >= 1 || throw(ArgumentError("SPD max attempts must be positive"))
    record.spd_reason in (:certified, :vacuous) ||
        throw(ArgumentError("point record does not contain a successful SPD proof"))
    record.spd_cholesky_completed || throw(ArgumentError("SPD Cholesky did not complete"))
    record.spd_n == record.dofs_after_constraints ||
        throw(ArgumentError("SPD dimension disagrees with constrained DOFs"))
    record.spd_nnz >= record.spd_n || throw(ArgumentError("invalid SPD nonzero count"))
    record.spd_ordering in (:amd, :natural) || throw(ArgumentError("unknown SPD ordering"))
    record.spd_iterations >= 1 || throw(ArgumentError("invalid SPD iteration count"))
    record.spd_iterations <= record.spd_max_attempts ||
        throw(ArgumentError("SPD iterations exceed max attempts"))
    record.spd_permutation_matched || throw(ArgumentError("SPD permutation did not match"))
    record.spd_factor_nnz >= record.spd_n || throw(ArgumentError("invalid factor nonzero count"))
    all(x -> isfinite(x) && x >= 0,
        (record.spd_rho, record.spd_input_radius_bound,
         record.spd_factor_product_norm_bound, record.spd_gamma_bound,
         record.spd_roundoff_bound, record.spd_underflow_bound,
         record.spd_total_error_bound, record.spd_strict_margin_lower)) ||
        throw(ArgumentError("invalid residual SPD audit value"))
    record.spd_strict_margin_lower > 0 ||
        throw(ArgumentError("residual SPD certificate has no strict positive margin"))
    _interval_matrix(record.gram)
    _require_sha256(record.fingerprints_sha256, "fingerprints_sha256")
    _require_sha256(record.environment_sha256, "environment_sha256")
    return record
end

"""Convert a successful `PointCertificate` into a provenance-bound record."""
function point_proof_record(id::AbstractString, certificate;
                            fingerprints_sha256::AbstractString,
                            environment_sha256::AbstractString,
                            spd_certificate = certificate.spd_certificate,
                            spd_initial_rho::Float64 = 0.0,
                            spd_max_attempts::Int = 8)
    family = certificate.family
    gram = family == :star ?
        Kappa3Certificate.star_vertex_gram(certificate.parameters_deg) :
        Kappa3Certificate.path_vertex_gram(certificate.parameters_deg)
    spd = spd_certificate
    spd isa Kappa3Certificate.RigorousSPD.SPDResidualCertificate ||
        throw(ArgumentError("final point records require SPDResidualCertificate"))
    spd.certified || throw(ArgumentError("cannot record an uncertified point"))
    record = PointProofRecord(
        id = String(id), family = family,
        parameters_deg = certificate.parameters_deg,
        refinement_level = certificate.refinement_level,
        elements = certificate.elements,
        dofs_before_constraints = certificate.dofs_before_constraints,
        dofs_after_constraints = certificate.dofs_after_constraints,
        lambda_approx = certificate.lambda_approx,
        lambda_lower = certificate.lambda_lower,
        relative_backoff = certificate.relative_backoff,
        hmax2_upper = certificate.hmax2_upper,
        u2_upper = certificate.u2_upper, C_upper = certificate.C_upper,
        basis = certificate.basis, mesh_strategy = certificate.mesh_strategy,
        spd_method = :residual, spd_initial_rho = spd_initial_rho,
        spd_max_attempts = spd_max_attempts,
        spd_reason = spd.reason, spd_n = spd.n, spd_nnz = spd.nnz,
        spd_cholesky_completed = spd.cholesky_completed,
        spd_iterations = spd.iterations, spd_rho = spd.rho,
        spd_input_radius_bound = spd.input_radius_bound,
        spd_factor_product_norm_bound = spd.factor_product_norm_bound,
        spd_gamma_bound = spd.gamma_bound,
        spd_roundoff_bound = spd.roundoff_bound,
        spd_underflow_bound = spd.underflow_bound,
        spd_total_error_bound = spd.total_error_bound,
        spd_strict_margin_lower = spd.strict_margin_lower,
        spd_ordering = spd.ordering,
        spd_permutation_matched = spd.permutation_matched,
        spd_factor_nnz = spd.factor_nnz, spd_scaled = spd.scaled,
        spd_scale_exponent_min = spd.scale_exponent_min,
        spd_scale_exponent_max = spd.scale_exponent_max,
        gram = _interval_matrix(gram),
        fingerprints_sha256 = String(fingerprints_sha256),
        environment_sha256 = String(environment_sha256),
    )
    return _validate_point(record)
end

function _point_values(r::PointProofRecord)
    return String[
        _escape(r.id), _symbol_text(r.family), _tuple3_text(r.parameters_deg),
        _int_text(r.refinement_level), _int_text(r.elements),
        _int_text(r.dofs_before_constraints), _int_text(r.dofs_after_constraints),
        _float_text(r.lambda_approx), _float_text(r.lambda_lower),
        _float_text(r.relative_backoff), _float_text(r.hmax2_upper),
        _float_text(r.u2_upper), _float_text(r.C_upper), _symbol_text(r.basis),
        _symbol_text(r.mesh_strategy), _symbol_text(r.spd_method),
        _float_text(r.spd_initial_rho), _int_text(r.spd_max_attempts),
        _symbol_text(r.spd_reason),
        _int_text(r.spd_n), _int_text(r.spd_nnz),
        _bool_text(r.spd_cholesky_completed), _int_text(r.spd_iterations),
        _float_text(r.spd_rho), _float_text(r.spd_input_radius_bound),
        _float_text(r.spd_factor_product_norm_bound), _float_text(r.spd_gamma_bound),
        _float_text(r.spd_roundoff_bound), _float_text(r.spd_underflow_bound),
        _float_text(r.spd_total_error_bound), _float_text(r.spd_strict_margin_lower),
        _symbol_text(r.spd_ordering), _bool_text(r.spd_permutation_matched),
        _int_text(r.spd_factor_nnz), _bool_text(r.spd_scaled),
        _int_text(r.spd_scale_exponent_min), _int_text(r.spd_scale_exponent_max),
        _gram_text(r.gram), r.fingerprints_sha256, r.environment_sha256,
    ]
end

function save_point_record(path::AbstractString, record::PointProofRecord)
    _validate_point(record)
    values = _point_values(record)
    return _atomic_write(path) do io
        println(io, _POINT_MAGIC)
        for (key, value) in zip(_POINT_KEYS, values)
            println(io, key, '\t', value)
        end
        println(io, "END")
    end
end

function load_point_record(path::AbstractString)
    values = _strict_key_values(readlines(path), _POINT_MAGIC, _POINT_KEYS)
    record = PointProofRecord(
        id = _unescape(values["id"]), family = _parse_symbol(values["family"]),
        parameters_deg = _parse_tuple3(values["parameters_deg"]),
        refinement_level = _parse_int(values["refinement_level"]),
        elements = _parse_int(values["elements"]),
        dofs_before_constraints = _parse_int(values["dofs_before_constraints"]),
        dofs_after_constraints = _parse_int(values["dofs_after_constraints"]),
        lambda_approx = _parse_float(values["lambda_approx"]),
        lambda_lower = _parse_float(values["lambda_lower"]),
        relative_backoff = _parse_float(values["relative_backoff"]),
        hmax2_upper = _parse_float(values["hmax2_upper"]),
        u2_upper = _parse_float(values["u2_upper"]),
        C_upper = _parse_float(values["C_upper"]),
        basis = _parse_symbol(values["basis"]),
        mesh_strategy = _parse_symbol(values["mesh_strategy"]),
        spd_method = _parse_symbol(values["spd_method"]),
        spd_initial_rho = _parse_float(values["spd_initial_rho"]),
        spd_max_attempts = _parse_int(values["spd_max_attempts"]),
        spd_reason = _parse_symbol(values["spd_reason"]),
        spd_n = _parse_int(values["spd_n"]), spd_nnz = _parse_int(values["spd_nnz"]),
        spd_cholesky_completed = _parse_bool(values["spd_cholesky_completed"]),
        spd_iterations = _parse_int(values["spd_iterations"]),
        spd_rho = _parse_float(values["spd_rho"]),
        spd_input_radius_bound = _parse_float(values["spd_input_radius_bound"]),
        spd_factor_product_norm_bound = _parse_float(values["spd_factor_product_norm_bound"]),
        spd_gamma_bound = _parse_float(values["spd_gamma_bound"]),
        spd_roundoff_bound = _parse_float(values["spd_roundoff_bound"]),
        spd_underflow_bound = _parse_float(values["spd_underflow_bound"]),
        spd_total_error_bound = _parse_float(values["spd_total_error_bound"]),
        spd_strict_margin_lower = _parse_float(values["spd_strict_margin_lower"]),
        spd_ordering = _parse_symbol(values["spd_ordering"]),
        spd_permutation_matched = _parse_bool(values["spd_permutation_matched"]),
        spd_factor_nnz = _parse_int(values["spd_factor_nnz"]),
        spd_scaled = _parse_bool(values["spd_scaled"]),
        spd_scale_exponent_min = _parse_int(values["spd_scale_exponent_min"]),
        spd_scale_exponent_max = _parse_int(values["spd_scale_exponent_max"]),
        gram = _parse_gram(values["gram"]),
        fingerprints_sha256 = _require_sha256(values["fingerprints_sha256"], "fingerprints_sha256"),
        environment_sha256 = _require_sha256(values["environment_sha256"], "environment_sha256"),
    )
    return _validate_point(record)
end

struct ReplayedPoint
    id::String
    family::Symbol
    gram::Matrix{IA}
    hmax2_upper::Float64
    u2_upper::Float64
    C_upper::Float64
    spd_certificate
end

function _require_same_float(label, actual::Float64, expected::Float64)
    _same_float(actual, expected) ||
        throw(ArgumentError("point replay mismatch for $label: got $actual, recorded $expected"))
end

function _check_spd_replay(actual, r::PointProofRecord)
    actual.certified || throw(ArgumentError("stored lambda_lower failed the replayed SPD proof"))
    actual.reason == r.spd_reason || throw(ArgumentError("SPD reason changed on replay"))
    actual.n == r.spd_n || throw(ArgumentError("SPD dimension changed on replay"))
    actual.nnz == r.spd_nnz || throw(ArgumentError("SPD sparsity changed on replay"))
    actual.cholesky_completed == r.spd_cholesky_completed ||
        throw(ArgumentError("SPD Cholesky status changed on replay"))
    actual.iterations == r.spd_iterations ||
        throw(ArgumentError("SPD iteration count changed on replay"))
    actual.ordering == r.spd_ordering || throw(ArgumentError("SPD ordering changed on replay"))
    actual.permutation_matched == r.spd_permutation_matched ||
        throw(ArgumentError("SPD permutation check changed on replay"))
    actual.factor_nnz == r.spd_factor_nnz ||
        throw(ArgumentError("SPD factor sparsity changed on replay"))
    actual.scaled == r.spd_scaled || throw(ArgumentError("SPD scaling changed on replay"))
    actual.scale_exponent_min == r.spd_scale_exponent_min ||
        throw(ArgumentError("SPD minimum scaling exponent changed on replay"))
    actual.scale_exponent_max == r.spd_scale_exponent_max ||
        throw(ArgumentError("SPD maximum scaling exponent changed on replay"))
    _require_same_float("SPD rho", actual.rho, r.spd_rho)
    _require_same_float("SPD input radius", actual.input_radius_bound,
                        r.spd_input_radius_bound)
    _require_same_float("SPD factor product norm", actual.factor_product_norm_bound,
                        r.spd_factor_product_norm_bound)
    _require_same_float("SPD gamma", actual.gamma_bound, r.spd_gamma_bound)
    _require_same_float("SPD roundoff bound", actual.roundoff_bound,
                        r.spd_roundoff_bound)
    _require_same_float("SPD underflow bound", actual.underflow_bound,
                        r.spd_underflow_bound)
    _require_same_float("SPD total error", actual.total_error_bound,
                        r.spd_total_error_bound)
    _require_same_float("SPD strict margin", actual.strict_margin_lower,
                        r.spd_strict_margin_lower)
    return nothing
end

"""
Replay a point proof without ARPACK.  The geometry, refined mesh, interval
pencil, and four-face constraint space are reconstructed.  The stored
`lambda_lower` is accepted only after a fresh local SPD proof; `hmax²` and the
Liu upper bound are then recomputed before any cover reference can be built.
"""
function replay_point(r::PointProofRecord)
    _validate_point(r)
    H, inverse_metric = Kappa3Certificate._reference_geometry(r.family, r.parameters_deg)
    _same_gram(H, r.gram) || throw(ArgumentError("reconstructed Gram matrix differs from point record"))
    mesh = r.mesh_strategy == :metric ?
        Kappa3Certificate.reference_mesh(r.refinement_level; metric = mid.(H)) :
        Kappa3Certificate.reference_mesh(r.refinement_level)
    mesh.NumElt == r.elements || throw(ArgumentError("element count changed on point replay"))
    mesh.NumF == r.dofs_before_constraints ||
        throw(ArgumentError("unconstrained DOF count changed on point replay"))
    A, M = Kappa3Certificate.assemble_cr_metric(mesh, H; inverse_metric)
    Z = Kappa3Certificate.constraint_basis(mesh; basis = r.basis)
    size(Z, 2) == r.dofs_after_constraints ||
        throw(ArgumentError("constrained DOF count changed on point replay"))

    AZ, MZ = A * Z, M * Z
    Braw = sparse(transpose(Z) * (AZ - interval(r.lambda_lower) * MZ))
    B = sparse((Braw + transpose(Braw)) / interval(2.0))
    r.spd_method == :residual ||
        throw(ArgumentError("legacy SPD methods cannot enter final replay"))
    spd = Kappa3Certificate.RigorousSPD.certify_spd_residual(
        B; scaling = r.spd_scaled, ordering = r.spd_ordering,
        initial_rho = r.spd_initial_rho,
        max_attempts = r.spd_max_attempts)
    _check_spd_replay(spd, r)

    h2 = Kappa3Certificate.hmax2_upper(mesh, H)
    u2_interval = inv(interval(r.lambda_lower)) +
                  Kappa3Certificate.CR_COEFFICIENT^2 * interval(0.0, h2)
    u2 = sup(u2_interval)
    C = sup(sqrt(u2_interval))
    _require_same_float("hmax2_upper", h2, r.hmax2_upper)
    _require_same_float("u2_upper", u2, r.u2_upper)
    _require_same_float("C_upper", C, r.C_upper)
    return ReplayedPoint(r.id, r.family, H, h2, u2, C, spd)
end

# ---------------------------------------------------------------------------
# Reference index and top-level artifact bundle
# ---------------------------------------------------------------------------

const _REFERENCE_VERTICES = ((0, 0, 0), (1, 0, 0),
                             (0, 1, 0), (0, 0, 1))

const _VERTEX_RELABEL_TRANSFORMS = let
    result = Set{NTuple{9, Int}}()
    for p0 in 1:4, p1 in 1:4, p2 in 1:4, p3 in 1:4
        p = (p0, p1, p2, p3)
        length(unique(p)) == 4 || continue
        T = ntuple(9) do k
            row = (k - 1) ÷ 3 + 1
            column = (k - 1) % 3 + 1
            _REFERENCE_VERTICES[p[row + 1]][column] -
                _REFERENCE_VERTICES[p[1]][column]
        end
        push!(result, T)
    end
    length(result) == 24 || error("internal vertex-relabel transform enumeration failed")
    result
end

function _normalize_transform(transform)
    values = collect(transform)
    length(values) == 9 || throw(ArgumentError("a reference transform needs nine entries"))
    T = ntuple(i -> Int(values[i]), 9)
    all(x -> x in (-1, 0, 1), T) ||
        throw(ArgumentError("reference transform entries must be -1, 0, or 1"))
    determinant = T[1] * (T[5] * T[9] - T[6] * T[8]) -
                  T[2] * (T[4] * T[9] - T[6] * T[7]) +
                  T[3] * (T[4] * T[8] - T[5] * T[7])
    abs(determinant) == 1 ||
        throw(ArgumentError("reference transform must be unimodular (determinant +/-1)"))
    T in _VERTEX_RELABEL_TRANSFORMS ||
        throw(ArgumentError("reference transform is not one of the 24 tetrahedron vertex relabellings"))
    return T
end

struct ReferenceProofRecord
    id::String
    source_point_id::String
    family::Symbol
    point_path::String
    point_sha256::String
    transform::NTuple{9, Int}
    gram::Matrix{IA}
    u2_upper::Float64
    function ReferenceProofRecord(id::AbstractString, source_point_id::AbstractString,
                                  family::Symbol,
                                  point_path::AbstractString,
                                  point_sha256::AbstractString,
                                  transform,
                                  gram::AbstractMatrix, u2_upper::Real)
        family in (:star, :path) || throw(ArgumentError("reference family must be star or path"))
        isempty(id) && throw(ArgumentError("reference id must not be empty"))
        isempty(source_point_id) && throw(ArgumentError("source point id must not be empty"))
        T = _normalize_transform(transform)
        u = Float64(u2_upper)
        isfinite(u) && u > 0 || throw(ArgumentError("reference u2 must be finite and positive"))
        new(String(id), String(source_point_id), family, _safe_relative_path(point_path),
            _require_sha256(point_sha256, "point_sha256"),
            T, _interval_matrix(gram), u)
    end
end

const _IDENTITY_TRANSFORM = (1, 0, 0, 0, 1, 0, 0, 0, 1)

_transform_text(transform::NTuple{9, Int}) = join(transform, ',')

function _parse_transform(text::AbstractString)
    fields = split(text, ','; keepempty = true)
    length(fields) == 9 || throw(ArgumentError("a reference transform needs nine entries"))
    return Tuple(_parse_int(field) for field in fields)
end

function _transform_gram(gram::AbstractMatrix, transform::NTuple{9, Int})
    T = [transform[3(i - 1) + j] for i in 1:3, j in 1:3]
    result = Matrix{IA}(undef, 3, 3)
    # The exact congruence T*G*T' is symmetric, so the enclosure of entry
    # (i,j) also encloses (j,i).  Mirroring it keeps the result exactly
    # symmetric; separate summation orders could differ by one ulp.
    for i in 1:3, j in i:3
        value = zero(gram[1, 1])
        for a in 1:3, b in 1:3
            sign = T[i, a] * T[j, b]
            iszero(sign) && continue
            value += sign == 1 ? gram[a, b] : -gram[a, b]
        end
        result[i, j] = value
        result[j, i] = value
    end
    return result
end

struct ReferenceIndex
    records::Vector{ReferenceProofRecord}
    function ReferenceIndex(records::AbstractVector{ReferenceProofRecord})
        refs = sort!(collect(records); by = r -> r.id)
        ids = getfield.(refs, :id)
        length(unique(ids)) == length(ids) || throw(ArgumentError("duplicate reference id"))
        new(refs)
    end
end

function reference_proof_record(id::AbstractString, point_path::AbstractString,
                                point_file::AbstractString;
                                transform = _IDENTITY_TRANSFORM)
    point = load_point_record(point_file)
    normalized = _normalize_transform(transform)
    # The constructor performs the exact domain and unimodularity checks.
    transformed = _transform_gram(point.gram, normalized)
    return ReferenceProofRecord(id, point.id, point.family, point_path,
                                sha256_file(point_file), normalized,
                                transformed, point.u2_upper)
end

function save_reference_index(path::AbstractString, index::ReferenceIndex)
    return _atomic_write(path) do io
        println(io, _REFERENCE_MAGIC)
        println(io, "references\t", length(index.records))
        for r in index.records
            println(io, join(("R", _escape(r.id), _escape(r.source_point_id),
                              _symbol_text(r.family), _escape(r.point_path),
                              r.point_sha256, _transform_text(r.transform),
                              _float_text(r.u2_upper), _gram_text(r.gram)), '\t'))
        end
        println(io, "END")
    end
end

function load_reference_index(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 3 && lines[1] == _REFERENCE_MAGIC ||
        throw(ArgumentError("not a Kappa3 reference index"))
    header = split(lines[2], '\t'; keepempty = true)
    length(header) == 2 && header[1] == "references" ||
        throw(ArgumentError("missing reference count"))
    count = _parse_int(header[2])
    count >= 0 || throw(ArgumentError("negative reference count"))
    length(lines) == count + 3 || throw(ArgumentError("reference count mismatch"))
    lines[end] == "END" || throw(ArgumentError("reference index has no END marker"))
    records = ReferenceProofRecord[]
    for line in lines[3:end-1]
        fields = split(line, '\t'; keepempty = true)
        length(fields) == 9 && fields[1] == "R" ||
            throw(ArgumentError("malformed reference entry"))
        push!(records, ReferenceProofRecord(
            _unescape(fields[2]), _unescape(fields[3]), _parse_symbol(fields[4]),
            _unescape(fields[5]), fields[6], _parse_transform(fields[7]),
            _parse_gram(fields[9]), _parse_float(fields[8])))
    end
    ids = getfield.(records, :id)
    issorted(ids) || throw(ArgumentError("reference index is not canonically sorted"))
    return ReferenceIndex(records)
end

"""
Resolve and replay every SHA-addressed point in a reference index.  The
returned dictionary contains freshly recomputed data, never trusted `u2`
values from the index.  `fingerprints_sha256` and `environment_sha256` bind
each point to the bundle provenance files.
"""
function replay_reference_index(index_path::AbstractString;
                                fingerprints_sha256::AbstractString,
                                environment_sha256::AbstractString)
    fhash = _require_sha256(fingerprints_sha256, "fingerprints_sha256")
    ehash = _require_sha256(environment_sha256, "environment_sha256")
    index = load_reference_index(index_path)
    base = dirname(abspath(index_path))
    result = Dict{String, ReplayedPoint}()
    point_cache = Dict{Tuple{String, String}, ReplayedPoint}()
    for ref in index.records
        point_path = _resolve_relative(base, ref.point_path)
        sha256_file(point_path) == ref.point_sha256 ||
            throw(ArgumentError("point SHA-256 mismatch for reference '$(ref.id)'"))
        point = load_point_record(point_path)
        point.id == ref.source_point_id ||
            throw(ArgumentError("reference source id differs from point id"))
        point.family == ref.family || throw(ArgumentError("reference family differs from point family"))
        point.fingerprints_sha256 == fhash ||
            throw(ArgumentError("point '$(point.id)' is bound to different source fingerprints"))
        point.environment_sha256 == ehash ||
            throw(ArgumentError("point '$(point.id)' is bound to a different environment snapshot"))
        cache_key = (point_path, ref.point_sha256)
        replayed = get!(point_cache, cache_key) do
            replay_point(point)
        end
        transformed_gram = _transform_gram(replayed.gram, ref.transform)
        _same_gram(transformed_gram, ref.gram) ||
            throw(ArgumentError("reference Gram matrix differs from replayed point"))
        _require_same_float("reference u2_upper", replayed.u2_upper, ref.u2_upper)
        result[ref.id] = ReplayedPoint(ref.id, ref.family, transformed_gram,
                                       replayed.hmax2_upper, replayed.u2_upper,
                                       replayed.C_upper, replayed.spd_certificate)
    end
    return result
end

struct ArtifactLink
    role::String
    path::String
    sha256::String
    function ArtifactLink(role::AbstractString, path::AbstractString, digest::AbstractString)
        name = String(role)
        name in _BUNDLE_ROLES || throw(ArgumentError("unknown proof-bundle role '$name'"))
        new(name, _safe_relative_path(path), _require_sha256(digest, "artifact SHA-256"))
    end
end

struct ProofBundle
    artifacts::Vector{ArtifactLink}
    function ProofBundle(artifacts::AbstractVector{ArtifactLink})
        links = sort!(collect(artifacts); by = a -> a.role)
        roles = getfield.(links, :role)
        Tuple(roles) == _BUNDLE_ROLES ||
            throw(ArgumentError("proof bundle must contain exactly $(join(_BUNDLE_ROLES, ", "))"))
        new(links)
    end
end

function save_proof_bundle(path::AbstractString, bundle::ProofBundle)
    return _atomic_write(path) do io
        println(io, _BUNDLE_MAGIC)
        println(io, "artifacts\t", length(bundle.artifacts))
        for a in bundle.artifacts
            println(io, join(("A", a.role, _escape(a.path), a.sha256), '\t'))
        end
        println(io, "END")
    end
end

function load_proof_bundle(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 3 && lines[1] == _BUNDLE_MAGIC ||
        throw(ArgumentError("not a Kappa3 proof bundle"))
    header = split(lines[2], '\t'; keepempty = true)
    length(header) == 2 && header[1] == "artifacts" ||
        throw(ArgumentError("missing proof-bundle artifact count"))
    count = _parse_int(header[2])
    count == length(_BUNDLE_ROLES) || throw(ArgumentError("wrong proof-bundle artifact count"))
    length(lines) == count + 3 || throw(ArgumentError("proof-bundle count mismatch"))
    lines[end] == "END" || throw(ArgumentError("proof bundle has no END marker"))
    links = ArtifactLink[]
    for line in lines[3:end-1]
        fields = split(line, '\t'; keepempty = true)
        length(fields) == 4 && fields[1] == "A" ||
            throw(ArgumentError("malformed proof-bundle artifact"))
        push!(links, ArtifactLink(fields[2], _unescape(fields[3]), fields[4]))
    end
    issorted(getfield.(links, :role)) || throw(ArgumentError("bundle artifacts are not sorted"))
    return ProofBundle(links)
end

"""Verify all bundle hashes and return a role-to-absolute-path dictionary."""
function verify_bundle_artifacts(bundle_path::AbstractString, bundle::ProofBundle)
    base = dirname(abspath(bundle_path))
    result = Dict{String, String}()
    for artifact in bundle.artifacts
        path = _resolve_relative(base, artifact.path)
        sha256_file(path) == artifact.sha256 ||
            throw(ArgumentError("SHA-256 mismatch for bundle artifact '$(artifact.role)'"))
        result[artifact.role] = path
    end
    return result
end

# ---------------------------------------------------------------------------
# Scratch layout
# ---------------------------------------------------------------------------

struct ScratchLayout
    root::String
    checkpoints::String
    logs::String
    points::String
    proof::String
    temporary::String
end

"""
Create the fixed production layout.  Production callers should retain the
default strict root.  Tests may pass `allow_noncanonical=true` with a fresh
temporary directory; this flag is intentionally explicit.
"""
function prepare_scratch_layout(root::AbstractString = CANONICAL_SCRATCH_ROOT;
                                allow_noncanonical::Bool = false)
    requested = abspath(normpath(root))
    canonical = abspath(CANONICAL_SCRATCH_ROOT)
    if !allow_noncanonical && requested != canonical
        throw(ArgumentError("production scratch root must be exactly '$canonical'"))
    end
    mkpath(requested)
    islink(requested) && throw(ArgumentError("scratch root must not be a symlink"))
    names = ("checkpoints", "logs", "points", "proof", "tmp")
    paths = map(name -> begin
        path = joinpath(requested, name)
        mkpath(path)
        islink(path) && throw(ArgumentError("scratch subdirectory must not be a symlink: '$path'"))
        path
    end, names)
    return ScratchLayout(requested, paths...)
end

end # module ProofRecords
