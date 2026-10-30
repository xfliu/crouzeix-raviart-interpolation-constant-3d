# Certified polynomial lower bounds for a list of shapes.
#
#   julia --project=LowerBound lower_batch.jl INPUT.tsv OUTPUT.tsv
#
# INPUT columns (tab separated, '#' comments): id family p1 p2 p3 degrees
# where degrees is a comma list such as 10 or 4,6,8,10,12,14.  Parameters are
# decimal degrees parsed exactly as binary64.  Output is appended one line
# per (id, degree); lines already present are skipped, so the job resumes.

using Printf
using SHA
include(joinpath(@__DIR__, "..", "src", "BernsteinLB.jl"))
using .BernsteinLB
const B = BernsteinLB

const SOURCE_SHA = bytes2hex(sha256(read(joinpath(@__DIR__, "..", "src", "BernsteinLB.jl"))))

hexf(x::Float64) = string("0x", string(reinterpret(UInt64, x); base = 16, pad = 16))

function done_keys(path)
    keys = Set{Tuple{String, Int}}()
    isfile(path) || return keys
    for line in eachline(path)
        (isempty(line) || startswith(line, "#")) && continue
        f = split(line, '\t')
        length(f) >= 6 || continue
        push!(keys, (String(f[1]), parse(Int, f[6])))
    end
    return keys
end

function main(input, output)
    done = done_keys(output)
    if !isfile(output)
        open(output, "w") do io
            println(io, "# id\tfamily\tp1\tp2\tp3\tdegree\tritz_C_float\tC_lower_certified\tC_lower_hex\tlambda_upper\tseconds\tstatus\tsource_sha256=", SOURCE_SHA)
        end
    end
    for line in eachline(input)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s)
        id, fam = String(f[1]), Symbol(f[2])
        p = (parse(Float64, f[3]), parse(Float64, f[4]), parse(Float64, f[5]))
        degrees = parse.(Int, split(f[6], ','))
        for M in degrees
            (id, M) in done && continue
            t0 = time()
            row = try
                r = B.point_lower(B.ShapeSpec(fam, p), M)
                @sprintf("%s\t%s\t%.17g\t%.17g\t%.17g\t%d\t%.15f\t%.15f\t%s\t%.17g\t%.2f\tok",
                         id, fam, p..., M, 1 / sqrt(r.ritz_lambda_float),
                         r.C_lower, hexf(r.C_lower), Float64(r.lambda_upper, RoundUp),
                         time() - t0)
            catch err
                @sprintf("%s\t%s\t%.17g\t%.17g\t%.17g\t%d\tNaN\tNaN\t-\tNaN\t%.2f\tFAILED:%s",
                         id, fam, p..., M, time() - t0,
                         replace(sprint(showerror, err), r"\s+" => " ")[1:min(end, 200)])
            end
            open(output, "a") do io
                println(io, row)
            end
            flush(stdout)
            println(row)
        end
    end
end

main(ARGS[1], ARGS[2])
