# Follow the running proof campaign and certify a polynomial lower bound at
# every reference point it creates.  Read-only with respect to the campaign.
#
#   julia --project=LowerBound lower_daemon.jl CAMPAIGN_ROOT OUTPUT.tsv [degree]
#
# Exits after the campaign's proof-bundle.txt exists and every point has a
# lower bound, or after 30 days.

using Printf
using SHA
include(joinpath(@__DIR__, "..", "src", "BernsteinLB.jl"))
using .BernsteinLB
const B = BernsteinLB

const SOURCE_SHA = bytes2hex(sha256(read(joinpath(@__DIR__, "..", "src", "BernsteinLB.jl"))))
hexf(x::Float64) = string("0x", string(reinterpret(UInt64, x); base = 16, pad = 16))

function parse_point(path)
    id = fam = nothing
    params = nothing
    for line in eachline(path)
        f = split(line, '\t')
        length(f) == 2 || continue
        if f[1] == "id"
            id = String(f[2])
        elseif f[1] == "family"
            fam = Symbol(f[2])
        elseif f[1] == "parameters_deg"
            vals = map(split(f[2], ',')) do t
                h = split(t, '@')[2]
                reinterpret(Float64, parse(UInt64, h[3:end]; base = 16))
            end
            params = (vals[1], vals[2], vals[3])
        end
    end
    return id, fam, params
end

function done_ids(path)
    ids = Set{String}()
    isfile(path) || return ids
    for line in eachline(path)
        (isempty(line) || startswith(line, "#")) && continue
        push!(ids, String(split(line, '\t')[1]))
    end
    return ids
end

function main(root, output, M)
    pointdir = joinpath(root, "points")
    if !isfile(output)
        open(output, "w") do io
            println(io, "# id\tfamily\tp1\tp2\tp3\tdegree\tritz_C_float\tC_lower_certified\tC_lower_hex\tlambda_upper\tseconds\tstatus\tsource_sha256=", SOURCE_SHA)
        end
    end
    deadline = time() + 30 * 24 * 3600
    while time() < deadline
        done = done_ids(output)
        files = isdir(pointdir) ? filter(endswith(".point"), readdir(pointdir)) : String[]
        pending = 0
        for file in sort(files)
            id0 = replace(file, ".point" => "")
            id0 in done && continue
            pending += 1
            id, fam, p = try
                parse_point(joinpath(pointdir, file))
            catch
                (nothing, nothing, nothing)
            end
            (id === nothing || fam === nothing || p === nothing) && continue
            t0 = time()
            row = try
                r = B.point_lower(B.ShapeSpec(fam, p), M)
                @sprintf("%s\t%s\t%.17g\t%.17g\t%.17g\t%d\t%.15f\t%.15f\t%s\t%.17g\t%.2f\tok",
                         id, fam, p..., M, 1 / sqrt(r.ritz_lambda_float), r.C_lower,
                         hexf(r.C_lower), Float64(r.lambda_upper, RoundUp), time() - t0)
            catch err
                @sprintf("%s\t%s\t%.17g\t%.17g\t%.17g\t%d\tNaN\tNaN\t-\tNaN\t%.2f\tFAILED:%s",
                         id, fam, p..., M, time() - t0,
                         replace(sprint(showerror, err), r"\s+" => " ")[1:min(end, 200)])
            end
            open(io -> println(io, row), output, "a")
            println(row); flush(stdout)
        end
        if pending == 0 && isfile(joinpath(root, "proof-bundle.txt"))
            println("campaign complete and all points have lower bounds; exiting")
            return
        end
        pending == 0 && sleep(600)
    end
end

main(ARGS[1], ARGS[2], length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10)
