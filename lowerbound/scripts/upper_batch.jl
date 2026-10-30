# Certified Crouzeix--Raviart upper bounds for a list of shapes, using the
# campaign's own Kappa3Certificate.point_certificate (run with
# --project=<Code>; this file lives outside Code/ so the campaign source
# fingerprints are unaffected).
#
#   julia --project=Code upper_batch.jl INPUT.tsv OUTPUT.tsv
#
# INPUT columns: id family p1 p2 p3 levels   (levels: comma list, e.g. 4,5)

using Printf
using Kappa3Certificate

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
            println(io, "# id\tfamily\tp1\tp2\tp3\tlevel\telements\tlambda_approx\tlambda_lower\thmax2_upper\tC_upper\tC_upper_hex\tseconds\tstatus")
        end
    end
    for line in eachline(input)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s)
        id, fam = String(f[1]), Symbol(f[2])
        p = (parse(Float64, f[3]), parse(Float64, f[4]), parse(Float64, f[5]))
        levels = parse.(Int, split(f[6], ','))
        for L in levels
            (id, L) in done && continue
            t0 = time()
            row = try
                c = point_certificate(fam, p; level = L, basis = :haar,
                                      initial_backoff = 1e-4,
                                      mesh_strategy = :metric)
                @sprintf("%s\t%s\t%.17g\t%.17g\t%.17g\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%s\t%.2f\tok",
                         id, fam, p..., L, c.elements, c.lambda_approx,
                         c.lambda_lower, c.hmax2_upper, c.C_upper,
                         hexf(c.C_upper), time() - t0)
            catch err
                @sprintf("%s\t%s\t%.17g\t%.17g\t%.17g\t%d\t0\tNaN\tNaN\tNaN\tNaN\t-\t%.2f\tFAILED:%s",
                         id, fam, p..., L, time() - t0,
                         replace(sprint(showerror, err), r"\s+" => " ")[1:min(end, 200)])
            end
            open(output, "a") do io
                println(io, row)
            end
            println(row)
            flush(stdout)
        end
    end
end

main(ARGS[1], ARGS[2])
