# Pilot: certified CR point bounds for star references, keeping u2_upper
# exactly (hex) so the cover pilot can rebuild ReferenceBound values.
#
#   julia --project=Code pilot_points.jl INPUT.tsv OUTPUT.tsv [SHARD NSHARDS]
#
# INPUT columns: id p1 p2 p3 level [family]   (family defaults to star)

using Printf
using Kappa3Certificate

hexf(x::Float64) = string("0x", string(reinterpret(UInt64, x); base = 16, pad = 16))

function main(input, output, shard, nshards)
    done = Set{String}()
    if isfile(output)
        for line in eachline(output)
            startswith(line, "#") || isempty(line) || push!(done, first(split(line, '\t')))
        end
    else
        open(output, "w") do io
            println(io, "# id\tp1\tp2\tp3\tlevel\tu2_upper_hex\tC_upper\tseconds\tstatus\tfamily")
        end
    end
    k = 0
    for line in eachline(input)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        k += 1
        (k - 1) % nshards == shard || continue
        f = split(s)
        id = String(f[1])
        id in done && continue
        p = (parse(Float64, f[2]), parse(Float64, f[3]), parse(Float64, f[4]))
        L = parse(Int, f[5])
        fam = length(f) >= 6 ? Symbol(f[6]) : :star
        t0 = time()
        row = try
            c = point_certificate(fam, p; level = L, basis = :haar,
                                  initial_backoff = 1e-4, mesh_strategy = :metric)
            @sprintf("%s\t%.17g\t%.17g\t%.17g\t%d\t%s\t%.17g\t%.2f\tok\t%s",
                     id, p..., L, hexf(c.u2_upper), c.C_upper, time() - t0, fam)
        catch err
            @sprintf("%s\t%.17g\t%.17g\t%.17g\t%d\t-\tNaN\t%.2f\tFAILED:%s\t%s",
                     id, p..., L, time() - t0,
                     replace(sprint(showerror, err), r"\s+" => " ")[1:min(end, 160)], fam)
        end
        open(io -> println(io, row), output, "a")
        println(row)
        flush(stdout)
    end
end

shard = length(ARGS) >= 4 ? parse(Int, ARGS[3]) : 0
nshards = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
main(ARGS[1], ARGS[2], shard, nshards)
