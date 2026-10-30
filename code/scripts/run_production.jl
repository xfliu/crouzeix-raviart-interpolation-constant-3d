using Kappa3Certificate

code_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(code_root, "src", "BoxCover.jl"))
include(joinpath(code_root, "src", "CoverSearch.jl"))
include(joinpath(code_root, "src", "ReferencePlan.jl"))
include(joinpath(code_root, "src", "ProofRecords.jl"))
include(joinpath(code_root, "src", "ProductionRunner.jl"))

try
    exit(Main.ProductionRunner.cli_main(ARGS; code_root))
catch exception
    println(stderr, "PRODUCTION REFUSED/FAILED: ", sprint(showerror, exception))
    Base.display_error(stderr, exception, catch_backtrace())
    exit(70)
end
