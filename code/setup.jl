using Pkg

Pkg.activate(@__DIR__)

vfem_path = get(ENV, "VFEM_PATH", "/home/xfliu/Workspace/Julia_LIB/Lib/VFEM.jl")
veigs_path = get(ENV, "VEIGS_PATH", "/home/xfliu/Workspace/Julia_LIB/Lib/Veigs.jl")

# Develop both unregistered local packages in one resolution transaction.  If
# they are added one at a time, Pkg tries to resolve the still-unregistered
# companion dependency before its path is known.
Pkg.develop([
    PackageSpec(path = veigs_path),
    PackageSpec(path = vfem_path),
])
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()

println("Kappa3Certificate environment ready")
Pkg.status()
