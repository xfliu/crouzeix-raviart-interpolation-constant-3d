using Pkg

Pkg.activate(@__DIR__)

# VFEM and Veigs are installed from GitHub at the exact commits used for the
# proof records.  Both are added in one transaction because neither package
# is registered.
Pkg.add([
    PackageSpec(url = "https://github.com/xfliu/veigs", subdir = "VEIGS.jl",
                rev = "ddd17a23758252ca59ef9190bde62729f390237d"),
    PackageSpec(url = "https://github.com/xfliu/vfem",
                rev = "ee15fc7872714151010cb197a2ef958ed2322241"),
])
Pkg.instantiate()
Pkg.precompile()

println("Kappa3Certificate environment ready")
Pkg.status()
