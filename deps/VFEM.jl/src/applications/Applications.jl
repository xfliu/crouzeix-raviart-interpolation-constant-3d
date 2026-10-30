# src/applications/Applications.jl
#
# Namespace for concrete problem drivers. Each application is a
# submodule that imports kernel functions from `..` (the parent
# `VFEM` module) and adds problem-specific parameters, drivers,
# meshes, and reference values.
#
# Today this file is a placeholder. As application-specific drivers
# are ported from `VFEM3D/cases/`, add them here as `include`s and
# document them in `src/applications/README.md`.
#
# Example skeleton (uncomment and populate when porting `Hydrogen`):
#
#   module Hydrogen
#       using ..VFEM: Mesh3D, mesh_load_from_folder, CoulombInfo,
#                     elem_V_coulomb_average, schrodinger_eig_cecr_3d
#       export hydrogen_lower_bound, exact_eigenvalues
#       include("hydrogen/exact.jl")
#       include("hydrogen/driver.jl")
#   end

module Applications
end
