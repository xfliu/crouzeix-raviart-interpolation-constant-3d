# src/applications/cecr_pipeline/CecrPipeline.jl
#
# CECR certification pipeline (m4–m7 + drivers).
# Included directly in the VFEM module (not a sub-module) to share
# the parent module's imports and exported types.

using ArnoldiMethod: partialschur

include("mesh2d_load_ne.jl")
include("pipeline_types.jl")
include("m4_constants.jl")
include("m6_lower_bound.jl")
include("m5_eps_h.jl")
include("m2_cecr_lb.jl")
include("m7_ceps_diag.jl")   # defines _assemble_Dh_2d used by m3
include("m3_p1_ub.jl")
include("run_pipeline.jl")
