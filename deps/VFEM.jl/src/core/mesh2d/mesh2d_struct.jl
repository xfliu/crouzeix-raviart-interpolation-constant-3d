# src/core/mesh2d/mesh2d_struct.jl
#
# 2D triangular mesh data type. Mirrors the field set in
# vfem2d/lib/mesh/read_mesh_from_folder.m so call sites stay
# grep-compatible.
#
# Field naming follows the vfem2d convention (lowercase) rather
# than the VFEM3D `NodeList`/`ElementList` style — same reason:
# match the MATLAB code already on disk.

"""
    Mesh2D

2D triangular mesh struct.

* `nodes`        :: `Matrix{Float64}` — `nv × 2` of vertex (x, y) coords.
* `elements`     :: `Matrix{Int}`     — `nt × 3`, 1-based; row order is
  the load-from-file order.
* `edges`        :: `Matrix{Int}`     — `ne × 2`, each row sorted ascending.
* `bd_edges`     :: `Matrix{Int}`     — `nb × 2`, boundary-edge endpoints.
* `bd_edge_ids`  :: `Vector{Int}`     — indices into `edges` of boundary edges.
* `tri2edge`     :: `Matrix{Int}`     — `nt × 3` triangle→edge map.
  Column k holds the global edge index of the local edge *opposite* vertex k.
* `nv`, `nt`, `ne`, `nb` :: `Int`.

Indices are 1-based.
"""
struct Mesh2D
    nodes::Matrix{Float64}
    elements::Matrix{Int}
    edges::Matrix{Int}
    bd_edges::Matrix{Int}
    bd_edge_ids::Vector{Int}
    tri2edge::Matrix{Int}
    nv::Int
    nt::Int
    ne::Int
    nb::Int
end

Base.show(io::IO, m::Mesh2D) =
    print(io, "Mesh2D(nv=", m.nv, ", nt=", m.nt,
              ", ne=", m.ne, ", nb=", m.nb, ")")
