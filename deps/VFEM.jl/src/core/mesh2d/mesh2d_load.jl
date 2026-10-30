# src/core/mesh2d/mesh2d_load.jl
#
# Port of vfem2d/lib/mesh/read_mesh_from_folder.m. Reads
#   <folder>/vert.dat   — nv × 2 floats
#   <folder>/tri.dat    — nt × 3 ints
#   <folder>/edge.dat   — ne × 2 ints
#   <folder>/bd.dat     — nb × 2 ints (subset of edge rows)
# and builds the full Mesh2D including tri2edge and bd_edge_ids.
#
# Unlike VFEM3D's mesh layer, the 2D version does NOT re-compute
# edges from elements — it reads the precomputed edge table and
# matches boundary edges by sorted-pair encoding (same as
# find_is_edge_bd).

using DelimitedFiles: readdlm

"""
    mesh2d_load(path::AbstractString) -> Mesh2D

Read a 2D triangular mesh from a folder containing `vert.dat`,
`tri.dat`, `edge.dat`, `bd.dat` (whitespace-separated). Returns a
fully-populated `Mesh2D` with `tri2edge` and `bd_edge_ids`.
"""
function mesh2d_load(path::AbstractString)
    nodes    = readdlm(joinpath(path, "vert.dat"), Float64)
    elements = readdlm(joinpath(path, "tri.dat"),  Int)
    edges    = readdlm(joinpath(path, "edge.dat"), Int)
    bd_edges = readdlm(joinpath(path, "bd.dat"),   Int)

    size(nodes, 2)    == 2 || throw(ArgumentError("vert.dat must have 2 columns"))
    size(elements, 2) == 3 || throw(ArgumentError("tri.dat must have 3 columns"))
    size(edges, 2)    == 2 || throw(ArgumentError("edge.dat must have 2 columns"))
    size(bd_edges, 2) == 2 || throw(ArgumentError("bd.dat must have 2 columns"))

    is_bd = find_is_edge_bd(edges, bd_edges)
    bd_edge_ids = findall(==(1), is_bd)
    tri2edge = find_tri2edge(elements, edges)

    return Mesh2D(nodes,
                  elements,
                  edges,
                  bd_edges,
                  bd_edge_ids,
                  tri2edge,
                  size(nodes, 1),
                  size(elements, 1),
                  size(edges, 1),
                  size(bd_edges, 1))
end
