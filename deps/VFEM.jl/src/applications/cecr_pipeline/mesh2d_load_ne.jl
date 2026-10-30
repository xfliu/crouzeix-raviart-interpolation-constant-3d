# src/applications/cecr_pipeline/mesh2d_load_ne.jl
#
# Load a 2D mesh from nodes.dat + elements.dat (the format produced by
# the Python mesh generators in MeshGenerator2/). Computes edges and
# boundary edges from scratch (not read from file) since these files
# are not produced by the generator.

using DelimitedFiles: readdlm

"""
    mesh2d_load_ne(path::AbstractString) -> Mesh2D

Read a 2D triangular mesh from a folder containing:
* `nodes.dat`    — `nv × 2` float coordinates
* `elements.dat` — `nt × 3` integer vertex indices (1-based)

Edges, boundary edges, and the tri2edge map are computed automatically.
This loader handles the format produced by `MeshGenerator2/`.
"""
function mesh2d_load_ne(path::AbstractString)
    nodes    = readdlm(joinpath(path, "nodes.dat"),    Float64)
    elements = readdlm(joinpath(path, "elements.dat"), Int)
    size(nodes, 2)    == 2 || throw(ArgumentError("nodes.dat must have 2 columns"))
    size(elements, 2) == 3 || throw(ArgumentError("elements.dat must have 3 columns"))

    nv = size(nodes, 1)
    nt = size(elements, 1)

    # --- Build edge list from triangles ---
    # Map sorted edge pair → (global_edge_idx, count).
    edge_map  = Dict{NTuple{2, Int}, Int}()    # edge → global index
    edge_cnt  = Dict{NTuple{2, Int}, Int}()    # edge → triangle count
    edge_list = NTuple{2, Int}[]               # global_idx → sorted (a, b)
    sizehint!(edge_map, 3 * nt ÷ 2)
    sizehint!(edge_cnt, 3 * nt ÷ 2)

    @inbounds for k in 1:nt
        v1 = elements[k, 1]; v2 = elements[k, 2]; v3 = elements[k, 3]
        local_edges = ((v2, v3), (v1, v3), (v1, v2))
        for (a, b) in local_edges
            a_, b_ = a < b ? (a, b) : (b, a)
            key = (a_, b_)
            if !haskey(edge_map, key)
                push!(edge_list, key)
                edge_map[key] = length(edge_list)
                edge_cnt[key] = 1
            else
                edge_cnt[key] += 1
            end
        end
    end

    ne = length(edge_list)
    edges = Matrix{Int}(undef, ne, 2)
    for (i, (a, b)) in enumerate(edge_list)
        edges[i, 1] = a; edges[i, 2] = b
    end

    # --- Boundary edges: those with count == 1 ---
    bd_list = findall(k -> edge_cnt[edge_list[k]] == 1, 1:ne)
    nb = length(bd_list)
    bd_edges = Matrix{Int}(undef, nb, 2)
    for (i, k) in enumerate(bd_list)
        bd_edges[i, 1] = edges[k, 1]; bd_edges[i, 2] = edges[k, 2]
    end
    bd_edge_ids = bd_list

    # --- tri2edge ---
    tri2edge = Matrix{Int}(undef, nt, 3)
    @inbounds for k in 1:nt
        v1 = elements[k, 1]; v2 = elements[k, 2]; v3 = elements[k, 3]
        for (j, (a, b)) in enumerate(((v2, v3), (v1, v3), (v1, v2)))
            a_, b_ = a < b ? (a, b) : (b, a)
            tri2edge[k, j] = edge_map[(a_, b_)]
        end
    end

    return Mesh2D(nodes, elements, edges, bd_edges, bd_edge_ids, tri2edge,
                  nv, nt, ne, nb)
end
