# src/core/mesh/mesh_load_from_folder.jl
#
# Port of VFEM3D/lib/mesh/mesh_load_from_folder.m. Reads
#   <folder>/nodes.dat     (NumNode × 3, whitespace-separated floats)
#   <folder>/elements.dat  (NumElt  × 4, 1-based ints)
# and builds the full Mesh3D — facets, edges, and connectivity tables.
#
# `elements.dat` rows are sorted ascending on load (MATLAB does the
# same), so all downstream routines can rely on `ElementList[e, :]`
# being sorted.

using DelimitedFiles: readdlm

"""
    mesh_load_from_folder(path::AbstractString) -> Mesh3D

Read a tetrahedral mesh from a folder containing `nodes.dat` and
`elements.dat` (both whitespace-separated). Returns a fully-populated
`Mesh3D` with derived `FacetList`, `EdgeList`, `Facet2Element`,
`Element2Facet`. The folder need not have a trailing slash.
"""
function mesh_load_from_folder(path::AbstractString)
    nodes = readdlm(joinpath(path, "nodes.dat"), Float64)
    size(nodes, 2) == 3 ||
        throw(ArgumentError("nodes.dat must have 3 columns (got $(size(nodes, 2)))"))

    elt_raw = readdlm(joinpath(path, "elements.dat"), Int)
    size(elt_raw, 2) == 4 ||
        throw(ArgumentError("elements.dat must have 4 columns (got $(size(elt_raw, 2)))"))

    # Sort each element row ascending (MATLAB convention).
    ElementList = Matrix{Int}(undef, size(elt_raw)...)
    @inbounds for r in axes(elt_raw, 1)
        v = (elt_raw[r, 1], elt_raw[r, 2], elt_raw[r, 3], elt_raw[r, 4])
        sv = sort([v...])
        ElementList[r, 1] = sv[1]
        ElementList[r, 2] = sv[2]
        ElementList[r, 3] = sv[3]
        ElementList[r, 4] = sv[4]
    end

    FacetList = get_facet_list(ElementList)
    EdgeList = get_edge_list(ElementList)
    Facet2Element, Element2Facet = facet_element_connectivity(ElementList, FacetList)

    return Mesh3D(nodes,
                  ElementList,
                  FacetList,
                  EdgeList,
                  Facet2Element,
                  Element2Facet,
                  size(nodes, 1),
                  size(ElementList, 1),
                  size(FacetList, 1),
                  size(EdgeList, 1))
end
