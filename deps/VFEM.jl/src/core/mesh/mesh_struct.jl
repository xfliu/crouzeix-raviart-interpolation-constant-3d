# src/core/mesh/mesh_struct.jl
#
# Tetrahedral mesh data type. Mirrors the field set the MATLAB code
# threads through everywhere: sorted (1-based) `ElementList`, raw
# `NodeList`, derived `FacetList`, `EdgeList`, plus the connectivity
# tables. We keep the same name capitalisation as the MATLAB struct
# fields so call-site searches across both code bases stay grep-able.
#
# Only the geometric data is stored; signed Facet2Element data is
# computed on demand by `facet2element_with_sign` since not every
# pipeline needs it.

"""
    Mesh3D

Tetrahedral mesh struct. Fields mirror the MATLAB layout:

* `NodeList`   :: `Matrix{Float64}` — `NumNode × 3` of vertex coords.
* `ElementList`:: `Matrix{Int}`     — `NumElt × 4`, sorted ascending.
* `FacetList`  :: `Matrix{Int}`     — `NumF × 3`, each row sorted.
* `EdgeList`   :: `Matrix{Int}`     — `NumEdge × 2`, each row sorted (row 1 < row 2).
* `Facet2Element` :: `Matrix{Int}`  — `NumF × 2`. Boundary faces have a 0 in column 2.
* `Element2Facet` :: `Matrix{Int}`  — `NumElt × 4`, with column k giving the
  global facet id of the local face *opposite* vertex k (matching the
  MATLAB convention).
* `NumNode`, `NumElt`, `NumF`, `NumEdge` :: `Int`.

Indices are 1-based at every layer (matching MATLAB).
"""
struct Mesh3D
    NodeList::Matrix{Float64}
    ElementList::Matrix{Int}
    FacetList::Matrix{Int}
    EdgeList::Matrix{Int}
    Facet2Element::Matrix{Int}
    Element2Facet::Matrix{Int}
    NumNode::Int
    NumElt::Int
    NumF::Int
    NumEdge::Int
end

Base.show(io::IO, m::Mesh3D) =
    print(io, "Mesh3D(NumNode=", m.NumNode, ", NumElt=", m.NumElt,
              ", NumF=", m.NumF, ", NumEdge=", m.NumEdge, ")")
