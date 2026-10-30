# src/core/mesh/facet_element_connectivity.jl
#
# Port of VFEM3D/lib/mesh/mesh_get_Facet2Element.m and
# mesh_get_Facet2Element_with_sign_fast.m.
#
# Convention (MATLAB-faithful):
#   The k-th local face of element e is the face *opposite* vertex k,
#   i.e. composed of vertices at positions (1:4) \ {k}. Element2Facet[e, k]
#   stores the global facet id of that face.
#   The signed variant returns +1 for the first element seen on a given
#   facet, -1 for the second.

"""
    facet_element_connectivity(ElementList::AbstractMatrix{<:Integer},
                               FacetList::AbstractMatrix{<:Integer})
        -> (Facet2Element::Matrix{Int}, Element2Facet::Matrix{Int})

Compute (a) the `NumF × 2` map from facets to their (≤ 2) parent
elements and (b) the `NumElt × 4` map from each element to its 4
global facet ids — one per local face, ordered to match MATLAB:
column k corresponds to the face opposite vertex k.

Boundary facets have a `0` in column 2 of `Facet2Element`.
"""
function facet_element_connectivity(ElementList::AbstractMatrix{<:Integer},
                                    FacetList::AbstractMatrix{<:Integer})
    size(ElementList, 2) == 4 ||
        throw(DimensionMismatch("ElementList must be NumElt × 4"))
    size(FacetList, 2) == 3 ||
        throw(DimensionMismatch("FacetList must be NumF × 3"))

    NumElt = size(ElementList, 1)
    NumF = size(FacetList, 1)
    Facet2Element = zeros(Int, NumF, 2)
    Element2Facet = zeros(Int, NumElt, 4)

    # Hash facet → row index. Keys are sorted ascending triples.
    facet_to_idx = Dict{NTuple{3, Int}, Int}()
    sizehint!(facet_to_idx, NumF)
    @inbounds for r in 1:NumF
        key = (Int(FacetList[r, 1]), Int(FacetList[r, 2]), Int(FacetList[r, 3]))
        # FacetList rows from get_facet_list are already sorted; assert here
        # so a malformed FacetList surfaces with a clear error.
        key[1] ≤ key[2] ≤ key[3] ||
            throw(ArgumentError("FacetList row $r is not sorted ascending"))
        facet_to_idx[key] = r
    end

    @inbounds for e in 1:NumElt
        v = (Int(ElementList[e, 1]), Int(ElementList[e, 2]),
             Int(ElementList[e, 3]), Int(ElementList[e, 4]))
        v[1] ≤ v[2] ≤ v[3] ≤ v[4] ||
            throw(ArgumentError("ElementList row $e is not sorted ascending"))
        # Local face k = vertices opposite vertex k. MATLAB column order:
        # k=1 → (v2, v3, v4), k=2 → (v1, v3, v4), k=3 → (v1, v2, v4),
        # k=4 → (v1, v2, v3). Already sorted ascending since v itself is.
        local_faces = ((v[2], v[3], v[4]),
                       (v[1], v[3], v[4]),
                       (v[1], v[2], v[4]),
                       (v[1], v[2], v[3]))
        for k in 1:4
            idx = facet_to_idx[local_faces[k]]
            Element2Facet[e, k] = idx
            if Facet2Element[idx, 1] == 0
                Facet2Element[idx, 1] = e
            else
                Facet2Element[idx, 2] = e
            end
        end
    end
    return Facet2Element, Element2Facet
end

"""
    facet_element_connectivity_with_sign(ElementList, FacetList)
        -> (Facet2Element, Element2Facet, ElementFacetDirectSign)

Same connectivity tables as `facet_element_connectivity`, plus an
`ElementFacetDirectSign[e, k] ∈ {+1, -1}` recording whether element
`e` is the first (+1) or second (-1) element to claim its local
face k. Used by RT/Hdiv assembly to fix interior-flux orientation.
"""
function facet_element_connectivity_with_sign(ElementList::AbstractMatrix{<:Integer},
                                              FacetList::AbstractMatrix{<:Integer})
    size(ElementList, 2) == 4 ||
        throw(DimensionMismatch("ElementList must be NumElt × 4"))
    size(FacetList, 2) == 3 ||
        throw(DimensionMismatch("FacetList must be NumF × 3"))

    NumElt = size(ElementList, 1)
    NumF = size(FacetList, 1)
    Facet2Element = zeros(Int, NumF, 2)
    Element2Facet = zeros(Int, NumElt, 4)
    ElementFacetDirectSign = zeros(Int, NumElt, 4)

    facet_to_idx = Dict{NTuple{3, Int}, Int}()
    sizehint!(facet_to_idx, NumF)
    @inbounds for r in 1:NumF
        facet_to_idx[(Int(FacetList[r, 1]),
                      Int(FacetList[r, 2]),
                      Int(FacetList[r, 3]))] = r
    end

    @inbounds for e in 1:NumElt
        v = (Int(ElementList[e, 1]), Int(ElementList[e, 2]),
             Int(ElementList[e, 3]), Int(ElementList[e, 4]))
        local_faces = ((v[2], v[3], v[4]),
                       (v[1], v[3], v[4]),
                       (v[1], v[2], v[4]),
                       (v[1], v[2], v[3]))
        for k in 1:4
            idx = facet_to_idx[local_faces[k]]
            Element2Facet[e, k] = idx
            if Facet2Element[idx, 1] == 0
                Facet2Element[idx, 1] = e
                ElementFacetDirectSign[e, k] = +1
            else
                Facet2Element[idx, 2] = e
                ElementFacetDirectSign[e, k] = -1
            end
        end
    end
    return Facet2Element, Element2Facet, ElementFacetDirectSign
end
