module BoxCover

using IntervalArithmetic: Interval, interval, inf, sup, isguaranteed

export TARGET_KAPPA, TARGET_KAPPA_SQUARED, STAR_TARGET, PATH_TARGET,
       family_target,
       Dyadic, DyadicInterval, DyadicBox, dyadic_box, midpoint,
       star_root_box, path_root_box, split_box, split_boxes,
       CoefficientBox, coefficient_corners,
       star_coefficient_enclosure, path_coefficient_enclosure,
       star_box_excluded, path_box_excluded,
       star_gram, path_vertex_gram,
       PD3Result, strict_pd3_result, strict_pd3,
       allowed_comparison_factor, CornerTest, LeafTestResult,
       certify_coefficient_leaf, certify_leaf,
       AbstractCoverNode, CoverLeaf, CoverSplit,
       TreeAudit, audit_tree, validate_tree,
       ReferenceBound, ReplayResult, replay_cover

const IA = Interval{Float64}

"""The exact requested universal bound, not a parsed decimal."""
const TARGET_KAPPA = 799 // 5000
const TARGET_KAPPA_SQUARED = 638401 // 25000000

"""Exact per-family cover targets; the star family is the critical one."""
const STAR_TARGET = TARGET_KAPPA
const PATH_TARGET = 31 // 200

function family_target(family::Symbol)
    family == :star && return STAR_TARGET
    family == :path && return PATH_TARGET
    throw(ArgumentError("family must be :star or :path"))
end

# ---------------------------------------------------------------------------
# Exact dyadic geometry
# ---------------------------------------------------------------------------

"""
    Dyadic(numerator, exponent)

The exact number `numerator / 2^exponent`.  Values are normalized, so two
equal dyadics have identical representations.  Cover topology never depends
on floating-point endpoint comparisons.
"""
struct Dyadic
    numerator::BigInt
    exponent::Int

    function Dyadic(numerator::Integer, exponent::Integer = 0)
        exponent >= 0 || throw(DomainError(exponent, "a dyadic exponent must be nonnegative"))
        n = BigInt(numerator)
        e = Int(exponent)
        if iszero(n)
            return new(BigInt(0), 0)
        end
        while e > 0 && iseven(n)
            n >>= 1
            e -= 1
        end
        return new(n, e)
    end
end

function Dyadic(x::Rational{<:Integer})
    d = BigInt(denominator(x))
    e = 0
    while d > 1 && iseven(d)
        d >>= 1
        e += 1
    end
    d == 1 || throw(ArgumentError("the rational denominator is not a power of two"))
    return Dyadic(BigInt(numerator(x)), e)
end

_dyadic(x::Dyadic) = x
_dyadic(x::Integer) = Dyadic(x)
_dyadic(x::Rational{<:Integer}) = Dyadic(x)

Base.:(==)(a::Dyadic, b::Dyadic) =
    a.exponent == b.exponent && a.numerator == b.numerator
Base.hash(x::Dyadic, h::UInt) = hash((x.numerator, x.exponent), h)

function Base.isless(a::Dyadic, b::Dyadic)
    e = max(a.exponent, b.exponent)
    an = a.numerator << (e - a.exponent)
    bn = b.numerator << (e - b.exponent)
    return an < bn
end

Base.:<=(a::Dyadic, b::Dyadic) = isless(a, b) || a == b
Base.:+(a::Dyadic, b::Dyadic) = begin
    e = max(a.exponent, b.exponent)
    Dyadic((a.numerator << (e - a.exponent)) +
            (b.numerator << (e - b.exponent)), e)
end
Base.:-(a::Dyadic, b::Dyadic) = begin
    e = max(a.exponent, b.exponent)
    Dyadic((a.numerator << (e - a.exponent)) -
            (b.numerator << (e - b.exponent)), e)
end

"""Return the exact dyadic midpoint."""
function midpoint(a::Dyadic, b::Dyadic)
    a <= b || throw(ArgumentError("midpoint endpoints are reversed"))
    e = max(a.exponent, b.exponent)
    an = a.numerator << (e - a.exponent)
    bn = b.numerator << (e - b.exponent)
    return Dyadic(an + bn, e + 1)
end

function Base.show(io::IO, x::Dyadic)
    if x.exponent == 0
        print(io, x.numerator)
    else
        print(io, x.numerator, "/2^", x.exponent)
    end
end

function _rational(x::Dyadic)
    return x.numerator // (BigInt(1) << x.exponent)
end

# Constructing the Float64 interval from the exact rational asks
# IntervalArithmetic to direct-round both endpoints.  No Float64 conversion
# of a cover endpoint occurs before this enclosure.
_interval(x::Dyadic) = interval(Float64, _rational(x))

struct DyadicInterval
    lo::Dyadic
    hi::Dyadic
    function DyadicInterval(lo::Dyadic, hi::Dyadic)
        lo <= hi || throw(ArgumentError("dyadic interval endpoints are reversed"))
        new(lo, hi)
    end
end

DyadicInterval(lo, hi) = DyadicInterval(_dyadic(lo), _dyadic(hi))

struct DyadicBox
    bounds::NTuple{3, DyadicInterval}
end

function dyadic_box(lo::Tuple, hi::Tuple)
    length(lo) == 3 == length(hi) || throw(ArgumentError("a cover box is three-dimensional"))
    return DyadicBox(ntuple(i -> DyadicInterval(lo[i], hi[i]), 3))
end

Base.:(==)(a::DyadicInterval, b::DyadicInterval) = a.lo == b.lo && a.hi == b.hi
Base.:(==)(a::DyadicBox, b::DyadicBox) = a.bounds == b.bounds

star_root_box() = dyadic_box((0, 0, 0), (60, 60, 60))
path_root_box() = dyadic_box((0, 0, 0), (60, 60, 90))

"""Bisect one coordinate of a dyadic box, returning its lower and upper child."""
function split_box(box::DyadicBox, axis::Integer)
    1 <= axis <= 3 || throw(ArgumentError("split axis must be 1, 2, or 3"))
    b = box.bounds[Int(axis)]
    m = midpoint(b.lo, b.hi)
    m == b.lo && throw(ArgumentError("cannot split a zero-width dyadic interval"))
    lower = DyadicBox(ntuple(i -> i == axis ? DyadicInterval(b.lo, m) : box.bounds[i], 3))
    upper = DyadicBox(ntuple(i -> i == axis ? DyadicInterval(m, b.hi) : box.bounds[i], 3))
    return lower, upper
end

"""
    split_boxes(box, axes)

Simultaneously bisect the listed axes.  Children use binary-mask order: bit
`j-1` selects the upper half of `axes[j]`.  This represents both binary trees
and the star calculation's eight-child octree without approximate endpoints.
"""
function split_boxes(box::DyadicBox, axes)
    ax = Tuple(Int(a) for a in axes)
    isempty(ax) && throw(ArgumentError("at least one split axis is required"))
    all(a -> 1 <= a <= 3, ax) || throw(ArgumentError("split axes must be in 1:3"))
    length(unique(ax)) == length(ax) || throw(ArgumentError("split axes must be distinct"))
    issorted(ax) || throw(ArgumentError("split axes must be in increasing order"))
    mids = Dict(a => midpoint(box.bounds[a].lo, box.bounds[a].hi) for a in ax)
    any(a -> mids[a] == box.bounds[a].lo, ax) &&
        throw(ArgumentError("cannot split a zero-width dyadic interval"))
    return [DyadicBox(ntuple(i -> begin
                j = findfirst(==(i), ax)
                if isnothing(j)
                    box.bounds[i]
                elseif ((mask >> (j - 1)) & 1) == 0
                    DyadicInterval(box.bounds[i].lo, mids[i])
                else
                    DyadicInterval(mids[i], box.bounds[i].hi)
                end
            end, 3)) for mask in 0:(2^length(ax) - 1)]
end

# ---------------------------------------------------------------------------
# Rigorous angular-to-coefficient enclosures
# ---------------------------------------------------------------------------

struct CoefficientBox
    lo::NTuple{3, Float64}
    hi::NTuple{3, Float64}
    function CoefficientBox(lo::NTuple{3, Float64}, hi::NTuple{3, Float64})
        all(isfinite, lo) && all(isfinite, hi) ||
            throw(ArgumentError("coefficient endpoints must be finite"))
        all(lo[i] <= hi[i] for i in 1:3) ||
            throw(ArgumentError("coefficient interval endpoints are reversed"))
        new(lo, hi)
    end
end

CoefficientBox(lo::NTuple{3, <:Real}, hi::NTuple{3, <:Real}) =
    CoefficientBox(ntuple(i -> inf(_ia(lo[i])), 3),
                   ntuple(i -> sup(_ia(hi[i])), 3))

function _trig_degree(x::Dyadic, f::F) where {F}
    return setprecision(BigFloat, 256) do
        xb = interval(BigFloat, _rational(x))
        yb = f(xb / interval(BigFloat, BigFloat(180)))
        interval(Float64, inf(yb), sup(yb))
    end
end

_cosd(x::Dyadic) = _trig_degree(x, cospi)
_sind(x::Dyadic) = _trig_degree(x, sinpi)

function _check_box_range(box::DyadicBox, maxima::NTuple{3, Int})
    z = Dyadic(0)
    for i in 1:3
        z <= box.bounds[i].lo || throw(ArgumentError("angle boxes must be nonnegative"))
        box.bounds[i].hi <= Dyadic(maxima[i]) ||
            throw(ArgumentError("angle box coordinate $i exceeds $(maxima[i]) degrees"))
    end
    return nothing
end

"""
Exact enclosing Gram-coordinate box for a star angular box.  The returned
coordinates are `(cos(alpha), cos(beta), cos(gamma))`.  Its binary Float64
endpoints are outward enclosures of the exact trigonometric endpoints.
"""
function star_coefficient_enclosure(box::DyadicBox)
    _check_box_range(box, (60, 60, 60))
    lo = ntuple(i -> inf(_cosd(box.bounds[i].hi)), 3)
    hi = ntuple(i -> sup(_cosd(box.bounds[i].lo)), 3)
    return CoefficientBox(lo, hi)
end

"""
Enclose the feasible image of a path angular box in `(x,y,z)`, where
`x=cos(theta1)`, `y=cos(theta2)`, and
`z=xy-sin(theta1)sin(theta2)cos(phi)`.  The diameter condition
`z <= x+y-1` is included in the upper endpoint.  The monotone formula is
valid on the full path root `[0,60]^2 x [0,90]`.  `nothing` is returned only
after the outward enclosures themselves prove that the feasible image is
empty.
"""
function path_coefficient_enclosure(box::DyadicBox)
    _check_box_range(box, (60, 60, 90))
    t1, t2, ph = box.bounds

    cb1, ca1 = _cosd(t1.hi), _cosd(t1.lo)
    cb2, ca2 = _cosd(t2.hi), _cosd(t2.lo)
    sb1, sa1 = _sind(t1.hi), _sind(t1.lo)
    sb2, sa2 = _sind(t2.hi), _sind(t2.lo)
    caph, cbph = _cosd(ph.lo), _cosd(ph.hi)

    xlo, xhi = inf(cb1), sup(ca1)
    ylo, yhi = inf(cb2), sup(ca2)

    # On this root z decreases with theta1 and theta2 and increases with phi.
    zlo = inf(cb1 * cb2 - sb1 * sb2 * caph)
    zhi_angle = sup(ca1 * ca2 - sa1 * sa2 * cbph)
    zhi_diameter = sup(ca1 + ca2 - interval(1.0))
    zhi = min(zhi_angle, zhi_diameter)
    zlo <= zhi || return nothing
    return CoefficientBox((xlo, ylo, zlo), (xhi, yhi, zhi))
end

"""Return the eight affine corners, in binary-mask order."""
function coefficient_corners(box::CoefficientBox)
    return [ntuple(i -> ((mask >> (i - 1)) & 1) == 0 ? box.lo[i] : box.hi[i], 3)
            for mask in 0:7]
end

"""
Prove a star angular box infeasible using an exact triangle inequality.
The strict feasible inequalities imply emptiness if one lower endpoint is at
least the sum of the other two upper endpoints.
"""
function star_box_excluded(box::DyadicBox)
    _check_box_range(box, (60, 60, 60))
    for i in 1:3
        j, k = filter(!=(i), 1:3)
        box.bounds[i].lo >= box.bounds[j].hi + box.bounds[k].hi && return true
    end
    return false
end

"""
Prove a path angular box infeasible.  Feasibility requires
`cos(phi) >= tan(theta1/2)tan(theta2/2)`.  Monotonicity reduces the test to
the three lower angular endpoints.  A strict, separated interval inequality
is required; equality is deliberately retained.
"""
function path_box_excluded(box::DyadicBox)
    _check_box_range(box, (60, 60, 90))
    a1, a2, a3 = (b.lo for b in box.bounds)
    lhs = _cosd(a3)
    t1 = _sind(a1) / (interval(1.0) + _cosd(a1)) # tan(a1/2)
    t2 = _sind(a2) / (interval(1.0) + _cosd(a2)) # tan(a2/2)
    sup(lhs) < inf(t1 * t2) && return true

    # This is an independent, equally rigorous emptiness proof and can be
    # stronger by a few ulps near a coefficient-box separation.
    return isnothing(path_coefficient_enclosure(box))
end

# ---------------------------------------------------------------------------
# Matrix corner tests
# ---------------------------------------------------------------------------

function _ia(x::Interval)
    isguaranteed(x) ||
        throw(ArgumentError("non-guaranteed interval input cannot enter a rigorous certificate"))
    return interval(Float64, inf(x), sup(x))
end
_ia(x::Rational) = interval(Float64, x)
_ia(x::Real) = interval(Float64, x)

function star_gram(c::NTuple{3, <:Real})
    a, b, d = _ia(c[1]), _ia(c[2]), _ia(c[3])
    o = interval(1.0)
    G = Matrix{IA}(undef, 3, 3)
    G[1, 1] = o; G[2, 2] = o; G[3, 3] = o
    G[1, 2] = G[2, 1] = a
    G[1, 3] = G[3, 1] = b
    G[2, 3] = G[3, 2] = d
    return G
end

"""
The vertex-edge Gram matrix used by the CR point solver for a path.  This is
`H(x,y,z)` from the manuscript, not the directed-chain matrix `G_p`.
"""
function path_vertex_gram(c::NTuple{3, <:Real})
    x, y, z = _ia(c[1]), _ia(c[2]), _ia(c[3])
    o = interval(1.0)
    H = Matrix{IA}(undef, 3, 3)
    H[1, 1] = o
    H[1, 2] = H[2, 1] = x
    H[1, 3] = H[3, 1] = x - z
    H[2, 2] = o
    H[2, 3] = H[3, 2] = o - y
    H[3, 3] = interval(2.0) - interval(2.0) * y
    return H
end

function _interval_matrix(A::AbstractMatrix)
    size(A) == (3, 3) || throw(DimensionMismatch("a Gram matrix must be 3 by 3"))
    B = Matrix{IA}(undef, 3, 3)
    for j in 1:3, i in 1:3
        B[i, j] = _ia(A[i, j])
    end
    return B
end

struct PD3Result
    passed::Bool
    leading_minor_lower::NTuple{3, Float64}
end

"""
Certify strict positive definiteness of every symmetric point matrix enclosed
by `A`.  Sylvester's criterion is evaluated with interval arithmetic.  This
does not compute eigenvalues, invert a possibly singular corner, or accept a
non-strict/ambiguous zero margin.
"""
function strict_pd3_result(A::AbstractMatrix)
    B = _interval_matrix(A)
    for i in 1:3, j in (i + 1):3
        (isequal(inf(B[i, j]), inf(B[j, i])) &&
         isequal(sup(B[i, j]), sup(B[j, i]))) ||
            throw(ArgumentError("strict_pd3_result requires a symmetric interval matrix"))
    end
    a, b, c = B[1, 1], B[1, 2], B[1, 3]
    d, e, f = B[2, 2], B[2, 3], B[3, 3]
    p1 = a
    p2 = a * d - b * b
    p3 = a * d * f + interval(2.0) * b * c * e -
         a * e * e - d * c * c - f * b * b
    lower = (inf(p1), inf(p2), inf(p3))
    return PD3Result(all(>(0.0), lower), lower)
end

strict_pd3(A::AbstractMatrix) = strict_pd3_result(A).passed

"""
Return a downward-rounded admissible comparison factor.  If `u2_upper` is a
certified upper endpoint for `C(K0)^2`, then the returned `q` satisfies
`q*u2_upper <= target^2`.  Working with squares avoids an unnecessary square
root in every leaf test.
"""
function allowed_comparison_factor(u2_upper::Real;
                                   target::Rational = TARGET_KAPPA)
    u = sup(_ia(u2_upper))
    isfinite(u) && u > 0 || throw(DomainError(u2_upper, "u2_upper must be finite and positive"))
    t = interval(Float64, target)
    return inf((t * t) / interval(u))
end

struct CornerTest
    coefficients::NTuple{3, Float64}
    pd::PD3Result
end

struct LeafTestResult
    passed::Bool
    vacuous::Bool
    q_allow::Float64
    coefficient_box::Union{Nothing, CoefficientBox}
    corners::Vector{CornerTest}
end

function certify_coefficient_leaf(family::Symbol, box::CoefficientBox,
                                  reference_gram::AbstractMatrix,
                                  u2_upper::Real;
                                  target::Rational = TARGET_KAPPA)
    family in (:star, :path) || throw(ArgumentError("family must be :star or :path"))
    G0 = _interval_matrix(reference_gram)
    strict_pd3(G0) ||
        throw(ArgumentError("the reference Gram matrix is not interval-certified positive definite"))
    q = allowed_comparison_factor(u2_upper; target)
    q > 0 || throw(DomainError(q, "the admissible comparison factor is not positive"))
    qI = interval(q)
    tests = CornerTest[]
    for c in coefficient_corners(box)
        G = family == :star ? star_gram(c) : path_vertex_gram(c)
        pd = strict_pd3_result(qI .* G0 .- G)
        push!(tests, CornerTest(c, pd))
    end
    return LeafTestResult(all(t -> t.pd.passed, tests), false, q, box, tests)
end

function certify_leaf(family::Symbol, angular_box::DyadicBox,
                      reference_gram::AbstractMatrix, u2_upper::Real;
                      target::Rational = TARGET_KAPPA)
    coeff = if family == :star
        star_coefficient_enclosure(angular_box)
    elseif family == :path
        path_coefficient_enclosure(angular_box)
    else
        throw(ArgumentError("family must be :star or :path"))
    end
    q = allowed_comparison_factor(u2_upper; target)
    isnothing(coeff) && return LeafTestResult(true, true, q, nothing, CornerTest[])
    return certify_coefficient_leaf(family, coeff, reference_gram, u2_upper; target)
end

# ---------------------------------------------------------------------------
# Exact cover-tree audit and independent replay
# ---------------------------------------------------------------------------

abstract type AbstractCoverNode end

"""
A terminal cover record.  `status` is `:retained` or `:excluded`; retained
leaves name a reference certificate, while excluded leaves are rechecked from
geometry during audit/replay.
"""
struct CoverLeaf <: AbstractCoverNode
    box::DyadicBox
    status::Symbol
    reference_id::Union{Nothing, String}
    function CoverLeaf(box::DyadicBox, status::Symbol,
                       reference_id::Union{Nothing, AbstractString} = nothing)
        status in (:retained, :excluded) ||
            throw(ArgumentError("leaf status must be :retained or :excluded"))
        id = isnothing(reference_id) ? nothing : String(reference_id)
        status == :retained && isnothing(id) &&
            throw(ArgumentError("a retained leaf must name a reference"))
        status == :excluded && !isnothing(id) &&
            throw(ArgumentError("an excluded leaf must not name a reference"))
        new(box, status, id)
    end
end

"""
An exact dyadic split.  `axes=(1,2,3)` has eight children; a singleton axis
has two.  Children must be stored in the binary-mask order of `split_boxes`.
"""
struct CoverSplit <: AbstractCoverNode
    box::DyadicBox
    axes::Tuple{Vararg{Int}}
    children::Vector{AbstractCoverNode}
    function CoverSplit(box::DyadicBox, axes,
                        children::AbstractVector{<:AbstractCoverNode})
        ax = Tuple(Int(a) for a in axes)
        new(box, ax, AbstractCoverNode[children...])
    end
end

struct TreeAudit
    passed::Bool
    nodes::Int
    splits::Int
    retained::Int
    excluded::Int
    errors::Vector{String}
end

function _audit_node!(errors, counters, node::AbstractCoverNode,
                      expected::DyadicBox, family::Union{Nothing, Symbol})
    counters[1] += 1
    if node isa CoverLeaf
        leaf = node::CoverLeaf
        leaf.box == expected || push!(errors, "leaf box does not equal its exact partition cell")
        if leaf.status == :retained
            counters[3] += 1
        else
            counters[4] += 1
            if family == :star
                star_box_excluded(leaf.box) ||
                    push!(errors, "star leaf marked excluded without an exact infeasibility proof")
            elseif family == :path
                path_box_excluded(leaf.box) ||
                    push!(errors, "path leaf marked excluded without an interval infeasibility proof")
            elseif !isnothing(family)
                push!(errors, "unknown family $family")
            end
        end
        return
    end

    split = node::CoverSplit
    counters[2] += 1
    split.box == expected || push!(errors, "split box does not equal its exact partition cell")
    local child_boxes
    try
        child_boxes = split_boxes(expected, split.axes)
    catch err
        push!(errors, "invalid split: $(sprint(showerror, err))")
        return
    end
    if length(split.children) != length(child_boxes)
        push!(errors, "split has $(length(split.children)) children; expected $(length(child_boxes))")
        return
    end
    for (child, child_box) in zip(split.children, child_boxes)
        _audit_node!(errors, counters, child, child_box, family)
    end
end

"""Audit exact coverage, split topology, and (optionally) excluded leaves."""
function audit_tree(root::AbstractCoverNode, expected_root::DyadicBox;
                    family::Union{Nothing, Symbol} = nothing)
    family in (nothing, :star, :path) || throw(ArgumentError("family must be nothing, :star, or :path"))
    errors = String[]
    counters = zeros(Int, 4) # nodes, splits, retained, excluded
    _audit_node!(errors, counters, root, expected_root, family)
    return TreeAudit(isempty(errors), counters[1], counters[2],
                     counters[3], counters[4], errors)
end

"""Throw on a failed tree audit and otherwise return its statistics."""
function validate_tree(root::AbstractCoverNode, expected_root::DyadicBox;
                       family::Union{Nothing, Symbol} = nothing)
    result = audit_tree(root, expected_root; family)
    result.passed || throw(ArgumentError(join(result.errors, "\n")))
    return result
end

struct ReferenceBound
    id::String
    family::Symbol
    gram::Matrix{IA}
    u2_upper::Float64
    function ReferenceBound(id::AbstractString, family::Symbol,
                            gram::AbstractMatrix, u2_upper::Real)
        family in (:star, :path) || throw(ArgumentError("family must be :star or :path"))
        u = sup(_ia(u2_upper))
        isfinite(u) && u > 0 || throw(DomainError(u2_upper, "u2_upper must be finite and positive"))
        G = _interval_matrix(gram)
        strict_pd3(G) ||
            throw(ArgumentError("the reference Gram matrix is not interval-certified positive definite"))
        new(String(id), family, G, u)
    end
end

struct ReplayResult
    passed::Bool
    audit::TreeAudit
    checked_leaves::Int
    checked_corners::Int
    vacuous_leaves::Int
    failures::Vector{String}
end

function _collect_leaves!(leaves::Vector{CoverLeaf}, node::AbstractCoverNode)
    if node isa CoverLeaf
        push!(leaves, node)
    else
        for child in (node::CoverSplit).children
            _collect_leaves!(leaves, child)
        end
    end
    return leaves
end

"""
Independently replay a complete cover.  First audit the exact dyadic tree and
all exclusions; then recompute every retained leaf's eight interval matrix
tests from its named reference.  Stored floating-point comparison factors are
neither accepted nor needed.
"""
function replay_cover(root::AbstractCoverNode, expected_root::DyadicBox,
                      references::AbstractDict{String, <:ReferenceBound};
                      family::Symbol, target::Rational = TARGET_KAPPA)
    audit = audit_tree(root, expected_root; family)
    failures = copy(audit.errors)
    leaves = _collect_leaves!(CoverLeaf[], root)
    checked_leaves = 0
    checked_corners = 0
    vacuous = 0
    for leaf in leaves
        leaf.status == :retained || continue
        checked_leaves += 1
        id = leaf.reference_id::String
        if !haskey(references, id)
            push!(failures, "retained leaf references missing certificate '$id'")
            continue
        end
        ref = references[id]
        if ref.id != id
            push!(failures, "reference dictionary key '$id' disagrees with stored id '$(ref.id)'")
            continue
        end
        if ref.family != family
            push!(failures, "reference '$id' belongs to $(ref.family), not $family")
            continue
        end
        result = certify_leaf(family, leaf.box, ref.gram, ref.u2_upper; target)
        checked_corners += length(result.corners)
        vacuous += result.vacuous
        result.passed || push!(failures, "corner comparison failed for reference '$id' on $(leaf.box)")
    end
    return ReplayResult(isempty(failures), audit, checked_leaves,
                        checked_corners, vacuous, failures)
end

end # module BoxCover
