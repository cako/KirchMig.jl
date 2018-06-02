export KirchMap

import Base: +, -, *, \, /, ==, transpose
import LinearMaps: LinearMap

struct KirchMap{T, F1, F2} <: LinearMap{T}
    f::F1
    fc::F2
    M::Int
    N::Int
    _ismutating::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
    _parallel_threaded_serial::String
end

# Constructors
function KirchMap{T}(f::F1, fc::F2, M::Int, N::Int;
                     ismutating::Bool = false,
                     issymmetric::Bool = false,
                     ishermitian::Bool=(T<:Real && issymmetric),
                     isposdef::Bool = false,
                     parallel_threaded_serial::String = "parallel") where {T,F1,F2}
    KirchMap{T,F1,F2}(f, fc, M, N, ismutating, issymmetric, ishermitian, isposdef, parallel_threaded_serial)
end

"""
`KirchMap{eltype(t)}(t, trav_r, [trav_s]; parallel_threaded_serial="serial")`

Construct a Kirchhoff LinearMap object, which can perform Kirchhoff modeling (or more accurately, demigration) as a
forward operator and Kirchhoff migration as an adjoint operation.

Parameters
----------
* `t` : `(nt,)`, `AbstractVector{<:Real}`

Contains time samples which correspond to the data domain. `KirchMap(t, trav)*m` will create data whose time axis is given by `t`. Its type defines the type of `KirchMap`.

* `trav_r` : `(nz, [nx, ny, ...], nr)`, `AbstractArray{<:Real, M}`

Contains traveltimes between each model parameter and receiver location. The first `M-1` dimensions are model dimensions `(z, x, y, ...)`, and the last dimension corresponds to receivers.

* `trav_s` : `(nz, [nx, ny, ...], ns)`, `AbstractArray{<:Real, M}`, optional

Like `trav_r` but for sources. If omitted, defaults to `trav_r`, i.e., sources and receivers are assumed colocated.

* `parallel_threaded_serial` : `String`

Defines which Kirchhoff methods to use. Can be the default, `parallel`, which uses Julia's distributed computing to parallelize over receivers; `threaded` which uses multi-threading to parallelize over receivers; or `serial`. It is highly recommended to not use the `serial` version, even when only using a single worker or thread.

Usage
-----
* Forward map

The forward map `L` multiplies a model vector of size `nz × nx × ny × ...` to create a data vector of size `nr × ns × nt`.

* Adjoint map

The adjoint map `L'` multiplies a data vector of size `nr × ns × nt` to create a model vector of size `nz × nx × ny × ...`

Description
-----------

The forward map computes the discretized version of the following operation
```math
d(r, s, t) = \\int m(x)\\,\\delta(t - \\tau_{sx} - \\tau_{xr})\\,\\mathrm{d}x.
```

and the adjoint map computes
```math
m(x) = \\int d(t, r, s)\\,\\delta(t - \\tau_{sx} - \\tau_{xr})\\,\\mathrm{d}r\\,\\mathrm{d}s\\,\\mathrm{d}t.
```

In both computations, the differential elements (which only affect amplitude) are neglected.
"""
KirchMap(t::AbstractVector{<:Real}, trav_r::AbstractArray{<:Real}; parallel_threaded_serial::String= "parallel") = KirchMap(t, trav_r, trav_r; parallel_threaded_serial=parallel_threaded_serial)
function KirchMap(t::AbstractVector{T},
                trav_r::AbstractArray{<:Real, N},
                trav_s::AbstractArray{<:Real, N};
                parallel_threaded_serial::String = "parallel") where {T<:Real, N}
    nr = size(trav_r)[end]
    ns = size(trav_s)[end]
    nzxy = size(trav_r)[1:N-1]
    NZXY = size(trav_s)[1:N-1]
    ot, dt, nt = t[1], t[2]-t[1], length(t)

    if nzxy != NZXY
        throw(DimensionMismatch("1st $(M-1) dimensions of trav_r and trav_s must be the same"))
    end

    if parallel_threaded_serial == "parallel"
        nworkers() == 1 ? warn("Using only one worker") : nothing
        return KirchMap{T}(x -> view(kirchmod_par(reshape(x, nzxy...), t, trav_r, trav_s), :),
                           x -> view(kirchmig_par(reshape(x, nr, ns, nt), t, trav_r, trav_s), :),
                           nr*ns*nt,
                           prod(nzxy))
    elseif parallel_threaded_serial == "threaded"
        Threads.nthreads() == 1 ? warn("Using only one thread") : nothing
        return KirchMap{T}(x -> view(kirchmod_thread(reshape(x, nzxy...), t, trav_r, trav_s), :),
                           x -> view(kirchmig_thread(reshape(x, nr, ns, nt), t, trav_r, trav_s), :),
                           nr*ns*nt,
                           prod(nzxy))
    else
        return KirchMap{T}(x -> view(kirchmod(reshape(x, nzxy...), t, trav_r, trav_s), :),
                           x -> view(kirchmig(reshape(x, nr, ns, nt), t, trav_r, trav_s), :),
                           nr*ns*nt,
                           prod(nzxy))
    end
end

# Inverse
\(A::KirchMap, b) = cg(A'A, A'b, maxiter=5, log=false);

# Show
function Base.show(io::IO, A::KirchMap{T}) where {T}
    print(io,"KirchMig.KirchMap{$T}($(A.f), $(A.fc), $(A.M), $(A.N); ismutating=$(A._ismutating), issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef), parallel_threaded_serial=$(A._parallel_threaded_serial))")
end

# Properties
Base.size(A::KirchMap) = (A.M, A.N)
Base.issymmetric(A::KirchMap) = A._issymmetric
Base.ishermitian(A::KirchMap) = A._ishermitian
Base.isposdef(A::KirchMap) = A._isposdef
ismutating(A::KirchMap) = A._ismutating
_ismutating(f) = first(methods(f)).nargs == 3

# Multiplication with vector
function Base.A_mul_B!(y::AbstractVector, A::KirchMap, x::AbstractVector)
    (length(x) == A.N && length(y) == A.M) || throw(DimensionMismatch())
    ismutating(A) ? A.f(y,x) : copy!(y,A.f(x))
    return y
end
function *(A::KirchMap, x::AbstractVector)
    length(x) == A.N || throw(DimensionMismatch())
    if ismutating(A)
        y = similar(x, promote_type(eltype(A), eltype(x)), A.M)
        A.f(y,x)
    else
        y = A.f(x)
    end
    return y
end

function Base.At_mul_B!(y::AbstractVector, A::KirchMap, x::AbstractVector)
    issymmetric(A) && return Base.A_mul_B!(y, A, x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A.fc != nothing
        if !isreal(A)
            x = conj(x)
        end
        (ismutating(A) ? A.fc(y,x) : copy!(y, A.fc(x)))
        if !isreal(A)
            conj!(y)
        end
        return y
    else
        error("transpose not implemented for $A")
    end
end
function Base.At_mul_B(A::KirchMap, x::AbstractVector)
    issymmetric(A) && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if A.fc != nothing
        if !isreal(A)
            x = conj(x)
        end
        if ismutating(A)
            y = similar(x, promote_type(eltype(A), eltype(x)), A.N)
            A.fc(y,x)
        else
            y = A.fc(x)
        end
        if !isreal(A)
            conj!(y)
        end
        return y
    else
        error("transpose not implemented for $A")
    end
end

function Base.Ac_mul_B!(y::AbstractVector, A::KirchMap, x::AbstractVector)
    ishermitian(A) && return Base.A_mul_B!(y,A,x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A.fc != nothing
        return (ismutating(A) ? A.fc(y, x) : copy!(y, A.fc(x)))
    else
        error("adjoint not implemented for $A")
    end
end
function Base.Ac_mul_B(A::KirchMap, x::AbstractVector)
    ishermitian(A) && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if A.fc != nothing
        if ismutating(A)
            y = similar(x, promote_type(eltype(A), eltype(x)), A.N)
            A.fc(y,x)
        else
            y = A.fc(x)
        end
        return y
    else
        error("adjoint not implemented for $A")
    end
end
