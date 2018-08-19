export eikonal_const_vel

"""
`eikonal_const_vel(src, z, x, [y, ] velocity)`

Compute traveltime tables between `src` locations and subsurface model locations given by `z`, `x`, and optionally `y`. It assumes `velocity` is constant.

Parameters
----------
* `src` : `(ns, n)`, `AbstractMatrix{<:Real}`

The `(z_s, x_s [, x_y])` coordinates for every source. In 2D, `n = 2` and in 3D `n = 3`.

* `z` : `(nz,)`, `AbstractVector{<:Real}`

Model depth `z` coordinates.

* `x` : `(nx,)`, `AbstractVector{<:Real}`

Model horizontal `x` coordinates.

* `y` : `(ny,)`, `AbstractVector{<:Real}`, optional

Model horizontal `y` coordinates.


* `velocity` : `Real`

Wave speed

Returns
-------

* `trav` : `(nz, nx, [ny, ] ns)`, `AbstractArray{<:Real, M}`

Traveltime between each src and locations given by `z`, `x` and optionally `y`:

```math
t(z, x, y, s) = \\frac{\\sqrt{(z-s_z)^2 + (x-s_x)^2 + (y-s_y)^2}}{v}
```
"""
function eikonal_const_vel(src::AbstractMatrix{<:Real},
                           z::AbstractVector{<:Real},
                           x::AbstractVector{<:Real},
                           velocity::Real)
    if velocity < 0
        @warn("Velocity cannot not be negative")
        throw(DomainError())
    end
    size(src, 2) == 2 || throw(DimensionMismatch("Second dimension of `src` must match number of model arguments"))

    ns = size(src, 1)
    trav = zeros(length(z), length(x), ns)
    Z = [i for i in z, j in x]
    X = [j for i in z, j in x]
    for is in 1:ns
        src_z, src_x = src[is, :]
        trav[:,:,is] = sqrt.((Z-src_z).^2 + (X - src_x).^2)./velocity
    end
    return trav
end
function eikonal_const_vel(src::AbstractMatrix{<:Real},
                           z::AbstractVector{<:Real},
                           x::AbstractVector{<:Real},
                           y::AbstractVector{<:Real},
                           velocity::Real)
    if velocity < 0
        @warn("Velocity cannot not be negative")
        throw(DomainError())
    end
    size(src, 2) == 3 || throw(DimensionMismatch("Second dimension of `src` must match number of model arguments"))

    ns = size(src, 1)
    trav = zeros(length(z), length(x), length(y), ns)
    Z = [i for i in z, j in x, k in y]
    X = [j for i in z, j in x, k in y]
    Y = [k for i in z, j in x, k in y]
    for is in 1:ns
        src_z, src_x, src_y = src[is, :]
        trav[:,:,:,is] = sqrt.((Z - src_z).^2 + (X-src_x).^2 + (Y-src_y).^2)./velocity
    end
    return trav
end
