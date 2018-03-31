export LaplacianMap, DiffZMap
import LinearMaps: FunctionMap

function diff_z(x, n...)
    x_ = zeros((eltype(n)[n...]+2)...)
    x_[[2:n_+1 for n_ in n]...] = reshape(x, n...)

    colons = [Colon() for i=1:length(n)-1]

    dz = zero(x_)
    @fastmath @inbounds @simd for i in 2:n[1]+1
        dz[i,colons...] = (x_[i+1,colons...] - x_[i-1,colons...])/2.
    end
    return dz[[2:n_+1 for n_ in n]...]
end

function laplacian(x, nz, nx)
    x_ = zeros(nz+2, nx+2)
    x_[2:nz+1, 2:nx+1] = reshape(x, nz, nx)

    dx = zero(x_)
    dz = zero(x_)
    @fastmath @inbounds @simd for i in 2:nz+1
        dz[i,:] = -x_[i-1,:] + 2x_[i,:] + -x_[i+1,:]
    end
    @fastmath @inbounds @simd for i in 2:nx+1
        dx[:,i] = -x_[:,i-1] + 2x_[:,i] + -x_[:,i+1]
    end
    return dx[2:nz+1, 2:nx+1] + dz[2:nz+1, 2:nx+1]
end

function laplacian(x, nz, nx, ny)
    x_ = zeros(nz+2, nx+2, ny+2)
    x_[2:nz+1, 2:nx+1, 2:ny+1] = reshape(x, nz, nx, ny)

    dx = zero(x_)
    dz = zero(x_)
    dy = zero(x_)
    @fastmath @inbounds @simd for i in 2:nz+1
        dz[i,:,:] = -x_[i-1,:,:] + 2x_[i,:,:] + -x_[i+1,:,:]
    end
    @fastmath @inbounds @simd for i in 2:nx+1
        dx[:,i,:] = -x_[:,i-1,:] + 2x_[:,i,:] + -x_[:,i+1,:]
    end
    @fastmath @inbounds @simd for i in 2:ny+1
        dy[:,:,i] = -x_[:,:,i-1] + 2x_[:,:,i] + -x_[:,:,i+1]
    end
    return dx[2:nz+1, 2:nx+1, 2:ny+1] + dy[2:nz+1, 2:nx+1, 2:ny+1] + dz[2:nz+1, 2:nx+1, 2:ny+1]
end

"""
`LaplacianMap([T,] n...) -> Δ`

Construct a discretized Laplacian operator Δ which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `Δ`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `Δ`.

Usage
-----
* Forward map and Adjoint maps

The forward map `Δ` multiplies a model vector of size `nz × nx × ny × ...` its second order derivative. It is symmetric.

Description
-----------

The forward and adjoint maps computes the following operation
```math
Δx = \\sum_{i} δ_i x
```
where 
```math
δ_l x_{i,j,k,...} = - x_{...,l+1,...} + 2x_{...,l,...} - x_{...,l-1,...}
```
"""
LaplacianMap(T::Type, n::Int...) = FunctionMap{T}(x -> laplacian(x, n...)[:], prod(n), issymmetric=true)
LaplacianMap(n::Int...) = LaplacianMap(Float64, n...)

"""
`DiffZMap([T,] n...) -> δz`

Construct a discretized z-derivative operator δz which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `δz`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `δz`.

Usage
-----
* Forward map and Adjoint maps

The forward map `δz` multiplies a model vector of size `nz × nx × ny × ...` its first order z-derivative. The adjoint map is minus the forward map.

Description
-----------

The forward map computes the following operation
```math
δz x_{i,j,k,...} = (x_{l+1,...} - x_{l-1,...})/2
```
and the adjoint map computes `-δz`.
"""
DiffZMap(T::Type, n::Int...) = FunctionMap{T}(x -> diff_z(x, n...)[:], x -> -diff_z(x, n...)[:], prod(n), prod(n))
DiffZMap(n::Int...) = DiffZMap(Float64, n...)
