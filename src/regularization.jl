export LaplacianMap, DiffZMap, DiffXMap, GradDivMap
import LinearMaps: FunctionMap

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
DiffZMap(T::Type, n::Int...) = FunctionMap{T}(x ->  deriv_z(x, n...)[:],
                                              x -> -deriv_z(x, n...)[:], prod(n), prod(n))
DiffZMap(n::Int...) = DiffZMap(Float64, n...)

"""
`DiffXMap([T,] n...) -> δx`

Construct a discretized x-derivative operator δx which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `δx`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `δx`.

Usage
-----
* Forward map and Adjoint maps

The forward map `δx` multiplies a model vector of size `nz × nx × ny × ...` its first order x-derivative. The adjoint map is minus the forward map.

Description
-----------

The forward map computes the following operation
```math
δx x_{i,j,k,...} = (x_{l+1,...} - x_{l-1,...})/2
```
and the adjoint map computes `-δx`.
"""
DiffXMap(T::Type, n::Int...) = FunctionMap{T}(x ->  deriv_x(x, n...)[:],
                                              x -> -deriv_x(x, n...)[:], prod(n), prod(n))
DiffXMap(n::Int...) = DiffXMap(Float64, n...)

"""
`GradDivMap([T,] n...) -> GD`

Construct a discretized gradient operator GD which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `GD`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `GD`.

Usage
-----
* Forward map

Calculates the discrete gradient of a `nz × nx × ny × ...` using first order central differences.
```math
∇x_{i,j,k,...} = [δx x_{i,j,k,...},  ..., δz x_{i,j,k,...}]
```

* Adjoint map

Calculates the discrete negative divergence of a `nz × nx × ny × ...` using first order central differences.
```math
∇\\cdot x_{i,j,k,...} = δx x_{i,j,k,...} +  ... + δz x_{i,j,k,...}
```
"""
GradDivMap(T::Type, nz, nx) = FunctionMap{T}(x -> gradient(x, nz, nx)[:],
                                             x -> -divergence(x, nz, nx)[:], 2nz*nx, nz*nx)
GradDivMap(T::Type, nz, nx, ny) = FunctionMap{T}(x -> gradient(x, nz, nx, ny)[:],
                                                 x -> -divergence(x, nz, nx, ny)[:], 3nz*nx*ny, nz*nx*ny)
GradDivMap(n...) = GradDivMap(Float64, n...)

laplacian(x::AbstractArray) = laplacian(x, size(x)...)
function laplacian(x::AbstractArray{T}, nz, nx) where T
    x_ = zeros(T, nz+2, nx+2)
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

function laplacian(x::AbstractArray{T}, nz, nx, ny) where T
    x_ = zeros(T, nz+2, nx+2, ny+2)
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

deriv_z(x::AbstractArray) = deriv_z(x, size(x)...)
function deriv_z(x::AbstractArray{T}, n...) where T
    x_ = zeros(T, n[1]+2, n[2:end]...)

    colons = [Colon() for i=2:length(n)]

    x_[2:n[1]+1, colons...] = reshape(x, n[1], n[2:end]...)

    dz = zero(x_)
    @fastmath @inbounds @simd for i in 2:n[1]+1
        dz[i,colons...] = (x_[i+1,colons...] - x_[i-1,colons...])/2.
    end
    return dz[2:n[1]+1, colons...]
end

deriv_x(x::AbstractArray) = deriv_x(x, size(x)...)
function deriv_x(x::AbstractArray{T}, n...) where T
    x_ = zeros(T, n[1], n[2]+2, n[3:end]...)

    colons = [Colon() for i=3:length(n)]

    x_[:, 2:n[2]+1, colons...] = reshape(x, n[1], n[2], n[3:end]...)

    dx = zero(x_)
    @fastmath @inbounds @simd for i in 2:n[2]+1
        dx[:,i,colons...] = (x_[:,i+1,colons...] - x_[:,i-1,colons...])/2.
    end
    return dx[:,2:n[2]+1, colons...]
end

deriv_y(x::AbstractArray) = deriv_y(x, size(x)...)
function deriv_y(x::AbstractArray{T}, n...) where T
    x_ = zeros(T, n[1:2]..., n[3]+2, n[4:end]...)

    colons = [Colon() for i=4:length(n)]

    x_[:, :, 2:n[3]+1, colons...] = reshape(x, n[1:2]..., n[3], n[4:end]...)

    dy = zero(x_)
    @fastmath @inbounds @simd for i in 2:n[3]+1
        dy[:,:,i,colons...] = (x_[:,:,i+1,colons...] - x_[:,:,i-1,colons...])/2.
    end
    return dy[:,:,2:n[3]+1, colons...]
end

gradient(x::AbstractArray) = gradient(x, size(x)...)
function gradient(x::AbstractArray{T}, nz, nx) where T
    grad = zeros(T, nz, nx, 2)
    grad[:,:,1] = deriv_x(x, nz, nx)
    grad[:,:,2] = deriv_z(x, nz, nx)
    return grad
end

function gradient(x::AbstractArray{T}, nz, nx, ny) where T
    grad = zeros(T, nz, nx, ny, 3)
    grad[:,:,:,1] = deriv_x(x, nz, nx, ny)
    grad[:,:,:,2] = deriv_y(x, nz, nx, ny)
    grad[:,:,:,3] = deriv_z(x, nz, nx, ny)
    return grad
end

divergence(x::AbstractArray) = divergence(x, size(x)...)
function divergence(x::AbstractArray, nz, nx)
    x_ = reshape(x, nz, nx, 2)
    return deriv_x(x_[:,:,1], nz, nx) + deriv_z(x_[:,:,2], nz, nx)
end

function divergence(x::AbstractArray, nz, nx, ny)
    x_ = reshape(x, nz, nx, ny, 3)
    return deriv_x(x_[:,:,:,1], nz, nx, ny) + deriv_y(x_[:,:,:,2], nz, nx, ny) + deriv_z(x_[:,:,:,3], nz, nx, ny)
end
