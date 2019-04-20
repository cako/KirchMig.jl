export ConvMap, LaplacianMap, DiffZMap, DiffXMap, GradDivMap
import DSP: hilbert, conv
import LinearMaps: LinearMap

"""
`ConvMap([T,] wavelet, n...) -> W`

Construct a convolution/correlation operator `W` which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `W`. Defaults to `Float64`.

* `wavelet`

One-dimensional wavelet.

* `n...`

Sequence of dimensions of the data.

Usage
-----
* Forward map and Adjoint maps

The forward map `W` convolves the last dimension of the input with `wavelet`.
The adjoint map `W` correlates the last dimension of the input with `wavelet`.

"""
ConvMap(T::Type, wav, n...) = LinearMap{T}(x -> wav_conv(x, wav, n...)[:],
                                           x -> wav_corr(x, wav, n...)[:], prod(n), prod(n))
ConvMap(wav, n...) = ConvMap(Float64, wav, n...)

"""
`LaplacianMap([T,] n...) -> Δ`

Construct a discretized Laplacian operator `Δ` which will act on an `AbstractVector`.

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
Δm = \\sum_{i} δ_i m
```
where 
```math
δ_l m_{i,j,k,...} = - m_{...,l+1,...} + 2m_{...,l,...} - m_{...,l-1,...}
```
"""
LaplacianMap(T::Type, n::Int...) = LinearMap{T}(x -> laplacian(x, n...)[:], prod(n), issymmetric=true)
LaplacianMap(n::Int...) = LaplacianMap(Float64, n...)

"""
`DiffZMap([T,] n...[; order=4]) -> δz`

Construct a discretized z-derivative operator `δz` of order `order` which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `δz`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `δz`.

* `order` : `Int`, optional

Order of central finite-difference approximation. Available options are 2, 4 and 8.


Usage
-----
The forward map `δz` multiplies a model vector of size `nz × nx × ny × ...` its first order z-derivative. The adjoint map is minus the forward map.

Description
-----------

```math
δ_z m_{i,...} = (m_{i+1,...} - m_{i-1,...})/2
```
for `order=2`,
```math
δ_z m_{i,...} =
\\left(\\frac{1}{12} m_{i+2,...} - \\frac{2}{3}  m_{i+1,...}
     + \\frac{2}{3}  m_{i-1,...} - \\frac{1}{12} m_{i-2,...}
\\right)
```
for `order=4`, and
```math
δ_z m_{i,...} =
\\left(\\frac{1}{280} m_{i+4,...} - \\frac{4}{105} m_{i+3,...}
     + \\frac{1}{5}   m_{i+2,...} - \\frac{4}{5}   m_{i+1,...}
     - \\frac{1}{280} m_{i+4,...} + \\frac{4}{105} m_{i+3,...}
     - \\frac{1}{5}   m_{i+2,...} + \\frac{4}{5}   m_{i+1,...}
\\right)
```
for `order=8`. The adjoint map for any order computes `-δz`.
"""
DiffZMap(T::Type, n::Int...; order=4) = LinearMap{T}(
    x -> copy( deriv_z(x, n...; order=order)[:]),
    x -> copy(-deriv_z(x, n...; order=order)[:]),
    prod(n), prod(n))
DiffZMap(n::Int...; order=4) = DiffZMap(Float64, n...; order=order)

"""
`DiffZMap([T,] n...[; order=4]) -> δx`

Construct a discretized x-derivative operator `δx` of order `order` which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `δx`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `δx`.

* `order` : `Int`, optional

Order of central finite-difference approximation. Available options are 2, 4 and 8.


Usage
-----
The forward map `δx` multiplies a model vector of size `nz × nx × ny × ...` its first order x-derivative. The adjoint map is minus the forward map.

Description
-----------

```math
δ_x m_{i,j,...} = (m_{i,j+1,...} - m_{i,j-1,...})/2
```
for `order=2`,
```math
δ_x m_{i,j,k,...} =
\\left(\\frac{1}{12} m_{i,j+2,...} - \\frac{2}{3}  m_{i,j+1,...}
     + \\frac{2}{3}  m_{i,j-1,...} - \\frac{1}{12} m_{i,j-2,...}
\\right)
```
for `order=4`, and
```math
δ_x m_{i,j,...} =
\\left(\\frac{1}{280} m_{i,j+4,...} - \\frac{4}{105} m_{i,j+3,...}
     + \\frac{1}{5}   m_{i,j+2,...} - \\frac{4}{5}   m_{i,j+1,...}
     - \\frac{1}{280} m_{i,j+4,...} + \\frac{4}{105} m_{i,j+3,...}
     - \\frac{1}{5}   m_{i,j+2,...} + \\frac{4}{5}   m_{i,j+1,...}
\\right)
```
for `order=8`. The adjoint map for any order computes `-δz`.
"""
DiffXMap(T::Type, n::Int...; order=4) = LinearMap{T}(
    x ->  deriv_x(x, n...; order=order)[:],
    x -> -deriv_x(x, n...; order=order)[:],
    prod(n), prod(n))
DiffXMap(n::Int...; order=4) = DiffXMap(Float64, n...; order=order)

"""
`GradDivMap([T,] n...[; order=4]) -> GD`

Construct a discretized gradient operator `GD` which will act on an `AbstractVector`.

Parameters
----------
* `T` : `Type`, optional

`Type` of argument of `GD`. Defaults to `Float64`.

* `n...`

Sequence of spatial dimensions of `GD`.

* `order` : `Int`, optional

Order of central finite-difference approximation. Available options are 2, 4 and 8.

Usage
-----
* Forward map

Calculates the discrete gradient of a `nz × nx × ny × ...` using central differences.
```math
∇m_{i,j,k,...} = [δ_x m_{i,j,k,...},  ..., δ_z m_{i,j,k,...}]
```

* Adjoint map

Calculates the discrete negative divergence of a `nz × nx × ny × ...` using central differences.
```math
-∇\\cdot m_{i,j,k,...} = -δ_x m_{i,j,k,...} -  ... - δ_z m_{i,j,k,...}
```
"""
GradDivMap(T::Type, nz, nx; order=4) = LinearMap{T}(
    x ->    gradient(x, nz, nx; order=order)[:],
    x -> -divergence(x, nz, nx; order=order)[:],
    2nz*nx, nz*nx)
GradDivMap(T::Type, nz, nx, ny; order=4) = LinearMap{T}(
    x ->    gradient(x, nz, nx, ny; order=order)[:],
    x -> -divergence(x, nz, nx, ny; order=order)[:],
    3nz*nx*ny, nz*nx*ny)
GradDivMap(n...; order=4) = GradDivMap(Float64, n...; order=order)

# Auxiliary functions
function wav_conv(data::AbstractArray, wavelet::AbstractVector, nr, nt)
    kt = argmax(abs.(hilbert(wavelet)))
    return mapslices(x->conv(x, wavelet)[kt:kt+nt-1],
                     reshape(copy(data), nr, nt), dims=[2])
end

function wav_corr(data::AbstractArray, wavelet::AbstractVector, nr, nt)
    kt = argmax(abs.(hilbert(wavelet)))
    return mapslices(x->conv(x, wavelet[end:-1:1])[nt-kt+1:2nt-kt],
                     reshape(copy(data), nr, nt), dims=[2]);
end

function wav_conv(data::AbstractArray, wavelet::AbstractVector, nr, ns, nt)
    kt = argmax(abs.(hilbert(wavelet)))
    return mapslices(x->conv(x, wavelet)[kt:kt+nt-1],
                     reshape(copy(data), nr, ns, nt), dims=[3])
end

function wav_corr(data::AbstractArray, wavelet::AbstractVector, nr, ns, nt)
    kt = argmax(abs.(hilbert(wavelet)))
    return mapslices(x->conv(x, wavelet[end:-1:1])[nt-kt+1:2nt-kt],
                     reshape(copy(data), nr, ns, nt), dims=[3]);
end

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

deriv_z(x::AbstractArray; order=4) = deriv_z(x, size(x)...; order=order)
function deriv_z(x::AbstractArray{T}, n...; order=4) where T
    colons = [Colon() for i=2:length(n)]

    x_ = reshape(x, n...)
    dz = zero(x_)

    zero_slice = zero(x_[1, colons...])
    if order == 2
        @fastmath @inbounds @simd for i in 1:n[1]
            xm1 = i-1 >= 1    ? x_[i-1, colons...] : zero_slice
            xp1 = i+1 <= n[1] ? x_[i+1, colons...] : zero_slice
            @. dz[i,colons...] = (xp1 - xm1)/T(2)
        end
    elseif order == 8
        @fastmath @inbounds @simd for i in 1:n[1]
            xm1 = i-1 >= 1    ? x_[i-1, colons...] : zero_slice
            xp1 = i+1 <= n[1] ? x_[i+1, colons...] : zero_slice

            xm2 = i-2 >= 1    ? x_[i-2, colons...] : zero_slice
            xp2 = i+2 <= n[1] ? x_[i+2, colons...] : zero_slice

            xm3 = i-3 >= 1    ? x_[i-3, colons...] : zero_slice
            xp3 = i+3 <= n[1] ? x_[i+3, colons...] : zero_slice

            xm4 = i-4 >= 1    ? x_[i-4, colons...] : zero_slice
            xp4 = i+4 <= n[1] ? x_[i+4, colons...] : zero_slice

            @. dz[i,colons...] = ( xm4/T(280) - T(4)*xm3/T(105) + xm2/T(5) - T(4)*xm1/T(5) +
                                  -xp4/T(280) + T(4)*xp3/T(105) - xp2/T(5) + T(4)*xp1/T(5))
        end
    else
        @fastmath @inbounds @simd for i in 1:n[1]
            xm1 = i-1 >= 1    ? x_[i-1, colons...] : zero_slice
            xp1 = i+1 <= n[1] ? x_[i+1, colons...] : zero_slice

            xm2 = i-2 >= 1    ? x_[i-2, colons...] : zero_slice
            xp2 = i+2 <= n[1] ? x_[i+2, colons...] : zero_slice

            @. dz[i,colons...] = xm2/T(12) - T(2)*xm1/T(3) + T(2)*xp1/T(3) - xp2/T(12)
        end
    end

    return dz
end

deriv_x(x::AbstractArray; order=4) = deriv_x(x, size(x)...; order=order)
function deriv_x(x::AbstractArray{T}, n...; order=4) where T
    colons = [Colon() for i=3:length(n)]

    x_ = reshape(x, n...)
    dz = zero(x_)

    zero_slice = zero(x_[:, 1, colons...])
    if order == 2
        @fastmath @inbounds @simd for i in 1:n[2]
            xm1 = i-1 >= 1    ? x_[:,i-1, colons...] : zero_slice
            xp1 = i+1 <= n[2] ? x_[:,i+1, colons...] : zero_slice
            @. dz[:,i,colons...] = (xp1 - xm1)/T(2)
        end
    elseif order == 8
        @fastmath @inbounds @simd for i in 1:n[2]
            xm1 = i-1 >= 1    ? x_[:,i-1, colons...] : zero_slice
            xp1 = i+1 <= n[2] ? x_[:,i+1, colons...] : zero_slice

            xm2 = i-2 >= 1    ? x_[:,i-2, colons...] : zero_slice
            xp2 = i+2 <= n[2] ? x_[:,i+2, colons...] : zero_slice

            xm3 = i-3 >= 1    ? x_[:,i-3, colons...] : zero_slice
            xp3 = i+3 <= n[2] ? x_[:,i+3, colons...] : zero_slice

            xm4 = i-4 >= 1    ? x_[:,i-4, colons...] : zero_slice
            xp4 = i+4 <= n[2] ? x_[:,i+4, colons...] : zero_slice

            @. dz[:,i,colons...] = ( xm4/T(280) - T(4)*xm3/T(105) + xm2/T(5) - T(4)*xm1/T(5) +
                                    -xp4/T(280) + T(4)*xp3/T(105) - xp2/T(5) + T(4)*xp1/T(5))
        end
    else
        @fastmath @inbounds @simd for i in 1:n[2]
            xm1 = i-1 >= 1    ? x_[:,i-1, colons...] : zero_slice
            xp1 = i+1 <= n[2] ? x_[:,i+1, colons...] : zero_slice

            xm2 = i-2 >= 1    ? x_[:,i-2, colons...] : zero_slice
            xp2 = i+2 <= n[2] ? x_[:,i+2, colons...] : zero_slice

            @. dz[:,i,colons...] = xm2/T(12) - T(2)*xm1/T(3) + T(2)*xp1/3 - xp2/T(12)
        end
    end

    return dz
end

deriv_y(x::AbstractArray; order=4) = deriv_y(x, size(x)...; order=order)
function deriv_y(x::AbstractArray{T}, n...; order=4) where T
    colons = [Colon() for i=4:length(n)]

    x_ = reshape(x, n...)
    dz = zero(x_)

    zero_slice = zero(x_[:, :, 1, colons...])
    if order == 2
        @fastmath @inbounds @simd for i in 1:n[3]
            xm1 = i-1 >= 1    ? x_[:,:,i-1, colons...] : zero_slice
            xp1 = i+1 <= n[3] ? x_[:,:,i+1, colons...] : zero_slice
            @. dz[:,:,i,colons...] = (xp1 - xm1)/T(2)
        end
    elseif order == 8
        @fastmath @inbounds @simd for i in 1:n[3]
            xm1 = i-1 >= 1    ? x_[:,:,i-1, colons...] : zero_slice
            xp1 = i+1 <= n[3] ? x_[:,:,i+1, colons...] : zero_slice

            xm2 = i-2 >= 1    ? x_[:,:,i-2, colons...] : zero_slice
            xp2 = i+2 <= n[3] ? x_[:,:,i+2, colons...] : zero_slice

            xm3 = i-3 >= 1    ? x_[:,:,i-3, colons...] : zero_slice
            xp3 = i+3 <= n[3] ? x_[:,:,i+3, colons...] : zero_slice

            xm4 = i-4 >= 1    ? x_[:,:,i-4, colons...] : zero_slice
            xp4 = i+4 <= n[3] ? x_[:,:,i+4, colons...] : zero_slice

            @. dz[:,:,i,colons...] = ( xm4/T(280) - T(4)*xm3/T(105) + xm2/T(5) - T(4)*xm1/T(5) +
                                      -xp4/T(280) + T(4)*xp3/T(105) - xp2/T(5) + T(4)*xp1/T(5))
        end
    else
        @fastmath @inbounds @simd for i in 1:n[3]
            xm1 = i-1 >= 1    ? x_[:,:,i-1, colons...] : zero_slice
            xp1 = i+1 <= n[3] ? x_[:,:,i+1, colons...] : zero_slice

            xm2 = i-2 >= 1    ? x_[:,:,i-2, colons...] : zero_slice
            xp2 = i+2 <= n[3] ? x_[:,:,i+2, colons...] : zero_slice

            @. dz[:,:,i,colons...] = xm2/T(12) - T(2)*xm1/T(3) + T(2)*xp1/3 - xp2/T(12)
        end
    end

    return dz
end

gradient(x::AbstractArray; order=4) = gradient(x, size(x)...; order=order)
function gradient(x::AbstractArray{T}, nz, nx; order=4) where T
    grad = zeros(T, nz, nx, 2)
    grad[:,:,1] = deriv_x(x, nz, nx; order=order)
    grad[:,:,2] = deriv_z(x, nz, nx; order=order)
    return grad
end

function gradient(x::AbstractArray{T}, nz, nx, ny; order=4) where T
    grad = zeros(T, nz, nx, ny, 3)
    grad[:,:,:,1] = deriv_x(x, nz, nx, ny; order=order)
    grad[:,:,:,2] = deriv_y(x, nz, nx, ny; order=order)
    grad[:,:,:,3] = deriv_z(x, nz, nx, ny; order=order)
    return grad
end

divergence(x::AbstractArray; order=4) = divergence(x, size(x)...; order=order)
function divergence(x::AbstractArray, nz, nx; order=4)
    x_ = reshape(x, nz, nx, 2)
    return deriv_x(x_[:,:,1], nz, nx; order=order) + deriv_z(x_[:,:,2], nz, nx; order=order)
end

function divergence(x::AbstractArray, nz, nx, ny; order=4)
    x_ = reshape(x, nz, nx, ny, 3)
    return deriv_x(x_[:,:,:,1], nz, nx, ny; order=order) +
           deriv_y(x_[:,:,:,2], nz, nx, ny; order=order) +
           deriv_z(x_[:,:,:,3], nz, nx, ny; order=order)
end
