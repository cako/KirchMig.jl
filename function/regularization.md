<a id='KirchMig.ConvMap' href='#KirchMig.ConvMap'>#</a>
**`KirchMig.ConvMap`** &mdash; *Function*.



`ConvMap([T,] wavelet, n...) -> W`

Construct a convolution/correlation operator `W` which will act on an `AbstractVector`.

**Parameters**

  * `T` : `Type`, optional

`Type` of argument of `W`. Defaults to `Float64`.

  * `wavelet`

One-dimensional wavelet.

  * `n...`

Sequence of dimensions of the data.

**Usage**

  * Forward map and Adjoint maps

The forward map `W` convolves the last dimension of the input with `wavelet`. The adjoint map `W` correlates the last dimension of the input with `wavelet`.


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/017cea17635b5c6f68de628dd0502e9a268fa3ff/src/regularization.jl#L5-L31' class='documenter-source'>source</a><br>

<a id='KirchMig.LaplacianMap' href='#KirchMig.LaplacianMap'>#</a>
**`KirchMig.LaplacianMap`** &mdash; *Function*.



`LaplacianMap([T,] n...) -> Δ`

Construct a discretized Laplacian operator `Δ` which will act on an `AbstractVector`.

**Parameters**

  * `T` : `Type`, optional

`Type` of argument of `Δ`. Defaults to `Float64`.

  * `n...`

Sequence of spatial dimensions of `Δ`.

**Usage**

  * Forward map and Adjoint maps

The forward map `Δ` multiplies a model vector of size `nz × nx × ny × ...` its second order derivative. It is symmetric.

**Description**

The forward and adjoint maps computes the following operation

$$
Δm = \sum_{i} δ_i m
$$

where 

$$
δ_l m_{i,j,k,...} = - m_{...,l+1,...} + 2m_{...,l,...} - m_{...,l-1,...}
$$


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/017cea17635b5c6f68de628dd0502e9a268fa3ff/src/regularization.jl#L36-L68' class='documenter-source'>source</a><br>

<a id='KirchMig.DiffZMap' href='#KirchMig.DiffZMap'>#</a>
**`KirchMig.DiffZMap`** &mdash; *Function*.



`DiffZMap([T,] n...) -> δz`

Construct a discretized z-derivative operator `δz` which will act on an `AbstractVector`.

**Parameters**

  * `T` : `Type`, optional

`Type` of argument of `δz`. Defaults to `Float64`.

  * `n...`

Sequence of spatial dimensions of `δz`.

**Usage**

  * Forward map and Adjoint maps

The forward map `δz` multiplies a model vector of size `nz × nx × ny × ...` its first order z-derivative. The adjoint map is minus the forward map.

**Description**

The forward map computes the following operation

$$
δ_z m_{i,j,k,...} = (m_{l+1,...} - m_{l-1,...})/2
$$

and the adjoint map computes `-δz`.


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/017cea17635b5c6f68de628dd0502e9a268fa3ff/src/regularization.jl#L72-L101' class='documenter-source'>source</a><br>

<a id='KirchMig.DiffXMap' href='#KirchMig.DiffXMap'>#</a>
**`KirchMig.DiffXMap`** &mdash; *Function*.



`DiffXMap([T,] n...) -> δx`

Construct a discretized x-derivative operator `δx` which will act on an `AbstractVector`.

**Parameters**

  * `T` : `Type`, optional

`Type` of argument of `δx`. Defaults to `Float64`.

  * `n...`

Sequence of spatial dimensions of `δx`.

**Usage**

  * Forward map and Adjoint maps

The forward map `δx` multiplies a model vector of size `nz × nx × ny × ...` its first order x-derivative. The adjoint map is minus the forward map.

**Description**

The forward map computes the following operation

$$
δ_x m_{i,j,k,...} = (m_{l+1,...} - m_{l-1,...})/2
$$

and the adjoint map computes `-δx`.


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/017cea17635b5c6f68de628dd0502e9a268fa3ff/src/regularization.jl#L106-L135' class='documenter-source'>source</a><br>

<a id='KirchMig.GradDivMap' href='#KirchMig.GradDivMap'>#</a>
**`KirchMig.GradDivMap`** &mdash; *Function*.



`GradDivMap([T,] n...) -> GD`

Construct a discretized gradient operator GD which will act on an `AbstractVector`.

**Parameters**

  * `T` : `Type`, optional

`Type` of argument of `GD`. Defaults to `Float64`.

  * `n...`

Sequence of spatial dimensions of `GD`.

**Usage**

  * Forward map

Calculates the discrete gradient of a `nz × nx × ny × ...` using first order central differences.

$$
∇m_{i,j,k,...} = [δ_x m_{i,j,k,...},  ..., δ_z m_{i,j,k,...}]
$$

  * Adjoint map

Calculates the discrete negative divergence of a `nz × nx × ny × ...` using first order central differences.

$$
∇\cdot m_{i,j,k,...} = δ_x m_{i,j,k,...} +  ... + δ_z m_{i,j,k,...}
$$


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/017cea17635b5c6f68de628dd0502e9a268fa3ff/src/regularization.jl#L140-L170' class='documenter-source'>source</a><br>

