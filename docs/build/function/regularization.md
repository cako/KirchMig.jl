<a id='KirchMig.LaplacianMap' href='#KirchMig.LaplacianMap'>#</a>
**`KirchMig.LaplacianMap`** &mdash; *Function*.



`LaplacianMap([T,] n...) -> Δ`

Construct a discretized Laplacian operator Δ which will act on an `AbstractVector`.

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
Δx = \sum_{i} δ_i x
$$

where 

$$
δ_l x_{i,j,k,...} = - x_{...,l+1,...} + 2x_{...,l,...} - x_{...,l-1,...}
$$


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/ad7452c642bdd0913a90176158cebda42a45619c/src/regularization.jl#L51-L83' class='documenter-source'>source</a><br>

<a id='KirchMig.DiffZMap' href='#KirchMig.DiffZMap'>#</a>
**`KirchMig.DiffZMap`** &mdash; *Function*.



`DiffZMap([T,] n...) -> δz`

Construct a discretized z-derivative operator δz which will act on an `AbstractVector`.

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
δz x_{i,j,k,...} = (x_{l+1,...} - x_{l-1,...})/2
$$

and the adjoint map computes `-δz`.


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/ad7452c642bdd0913a90176158cebda42a45619c/src/regularization.jl#L87-L116' class='documenter-source'>source</a><br>

