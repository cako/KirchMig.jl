
<a id='Traveltime-computations-1'></a>

## Traveltime computations

<a id='KirchMig.eikonal_const_vel' href='#KirchMig.eikonal_const_vel'>#</a>
**`KirchMig.eikonal_const_vel`** &mdash; *Function*.



`eikonal_const_vel(src, z, x, [y, ] velocity)`

Compute traveltime tables between `src` locations and subsurface model locations given by `z`, `x`, and optionally `y`. It assumes `velocity` is constant.

**Parameters**

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

**Returns**

  * `trav` : `(nz, nx, [ny, ] ns)`, `AbstractArray{<:Real, M}`

Traveltime between each src and locations given by `z`, `x` and optionally `y`:

$$
t(z, x, y, s) = \frac{\sqrt{(z-s_z)^2 + (x-s_x)^2 + (y-s_y)^2}}{v}
$$


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/ad7452c642bdd0913a90176158cebda42a45619c/src/eikonal.jl#L3-L41' class='documenter-source'>source</a><br>

