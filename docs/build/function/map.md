
<a id='Kirchhoff-map-1'></a>

## Kirchhoff map

<a id='KirchMig.KirchMap' href='#KirchMig.KirchMap'>#</a>
**`KirchMig.KirchMap`** &mdash; *Type*.



`KirchMap{eltype(t)}(t, trav_r, [trav_s]; parallel_threaded_serial="serial")`

Construct a Kirchhoff LinearMap object, which can perform Kirchhoff modeling (or more accurately, demigration) as a forward operator and Kirchhoff migration as an adjoint operation.

**Parameters**

  * `t` : `(nt,)`, `AbstractVector{<:Real}`

Contains time samples which correspond to the data domain. `KirchMap(t, trav)*m` will create data whose time axis is given by `t`. Its type defines the type of `KirchMap`.

  * `trav_r` : `(nz, [nx, ny, ...], nr)`, `AbstractArray{<:Real, M}`

Contains traveltimes between each model parameter and receiver location. The first `M-1` dimensions are model dimensions `(z, x, y, ...)`, and the last dimension corresponds to receivers.

  * `trav_s` : `(nz, [nx, ny, ...], ns)`, `AbstractArray{<:Real, M}`, optional

Like `trav_r` but for sources. If omitted, defaults to `trav_r`, i.e., sources and receivers are assumed colocated.

  * `parallel_threaded_serial` : `String`

Defines which Kirchhoff methods to use. Can be the default, `parallel`, which uses Julia's distributed computing to parallelize over receivers; `threaded` which uses multi-threading to parallelize over receivers; or `serial`. It is highly recommended to not use the `serial` version, even when only using a single worker or thread.

**Usage**

  * Forward map

The forward map `L` multiplies a model vector of size `nz × nx × ny × ...` to create a data vector of size `nr × ns × nt`.

  * Adjoint map

The adjoint map `L'` multiplies a data vector of size `nr × ns × nt` to create a model vector of size `nz × nx × ny × ...`

**Description**

The forward map computes the discretized version of the following operation

$$
d(r, s, t) = \int m(x)\,\delta(t - \tau_{sx} - \tau_{xr})\,\mathrm{d}x.
$$

and the adjoint map computes

$$
m(x) = \int d(t, r, s)\,\delta(t - \tau_{sx} - \tau_{xr})\,\mathrm{d}r\,\mathrm{d}s\,\mathrm{d}t.
$$

In both computations, the differential elements (which only affect amplitude) are neglected.

