
A **simple** conjugate gradients solver is provided, but using [IterativeSolvers.jl](https://juliamath.github.io/IterativeSolvers.jl/latest/) is highly recommended.

<a id='KirchMig.cg' href='#KirchMig.cg'>#</a>
**`KirchMig.cg`** &mdash; *Function*.



`cg(A, b [, x0];  maxiter=10, tol=1e-3, log=false) -> x [, history_x, history_r]`

Solve the system `Ax = b` of a symmetric, positive definite linear map given by `A` using the conjugate gradients method.

**Parameters**

  * `A`

Linear map, which must have two methods defined: multiplication by `AbstractVector` and transpose multiplication by `AbstractVector`.

  * `b` : `AbstractVector`

RHS of equation `Ax = b`.

  * `x0` : `AbstractVector`

Starting solution.

  * `maxiter` : Int, optional

Maximum number of iterations. Defaults to 10.

  * `tol` : Real, optional

Stopping criterion tolerance. Stops if the square root of the ration between the previous and current residuals is smaller than `tol`. Defaults to 1e-3.

  * `log` : Bool, optional

If `true`, returns also model and residuals at each iterations.


<a target='_blank' href='https://github.com/cako/KirchMig.jl/blob/55caf82a173be64bbe0144a7fd4edc6dbe2228a7/src/optimization.jl#L3-L33' class='documenter-source'>source</a><br>

