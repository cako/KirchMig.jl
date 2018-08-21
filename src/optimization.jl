export cg
using LinearAlgebra

"""
`cg(A, b [, x0];  maxiter=10, tol=1e-3, log=false) -> x [, history_x, history_r]`

Solve the system `Ax = b` of a symmetric, positive definite linear map given by `A` using the conjugate gradients method.

Parameters
----------
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
"""
function cg(A, b::AbstractVector, x0::AbstractVector; maxiter::Int=10, tol::Real=1e-3, log::Bool=false)
    x = x0[:]
    r = b - A * x
    p = r
    rsold = dot(r, r)
    r1 = rsold
    if log
        history_x = [x]
        history_r = [rsold]
    end

    for i = 1:minimum([length(b), maxiter])≈
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = dot(r, r)
        p = r + (rsnew / rsold) * p
        if log
            push!(history_x, x)
            push!(history_r, rsnew)
        end
        if sqrt(rsnew/rsold) < tol || p ≈ zero(p)
            break
        end
        rsold = rsnew
    end
    if log
        return x, history_x, history_r
    else
        return x
    end
end
cg(A, b; kwargs...) = cg(A, b, zeros(length(b)); kwargs...)

