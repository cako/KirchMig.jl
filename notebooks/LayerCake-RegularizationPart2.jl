
#addprocs(Sys.CPU_CORES)
import PyPlot; const plt = PyPlot
import KirchMig

dz, dx = 15, 15
x = 0:dx:1000; nx = length(x)
z = 0:dz:1000; nz = length(z)

rho = 1000ones(nz, nx)
rho[div(330,dx):end,:] += 1000
rho[div(670,dz):end,:] -= 500

vel = similar(rho); vel .= 2000;

blk = 1e-9rho.*vel.^2;

refl = [zeros(nx)'
    (blk[2:end,:] - blk[1:end-1,:])./(blk[2:end,:]+blk[1:end-1,:])];

ricker(to, f) = (1 - 2pi^2 * f^2 * to.^2) .* exp.(-pi^2 * f^2 * to.^2)
rick = ricker(z-z[div(nz-1,2)+1], 0.01); 

mod_bl = hcat( (conv(refl[:,ix], rick)[div(nz-1,2)+1:div(3nz-1,2)] for ix in 1:nx)... );

nr = 46

rec_z = zeros(nr)
rec_x = range(x[1], stop=x[end], length=nr)

ns = 10
Random.seed!(12)
src_z = zeros(ns)
src_x = sort(rec_x[randperm(nr)][1:ns])


trav_r = KirchMig.eikonal_const_vel([rec_z rec_x], z, x, vel[1]);
trav_s = KirchMig.eikonal_const_vel([src_z src_x], z, x, vel[1]);

t = 0:0.008:1; nt = length(t)
L = KirchMig.KirchMap(t, trav_r, trav_s)

@time d = L*mod_bl[:];

Random.seed!(1)
n = randn(size(d))
d += n*std(d)/std(n);

@time m_mig = L'd;

J0(m) = norm(L*m - d).^2

function ∇J0!(storage, m)
    storage[1:end] = 2L'L*m - 2m_mig
end

Random.seed!(123);

nL = size(L, 2)
u = randn(nL)
v = randn(nL)
h = 0.001*maximum(abs.(u))

storage = zero(u)
∇J0!(storage, u)

g1 = (J0(u + h*v) - J0(u - h*v))/2h
g2 = dot(storage, v)
err = 100abs(g1 - g2)/((g1+g2)/2)
println("$(@sprintf("%.2f", err))% error")

import Optim

@time res = Optim.optimize(J0, ∇J0!, zeros(size(L, 2)),
                           Optim.ConjugateGradient(),
                           Optim.Options(iterations = 20, show_trace=true))

xran = x[end]-x[1]
mod_lsm_0 = reshape(res.minimizer, nz, nx)
pclip=0.5

fig, ax = plt.subplots(figsize=(6,4))
vmin, vmax = pclip*maximum(abs.(mod_lsm_0))*[-1,1]
cax = ax[:imshow](mod_lsm_0, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray")
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod_lsm_0[:,div(end,2)]/(4vmax) + xran/2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="LSM image: no reg.")
fig[:tight_layout]()

@time λ, ϕ = eigs(L'L; nev=1, maxiter=2); λ = λ[1]

function TV(x)
    grad = KirchMig.gradient(x, nz, nx) # Returns a nz × nx × 2 array in 2D
    norm_grad = sqrt.(sum(grad.^2, 3))
    e = maximum([100eps(), 0.01maximum(norm_grad)])
    return 0.005*λ*sum(sqrt.(norm_grad.^2 + e^2))
end
function ∇TV!(storage, x)
    grad = KirchMig.gradient(x, nz, nx)
    norm_grad = sqrt.(sum(grad.^2, 3))
    e = maximum([100eps(), 0.01maximum(norm_grad)])
    grad ./= sqrt.(norm_grad.^2 + e^2)
    
    storage[1:end] = -0.005*λ*KirchMig.divergence(grad[:], nz, nx)[:]
end

Random.seed!(123);

nL = size(L, 2)
u = randn(nL)
v = randn(nL)
h = 0.001*maximum(abs.(u))

storage = zero(u)
∇TV!(storage, u)

g1 = (TV(u + h*v) - TV(u - h*v))/2h
g2 = dot(storage, v)
err = 100abs(g1 - g2)/((g1+g2)/2)
println("$(@sprintf("%.2f", err))% error")

JTV(x) = J0(x) + TV(x)

function ∇JTV!(storage, x)
    ∇J0!(storage, x)
    s = storage[:]
    ∇TV!(storage, x)
    storage[1:end] = s + storage[1:end]
end

# Check gradient
Random.seed!(123);

nL = size(L, 2)
u = randn(nL)
v = randn(nL)
h = 0.001*maximum(abs.(u))

storage = zero(u)
∇JTV!(storage, u)

g1 = (JTV(u + h*v) - JTV(u - h*v))/2h
g2 = dot(storage, v)
err = 100abs(g1 - g2)/((g1+g2)/2)
println("$(@sprintf("%.2f", err))% error")

@time res_tv = Optim.optimize(JTV, ∇JTV!, zeros(size(L, 2)),
                              Optim.ConjugateGradient(),
                              Optim.Options(iterations = 20, show_trace=true))

mod_lsm_tv = reshape(res_tv.minimizer, nz, nx)
pclip=0.5

fig, ax = plt.subplots(figsize=(6,4))
vmin, vmax = pclip*maximum(abs.(mod_lsm_tv))*[-1,1]
cax = ax[:imshow](mod_lsm_tv, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray")
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod_lsm_tv[:,div(end,2)]/(4vmax) + xran/2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="LSM image: TV reg.")
fig[:tight_layout]()
