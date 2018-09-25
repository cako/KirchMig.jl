
using Distributed
addprocs(Sys.CPU_THREADS)

import PyPlot; const plt = PyPlot
import DSP: conv
import Random: seed!, randperm
import Statistics: std

import KirchMig

dz, dx = 15, 15
x = 0:dx:1000; nx = length(x)
z = 0:dz:1000; nz = length(z)

rho = 1000ones(nz, nx)
rho[div(330,dx):end,:] .+= 1000
rho[div(670,dz):end,:] .-= 500

vel = 2000 .+ zero(rho)

imp = 1e-3rho.*vel;

refl = [zeros(nx)'
    (imp[2:end,:] .- imp[1:end-1,:])./(imp[2:end,:] .+ imp[1:end-1,:])]

refl[div(nz, 2)-1, div(nx, 2)] += 0.2
refl[div(nz, 2)+1, div(nx, 2)] += 0.2;

fig, ax = plt.subplots(1, 3, figsize=(10,2.5))

# Density
cax = ax[1][:imshow](rho, extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Blues", interpolation=nothing)
ax[1][:set](xlabel="Position [m]", ylabel="Depth [m]")
cbar = fig[:colorbar](cax, ax=ax[1])
cbar[:ax][:set](ylabel="Density [kg/m³]")

# Velocity
cax = ax[2][:imshow](vel, extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Blues", interpolation=nothing)
ax[2][:set](xlabel="Position [m]", yticks=[])
cbar = fig[:colorbar](cax, ax=ax[2])
cbar[:ax][:set](ylabel="Velocity [m/s]");

cax = ax[3][:imshow](imp, extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Blues", interpolation=nothing)
ax[3][:set](xlabel="Position [m]", yticks=[])
cbar = fig[:colorbar](cax, ax=ax[3])
cbar[:ax][:set](ylabel="Impedance [kPa·s/m]");
fig[:tight_layout]()

xran = x[end]-x[1]
mod = reshape(refl, nz, nx)
vmin, vmax = maximum(abs.(mod))*[-1,1]
fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](mod, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod[:,div(end,2)]./(4vmax) .+ xran./2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end-1], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="Reflectivity")
fig[:tight_layout]()

nr = 46

rec_z = zeros(nr)
rec_x = range(x[1], stop=x[end], length=nr)

ns = 10
seed!(12)
src_z = zeros(ns)
src_x = sort(rec_x[randperm(nr)][1:ns])

trav_r = KirchMig.eikonal_const_vel([rec_z rec_x], z, x, vel[1]);
trav_s = KirchMig.eikonal_const_vel([src_z src_x], z, x, vel[1]);

fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](trav_s[:,:,div(ns,3)], extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Paired",
    interpolation="bilinear", vmin=0, vmax=maximum(trav_s[:,:,div(ns,3)]))
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Traveltime [s]")
cbar[:ax][:invert_yaxis]() 
ax[:scatter](rec_x[1:3:end], rec_z[1:3:end], color="#1f77b4", marker="v", s=100, clip_on=false, zorder=100)
ax[:scatter](src_x, src_z, color="#d62728", alpha=0.4, marker="*", s=200, clip_on=false, zorder=100)
ax[:scatter](src_x[div(ns,3)], src_z[div(ns,3)], color="#d62728", marker="*", s=200, clip_on=false, zorder=100)
con = ax[:contour](trav_s[:,:,div(ns,3)], extent=[x[1], x[end], z[end], z[1]], origin="upper", colors="k")
plt.clabel(con, fontsize=9, inline=1, fmt="%1.2fs");
ax[:set](xlabel="Position [m]", ylabel="Depth [m]", xlim=(x[1], x[end]), ylim=(z[end-1], z[1]))
fig[:tight_layout]()

t = 0:0.008:1; nt = length(t)
G = KirchMig.KirchMap(t, trav_r, trav_s);

@time Gm = G*refl[:]; # refl is 67×67 a array, refl[:] is a 4489-element vector

ricker(t0, f) = @. (1 - 2pi^2 * f^2 * t0^2) * exp(-pi^2 * f^2 * t0^2)
rick_dtt = ricker(t .- t[div(nt,5)], 15);
@views rick_dtt[2:end-1] = (rick_dtt[1:end-2] - 2.0*rick_dtt[2:end-1] + rick_dtt[3:end])/(t[2] - t[1])^2;
rick_dtt /= maximum(abs.(rick_dtt));

W = KirchMig.ConvMap(rick_dtt, nr, ns, nt);

d = W*Gm;

seed!(1)
n = randn(size(d))
d += n*std(d)/std(n);

data = reshape(d, nr, ns, nt)

rows = 2
fig, ax = plt.subplots(rows, div(ns, rows), figsize=(10,4rows))
idx = reshape(1:rows*div(ns, rows), div(ns, rows), rows)'
for (i, axi) in enumerate(ax)
    csg = Array(data[:,idx[i],:]')
    axi[:imshow](csg, extent=[rec_x[1], rec_x[end], t[end], t[1]],
        vmin=-0.5maximum(abs.(csg)), vmax=0.5maximum(abs.(csg)), aspect="auto", cmap="gray", interpolation="none")
    axi[:scatter](src_x[idx[i]], t[1], color="#d62728", marker="*", s=200, clip_on=false, zorder=100)
    axi[:set](xlim=(rec_x[1],rec_x[end]), ylim=(t[end], t[1]), xlabel="Position [m]",
        xticks = range(minimum(rec_x), stop=maximum(rec_x), length=6))
    if idx[i] != 1 && idx[i] != div(ns,rows)+1
        axi[:set](yticks=[])
    else
        axi[:set](ylabel="Time [s]")
    end
end
fig[:tight_layout]()

L = W*G
@time m_mig = L'd;

xran = x[end]-x[1]
mod_mig = reshape(m_mig, nz, nx)
vmin, vmax = maximum(abs.(mod_mig))*[-1,1]
fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](mod_mig, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod_mig[:,div(end,2)]./(4vmax) .+ xran./2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="Migrated image")
fig[:tight_layout]()

import IterativeSolvers: cg
@time m_lsm_0, hist_0 = cg(L'L, m_mig, maxiter=20, log=true);

using Arpack
@time λ, ϕ = eigs(L'L; nev=1, maxiter=2); λ = λ[1];

import LinearMaps: IdentityMap

Id = IdentityMap(size(L'L, 1))
@time m_lsm_id, hist_id = cg(L'L + 0.01λ*Id, m_mig, maxiter=20, log=true);

Δ = KirchMig.LaplacianMap(nz, nx);

@time m_lsm_lap, hist_lap = cg(L'L + 0.01λ*Δ'Δ, m_mig, maxiter=20, log=true);

mod_lsm = reshape(m_lsm_0, nz, nx)
mod_lsm_id = reshape(m_lsm_id, nz, nx)
mod_lsm_lap = reshape(m_lsm_lap, nz, nx)

fig, ax = plt.subplots(2, 2, figsize=(6,6))
pclip=1
vmin, vmax = pclip*maximum(abs.(mod_lsm))*[-1,1]
ax[1][:imshow](mod_lsm, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
ax[1][:set](xlabel="Position [m]", ylabel="Depth [m]", title="LSM image: no reg.")
ax[1][:plot](250mod_lsm[:,50]./vmax .+ 500, z, color="#d62728");

vmin, vmax = pclip*maximum(abs.(mod_lsm_id))*[-1,1]
ax[2][:imshow](mod_lsm_id, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
ax[2][:set](xlabel="Position [m]",title="LSM image: min. norm", yticks=[])
ax[2][:plot](250mod_lsm_id[:,50]./vmax .+ 500, z, color="#d62728");

vmin, vmax = pclip*maximum(abs.(mod_lsm_lap))*[-1,1]
ax[3][:imshow](mod_lsm_lap, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
ax[3][:set](xlabel="Position [m]",title="LSM image: min. curv.", yticks=[])
ax[3][:plot](250mod_lsm_lap[:,50]./vmax .+ 500, z, color="#d62728");

vmin, vmax = pclip*maximum(abs.(mod_mig))*[-1,1]
ax[4][:imshow](mod_mig, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
ax[4][:set](xlabel="Position [m]",title="Migrated image", yticks=[])
ax[4][:plot](250mod_mig[:,50]./vmax .+ 500, z, color="#d62728");
fig[:tight_layout]()

fig, ax = plt.subplots(figsize=(6,4))
iter = length(hist_0.data[:resnorm])
ax[:semilogy](1:iter, hist_0.data[:resnorm], "k", label="No reg.", linewidth=2)

iter = length(hist_id.data[:resnorm])
ax[:semilogy](1:iter, hist_id.data[:resnorm], "#1f77b4", label="Min. norm", linewidth=2)

iter = length(hist_lap.data[:resnorm])
ax[:semilogy](1:iter, hist_lap.data[:resnorm], "#d62728", label="Min. curv.", linewidth=2)

ax[:set](xlabel="Iteration", ylabel="Residual", xlim=(1, iter))
plt.legend(loc=3)
fig[:tight_layout]()
