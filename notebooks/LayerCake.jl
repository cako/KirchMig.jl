
#using Distributed
#addprocs(Sys.CPU_THREADS)

import PyPlot; const plt = PyPlot

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

ns = 50

src_z = zeros(ns)
src_x = 0:20:(ns-1)*20

trav = KirchMig.eikonal_const_vel([src_z src_x], z, x, vel[1]);

fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](trav[:,:,div(ns,3)], extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Paired",
    interpolation="bilinear", vmin=0, vmax=maximum(trav[:,:,div(ns,3)]))
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Traveltime [s]")
cbar[:ax][:invert_yaxis]() 
ax[:scatter](src_x[1:4:end], src_z[1:4:end], color="#1f77b4", marker="v", s=100, clip_on=false, zorder=100)
ax[:scatter](src_x[3:4:end], src_z[3:4:end], color="#d62728", alpha=0.4, marker="*", s=200, clip_on=false, zorder=100)
ax[:scatter](src_x[div(ns,3)], src_z[div(ns,3)], color="#d62728", marker="*", s=200, clip_on=false, zorder=100)
con = ax[:contour](trav[:,:,div(ns,3)], extent=[x[1], x[end], z[end], z[1]], origin="upper", colors="k")
plt.clabel(con, fontsize=9, inline=1, fmt="%1.2fs");
ax[:set](xlabel="Position [m]", ylabel="Depth [m]", xlim=(x[1], x[end]), ylim=(z[end-1], z[1]))
fig[:tight_layout]()

t = 0:0.008:1; nt = length(t)
G = KirchMig.KirchMap(t, trav);

@time Gm = G*refl[:]; # refl is 67×67 a array, refl[:] is a 4489-element vector

ricker(t0, f) = @. (1 - 2pi^2 * f^2 * t0^2) * exp(-pi^2 * f^2 * t0^2)
rick_dtt = ricker(t .- t[div(nt,5)], 15);
@views rick_dtt[2:end-1] = (rick_dtt[1:end-2] - 2.0*rick_dtt[2:end-1] + rick_dtt[3:end])/(t[2] - t[1])^2;
rick_dtt /= maximum(abs.(rick_dtt));

W = KirchMig.ConvMap(rick_dtt, ns, ns, nt);

d = W*Gm;

data = reshape(d, ns, ns, nt)

rows = 10
fig, ax = plt.subplots(rows, div(ns, rows), figsize=(12,4rows))
idx = reshape(1:rows*div(ns, rows), div(ns, rows), rows)'
for (i, axi) in enumerate(ax)
    csg = Array(data[:,idx[i],:]')
    cax = axi[:imshow](csg, extent=[src_x[1], src_x[end], t[end], t[1]],
        vmin=-0.5maximum(abs.(csg)), vmax=0.5maximum(abs.(csg)), aspect="auto", cmap="gray", interpolation="none")
    axi[:scatter](src_x[idx[i]], t[1], color="#d62728", marker="*", s=200, clip_on=false, zorder=100)
    axi[:set](xlim=(src_x[1],src_x[end]), ylim=(t[end], t[1]), xlabel="Position [m]",
        xticks = range(minimum(src_x), stop=maximum(src_x), length=5))
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

@time m_lsm, hist_m, hist_r = KirchMig.cg(L'L, m_mig, maxiter=15, log=true);

xran = x[end]-x[1]
mod_lsm = reshape(m_lsm, nz, nx)
vmin, vmax = maximum(abs.(mod_lsm))*[-1,1]
fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](mod_lsm, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation=nothing)
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod_lsm[:,div(end,2)]./(4vmax) .+ xran./2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="Migrated image")
fig[:tight_layout]()

import LinearAlgebra: norm
model_residuals = [ norm(m - refl[:])/norm(refl) for m in hist_m];

its = 0:(length(hist_r)-1)

fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex=true)
ax[1][:plot](its, hist_r, "k")
ax[1][:set](ylabel="Data residual")
ax[3][:semilogy](its, hist_r, "k")
ax[3][:set](ylabel="Data residual")
ax[2][:plot](its, model_residuals, "k")
ax[2][:set](xlabel="Iteration", ylabel="Model residual")
ax[4][:semilogy](its, model_residuals, "k")
ax[4][:set](xlim=(its[1], its[end]), xlabel="Iteration", ylabel="Model residual")
ax[4][:xaxis][:set_major_locator](plt.matplotlib[:ticker][:MaxNLocator](integer=true))
fig[:suptitle]("Residuals", y=1.)
fig[:tight_layout]();

err_mig = refl/maximum(refl) - mod_mig/maximum(mod_mig)
err_lsm = refl/maximum(refl) - mod_lsm/maximum(mod_lsm)

fig, ax = plt.subplots(1, 2, figsize=(12,4))
pclip=.5
cax = ax[1][:imshow](err_mig, extent=[x[1], x[end], z[end], z[1]],
    vmin=-pclip, vmax=pclip, aspect="equal", cmap="gray", interpolation="none")
ax[1][:set](xlabel="Position [m]", ylabel="Depth [m]", title="Normalized migration error")
cbar = fig[:colorbar](cax, ax=ax[1])
cbar[:ax][:set](ylabel="Amplitude");

vmin, vmax = pclip*maximum(abs.(mod_lsm))*[-1,1]
cax = ax[2][:imshow](err_lsm, extent=[x[1], x[end], z[end], z[1]],
    vmin=-pclip, vmax=pclip, aspect="equal", cmap="gray", interpolation="none")
ax[2][:set](xlabel="Position [m]",title="Normalized LSM error", yticks=[]);
cbar = fig[:colorbar](cax, ax=ax[2])
cbar[:ax][:set](ylabel="Amplitude");

fig, ax = plt.subplots(2, 1, figsize=(8,4))
ax[1][:plot](mod_mig[:,div(end,2)], "k", linewidth=2)
ax[1][:set](ylabel="Mig. Amplitude", ylim=[-1.1,1.1]*maximum(abs.(mod_mig[:,div(end,2)])))
ax2 = ax[1][:twinx]()
ax2[:plot](mod_lsm[:,div(end,2)], "--", color="#1f77b4", linewidth=2)
ax2[:set](ylim=[-1.1,1.1]*maximum(abs.(mod_lsm[:,div(end,2)])))
ax2[:set_ylabel]("LSM Amplitude", color="#1f77b4")

ax[2][:plot](mod_mig[:,1], "k", linewidth=2)
ax[2][:set](ylabel="Mig. Amplitude", ylim=[-1.1,1.1]*maximum(abs.(mod_mig[:,1])))
ax2 = ax[2][:twinx]()
ax2[:plot](mod_lsm[:,1], "--", color="#1f77b4", linewidth=2)
ax2[:set](ylim=[-1.1,1.1]*maximum(abs.(mod_lsm[:,1])))
ax2[:set_ylabel]("LSM Amplitude", color="#1f77b4");
