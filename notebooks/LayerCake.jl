
#addprocs(Sys.CPU_THREADS)
import PyPlot; const plt = PyPlot
import KirchMig; const km = KirchMig;

dz, dx = 15, 15
x = 0:dx:1000; nx = length(x)
z = 0:dz:1000; nz = length(z)

rho = 1000ones(nz, nx)
rho[div(330,dx):end,:] += 1000
rho[div(670,dz):end,:] -= 500

vel = similar(rho); vel .= 2000;

blk = 1e-9rho.*vel.^2;

fig, ax = plt.subplots(1, 3, figsize=(12,3))

# Density
cax = ax[1][:imshow](rho, extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Blues", interpolation="none")
ax[1][:set](xlabel="Position [m]", ylabel="Depth [m]")
cbar = fig[:colorbar](cax, ax=ax[1])
cbar[:ax][:set](ylabel="Density [kg/m³]")

# Velocity
cax = ax[2][:imshow](vel, extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Blues", interpolation="none")
ax[2][:set](xlabel="Position [m]", yticks=[])
cbar = fig[:colorbar](cax, ax=ax[2])
cbar[:ax][:set](ylabel="Velocity [m/s]");

cax = ax[3][:imshow](blk, extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Blues", interpolation="none")
ax[3][:set](xlabel="Position [m]", yticks=[])
cbar = fig[:colorbar](cax, ax=ax[3])
cbar[:ax][:set](ylabel="Bulk modulus [GPa]");
fig[:tight_layout]()

refl = [zeros(nx)'
    (blk[2:end,:] - blk[1:end-1,:])./(blk[2:end,:]+blk[1:end-1,:])];

ricker(to, f) = (1 - 2pi^2 * f^2 * to.^2) .* exp.(-pi^2 * f^2 * to.^2)
rick = ricker(z-z[div(nz-1,2)+1], 0.01); 

mod_bl = hcat( (conv(refl[:,ix], rick)[div(nz-1,2)+1:div(3nz-1,2)] for ix in 1:nx)... );

fig, ax = plt.subplots(1, 2, figsize=(8,3))
vmin, vmax = maximum(abs.(refl))*[-1,1]
cax = ax[1][:imshow](refl, extent=[x[1], x[end], z[end], z[1]],
    vmin=-maximum(abs.(refl)), vmax=maximum(abs.(refl)), cmap="gray", interpolation="none")
ax[1][:set](xlabel="Position [m]", ylabel="Depth [m]", title="Reflectivity")
cbar = fig[:colorbar](cax, ax=ax[1])
ax[1][:plot](250refl[:,50]/vmax+500, z, color="#d62728")

vmin, vmax = maximum(abs.(mod_bl))*[-1,1]
cax = ax[2][:imshow](mod_bl, extent=[x[1], x[end], z[end], z[1]], vmin=vmin, vmax=vmax, cmap="gray",
    interpolation="none")
ax[2][:set](xlabel="Position [m]", title="Band-limited reflectivity", yticks=[])
cbar = fig[:colorbar](cax, ax=ax[2])
ax[2][:plot](250mod_bl[:,50]/vmax+500, z, color="#d62728");

ns = 50

src_z = zeros(ns)
src_x = 0:20:(ns-1)*20

trav = km.eikonal_const_vel([src_z src_x], z, x, vel[1]);

fig, ax = plt.subplots(figsize=(5,3))
cax = ax[:imshow](trav[:,:,div(ns,3)], extent=[x[1], x[end], z[end], z[1]], aspect="equal", cmap="Paired",
    vmin=0, vmax=maximum(trav[:,:,div(ns,3)]))
ax[:set](xlabel="Position [m]", ylabel="Depth [m]")
cbar = fig[:colorbar](cax, ax=ax)
ax[:scatter](src_x[1:3:end], src_z[1:3:end], color="#1f77b4", marker="v", s=100, clip_on=false, zorder=100)
ax[:scatter](src_x[div(ns,3)], src_z[div(ns,3)], color="#d62728", marker="*", s=200, clip_on=false, zorder=100)
con = ax[:contour](trav[:,:,div(ns,3)], extent=[x[1], x[end], z[end], z[1]], origin="upper", colors="k")
plt.clabel(con, fontsize=9, inline=1, fmt="%1.2fs");
ax[:set](ylabel="Traveltime [s]", xlim=(x[1], x[end]), ylim=(z[end], z[1]));

t = 0:0.008:1; nt = length(t)
L = km.KirchMap(t, trav)

@time d = L*view(mod_bl, :); # mod_bl is 67x67 array, L takes 4489 vector

data = reshape(d, ns, ns, nt)

rows = 10
fig, ax = plt.subplots(rows, div(ns, rows), figsize=(12,4rows))
idx = reshape(1:rows*div(ns, rows), div(ns, rows), rows)'
for (i, axi) in enumerate(ax)
    csg = data[:,idx[i],:]'
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

@time m_mig = L'd;

xran = x[end]-x[1]
mod_mig = reshape(m_mig, nz, nx)
vmin, vmax = maximum(abs.(mod_mig))*[-1,1]
fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](mod_mig, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation="none")
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod_mig[:,div(end,2)]/(4vmax) + xran/2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="Migrated image")
fig[:tight_layout]()

@time m_lsm, _, hist_r = km.cg(L'L, m_mig, maxiter=20, log=true);

mod_lsm = reshape(m_lsm, nz, nx)
vmin, vmax = maximum(abs.(mod_lsm))*[-1,1]
fig, ax = plt.subplots(figsize=(6,4))
cax = ax[:imshow](mod_lsm, extent=[x[1], x[end], z[end], z[1]],
    vmin=vmin, vmax=vmax, aspect="equal", cmap="gray", interpolation="none")
cbar = fig[:colorbar](cax, ax=ax)
cbar[:ax][:set](ylabel="Amplitude");
ax[:plot](xran*mod_lsm[:,div(end,2)]/(4vmax) + xran/2, z, color="#d62728");
ax[:set](xlim=(x[1], x[end]), ylim=(z[end], z[1]), xlabel="Position [m]", ylabel="Depth [m]",
    title="Least-squares image")
fig[:tight_layout]()

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[1][:plot](1:length(hist_r), hist_r, "k")
ax[1][:set](xlim=(1, length(hist_r)), xlabel="Iteration", ylabel="Residual");
ax[2][:semilogy](1:length(hist_r), hist_r, "k")
ax[2][:set](xlim=(1, length(hist_r)), xlabel="Iteration", ylabel="Residual");
fig[:suptitle]("CG residual")
fig[:tight_layout]()

err_mig = mod_bl/maximum(mod_bl) - mod_mig/maximum(mod_mig)
err_lsm = mod_bl/maximum(mod_bl) - mod_lsm/maximum(mod_lsm)

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
