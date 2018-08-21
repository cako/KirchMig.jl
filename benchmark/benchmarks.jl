using KirchMig

# Setup 2D
nR = nS = 64
dR = dS = 32
oR = oS = 0

t = 1e-2:1e-4:1e-1

dy = dx = dz = 50
x = 0:dx:oR+(nR-1)*dR
z = 0:dz:400
nx = length(x)
nz = length(z)
ny = 10
y = (0:ny-1)*dy

vel = 2700

src_x = oR:dR:oR+(nR-1)*dR
src_z = zeros(nR)
src_y = zeros(nR)
trav = eikonal_const_vel([src_x src_y src_z], x, y, z, vel)

pts = length(ARGS) == 0 ? "serial" : ARGS[1]
L = KirchMap(t, trav; parallel_threaded_serial=pts)
Random.seed!(1234)
u = rand(size(L, 2))
v = rand(size(L, 1))

v_hat = L*u;
tic()
v_hat = L*u;
fwd_t = toq()

u_hat = L'*v;
tic()
u_hat = L'*v;
adj_t = toq()

if pts == "serial"
    i = 1
elseif pts == "threaded"
    i = Threads.nthreads()
else
    i = nworkers()
end
@printf("    %02d   |%9.2f    |%9.2f \n", i, fwd_t, adj_t)

