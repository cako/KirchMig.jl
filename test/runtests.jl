addprocs(Sys.CPU_CORES)
using KirchMig
using Base.Test
import LinearMaps: LinearMap

# Setup 2D
nR = nS = 64
dR = dS = 14
oR = oS = 0

t = 1e-2:1e-4:1e-1


dx = dz = 10
x = 0:dx:oR+(nR-1)*dR
z = 0:dz:400
nx = length(x)
nz = length(z)
vel = 2700

src_x = oR:dR:oR+(nR-1)*dR
src_z = zeros(nR)
trav = eikonal_const_vel([src_z src_x], z, x, vel)

X = [i for i in x, j in z]
Z = [j for i in x, j in z]
trav = zeros(nx, nz, nR)
for iR=1:nR
    rpos = oR + (iR-1)*dR
    trav[:,:,iR] = sqrt.((X - rpos).^2 + Z.^2)./vel
end

# Dot test 2D
for pts in ["parallel", "threaded", "serial"]
    pts == "parallel" ? println("Using ", nworkers(), " workers") : nothing
    pts == "threaded" ? println("Using ", Threads.nthreads(), " threads") : nothing
    pts == "serial" ? println("Using serial version") : nothing
    L = KirchMap(t, trav; parallel_threaded_serial=pts)
    srand(1234)
    u = rand(size(L, 2))
    v = rand(size(L, 1))
    print("Forward:")
    @time v_hat = L*u;
    print("Adjoint:")
    @time u_hat = L'*v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("2D Dot test: ⟨L'y, x⟩ = %.2f", utu))
    println(@sprintf("             ⟨L x, y⟩ = %.2f", vtv))
    @test (utu - vtv)/((utu+vtv)/2) <= 100*eps()
    println("")
end


println("")
# Setup 3D
t = convert(Vector{Float32}, t)

dy = 10
ny = 3
y = (0:ny-1)*dy

X = [i for i in x, j in z, k in y]
Z = [j for i in x, j in z, k in y]
Y = [k for i in x, j in z, k in y]
trav = zeros(nx, nz, ny, nR)
for iR=1:nR
    rpos = oR + (iR-1)*dR
    trav[:,:,:,iR] = sqrt.((X - rpos).^2 + Z.^2 + Y.^2)./vel
end
src_y = zeros(nR)
trav = eikonal_const_vel([src_x src_y src_z], x, y, z, vel)

# Dot test 3D
for pts in ["parallel", "threaded", "serial"]
    pts == "parallel" ? println("Using ", nworkers(), " workers") : nothing
    pts == "threaded" ? println("Using ", Threads.nthreads(), " threads") : nothing
    pts == "serial" ? println("Using serial version") : nothing
    L = KirchMap(t, trav; parallel_threaded_serial=pts)
    srand(1234)
    u = rand(size(L, 2))
    v = rand(size(L, 1))
    print("Forward:")
    @time v_hat = L*u;
    print("Adjoint:")
    @time u_hat = L'*v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("3D Dot test: ⟨L'y, x⟩ = %.2f", utu))
    println(@sprintf("             ⟨L x, y⟩ = %.2f", vtv))

    @test (utu - vtv)/((utu+vtv)/2) <= 100*eps()
    println("")
end

# cg
Id = LinearMap(x->2*x, x->2*x, 10, 10)
b = rand(size(Id, 1))
x = cg(Id'Id, Id'b)
@test 2x ≈ b
