print("Checking cores... ")
Sys.CPU_CORES > 1 && addprocs(2)
println("done")
print("Loading KirchMig... ")
using KirchMig
println("done")
print("Loading Base.Test... ")
using Base.Test
println("done")
print("Loading LinearMaps... ")
import LinearMaps: LinearMap
println("done")

# Setup 2D
print("2D setup... ")
nR = nS = 64
dR = dS = 14
oR = oS = 0

t = 1e-2:1e-4:1e-1


dx = dz = 100
x = 0:dx:oR+(nR-1)*dR
z = 0:dz:400
nx = length(x)
nz = length(z)
vel = 2700
println("done")

print("2D traveltimes... ")
src_x = oR:dR:oR+(nR-1)*dR
src_z = zeros(nR)
trav = eikonal_const_vel([src_z src_x], z, x, vel)
println("done")

# Dot test 2D
for pts in ["parallel", "threaded", "serial"]
    pts == "parallel" ? println("  Using ", nworkers(), " workers") : nothing
    pts == "threaded" ? println("  Using ", Threads.nthreads(), " threads") : nothing
    pts == "serial" ? println("  Using serial version") : nothing
    L = KirchMap(t, trav; parallel_threaded_serial=pts)
    srand(1234)
    u = rand(size(L, 2))
    v = rand(size(L, 1))
    print("    Forward:")
    @time v_hat = L*u;
    print("    Adjoint:")
    @time u_hat = L'*v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("    dot test: ⟨L'y, x⟩ = %.2f", utu))
    println(@sprintf("              ⟨L x, y⟩ = %.2f", vtv))
    @test (utu - vtv)/((utu+vtv)/2) <= 100*eps()
    println("")
end


println("")
# Setup 3D
print("3D setup... ")
t = convert(Vector{Float32}, t)

dy = 10
ny = 3
y = (0:ny-1)*dy
println("done")

print("3D traveltimes... ")
src_y = zeros(nR)
trav = eikonal_const_vel([src_x src_y src_z], x, y, z, vel)
println("done")

# Dot test 3D
for pts in ["parallel", "threaded", "serial"]
    pts == "parallel" ? println("  Using ", nworkers(), " workers") : nothing
    pts == "threaded" ? println("  Using ", Threads.nthreads(), " threads") : nothing
    pts == "serial" ? println("  Using serial version") : nothing
    L = KirchMap(t, trav; parallel_threaded_serial=pts)
    srand(1234)
    u = rand(size(L, 2))
    v = rand(size(L, 1))
    print("    Forward:")
    @time v_hat = L*u;
    print("    Adjoint:")
    @time u_hat = L'*v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("    dot test: ⟨L'y, x⟩ = %.2f", utu))
    println(@sprintf("              ⟨L x, y⟩ = %.2f", vtv))

    @test (utu - vtv)/((utu+vtv)/2) <= 100*eps()
    println("")
end

# cg
print("Conjugate gradients... ")
Id = LinearMap(x->2*x, x->2*x, 10, 10)
b = rand(size(Id, 1))
x, hist_x, hist_r = cg(Id'Id, Id'b, log=true)
@test 2x ≈ b
println("done")

# Regularization
println("2D Laplacian... ")
Δ = KirchMig.LaplacianMap(nz, nx)

srand(1234)
u = rand(size(Δ, 2))
v = rand(size(Δ, 1))
v_hat = Δ*u;
u_hat = Δ'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨Δ'y, x⟩ = %.5f", utu))
println(@sprintf("            ⟨Δ x, y⟩ = %.5f", vtv))
@test (utu - vtv)/((utu+vtv)/2) <= 0*eps()
println("done")

println("3D Laplacian... ")
Δ = KirchMig.LaplacianMap(nz, nx, ny)

srand(1234)
u = rand(size(Δ, 2))
v = rand(size(Δ, 1))
v_hat = Δ*u;
u_hat = Δ'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨Δ'y, x⟩ = %.5f", utu))
println(@sprintf("            ⟨Δ x, y⟩ = %.5f", vtv))
@test (utu - vtv)/((utu+vtv)/2) <= 0*eps()
println("done")

println("3D DiffZ... ")
Δ = KirchMig.DiffZMap(nz, nx, ny)

srand(1234)
u = rand(size(Δ, 2))
v = rand(size(Δ, 1))
v_hat = Δ*u;
u_hat = Δ'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨Δ'y, x⟩ = %.5f", utu))
println(@sprintf("            ⟨Δ x, y⟩ = %.5f", vtv))
@test (utu - vtv)/((utu+vtv)/2) <= eps()
println("done")
