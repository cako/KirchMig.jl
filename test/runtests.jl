using Test
using Printf
using Distributed
using Random
using LinearAlgebra
import LinearMaps: LinearMap, _ismutating

print("Checking cores... ")
Sys.CPU_THREADS > 1 && addprocs(2)
println("done")
print("Loading KirchMig... ")
using KirchMig
println("done")

# Setup 2D
print("2D setup... ")
nR = nS = 64
dR = dS = 14
oR = oS = 0

ot = 1e-2
dt = 1e-4
nt = 901
t = ot:dt:ot+dt*(nt-1)

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

try
   L = KirchMap(t, trav, trav[1:end-1, :, :])
catch err
    @test isa(err, DimensionMismatch)
end

# Dot test 2D
println("2D Kirchhoff...")
for pts in ["parallel", "threaded", "serial"]
    pts == "parallel" ? println("  Using ", nworkers(), " workers") : nothing
    pts == "threaded" ? println("  Using ", Threads.nthreads(),
                                " thread" * (Threads.nthreads() > 1 ? "s" : "") ) : nothing
    pts == "serial" ? println("  Using serial version") : nothing
    L = KirchMap(t, trav; parallel_threaded_serial=pts)
    Random.seed!(1234)
    u = rand(size(L, 2))
    v = rand(size(L, 1))
    print("    Forward:")
    @time v_hat = L*u;
    print("    Adjoint:")
    @time u_hat = L'*v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("    dot test: ⟨L'v, u⟩ = %.2f", utu))
    println(@sprintf("              ⟨L u, v⟩ = %.2f", vtv))
    @test abs(utu - vtv)/((utu+vtv)/2) <= 100*eps()

    issymmetric(L) == false
    isposdef(L) == false
    _ismutating(L.f)
    println("done")
    println("")
end

# Setup 3D
print("3D setup... ")

dy = 10
ny = 3
y = (0:ny-1)*dy
println("done")

print("3D traveltimes... ")
src_y = zeros(nR)
trav = eikonal_const_vel([src_x src_y src_z], x, y, z, vel)
println("done")


# Dot test 3D
println("3D Kirchhoff...")
for pts in ["parallel", "threaded", "serial"]
    pts == "parallel" ? println("  Using ", nworkers(), " workers") : nothing
    pts == "threaded" ? println("  Using ", Threads.nthreads(), " threads") : nothing
    pts == "serial" ? println("  Using serial version") : nothing
    L = KirchMap(t, trav; parallel_threaded_serial=pts)
    Random.seed!(1234)
    u = rand(size(L, 2))
    v = rand(size(L, 1))
    print("    Forward:")
    @time v_hat = L*u;
    print("    Adjoint:")
    @time u_hat = L'*v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("    dot test: ⟨L'v, u⟩ = %.2f", utu))
    println(@sprintf("              ⟨L u, v⟩ = %.2f", vtv))

    @test abs(utu - vtv)/((utu+vtv)/2) <= 100*eps()
    println("")
end

# cg
println("Conjugate gradients... ")
M, N = 10, 5
function fwd(x)
    y = zeros(eltype(x), M)
    y[1:N] = 2x .+ x[end:-1:1]
    return y
end
function adj(x)
    return 2x[1:N] .+ x[1:N][end:-1:1]
end
Id = LinearMap(fwd, adj, M, N)
u = rand(size(Id, 2))
v = rand(size(Id, 1))
v_hat = Id*u;
u_hat = Id'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨corr v, u⟩ = %.5f", utu))
println(@sprintf("            ⟨conv u, v⟩ = %.5f", vtv))
@test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()

b = rand(size(Id, 1))
x, hist_x, hist_r = cg(Id'Id, Id'b, log=true)
@test x ≈ inv(Matrix(Id'Id))*(Id'b)
println("done")
println("")

# Regularization
println("2D ConvMap... ")
ricker(t0, f) = @. (1 - 2pi^2 * f^2 * t0^2) * exp(-pi^2 * f^2 * t0^2)
rick_dt = ricker(t .- t[div(nt,3)], 15);
rick_dt[2:end-1] = @. (rick_dt[1:end-2] - rick_dt[3:end])/2(t[2] - t[1])

W = ConvMap(rick_dt, nS, nt);
Random.seed!(1234)
u = rand(size(W, 2))
v = rand(size(W, 1))
v_hat = W*u;
u_hat = W'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨corr v, u⟩ = %.5f", utu))
println(@sprintf("            ⟨conv u, v⟩ = %.5f", vtv))
@test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
println("")

println("3D ConvMap... ")
W = ConvMap(rick_dt, nR, nS, nt);
Random.seed!(1234)
u = rand(size(W, 2))
v = rand(size(W, 1))
v_hat = W*u;
u_hat = W'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨corr v, u⟩ = %.5f", utu))
println(@sprintf("            ⟨conv u, v⟩ = %.5f", vtv))
@test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
println("")

println("2D Laplacian... ")
Δ = KirchMig.LaplacianMap(nz, nx)

Random.seed!(1234)
u = rand(size(Δ, 2))
v = rand(size(Δ, 1))
v_hat = Δ*u;
u_hat = Δ'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨Δ'v, u⟩ = %.5f", utu))
println(@sprintf("            ⟨Δ u, v⟩ = %.5f", vtv))
@test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
println("")

println("3D Laplacian... ")
Δ = KirchMig.LaplacianMap(nz, nx, ny)

Random.seed!(1234)
u = rand(size(Δ, 2))
v = rand(size(Δ, 1))
v_hat = Δ*u;
u_hat = Δ'v;
utu =  dot(u_hat, u)
vtv =  dot(v_hat, v)
println(@sprintf("  Dot test: ⟨Δ'v, u⟩ = %.5f", utu))
println(@sprintf("            ⟨Δ u, v⟩ = %.5f", vtv))
@test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
println("")

for order in [2, 4, 8]
    println("3D DiffZ (order $order)... ")
    δz = KirchMig.DiffZMap(nz, nx, ny; order=order)

    Random.seed!(1234)
    u = rand(size(δz, 2))
    v = rand(size(δz, 1))
    v_hat = δz*u;
    u_hat = δz'v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("  Dot test: ⟨-δz v, u⟩ = %.5f", utu))
    println(@sprintf("            ⟨ δz u, v⟩ = %.5f", vtv))
    @test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
    println("")

    println("2D DiffX (order $order)... ")
    δx = KirchMig.DiffXMap(nz, nx; order=order)

    Random.seed!(1234)
    u = rand(size(δx, 2))
    v = rand(size(δx, 1))
    v_hat = δx*u;
    u_hat = δx'v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("  Dot test: ⟨-δx v, u⟩ = %.5f", utu))
    println(@sprintf("            ⟨ δx u, v⟩ = %.5f", vtv))
    @test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
    println("")

    println("2D GradDiv (order $order)... ")
    GD = KirchMig.GradDivMap(nz, nx; order=order)

    Random.seed!(1234)
    u = rand(size(GD, 2))
    v = rand(size(GD, 1))
    v_hat = GD*u;
    u_hat = GD'v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("  Dot test: ⟨-div v, u⟩ = %.5f", utu))
    println(@sprintf("            ⟨grad u, v⟩ = %.5f", vtv))
    @test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
    println("")

    println("3D GradDiv (order $order)... ")
    GD = KirchMig.GradDivMap(nz, nx, ny; order=order)

    Random.seed!(1234)
    u = rand(size(GD, 2))
    v = rand(size(GD, 1))
    v_hat = GD*u;
    u_hat = GD'v;
    utu =  dot(u_hat, u)
    vtv =  dot(v_hat, v)
    println(@sprintf("  Dot test: ⟨-div v, u⟩ = %.5f", utu))
    println(@sprintf("            ⟨grad u, v⟩ = %.5f", vtv))
    @test abs(utu - vtv)/((utu+vtv)/2) <= 100eps()
    println("")
end
