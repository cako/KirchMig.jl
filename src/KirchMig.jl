__precompile__(true)
"""
Main module for `KirchMig.jl` -- a Julia package for Kirchhoff migration.
"""
module KirchMig

include("migration.jl")
include("map.jl")
include("eikonal.jl")
include("optimization.jl")
include("regularization.jl")

end
