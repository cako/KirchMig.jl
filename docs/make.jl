using Documenter, KirchMig
 
makedocs(
    modules = [KirchMig],
    doctest = true,
)
 
deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/cako/KirchMig.jl",
    julia  = "0.7",
    osname = "linux"
)
