using Documenter, KirchMig
 
makedocs(
    modules = [KirchMig],
    doctest = true,
    sitename = "KirchMig.jl",
)
 
deploydocs(
    deps   = Deps.pip("mkdocs", "mkdocs-material", "python-markdown-math"),
    repo = "github.com/cako/KirchMig.jl",
    julia  = "1.0",
    osname = "linux"
)
