<p align="center">
<img src="https://s3.eu-west-2.amazonaws.com/cdacosta-londonbucket/github/forward-diagram.png?raw=true" width=400px alt="Forward map diagram"/>
</p>

# KirchMig.jl
Kirchhoff migration is a method used in geophysics, nondestructive testing and other areas to obtain high-frequency representations of the impedance contrasts of a target medium.
In geophysics, it is also known as the *diffraction stack*, and in nondestructive testing as *total focusing method* (TFM).

What this package is *not*: This package is not focused in providing extensive traveltime computation methods, such as eikonal solvers, ray-tracers, etc.
Many of these can be already found in software suits like [Madagascar](http://ahay.org).
I recommend the [excellent `sfeikonal` solver](http://ahay.org/blog/2014/06/11/program-of-the-month-sfeikonal/). In the future, I might incorporate `sfeikonal` and other Madagascar tools within the library.
This package is also not, a priori, designed to be an advanced, true-amplitude migration tool with many bells and whistles, although it might mutate into that in the future. Of course I am more than happy to incorporate more advanced Kirchhoff migration (and demigration).

What this package is: At the moment I am interested in providing a simple-to-use and simple-to-grasp Kirchhoff inversion toolset.
I am interested in writing this with a linear mapping framework taking special care of providing forward and adjoint maps which respect the [dot product test](http://sepwww.stanford.edu/sep/prof/pvi/conj/paper_html/node9.html).
In this way, the linear maps provided can be used in inversion methods, for example in so-called *least-squares migration*.

## Get started

Once [Julia 0.6.2](https://julialang.org/downloads/) is installed, open the REPL and run

```bash
Pkg.clone("https://github.com/cako/KirchMig.jl")
```

A few examples are provided as scripts and Jupyter notebooks [here](https://github.com/cako/KirchMig.jl/tree/master/notebooks).

<!--```@contents-->
<!--Pages = [-->
    <!--"modules/map.md",-->
    <!--"modules/eikonal.md",-->
    <!--]-->
<!--Depth = 2-->
<!--```-->
