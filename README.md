# ytterbium

[Kilian W. Kluge](http://github.com/ionicsolutions)

`ytterbium` is a small Python package built on [QuTiP](http://github.com/qutip)
to simulate the full level-structure of <sup>171</sup>Yb<sup>+</sup>
and <sup>174</sup>Yb<sup>+</sup> ions.


## Requirements

To run `ytterbium`, the [General Requirements for running QuTiP](http://qutip.org/docs/3.1.0/installation.html)
need to be fulfilled and `qutip` needs to be installed.

While it is possible to run `qutip` on Python 2.7.x, `ytterbium` is developed on Python 3.6
and will not work correctly on Python 2.7.x without modifications.


## Getting Started

If you have never worked with `qutip` before, starting with the two-level model of the
S-P transition in 174Yb+ (`Yb174.twolevel`) is a good idea. Once you are familiar with
the way `qutip` works and how the models in `ytterbium` are structured, looking at some
of the example simulations provided is helpful.


## Literature

Throughout the package, we reference the following literature:

- S. Ejtemaee et al.: *Optimization of Yb+ fluorescence and hyperfine-qubit detection.*
  Phys. Rev. A **82**, 063419 (2010)
- H. Meyer: *A fibre-cavity based photonic interface for a single ion.*
  Dissertation, University of Cambridge (2014)
- S. Olmschenk et al.: *Manipulation and detection of a trapped Yb+ hyperfine qubit.*
  Phys. Rev. A **76**, 052314 (2007)


