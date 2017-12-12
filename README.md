# ytterbium

[Kilian W. Kluge](http://github.com/ionicsolutions)

`ytterbium` is a Python package built on [QuTiP](http://github.com/qutip)
to simulate optical transitions in <sup>171</sup>Yb<sup>+</sup>
and <sup>174</sup>Yb<sup>+</sup> ions by solving Lindblad form master equations.

In addition to various models of commonly encountered experimental situations,
`ytterbium` contains a small library for parallelized computations which
provides a significant speedup when a system needs to be solved for many
different parameter values (`parallelize`).

[![build-status](https://api.travis-ci.org/ionicsolutions/ytterbium.svg?branch=master)](http://travis-ci.org/ionicsolutions/ytterbium)

## Requirements

To run `ytterbium`, the [General Requirements for running QuTiP](http://qutip.org/docs/3.1.0/installation.html)
need to be fulfilled and `qutip` needs to be installed.

While it is possible to run `qutip` on Python 2.7.x, `ytterbium` is developed
on Python 3.6 and will not work correctly on Python <3.6 without modifications.


## Getting Started

If you have never worked with `qutip` before, starting with the two-level model
of the S-P transition in <sup>174</sup>Yb<sup>+</sup> (`Yb174.twolevel`) and
the corresponding tests (`test.Yb174.test_twolevel`) is recommended.

Once you are familiar with the way `qutip` works and how the models in
`ytterbium` are structured, looking at some of the example simulations provided
is helpful to set up your own calculations.


## Literature

Throughout the package, we reference the following literature:

- S. Ejtemaee et al.:
  *Optimization of Yb+ fluorescence and hyperfine-qubit detection.*
  Phys. Rev. A **82**, 063419 (2010)
- H. Meyer et al.:
  *Laser spectroscopy and cooling of Yb<sup>+</sup> ions on a
  deep-UV transition* Phys. Rev. A **85**, 012502 (2012)
- H. Meyer:
  *A fibre-cavity based photonic interface for a single ion.*
  Dissertation, University of Cambridge (2014)
- S. Olmschenk et al.:
  *Manipulation and detection of a trapped Yb<sup>+</sup> hyperfine qubit.*
  Phys. Rev. A **76**, 052314 (2007)
- S. Olmschenk: *Quantum teleportation between distant matter qubits.*
  Dissertation, University of Michigan (2009)
