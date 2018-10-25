# Solving Burgers' Equation in 1D using the Finite Element Method

This is a Python 3 implementation of a finite element method (FEM) solver for Burgers' equation in 1D. There also a [version of this program in C++](https://github.com/michelrobijns/BurgersFEM). The C++ version is a bit harder to understand, but it's rediculously fast :)

## Features

* Solves Burgers' equation in 1D
* Uses piecewise-linear basis functions
* Uses the backward Euler method (or implicit Euler method) to advance the solution through time
* Uses Newton's method to solve the system of nonlinear equations
* Supports Dirichlet and periodic boundary conditions
* Supports uniformly and nonuniformly spaced meshes

## Dependencies

* Numpy
* matplotlib

## Usage

Run `main.py` and then `plot.sh` to plot the solution. The forcing function, boundary values, and initial values are defined as functions in `main.py`. The code should be self-explanatory.
