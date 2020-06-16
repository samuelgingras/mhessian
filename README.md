mhessian - A MATLAB toolbox for simulation smoothing
====================================================

## Description

This is a Matlab toolbox that implement the HESSIAN (highly efficient simulation smoothing in a nutshell) method to a set of specific univariate state-space models with a Gaussian unobserved latent state equation and non-linear and non-Gaussian measurement equation. The method is a generic algorithm that approximate the state posterior distribution in order to draw a complete vector of states and evaluate it log posterior density. 

Model specific requirement of the HESSIAN method are routine to evaluate the log likelihood of an individual observation given the state and the model specific parameters and it first five derivatives. Computational routine to perform those model specific evaluation are implemented for the following observation models:

1. Stochastic volatility model with standard Gaussian innovations
2. Stochastic volatility model with finite mixture of Gaussian innovations
3. Stochastic volatility model with Student's t innovations
4. Stochastic volatility model with Auto-regressive Student's t innovations

5. Duration model with Exponential distribution
6. Duration model with finite mixture of Exponential distribution

7. Dynamic counts model with Poisson distribution
8. Dynamic counts model with Gamma-Poisson distribution

The HESSIAN approximation can then be used for any of the previous eight models as an importance density to compute approximations of integrals, as a proposal density for a Markov chain Monte Carlo block updating the conditional distribution of latent state or to perform Laplace-like approximation of a likelihood function.

## Content
To be completed ...

## Installation
To be completed ...

## Documentation
Per-function documentation is available for information about each individual syntax and input/output arguments. To access specific documentation, simply enter: 
    help NAME_OF_THE_FUNCTION