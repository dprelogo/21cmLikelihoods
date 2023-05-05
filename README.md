# 21cmLikelihoods

Neural density estimators (NDEs) for the cosmic 21-cm power spectrum likelihoods.

Common assumptions of the classical Bayesian inferences with the 21-cm PS are:
- the likelihood shape is a Gaussian,
- the covariance matrix is usually fixed and pre-calculated at some fiducial parameter values,
- very often only diagonal covariance is used, ignoring all other correlations,
- the Gaussian mean at each point in parameter space is estimated from only one realization.
All of these assumptions mostly come in order to reduce computational costs.

In order to bypass all of these, we make use of Simulation-Based Inference (SBI).
It consists of three main steps:
- draw parameter sample from some distribution (possibly prior) - $\tilde{\boldsymbol{\theta}} \sim \pi(\boldsymbol{\theta})$

<!-- train NDEs on a database of 21-cm PS, and use
standard MCMC (or nested sampling) to recover posterior. Such procedure is a basis
of a Simulation-Based Inference (SBI).  -->