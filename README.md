# 21cmLikelihoods

Neural density estimators (NDEs) for the cosmic 21-cm power spectrum likelihoods.

Common assumptions of the classical Bayesian inferences with the 21-cm PS are:
- the likelihood shape is a Gaussian,
- the covariance matrix is usually fixed and pre-calculated at some fiducial parameter values,
- often only diagonal covariance is used, ignoring other correlations,
- the Gaussian mean at each point in parameter space is estimated from only one realization.
All of these assumptions mostly come in order to reduce computational costs,
and have a potentially significant impact on the final posterior.

In order to bypass all of these, we use Simulation-Based Inference (SBI).
It can be summarized into two main steps:
- draw parameter sample from some distribution (possibly prior) - $\tilde{\boldsymbol{\theta}} \sim \pi(\boldsymbol{\theta})$,
- draw a data sample by using a realistic data simulator - $\tilde{\boldsymbol{d}} \sim \mathcal{L}(\boldsymbol{d} | \tilde{\boldsymbol{\theta}})$,
- Repeat many times.
A database of (parameter, data sample) pairs follow full distribution 
$P(\boldsymbol{d}, \boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{d} | \boldsymbol{\theta}) \cdot \pi(\boldsymbol{\theta})$.
Using a NN-parameterized likelihood NDE $\mathcal{L}_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta})$ and training it to
minimize KL divergence, we recover a data-driven likelihood estimator.
With that in hand, one can use standard MCMC (or nested sampling) to recover posterior.


## Implemented likelihoods
We implement three main likelihood categories, by relaxing classical inference constraints.

### Mean constraint

### Covariance constraint

### Gaussian constraint

<!-- train NDEs on a database of 21-cm PS, and use
standard MCMC (or nested sampling) to recover posterior. Such procedure is a basis
of a Simulation-Based Inference (SBI).  -->