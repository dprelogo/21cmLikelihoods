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

See [examples](https://github.com/dprelogo/21cmLikelihoods/tree/main/examples) and [article](https://arxiv.org/) for more details.

## Implemented likelihoods
We implement three main likelihood categories, by relaxing classical inference constraints.

### Mean constraint
In order to estimate the mean better, a feed-forward NN is used which takes parameters $\boldsymbol{\theta}$ and outputs the mean:
$$\boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) \, . $$
The possible Gaussian likelihoods are then:
$$\mathcal{L}_{\text{NN}}(\boldsymbol{d}_{PS} | \boldsymbol{\theta}) &= \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}), \boldsymbol{\sigma}^2(\boldsymbol{\theta}_{\text{fid}})) \, ,$$
$$\mathcal{L}_{\text{NN}}(\boldsymbol{d}_{PS} | \boldsymbol{\theta}) &= \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}), \Sigma(\boldsymbol{\theta}_{\text{fid}})) \, .$$
Here $\boldsymbol{\sigma}^2(\boldsymbol{\theta}_{\text{fid}})$ and $\Sigma(\boldsymbol{\theta}_{\text{fid}})$ represent the variance and covariance estimated at the fiducial parameter values.

In code, one can create such likelihoods as:
```python
import numpy as np
from py21cmlikelihoods import ConditionalGaussian

fiducial_covariance = np.load("cov.npy")

NDE = ConditionalGaussian(
    n_parameters = 2, 
    n_data = 5, 
    covariance = fiducial_covariance,
)
```
where `fiducial_covariance` can be 1D or 2D, depending if full or diagonal covariance is needed.
### Covariance constraint
Likewise, we can also estimate the (co)variance matrix with a NN. In this scenario, the network can output one of the following:
$$\boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}), \boldsymbol{\sigma}^2_{\text{NN}}(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) \, , $$
$$\boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}), \Sigma_{\text{NN}}(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) \, , $$
with their respective likelihoods:
$$\mathcal{L}_{\text{NN}}(\boldsymbol{d}_{PS} | \boldsymbol{\theta}) &= \mathcal{N}(\boldsymbol{d}_{PS}| \boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}), \boldsymbol{\sigma}^2_{\text{NN}}(\boldsymbol{\theta})) \, ,$$
$$\mathcal{L}_{\text{NN}}(\boldsymbol{d}_{PS} | \boldsymbol{\theta}) &= \mathcal{N}(\boldsymbol{d}_{PS}| \boldsymbol{\mu}_{\text{NN}}(\boldsymbol{\theta}), \Sigma_{\text{NN}}(\boldsymbol{\theta})) \, .$$

In code:
```python
from py21cmlikelihoods import ConditionalGaussian

NDE_diagonal = ConditionalGaussian(
    n_parameters = 2, 
    n_data = 5, 
    diagonal_covariance = True,
)

NDE_full = ConditionalGaussian(
    n_parameters = 2, 
    n_data = 5, 
    diagonal_covariance = False,
)
```
### Gaussian constraint


# Installation
To install and use the code, clone the repository and run
```bash
pip install -e .
```
For a full setup needed to run [examples](https://github.com/dprelogo/21cmLikelihoods/tree/main/examples),
check the `environment.yml` which one can install as
```bash
conda env create -f environment.yml
```

# Citing
