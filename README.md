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
Using a NN-parameterized likelihood NDE $\mathcal{L}\_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta})$ and training it to
minimize KL divergence, we recover a data-driven likelihood estimator.
Once trained, one can use standard MCMC (or nested sampling) to recover posterior for a particular observed data $\boldsymbol{d}_{\text{obs}}$.

See [examples](https://github.com/dprelogo/21cmLikelihoods/tree/main/examples) and [article](https://arxiv.org/abs/2305.03074) for more details.

# Implemented likelihoods
We implement three main likelihood categories, by relaxing classical inference constraints.

## Mean constraint
In order to estimate the mean better, a feed-forward NN is used which takes parameters $\boldsymbol{\theta}$ and outputs the mean:
$\boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) .$

The possible Gaussian likelihoods are then:

$$\mathcal{L}\_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}), \boldsymbol{\sigma}^2(\boldsymbol{\theta}\_{\text{fid}})) , $$

$$\mathcal{L}\_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}), \Sigma(\boldsymbol{\theta}\_{\text{fid}})) . $$

Here $\boldsymbol{\sigma}^2(\boldsymbol{\theta}\_{\text{fid}})$ and $\Sigma(\boldsymbol{\theta}\_{\text{fid}})$ represent the variance and covariance estimated at the fiducial parameter values.

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
## Covariance constraint
Likewise, we can also estimate the (co)variance matrix with a NN. In this scenario, the network can output one of the following:

$$\boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}), \boldsymbol{\sigma}^2\_{\text{NN}}(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) ,$$

$$\boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}), \Sigma\_{\text{NN}}(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) , $$

with their respective likelihoods:

$$\mathcal{L}\_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}), \boldsymbol{\sigma}^2\_{\text{NN}}(\boldsymbol{\theta})) ,$$

$$\mathcal{L}\_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}\_{\text{NN}}(\boldsymbol{\theta}), \Sigma\_{\text{NN}}(\boldsymbol{\theta})) .$$

In code:
```python
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
## Gaussian constraint
Finally, we can relax the Gaussian constraint as well. This can be done in a parametric way
by using Gaussian mixture networks, or non-parametric way with Conditional
Masked Autoregressive Flows (CMAF).

### Gaussian mixture network
The setup here is exactly the same as previous cases, with the difference that NN outputs
a Gaussian mixture:

$$\boldsymbol{\mu}\_{\text{NN}, 1}(\boldsymbol{\theta}), \Sigma\_{\text{NN}, 1}(\boldsymbol{\theta}), \phi\_1(\boldsymbol{\theta}), \ldots, \boldsymbol{\mu}\_{\text{NN}, K}(\boldsymbol{\theta}), \Sigma\_{\text{NN}, K}(\boldsymbol{\theta}), \phi\_K(\boldsymbol{\theta}) = \text{NN}(\boldsymbol{\theta}) ,$$

where $\boldsymbol{\mu}\_{\text{NN}, i}(\boldsymbol{\theta}), \Sigma\_{\text{NN}, i}(\boldsymbol{\theta})$ describe mean and covariance of the $i-\text{th}$ Gaussian and $\phi\_i(\boldsymbol{\theta})$ its relative (positive) weight, $\sum\_i \phi_i(\boldsymbol{\theta}) = 1$. Therefore, the full likelihood can be written as:

$$\mathcal{L}\_{\text{NN}}(\boldsymbol{d} | \boldsymbol{\theta}) = \sum\_{i=1}^K \phi\_i(\boldsymbol{\theta}) \cdot \mathcal{N}(\boldsymbol{d}| \boldsymbol{\mu}\_{\text{NN}, i}(\boldsymbol{\theta}), \Sigma\_{\text{NN}, i}(\boldsymbol{\theta})) .$$

In code:
```python
from py21cmlikelihoods import ConditionalGaussianMixture

NDE = ConditionalGaussianMixture(
    n_parameters = 2, 
    n_data = 5, 
    n_components = 3,
)
```

### Conditional Masked Autoregressive Flows
CMAF represents non-parametric density estimator, with large expressivity in the 
shape of the final distribution. Minimal example is the following:
```python
from py21cmlikelihoods import ConditionalMaskedAutoregressiveFlow

NDE = ConditionalMaskedAutoregressiveFlow(
    n_dim = 5,
    cond_n_dim = 2,
)
```

# Training NDE likelihood
To train NDE, simply format the training set and call the training function.
```python
from py21cmlikelihoods.utils import prepare_dataset

data_samples = np.load("data.npy")
param_samples = np.load("params.npy")
batch_size = 100
training_set = prepare_dataset(NDE, data_samples, param_samples, batch_size)

NDE.train(
    epochs = 100,
    dataset = training_set,
)
```
# Installation
To install and use the code, clone the repository and run
```bash
pip install -e .
```
For a full setup needed to run [examples](https://github.com/dprelogo/21cmLikelihoods/tree/main/examples),
check the the conda `environment.yml` and install it as
```bash
conda env create -f environment.yml
conda activate 21cmLikelihoods
pip install -e .
```

# Acknowledging
If you use the code in your research, please cite the original paper:
```
@ARTICLE{Prelogovic2023,
       author = {{Prelogovi{\'c}}, David and {Mesinger}, Andrei},
        title = "{The likelihood of the 21-cm power spectrum}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = may,
          eid = {arXiv:2305.03074},
        pages = {arXiv:2305.03074},
archivePrefix = {arXiv},
       eprint = {2305.03074},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230503074P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
