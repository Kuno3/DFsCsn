# DFsCsn

DFsCsn is a Julia package that provides tools for simulating statistical models considering spatio-temporal structures and skewness. It also facilitates sampling from the posterior distribution of these models. Please refer to Kuno and Murakami (2024) for details.

## Installation

To install DFsCsn, use the following command in Julia's REPL:

```julia
using Pkg
Pkg.add("https://github.com/Kuno3/DFsCsn")
```

## Getting Started

### Importing the Package

To start using DFsCsn, simply import the package:

```julia
using DFsCsn
```

### Simulation

DFsCsn allows you to specify parameters for your spatiotemporal model and run simulations. Here’s an example of how to set parameters and execute a simulation:

```julia
# Define parameters
# Below is an example
T = 10 # Time length
W = [0, 1, 1, 0; 1, 0, 0, 1; 1, 0, 0, 1; 0, 1, 1, 0] # Adjacency matrix
feature = randn(T, length(W), 2) # Features
# You can also define other parameters 
simulator = DFsCsnSimulator(T, W, feature)
simulate(simulator)
```

### Sampling from the Posterior Distribution

You can sample from the posterior distribution as follows:

```julia
# Define observations
# Below is an example
y = simulator.y
W = simulator.W
feature = simulator.feature

# Sample from the posterior distribution
# You can also define the parameters for prior distributions
sampler = DFsCsnPostSampler(y, W, feature)
results = sampling(sampler, 1000)
```

## License

DFsCsn is licensed under the MIT License.

## Reference
Kuno, H., Murakami, D., 2024. Efficient bayesian dynamic closed skew-normal model for spatio-
temporal data with preserving mean and covariance. Preprint