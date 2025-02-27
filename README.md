# NeuroTreeModels.jl

> Differentiable tree-based models for tabular data. 

| Documentation | CI Status | DOI |
|:------------------------:|:----------------:|:----------------:|
| [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] | [![][DOI-img]][DOI-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://evovest.github.io/NeuroTreeModels.jl/dev

[ci-img]: https://github.com/Evovest/NeuroTreeModels.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Evovest/NeuroTreeModels.jl/actions?query=workflow%3ACI+branch%3Amain

[DOI-img]: https://zenodo.org/badge/762536508.svg
[DOI-url]: https://zenodo.org/doi/10.5281/zenodo.10725028

## Installation

```julia
] add NeuroTreeModels
```

⚠ Compatible with Julia >= v1.10

## Configuring a model

A model configuration is defined with on of the constructor:
- [NeuroTreeRegressor](https://evovest.github.io/NeuroTreeModels.jl/dev/models#NeuroTreeModels.NeuroTreeRegressor)
- [NeuroTreeClassifier](https://evovest.github.io/NeuroTreeModels.jl/dev/models#NeuroTreeModels.NeuroTreeClassifier)

```julia
using NeuroTreeModels, DataFrames

config = NeuroTreeRegressor(
    loss = :mse,
    nrounds = 10,
    num_trees = 16,
    depth = 5,
    device = :cpu
)
```

For training on GPU, use `device=:gpu` in the constructor, and optionally `gpuID=0` to target a specific a device.

## Training

Building and training a model according to the above `config` is done with [NeuroTreeModels.fit](https://evovest.github.io/NeuroTreeModels.jl/dev/API#NeuroTreeModels.fit).
See the docs for additional features, notably early stopping support through the tracking of an evaluation metric on evaluation data.

```julia
nobs, nfeats = 1_000, 5
dtrain = DataFrame(randn(nobs, nfeats), :auto)
dtrain.y = rand(nobs)
feature_names, target_name = names(dtrain, r"x"), "y"

m = NeuroTreeModels.fit(config, dtrain; feature_names, target_name)
```

## Inference

```julia
p = m(dtrain)
p = m(dtrain; device=:gpu)
```

## MLJ

NeuroTreeModels.jl supports the [MLJ](https://github.com/alan-turing-institute/MLJ.jl) Interface. 

```julia
using MLJBase, NeuroTreeModels
m = NeuroTreeRegressor(depth=5, nrounds=10)
X, y = @load_boston
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```

## Benchmarks

Benchmarking against prominent ML libraries for tabular data is performed at [MLBenchmarks.jl](https://github.com/Evovest/MLBenchmarks.jl).
