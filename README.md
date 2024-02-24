# NeuroTreeModels.jl

> Differentiable tree-based models for tabular data. 

| Documentation | CI Status |
|:------------------------:|:----------------:|
| [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://evovest.github.io/NeuroTreeModels.jl/dev

[ci-img]: https://github.com/Evovest/NeuroTreeModels.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Evovest/NeuroTreeModels.jl/actions?query=workflow%3ACI+branch%3Amain

## Installation

```julia
] add NeuroTreeModels
```

## Configuring a model

A model configuration is defined with the [NeuroTreeRegressor](@ref) constructor:

```julia
using NeuroTreeModels, DataFrames

config = NeuroTreeRegressor(
    loss = :mse,
    nrounds = 10,
    num_trees = 16,
    depth = 5,
)
```

## Training

Building a training a model according to the above `config` is done [NeuroTreeModels.fit](@ref).
See the docs for additinal features, notably early stopping support through the tracking of an evaluation metric.

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

Benchmarks against prominent ML librairies for tabular is performed at [MLBenchmarks.jl](https://github.com/Evovest/MLBenchmarks.jl).
