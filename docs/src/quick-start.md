# Getting started with NeuroTreeModels.jl

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
    device = :cpu
)
```

For training on GPU, use `device=:gpu` in the constructor, and optionally `gpuID=0` to target a specific a device.

## Training

Building and training a model according to the above `config` is done with [NeuroTreeModels.fit](@ref).
See the docs for additional features, notably early stopping support through the tracking of an evaluation metric.

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