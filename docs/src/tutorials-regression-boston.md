# Regression on Boston Housing Dataset

We will use the Boston Housing dataset, which is included in the MLDatasets package. It's derived from information collected by the U.S. Census Service concerning housing in the area of Boston. Target variable represents the median housing value.

## Getting started

To begin, we will load the required packages and the dataset:

```julia
using NeuroTreeModels
using MLDatasets
using DataFrames
using Statistics: mean, std
using CategoricalArrays
using Random
Random.seed!(123)
```

## Preprocessing

Before we can train our model, we need to preprocess the dataset. We will split our data according to train and eval indices, and separate features from the target variable.

```julia
df = MLDatasets.BostonHousing().dataframe
feature_names = setdiff(names(df), ["MEDV"])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

_mean, _std = mean(dtrain.MEDV), std(dtrain.MEDV)
transform!(dtrain, :MEDV => (x -> (x .- _mean) ./ _std) => "target")
transform!(deval, :MEDV => (x -> (x .- _mean) ./ _std) => "target")

target_name = "target"
```

## Training

Now we are ready to train our model. We first define a model configuration using the [`NeuroTreeRegressor`](@ref) model constructor. 
Then, we use [`NeuroTreeModels.fit`](@ref) to train a boosted tree model. We pass the optional `deval` argument to enable the usage of early stopping. 

```julia
config = NeuroTreeRegressor(
    loss=:mse,
    nrounds=400,
    depth=5,
    lr=2e-2,
    early_stopping_rounds=2,
    device=:cpu
)

m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n=10,
)
```

## Diagnosis

Finally, we can get predictions by passing training and testing data to our model. We can then apply various evaluation metric, such as the MAE (mean absolute error):  

```julia
p_train = m(dtrain) .* _std .+ _mean
p_eval = m(deval) .* _std .+ _mean
```

```julia-repl
julia> mean(abs.(p_train .- dtrain[!, "MEDV"]))
0.8985784079860025

julia> mean(abs.(p_eval .- deval[!, "MEDV"]))
2.3287859731914597
```