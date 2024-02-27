# Logistic Regression on Titanic Dataset

We will use the Titanic dataset, which is included in the MLDatasets package. It describes the survival status of individual passengers on the Titanic. The model will be approached as a logistic regression problem, although a Classifier model could also have been used (see the `Classification - Iris` tutorial). 

## Getting started

To begin, we will load the required packages and the dataset:

```julia
using NeuroTreeModels
using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
Random.seed!(123)
```

## Preprocessing

A first step in data processing is to prepare the input features in a model compatible format. 

EvoTrees' Tables API supports input that are either `Real` (incl. `Bool`) or `Categorical`. `Bool` variables are treated as unordered, 2-levels categorical variables.
A recommended approach for `String` features such as `Sex` is to convert them into an unordered `Categorical`. 

For dealing with features with missing values such as `Age`, a common approach is to first create an `Bool` indicator variable capturing the info on whether a value is missing.
Then, the missing values can be imputed (replaced by some default values such as `mean` or `median`, or more sophisticated approach such as predictions from another model).

```julia
df = MLDatasets.Titanic().dataframe

# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)
transform!(df, :Sex => ByRow(levelcode) => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);

# remove unneeded variables
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

```

The full data can now be split according to train and eval indices. 
Target and feature names are also set.

```julia
train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), [target_name])
```

## Training

Now we are ready to train our model. We first define a model configuration using the [`NeuroTreeRegressor`](@ref) model constructor. 
Then, we use [`NeuroTreeModels.fit`](@ref) to train a boosted tree model. We pass the optional `deval` argument to enable the usage of early stopping. 

```julia
config = NeuroTreeRegressor(
    device=:cpu,
    loss=:logloss,
    nrounds=400,
    depth=4,
    lr=2e-2,
)

m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    metric=:logloss,
    print_every_n=10,
    early_stopping_rounds=2,
)
```


## Diagnosis

We can get predictions by passing training and testing data to our model. We can then evaluate the accuracy of our model, which should be around 85%. 

```julia
p_train = m(dtrain)
p_eval = m(deval)
```

```julia-repl
julia> mean((p_train .> 0.5) .== dtrain[!, target_name])
0.8527349228611

julia> mean((p_eval .> 0.5) .== deval[!, target_name])
0.8426966292134831
```
