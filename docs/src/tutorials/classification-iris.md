# Classification on Iris dataset

We will use the iris dataset, which is included in the MLDatasets package. This dataset consists of measurements of the sepal length, sepal width, petal length, and petal width for three different types of iris flowers: Setosa, Versicolor, and Virginica.

## Getting started

To begin, we will load the required packages and the dataset:

```julia
using NeuroTreeModels
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
Random.seed!(123)
```

## Preprocessing

Before we can train our model, we need to preprocess the dataset. We will convert the class variable, which specifies the type of iris flower, into a categorical variable.

```julia
df = MLDatasets.Iris().dataframe

df[!, :class] = categorical(df[!, :class])
df[!, :class] .= levelcode.(df[!, :class])
target_name = "class"
feature_names = setdiff(names(df), [target_name])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]
```

## Training

Now we are ready to train our model. We first define a model configuration using the [`NeuroTreeRegressor`](@ref) model constructor. 
Then, we use [`NeuroTrees.fit`](@ref) to train a boosted tree model. We pass the optional `deval` argument to enable the usage of early stopping. 

```julia
config = NeuroTreeRegressor(
    device=:cpu,
    loss=:mlogloss,
    nrounds=400,
    outsize=3,
    depth=4,
    lr=2e-2,
)

m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    metric=:mlogloss,
    print_every_n=10,
    early_stopping_rounds=2,
)
```

## Diagnosis

Finally, we can get predictions by passing training and testing data to our model. We can then evaluate the accuracy of our model, which should be over 95% for this simple classification problem. 

```julia
p_train = m(dtrain)
p_eval = m(deval)
```

```julia-repl
julia> mean(dtrain[!, target_name] .== NeuroTreeModels.onecold(p_train'))
0.9833333333333333

julia> mean(deval[!, target_name] .== NeuroTreeModels.onecold(p_eval'))
1.0
```