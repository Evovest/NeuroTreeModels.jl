using NeuroTreeModels
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random
Random.seed!(123)

df = MLDatasets.Iris().dataframe

df[!, :class] = categorical(df[!, :class])
df[!, :class] .= levelcode.(df[!, :class])
target_name = "class"
feature_names = setdiff(names(df), [target_name])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

config = NeuroTreeRegressor(
    device = :cpu,
    loss = :mlogloss,
    nrounds = 400,
    outsize = 3,
    depth = 4,
    lr = 2e-2,
)

m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    metric = :mlogloss,
    print_every_n = 10,
    early_stopping_rounds = 2,
)

# Predictions depend on the number of samples in the dataset
m(dtrain[1:1, :])[1:1, :]
m(dtrain[1:2, :])[1:1, :]
m(dtrain[1:3, :])[1:1, :]
m(dtrain[1:10, :])[1:1, :]
