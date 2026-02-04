using NeuroTreeModels
using MLDatasets
using DataFrames
using Statistics: mean
using StatsBase: median
using CategoricalArrays
using Random
using CUDA
using CategoricalArrays

Random.seed!(123)

df = MLDatasets.Titanic().dataframe

# convert string feature to Categorical
transform!(df, :Sex => categorical => :Sex)
transform!(df, :Sex => ByRow(levelcode) => :Sex)

# treat string feature and missing values
transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age);

# remove unneeded variables
df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio*nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "Survived"
feature_names = setdiff(names(df), ["Survived"])

config = NeuroTreeRegressor(;
    loss = :logloss,
    actA = :identity,
    nrounds = 400,
    depth = 4,
    lr = 5e-2,
    early_stopping_rounds = 3,
    device = :cpu,
)

m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n = 10,
)

p_train = m(dtrain; device = :cpu)
p_eval = m(deval; device = :cpu)

@info mean((p_train .> 0.5) .== (dtrain[!, target_name] .> 0.5))
@info mean((p_eval .> 0.5) .== (deval[!, target_name] .> 0.5))

###################################
# MLJ
###################################
using MLJBase, NeuroTreeModels
m = NeuroTreeRegressor(depth = 5, nrounds = 40, batchsize = 1024, device = :cpu)
mach = machine(m, dtrain[:, feature_names], Float32.(dtrain[!, target_name])) |> fit!
p = predict(mach, dtrain[:, feature_names])
@info mean((p .> 0.5) .== (dtrain[!, target_name] .> 0.5))

m = NeuroTreeRegressor(depth = 5, nrounds = 40, batchsize = 1024, device = :gpu)
mach = machine(m, dtrain[:, feature_names], Float32.(dtrain[!, target_name])) |> fit!
p = predict(mach, dtrain[:, feature_names])
@info mean((p .> 0.5) .== (dtrain[!, target_name] .> 0.5))
