using NeuroTreeModels
using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using PlotlyLight
using Random
Random.seed!(123)

df = MLDatasets.Iris().dataframe

df[!, :class] = categorical(df[!, :class])
target_name = "class"
feature_names = setdiff(names(df), [target_name])

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

config =
    NeuroTreeClassifier(nrounds = 400, depth = 3, ntrees = 4, lr = 5e-2, batchsize = 60)

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

p_train = m(dtrain)
p_eval = m(deval)
mean(levelcode.(dtrain[!, target_name]) .== NeuroTreeModels.onecold(p_train'))
mean(levelcode.(deval[!, target_name]) .== NeuroTreeModels.onecold(p_eval'))

nts = m.chain.layers[2]
nt = nts.trees[1]
w = nt.w

xnames = "feat" .* string.(1:4)
ynames = ["T$j/N$i" for i = 1:(2^config.depth-1), j = 1:config.ntrees]
ynames = vec(ynames)

# p = plot(z=w, type="heatmap")  # Make plot
p = plot(z = w, x = xnames, y = ynames, type = "heatmap")  # Make plot
