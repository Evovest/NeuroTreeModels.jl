using Test
using NeuroTrees
using Tables
using DataFrames

config = NeuroTreeRegressor(
    device=:cpu,
    loss=:mse,
    tree_type=:stack,
    actA=:identity,
    init_scale=1.0,
    nrounds=200,
    depth=4,
    ntrees=32,
    stack_size=1,
    hidden_size=1,
    outsize=1,
    batchsize=2048,
    lr=1e-3,
)

# stack tree
nobs = 1_000
nfeats = 100
x = rand(Float32, nfeats, nobs)
feature_names = "var_" .* string.(1:nobs)

loss = NeuroTreeModels.get_loss_fn(config)
L = NeuroTreeModels.get_loss_type(config)
chain = NeuroTreeModels.get_model_chain(L; config, nfeats)
info = Dict(
    :device => config.device,
    :nrounds => 0,
    :feature_names => feature_names
)
m = NeuroTreeModel(L, chain, info)
@time m(x);
