using NeuroTreeModels
using DataFrames
using PlotlyLight

#################################
# vanilla DataFrame
#################################
nobs = 1000
nfeats = 10
x, y = randn(nobs, nfeats), randn(nobs);
df = DataFrame(x, :auto);
df.y = y;

config = NeuroTreeRegressor()

chain = get_model_chain(L; config, nfeats)
info = Dict(
    :device => config.device,
    :nrounds => 0,
    :feature_names => feature_names
)
m = NeuroTreeModel(L, chain, info)
if config.device == :gpu
    m = m |> gpu
end

optim = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
opts = Optimisers.setup(optim, m)


deval = NeuroTrees.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end
