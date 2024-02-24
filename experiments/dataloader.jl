using NeuroTreeModels
using DataFrames

#################################
# vanilla DataFrame
#################################
nobs=100
nfeats=10
x = rand(nobs, nfeats);
df = DataFrame(x, :auto);
df.y = rand(nobs);

target_name = "y"
feature_names = setdiff(names(df), [target_name])

dtrain = NeuroTrees.get_df_loader_train(df; feature_names, target_name, batchsize=32)
for d in dtrain
    @info length(d)
    @info size(d[1])
end

deval = NeuroTrees.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end
