using NeuroTreeModels
using DataFrames
using CategoricalArrays

#################################
# vanilla DataFrame
#################################
nobs = 100
nfeats = 10
x = rand(nobs, nfeats);
df = DataFrame(x, :auto);
df.y = rand(nobs);

target_name = "y"
feature_names = Symbol.(setdiff(names(df), [target_name]))
batchsize = 32

###################################
# CPU
###################################
device = :cpu
dtrain =
    NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)

for d in dtrain
    @info length(d)
    @info size(d[1])
end

deval = NeuroTreeModels.get_df_loader_infer(df; feature_names, batchsize = 32)
for d in deval
    @info size(d)
end

###################################
# GPU
###################################
device = :gpu
dtrain =
    NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)

for d in dtrain
    @info length(d)
    @info size(d[1])
end

deval = NeuroTreeModels.get_df_loader_infer(df; feature_names, batchsize = 32)
for d in deval
    @info size(d)
end

###################################
# Categorical
###################################
target_name = "y"
feature_names = Symbol.(setdiff(names(df), [target_name]))
batchsize = 32
device = :gpu

x = rand(nobs, nfeats);
df = DataFrame(x, :auto);
df.y = categorical(rand(1:2, nobs));

dtrain =
    NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[2])
end
