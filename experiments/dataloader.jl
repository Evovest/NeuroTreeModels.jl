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
dtrain = NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)

for d in dtrain
    @info length(d)
    @info size(d[1])
end

deval = NeuroTreeModels.get_df_loader_infer(df; feature_names, batchsize=32)
for d in deval
    @info size(d)
end

<<<<<<< HEAD
rand(2,3,4)

struct S5{N}
    b
end
s = S5{1}(rand(3))

struct S6{N}
    N
    b
end
s = S6{1}(1, rand(3))
typeof(s)
s.N
s = S6(1, rand(3))
S6(outsize::Int, vec) = S6{outsize}(outsize, vec)
s = S6(1, rand(3))
typeof(s)

struct S7{N,T}
    N
    b::T
end
S7(outsize, vec) = S7{outsize, typeof(vec)}(outsize, vec)
s = S7(2.2, rand(3))
typeof(s)
s.N


struct S8{N,T}
    N::Int
    b::T
end
S8(outsize, vec) = S8{outsize, typeof(vec)}(outsize, vec)
s = S8(2.2, rand(3))
typeof(s)
s.N
=======
###################################
# GPU
###################################
device = :gpu
dtrain = NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)

for d in dtrain
    @info length(d)
    @info size(d[1])
end

deval = NeuroTreeModels.get_df_loader_infer(df; feature_names, batchsize=32)
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

dtrain = NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, batchsize, device)
for d in dtrain
    @info length(d)
    @info size(d[1])
    @info typeof(d[2])
end
>>>>>>> main
