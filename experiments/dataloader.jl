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
