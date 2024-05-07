using NeuroTreeModels
using DataFrames
using PlotlyLight

#################################
# vanilla DataFrame
#################################
nobs = 8
nfeats = 5
depth = 3
ntrees = 12
x, y = randn(Float32, nfeats, nobs), randn(Float32, nobs);

n1 = NeuroTree(5 => 4; depth, ntrees, actA=tanh)
n1(x)
@code_warntype n1(x)

nw = rand(Float32, 2^depth-1, ntrees, nobs)
@code_warntype NeuroTreeModels.leaf_weights!(nw)
