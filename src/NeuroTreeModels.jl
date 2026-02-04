module NeuroTreeModels

using Base.Threads: @threads, nthreads
import Tables
using DataFrames
using CategoricalArrays
using Statistics: mean, std
using Random

using CUDA
using CUDA: CuIterator

import Optimisers
import Optimisers:
    OptimiserChain, WeightDecay, Adam, NAdam, Nesterov, Descent, Momentum, AdaDelta

import Flux
import Flux: @layer, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: onecold, onehotbatch
import Flux: BatchNorm, Dense, MultiHeadAttention, Parallel
import Flux: logσ, logsoftmax, sigmoid, softmax!

using ChainRulesCore
import ChainRulesCore: rrule

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export NeuroTreeRegressor, NeuroTreeClassifier, NeuroTreeModel, NeuroTree

include("data.jl")
include("learners.jl")
include("model.jl")
include("loss.jl")
include("metrics.jl")
include("callback.jl")
using .CallBacks
include("infer.jl")
include("entmax.jl")
include("fit.jl")
include("MLJ.jl")

end # module
