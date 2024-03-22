module NeuroTreeModels

using Base.Threads: @threads, nthreads
import Tables
using DataFrames
using Statistics: mean, std
using Random

using CUDA
import Optimisers
import Optimisers: OptimiserChain, WeightDecay, Adam, NAdam, Nesterov, Descent, Momentum, AdaDelta

import Flux
import Flux: @functor, trainmode!, gradient, Chain, DataLoader, cpu, gpu
import Flux: logσ, logsoftmax, softmax, softmax!, sigmoid, sigmoid_fast, hardsigmoid, tanh, tanh_fast, hardtanh, softplus, onecold, onehotbatch
import Flux: BatchNorm, Dense, MultiHeadAttention, Parallel

using ChainRulesCore
import ChainRulesCore: rrule

import MLJModelInterface as MMI

export NeuroTreeRegressor, NeuroTreeModel, NeuroTree

include("data.jl")
include("utils.jl")
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
