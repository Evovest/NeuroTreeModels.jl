abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end

const _loss_type_dict = Dict(
    :mse => MSE,
    :mae => MAE,
    :logloss => LogLoss,
    :mlogloss => MLogLoss,
    :gaussian_mle => GaussianMLE,
)

mutable struct NeuroTreeRegressor <: MMI.Deterministic
    loss::Symbol
    nrounds::Int
    lr::Float32
    wd::Float32
    batchsize::Int
    actA::Symbol
    outsize::Int
    depth::Int
    ntrees::Int
    hidden_size::Int
    stack_size::Int
    init_scale::Float32
    MLE_tree_split::Bool
    rng::Any
    device::Symbol
    gpuID::Int
end

"""
  NeuroTreeRegressor(;kwargs...)

A model type for constructing a NeuroTreeRegressor, based on [NeuroTreeModels.jl](https://github.com/Evovest/NeuroTreeModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `loss=:mse`:              Loss to be be minimized during training. One of:
  - `:mse`
  - `:mae`
  - `:logloss`
  - `:mlogloss`
  - `:gaussian_mle`
- `nrounds=10`:             Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `actA=:tanh`:             Activation function applied to each of input variable for determination of split node weight. Can be one of:
    - `:tanh`
    - `:identity`
- `outsize=1`:              Number of predictions returned by the model. Typically only used for classification tasks and set to the number of target levels / classes.
- `depth=6`:            Depth of a tree. Must be >= 1. A tree of depth 1 has 2 prediction leaf nodes. A complete tree of depth N contains `2^N` terminal leaves and `2^N - 1` split nodes.
  Compute cost is proportional to `2^depth`. Typical optimal values are in the 3 to 5 range.
- `ntrees=64`:              Number of trees (per stack).
- `hidden_size=16`:         Size of hidden layers. Applicable only when `stack_size` > 1.
- `stack_size=1`:           Number of stacked NeuroTree blocks.
- `init_scale=1.0`:         Scaling factor applied to the predictions weights. Values in the `]0, 1]` short result in best performance. 
- `MLE_tree_split=false`:   Whether independent models are buillt for each of the 2 parameters (mu, sigma) of the the `gaussian_mle` loss.
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`:            Device to use. Either `:cpu` or `:gpu` (recommended as it improves significantly the training speed). 
- `gpuID=0`:                ID of the GPU to use for training.

# Internal API

Do `config = NeuroTreeRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in NeuroTreeRegressor(loss=...).

## Training model

A model is trained using [`fit`](@ref):

```julia
m = fit(config, dtrain; feature_names, target_name, kwargs...)
```

## Inference

Models act as a functor. returning predictions when called as a function with features as argument:

```julia
m(data)
```

# MLJ Interface

From MLJ, the type can be imported using:

```julia
NeuroTreeRegressor = @load NeuroTreeRegressor pkg=NeuroTreeModels
```

Do `model = NeuroTreeRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTreeRegressor(loss=...)`.

## Training model

In MLJ or MLJBase, bind an instance `model` to data with
    `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

## Operations

- `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above.

## Fitted parameters

The fields of `fitted_params(mach)` are:
  - `:fitresult`: The `NeuroTreeModel` object.

## Report

The fields of `report(mach)` are:
  - `:features`: The names of the features encountered in training.

# Examples

## Internal API

```julia
using NeuroTreeModels, DataFrames
config = NeuroTreeRegressor(depth=5, nrounds=10)
nobs, nfeats = 1_000, 5
dtrain = DataFrame(randn(nobs, nfeats), :auto)
dtrain.y = rand(nobs)
feature_names, target_name = names(dtrain, r"x"), "y"
m = fit(config, dtrain; feature_names, target_name)
p = m(dtrain)
```

## MLJ Interface

```julia
using MLJBase, NeuroTreeModels
m = NeuroTreeRegressor(depth=5, nrounds=10)
X, y = @load_boston
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```
"""
function NeuroTreeRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :nrounds => 10,
        :lr => 1.0f-2,
        :wd => 0.0f0,
        :batchsize => 2048,
        :actA => :tanh,
        :outsize => 1,
        :depth => 4,
        :ntrees => 64,
        :hidden_size => 1,
        :stack_size => 1,
        :init_scale => 0.1,
        :MLE_tree_split => false,
        :rng => 123,
        :device => :cpu,
        :gpuID => 0,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 &&
        @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])

    config = NeuroTreeRegressor(
        Symbol(args[:loss]),
        args[:nrounds],
        Float32(args[:lr]),
        Float32(args[:wd]),
        args[:batchsize],
        Symbol(args[:actA]),
        args[:outsize],
        args[:depth],
        args[:ntrees],
        args[:hidden_size],
        args[:stack_size],
        args[:init_scale],
        args[:MLE_tree_split],
        args[:rng],
        Symbol(args[:device]),
        args[:gpuID],
    )

    return config
end

get_loss_type(config::NeuroTreeRegressor) = _loss_type_dict[config.loss]

struct NeuroTree{W,B,P}
    w::W
    b::B
    p::P
    actA::Function
end

@functor NeuroTree
# Flux.trainable(m::NeuroTree) = (w=m.w, b=m.b, p=m.p)

function node_weights(m::NeuroTree, x)
    # [N X T, F] * [F, B] => [N x T, B]
    # nw = sigmoid_fast.(m.w * x .+ m.b)
    nw = sigmoid_fast.(m.actA.(m.w) * x .+ m.b)
    # [N x T, B] -> [N, T, B]
    return reshape(nw, :, size(m.p, 3), size(x, 2))
end

include("leaf_weights.jl")

function (m::NeuroTree{W,B,P})(x::W) where {W,B,P}
    # [F, B] -> [N, T, B]
    nw = node_weights(m, x)
    # [N, T, B] -> [L, T, B]
    (_, lw) = leaf_weights!(nw)
    # [L, T, B], [P, L, T] -> [P, B]
    pred = dot_prod_agg(lw, m.p) ./ size(m.p, 3)
    return pred
end

dot_prod_agg(lw, p) = dropdims(sum(reshape(lw, 1, size(lw)...) .* p, dims=(2, 3)), dims=(2, 3))

"""
    NeuroTree

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1.0)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    nt = NeuroTree(
        Flux.glorot_uniform(nnodes * ntrees, ins), # w
        zeros(Float32, nnodes * ntrees), # b
        Float32.((rand(outs, nleaves, ntrees) .- 0.5) .* sqrt(12) .* init_scale), # p
        # Float32.(randn(outs, nleaves, ntrees) ./ 1 .* init_scale), # p
        actA,
    )
    return nt
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1.0)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    nt = NeuroTree(
        Flux.glorot_uniform(nnodes * ntrees, ins), # w
        zeros(Float32, nnodes * ntrees), # b
        Float32.((rand(outs, nleaves, ntrees) .- 0.5) .* sqrt(12) .* init_scale), # p
        # Float32.(randn(outs, nleaves, ntrees) ./ 1 .* init_scale), # p
        actA,
    )
    return nt
end

"""
    StackTree
A StackTree is made of a collection of NeuroTrees.
"""
struct StackTree
    trees::Vector{NeuroTree}
end
@functor StackTree

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, stack_size=2, hidden_size=8, actA=identity, init_scale=1.0)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; depth, ntrees, actA, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; depth, ntrees, actA, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; depth, ntrees, actA, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; depth, ntrees, actA, init_scale)
            push!(trees, tree)
        end
    end
    m = StackTree(trees)
    return m
end

function (m::StackTree)(x::AbstractMatrix)
    p = m.trees[1](x)
    for i in 2:length(m.trees)
        if i < length(m.trees)
            p = p .+ m.trees[i](p)
        else
            _p = m.trees[i](p)
            p = view(p, 1:size(_p, 1), :) .+ _p
        end
    end
    return p
end
# function (m::StackTree)(x::AbstractMatrix)
#     p = m.trees[1](x)
#     for i in 2:length(m.trees)
#         p = m.trees[i](p)
#     end
#     return p
# end


"""
    NeuroTreeModel
A NeuroTreeModel is made of a collection of Tree, either regular `NeuroTree` or `StackTree`.
Prediction is the sum of all the trees composing a NeuroTreeModel.
"""
struct NeuroTreeModel{L<:LossType}
    _loss_type::Type{L}
    chain::Chain
    info::Dict{Symbol,Any}
end
@functor NeuroTreeModel

"""
    (m::NeuroTreeModel)(x::AbstractMatrix)
    (m::NeuroTreeModel)(data::AbstractDataFrame)

Inference for NeuroTreeModel
"""
function (m::NeuroTreeModel)(x::AbstractMatrix)
    p = m.chain(x)
    if size(p, 1) == 1
        p = dropdims(p; dims=1)
    end
    return p
end
function (m::NeuroTreeModel)(data::AbstractDataFrame)
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048, device=m.info[:device])
    p = infer(m, dinfer)
    return p
end

const _act_dict = Dict(
    :identity => identity,
    :tanh => tanh,
    :hardtanh => hardtanh,
    :sigmoid => sigmoid,
    :hardsigmoid => hardsigmoid
)

function get_model_chain(L; config, nfeats)

    if L <: GaussianMLE && config.MLE_tree_split
        chain = Chain(
            BatchNorm(nfeats),
            Parallel(
                vcat,
                StackTree(nfeats => config.outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale),
                StackTree(nfeats => config.outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale)
            )
        )
    else
        outsize = L <: GaussianMLE ? 2 * config.outsize : config.outsize
        chain = Chain(
            BatchNorm(nfeats),
            StackTree(nfeats => outsize;
                depth=config.depth,
                ntrees=config.ntrees,
                stack_size=config.stack_size,
                hidden_size=config.hidden_size,
                actA=_act_dict[config.actA],
                init_scale=config.init_scale)
        )
    end

    return chain

end
