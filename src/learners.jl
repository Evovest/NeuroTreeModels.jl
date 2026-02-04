abstract type LossType end
abstract type MSE <: LossType end
abstract type MAE <: LossType end
abstract type LogLoss <: LossType end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: LossType end
abstract type Tweedie <: LossType end

"""
    mk_rng

make a Random Number Generator object
"""
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(rng::T) where {T<:Integer} = Random.MersenneTwister(rng)

const _loss_type_dict = Dict(
    :mse => MSE,
    :mae => MAE,
    :logloss => LogLoss,
    :tweedie => Tweedie,
    :gaussian_mle => GaussianMLE,
    :mlogloss => MLogLoss,
)

mutable struct NeuroTreeRegressor <: MMI.Deterministic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    early_stopping_rounds::Int
    lr::Float32
    wd::Float32
    batchsize::Int
    actA::Symbol
    depth::Int
    ntrees::Int
    hidden_size::Int
    stack_size::Int
    scaler::Bool
    init_scale::Float32
    MLE_tree_split::Bool
    rng::AbstractRNG
    device::Symbol
    gpuID::Int
end

"""
  NeuroTreeRegressor(; kwargs...)

A model type for constructing a NeuroTreeRegressor, based on [NeuroTreeModels.jl](https://github.com/Evovest/NeuroTreeModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `loss=:mse`:              Loss to be be minimized during training. One of:
  - `:mse`
  - `:mae`
  - `:logloss`
  - `:gaussian_mle`
- `metric=nothing`: evaluation metric tracked on `deval`. Can be one of:
  - `:mse`
  - `:mae`
  - `:logloss`
  - `:gaussian_mle`
- `nrounds=100`:            Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `actA=:tanh`:             Activation function applied to each of input variable for determination of split node weight. Can be one of:
    - `:tanh`
    - `:identity`
- `depth=6`:                Depth of a tree. Must be >= 1. A tree of depth 1 has 2 prediction leaf nodes. A complete tree of depth N contains `2^N` terminal leaves and `2^N - 1` split nodes.
  Compute cost is proportional to `2^depth`. Typical optimal values are in the 3 to 5 range.
- `ntrees=64`:              Number of trees (per stack).
- `hidden_size=16`:         Size of hidden layers. Applicable only when `stack_size` > 1.
- `stack_size=1`:           Number of stacked NeuroTree blocks.
- `scaler=true`:            Whether a learnable scaling factor, prior to the sigmoid activation, should be used. Otherwise a fixed scaling of 1.0 is used if `scaler=false`.  
- `init_scale=0.1`:         Scaling factor applied to the predictions weights. Values in the range `]0, 1]` should result in best convergence. 
- `MLE_tree_split=false`:   Whether independent models are buillt for each of the 2 parameters (mu, sigma) of the the `gaussian_mle` loss.
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`:            Device on which to perform the computation, either `:cpu` or `:gpu`
- `gpuID=0`:                GPU device to use, only relveant if `device = :gpu` 

# Internal API

Do `config = NeuroTreeRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTreeRegressor(loss=:logloss, depth=5, ...)`.

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
        :metric => nothing,
        :nrounds => 100,
        :early_stopping_rounds => typemax(Int),
        :lr => 1.0f-2,
        :wd => 0.0f0,
        :batchsize => 2048,
        :actA => :tanh,
        :depth => 4,
        :ntrees => 64,
        :hidden_size => 1,
        :stack_size => 1,
        :scaler => true,
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

    loss = Symbol(args[:loss])
    loss ∉ [:mse, :mae, :logloss, :tweedie, :gaussian_mle] &&
        error("The provided kwarg `loss`: $loss is not supported.")

    _metric_list = [:mse, :mae, :logloss, :tweedie, :gaussian_mle]
    if isnothing(args[:metric])
        metric = loss
    else
        metric = Symbol(args[:metric])
    end
    if metric ∉ _metric_list
        error("Invalid metric. Must be one of: $_metric_list")
    end

    rng = mk_rng(args[:rng])
    device = Symbol(args[:device])

    config = NeuroTreeRegressor(
        loss,
        metric,
        args[:nrounds],
        args[:early_stopping_rounds],
        Float32(args[:lr]),
        Float32(args[:wd]),
        args[:batchsize],
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:hidden_size],
        args[:stack_size],
        args[:scaler],
        args[:init_scale],
        args[:MLE_tree_split],
        rng,
        device,
        args[:gpuID],
    )

    return config
end


mutable struct NeuroTreeClassifier <: MMI.Probabilistic
    loss::Symbol
    metric::Symbol
    nrounds::Int
    early_stopping_rounds::Int
    lr::Float32
    wd::Float32
    batchsize::Int
    actA::Symbol
    depth::Int
    ntrees::Int
    hidden_size::Int
    stack_size::Int
    scaler::Bool
    init_scale::Float32
    MLE_tree_split::Bool
    rng::AbstractRNG
    device::Symbol
    gpuID::Int
end

"""
    NeuroTreeClassifier(; kwargs...)

A model type for constructing a NeuroTreeClassifier, based on [NeuroTreeModels.jl](https://github.com/Evovest/NeuroTreeModels.jl), and implementing both an internal API and the MLJ model interface.

# Hyper-parameters

- `nrounds=100`:             Max number of rounds (epochs).
- `lr=1.0f-2`:              Learning rate. Must be > 0. A lower `eta` results in slower learning, typically requiring a higher `nrounds`.   
- `wd=0.f0`:                Weight decay applied to the gradients by the optimizer.
- `batchsize=2048`:         Batch size.
- `actA=:tanh`:             Activation function applied to each of input variable for determination of split node weight. Can be one of:
    - `:tanh`
    - `:identity`
- `depth=6`:            Depth of a tree. Must be >= 1. A tree of depth 1 has 2 prediction leaf nodes. A complete tree of depth N contains `2^N` terminal leaves and `2^N - 1` split nodes.
  Compute cost is proportional to `2^depth`. Typical optimal values are in the 3 to 5 range.
- `ntrees=64`:              Number of trees (per stack).
- `hidden_size=16`:         Size of hidden layers. Applicable only when `stack_size` > 1.
- `stack_size=1`:           Number of stacked NeuroTree blocks.
- `scaler=true`:            Whether a learnable scaling factor, prior to the sigmoid activation, should be used. Otherwise a fixed scaling of 1.0 is used if `scaler=false`.  
- `init_scale=0.1`:         Scaling factor applied to the predictions weights. Values in the range `]0, 1]` should result in best convergence. 
- `MLE_tree_split=false`:   Whether independent models are buillt for each of the 2 parameters (mu, sigma) of the the `gaussian_mle` loss.
- `rng=123`:                Either an integer used as a seed to the random number generator or an actual random number generator (`::Random.AbstractRNG`).
- `device=:cpu`:            Device on which to perform the computation, either `:cpu` or `:gpu`
- `gpuID=0`:                GPU device to use, only relveant if `device = :gpu` 

# Internal API

Do `config = NeuroTreeClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTreeClassifier(depth=5, ...)`.

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
NeuroTreeClassifier = @load NeuroTreeClassifier pkg=NeuroTreeModels
```

Do `model = NeuroTreeClassifier()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `NeuroTreeClassifier(loss=...)`.

## Training model

In MLJ or MLJBase, bind an instance `model` to data with
    `mach = machine(model, X, y)` where
- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Finite`; check the scitype
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
using NeuroTreeModels, DataFrames, CategoricalArrays, Random 
config = NeuroTreeClassifier(depth=5, nrounds=10)
nobs, nfeats = 1_000, 5
dtrain = DataFrame(randn(nobs, nfeats), :auto)
dtrain.y = categorical(rand(1:2, nobs))
feature_names, target_name = names(dtrain, r"x"), "y"
m = fit(config, dtrain; feature_names, target_name)
p = m(dtrain)
```

## MLJ Interface

```julia
using MLJBase, NeuroTreeModels
m = NeuroTreeClassifier(depth=5, nrounds=10)
X, y = @load_crabs
mach = machine(m, X, y) |> fit!
p = predict(mach, X)
```
"""
function NeuroTreeClassifier(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :nrounds => 100,
        :early_stopping_rounds => typemax(Int),
        :lr => 1.0f-2,
        :wd => 0.0f0,
        :batchsize => 2048,
        :actA => :tanh,
        :depth => 4,
        :ntrees => 64,
        :hidden_size => 1,
        :stack_size => 1,
        :scaler => true,
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

    loss = :mlogloss
    metric = :mlogloss

    rng = mk_rng(args[:rng])
    device = Symbol(args[:device])

    config = NeuroTreeClassifier(
        loss,
        metric,
        args[:nrounds],
        args[:early_stopping_rounds],
        Float32(args[:lr]),
        Float32(args[:wd]),
        args[:batchsize],
        Symbol(args[:actA]),
        args[:depth],
        args[:ntrees],
        args[:hidden_size],
        args[:stack_size],
        args[:scaler],
        args[:init_scale],
        args[:MLE_tree_split],
        rng,
        device,
        args[:gpuID],
    )

    return config
end

const NeuroTypes = Union{NeuroTreeRegressor,NeuroTreeClassifier}
get_loss_type(config::NeuroTypes) = _loss_type_dict[config.loss]
