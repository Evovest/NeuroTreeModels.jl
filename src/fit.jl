function init(
    config::NeuroTreeRegressor,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing)

    batchsize = config.batchsize
    feature_names = Symbol.(feature_names)
    if config.device == :gpu
        device = Flux.gpu
        CUDA.device!(config.gpuID)
    else
        device = Flux.cpu
    end

    dtrain = NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize)
    (config.device == :gpu) && (dtrain = CuIterator(dtrain))

    nfeats = length(feature_names)
    loss = get_loss_fn(config)
    L = get_loss_type(config)
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

    cache = (dtrain=dtrain, loss=loss, opts=opts, device=device, info=info)
    return m, cache
end


"""
    function fit(
        config::NeuroTreeRegressor,
        dtrain;
        feature_names,
        target_name,
        weight_name=nothing,
        offset_name=nothing,
        deval=nothing,
        metric=nothing,
        print_every_n=9999,
        early_stopping_rounds=9999,
        verbosity=1,
        return_logger=false
    )

Training function of NeuroTreeModels' internal API.

# Arguments

- `config::NeuroTreeRegressor`
- `dtrain`: Must be a `AbstractDataFrame`  

# Keyword arguments

- `feature_names`:          Required kwarg, a `Vector{Symbol}` or `Vector{String}` of the feature names.
- `target_name`             Required kwarg, a `Symbol` or `String` indicating the name of the target variable.  
- `weight_name=nothing`
- `offset_name=nothing`
- `deval=nothing`           Data for tracking evaluation metric and perform early stopping.
- `metric=nothing`: evaluation metric tracked on `deval`. Can be one of:
    - `:mse`
    - `:mae`
    - `:logloss`
    - `:mlogloss`
    - `:gaussian_mle`
- `print_every_n=9999`
- `early_stopping_rounds=9999`
- `verbosity=1`
- `return_logger=false`

"""
function fit(
    config::NeuroTreeRegressor,
    dtrain;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    deval=nothing,
    metric=nothing,
    print_every_n=9999,
    early_stopping_rounds=9999,
    verbosity=1,
    return_logger=false
)

    feature_names = Symbol.(feature_names)
    if config.device == :gpu
        CUDA.device!(config.gpuID)
    end

    # initialize callback and logger if tracking eval data
    metric = isnothing(metric) ? nothing : Symbol(metric)
    logging_flag = !isnothing(metric) && !isnothing(deval)
    any_flag = !isnothing(metric) || !isnothing(deval)
    if !logging_flag && any_flag
        @warn "For logger and eval metric to be tracked, `metric` and `deval` must both be provided."
    end
    logger = Dict[]
    logger = nothing

    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name)

    if logging_flag
        cb = CallBack(config, deval; metric, feature_names, target_name, weight_name, offset_name)
        logger = init_logger(; metric, early_stopping_rounds)
        cb(logger, 0, m)
        (verbosity > 0) && @info "Init training" metric = logger[:metrics][end]
    else
        (verbosity > 0) && @info "Init training"
    end

    # for iter = 1:config.nrounds
    while m.info[:nrounds] < config.nrounds
        fit_iter!(m, cache)
        iter = m.info[:nrounds]
        if !isnothing(logger)
            cb(logger, iter, m)
            if verbosity > 0 && iter % print_every_n == 0
                @info "iter $iter" metric = logger[:metrics][:metric][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end

    if return_logger
        return (m, logger)
    else
        return m
    end
end

function fit_iter!(m, cache)
    loss, opts, data = cache[:loss], cache[:opts], cache[:dtrain]
    GC.gc(true)
    if m.info[:device] == :gpu
        CUDA.reclaim()
    end
    for d in data
        grads = gradient(model -> loss(model, d...), m)[1]
        Optimisers.update!(opts, m, grads)
    end
    m.info[:nrounds] += 1
    return nothing
end
