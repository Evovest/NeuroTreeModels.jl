function init(
    config::NeuroTypes,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    device=:cpu,
)

    batchsize = config.batchsize
    nfeats = length(feature_names)
    loss = get_loss_fn(config)
    L = get_loss_type(config)

    target_levels = nothing
    target_isordered = false
    outsize = 1
    if L <: MLogLoss
        eltype(df[!, target_name]) <: CategoricalValue || error("Target variable `$target_name` must have its elements `<: CategoricalValue`")
        target_levels = CategoricalArrays.levels(df[!, target_name])
        target_isordered = isordered(df[!, target_name])
        outsize = length(target_levels)
    end
    dtrain = NeuroTreeModels.get_df_loader_train(df; feature_names, target_name, weight_name, offset_name, batchsize, device)

    chain = get_model_chain(L; config, nfeats, outsize)
    info = Dict(
        :device => device,
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered)
    m = NeuroTreeModel(L, chain, info)
    if device == :gpu
        m = m |> gpu
    end

    optim = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    opts = Optimisers.setup(optim, m)

    cache = (dtrain=dtrain, loss=loss, opts=opts, info=info)
    return m, cache
end


"""
    function fit(
        config::NeuroTypes,
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
        device=:cpu,
        gpuID=0,
    )

Training function of NeuroTreeModels' internal API.

# Arguments

- `config::NeuroTypes`
- `dtrain`: Must be `<:AbstractDataFrame`  

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
- `device=:cpu`: device on which to perform the computation, either `:cpu` or `:gpu`
- `gpuID=0`: gpu device to use, only relveant if `device = :gpu` 

"""
function fit(
    config::NeuroTypes,
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
    device=:cpu,
    gpuID=0,
)

    device = Symbol(device)
    if device == :gpu
        CUDA.device!(gpuID)
    end

    feature_names = Symbol.(feature_names)
    target_name = Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    offset_name = isnothing(offset_name) ? nothing : Symbol(offset_name)
    metric = isnothing(metric) ? nothing : Symbol(metric)

    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name, device)

    # initialize callback and logger if tracking eval data
    logging_flag = !isnothing(metric) && !isnothing(deval)
    any_flag = !isnothing(metric) || !isnothing(deval)
    if !logging_flag && any_flag
        @warn "For logger and eval metric to be tracked, `metric` and `deval` must both be provided."
    end

    logger = nothing
    if logging_flag
        cb = CallBack(config, deval; metric, feature_names, target_name, weight_name, offset_name, device)
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

    m.info[:logger] = logger
    return m
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
