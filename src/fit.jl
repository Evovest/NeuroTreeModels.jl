function init(
    config::NeuroTypes,
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name = nothing,
    offset_name = nothing,
)

    device = config.device
    batchsize = config.batchsize
    nfeats = length(feature_names)
    loss = get_loss_fn(config)
    L = get_loss_type(config)

    target_levels = nothing
    target_isordered = false
    outsize = 1
    if L <: MLogLoss
        eltype(df[!, target_name]) <: CategoricalValue || error(
            "Target variable `$target_name` must have its elements `<: CategoricalValue`",
        )
        target_levels = CategoricalArrays.levels(df[!, target_name])
        target_isordered = isordered(df[!, target_name])
        outsize = length(target_levels)
    end
    dtrain = NeuroTreeModels.get_df_loader_train(
        df;
        feature_names,
        target_name,
        weight_name,
        offset_name,
        batchsize,
        device,
    )

    chain = get_model_chain(L; config, nfeats, outsize)
    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
    )
    m = NeuroTreeModel(L, chain, info)
    if device == :gpu
        m = m |> gpu
    end

    optim = OptimiserChain(NAdam(config.lr), WeightDecay(config.wd))
    opts = Optimisers.setup(optim, m)

    cache = (dtrain = dtrain, loss = loss, opts = opts, info = info)
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
        print_every_n=9999,
        verbosity=1,
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
- `print_every_n=9999`
- `verbosity=1`
"""
function fit(
    config::NeuroTypes,
    dtrain;
    feature_names,
    target_name,
    weight_name = nothing,
    offset_name = nothing,
    deval = nothing,
    print_every_n = 9999,
    verbosity = 1,
)

    device = Symbol(config.device)
    if device == :gpu
        CUDA.device!(config.gpuID)
    end

    feature_names = Symbol.(feature_names)
    target_name = Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    offset_name = isnothing(offset_name) ? nothing : Symbol(offset_name)

    m, cache = init(config, dtrain; feature_names, target_name, weight_name, offset_name)

    # initialize callback and logger if tracking eval data
    logger = nothing
    if !isnothing(deval)
        cb = CallBack(config, deval; feature_names, target_name, weight_name, offset_name)
        logger = init_logger(config)
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
    return m |> cpu
end

function fit_iter!(m, cache)
    loss, opts, data = cache[:loss], cache[:opts], cache[:dtrain]
    GC.gc(true)
    if typeof(cache[:dtrain]) <: CuIterator
        CUDA.reclaim()
    end
    for d in data
        grads = gradient(model -> loss(model, d...), m)[1]
        Optimisers.update!(opts, m, grads)
    end
    m.info[:nrounds] += 1
    return nothing
end
