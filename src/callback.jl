module CallBacks

using DataFrames
using Statistics: mean, median
using Flux: cpu, gpu
using CUDA: CuIterator
using ..NeuroTreeModels: NeuroTypes
using ..NeuroTreeModels: get_df_loader_train
using ..NeuroTreeModels.Metrics

export CallBack, init_logger, update_logger!, agg_logger

struct CallBack{F,D}
    feval::F
    deval::D
end

function (cb::CallBack)(logger, iter, m)
    metric = Metrics.evaluate(m, cb.feval, cb.deval)
    update_logger!(logger; iter, metric)
    return nothing
end

function CallBack(
    config::NeuroTypes,
    deval::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name = nothing,
    offset_name = nothing,
)

    device = config.device
    batchsize = config.batchsize
    feval = metric_dict[config.metric]
    deval = get_df_loader_train(
        deval;
        feature_names,
        target_name,
        weight_name,
        offset_name,
        batchsize,
        device,
    )
    return CallBack(feval, deval)
end

function init_logger(config::NeuroTypes)
    logger = Dict(
        :name => String(config.metric),
        :maximise => is_maximise(metric_dict[config.metric]),
        :early_stopping_rounds => config.early_stopping_rounds,
        :nrounds => 0,
        :metrics => (iter = Int[], metric = Float64[]),
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger; iter, metric)
    logger[:nrounds] = iter
    push!(logger[:metrics][:iter], iter)
    push!(logger[:metrics][:metric], metric)
    if iter == 0
        logger[:best_metric] = metric
    else
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] +=
                logger[:metrics][:iter][end] - logger[:metrics][:iter][end-1]
        end
    end
end

function agg_logger(logger_raw::Vector{Dict})

    _l1 = first(logger_raw)
    best_iters = [d[:best_iter] for d in logger_raw]
    best_iter = ceil(Int, median(best_iters))

    best_metrics = [d[:best_metric] for d in logger_raw]
    best_metric = last(best_metrics)

    metrics = (layer = Int[], iter = Int[], metric = Float64[])
    for i in eachindex(logger_raw)
        _l = logger_raw[i]
        append!(metrics[:layer], zeros(Int, length(_l[:metrics][:iter])) .+ i)
        append!(metrics[:iter], _l[:metrics][:iter])
        append!(metrics[:metric], _l[:metrics][:metric])
    end

    logger = Dict(
        :name => _l1[:name],
        :maximise => _l1[:maximise],
        :early_stopping_rounds => _l1[:name],
        :metrics => metrics,
        :best_iters => best_iters,
        :best_iter => best_iter,
        :best_metrics => best_metrics,
        :best_metric => best_metric,
    )

    return logger
end

end
