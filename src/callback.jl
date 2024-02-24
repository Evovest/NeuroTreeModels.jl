module CallBacks

using DataFrames
using Statistics: mean, median
using Flux: cpu, gpu
using CUDA: CuIterator
using ..NeuroTreeModels: NeuroTreeRegressor
using ..NeuroTreeModels: get_df_loader_train
using ..NeuroTreeModels.Metrics

export CallBack, init_logger, update_logger!, agg_logger

struct CallBack{F,D}
    feval::F
    deval::D
end

function (cb::CallBack)(logger, iter, m)
    metric = Metrics.eval(m, cb.feval, cb.deval)
    update_logger!(logger; iter, metric)
    return nothing
end

# function CallBack(config::NeuroTreeRegressor; metric, x_eval, y_eval, w_eval=nothing, offset_eval=nothing)
#     feval = metric_dict[metric]

#     y_eval = ndims(y_eval) == 1 ? y_eval : y_eval'
#     w_eval = isnothing(w_eval) ? ones(Float32, size(y_eval)[end]) : Vector{Float32}(w_eval)
#     offset_eval = isnothing(offset_eval) ? zeros(Float32, size(y_eval)[end]) : Vector{Float32}(offset_eval)

#     deval = DataLoader(
#         (
#             x=Matrix{Float32}(x_eval'),
#             y=Float32.(y_eval),
#             w=Float32.(w_eval),
#             offset = Float32.(offset_eval),
#         ),
#         batchsize=config.batchsize,
#         partial=true,
#         shuffle=false,
#         parallel=true,
#         buffer=false,
#     )
#     (config.device == :gpu) && (deval = CuIterator(deval))
#     return CallBack(feval, deval)
# end

function CallBack(
    config::NeuroTreeRegressor,
    deval::AbstractDataFrame;
    metric,
    feature_names,
    target_name,
    weight_name=nothing,
    offset_name=nothing)

    batchsize = config.batchsize
    feval = metric_dict[metric]
    deval = get_df_loader_train(deval; feature_names, target_name, weight_name, offset_name, batchsize)
    (config.device == :gpu) && (deval = CuIterator(deval))
    return CallBack(feval, deval)
end

function init_logger(; metric, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => is_maximise(metric_dict[metric]),
        :early_stopping_rounds => early_stopping_rounds,
        :nrounds => 0,
        :metrics => (iter=Int[], metric=Float64[]),
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
            logger[:iter_since_best] += logger[:metrics][:iter][end] - logger[:metrics][:iter][end-1]
        end
    end
end

function agg_logger(logger_raw::Vector{Dict})

    _l1 = first(logger_raw)
    best_iters =  [d[:best_iter] for d in logger_raw]
    best_iter =  ceil(Int, median(best_iters))

    best_metrics =  [d[:best_metric] for d in logger_raw]
    best_metric =  last(best_metrics)

    metrics = (layer=Int[], iter=Int[], metric=Float64[])
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