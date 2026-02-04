function MMI.fit(model::NeuroTypes, verbosity::Int, A, y, w = nothing)

    Tables.istable(A) ? dtrain = DataFrame(A) : error("`A` must be a Table")
    feature_names = string.(collect(Tables.schema(dtrain).names))
    @assert "_target" ∉ feature_names
    dtrain._target = y
    target_name = "_target"

    if !isnothing(w)
        @assert "_weight" ∉ feature_names
        dtrain._weight = w
        weight_name = "_weight"
    else
        weight_name = nothing
    end
    offset_name = nothing

    fitresult, cache =
        init(model, dtrain; feature_names, target_name, weight_name, offset_name)

    while fitresult.info[:nrounds] < model.nrounds
        fit_iter!(fitresult, cache)
    end

    report = (features = fitresult.info[:feature_names],)
    return fitresult, cache, report
end

function okay_to_continue(model, fitresult, cache)
    return model.nrounds - fitresult.info[:nrounds] >= 0
end

# For EarlyStopping.jl support
MMI.iteration_parameter(::Type{<:NeuroTypes}) = :nrounds

function MMI.update(
    model::NeuroTypes,
    verbosity::Integer,
    fitresult,
    cache,
    A,
    y,
    w = nothing,
)
    if okay_to_continue(model, fitresult, cache)
        while fitresult.info[:nrounds] < model.nrounds
            fit_iter!(fitresult, cache)
        end
        report = (features = fitresult.info[:feature_names],)
    else
        fitresult, cache, report = fit(model, verbosity, A, y, w)
    end
    return fitresult, cache, report
end

function predict(::NeuroTreeRegressor, fitresult, A)
    df = DataFrame(A)
    Tables.istable(A) ? df = DataFrame(A) : error("`A` must be a Table")
    pred = fitresult(df)
    return pred
end

function predict(::NeuroTreeClassifier, fitresult, A)
    df = DataFrame(A)
    Tables.istable(A) ? df = DataFrame(A) : error("`A` must be a Table")
    pred = fitresult(df)
    return MMI.UnivariateFinite(
        fitresult.info[:target_levels],
        pred,
        pool = missing,
        ordered = fitresult.info[:target_isordered],
    )
end

# Metadata
MMI.metadata_pkg.(
    (NeuroTreeRegressor, NeuroTreeClassifier),
    name = "NeuroTreeModels",
    uuid = "1db4e0a5-a364-4b0c-897c-2bd5a4a3a1f2",
    url = "https://github.com/Evovest/NeuroTreeModels.jl",
    julia = true,
    license = "Apache",
    is_wrapper = false,
)

MMI.metadata_model(
    NeuroTreeRegressor,
    input_scitype = MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),
    target_scitype = AbstractVector{<:MMI.Continuous},
    weights = true,
    path = "NeuroTreeModels.NeuroTreeRegressor",
)

MMI.metadata_model(
    NeuroTreeClassifier,
    input_scitype = MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),
    target_scitype = AbstractVector{<:MMI.Finite},
    weights = true,
    path = "NeuroTreeModels.NeuroTreeClassifier",
)
