import Base: length, getindex

"""
    ContainerTrain

"""
struct ContainerTrain{D<:AbstractDataFrame}
    df::D
    feature_names::Vector{Symbol}
    target_name::String
    weight_name::Union{Symbol,Nothing}
    offset_name::Union{Symbol,Vector{Symbol},Nothing}
end

function ContainerTrain(
    df;
    feature_names::Vector{Symbol},
    target_name,
    weight_name=nothing,
    offset_name=nothing)

    container = ContainerTrain(
        df,
        feature_names,
        target_name,
        weight_name,
        offset_name)

    return container
end

length(data::ContainerTrain{<:AbstractDataFrame}) = nrow(data.df)

function getindex(data::ContainerTrain{<:AbstractDataFrame}, idx::AbstractVector)
    df = view(data.df, idx, :)
    x = Matrix{Float32}(Matrix{Float32}(select(df, data.feature_names))')
    y = Float32.(df[!, data.target_name])
    if isnothing(data.weight_name) && isnothing(data.offset_name)
        return (x, y)
    elseif isnothing(data.offset_name)
        w = Float32.(df[!, data.weight_name])
        return (x, y, w)
    elseif isnothing(data.weight_name)
        w = ones(Float32, length(y))
        isa(data.offset_name, String) ? offset = Float32.(df[!, data.offset_name]) : offset = Matrix{Float32}(Matrix{Float32}(df[!, data.offset_name])')
        return (x, y, w, offset)
    else
        w = Float32.(df[!, data.weight_name])
        isa(data.offset_name, String) ? offset = Float32.(df[!, data.offset_name]) : offset = Matrix{Float32}(Matrix{Float32}(df[!, data.offset_name])')
        return (x, y, w, offset)
    end
end

function get_df_loader_train(
    df::AbstractDataFrame;
    feature_names::Vector{Symbol},
    target_name,
    weight_name=nothing,
    offset_name=nothing,
    batchsize,
    shuffle=true,
    device=:cpu)

    container = ContainerTrain(df; feature_names, target_name, weight_name, offset_name)
    batchsize = min(batchsize, length(container))
    dtrain = DataLoader(container; shuffle, batchsize, partial=true, parallel=false)
    if Symbol(device) == :gpu
        return CuIterator(dtrain)
    else
        return dtrain
    end
end


"""
    ContainerInfer

"""
struct ContainerInfer{D<:AbstractDataFrame}
    df::D
    feature_names::Vector{Symbol}
    offset_name::Union{Symbol,Nothing}
end

function ContainerInfer(
    df;
    feature_names::Vector{Symbol},
    offset_name=nothing)

    container = ContainerInfer(
        df,
        feature_names,
        offset_name)

    return container
end

length(data::ContainerInfer{<:AbstractDataFrame}) = nrow(data.df)

function getindex(data::ContainerInfer{<:AbstractDataFrame}, idx::AbstractVector)
    df = view(data.df, idx, :)
    x = Matrix{Float32}(Matrix{Float32}(select(df, data.feature_names))')
    return x
end

function get_df_loader_infer(
    df::AbstractDataFrame;
    feature_names,
    offset_name=nothing,
    batchsize,
    device=:cpu)

    feature_names = Symbol.(feature_names)
    container = ContainerInfer(df; feature_names, offset_name)
    batchsize = min(batchsize, length(container))
    dinfer = DataLoader(container; shuffle=false, batchsize, partial=true, parallel=false)
    if Symbol(device) == :gpu
        return CuIterator(dinfer)
    else
        return dinfer
    end
end
