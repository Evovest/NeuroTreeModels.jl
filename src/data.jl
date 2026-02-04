import Base: length, getindex

"""
    ContainerTrain

"""
struct ContainerTrain{A<:AbstractMatrix,B<:AbstractVector,C,D}
    x::A
    y::B
    w::C
    offset::D
end

length(data::ContainerTrain) = size(data.x, 2)

function getindex(
    data::ContainerTrain{A,B,C,D},
    idx::AbstractVector,
) where {A,B,C<:Nothing,D<:Nothing}
    x = data.x[:, idx]
    y = data.y[idx]
    return (x, y)
end
function getindex(
    data::ContainerTrain{A,B,C,D},
    idx::AbstractVector,
) where {A,B,C<:AbstractVector,D<:Nothing}
    x = data.x[:, idx]
    y = data.y[idx]
    w = data.w[idx]
    return (x, y, w)
end
function getindex(
    data::ContainerTrain{A,B,C,D},
    idx::AbstractVector,
) where {A,B,C<:AbstractVector,D<:AbstractVector}
    x = data.x[:, idx]
    y = data.y[idx]
    w = data.w[idx]
    offset = data.offset[idx]
    return (x, y, w, offset)
end
function getindex(
    data::ContainerTrain{A,B,C,D},
    idx::AbstractVector,
) where {A,B,C<:AbstractVector,D<:AbstractMatrix}
    x = data.x[:, idx]
    y = data.y[idx]
    w = data.w[idx]
    offset = data.offset[:, idx]
    return (x, y, w, offset)
end


function get_df_loader_train(
    df::AbstractDataFrame;
    feature_names,
    target_name,
    weight_name = nothing,
    offset_name = nothing,
    batchsize,
    shuffle = true,
    device = :cpu,
)

    feature_names = Symbol.(feature_names)
    x = Matrix{Float32}(Matrix{Float32}(select(df, feature_names))')

    if eltype(df[!, target_name]) <: CategoricalValue
        y = UInt32.(CategoricalArrays.levelcode.(df[!, target_name]))
    else
        y = Float32.(df[!, target_name])
    end

    w = isnothing(weight_name) ? nothing : Float32.(df[!, weight_name])

    offset = if isnothing(offset_name)
        nothing
    else
        isa(offset_name, String) ? Float32.(df[!, offset_name]) :
        offset = Matrix{Float32}(Matrix{Float32}(df[!, data.offset_name])')
    end

    container = ContainerTrain(x, y, w, offset)
    batchsize = min(batchsize, length(container))
    dtrain = DataLoader(container; shuffle, batchsize, partial = true, parallel = false)
    if device == :gpu
        return CuIterator(dtrain)
    else
        return dtrain
    end
end


"""
    ContainerInfer

"""
struct ContainerInfer{A<:AbstractMatrix,D}
    x::A
    offset::D
end

length(data::ContainerInfer) = size(data.x, 2)

function getindex(data::ContainerInfer{A,D}, idx::AbstractVector) where {A,D<:Nothing}
    x = data.x[:, idx]
    return x
end
function getindex(
    data::ContainerTrain{A,D},
    idx::AbstractVector,
) where {A,D<:AbstractVector}
    x = data.x[:, idx]
    offset = data.offset[idx]
    return (x, offset)
end
function getindex(
    data::ContainerTrain{A,D},
    idx::AbstractVector,
) where {A,D<:AbstractMatrix}
    x = data.x[:, idx]
    offset = data.offset[:, idx]
    return (x, offset)
end

function get_df_loader_infer(
    df::AbstractDataFrame;
    feature_names,
    offset_name = nothing,
    batchsize,
    device = :cpu,
)

    feature_names = Symbol.(feature_names)
    x = Matrix{Float32}(Matrix{Float32}(select(df, feature_names))')

    offset = if isnothing(offset_name)
        nothing
    else
        isa(offset_name, String) ? Float32.(df[!, offset_name]) :
        offset = Matrix{Float32}(Matrix{Float32}(df[!, data.offset_name])')
    end

    container = ContainerInfer(x, offset)
    batchsize = min(batchsize, length(container))
    dinfer =
        DataLoader(container; shuffle = false, batchsize, partial = true, parallel = false)
    if device == :gpu
        return CuIterator(dinfer)
    else
        return dinfer
    end
end
