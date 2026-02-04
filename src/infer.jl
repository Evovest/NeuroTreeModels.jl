"""
    DL

Union{CuIterator, DataLoader}
"""
const DL = Union{CuIterator,DataLoader}

"""
infer(m::NeuroTreeModel, data)

Return the inference of a `NeuroTreeModel` over `data`, where `data` is `AbstractDataFrame`.
"""
function infer(m::NeuroTreeModel, data::AbstractDataFrame)
    dinfer = get_df_loader_infer(
        data;
        feature_names = m.info[:feature_names],
        batchsize = 2048,
        device = m.info[:device],
    )
    p = infer(m, dinfer)
    return p
end

function infer(m::NeuroTreeModel{L}, data::DL) where {L<:Union{MSE,MAE}}
    preds = Vector{Float32}[]
    for x in data
        push!(preds, Vector(m(x)))
    end
    p = vcat(preds...)
    return p
end

function infer(m::NeuroTreeModel{<:LogLoss}, data::DL)
    preds = Vector{Float32}[]
    for x in data
        push!(preds, Vector(m(x)))
    end
    p = vcat(preds...)
    p .= sigmoid(p)
    return p
end

function infer(m::NeuroTreeModel{<:MLogLoss}, data::DL)
    preds = Matrix{Float32}[]
    for x in data
        push!(preds, Matrix(m(x)'))
    end
    p = vcat(preds...)
    softmax!(p; dims = 2)
    return p
end

function infer(m::NeuroTreeModel{<:GaussianMLE}, data::DL)
    preds = Matrix{Float32}[]
    for x in data
        push!(preds, Matrix(m(x)'))
    end
    p = vcat(preds...)
    p[:, 2] .= exp.(p[:, 2]) # reproject log(σ) into σ 
    return p
end

function infer(m::NeuroTreeModel{L}, data::DL) where {L<:Union{Tweedie}}
    preds = Vector{Float32}[]
    for x in data
        push!(preds, Vector(m(x)))
    end
    p = vcat(preds...)
    p .= exp.(p)
    return p
end
