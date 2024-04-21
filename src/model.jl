
struct NeuroTree{outsize,M,V}
    outsize::Int
    w::M
    b::V
    p::M
    actA::Function
end
NeuroTree(outsize::Int, w::M, b::V, p::M, actA::Function) where {M,V} = NeuroTree{outsize,M,V}(outsize, w, b, p, actA)

@functor NeuroTree
# Flux.trainable(m::NeuroTree) = (w=m.w, b=m.b, p=m.p)

function node_weights(m::NeuroTree, x)
    # [N X T, F] * [F, B] => [N x T, B]
    # nw = sigmoid_fast.(m.w * x .+ m.b)
    nw = sigmoid_fast.(m.actA.(m.w) * x .+ m.b)
    # [N x T, B] -> [N, T, B]
    return reshape(nw, :, size(m.p, 2), size(x, 2))
end

include("leaf_weights.jl")

# function (m::NeuroTree{W,B,P})(x::W) where {W,B,P}
#     # [F, B] -> [N, T, B]
#     nw = node_weights(m, x)
#     # [N, T, B] -> [L, T, B]
#     (_, lw) = leaf_weights!(nw)
#     # [L, T, B], [P, L, T] -> [P, B]
#     pred = dot_prod_agg(lw, m.p) ./ size(m.p, 3)
#     return pred
# end

# dot_prod_agg(lw, p) = dropdims(sum(reshape(lw, 1, size(lw)...) .* p, dims=(2, 3)), dims=(2, 3))

function (m::NeuroTree{N,M,V})(x::M) where {N,M,V}
    # [F, B] -> [N, T, B]
    nw = node_weights(m, x)
    # [N, T, B] -> [L, T, B]
    (_, lw) = leaf_weights!(nw)
    # [L, T, B], [L, T] -> [1, T, B]
    pred = sum(lw .* m.p, dims=1)
    # reshape to keep N outputs
    pred = reshape(pred, size(pred, 2) ÷ N, N, size(pred, 3))
    # [P, B] aggregate over trees per output
    pred = dropdims(mean(pred, dims=1), dims=1)
    return pred
end

# dot_prod_agg(lw, p) = dropdims(sum(lw .* p, dims=2), dims=2)

"""
    NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1.0)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1.0)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1.0)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    _ntrees_tot = outs * ntrees
    nt = NeuroTree(
        Int(outs),
        Flux.glorot_uniform(nnodes * _ntrees_tot, ins), # w
        zeros(Float32, nnodes * _ntrees_tot), # b
        Float32.((rand(nleaves, _ntrees_tot) .- 0.5) .* sqrt(12) .* init_scale), # p
        # Float32.(randn(outs, nleaves, _ntrees_tot) ./ 1 .* init_scale), # p
        actA,
    )
    return nt
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1.0)
    nnodes = 2^depth - 1
    nleaves = 2^depth
    _ntrees_tot = outs * ntrees
    nt = NeuroTree(
        Int(outs),
        Flux.glorot_uniform(nnodes * _ntrees_tot, ins), # w
        zeros(Float32, nnodes * _ntrees_tot), # b
        Float32.((rand(nleaves, _ntrees_tot) .- 0.5) .* sqrt(12) .* init_scale), # p
        # Float32.(randn(outs, nleaves, _ntrees_tot) ./ 1 .* init_scale), # p
        actA,
    )
    return nt
end

"""
    StackTree
A StackTree is made of a collection of NeuroTrees.
"""
struct StackTree
    trees::Vector{NeuroTree}
end
@functor StackTree

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, stack_size=2, hidden_size=8, actA=identity, init_scale=1.0)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; depth, ntrees, actA, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; depth, ntrees, actA, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; depth, ntrees, actA, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; depth, ntrees, actA, init_scale)
            push!(trees, tree)
        end
    end
    m = StackTree(trees)
    return m
end

function (m::StackTree)(x::AbstractMatrix)
    p = m.trees[1](x)
    for i in 2:length(m.trees)
        if i < length(m.trees)
            p = p .+ m.trees[i](p)
        else
            _p = m.trees[i](p)
            p = view(p, 1:size(_p, 1), :) .+ _p
        end
    end
    return p
end
# function (m::StackTree)(x::AbstractMatrix)
#     p = m.trees[1](x)
#     for i in 2:length(m.trees)
#         p = m.trees[i](p)
#     end
#     return p
# end

"""
    NeuroTreeModel
A NeuroTreeModel is made of a collection of Tree, either regular `NeuroTree` or `StackTree`.
Prediction is the sum of all the trees composing a NeuroTreeModel.
"""
struct NeuroTreeModel{L<:LossType,C<:Chain}
    _loss_type::Type{L}
    chain::C
    info::Dict{Symbol,Any}
end
@functor NeuroTreeModel

"""
    (m::NeuroTreeModel)(x::AbstractMatrix)
    (m::NeuroTreeModel)(data::AbstractDataFrame)

Inference for NeuroTreeModel
"""
function (m::NeuroTreeModel)(x::AbstractMatrix)
    p = m.chain(x)
    if size(p, 1) == 1
        p = dropdims(p; dims=1)
    end
    return p
end
function (m::NeuroTreeModel)(data::AbstractDataFrame)
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048, device=m.info[:device])
    p = infer(m, dinfer)
    return p
end

const _act_dict = Dict(
    :identity => identity,
    :tanh => tanh,
    :hardtanh => hardtanh,
    :sigmoid => sigmoid,
    :hardsigmoid => hardsigmoid
)

function get_model_chain(L; config, nfeats, outsize)

    if L <: GaussianMLE && config.MLE_tree_split
        chain = Chain(
            BatchNorm(nfeats),
            Parallel(
                vcat,
                StackTree(nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale),
                StackTree(nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=_act_dict[config.actA],
                    init_scale=config.init_scale)
            )
        )
    else
        outsize = L <: GaussianMLE ? 2 * outsize : outsize
        chain = Chain(
            BatchNorm(nfeats),
            StackTree(nfeats => outsize;
                depth=config.depth,
                ntrees=config.ntrees,
                stack_size=config.stack_size,
                hidden_size=config.hidden_size,
                actA=_act_dict[config.actA],
                init_scale=config.init_scale)
        )

    end

    return chain

end
