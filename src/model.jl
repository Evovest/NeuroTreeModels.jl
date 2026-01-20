
struct NeuroTree{DI,DP,M,P}
    d_in::DI
    d_proj::DP
    mask::M
    p::P
end
@layer NeuroTree trainable = (d_in, d_proj, p)

function (m::NeuroTree)(x)
    h = m.d_in(x) # [F,B] => [HNT,B]
    h = reshape(h, size(m.d_proj.weight, 2), :) # [HNT,B] => [H,NTB]
    nw = m.d_proj(h) # [H,NTB] => [1,NTB]
    nw = reshape(nw, size(m.mask, 1), :) # [1,NTB] => [N,TB]
    lw = softmax(m.mask' * nw) # [N,TB] => [L,TB]
    lw = reshape(lw, :, size(x, 2)) # [L,TB] => [LT,B]
    p = m.p * lw ./ 16 # [LT,B] => [P,B]
    return p
end
# function (m::NeuroTree)(x)
#     nw = m.d_in(x) # [F,B] => [NT,B]
#     nw = reshape(nw, 7, :) # [NT,B] => [N,TB]
#     lw = softmax(m.mask' * nw) # [N,TB] => [L,TB]
#     lw = reshape(lw, 8 * 16, :) # [L,TB] => [LT,B]
#     p = m.p * lw ./ 16 # [LT,B] => [P,B]
#     return p
# end

"""
    NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, init_scale=1e-1)
    NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, init_scale=1e-1)

Initialization of a NeuroTree.
"""
function NeuroTree(; ins, outs, depth=4, ntrees=64, actA=identity, scaler=true, init_scale=1e-1)
    h = 8
    layout = :binary
    mask = get_mask(Val(layout), depth)
    nnodes = size(mask, 1)
    nleaves = size(mask, 2)

    op = NeuroTree(
        Dense(ins => h * nnodes * ntrees, relu), # w
        # Dense(ins => nnodes * ntrees), # w
        Dense(h => 1), # s
        mask,
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
    )
    return op
end
function NeuroTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, actA=identity, scaler=true, init_scale=1e-1)
    h = 8
    layout = :binary
    mask = get_mask(Val(layout), depth)
    nnodes = size(mask, 1)
    nleaves = size(mask, 2)

    op = NeuroTree(
        Dense(ins => h * nnodes * ntrees, relu), # w
        # Dense(ins => nnodes * ntrees), # w
        Dense(h => 1), # s
        mask,
        Float32.(randn(outs, nleaves * ntrees) .* init_scale), # p
    )
    return op
end

function get_mask(::Val{:binary}, depth::Integer)
    nodes = 2^depth - 1
    leaves = 2^depth
    mask = zeros(Bool, nodes, leaves)

    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, 2^(d - 1) + b - 1, (b-1)*stride+1:(b-1)*stride+k,) .= true
        end
    end
    return mask
end

function get_mask(::Val{:oblivious}, depth::Integer)
    leaves = 2^depth
    mask = zeros(Bool, depth, leaves)

    for d in 1:depth
        blocks = 2^(d - 1)
        k = 2^(depth - d)
        stride = 2 * k
        for b in 1:blocks
            view(mask, d, (b-1)*stride+1:(b-1)*stride+k,) .= true
        end
    end
    return mask
end

"""
    StackTree
A StackTree is made of a collection of NeuroTrees.
"""
struct StackTree
    trees::Vector{NeuroTree}
end
@layer StackTree

function StackTree((ins, outs)::Pair{<:Integer,<:Integer}; depth=4, ntrees=64, stack_size=2, hidden_size=8, actA=identity, scaler=true, init_scale=1e-1)
    @assert stack_size == 1 || hidden_size >= outs
    trees = []
    for i in 1:stack_size
        if i == 1
            if i < stack_size
                tree = NeuroTree(ins => hidden_size; depth, ntrees, actA, scaler, init_scale)
                push!(trees, tree)
            else
                tree = NeuroTree(ins => outs; depth, ntrees, actA, scaler, init_scale)
                push!(trees, tree)
            end
        elseif i < stack_size
            tree = NeuroTree(hidden_size => hidden_size; depth, ntrees, actA, scaler, init_scale)
            push!(trees, tree)
        else
            tree = NeuroTree(hidden_size => outs; depth, ntrees, actA, scaler, init_scale)
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
@layer NeuroTreeModel

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
function (m::NeuroTreeModel)(data::AbstractDataFrame; device=:cpu, gpuID=0)
    if device == :gpu
        CUDA.device!(gpuID)
    end
    m = device == :cpu ? m |> cpu : m |> gpu
    dinfer = get_df_loader_infer(data; feature_names=m.info[:feature_names], batchsize=2048, device)
    p = infer(m, dinfer)
    return p
end


function _identity_act(x)
    return x ./ sum(abs.(x), dims=2)
end
function _tanh_act(x)
    x = Flux.tanh_fast.(x)
    return x ./ sum(abs.(x), dims=2)
end
function _hardtanh_act(x)
    x = Flux.hardtanh.(x)
    return x ./ sum(abs.(x), dims=2)
end

"""
    act_dict = Dict(
        :identity => _identity_act,
        :tanh => _tanh_act,
        :hardtanh => _hardtanh_act,
    )

Dictionary mapping features activation name to their function.
"""
const act_dict = Dict(
    :identity => _identity_act,
    :tanh => _tanh_act,
    :hardtanh => _hardtanh_act,
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
                    actA=act_dict[config.actA],
                    scaler=config.scaler,
                    init_scale=config.init_scale),
                StackTree(nfeats => outsize;
                    depth=config.depth,
                    ntrees=config.ntrees,
                    stack_size=config.stack_size,
                    hidden_size=config.hidden_size,
                    actA=act_dict[config.actA],
                    scaler=config.scaler,
                    init_scale=config.init_scale)
            )
        )
    else
        outsize = L <: GaussianMLE ? 2 * outsize : outsize
        chain = Chain(
            BatchNorm(nfeats),
            StackTree(
                nfeats => outsize;
                depth=config.depth,
                ntrees=config.ntrees,
                stack_size=config.stack_size,
                hidden_size=config.hidden_size,
                actA=act_dict[config.actA],
                scaler=config.scaler,
                init_scale=config.init_scale
            )
        )
    end

    return chain

end
