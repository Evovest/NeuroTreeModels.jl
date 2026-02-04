"""
    leaf_weights!(cw, nw)

Compute the cumulative probability associated with each node, down to terminal leaves
"""
function leaf_weights!(nw)
    cw = ones(eltype(nw), 2 * size(nw, 1) + 1, size(nw)[2:3]...)
    @threads for batch in axes(nw, 3)
        @inbounds for tree in axes(nw, 2)
            for i = 2:2:size(cw, 1)
                # child cumulative probability is obtained from the product of the parent cumulative prob (cw[parent]) and the parent probability (np[parent])
                cw[i, tree, batch] = cw[i>>1, tree, batch] * nw[i>>1, tree, batch]
                cw[i+1, tree, batch] = cw[i>>1, tree, batch] * (1 - nw[i>>1, tree, batch])
            end
        end
    end
    @views lw = cw[(size(nw, 1)+1):size(cw, 1), :, 1:size(nw, 3)]
    return (cw, lw)
end

"""
    leaf_weights!(cw::AnyCuArray, nw::AnyCuArray)

Compute the cumulative probability associated with each node, down to terminal leaves. 
"""
function leaf_weights!(nw::AnyCuArray)
    cw = CUDA.ones(eltype(nw), 2 * size(nw, 1) + 1, size(nw)[2:3]...)
    blocks = size(nw, 3)
    threads = size(nw, 2)
    @cuda threads = threads blocks = blocks leaf_weights!_kernel!(cw, nw)
    CUDA.synchronize()
    @views lw = cw[(size(nw, 1)+1):size(cw, 1), :, 1:size(nw, 3)]
    return (cw, lw)
end

function leaf_weights!_kernel!(cw::CuDeviceArray, nw::CuDeviceArray)

    tree = threadIdx().x # one thread per tree
    batch = blockIdx().x # one block per batch

    @inbounds for i = 2:2:size(cw, 1)
        # child cumulative probability is obtained from the product of the parent cumulative prob (cw[parent]) and the parent probability (np[parent])
        parent = cw[i>>1, tree, batch]
        child = nw[i>>1, tree, batch]
        cw[i, tree, batch] = parent * child
        cw[i+1, tree, batch] = parent * (1 - child)
    end
    return nothing
end

"""
    rrule(::leaf_weights!, cw, nw)

Backpropagation rule for the leaf probability calculation function
"""
function rrule(::typeof(leaf_weights!), nw)
    cw, lw = leaf_weights!(nw)
    max_depth = floor(Int, log2(size(nw, 1) + 1))
    node_offset = size(nw, 1) # offset on the leaf row
    leaf_weights!_pullback(ȳ) =
        NoTangent(), Δ_leaf_weights!(unthunk(ȳ[2]), cw, nw, max_depth, node_offset)
    return (cw, lw), leaf_weights!_pullback
end

"""
    Δ_leaf_weights!(Δnw, ȳ, cw, nw, max_depth, node_offset)

Kernel launcher for backpropagation rule of the leaf probability calculation
"""
function Δ_leaf_weights!(ȳ, cw, nw, max_depth, node_offset)
    Δnw = zeros(eltype(nw), size(nw)...)
    Δ_leaf_weights_kernel!(Δnw, ȳ, cw, nw, max_depth, node_offset)
    return Δnw
end
function Δ_leaf_weights!(ȳ, cw::AnyCuArray, nw::AnyCuArray, max_depth, node_offset)
    Δnw = CUDA.zeros(eltype(nw), size(nw)...)
    blocks = size(nw, 3)
    threads = size(nw, 2)
    @cuda threads = threads blocks = blocks Δ_leaf_weights_kernel!(
        Δnw,
        ȳ,
        cw,
        nw,
        max_depth,
        node_offset,
    )
    CUDA.synchronize()
    return Δnw
end

"""
    leaf_weights!_pullback_kernel!(nw̄, ȳ, cw, nw, max_depth, node_offset)

Kernel for backpropagation rule of the leaf probability calculation
"""
function Δ_leaf_weights_kernel!(Δnw, ȳ, cw, nw, max_depth, node_offset)

    @threads for batch in axes(nw, 3)
        @inbounds for tree in axes(nw, 2)
            # loop on node weights
            for i in axes(nw, 1)
                depth = floor(Int, log2(i)) # current depth level - starting at 0
                step = 2^(max_depth - depth) # iteration length
                leaf_offset = step * (i - 2^depth) # offset on the leaf row
                # loop on node weight leaf dependencies - half positive + half negative 
                for j = (1+leaf_offset):(step÷2+leaf_offset)
                    k = j + node_offset # move from leaf  position to full tree position 
                    Δnw[i, tree, batch] +=
                        ȳ[j, tree, batch] * cw[k, tree, batch] /
                        max(1.0f-8, nw[i, tree, batch])
                end
                for j = (1+leaf_offset+step÷2):(step+leaf_offset)
                    k = j + node_offset
                    Δnw[i, tree, batch] -=
                        ȳ[j, tree, batch] * cw[k, tree, batch] /
                        max(1.0f-8, (1 - nw[i, tree, batch]))
                end
            end
        end
    end
    return nothing
end

function Δ_leaf_weights_kernel!(
    Δnw::CuDeviceArray,
    ȳ,
    cw::CuDeviceArray,
    nw::CuDeviceArray,
    max_depth,
    node_offset,
)

    tree = threadIdx().x # one thread per tree
    batch = blockIdx().x # one block per batch

    @inbounds for i in axes(nw, 1)
        depth = floor(Int, log2(i)) # current depth level - starting at 0
        step = 2^(max_depth - depth) # iteration length
        leaf_offset = step * (i - 2^depth) # offset on the leaf row
        child = nw[i, tree, batch]
        # loop on node weight leaf dependencies - half positive + half negative 
        @inbounds for j in ((1+leaf_offset):(step÷2+leaf_offset))
            k = j + node_offset # move from leaf  position to full tree position
            Δnw[i, tree, batch] +=
                ȳ[j, tree, batch] * cw[k, tree, batch] / max(1.0f-8, child)
        end
        @inbounds for j = (1+leaf_offset+step÷2):(step+leaf_offset)
            k = j + node_offset
            Δnw[i, tree, batch] -=
                ȳ[j, tree, batch] * cw[k, tree, batch] / max(1.0f-8, (1 - child))
        end
    end
    return nothing
end
