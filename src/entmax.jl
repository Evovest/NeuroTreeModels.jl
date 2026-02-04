# Reference: https://github.com/deep-spin/entmax/blob/master/entmax/activations.py
one_to_vec(x) = reshape(vec(1:size(x, 2)), 1, :)
one_to_vec(x::AnyCuArray) = reshape(CUDA.CuArray(1:size(x, 2)), 1, :)

function entmax_threshold_and_support(x)

    rho = one_to_vec(x)
    x_sort = sort(x, dims = 2, rev = true)
    x_μ = cumsum(x_sort, dims = 2) ./ rho
    x_μ² = cumsum(x_sort .^ 2, dims = 2) ./ rho
    ss = rho .* (x_μ² .- x_μ .^ 2)
    delta = (1 .- ss) ./ rho

    delta_nz = max.(delta, 0)
    τ = x_μ .- sqrt.(delta_nz)

    support = dropdims(sum(x_sort .>= τ, dims = 2), dims = 2)
    idx = collect(zip(1:size(x, 1), support))
    τ′ = Flux.NNlib.gather(τ, idx) # !!! TO DO
    return τ′

end

"""
    entmax15(x)

WIP: entmax15 activation function for sparse feature selection. 
Doesn't improve performanceon YEAR dataset as stated in NODE paper.
"""
function entmax15(x)
    x_norm = (x .- maximum(x, dims = 2)) ./ 2
    τ′ = entmax_threshold_and_support(x_norm)
    out = max.(x_norm .- reshape(τ′, :, 1), 0) .^ 2
    return out
end

function rrule(::typeof(entmax15), x)
    out = entmax15(x)
    function entmax15_pullback(ȳ)
        Δ_entmax15(ȳ, out)
    end
    return out, entmax15_pullback
end

function Δ_entmax15(ȳ, out)
    gppr = sqrt.(out)  # = 1 / g'' (Y) 
    Δ = ȳ .* gppr
    q = sum(Δ, dims = 2) ./ sum(gppr, dims = 2)
    Δ = Δ .- q .* gppr
    return NoTangent(), Δ
end
