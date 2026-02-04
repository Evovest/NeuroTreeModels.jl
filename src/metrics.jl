module Metrics

export metric_dict, is_maximise

import Statistics: mean, std
import Flux: Chain, logσ, logsoftmax, onehotbatch

"""
    mse(x, y; agg=mean)
    mse(x, y, w; agg=mean)
    mse(x, y, w, offset; agg=mean)
"""
function mse(m, x, y; agg = mean)
    metric = agg((m(x) .- y) .^ 2)
    return metric
end
function mse(m, x, y, w; agg = mean)
    metric = agg((m(x) .- y) .^ 2 .* w)
    return metric
end
function mse(m, x, y, w, offset; agg = mean)
    metric = agg((m(x) .+ offset .- y) .^ 2 .* w)
    return metric
end

"""
    mae(x, y; agg=mean)
    mae(x, y, w; agg=mean)
    mae(x, y, w, offset; agg=mean)
"""
function mae(m, x, y; agg = mean)
    metric = agg(abs.(m(x) .- y))
    return metric
end
function mae(m, x, y, w; agg = mean)
    metric = agg(abs.(m(x) .- y) .* w)
    return metric
end
function mae(m, x, y, w, offset; agg = mean)
    metric = agg(abs.(m(x) .+ offset .- y) .* w)
    return metric
end


"""
    logloss(x, y; agg=mean)
    logloss(x, y, w; agg=mean)
    logloss(x, y, w, offset; agg=mean)
"""
function logloss(m, x, y; agg = mean)
    p = m(x)
    metric = agg((1 .- y) .* p .- logσ.(p))
    return metric
end
function logloss(m, x, y, w; agg = mean)
    p = m(x)
    metric = agg(((1 .- y) .* p .- logσ.(p)) .* w)
    return metric
end
function logloss(m, x, y, w, offset; agg = mean)
    p = m(x) .+ offset
    metric = agg(((1 .- y) .* p .- logσ.(p)) .* w)
    return metric
end


"""
    tweedie(x, y; agg=mean)
    tweedie(x, y, w; agg=mean)
    tweedie(x, y, w, offset; agg=mean)
"""
function tweedie(m, x, y; agg = mean)
    rho = eltype(x)(1.5)
    p = exp.(m(x))
    agg(
        2 .* (
            y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* p .^ (1 - rho) / (1 - rho) +
            p .^ (2 - rho) / (2 - rho)
        ),
    )
end
function tweedie(m, x, y, w)
    agg = mean
    rho = eltype(x)(1.5)
    p = exp.(m(x))
    agg(
        w .* 2 .* (
            y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* p .^ (1 - rho) / (1 - rho) +
            p .^ (2 - rho) / (2 - rho)
        ),
    )
end
function tweedie(m, x, y, w, offset; agg = mean)
    rho = eltype(x)(1.5)
    p = exp.(m(x) .+ offset)
    agg(
        w .* 2 .* (
            y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* p .^ (1 - rho) / (1 - rho) +
            p .^ (2 - rho) / (2 - rho)
        ),
    )
end

"""
    mlogloss(x, y; agg=mean)
    mlogloss(x, y, w; agg=mean)
    mlogloss(x, y, w, offset; agg=mean)
"""
function mlogloss(m, x, y; agg = mean)
    p = logsoftmax(m(x); dims = 1)
    k = size(p, 1)
    raw = dropdims(-sum(onehotbatch(y, 1:k) .* p; dims = 1); dims = 1)
    metric = agg(raw)
    return metric
end
function mlogloss(m, x, y, w; agg = mean)
    p = logsoftmax(m(x); dims = 1)
    k = size(p, 1)
    raw = dropdims(-sum(onehotbatch(y, 1:k) .* p; dims = 1); dims = 1)
    metric = agg(raw .* w)
    return metric
end
function mlogloss(m, x, y, w, offset; agg = mean)
    p = logsoftmax(m(x) .+ offset; dims = 1)
    k = size(p, 1)
    raw = dropdims(-sum(onehotbatch(y, 1:k) .* p; dims = 1); dims = 1)
    metric = agg(raw .* w)
    return metric
end


"""
    gaussian_mle(μ::T, σ::T, y::T, w::T) where {T<:AbstractFloat}
"""
gaussian_mle(μ::T, σ::T, y::T) where {T<:AbstractFloat} =
    -σ - (y - μ)^2 / (2 * max(T(2.0f-7), exp(2 * σ)))

gaussian_mle(μ::T, σ::T, y::T, w::T) where {T<:AbstractFloat} =
    (-σ - (y - μ)^2 / (2 * max(T(2.0f-7), exp(2 * σ)))) * w

""""
    gaussian_mle(x, y; agg=mean)
    gaussian_mle(x, y, w; agg=mean)
    gaussian_mle(x, y, w, offset; agg=mean)
"""
function gaussian_mle(m, x, y; agg = mean)
    p = m(x)
    metric = agg(gaussian_mle.(view(p, 1, :), view(p, 2, :), y))
    return metric
end
function gaussian_mle(m, x, y, w; agg = mean)
    p = m(x)
    metric = agg(gaussian_mle.(view(p, 1, :), view(p, 2, :), y, w))
    return metric
end
function gaussian_mle(m, x, y, w, offset; agg = mean)
    p = m(x) .+ offset
    metric = agg(gaussian_mle.(view(p, 1, :), view(p, 2, :), y, w))
    return metric
end

function evaluate(m, f::Function, data)
    metric = 0.0f0
    ws = 0.0f0
    for d in data
        metric += f(m, d...; agg = sum)
        if length(d) >= 3
            ws += sum(d[3])
        else
            ws += last(size(d[2]))
        end
    end
    metric = metric / ws
    return metric
end

const metric_dict = Dict(
    :mse => mse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :gaussian_mle => gaussian_mle,
    :tweedie => tweedie,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(tweedie)) = false

end
