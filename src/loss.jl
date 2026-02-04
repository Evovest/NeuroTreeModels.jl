function mse(m, x, y)
    mean((m(x) .- y) .^ 2)
end
function mse(m, x, y, w)
    sum((m(x) .- y) .^ 2 .* w) / sum(w)
end
function mse(m, x, y, w, offset)
    sum((m(x) .+ offset .- y) .^ 2 .* w) / sum(w)
end

function mae(m, x, y)
    mean(abs.(m(x) .- y))
end
function mae(m, x, y, w)
    sum(abs.(m(x) .- y) .* w) / sum(w)
end
function mae(m, x, y, w, offset)
    sum(abs.(m(x) .+ offset .- y) .* w) / sum(w)
end

function logloss(m, x, y)
    p = m(x)
    mean((1 .- y) .* p .- logσ.(p))
end
function logloss(m, x, y, w)
    p = m(x)
    sum(w .* ((1 .- y) .* p .- logσ.(p))) / sum(w)
end
function logloss(m, x, y, w, offset)
    p = m(x) .+ offset
    sum(w .* ((1 .- y) .* p .- logσ.(p))) / sum(w)
end

function tweedie(m, x, y)
    rho = eltype(x)(1.5)
    p = exp.(m(x))
    mean(
        2 .* (
            y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* p .^ (1 - rho) / (1 - rho) +
            p .^ (2 - rho) / (2 - rho)
        ),
    )
end
function tweedie(m, x, y, w)
    rho = eltype(x)(1.5)
    p = exp.(m(x))
    sum(
        w .* 2 .* (
            y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* p .^ (1 - rho) / (1 - rho) +
            p .^ (2 - rho) / (2 - rho)
        ),
    ) / sum(w)
end
function tweedie(m, x, y, w, offset)
    rho = eltype(x)(1.5)
    p = exp.(m(x) .+ offset)
    sum(
        w .* 2 .* (
            y .^ (2 - rho) / (1 - rho) / (2 - rho) - y .* p .^ (1 - rho) / (1 - rho) +
            p .^ (2 - rho) / (2 - rho)
        ),
    ) / sum(w)
end

function mlogloss(m, x, y)
    p = logsoftmax(m(x); dims = 1)
    k = size(p, 1)
    mean(-sum(onehotbatch(y, 1:k) .* p; dims = 1))
end
function mlogloss(m, x, y, w)
    p = logsoftmax(m(x); dims = 1)
    k = size(p, 1)
    sum(-sum(onehotbatch(y, 1:k) .* p; dims = 1) .* w) / sum(w)
end
function mlogloss(m, x, y, w, offset)
    p = logsoftmax(m(x) .+ offset; dims = 1)
    k = size(p, 1)
    sum(-sum(onehotbatch(y, 1:k) .* p; dims = 1) .* w) / sum(w)
end

gaussian_mle_loss(
    μ::AbstractVector{T},
    σ::AbstractVector{T},
    y::AbstractVector{T},
) where {T} = -sum(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ))))
gaussian_mle_loss(
    μ::AbstractVector{T},
    σ::AbstractVector{T},
    y::AbstractVector{T},
    w::AbstractVector{T},
) where {T} =
    -sum((-σ .- (y .- μ) .^ 2 ./ (2 .* max.(T(2e-7), exp.(2 .* σ)))) .* w) / sum(w)

function gaussian_mle(m, x, y)
    p = m(x)
    gaussian_mle_loss(view(p, 1, :), view(p, 2, :), y)
end
function gaussian_mle(m, x, y, w)
    p = m(x)
    gaussian_mle_loss(view(p, 1, :), view(p, 2, :), y, w)
end
function gaussian_mle(m, x, y, w, offset)
    p = m(x) .+ offset
    gaussian_mle_loss(view(p, 1, :), view(p, 2, :), y, w)
end

const _loss_fn_dict = Dict(
    :mse => mse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :gaussian_mle => gaussian_mle,
    :tweedie => tweedie,
)

get_loss_fn(config::NeuroTypes) = _loss_fn_dict[config.loss]
