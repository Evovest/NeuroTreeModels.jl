using CSV
using DataFrames
using NeuroTreeModels
using StatsBase: sample
using Random: seed!
using EvoTrees

using AWS: AWSCredentials, AWSConfig, @service
@service S3

aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds = aws_creds, region = "ca-central-1")

path = "share/data/aicrowd/insurance-aicrowd.csv"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
df = DataFrame(CSV.File(raw))
transform!(df, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

target_name = "event"
feature_names = [
    "vh_age",
    "vh_value",
    "vh_speed",
    "vh_weight",
    "drv_age1",
    "pol_no_claims_discount",
    "pol_coverage",
    "pol_duration",
    "pol_sit_duration",
]

pol_cov_dict = Dict{String,Float64}("Min" => 1, "Med1" => 2, "Med2" => 3, "Max" => 4)
pol_cov_map(x) = get(pol_cov_dict, x, 4)
transform!(df, "pol_coverage" => ByRow(pol_cov_map) => "pol_coverage")

setdiff(feature_names, names(df))

seed!(123)
nobs = nrow(df)
id_train = sample(1:nobs, Int(round(0.8 * nobs)), replace = false)

dtrain = dropmissing(df[id_train, [feature_names..., target_name]])
deval = dropmissing(df[Not(id_train), [feature_names..., target_name]])

##############################
# NeuroTrees
##############################
config = NeuroTreeRegressor(
    loss = :logloss,
    nrounds = 400,
    actA = :tanh,
    depth = 4,
    ntrees = 32,
    batchsize = 4096,
    rng = 123,
    lr = 3e-3,
    early_stopping_rounds = 5,
    device = :gpu,
)

@time m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    feature_names,
    target_name,
    print_every_n = 5,
);
pred_eval_neuro = m(deval)

##############################
# EvoTrees
##############################
config = EvoTreeRegressor(
    T = Float32,
    loss = :logistic,
    nrounds = 1000,
    eta = 0.02,
    L2 = 1,
    lambda = 0.02,
    nbins = 32,
    max_depth = 5,
    rowsample = 0.5,
    colsample = 0.8,
    early_stopping_rounds = 50,
)

@time m =
    EvoTrees.fit(config, dtrain; deval, feature_names, target_name, print_every_n = 25);
pred_eval_evo = m(deval)

function logloss(p::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}
    eval = zero(T)
    @inbounds for i in eachindex(y)
        eval -= (y[i] * log(p[i]) + (1 - y[i]) * log(1 - p[i]))
    end
    eval /= length(p)
    return eval
end

# Neuro: 0.3170 | Evo: 0.3182
logloss(pred_eval_neuro, deval[!, target_name])
logloss(pred_eval_evo, deval[!, target_name])
