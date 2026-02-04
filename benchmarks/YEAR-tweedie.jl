#####################################################################
# WIP: need to adapt the fit! function to support normal DataFrame (not just GroupedOne)
#    Have dataloader adapted to DF vs GDF (both at fit init and callback init)
#####################################################################

using Revise
using Random
using CSV
using DataFrames
using StatsBase
using Statistics: mean, std
using NeuroTreeModels

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds = aws_creds, region = "ca-central-1")

path = "share/data/year/year.csv"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
df = DataFrame(CSV.File(raw, header = false))
df_tot = copy(df)

path = "share/data/year/year-train-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
train_idx = DataFrame(CSV.File(raw, header = false))[:, 1] .+ 1

path = "share/data/year/year-eval-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
eval_idx = DataFrame(CSV.File(raw, header = false))[:, 1] .+ 1

transform!(df_tot, "Column1" => identity => "y_raw")
transform!(df_tot, "y_raw" => (x -> (x .- minimum(x)) ./ std(x)) => "y_norm")
select!(df_tot, Not("Column1"))
feature_names = setdiff(names(df_tot), ["y_raw", "y_norm"])
target_name = "y_norm"

dtrain = df_tot[train_idx, :];
deval = df_tot[eval_idx, :];
dtest = df_tot[(end-51630+1):end, :];

config = NeuroTreeRegressor(
    loss = :tweedie,
    actA = :identity,
    init_scale = 0.1,
    nrounds = 200,
    depth = 5,
    ntrees = 32,
    stack_size = 1,
    hidden_size = 8,
    batchsize = 2048,
    lr = 1e-3,
    early_stopping_rounds = 2,
    device = :gpu,
)

@time m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n = 5,
);

p_eval = m(deval);
mse_eval = mean((p_eval .- deval.y_norm) .^ 2)
@info "MSE raw - deval" mse_eval

p_test = m(dtest);
mse_test = mean((p_test .- dtest.y_norm) .^ 2) * std(df_tot.y_raw)^2
@info "MSE - dtest" mse_test
