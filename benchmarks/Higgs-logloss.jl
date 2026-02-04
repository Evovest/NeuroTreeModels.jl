using Revise
using Random
using CSV
using DataFrames
using StatsBase
using Statistics: mean, std
using NeuroTreeModels
using Solage: Connectors
using AWS: AWSCredentials, AWSConfig, @service

@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds = aws_creds, region = "ca-central-1")
bucket = "jeremiedb"

path = "share/data/higgs/HIGGS.arrow"
df_tot = Connectors.read_arrow_aws(path; bucket = "jeremiedb", aws_config)

rename!(df_tot, "Column1" => "y")
feature_names = setdiff(names(df_tot), ["y"])
target_name = "y"

# function percent_rank(x::AbstractVector{T}) where {T}
#     return tiedrank(x) / (length(x) + 1)
# end

# transform!(df_tot, feature_names .=> percent_rank .=> feature_names)

dtrain = df_tot[1:(end-1_000_000), :];
deval = df_tot[(end-1_000_000+1):(end-500_000), :];
dtest = df_tot[(end-500_000+1):end, :];

config = NeuroTreeRegressor(
    loss = :logloss,
    nrounds = 200,
    depth = 4,
    lr = 2e-3,
    ntrees = 128,
    stack_size = 2,
    hidden_size = 16,
    batchsize = 8092,
    early_stopping_rounds = 2,
    device = :gpu,
)

@time m = NeuroTreeModels.fit(
    config,
    dtrain;
    deval,
    target_name,
    feature_names,
    print_every_n = 1,
);

dinfer_eval = NeuroTreeModels.get_df_loader_infer(
    deval;
    feature_names,
    batchsize = config.batchsize,
    device = config.device,
);
p_eval = m(dinfer_eval);
error_eval = 1 - mean(round.(Int, p_eval) .== deval.y)
@info "ERROR - deval" error_eval

dinfer_test = NeuroTreeModels.get_df_loader_infer(
    dtest;
    feature_names,
    batchsize = config.batchsize,
    device = config.device,
);
p_test = m(dinfer_test);
error_test = 1 - mean(round.(Int, p_test) .== dtest.y)
@info "ERROR - dtest" error_test

# depth:4, num_trees=256, stack_size=2, hidden_size=16, boosting_size=1, batchsize=2048, lr=1e-3
# ┌ Info: iter 30
# └   metric = 0.4679296910762787
# 10128.021110 seconds (806.60 M allocations: 206.595 GiB, 0.40% gc time, 0.00% compilation time)
# ┌ Info: ERROR - dtest
# └   error_test = 0.22794599999999998

# depth:5, num_trees=256, stack_size=1, hidden_size=1, boosting_size=1, batchsize=2048,
# ┌ Info: iter 40
# └   metric = 0.4786278009414673
# 10985.068111 seconds (959.42 M allocations: 259.180 GiB, 0.38% gc time)
# ┌ Info: ERROR - dtest
# └   error_test = 0.23524

# depth:5, num_trees=256, stack_size=3, hidden_size=16, boosting_size=3, batchsize=2048,
# ┌ Info: iter 33
# └   metric = 0.4564650058746338
# 34568.885039 seconds (7.51 G allocations: 1.109 TiB, 1.01% gc time)
# ┌ Info: ERROR - dtest
# └   error_test = 0.22153599999999996
