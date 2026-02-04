using Revise
using CSV
using DataFrames
using StatsBase: sample, tiedrank
using Statistics
using Random: seed!
using ReadLIBSVM
using NeuroTreeModels

# data is C14 - Yahoo! Learning to Rank Challenge
# data can be obtained though a request to https://webscope.sandbox.yahoo.com/
# Yahoo's paper overview with benchmarks (0.790 bot GBT: https://www.khoury.northeastern.edu/home/vip/teach/IRcourse/6_ML/other_notes/chapelle11a.pdf)
# CatBoost benchmarks: https://github.com/catboost/benchmarks/blob/master/ranking/Readme.md#4-results

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds = aws_creds, region = "ca-central-1")

function read_libsvm_aws(file::String; has_query = false, aws_config = AWSConfig())
    raw = S3.get_object(
        "jeremiedb",
        file,
        Dict("response-content-type" => "application/octet-stream");
        aws_config,
    )
    return read_libsvm(raw; has_query)
end

function ndcg(p, y, k = 10)
    k = min(k, length(p))
    p_order = partialsortperm(p, 1:k, rev = true)
    y_order = partialsortperm(y, 1:k, rev = true)
    _y = y[p_order]
    gains = 2 .^ _y .- 1
    discounts = log2.((1:k) .+ 1)
    ndcg = sum(gains ./ discounts)

    y_order = partialsortperm(y, 1:k, rev = true)
    _y = y[y_order]
    gains = 2 .^ _y .- 1
    discounts = log2.((1:k) .+ 1)
    idcg = sum(gains ./ discounts)

    return idcg == 0 ? 1.0 : ndcg / idcg
end

@time train_raw =
    read_libsvm_aws("share/data/yahoo-ltrc/set1.train.txt"; has_query = true, aws_config);
@time eval_raw =
    read_libsvm_aws("share/data/yahoo-ltrc/set1.valid.txt"; has_query = true, aws_config);
@time test_raw =
    read_libsvm_aws("share/data/yahoo-ltrc/set1.test.txt"; has_query = true, aws_config);

colsums_train = map(sum, eachcol(train_raw[:x]))
# colsums_eval = map(sum, eachcol(eval_raw[:x]))
colsums_test = map(sum, eachcol(test_raw[:x]))

sum(colsums_train .== 0)
sum(colsums_test .== 0)
@assert all((colsums_train .== 0) .== (colsums_test .== 0))
drop_cols = colsums_train .== 0

x_train = train_raw[:x][:, .!drop_cols]
x_eval = eval_raw[:x][:, .!drop_cols]
x_test = test_raw[:x][:, .!drop_cols]

x_train_miss = x_train .== 0
x_eval_miss = x_eval .== 0
x_test_miss = x_test .== 0

x_train[x_train .== 0] .= 0.5
x_eval[x_eval .== 0] .= 0.5
x_test[x_test .== 0] .= 0.5

x_train = hcat(x_train, x_train_miss)
x_eval = hcat(x_eval, x_eval_miss)
x_test = hcat(x_test, x_test_miss)

#####################################
# create DataFrames
#####################################
dtrain = DataFrame(x_train, :auto)
dtrain.y_raw .= train_raw[:y]
dtrain.y .= train_raw[:y] ./ 4
dtrain.q .= train_raw[:q]

deval = DataFrame(x_eval, :auto)
deval.y_raw .= eval_raw[:y]
deval.y .= eval_raw[:y] ./ 4
deval.q .= eval_raw[:q]

dtest = DataFrame(x_test, :auto)
dtest.y_raw .= test_raw[:y]
dtest.y .= test_raw[:y] ./ 4
dtest.q .= test_raw[:q]

feature_names = setdiff(names(dtrain), ["y", "y_raw", "q", "offset", "batches"])
target_name = "y"

#####################################
# training
#####################################
config = NeuroTreeRegressor(
    loss = :logloss,
    nrounds = 400,
    actA = :identity,
    init_scale = 1.0,
    scaler = true,
    depth = 4,
    ntrees = 256,
    hidden_size = 1,
    stack_size = 1,
    batchsize = 1024,
    shuffle = true,
    lr = 3e-4,
    metric = :logloss,
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

dinfer = NeuroTreeModels.get_df_loader_infer(
    dtest;
    feature_names,
    batchsize = config.batchsize,
    device = config.device,
);
p_test = m(dinfer) .* 4;

test_df = DataFrame(p = p_test, y = dtest.y_raw, q = dtest.q)
test_df_agg = combine(groupby(test_df, "q"), ["p", "y"] => ndcg => "ndcg")
ndcg_test = mean(test_df_agg.ndcg)
mse_test = mean((p_test .- dtest.y_raw) .^ 2)
@info "MSE - full - test data" mse_test
@info "NDCG - full - test data" ndcg_test


# Tuning fit - depth 5 boosting_size 4 stack 1
# 6920.016444 seconds (505.02 M allocations: 3.038 TiB, 1.12% gc time, 0.00% compilation time)
# ┌ Info: MSE - full - test data
# └   mse_test = 0.5648856522739414
# ┌ Info: NDCG - full - test data
# └   ndcg_test = 0.7930367325602978

# 7193.781178 seconds (523.91 M allocations: 3.154 TiB, 1.12% gc time)
# ┌ Info: MSE - full - test data
# └   mse_test = 0.5688543564886467
# ┌ Info: NDCG - full - test data
# └   ndcg_test = 0.7899568312166952
