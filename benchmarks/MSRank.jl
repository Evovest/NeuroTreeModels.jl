using Revise
using Random
using CSV
using DataFrames
using StatsBase
using Statistics: mean, std
using NeuroTreeModels
using Solage: Connectors
using ReadLIBSVM
using AWS: AWSCredentials, AWSConfig, @service

# https://www.microsoft.com/en-us/research/project/mslr/

@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds = aws_creds, region = "ca-central-1")
bucket = "jeremiedb"

# initial prep
function read_libsvm_aws(file::String; has_query = false, aws_config = AWSConfig())
    raw = S3.get_object(
        "jeremiedb",
        file,
        Dict("response-content-type" => "application/octet-stream");
        aws_config,
    )
    return read_libsvm(raw; has_query)
end

@time train_raw =
    read_libsvm_aws("share/data/msrank/train.txt"; has_query = true, aws_config);
@time eval_raw = read_libsvm_aws("share/data/msrank/vali.txt"; has_query = true, aws_config);
@time test_raw = read_libsvm_aws("share/data/msrank/test.txt"; has_query = true, aws_config);

dtrain = DataFrame(train_raw[:x], :auto)
dtrain.y_raw = train_raw[:y]
dtrain.y = dtrain.y_raw ./ 4
dtrain.q = train_raw[:q]

deval = DataFrame(eval_raw[:x], :auto)
deval.y_raw = eval_raw[:y]
deval.y = deval.y_raw ./ 4
deval.q = eval_raw[:q]

dtest = DataFrame(test_raw[:x], :auto)
dtest.y_raw = test_raw[:y]
dtest.y = dtest.y_raw ./ 4
dtest.q = test_raw[:q]

feature_names = setdiff(names(dtrain), ["y", "y_raw", "q"])
target_name = "y_raw"

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end

transform!(dtrain, feature_names .=> percent_rank .=> feature_names)
transform!(deval, feature_names .=> percent_rank .=> feature_names)
transform!(dtest, feature_names .=> percent_rank .=> feature_names)

config = NeuroTreeRegressor(
    loss = :mse,
    nrounds = 2,
    actA = :tanh,
    depth = 4,
    ntrees = 64,
    stack_size = 2,
    hidden_size = 16,
    batchsize = 4096,
    lr = 3e-4,
    early_stopping_rounds = 3,
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

p_eval = m(deval);
mse_eval = mean((p_eval .- deval.y_raw) .^ 2)
@info "MSE - deval" mse_eval

p_test = m(dtest);
mse_test = mean((p_test .- dtest.y_raw) .^ 2)
@info "MSE - dtest" mse_test
