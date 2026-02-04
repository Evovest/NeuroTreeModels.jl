using StatsBase
using Statistics: mean, std
using CUDA

using NeuroTreeModels
using BenchmarkTools
using Random: seed!

using ChainRulesCore
import ChainRulesCore: rrule

Threads.nthreads()

seed!(123)
nobs = Int(1e6)
num_feat = Int(100)
@info "testing with: $nobs observations | $num_feat features."
X = rand(Float32, nobs, num_feat)
Y = randn(Float32, size(X, 1))

config = NeuroTreeRegressor(
    loss = :mse,
    nrounds = 1,
    actA = :identity,
    depth = 5,
    ntrees = 64,
    batchsize = 2048,
    rng = 123,
    lr = 1e-3,
    device = :gpu,
)

CUDA.@time m, cache = NeuroTreeModels.init(config, x_train = X, y_train = Y);
CUDA.@time NeuroTreeModels.fit!(m, cache);
CUDA.@time NeuroTreeModels.fit(config, x_train = X, y_train = Y);
@time NeuroTrees.fit(config, x_train = X, y_train = Y);
CUDA.@time NeuroTreeModels.fit(config, x_train = X, y_train = Y, x_eval = X, y_eval = Y);

#######
# mse
#######
# scaler true - 1e6 cpu: 
# scaler true - 1e6 gpu: 
# No callback
# scaler true: 1e6 cpu:
# scaler true: 1e6 gpu:
_device = config.device == "cpu" ? NeuroTrees.cpu : NeuroTrees.gpu
dinfer = NeuroTreeModels.DataLoader(
    Matrix{Float32}(X') |> _device,
    batchsize = config.batchsize,
    shuffle = false,
    partial = true,
)
@time pred = NeuroTreeModels.infer(m, dinfer);
@btime pred = NeuroTreeModels.infer($m, $dinfer);
#######
# mse
#######
# scaler true - 1e6 cpu:
# scaler true - 1e6 gpu: 

# No BatchNorm
# scaler true: 1e6 cpu: 1.119 s (366929 allocations: 512.58 MiB)
# scaler true: 1e6 gpu: 



#### forward speed test
x = rand(Float32, num_feat, config.batchsize) |> gpu;
y = rand(Float32, config.batchsize) |> gpu;

# cpu: 
# gpu: 12.371 ms (495 allocations: 28.27 KiB)
@time m(x);
@btime CUDA.@sync m($x);
# CUDA.@time m(x);

# cpu: 
# gpu: 1.102 ms (350 allocations: 19.12 KiB)
@time nw = NeuroTreeModels.node_weights(m, copy(x));
@btime NeuroTreeModels.node_weights($m, $copy(x));
@btime CUDA.@sync NeuroTreeModels.node_weights($m, $x);

# cpu: 
@time _, lw = NeuroTreeModels.leaf_weights!(nw);
# gpu pre: 10.282 ms (77 allocations: 4.70 KiB)
# gpu post: 6.574 ms (78 allocations: 4.81 KiB)
@btime CUDA.@sync NeuroTreeModels.leaf_weights!($nw);

# cpu: 
# gpu: 899.400 μs (130 allocations: 9.11 KiB)
@time pred = NeuroTreeModels.dot_prod_tullio!(lw, m.p);
@btime CUDA.@sync NeuroTreeModels.dot_prod_tullio!($lw, $m.p);


############################################
# node_weights breakdown
############################################
# cpu: 125.000 μs (26 allocations: 190.42 KiB)
# gpu: 55.500 μs (128 allocations: 7.38 KiB)
@time fw = m.actA(m.w) .* m.mask;
@btime m.actA(m.w) .* m.mask;
@btime CUDA.@sync m.actA(m.w) .* m.mask;

# cpu: 138.100 μs (274 allocations: 15.30 KiB)
# gpu: 83.100 μs (44 allocations: 2.08 KiB)
@time feat_proj = NeuroTreeModels.feat_proj!(fw, x);
@btime NeuroTreeModels.feat_proj!($fw, $x);
@btime CUDA.@sync NeuroTreeModels.feat_proj!($fw, $x);

# cpu: 376.214 ns (5 allocations: 320 bytes)
# gpu: 940.625 ns (3 allocations: 160 bytes)
@time feat_proj = reshape(feat_proj, size(m.b, 1), size(m.b, 2), :);
@btime reshape($feat_proj, size(m.b, 1), size(m.b, 2), :);
@btime CUDA.@sync reshape($feat_proj, size(m.b, 1), size(m.b, 2), :);

# cpu: 52.900 μs (245 allocations: 18.14 KiB)
# gpu: 39.200 μs (44 allocations: 4.80 KiB)
@time nw = NeuroTreeModels.nw_scale!(feat_proj, m.s, m.b);
@btime NeuroTreeModels.nw_scale!($feat_proj, $m.s, $m.b);
@btime CUDA.@sync NeuroTreeModels.nw_scale!($feat_proj, $m.s, $m.b);

# cpu: 275.200 μs (81 allocations: 8.25 KiB)
# gpu: 25.800 μs (4 allocations: 160 bytes)
@time NeuroTreeModels.sigmoid_act!(nw);
@btime NeuroTreeModels.sigmoid_act!($nw);
@btime CUDA.@sync NeuroTreeModels.sigmoid_act!($nw);

##########################
# grads
##########################
x = rand(Float32, num_feat, config.batchsize) |> gpu;
y = rand(Float32, config.batchsize) |> gpu;
w = ones(Float32, config.batchsize) |> gpu;
θ = NeuroTrees.params(m)

function forward_test_1(m, x)
    for i = 1:488
        vec(m(x))
    end
    return nothing
end

function grad_test1(m, x, y, w, θ)
    for i = 1:488
        gs = NeuroTreeModels.gradient(θ) do
            NeuroTreeModels.mse(m, x, y, w)
        end
    end
    return nothing
end

# 1.486592 seconds (257.94 k CPU allocations: 14.050 MiB, 0.83% gc time) (5.86 k GPU allocations: 5.948 GiB, 3.63% memmgmt time)
@time forward_test_1(m, x)
@btime forward_test_1($m, $x)
CUDA.@time CUDA.@sync forward_test_1(m, x)
# 3.587317 seconds (995.79 k CPU allocations: 63.791 MiB, 0.77% gc time) (21.47 k GPU allocations: 11.557 GiB, 4.12% memmgmt time)
@time grad_test1(m, x, y, w, θ)
@btime grad_test1($m, $x, $y, $w, $θ)
CUDA.@time CUDA.@sync grad_test1(m, x, y, w, θ)

function model_2(x)
    nw = NeuroTreeModels.node_weights(m, x)
    lw = NeuroTreeModels.leaf_weights!(m.cnw, nw)
    pred = NeuroTreeModels.dot_prod_tullio!(lw, m.p) ./ size(m.p, 3)
    return pred
end

function test_forward_2(x)
    for i = 1:488
        model_2(x)
    end
    return nothing
end
function test_loss_2(x, y)
    for i = 1:488
        NeuroTreeModels.mse(vec(model_2(x)), y)
    end
    return nothing
end

function grad_test2(x, y, θ)
    for i = 1:488
        gs = NeuroTreeModels.gradient(θ) do
            NeuroTreeModels.mse(vec(model_2(x)), y)
        end
    end
    return nothing
end

# 2.894989 seconds (213.37 k allocations: 2.785 GiB, 7.06% gc time, 0.23% compilation time)
@time test_forward_2(x)
# @btime test_forward_2($x)

# 2.938949 seconds (198.51 k allocations: 2.788 GiB, 6.84% gc time)
@time test_loss_2(x, y)
# @btime test_loss_2($x, $y)

# 7.582608 seconds (1.27 M allocations: 6.116 GiB, 5.35% gc time, 6.91% compilation time)
@time grad_test2(x, y, θ)
# @btime grad_test2($x, $y, $θ)



#######################
# post node_neights
#######################
function model_3(nw)
    lw = NeuroTreeModels.leaf_weights!(m.cnw, nw)
    pred = NeuroTreeModels.dot_prod_tullio!(lw, m.p) ./ size(m.p, 3)
    return pred
end

function test_forward_3(nw)
    for i = 1:488
        model_3(nw)
    end
    return nothing
end
function test_loss_3(nw, y)
    for i = 1:488
        NeuroTreeModels.mse(vec(model_3(nw)), y)
    end
    return nothing
end

function grad_test3(nw, y, θ)
    for i = 1:488
        gs = NeuroTreeModels.gradient(θ) do
            NeuroTreeModels.mse(vec(model_3(nw)), y)
        end
    end
    return nothing
end

# 198.227 ms (56696 allocations: 8.65 MiB)
@time test_forward_3(nw)
@btime test_forward_3($nw)

# 200.415 ms (58743 allocations: 12.57 MiB)
@time test_loss_3(nw, y)
@btime test_loss_3($nw, $y)

# 2.019 s (180016 allocations: 972.06 MiB)
@time grad_test3(nw, y, θ)
@btime grad_test3($nw, $y, $θ)


#######################
# post lw
#######################
function model_4(lw)
    pred = NeuroTreeModels.dot_prod_tullio!(lw, m.p) ./ size(m.p, 3)
    return pred
end

function test_forward_4(lw)
    for i = 1:488
        model_3(lw)
    end
    return nothing
end
function test_loss_4(lw, y)
    for i = 1:488
        NeuroTreeModels.mse(vec(model_3(lw)), y)
    end
    return nothing
end

function grad_test4(lw, y, θ)
    for i = 1:488
        gs = NeuroTreeModels.gradient(θ) do
            NeuroTreeModels.mse(vec(model_4(lw)), y)
        end
    end
    return nothing
end

# 57.251 ms (14156 allocations: 4.60 MiB)
@time test_forward_4(lw)
@btime test_forward_4($lw)

# 61.229 ms (16107 allocations: 8.52 MiB)
@time test_loss_4(lw, y)
@btime test_loss_4($lw, $y)

# 548.048 ms (103333 allocations: 49.62 MiB)
@time grad_test4(lw, y, θ)
@btime grad_test4($lw, $y, $θ)


function data_loop(data)
    for d in dtrain
    end
    return nothing
end
# 325.788 ms (4409 allocations: 408.22 MiB)
@time data_loop(dtrain)
@btime data_loop($dtrain)
