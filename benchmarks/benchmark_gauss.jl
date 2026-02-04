# using Flux: params, gpu
using LoopVectorization
using Tullio
using StatsBase
using Statistics: mean, std

using Revise
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
X = rand(Float32, num_feat, nobs)
Y = randn(Float32, size(X, 2))

config = Dict(
    :loss_type => "gauss",
    :metric => "gauss",
    :actA => "softmax",
    :scaler => true,
    :dropout => 0.0,
    :outsize => 2,
    :device => "cpu",
    :num_feat => size(X, 1),
    :depth => 4,
    :num_trees => 16,
    :batch_size => 2048,
    :shuffle => true,
    :early_stopping_rounds => 0,
    :opt => Dict(:type => "nadam", :lr => 1e-3, :rho => 0.9),
)

@time NeuroTreeModels.fit!(loss, θ, dtrain, opt, nrounds = 1, cb = cb);
#######
# mse
#######
# scaler true: 1e6 cpu: 5.231904 seconds (1.59 M allocations: 1.393 GiB, 3.03% gc time)
# scaler true: 1e6 gpu: 

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
x = rand(Float32, num_feat, config[:batch_size]);
y = rand(Float32, config[:batch_size]);

# cpu: 1.561 ms (814 allocations: 1.04 MiB)
# gpu: 388.300 μs (338 allocations: 23.06 KiB)
@time m(copy(x));
@btime m($copy(x));

# cpu: 1.114 ms (667 allocations: 1.01 MiB)
# gpu: 89.800 μs (218 allocations: 13.50 KiB)
@time nw = NeuroTreeModels.node_weights(m, copy(x));
@btime NeuroTreeModels.node_weights($m, $copy(x));
@btime CUDA.@sync NeuroTreeModels.node_weights($m, $x);

# cpu: 120.600 μs (81 allocations: 8.27 KiB)
# gpu: 132.100 μs (18 allocations: 1.17 KiB)
@time _, lw = NeuroTreeModels.leaf_weights!(nw);
@btime NeuroTrees.leaf_weights!($nw);

# cpu: 66.800 μs (56 allocations: 3.66 KiB)
# gpu: 18.000 μs (67 allocations: 5.62 KiB)
@time pred = NeuroTreeModels.dot_prod_tullio!(lw, m.p);
@btime NeuroTreeModels.dot_prod_tullio!($lw, $m.p);


# cpu: 3.042 ms (1639 allocations: 1.33 MiB)
# gpu: 
@time loss(copy(x), y);
@btime loss($copy(x), $y);
preds = m(x)
function gauss1(μ, σ, y)
    @tullio gauss := -σ[i] - (y[i] - μ[i]) ^ 2 / (2 * exp(2 * σ[i]))
    return gauss / length(y)
end

@btime NeuroTreeModels.gauss_𝑙(preds[1, :], preds[1, :], y)
@btime gauss1(preds[1, :], preds[1, :], y)

# gauss_𝑙(μ, σ, y) = mean(-σ .- (y .- μ) .^ 2 ./ (2 .* max.(Float32(2e-7), exp.(2 .* σ))))

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
@btime NeuroTrees.feat_proj!($fw, $x);
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
