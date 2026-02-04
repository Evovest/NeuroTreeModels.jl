using NeuroTreeModels
using DataFrames
using PlotlyLight

#################################
# vanilla DataFrame
#################################
nobs = 1000
nfeats = 100
x, y = randn(Float32, nobs, nfeats), randn(Float32, nobs);
df = DataFrame(x, :auto);
df.y = y;
feature_names = Symbol("x" .* string.(1:nfeats))

config = NeuroTreeRegressor(; actA = :identity, depth = 3, ntrees = 32, init_scale = 0.1)

loss = NeuroTreeModels.get_loss_fn(config)
L = NeuroTreeModels.get_loss_type(config)
chain = NeuroTreeModels.get_model_chain(L; config, nfeats, outsize = 1)
info = Dict(:device => :cpu, :nrounds => 0, :feature_names => feature_names)
m = NeuroTreeModel(L, chain, info)
xb = x'
yb = y
m(xb)

@code_warntype m(xb)

w = m.chain.layers[2].trees[1].w
b = m.chain.layers[2].trees[1].b
p = m.chain.layers[2].trees[1].p

grads = NeuroTreeModels.gradient(model -> loss(model, xb, yb), m)[1]
grad_layers = grads[:chain][:layers]
grad_neuro = grad_layers[2][:trees][1]
dw = grad_neuro[:w]
db = grad_neuro[:b]
dp = grad_neuro[:p]

# fig =  plot(x=vec(w); type=:scatter, mode="markers")
fig = plot(x = vec(w); type = :histogram)
fig = plot(x = vec(dw); type = :histogram)

fig = plot(x = vec(b); type = :histogram)
fig = plot(x = vec(db); type = :histogram)

fig = plot(x = vec(p); type = :histogram)
fig = plot(x = vec(dp); type = :histogram)
