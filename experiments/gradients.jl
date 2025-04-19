using NeuroTreeModels
using DataFrames
# using PlotlyLight

#################################
# vanilla DataFrame
#################################
nobs = 1000
nfeats = 100
x, y = randn(Float32, nobs, nfeats), randn(Float32, nobs);
df = DataFrame(x, :auto);
df.y = y;
feature_names = Symbol("x" .* string.(1:nfeats))

config = NeuroTreeRegressor(;
    actA=:identity,
    depth=3,
    ntrees=32,
    init_scale=0.1,
)

loss = NeuroTreeModels.get_loss_fn(config)
L = NeuroTreeModels.get_loss_type(config)
chain = NeuroTreeModels.get_model_chain(L; config, nfeats, outsize=1)
info = Dict(
    :device => :cpu,
    :nrounds => 0,
    :feature_names => feature_names
)
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
fig = plot(x=vec(w); type=:histogram)
fig = plot(x=vec(dw); type=:histogram)

fig = plot(x=vec(b); type=:histogram)
fig = plot(x=vec(db); type=:histogram)

fig = plot(x=vec(p); type=:histogram)
fig = plot(x=vec(dp); type=:histogram)



using CairoMakie
# Define the activation function
function act_xtanhx(x)
    return abs(x) * tanh(x)
end
# Define the activation function
function act_gauss(x)
    return x * (1 - exp(-x^2 / 0.01))
end
# Define the activation function
function act_sqrtx2(x)
    return sign(x) * √(x^2 + 0.001)
end
# Define the activation function
function act_logexp(x)
    return log(exp(x^2 + 1)) - x
end


function diff_act_xtanhx(x)
    return x * (x * sech(x)^2 + tanh(x)) / abs(x)
end


# Generate x values
x = range(-2.0, 2.0, length=1000)
y1 = act_xtanhx.(x)
dy1 = diff_act_xtanhx.(x)
y2 = act_gauss.(x)
y3 = act_sqrtx2.(x)
y4 = act_logexp.(x)
identity_line = x  # For y = x

# Create the plot
fig = Figure()
ax = Axis(fig[1, 1],
    xlabel="x",
    ylabel="f(x)",
    title="Activation Function: f(x) = |x| * tanh(x)")

lines!(ax, x, identity_line, color=:gray, linestyle=:dash, alpha=0.5, label="y = x")
lines!(ax, x, y1, color=:blue, linewidth=2, label="|x| * tanh(x)")
# lines!(ax, x, y2, color=:red, linewidth=2, label="x * (1 - exp(x^2 / 0.01))")
# lines!(ax, x, y3, color=:green, linewidth=2, label="sign(x) * √(x^2 + 0.01)")
# lines!(ax, x, y4, color=:purple, linewidth=2, label="log(exp(x^2 + 1)) - x")
axislegend(ax, position=:ct)
fig

# Create the plot
fig = Figure()
ax = Axis(fig[1, 1],
    xlabel="x",
    ylabel="f(x)",
    title="∇ Activation Function")

lines!(ax, x, dy1, color=:blue, linewidth=2, label="∇ |x| * tanh(x)")
# lines!(ax, x, y2, color=:red, linewidth=2, label="x * (1 - exp(x^2 / 0.01))")
# lines!(ax, x, y3, color=:green, linewidth=2, label="sign(x) * √(x^2 + 0.01)")
# lines!(ax, x, y4, color=:purple, linewidth=2, label="log(exp(x^2 + 1)) - x")
axislegend(ax, position=:lb)
fig
