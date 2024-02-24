"""
    mk_rng

make a Random Number Generator object
"""
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(rng::T) where {T<:Integer} = Random.MersenneTwister(rng)

"""
    optimiser
"""
# function get_optim(conf)
#     if conf["type"] == "descent"
#         opt = Optimisers.Descent(conf["lr"])
#     elseif conf["type"] == "momentum"
#         opt = Optimisers.Momentum(conf["lr"], conf["rho"])
#     elseif conf["type"] == "nesterov"
#         opt = Optimisers.Nesterov(conf["lr"], conf["rho"])
#     elseif conf["type"] == "adam"
#         opt = Optimisers.Adam(conf["lr"])
#     elseif conf["type"] == "nadam"
#         opt = Optimisers.NAdam(conf["lr"])
#     elseif conf["type"] == "radam"
#         opt = Optimisers.RADAM(conf["lr"])
#     elseif conf["type"] == "adamw"
#         opt = Optimisers.AdamW(conf["lr"], (0.9, 0.999), conf["wd"])
#     elseif conf["type"] == "amsgrad"
#         opt = Optimisers.AMSGrad(conf["lr"])
#     elseif conf["type"] == "rmsprop"
#         opt = Optimisers.RMSProp(conf["lr"])
#     elseif conf["type"] == "adadelta"
#         opt = Optimisers.AdaDelta(conf["rho"])
#     else
#         @warn "Invalid optimizer type. Defaulting to ADAM(1e-3)."
#         opt = Optimisers.ADAM(1e-3)
#     end
#     return opt
# end
