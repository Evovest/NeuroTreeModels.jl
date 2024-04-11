@testset "Regression test" begin

    config = NeuroTreeRegressor(
        device=:cpu,
        loss=:mse,
        actA=:identity,
        init_scale=1.0,
        nrounds=200,
        depth=4,
        ntrees=32,
        stack_size=1,
        hidden_size=1,
        outsize=1,
        batchsize=2048,
        lr=1e-3,
    )

    # stack tree
    nobs = 1_000
    nfeats = 10
    x = rand(Float32, nfeats, nobs)
    feature_names = "var_" .* string.(1:nobs)

    loss = NeuroTreeModels.get_loss_fn(config)
    L = NeuroTreeModels.get_loss_type(config)
    chain = NeuroTreeModels.get_model_chain(L; config, nfeats)
    info = Dict(
        :device => config.device,
        :nrounds => 0,
        :feature_names => feature_names
    )
    m = NeuroTreeModel(L, chain, info)

end

@testset "Classification test" begin

    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] .= levelcode.(y)
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio * nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    config = NeuroTreeRegressor(
        device=:cpu,
        loss=:mlogloss,
        nrounds=100,
        outsize=3,
        depth=3,
        lr=1e-1,
    )

    m = NeuroTreeModels.fit(
        config,
        dtrain;
        deval,
        target_name,
        feature_names,
        metric=:mlogloss,
        print_every_n=10,
        early_stopping_rounds=2,
    )

    # Predictions depend on the number of samples in the dataset
    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== dtrain.class) > 0.95
    @test mean(peval .== deval.class) > 0.95

end