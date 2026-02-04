@testset "Core - internals test" begin

    config = NeuroTreeRegressor(
        loss = :mse,
        actA = :identity,
        init_scale = 1.0,
        nrounds = 200,
        depth = 4,
        ntrees = 32,
        stack_size = 1,
        hidden_size = 1,
        batchsize = 2048,
        lr = 1e-3,
    )

    # stack tree
    nobs = 1_000
    nfeats = 10
    x = rand(Float32, nfeats, nobs)
    feature_names = "var_" .* string.(1:nobs)

    outsize = 1
    loss = NeuroTreeModels.get_loss_fn(config)
    L = NeuroTreeModels.get_loss_type(config)
    chain = NeuroTreeModels.get_model_chain(L; config, nfeats, outsize)
    info = Dict(:device => :cpu, :nrounds => 0, :feature_names => feature_names)
    m = NeuroTreeModel(L, chain, info)

end

@testset "Core - Regression" begin

    Random.seed!(123)
    X, y = rand(1000, 10), randn(1000)
    df = DataFrame(X, :auto)
    df[!, :y] = y
    target_name = "y"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    config = NeuroTreeRegressor(loss = :mse, nrounds = 20, depth = 3, lr = 1e-1)

    m = NeuroTreeModels.fit(config, dtrain; target_name, feature_names)

    m = NeuroTreeModels.fit(config, dtrain; target_name, feature_names, deval)

end

@testset "Classification test" begin

    Random.seed!(123)
    X, y = @load_crabs
    df = DataFrame(X)
    df[!, :class] = y
    target_name = "class"
    feature_names = setdiff(names(df), [target_name])

    train_ratio = 0.8
    train_indices = randperm(nrow(df))[1:Int(train_ratio*nrow(df))]

    dtrain = df[train_indices, :]
    deval = df[setdiff(1:nrow(df), train_indices), :]

    config = NeuroTreeClassifier(
        nrounds = 100,
        depth = 4,
        lr = 3e-2,
        batchsize = 64,
        early_stopping_rounds = 10,
        device = :cpu,
    )

    m = NeuroTreeModels.fit(config, dtrain; deval, target_name, feature_names)

    # Predictions depend on the number of samples in the dataset
    ptrain = [argmax(x) for x in eachrow(m(dtrain))]
    peval = [argmax(x) for x in eachrow(m(deval))]
    @test mean(ptrain .== levelcode.(dtrain.class)) > 0.95
    @test mean(peval .== levelcode.(deval.class)) > 0.95

end
