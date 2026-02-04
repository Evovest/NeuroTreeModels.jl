logit(x) = log(x / (1 - x))
logit(x::AbstractVector) = logit.(x)
sigmoid(x) = 1 / (1 + exp(-x))
sigmoid(x::AbstractVector) = sigmoid.(x)

@testset "generic interface tests" begin
    @testset "NeuroTreeRegressor" begin
        failures, summary = MLJTestInterface.test(
            [NeuroTreeRegressor],
            MLJTestInterface.make_regression()...;
            mod = @__MODULE__,
            verbosity = 0, # bump to debug
            throw = true, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "NeuroTreeClassifier" begin

        failures, summary = MLJTestInterface.test(
            [NeuroTreeClassifier],
            MLJTestInterface.make_binary()...;
            mod = @__MODULE__,
            verbosity = 0, # bump to debug
            throw = true, # set to true to debug
        )
        @test isempty(failures)

        failures, summary = MLJTestInterface.test(
            [NeuroTreeClassifier],
            MLJTestInterface.make_multiclass()...;
            mod = @__MODULE__,
            verbosity = 0, # bump to debug
            throw = true, # set to true to debug
        )
        @test isempty(failures)

    end
end

##################################################
### Regression
##################################################
@testset "MLJ - regression" begin

    features = rand(1_000) .* 5 .- 2
    X = reshape(features, (size(features)[1], 1))
    Y = sin.(features) .* 0.5 .+ 0.5
    Y = logit(Y) + randn(size(Y))
    Y = sigmoid(Y)
    y = Y
    X = MLJBase.table(X)

    tree_model = NeuroTreeRegressor(max_depth = 5, eta = 0.05, nrounds = 10)
    mach = machine(tree_model, X, y)
    train, test = partition(eachindex(y), 0.7, shuffle = true) # 70:30 split
    fit!(mach, rows = train, verbosity = 1)

    mach.model.nrounds += 10
    fit!(mach, rows = train, verbosity = 1)
    _report = report(mach)

    # predict on train data
    pred_train = predict(mach, selectrows(X, train))
    mean(abs.(pred_train - selectrows(Y, train)))

    # predict on test data
    pred_test = predict(mach, selectrows(X, test))
    mean(abs.(pred_test - selectrows(Y, test)))

    @test MLJBase.iteration_parameter(NeuroTreeRegressor) == :nrounds
end

@testset "MLJ - rowtables - NeuroTreeRegressor" begin
    X, y = make_regression(1000, 5)
    X = Tables.rowtable(X)
    booster = NeuroTreeRegressor()
    # smoke tests:
    mach = machine(booster, X, y) |> fit!
    fit!(mach)
    report(mach)
    predict(mach, X)
end

@testset "MLJ - named tuples - NeuroTreeRegressor" begin
    X, y = (x1 = rand(100), x2 = rand(100)), rand(100)
    booster = NeuroTreeRegressor()
    # smoke tests:
    mach = machine(booster, X, y) |> fit!
    fit!(mach)
    report(mach)
    predict(mach, X)
end

@testset "MLJ - classification" begin
    X, y = @load_crabs

    tree_model = NeuroTreeClassifier(depth = 4, lr = 0.1, nrounds = 20, batchsize = 64)

    # @load EvoTreeRegressor
    mach = machine(tree_model, X, y)
    train, test = partition(eachindex(y), 0.7, shuffle = true) # 70:30 split
    fit!(mach, rows = train, verbosity = 1)

    mach.model.nrounds += 50
    fit!(mach, rows = train, verbosity = 1)

    pred_train = predict(mach, selectrows(X, train))
    pred_train_mode = predict_mode(mach, selectrows(X, train))
    sum(pred_train_mode .== y[train]) / length(y[train])

    pred_test = predict(mach, selectrows(X, test))
    pred_test_mode = predict_mode(mach, selectrows(X, test))
    pred_test_mode = predict_mode(mach, selectrows(X, test))
    sum(pred_test_mode .== y[test]) / length(y[test])
end

@testset "MLJ - support for ordered factor predictions" begin
    X = (; x = rand(10))
    y = coerce(rand("ab", 10), OrderedFactor)
    model = NeuroTreeClassifier()
    mach = machine(model, X, y) |> fit!
    yhat = predict(mach, X)
    @assert isordered(yhat)
end
