import{_ as e,c as s,o as i,a7 as a}from"./chunks/framework.DOlMkYgI.js";const u=JSON.parse('{"title":"Models","description":"","frontmatter":{},"headers":[],"relativePath":"models.md","filePath":"models.md","lastUpdated":null}'),t={name:"models.md"},n=a(`<h1 id="Models" tabindex="-1">Models <a class="header-anchor" href="#Models" aria-label="Permalink to &quot;Models {#Models}&quot;">​</a></h1><h2 id="NeuroTreeRegressor" tabindex="-1">NeuroTreeRegressor <a class="header-anchor" href="#NeuroTreeRegressor" aria-label="Permalink to &quot;NeuroTreeRegressor {#NeuroTreeRegressor}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="NeuroTreeModels.NeuroTreeRegressor" href="#NeuroTreeModels.NeuroTreeRegressor">#</a> <b><u>NeuroTreeModels.NeuroTreeRegressor</u></b> — <i>Type</i>. <p>NeuroTreeRegressor(; kwargs...)</p><p>A model type for constructing a NeuroTreeRegressor, based on <a href="https://github.com/Evovest/NeuroTreeModels.jl" target="_blank" rel="noreferrer">NeuroTreeModels.jl</a>, and implementing both an internal API and the MLJ model interface.</p><p><strong>Hyper-parameters</strong></p><ul><li><p><code>loss=:mse</code>: Loss to be be minimized during training. One of:</p><ul><li><p><code>:mse</code></p></li><li><p><code>:mae</code></p></li><li><p><code>:logloss</code></p></li><li><p><code>:mlogloss</code></p></li><li><p><code>:gaussian_mle</code></p></li></ul></li><li><p><code>nrounds=10</code>: Max number of rounds (epochs).</p></li><li><p><code>lr=1.0f-2</code>: Learning rate. Must be &gt; 0. A lower <code>eta</code> results in slower learning, typically requiring a higher <code>nrounds</code>.</p></li><li><p><code>wd=0.f0</code>: Weight decay applied to the gradients by the optimizer.</p></li><li><p><code>batchsize=2048</code>: Batch size.</p></li><li><p><code>actA=:tanh</code>: Activation function applied to each of input variable for determination of split node weight. Can be one of:</p><ul><li><p><code>:tanh</code></p></li><li><p><code>:identity</code></p></li></ul></li><li><p><code>depth=6</code>: Depth of a tree. Must be &gt;= 1. A tree of depth 1 has 2 prediction leaf nodes. A complete tree of depth N contains <code>2^N</code> terminal leaves and <code>2^N - 1</code> split nodes. Compute cost is proportional to <code>2^depth</code>. Typical optimal values are in the 3 to 5 range.</p></li><li><p><code>ntrees=64</code>: Number of trees (per stack).</p></li><li><p><code>hidden_size=16</code>: Size of hidden layers. Applicable only when <code>stack_size</code> &gt; 1.</p></li><li><p><code>stack_size=1</code>: Number of stacked NeuroTree blocks.</p></li><li><p><code>init_scale=1.0</code>: Scaling factor applied to the predictions weights. Values in the <code>]0, 1]</code> short result in best performance.</p></li><li><p><code>MLE_tree_split=false</code>: Whether independent models are buillt for each of the 2 parameters (mu, sigma) of the the <code>gaussian_mle</code> loss.</p></li><li><p><code>rng=123</code>: Either an integer used as a seed to the random number generator or an actual random number generator (<code>::Random.AbstractRNG</code>).</p></li></ul><p><strong>Internal API</strong></p><p>Do <code>config = NeuroTreeRegressor()</code> to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in NeuroTreeRegressor(loss=...).</p><p><strong>Training model</strong></p><p>A model is trained using <a href="/NeuroTreeModels.jl/previews/PR11/API#MLJModelInterface.fit"><code>fit</code></a>:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> fit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(config, dtrain; feature_names, target_name, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p><strong>Inference</strong></p><p>Models act as a functor. returning predictions when called as a function with features as argument:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">m</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(data)</span></span></code></pre></div><p><strong>MLJ Interface</strong></p><p>From MLJ, the type can be imported using:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">NeuroTreeRegressor </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @load</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NeuroTreeRegressor pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">NeuroTreeModels</span></span></code></pre></div><p>Do <code>model = NeuroTreeRegressor()</code> to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in <code>NeuroTreeRegressor(loss=...)</code>.</p><p><strong>Training model</strong></p><p>In MLJ or MLJBase, bind an instance <code>model</code> to data with <code>mach = machine(model, X, y)</code> where</p><ul><li><p><code>X</code>: any table of input features (eg, a <code>DataFrame</code>) whose columns each have one of the following element scitypes: <code>Continuous</code>, <code>Count</code>, or <code>&lt;:OrderedFactor</code>; check column scitypes with <code>schema(X)</code></p></li><li><p><code>y</code>: is the target, which can be any <code>AbstractVector</code> whose element scitype is <code>&lt;:Continuous</code>; check the scitype with <code>scitype(y)</code></p></li></ul><p>Train the machine using <code>fit!(mach, rows=...)</code>.</p><p><strong>Operations</strong></p><ul><li><code>predict(mach, Xnew)</code>: return predictions of the target given features <code>Xnew</code> having the same scitype as <code>X</code> above.</li></ul><p><strong>Fitted parameters</strong></p><p>The fields of <code>fitted_params(mach)</code> are:</p><ul><li><code>:fitresult</code>: The <code>NeuroTreeModel</code> object.</li></ul><p><strong>Report</strong></p><p>The fields of <code>report(mach)</code> are:</p><ul><li><code>:features</code>: The names of the features encountered in training.</li></ul><p><strong>Examples</strong></p><p><strong>Internal API</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> NeuroTreeModels, DataFrames</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">config </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> NeuroTreeRegressor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, nrounds</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nobs, nfeats </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1_000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dtrain </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> DataFrame</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">randn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nobs, nfeats), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:auto</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dtrain</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nobs)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">feature_names, target_name </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> names</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dtrain, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">r&quot;</span><span style="--shiki-light:#032F62;--shiki-dark:#DBEDFF;">x</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;y&quot;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> fit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(config, dtrain; feature_names, target_name)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">p </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> m</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dtrain)</span></span></code></pre></div><p><strong>MLJ Interface</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLJBase, NeuroTreeModels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> NeuroTreeRegressor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(depth</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, nrounds</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">X, y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @load_boston</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">mach </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> machine</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(m, X, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> fit!</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">p </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> predict</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mach, X)</span></span></code></pre></div><p><a href="https://github.com/Evovest/NeuroTreeModels.jl/blob/e83c1d56c3fdce51b00a792a772d499a46888118/src/learners.jl#L32-L145" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="NeuroTreeModel" tabindex="-1">NeuroTreeModel <a class="header-anchor" href="#NeuroTreeModel" aria-label="Permalink to &quot;NeuroTreeModel {#NeuroTreeModel}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="NeuroTreeModels.NeuroTreeModel" href="#NeuroTreeModels.NeuroTreeModel">#</a> <b><u>NeuroTreeModels.NeuroTreeModel</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">NeuroTreeModel</span></span></code></pre></div><p>A NeuroTreeModel is made of a collection of Tree, either regular <code>NeuroTree</code> or <code>StackTree</code>. Prediction is the sum of all the trees composing a NeuroTreeModel.</p><p><a href="https://github.com/Evovest/NeuroTreeModels.jl/blob/e83c1d56c3fdce51b00a792a772d499a46888118/src/model.jl#L118-L122" target="_blank" rel="noreferrer">source</a></p></div><br>`,7),r=[n];function l(o,p,d,h,c,k){return i(),s("div",null,r)}const E=e(t,[["render",l]]);export{u as __pageData,E as default};
