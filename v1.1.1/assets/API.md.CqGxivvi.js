import{_ as i,c as s,o as e,a7 as a}from"./chunks/framework.DzDZdJdS.js";const E=JSON.parse('{"title":"API","description":"","frontmatter":{},"headers":[],"relativePath":"API.md","filePath":"API.md","lastUpdated":null}'),n={name:"API.md"},l=a(`<h1 id="API" tabindex="-1">API <a class="header-anchor" href="#API" aria-label="Permalink to &quot;API {#API}&quot;">​</a></h1><h2 id="Training" tabindex="-1">Training <a class="header-anchor" href="#Training" aria-label="Permalink to &quot;Training {#Training}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="NeuroTreeModels.fit" href="#NeuroTreeModels.fit">#</a> <b><u>NeuroTreeModels.fit</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> fit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    config</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NeuroTreeRegressor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dtrain;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    feature_names,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    target_name,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    weight_name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    offset_name</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    deval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    metric</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">nothing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    print_every_n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    early_stopping_rounds</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">9999</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    verbosity</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    return_logger</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Training function of NeuroTreeModels&#39; internal API.</p><p><strong>Arguments</strong></p><ul><li><p><code>config::NeuroTreeRegressor</code></p></li><li><p><code>dtrain</code>: Must be a <code>AbstractDataFrame</code></p></li></ul><p><strong>Keyword arguments</strong></p><ul><li><p><code>feature_names</code>: Required kwarg, a <code>Vector{Symbol}</code> or <code>Vector{String}</code> of the feature names.</p></li><li><p><code>target_name</code> Required kwarg, a <code>Symbol</code> or <code>String</code> indicating the name of the target variable.</p></li><li><p><code>weight_name=nothing</code></p></li><li><p><code>offset_name=nothing</code></p></li><li><p><code>deval=nothing</code> Data for tracking evaluation metric and perform early stopping.</p></li><li><p><code>metric=nothing</code>: evaluation metric tracked on <code>deval</code>. Can be one of:</p><ul><li><p><code>:mse</code></p></li><li><p><code>:mae</code></p></li><li><p><code>:logloss</code></p></li><li><p><code>:mlogloss</code></p></li><li><p><code>:gaussian_mle</code></p></li></ul></li><li><p><code>print_every_n=9999</code></p></li><li><p><code>early_stopping_rounds=9999</code></p></li><li><p><code>verbosity=1</code></p></li><li><p><code>return_logger=false</code></p></li></ul><p><a href="https://github.com/Evovest/NeuroTreeModels.jl/blob/b8afb2fd6fec7b351cca3f1fd5209a6a00ad3c53/src/fit.jl#L43-L84" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Inference" tabindex="-1">Inference <a class="header-anchor" href="#Inference" aria-label="Permalink to &quot;Inference {#Inference}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="NeuroTreeModels.infer" href="#NeuroTreeModels.infer">#</a> <b><u>NeuroTreeModels.infer</u></b> — <i>Function</i>. <p>infer(m::NeuroTreeModel, data)</p><p>Return the inference of a <code>NeuroTreeModel</code> over <code>data</code>, where <code>data</code> is <code>AbstractDataFrame</code>.</p><p><a href="https://github.com/Evovest/NeuroTreeModels.jl/blob/b8afb2fd6fec7b351cca3f1fd5209a6a00ad3c53/src/infer.jl#L8-L12" target="_blank" rel="noreferrer">source</a></p></div><br>`,7),t=[l];function r(o,p,d,h,c,k){return e(),s("div",null,t)}const u=i(n,[["render",r]]);export{E as __pageData,u as default};
