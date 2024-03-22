import{_ as t,c as e,o as i,a7 as s}from"./chunks/framework.CUwTDK2x.js";const a="/NeuroTreeModels.jl/dev/assets/decision-tree.Dqh78YdA.png",n="/NeuroTreeModels.jl/dev/assets/neurotree.vCO5vhkM.png",f=JSON.parse('{"title":"NeuroTree - A differentiable tree operator for tabular data","description":"","frontmatter":{},"headers":[],"relativePath":"design.md","filePath":"design.md","lastUpdated":null}'),r={name:"design.md"},l=s('<h1 id="NeuroTree-A-differentiable-tree-operator-for-tabular-data" tabindex="-1">NeuroTree - A differentiable tree operator for tabular data <a class="header-anchor" href="#NeuroTree-A-differentiable-tree-operator-for-tabular-data" aria-label="Permalink to &quot;NeuroTree - A differentiable tree operator for tabular data {#NeuroTree-A-differentiable-tree-operator-for-tabular-data}&quot;">​</a></h1><h2 id="Overview" tabindex="-1">Overview <a class="header-anchor" href="#Overview" aria-label="Permalink to &quot;Overview {#Overview}&quot;">​</a></h2><p>This work introduces <code>NeuroTree</code> a differentiable binary tree operator adapted for the treatment of tabular data.</p><ul><li><p>Address the shortcoming of traditional trees greediness: all node and leaves are learned simultaneously. It provides the ability to learn an optimal configuration across all the tree levels. The notion extent also to the collection of trees that are simultaneously learned.</p></li><li><p>Extend the notion of forest/bagging and boosting.</p><ul><li>Although the predictions from the all the trees forming a NeuroTree operator are averaged, each of the tree prediction is tuned simultaneously. This is different from boosting (ex XGBoost) where each tree is learned sequentially and over the residual from previous trees. Also, unlike random forest and bagging, trees aren&#39;t learned in isolation but tuned collaboratively, resulting in predictions that account for all of the other tree predictions.</li></ul></li><li><p>General operator compatible for composition.</p><ul><li>Allows integration within Flux&#39;s Chain like other standard operators from NNLib. Composition is also illustrated through the built-in StackTree layer, a residual composition of multiple NeuroTree building blocks.</li></ul></li><li><p>Compatible with general purpose machine learning framework.</p><ul><li>MLJ integration</li></ul></li></ul><h2 id="Architecture" tabindex="-1">Architecture <a class="header-anchor" href="#Architecture" aria-label="Permalink to &quot;Architecture {#Architecture}&quot;">​</a></h2><p>A NeuroTree operator acts as collection of complete binary trees, ie. trees without any pruned node. To be differentiable, hence trainable using first-order gradient based methods (ex. Adam optimiser), each tree path implements a soft decision rather than a hard one like in traditional decision tree.</p><p>To introduce the implementation of a NeuroTree, we first get back to the architecture of a basic decision tree.</p><p><img src="'+a+'" alt=""></p><p>The above is a binary decision tree of <code>depth</code> 2.</p><p>Highlighted in green is the decision path taken for a given sample. It goes into <code>depth</code> number of binary decisions, resulting in the path <code>node1 → node3 → leaf3</code>.</p><p>One way to view the role of the decision nodes (gray background) is to provide an index of the leaf prediction to fetch (index <code>3</code> in the figure). Such indexing view is applicable given that node routing relies on hard conditions: either <code>true</code> or <code>false</code>.</p><p>An alternative perspective that we adopt here is that tree nodes collectively provide weights associated to each leaf. A tree prediction becomes the weighted sum of the leaf&#39;s values and the leaf&#39;s weights. In regular decision trees, since all conditions are binary, leaf weights take the form of a mask. In the above example, the mask is <code>[0, 0, 1, 0]</code>.</p><p>By relaxing these hard conditions into soft ones, the mask takes the form of a probability vector associated to each leaf, where <code>∑(leaf_weights) = 1</code> and where each each <code>leaf_weight</code> element is <code>[0, 1]</code>. A tree prediction can be obtained with the dot product: <code>leaf_values&#39; * leaf_weights</code>.</p><p>The following illustrate how a basic decision tree is represented as a single differentiable tree within NeuroTree:</p><p><img src="'+n+`" alt=""></p><h3 id="Node-weights" tabindex="-1">Node weights <a class="header-anchor" href="#Node-weights" aria-label="Permalink to &quot;Node weights {#Node-weights}&quot;">​</a></h3><p>To illustrate how a NeuroTree derives the soft decision probability (referred to <code>NW1 - NW3</code> in the above figure), we first break down how a traditional tree split condition is derived from 2 underlying decisions:</p><ol><li><em>Selection of the feature on which to perform the condition</em>.</li></ol><p>Such selection can be represented as the application of a binary mask where all elements are set to <code>false</code> except for that single selected feature where it&#39;s set to <code>true</code>.</p><ol><li><em>Selection of the condition&#39;s threshold value</em>.</li></ol><p>For a given observation, if the selected feature&#39;s value is below that threshold, then the node decision is set to <code>false</code> (pointing to left child), and <code>true</code> otherwise (pooinnting to right child).</p><p>In NeuroTree, these 2 hard steps are translated into soft, differentiable ones.</p><h3 id="Leaf-weights" tabindex="-1">Leaf weights <a class="header-anchor" href="#Leaf-weights" aria-label="Permalink to &quot;Leaf weights {#Leaf-weights}&quot;">​</a></h3><p>Computing the leaf weights consists of accumulating the weights through each tree branch. It&#39;s the technically more challenging part as such computation cannot be represented as a form of matrix multiplication, unlike other common operators like <code>Dense</code>, <code>Conv</code> or <code>MultiHeadAttention</code> / <code>Transformer</code>. Performing probability accumulation though a tree index naturally leads to in-place element wise operations, which are notoriously not friendly for auto-differentiation engines. Since NeuroTree was intended to integrate with the Flux.jl ecosystem, Zygote.jl acts as the underlying AD, the approach used was to manually implement <code>backward</code> / <code>adjoint</code> of the terminal leaf function and instruct the AD to use that custom rule rather than attempt to differentiate a non-AD compliant function.</p><p>Below are the algo and actual implementation of the forward and backward function that compute the leaf weights. For brevity, the loops over each observation of the batch and each tree are omitted. Parallelism, both on CPU and GPU, is obtained through parallelization over the <code>tree</code> and <code>batch</code> dimensions.</p><p><strong>Forward</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> leaf_weights!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nw)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cw </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ones</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nw), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(cw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        cw[i] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cw[i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nw[i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        cw[i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tree, batch] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cw[i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nw[i</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    </span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lw </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cw[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(cw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (cw, lw)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p><strong>Backward</strong></p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> Δ_leaf_weights!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Δnw, ȳ, cw, nw, max_depth, node_offset)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> axes</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(nw, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)        </span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        depth </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> floor</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Int, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">log2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(i)) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># current depth level - starting at 0</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        step </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">^</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(max_depth </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> depth) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># iteration length</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        leaf_offset </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> step </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (i </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">^</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">depth) </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># offset on the leaf row</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> j </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">leaf_offset)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(step</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">leaf_offset)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            k </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> j </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> node_offset </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># move from leaf position to full tree position </span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            Δnw[i] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ȳ[j] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cw[k] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nw[i]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    </span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> j </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">leaf_offset</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">step</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(step</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">leaf_offset)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            k </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> j </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> node_offset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            Δnw[i] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ȳ[j] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cw[k] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> nw[i])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nothing</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h3 id="Tree-prediction" tabindex="-1">Tree prediction <a class="header-anchor" href="#Tree-prediction" aria-label="Permalink to &quot;Tree prediction {#Tree-prediction}&quot;">​</a></h3><h2 id="Composability" tabindex="-1">Composability <a class="header-anchor" href="#Composability" aria-label="Permalink to &quot;Composability {#Composability}&quot;">​</a></h2><ul><li><p>StackTree</p></li><li><p>General operator: Chain <code>NeuroTree</code> with MLP</p></li></ul><h2 id="Benchmarks" tabindex="-1">Benchmarks <a class="header-anchor" href="#Benchmarks" aria-label="Permalink to &quot;Benchmarks {#Benchmarks}&quot;">​</a></h2><p>For each dataset and algo, the following methodology is followed:</p><ul><li><p>Data is split in three parts: <code>train</code>, <code>eval</code> and <code>test</code></p></li><li><p>A random grid of 16 hyper-parameters is generated</p></li><li><p>For each parameter configuration, a model is trained on <code>train</code> data until the evaluation metric tracked against the <code>eval</code> stops improving (early stopping)</p></li><li><p>The trained model is evaluated against the <code>test</code> data</p></li><li><p>The metric presented in below are the ones obtained on the <code>test</code> for the model that generated the best <code>eval</code> metric.</p></li></ul><p>Source code available at <a href="https://github.com/Evovest/MLBenchmarks.jl" target="_blank" rel="noreferrer">MLBenchmarks.jl</a>.</p><p>For performance assessment, benchmarks is run on the following selection of common Tabular datasets:</p><ul><li><p><a href="https://archive.ics.uci.edu/dataset/203/yearpredictionmsd" target="_blank" rel="noreferrer">Year</a>: min squared error regression. 515,345 observations, 90 features.</p></li><li><p><a href="https://www.microsoft.com/en-us/research/project/mslr/" target="_blank" rel="noreferrer">MSRank</a>: ranking problem with min squared error regression. 1,200,192 observations, 136 features.</p></li><li><p><a href="https://webscope.sandbox.yahoo.com/" target="_blank" rel="noreferrer">YahooRank</a>: ranking problem with min squared error regression. 709,877 observations, 519 features.</p></li><li><p><a href="https://archive.ics.uci.edu/dataset/280/higgs" target="_blank" rel="noreferrer">Higgs</a>: 2-level classification with logistic regression. 11,000,000 observations, 28 features.</p></li><li><p><a href="https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.BostonHousing" target="_blank" rel="noreferrer">Boston Housing</a>: min squared error regression.</p></li><li><p><a href="https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.Titanic" target="_blank" rel="noreferrer">Titanic</a>: 2-level classification with logistic regression. 891 observations, 7 features.</p></li></ul><p>Comparison is performed against the following algos (implementation in link) considered as state of the art on classification tasks:</p><ul><li><p><a href="https://github.com/Evovest/EvoTrees.jl" target="_blank" rel="noreferrer">EvoTrees</a></p></li><li><p><a href="https://github.com/dmlc/XGBoost.jl" target="_blank" rel="noreferrer">XGBoost</a></p></li><li><p><a href="https://github.com/IQVIA-ML/LightGBM.jl" target="_blank" rel="noreferrer">LightGBM</a></p></li><li><p><a href="https://github.com/JuliaAI/CatBoost.jl" target="_blank" rel="noreferrer">CatBoost</a></p></li><li><p><a href="https://github.com/manujosephv/pytorch_tabular" target="_blank" rel="noreferrer">NODE</a></p></li></ul><h4 id="Boston" tabindex="-1">Boston <a class="header-anchor" href="#Boston" aria-label="Permalink to &quot;Boston {#Boston}&quot;">​</a></h4><table><thead><tr><th style="text-align:center;"><strong>model_type</strong></th><th style="text-align:center;"><strong>train_time</strong></th><th style="text-align:center;"><strong>mse</strong></th><th style="text-align:center;"><strong>gini</strong></th></tr></thead><tbody><tr><td style="text-align:center;">neurotrees</td><td style="text-align:center;">12.8</td><td style="text-align:center;">18.9</td><td style="text-align:center;"><strong>0.947</strong></td></tr><tr><td style="text-align:center;">evotrees</td><td style="text-align:center;">0.206</td><td style="text-align:center;">19.7</td><td style="text-align:center;">0.927</td></tr><tr><td style="text-align:center;">xgboost</td><td style="text-align:center;">0.0648</td><td style="text-align:center;">19.4</td><td style="text-align:center;">0.935</td></tr><tr><td style="text-align:center;">lightgbm</td><td style="text-align:center;">0.865</td><td style="text-align:center;">25.4</td><td style="text-align:center;">0.926</td></tr><tr><td style="text-align:center;">catboost</td><td style="text-align:center;">0.0511</td><td style="text-align:center;"><strong>13.9</strong></td><td style="text-align:center;">0.946</td></tr></tbody></table><h4 id="Titanic" tabindex="-1">Titanic <a class="header-anchor" href="#Titanic" aria-label="Permalink to &quot;Titanic {#Titanic}&quot;">​</a></h4><table><thead><tr><th style="text-align:center;"><strong>model_type</strong></th><th style="text-align:center;"><strong>train_time</strong></th><th style="text-align:center;"><strong>logloss</strong></th><th style="text-align:center;"><strong>accuracy</strong></th></tr></thead><tbody><tr><td style="text-align:center;">neurotrees</td><td style="text-align:center;">7.58</td><td style="text-align:center;">0.407</td><td style="text-align:center;">0.828</td></tr><tr><td style="text-align:center;">evotrees</td><td style="text-align:center;">0.673</td><td style="text-align:center;">0.382</td><td style="text-align:center;">0.828</td></tr><tr><td style="text-align:center;">xgboost</td><td style="text-align:center;">0.0379</td><td style="text-align:center;"><strong>0.375</strong></td><td style="text-align:center;">0.821</td></tr><tr><td style="text-align:center;">lightgbm</td><td style="text-align:center;">0.615</td><td style="text-align:center;">0.390</td><td style="text-align:center;"><strong>0.836</strong></td></tr><tr><td style="text-align:center;">catboost</td><td style="text-align:center;">0.0326</td><td style="text-align:center;">0.388</td><td style="text-align:center;"><strong>0.836</strong></td></tr></tbody></table><h4 id="Year" tabindex="-1">Year <a class="header-anchor" href="#Year" aria-label="Permalink to &quot;Year {#Year}&quot;">​</a></h4><table><thead><tr><th style="text-align:center;"><strong>model_type</strong></th><th style="text-align:center;"><strong>train_time</strong></th><th style="text-align:center;"><strong>mse</strong></th><th style="text-align:center;"><strong>gini</strong></th></tr></thead><tbody><tr><td style="text-align:center;">neurotrees</td><td style="text-align:center;">280.0</td><td style="text-align:center;"><strong>76.4</strong></td><td style="text-align:center;"><strong>0.652</strong></td></tr><tr><td style="text-align:center;">evotrees</td><td style="text-align:center;">18.6</td><td style="text-align:center;">80.1</td><td style="text-align:center;">0.627</td></tr><tr><td style="text-align:center;">xgboost</td><td style="text-align:center;">17.2</td><td style="text-align:center;">80.2</td><td style="text-align:center;">0.626</td></tr><tr><td style="text-align:center;">lightgbm</td><td style="text-align:center;">8.11</td><td style="text-align:center;">80.3</td><td style="text-align:center;">0.624</td></tr><tr><td style="text-align:center;">catboost</td><td style="text-align:center;">80.0</td><td style="text-align:center;">79.2</td><td style="text-align:center;">0.635</td></tr></tbody></table><h4 id="MSRank" tabindex="-1">MSRank <a class="header-anchor" href="#MSRank" aria-label="Permalink to &quot;MSRank {#MSRank}&quot;">​</a></h4><table><thead><tr><th style="text-align:center;"><strong>model_type</strong></th><th style="text-align:center;"><strong>train_time</strong></th><th style="text-align:center;"><strong>mse</strong></th><th style="text-align:center;"><strong>ndcg</strong></th></tr></thead><tbody><tr><td style="text-align:center;">neurotrees</td><td style="text-align:center;">39.1</td><td style="text-align:center;">0.578</td><td style="text-align:center;">0.462</td></tr><tr><td style="text-align:center;">evotrees</td><td style="text-align:center;">37.0</td><td style="text-align:center;">0.554</td><td style="text-align:center;"><strong>0.504</strong></td></tr><tr><td style="text-align:center;">xgboost</td><td style="text-align:center;">12.5</td><td style="text-align:center;">0.554</td><td style="text-align:center;">0.503</td></tr><tr><td style="text-align:center;">lightgbm</td><td style="text-align:center;">37.5</td><td style="text-align:center;"><strong>0.553</strong></td><td style="text-align:center;">0.503</td></tr><tr><td style="text-align:center;">catboost</td><td style="text-align:center;">15.1</td><td style="text-align:center;">0.558</td><td style="text-align:center;">0.497</td></tr></tbody></table><h4 id="Yahoo" tabindex="-1">Yahoo <a class="header-anchor" href="#Yahoo" aria-label="Permalink to &quot;Yahoo {#Yahoo}&quot;">​</a></h4><table><thead><tr><th style="text-align:center;"><strong>model_type</strong></th><th style="text-align:center;"><strong>train_time</strong></th><th style="text-align:center;"><strong>mse</strong></th><th style="text-align:center;"><strong>ndcg</strong></th></tr></thead><tbody><tr><td style="text-align:center;">neurotrees</td><td style="text-align:center;">417.0</td><td style="text-align:center;">0.584</td><td style="text-align:center;">0.781</td></tr><tr><td style="text-align:center;">evotrees</td><td style="text-align:center;">687.0</td><td style="text-align:center;">0.545</td><td style="text-align:center;">0.797</td></tr><tr><td style="text-align:center;">xgboost</td><td style="text-align:center;">120.0</td><td style="text-align:center;">0.547</td><td style="text-align:center;"><strong>0.798</strong></td></tr><tr><td style="text-align:center;">lightgbm</td><td style="text-align:center;">244.0</td><td style="text-align:center;"><strong>0.540</strong></td><td style="text-align:center;">0.796</td></tr><tr><td style="text-align:center;">catboost</td><td style="text-align:center;">161.0</td><td style="text-align:center;">0.561</td><td style="text-align:center;">0.794</td></tr></tbody></table><h4 id="Higgs" tabindex="-1">Higgs <a class="header-anchor" href="#Higgs" aria-label="Permalink to &quot;Higgs {#Higgs}&quot;">​</a></h4><table><thead><tr><th style="text-align:center;"><strong>model_type</strong></th><th style="text-align:center;"><strong>train_time</strong></th><th style="text-align:center;"><strong>logloss</strong></th><th style="text-align:center;"><strong>accuracy</strong></th></tr></thead><tbody><tr><td style="text-align:center;">neurotrees</td><td style="text-align:center;">12300.0</td><td style="text-align:center;"><strong>0.452</strong></td><td style="text-align:center;"><strong>0.781</strong></td></tr><tr><td style="text-align:center;">evotrees</td><td style="text-align:center;">2620.0</td><td style="text-align:center;">0.464</td><td style="text-align:center;">0.776</td></tr><tr><td style="text-align:center;">xgboost</td><td style="text-align:center;">1390.0</td><td style="text-align:center;">0.462</td><td style="text-align:center;">0.776</td></tr><tr><td style="text-align:center;">lightgbm</td><td style="text-align:center;">1330.0</td><td style="text-align:center;">0.461</td><td style="text-align:center;">0.779</td></tr><tr><td style="text-align:center;">catboost</td><td style="text-align:center;">7180.0</td><td style="text-align:center;">0.464</td><td style="text-align:center;">0.775</td></tr></tbody></table><h2 id="Discussion" tabindex="-1">Discussion <a class="header-anchor" href="#Discussion" aria-label="Permalink to &quot;Discussion {#Discussion}&quot;">​</a></h2><p>NeuroTreeModels can achieve top tier performance on both small (Boston) and large (Higgs) datasets. Its performance trailed on the two ranking regression problems (MSRank and Yahoo). Although the large number of features is a distinguishing characteristic of the Yahoo dataset, the 136 features of MSRank are not materially different for the YEAR dataset (90 features), and on which NeuroTreeMoels outperform all other algos. Considering that no sparsity mechanism is present in the feature selection for the node conditions, datasets with a very large number of features may present a challenge. Substituting the default <code>tanh</code> activation with a sparsity inducing one such as <code>hardsigmoid</code> or <code>EntrOpt</code> has not resulted in improvement from the experiments.</p><p>Another potential weakness may stem from the soft nature of the decision criteria. Traditional trees can isolate the effect of a specific feature value. This can be notably meaningful in a situation where a numeric feature taking a value of 0 may carry a particular meaning (ex. missing, unknown value). Such stump the effect of a feature should be harder to pick with NeuroTree&#39;s soft condition.</p><h2 id="References" tabindex="-1">References <a class="header-anchor" href="#References" aria-label="Permalink to &quot;References {#References}&quot;">​</a></h2><ul><li><p><a href="https://arxiv.org/abs/1909.06312v2" target="_blank" rel="noreferrer">Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data</a>.</p><ul><li><p><a href="https://github.com/Qwicen/node" target="_blank" rel="noreferrer">https://github.com/Qwicen/node</a></p></li><li><p><a href="https://github.com/manujosephv/pytorch_tabular" target="_blank" rel="noreferrer">https://github.com/manujosephv/pytorch_tabular</a></p></li></ul></li><li><p><a href="https://arxiv.org/abs/2010.02921" target="_blank" rel="noreferrer">Attention augmented differentiable forest for tabular data</a></p></li><li><p><a href="https://arxiv.org/abs/2307.12198" target="_blank" rel="noreferrer">NCART: Neural Classification and Regression Tree for Tabular Data</a></p></li><li><p><a href="https://arxiv.org/abs/1709.01507" target="_blank" rel="noreferrer">Squeeze-and-Excitation Networks</a></p></li><li><p><a href="https://arxiv.org/abs/1806.06988" target="_blank" rel="noreferrer">Deep Neural Decision Trees</a></p></li><li><p><a href="https://arxiv.org/abs/1603.02754" target="_blank" rel="noreferrer">XGBoost: A Scalable Tree Boosting System</a></p></li><li><p><a href="https://arxiv.org/abs/1810.11363" target="_blank" rel="noreferrer">CatBoost: gradient boosting with categorical features support</a></p></li><li><p><a href="https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html" target="_blank" rel="noreferrer">LightGBM: A Highly Efficient Gradient Boosting Decision Tree</a></p></li><li><p><a href="https://arxiv.org/abs/1806.06988" target="_blank" rel="noreferrer">Deep Neural Decision Trees</a></p></li><li><p><a href="https://arxiv.org/abs/1702.07360" target="_blank" rel="noreferrer">Neural Decision Trees</a></p></li></ul>`,57),h=[l];function o(d,p,g,k,c,y){return i(),e("div",null,h)}const u=t(r,[["render",o]]);export{f as __pageData,u as default};
