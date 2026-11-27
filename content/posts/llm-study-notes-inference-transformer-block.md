---
title: "第四步：Transformer 块，逐个拆"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Transformer", "RMSNorm", "RoPE", "注意力", "FFN"]
description: "大模型推理第四步：Transformer 块逐个拆，RMSNorm、RoPE、注意力、FFN 都在这一页。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第四步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第三步：最基础的两个模块](/posts/llm-study-notes-inference-linear-embedding/)。

第二步我们看了 Transformer 整体长什么样，现在把一个块拆开，一个模块一个模块讲透，**都在这一页**，不用跳来跳去。

## 一个块里有什么

一个 Transformer 块（Pre-Norm 结构）是两条子层，每条子层前面一个 RMSNorm，后面一个残差相加：

| 顺序 | 模块 | 干什么 |
| --- | --- | --- |
| ① | RMSNorm | 归一化，把每个 token 的向量缩放到合适大小 |
| ② | 多头自注意力 | token 之间互相看，交换信息 |
| ③ | ＋x（残差） | 把输入加回来，形状不变 |
| ④ | RMSNorm | 再归一化一次 |
| ⑤ | FFN（SwiGLU） | 逐个 token 独立加工，升维再降维 |
| ⑥ | ＋x（残差） | 再把输入加回来，形状不变 |

形状全程不变：输入 3 行 × 4 列，输出还是 3 行 × 4 列。


## 4.1 RMSNorm：均方根归一化

原始 Transformer 用层归一化（LayerNorm），我们用它的简化版：均方根归一化（RMSNorm）。先把一句话记下来：

**RMSNorm 就是把每个 token 的向量，除以它自己的均方根，再乘一个可学习的增益 $g$。**

### 公式，一个符号一个符号讲

给定激活向量 $\boldsymbol{a} \in \mathbb{R}^{d_{model}}$，RMSNorm 对每个激活分量 $a_i$ 的缩放计算如下：

$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(\boldsymbol{a})} g_i$

其中：

$\text{RMS}(\boldsymbol{a}) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \epsilon}$

| 符号 | 是什么 | 记住它 |
| --- | --- | --- |
| $\boldsymbol{a}$ | 一个 token 的向量，$d_{model}$ 个数 | 就是要归一化的那一行 |
| $a_i$ | 向量里的第 $i$ 个数 | 逐个算，每个数单独除 |
| $a_i^2$ | 每个数先平方 | 平方是为了让正负都变成正，大的更大 |
| $\frac{1}{d_{model}} \sum a_i^2$ | 平方之后求平均 | 这一行整体有多大 |
| $\epsilon$ | 很小的数，通常固定取 $10^{-5}$ | 防止除以 0，平时可以忽略 |
| $\text{RMS}(\boldsymbol{a})$ | 上面那个平均数开根号 | 这一行的“均方根”，一个数 |
| $g_i$ | 可学习的增益参数，$d_{model}$ 个 | 模型自己学，决定每个维度放大还是缩小 |

### 输入输出长什么样

|  | 形状 | 说明 |
| --- | --- | --- |
| 输入张量 x | (batch_size, sequence_length, d_model) | 一批数据，每批几个样本，每个样本几个 token，每个 token 几维 |
| 输出张量 x_norm | (batch_size, sequence_length, d_model) | 形状完全不变，只改数值大小 |

**形状不变**，这是归一化最重要的一点：进来什么形状，出去还是什么形状。

### 工作流程：拿一个 4×4 的张量走一遍

输入张量 X，4 个 token，每个 4 维：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| Token 0 | 2 | 2 | -2 | -2 |
| Token 1 | 0 | 3 | 0 | 4 |
| Token 2 | 1 | 1 | 1 | 1 |
| Token 3 | 10 | 20 | 20 | 10 |

**先看一眼**：Token 3 那一行是 10、20、20、10，比别人大一个数量级。如果不处理，它进注意力一乘就更大，梯度直接炸掉。这就是 RMSNorm 要解决的问题。

### 演示：点任意一行，看它怎么算

下面是完整的 4 行，每行点一下，就能看到这一行的平方、均值、均方根、除完的结果。也可以点“播放”，一行一行自动走。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="rms-anim" style="overflow-x:auto;"></div>
<p id="rms-explain">点下面任意一行，或者点“播放”。</p>
<button id="rms-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="rms-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一行</button>
<button id="rms-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="rms-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var X=[[2,2,-2,-2],[0,3,0,4],[1,1,1,1],[10,20,20,10]];
  var names=['Token 0','Token 1','Token 2','Token 3'];
  var sq=X.map(function(r){ return r.map(function(v){return v*v;}); });
  var mean=X.map(function(r,i){ return sq[i].reduce(function(a,b){return a+b;},0)/4; });
  var rms=mean.map(function(m){ return Math.sqrt(m+1e-5); });
  var out=X.map(function(r,i){ return r.map(function(v){ return Math.round(v/rms[i]*100)/100; }); });
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function render(){
    var html='<table style="border-collapse:collapse;text-align:center;">';
    html+='<tr><td style="padding:4px;">输入 X</td><td style="padding:4px;">平方</td><td style="padding:4px;">均值</td><td style="padding:4px;">均方根</td><td style="padding:4px;">除完 x′</td></tr>';
    for(var i=0;i<4;i++){
      var done = idx>=i, cur = idx===i;
      html+='<tr data-i="'+i+'" style="cursor:pointer;">';
      html+='<td style="border:1px solid #888;padding:8px 12px;background:'+(cur?'#facc15':(done?'#2563eb':'#1f2937'))+';color:'+(cur?'#000':'#fff')+'">'+names[i]+'：'+X[i].join(', ')+'</td>';
      html+='<td style="border:1px solid #888;padding:8px 12px;">'+(done?sq[i].join(', '):'?')+'</td>';
      html+='<td style="border:1px solid #888;padding:8px 12px;">'+(done?Math.round(mean[i]*100)/100:'?')+'</td>';
      html+='<td style="border:1px solid #888;padding:8px 12px;">'+(done?Math.round(rms[i]*100)/100:'?')+'</td>';
      html+='<td style="border:1px solid #888;padding:8px 12px;">'+(done?out[i].join(', '):'?')+'</td>';
      html+='</tr>';
    }
    html+='</table>';
    document.getElementById('rms-anim').innerHTML=html;
    var ex=document.getElementById('rms-explain');
    if(idx<0){ ex.textContent='点下面任意一行，或者点“播放”。'; return; }
    ex.textContent=names[idx]+'：平方 ['+sq[idx].join(', ')+']，均值 = '+Math.round(mean[idx]*100)/100+'，均方根 ≈ '+Math.round(rms[idx]*100)/100+'，每格除以它，得到 ['+out[idx].join(', ')+']。';
    document.querySelectorAll('#rms-anim tr[data-i]').forEach(function(tr){
      tr.addEventListener('click', function(){ stop(); idx=+tr.getAttribute('data-i'); render(); });
    });
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('rms-play').textContent='▶ 播放'; }
  function step(){ if(idx>=3){ stop(); return; } idx++; render(); if(idx>=3) stop(); }
  document.getElementById('rms-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=3){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 1200);
  });
  document.getElementById('rms-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('rms-all').addEventListener('click', function(){ stop(); idx=3; render(); });
  document.getElementById('rms-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

### 算完长什么样：前后对比

|  | 归一化前 | 均方根 | 归一化后 |
| --- | --- | --- | --- |
| Token 0 | 2, 2, -2, -2 | 2.00 | 1.0, 1.0, -1.0, -1.0 |
| Token 1 | 0, 3, 0, 4 | 2.50 | 0.0, 1.2, 0.0, 1.6 |
| Token 2 | 1, 1, 1, 1 | 1.00 | 1.0, 1.0, 1.0, 1.0 |
| Token 3 | 10, 20, 20, 10 | 15.81 | 0.63, 1.26, 1.26, 0.63 |

看 Token 3 那一行最清楚：**归一化前是 10、20，归一化后是 0.63、1.26**，一下子回到 1 附近。每一行内部的比例没变（20 还是 10 的两倍），但整行的大小被拉平了。

### 最后一步：乘上可学习的增益 g

归一化之后，再逐元素乘一个可学习的参数 $g$（$d_{model}$ 个数，形状和 x 的最后一维一样）：

$Y = X_{norm} \odot g$

$g$ 是模型自己学出来的：归一化把所有维度拉到同一量级，$g$ 再决定每个维度放大还是缩小。推理时 $g$ 是固定的数。上面演示里 $g$ 都是 1，所以 $Y = X_{norm}$；真实模型里 $g$ 每个数都不一样。

### RMSNorm 到底好在哪

1. **防止数值爆炸**：Token 3 那行 10、20 不处理，进注意力一乘就更大，梯度传着传着就炸了。RMSNorm 把每行都拉到 1 附近，有效防止深层网络的前向传播数值失控。
2. **比 LayerNorm 简单**：LayerNorm 要先减均值再除以标准差，RMSNorm 省略了计算去均值的步骤，少一次计算，效果差不多，LLaMA 这些模型都用它。

### 和 LayerNorm 的区别，一句话

|  | LayerNorm | RMSNorm |
| --- | --- | --- |
| 减均值 | 要 | 不要 |
| 除什么 | 标准差 | 均方根 |
| 计算量 | 多一步 | 少一步 |
| 现在谁用 | 原始 Transformer | LLaMA、GPT 等现代模型 |

### 自测：RMSNorm 过关了吗

**第 1 题**：Token 1 [0, 3, 0, 4] 的均方根是多少？归一化后第一格是多少？

<details>
<summary>点开看解析和答案</summary>

平方是 [0, 9, 0, 16]，均值 = 25 / 4 = 6.25，均方根 = √6.25 = 2.5。归一化后第一格 = 0 / 2.5 = 0，第二格 = 3 / 2.5 = 1.2。

</details>

**第 2 题**：Token 3 归一化前是 10、20，归一化后是 0.63、1.26，比例变了吗？大小变了吗？

<details>
<summary>点开看解析和答案</summary>

比例没变：20 还是 10 的两倍，1.26 还是 0.63 的两倍。大小变了：整行除以均方根 15.81，从 10、20 拉回 1 附近。这就是归一化干的事：改大小，不改比例。

</details>

**第 3 题**：RMSNorm 是按行算还是按列算？token 之间会互相影响吗？

<details>
<summary>点开看解析和答案</summary>

按行算，每个 token 独立归一化。token 之间互不影响，交换信息是注意力干的事，不是归一化。

</details>

**第 4 题**：归一化之后形状变了吗？$g$ 是干什么的？

<details>
<summary>点开看解析和答案</summary>

没变。输入 (batch, seq, d_model)，输出还是 (batch, seq, d_model)。$g$ 是可学习的参数，归一化之后逐元素乘它，模型可以自己决定每个维度放大还是缩小。推理时它是固定的数。

</details>

**第 5 题**：RMSNorm 和 LayerNorm 有什么区别？为什么现代模型用 RMSNorm？

<details>
<summary>点开看解析和答案</summary>

LayerNorm 先减均值再除以标准差，RMSNorm 不减均值，只除以均方根。RMSNorm 少一次计算，效果差不多，现代模型都用它。

</details>

全答得出来，4.1 就过了。

## 4.2 RoPE：旋转位置编码，给 Q、K 转个角度

注意力本身不知道词的先后顺序：“猫吃鱼”和“鱼吃猫”算出来完全一样，只是换了行。所以要给位置信息。

LLaMA 这些模型用 RoPE：**不在输入上加位置向量，而是在算 Q、K 的时候，把每一对数转一个角度**，角度由位置决定。

### 旋转是怎么转的

把向量的每两个数看成一对 $(x, y)$，转角度 $\theta$：

$x' = x \cos\theta - y \sin\theta$
$y' = x \sin\theta + y \cos\theta$

$\theta$ 由位置 pos 和这一对是第几对决定：$\theta = pos \times 10000^{-2i/d}$。d=4 时，第 0 对 $\theta = pos \times 1$，第 1 对 $\theta = pos \times 0.01$。位置越大，转得越多；第 0 对转得快，第 1 对转得慢。

关键性质：**相对位置决定相对角度**。两个词位置差多少，它们的 Q、K 之间就差多少角度，点积只和相对位置有关，和绝对位置无关。

### 动画：给“吃”的 Q 转一度

“吃”在位置 1，Q 的第一对是 [0, 2]，转 $\theta = 1$ 弧度（cos 1 ≈ 0.54，sin 1 ≈ 0.84）。点“播放”看它怎么转。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="rope-anim" style="overflow-x:auto;"></div>
<p id="rope-explain">点“播放”开始。</p>
<button id="rope-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="rope-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="rope-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="rope-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var steps=[
    {t:'cos', v:'cos 1 ≈ 0.54，sin 1 ≈ 0.84。位置 1，第 0 对，角度就是 1 弧度。'},
    {t:'x', v:"x' = x cosθ - y sinθ = 0×0.54 - 2×0.84 = -1.68。"},
    {t:'y', v:"y' = x sinθ + y cosθ = 0×0.84 + 2×0.54 = 1.08。"},
    {t:'done', v:'转完了：[0, 2] → [-1.68, 1.08]。长度没变（√(0²+2²)=2，√(1.68²+1.08²)=2），只是方向转了。'}
  ];
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:10px 14px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var html='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;"><tr><td style="padding:4px;">转前 q =</td>'+cell(0,'')+cell(2,'')+'</tr>';
    html+='<tr><td style="padding:4px;">转后 q′ =</td>';
    html+=cell(s&&idx>=1?-1.68:'?', s&&idx>=1?'done':'');
    html+=cell(s&&idx>=2?1.08:'?', s&&idx>=2?'done':'');
    html+='</tr></table>';
    document.getElementById('rope-anim').innerHTML=html;
    document.getElementById('rope-explain').textContent = s ? s.v : '点“播放”开始。';
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('rope-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('rope-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 1000);
  });
  document.getElementById('rope-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('rope-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('rope-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

位置 0 的词角度是 0，不转；位置 1 转 1 弧度；位置 2 转 2 弧度。第二对转得慢，位置 1 只转 0.01 弧度，[1, 1] → [0.99, 1.01]，几乎不动。

### 和原始位置编码的区别

|  | 原始位置编码 | RoPE |
| --- | --- | --- |
| 在哪加 | 输入上加一个位置向量 P | 算 Q、K 时旋转 |
| 相对位置 | 靠模型自己学 | 天生就是相对的，点积只看位置差 |
| 现在谁用 | 原始 Transformer | LLaMA、GPT 等现代模型 |

### 自测：RoPE 过关了吗

**第 1 题**：RoPE 是在哪里加位置信息的？和原始位置编码有什么不同？

<details>
<summary>点开看解析和答案</summary>

原始位置编码在输入上加一个位置向量 P；RoPE 不在输入上加，而是在算 Q、K 的时候把每对数旋转一个角度，角度由位置决定。

</details>

**第 2 题**：[0, 2] 转 1 弧度之后是多少？长度变了吗？

<details>
<summary>点开看解析和答案</summary>

x' = 0×0.54 - 2×0.84 = -1.68，y' = 0×0.84 + 2×0.54 = 1.08，变成 [-1.68, 1.08]。长度没变，都是 2，旋转只改方向不改长度。

</details>

**第 3 题**：为什么说 RoPE 的点积只和相对位置有关？

<details>
<summary>点开看解析和答案</summary>

两个词的 Q、K 各转了自己的角度，点积里的角度差就是它们的位置差。绝对位置同时加多少，角度差不变，所以点积只看相对位置。

</details>

## 4.3 多头自注意力：QKV、分数、遮罩、加权求和、拼接

Q、K、V 都是 X 乘一个权重矩阵得到的，都是 Linear（第三步讲过的）：$Q = X W_Q^T$，$K = X W_K^T$，$V = X W_V^T$。

- **Q（查询）**：每个词问“我要找什么”
- **K（键）**：每个词说“我这里有什么”
- **V（值）**：每个词真正要传出去的内容

### 动画 1：Q = X × W_Q^T

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="qkv-q-anim" style="overflow-x:auto;"></div>
<p id="qkv-q-explain">点“播放”开始。</p>
<button id="qkv-q-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="qkv-q-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="qkv-q-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="qkv-q-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>

### 动画 2：K = X × W_K^T

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="qkv-k-anim" style="overflow-x:auto;"></div>
<p id="qkv-k-explain">点“播放”开始。</p>
<button id="qkv-k-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="qkv-k-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="qkv-k-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="qkv-k-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>

### 动画 3：V = X × W_V^T

这里 W_V 是单位矩阵，所以 V = X，直接看结果：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 0 | 1 | 0 |
| 吃 | 0 | 1 | 0 | 1 |
| 鱼 | 1 | 1 | 0 | 0 |

算出来的 Q 和 K 是：

Q（3 行 × 4 列）：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 2 | 0 | 1 | 1 |
| 吃 | 0 | 2 | 1 | 1 |
| 鱼 | 1 | 1 | 1 | 1 |

K（3 行 × 4 列）：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 1 | 1 | 1 |
| 吃 | 1 | 1 | 1 | 1 |
| 鱼 | 2 | 1 | 0 | 1 |

<script>
function matAnim(p, A, BT, C, names){
  // A (m x n) x BT (n x p) = C (m x p); BT is already transposed
  var m=A.length, n=A[0].length, pc=BT[0].length;
  var steps=[];
  for(var i=0;i<m;i++) for(var j=0;j<pc;j++){
    for(var k=0;k<n;k++) steps.push({type:'mul', i:i, j:j, k:k});
    steps.push({type:'sum', i:i, j:j});
  }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, hi, hj, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+(names?names[r]:'')+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(hi!==null && hi.r===r && hi.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var aHi=null, bHi=null, cCur=null, cDone={};
    if(s){
      aHi={r:s.i, c:s.k}; bHi={r:s.k, c:s.j}; cCur={r:s.i, c:s.j};
      for(var t=0;t<=idx;t++){ var st=steps[t]; if(st.type==='sum') cDone[st.i+','+st.j]=1; }
    }
    var html=mat(A, aHi, null, null);
    html+='<span style="margin:0 8px;">×</span>';
    html+=mat(BT, bHi, null, null);
    html+='<span style="margin:0 8px;">=</span>';
    html+=mat(C, cCur, null, cDone);
    document.getElementById(p+'-anim').innerHTML=html;
    var ex=document.getElementById(p+'-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mul'){
      ex.textContent='算第 '+(s.i+1)+' 行第 '+(s.j+1)+' 列：'+A[s.i][s.k]+'×'+BT[s.k][s.j]+' = '+A[s.i][s.k]*BT[s.k][s.j]+'。黄色是正在乘的两个数。';
    } else {
      var parts=[]; for(var k2=0;k2<n;k2++) parts.push(A[s.i][k2]+'×'+BT[k2][s.j]);
      ex.textContent='第 '+(s.i+1)+' 行第 '+(s.j+1)+' 列 = '+parts.join(' + ')+' = '+C[s.i][s.j]+'。';
    }
    // make result cells clickable to jump
    var tbls=document.getElementById(p+'-anim').querySelectorAll('table');
    var ct=tbls[tbls.length-1];
    ct.querySelectorAll('tr').forEach(function(tr, r){
      tr.querySelectorAll('td').forEach(function(td, c){
        if(c===0) return;
        var cc=c-1;
        td.style.cursor='pointer';
        td.onclick=function(){
          stop();
          for(var t2=0;t2<steps.length;t2++){ if(steps[t2].type==='sum'&&steps[t2].i===r&&steps[t2].j===cc){ idx=t2; break; } }
          render();
        };
      });
    });
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById(p+'-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById(p+'-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById(p+'-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById(p+'-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById(p+'-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
}
(function(){
  var X=[[1,0,1,0],[0,1,0,1],[1,1,0,0]];
  var names=['猫','吃','鱼'];
  var WqT=[[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]];
  var WkT=[[1,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1]];
  var Q=[[2,0,1,1],[0,2,1,1],[1,1,1,1]];
  var K=[[1,1,1,1],[1,1,1,1],[2,1,0,1]];
  matAnim('qkv-q', X, WqT, Q, names);
  matAnim('qkv-k', X, WkT, K, names);
})();
</script>

## 第二步：注意力分数，谁在看谁

每个头单独算。头 1 用 Q、K 的第 1–2 列：$S_1 = Q_1 K_1^T \div \sqrt{d_h}$，$d_h = 2$，$\sqrt{2} \approx 1.41$。

形状：(3 行 × 2 列) × (2 行 × 3 列) → 3 行 × 3 列。行是“谁在看”，列是“被看的是谁”。分数越大，第 i 个词越“关注”第 j 个词。

除以 $\sqrt{d_h}$ 是为了防止点积太大，让后面的 Softmax 不至于太尖锐。

### 动画 4：头 1 的分数 S₁ = Q₁ × K₁^T ÷ √2

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="score-anim" style="overflow-x:auto;"></div>
<p id="score-explain">点“播放”开始。</p>
<button id="score-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="score-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="score-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="score-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var Q1=[[2,0],[0,2],[1,1]], K1T=[[1,1,2],[1,1,1]];
  var S1=[[1.41,1.41,2.83],[1.41,1.41,1.41],[1.41,1.41,2.12]];
  var names=['猫','吃','鱼'];
  var steps=[];
  for(var i=0;i<3;i++) for(var j=0;j<3;j++){
    for(var k=0;k<2;k++) steps.push({type:'mul', i:i, j:j, k:k});
    steps.push({type:'sum', i:i, j:j});
  }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, hi, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(hi && hi.r===r && hi.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var aHi=null, bHi=null, cDone={};
    if(s){ aHi={r:s.i, c:s.k}; bHi={r:s.k, c:s.j}; for(var t=0;t<=idx;t++){ var st=steps[t]; if(st.type==='sum') cDone[st.i+','+st.j]=1; } }
    var html=mat(Q1, aHi, null)+'<span style="margin:0 8px;">×</span>'+mat(K1T, bHi, null)+'<span style="margin:0 8px;">÷ √2 =</span>'+mat(S1, s?{r:s.i,c:s.j}:null, cDone);
    document.getElementById('score-anim').innerHTML=html;
    var ex=document.getElementById('score-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mul') ex.textContent='算 S['+s.i+']['+s.j+']：'+Q1[s.i][s.k]+'×'+K1T[s.k][s.j]+' = '+Q1[s.i][s.k]*K1T[s.k][s.j]+'。';
    else {
      var raw=0; for(var k2=0;k2<2;k2++) raw+=Q1[s.i][k2]*K1T[k2][s.j];
      ex.textContent='S['+s.i+']['+s.j+'] = '+raw+' ÷ 1.41 = '+S1[s.i][s.j]+'。第 '+(s.i+1)+' 个词对第 '+(s.j+1)+' 个词的关注分数。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('score-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('score-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById('score-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('score-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('score-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

头 1 的分数（除以 √2 之后）：

|  | 猫 | 吃 | 鱼 |
| --- | --- | --- | --- |
| 猫 | 1.41 | 1.41 | 2.83 |
| 吃 | 1.41 | 1.41 | 1.41 |
| 鱼 | 1.41 | 1.41 | 2.12 |

“猫”最关注“鱼”（2.83），“鱼”也最关注“鱼”（2.12）。头 2 同理，算法一模一样，只是用的列不同。

## 第三步：遮罩 + Softmax，变成权重

解码器是自回归的：生成第 i 个词时，只能看前面的词，不能看后面的。所以分数矩阵的上三角要遮掉，变成 -∞，Softmax 之后就是 0。

### 动画 5：遮罩 + Softmax，一行一行算

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="mask-anim" style="overflow-x:auto;"></div>
<p id="mask-explain">点“播放”开始。</p>
<button id="mask-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="mask-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="mask-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="mask-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var S=[[1.41,1.41,2.83],[1.41,1.41,1.41],[1.41,1.41,2.12]];
  var names=['猫','吃','鱼'];
  var W=[[1,0,0],[0.5,0.5,0],[0.25,0.25,0.5]];
  // steps: for each row: mask step, then softmax step
  var steps=[];
  for(var i=0;i<3;i++){ steps.push({type:'mask', i:i}); steps.push({type:'soft', i:i}); }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : hl==='mask' ? '#7f1d1d' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    function mat(M, rowHi, doneRows){
      var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
      for(var r=0;r<3;r++){
        h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
        for(var c=0;c<3;c++){
          var hl='';
          if(rowHi===r) hl='cur';
          else if(doneRows && doneRows.indexOf(r)>=0) hl='done';
          h+=cell(M[r][c], hl);
        }
        h+='</tr>';
      }
      return h+'</table>';
    }
    var masked=S.map(function(row,i){ return row.map(function(v,j){ return j<=i ? v : '-∞'; }); });
    var html=mat(S, s&&s.type==='mask'?s.i:null, null);
    html+='<span style="margin:0 8px;">→ 遮罩 →</span>'+mat(masked, s&&s.type==='mask'?s.i:null, null);
    html+='<span style="margin:0 8px;">→ Softmax →</span>'+mat(W, s&&s.type==='soft'?s.i:null, null);
    document.getElementById('mask-anim').innerHTML=html;
    var ex=document.getElementById('mask-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mask'){
      var kept=[]; for(var j=0;j<=s.i;j++) kept.push(S[s.i][j]);
      ex.textContent='第 '+(s.i+1)+' 行遮罩：只能看前 '+(s.i+1)+' 个词，后面的变成 -∞（红色）。保留 '+kept.join(', ')+'。';
    } else {
      var row=W[s.i].filter(function(v){return v>0;});
      ex.textContent='第 '+(s.i+1)+' 行 Softmax：e 的指数归一化，得到权重 ['+W[s.i].join(', ')+']，一行加起来是 1。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('mask-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('mask-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 1000);
  });
  document.getElementById('mask-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('mask-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('mask-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

遮罩后的权重（头 1）：

|  | 猫 | 吃 | 鱼 |
| --- | --- | --- | --- |
| 猫 | 1 | 0 | 0 |
| 吃 | 0.5 | 0.5 | 0 |
| 鱼 | 0.25 | 0.25 | 0.5 |

每一行加起来都是 1。“猫”只能看自己，“吃”看猫和自己各一半，“鱼”看猫、吃各 0.25，看自己 0.5。

## 第四步：加权求和，拿权重去乘 V

输出 = 权重 × V。每个词的新表示，就是它关注的那些词的 V 按权重加权平均。

### 动画 6：头 1 输出 = 权重 × V₁

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="ws-anim" style="overflow-x:auto;"></div>
<p id="ws-explain">点“播放”开始。</p>
<button id="ws-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="ws-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="ws-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="ws-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var W=[[1,0,0],[0.5,0.5,0],[0.25,0.25,0.5]];
  var V1=[[1,0],[0,1],[1,1]];
  var O1=[[1,0],[0.5,0.5],[0.75,0.75]];
  var names=['猫','吃','鱼'];
  var steps=[];
  for(var i=0;i<3;i++) for(var j=0;j<2;j++){
    for(var k=0;k<3;k++) steps.push({type:'mul', i:i, j:j, k:k});
    steps.push({type:'sum', i:i, j:j});
  }
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, hi, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(hi && hi.r===r && hi.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var aHi=null, bHi=null, cDone={};
    if(s){ aHi={r:s.i, c:s.k}; bHi={r:s.k, c:s.j}; for(var t=0;t<=idx;t++){ var st=steps[t]; if(st.type==='sum') cDone[st.i+','+st.j]=1; } }
    var html=mat(W, aHi, null)+'<span style="margin:0 8px;">×</span>'+mat(V1, bHi, null)+'<span style="margin:0 8px;">=</span>'+mat(O1, s?{r:s.i,c:s.j}:null, cDone);
    document.getElementById('ws-anim').innerHTML=html;
    var ex=document.getElementById('ws-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    if(s.type==='mul') ex.textContent='权重 '+W[s.i][s.k]+' × V['+s.k+']['+s.j+'] = '+W[s.i][s.k]*V1[s.k][s.j]+'。';
    else {
      var parts=[]; for(var k2=0;k2<3;k2++) parts.push(W[s.i][k2]+'×'+V1[k2][s.j]);
      ex.textContent='输出['+s.i+']['+s.j+'] = '+parts.join(' + ')+' = '+O1[s.i][s.j]+'。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('ws-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('ws-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById('ws-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('ws-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('ws-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

头 1 输出（3 行 × 2 列）：

|  | 1 | 2 |
| --- | --- | --- |
| 猫 | 1 | 0 |
| 吃 | 0.5 | 0.5 |
| 鱼 | 0.75 | 0.75 |

头 2 同理，输出是 [[1, 0], [0.5, 0.5], [0.4, 0.4]]。

## 第五步：拼接 + 投影，拼回 4 维

两个头的输出拼起来，3 行 × 2 列 + 3 行 × 2 列 = 3 行 × 4 列，再乘一个投影矩阵 $W_O$（这里是单位矩阵，所以结果不变）：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 猫 | 1 | 0 | 1 | 0 |
| 吃 | 0.5 | 0.5 | 0.5 | 0.5 |
| 鱼 | 0.75 | 0.75 | 0.4 | 0.4 |

这就是多头注意力的输出，形状和输入一样，都是 3 行 × 4 列，可以直接加残差进下一层。

### 动画 7：拼接，一格一格拼

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="cat-anim" style="overflow-x:auto;"></div>
<p id="cat-explain">点“播放”开始。</p>
<button id="cat-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="cat-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="cat-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="cat-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var O1=[[1,0],[0.5,0.5],[0.75,0.75]], O2=[[1,0],[0.5,0.5],[0.4,0.4]];
  var O=[[1,0,1,0],[0.5,0.5,0.5,0.5],[0.75,0.75,0.4,0.4]];
  var names=['猫','吃','鱼'];
  var steps=[];
  for(var i=0;i<3;i++) for(var j=0;j<4;j++) steps.push({i:i, j:j});
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, cur, doneSet){
    var h='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++){
        var hl='';
        if(cur && cur.r===r && cur.c===c) hl='cur';
        else if(doneSet && doneSet[r+','+c]) hl='done';
        h+=cell(M[r][c], hl);
      }
      h+='</tr>';
    }
    return h+'</table>';
  }
  function render(){
    var s = idx>=0 ? steps[idx] : null;
    var done={};
    for(var t=0;t<=idx;t++) done[steps[t].i+','+steps[t].j]=1;
    var html=mat(O1, null, null)+'<span style="margin:0 8px;">+</span>'+mat(O2, null, null)+'<span style="margin:0 8px;">=</span>'+mat(O, s, done);
    document.getElementById('cat-anim').innerHTML=html;
    var ex=document.getElementById('cat-explain');
    if(!s){ ex.textContent='点“播放”开始。'; return; }
    var src = s.j<2 ? '头 1' : '头 2';
    ex.textContent='输出['+s.i+']['+s.j+'] = '+O[s.i][s.j]+'，来自'+src+'的第 '+(s.j<2?s.j+1:s.j-1)+' 列。两个头拼起来，2 列 + 2 列 = 4 列。';
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('cat-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('cat-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 600);
  });
  document.getElementById('cat-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('cat-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('cat-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

## 自测：注意力过关了吗

### 第 1 题

Q、K、V 分别是干什么的？都是怎么算出来的？

<details>
<summary>点开看解析和答案</summary>

Q 是查询，K 是键，V 是值。都是 X 乘一个权重矩阵得到的 Linear：$Q = X W_Q^T$，$K = X W_K^T$，$V = X W_V^T$。形状都是 3 行 × 4 列。

</details>

### 第 2 题

注意力分数 S = QKᵀ ÷ √d 是怎么算的？行和列分别是什么意思？

<details>
<summary>点开看解析和答案</summary>

每个头单独算，头 1 用 Q、K 的第 1–2 列，(3×2) × (2×3) → 3×3。行是“谁在看”，列是“被看的是谁”，分数越大越关注。除以 √2 是防止点积太大，Softmax 太尖锐。

</details>

### 第 3 题

遮罩是干什么的？为什么“猫”那一行只有第一个数不是 0？

<details>
<summary>点开看解析和答案</summary>

解码器自回归，生成第 i 个词只能看前面的词，不能看后面的。遮罩把上三角变成 -∞，Softmax 后就是 0。“猫”是第 1 个词，前面没有词，只能看自己，所以权重是 [1, 0, 0]。

</details>

### 第 4 题

加权求和是怎么算的？“鱼”那一行的输出 0.75 是怎么来的？

<details>
<summary>点开看解析和答案</summary>

输出 = 权重 × V。鱼的权重是 [0.25, 0.25, 0.5]，V₁ 的第一列是 [1, 0, 1]，所以 0.25×1 + 0.25×0 + 0.5×1 = 0.75。

</details>

### 第 5 题

为什么要多头？两个头拼起来之后形状是多少？

<details>
<summary>点开看解析和答案</summary>

每个头关注的东西不一样，一个头看语法，一个头看语义，合起来更全面。两个头各输出 3 行 × 2 列，拼起来是 3 行 × 4 列，再乘投影矩阵 $W_O$，形状不变。

</details>

全答得出来，4.3 就过了。

## 4.4 FFN（SwiGLU）：前馈网络，逐个 token 独立加工

注意力是让 token 之间互相看，FFN 是每个 token 自己加工自己，token 之间互不影响。公式是：

$\text{SwiGLU}(x) = (\text{SiLU}(x W_1) \odot x W_3) W_2$

- $x W_1$：门，升维
- $x W_3$：上投影，升维
- $\text{SiLU}(z) = z \times \text{sigmoid}(z)$：激活函数，$z$ 越大越接近 $z$，$z$ 越小越接近 0
- $\odot$：逐元素相乘，门决定哪些信息放过去
- $W_2$：降维，回到 d_model

形状：4 维 → 升到 2 维（玩具模型，真模型是升到几倍）→ 再降回 4 维。先升维再降维，中间有非线性，模型才能学复杂的东西。

### 动画：一个 token 走完 FFN

拿一个 token x = [1, 2, 0.5, -1]，点“播放”看它一步步走。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="ffn-anim" style="overflow-x:auto;"></div>
<p id="ffn-explain">点“播放”开始。</p>
<button id="ffn-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="ffn-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="ffn-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="ffn-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var stages=[
    {label:'x', vals:[1,2,0.5,-1], text:'输入 x = [1, 2, 0.5, -1]，4 维。'},
    {label:'门 xW₁', vals:[1.5,2.5], text:'门 xW₁ = [1.5, 2.5]，升维到 2 维。'},
    {label:'SiLU(门)', vals:[1.23,2.31], text:'SiLU 激活：1.5×sigmoid(1.5) ≈ 1.23，2.5×sigmoid(2.5) ≈ 2.31。'},
    {label:'上投影 xW₃', vals:[1,1.5], text:'上投影 xW₃ = [1, 1.5]，也是 2 维。'},
    {label:'门 × 上投影', vals:[1.23,3.46], text:'逐元素相乘：1.23×1 = 1.23，2.31×1.5 = 3.46。门决定放多少过去。'},
    {label:'输出 ×W₂', vals:[1.23,3.46,1.23,3.46], text:'再乘 W₂ 降维回 4 维：输出 [1.23, 3.46, 1.23, 3.46]。'}
  ];
  var idx=-1, timer=null, playing=false;
  function render(){
    var html='<table style="border-collapse:collapse;text-align:center;">';
    for(var i=0;i<stages.length;i++){
      var on = idx>=i;
      var cur = idx===i;
      html+='<tr style="background:'+(cur?'#facc15':(on?'#2563eb':'#1f2937'))+';color:'+(cur?'#000':'#fff')+'"><td style="border:1px solid #888;padding:8px 12px;">'+stages[i].label+'</td>';
      html+=stages[i].vals.map(function(v){return '<td style="border:1px solid #888;padding:8px 12px;">'+(on?v:'?')+'</td>';}).join('');
      html+='</tr>';
    }
    html+='</table>';
    document.getElementById('ffn-anim').innerHTML=html;
    document.getElementById('ffn-explain').textContent = idx>=0 ? stages[idx].text : '点“播放”开始。';
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('ffn-play').textContent='▶ 播放'; }
  function step(){ if(idx>=stages.length-1){ stop(); return; } idx++; render(); if(idx>=stages.length-1) stop(); }
  document.getElementById('ffn-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=stages.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 1000);
  });
  document.getElementById('ffn-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('ffn-all').addEventListener('click', function(){ stop(); idx=stages.length-1; render(); });
  document.getElementById('ffn-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

### 自测：FFN 过关了吗

**第 1 题**：SwiGLU 里三条路分别是什么？门是干什么的？

<details>
<summary>点开看解析和答案</summary>

$x W_1$ 是门，$x W_3$ 是上投影，两个都升维，SiLU 激活门之后逐元素相乘，再乘 $W_2$ 降维。门决定哪些信息放过去。

</details>

**第 2 题**：FFN 是按行算还是按列算？token 之间会互相影响吗？

<details>
<summary>点开看解析和答案</summary>

按行算，每个 token 独立加工，token 之间互不影响。交换信息是注意力干的事。

</details>

**第 3 题**：为什么要先升维再降维？

<details>
<summary>点开看解析和答案</summary>

升维之后有更多空间放中间结果，中间有 SiLU 非线性，模型才能学复杂的东西。再降回 d_model，形状不变，可以加残差。

</details>

---

下一步：[第五步：拼成完整的模型](/posts/llm-study-notes-inference/)（还没写，先回目录）

回目录：[大模型推理](/posts/llm-study-notes-inference/)
