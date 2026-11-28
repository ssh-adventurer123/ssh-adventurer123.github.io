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

### 为什么不用 BatchNorm

归一化还有一种叫 BatchNorm（批量归一化），在图像网络里很常见，但 Transformer 不用它。区别只在**沿哪个方向算均值**：

|  | BatchNorm | LayerNorm / RMSNorm |
| --- | --- | --- |
| 沿哪个方向算 | 沿 batch 方向：同一维（同一列），算所有 token 的均值 | 沿特征方向：一个 token 的一行，算它 4 个数的均值 |
| 依赖什么 | 依赖同一批里的其他样本 | 只依赖自己这一个 token |
| 推理时 batch=1 | 算不出来，1 个样本没有均值可算 | 没问题，每个 token 自己算自己的 |

点下面两个表，看均值是沿哪条线算的。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="bn-anim" style="overflow-x:auto;"></div>
<p id="bn-explain">点下面两个按钮。</p>
<button class="bn-btn" data-k="bn" style="padding:6px 12px;margin:2px;cursor:pointer;">BatchNorm：按列算</button>
<button class="bn-btn" data-k="ln" style="padding:6px 12px;margin:2px;cursor:pointer;">LayerNorm：按行算</button>
</div>
<script>
(function(){
  var data={
    bn:{t:'BatchNorm：算第 1 列的均值，要把猫、吃、鱼三个 token 的第 1 个数加起来：(2 + 0 + 1) / 3 = 1。均值依赖同一批里的其他 token，batch 变了，均值就变了。', hl:'col'},
    ln:{t:'LayerNorm：算猫这一行的均值，只用猫自己的 4 个数：(2 + 2 - 2 - 2) / 4 = 0。均值只依赖自己，batch 里有几个 token、句子多长，都不影响。', hl:'row'}
  };
  var rows=[[2,2,-2,-2],[0,3,0,4],[1,1,1,1]];
  function render(k){
    var d=data[k];
    var html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;"></td><td style="padding:6px;">第 1 维</td><td style="padding:6px;">第 2 维</td><td style="padding:6px;">第 3 维</td><td style="padding:6px;">第 4 维</td></tr>';
    rows.forEach(function(r,i){
      html+='<tr><td style="padding:6px;">'+['猫','吃','鱼'][i]+'</td>';
      r.forEach(function(v,j){
        var on = (d.hl==='col' && j===0) || (d.hl==='row' && i===0);
        html+='<td style="border:1px solid #888;padding:8px 12px;background:'+(on?'#2563eb':'#fff')+';color:'+(on?'#fff':'#000')+';">'+v+'</td>';
      });
      html+='</tr>';
    });
    html+='</table>';
    document.getElementById('bn-anim').innerHTML=html;
    document.getElementById('bn-explain').textContent=d.t;
  }
  document.querySelectorAll('.bn-btn').forEach(function(b){
    b.addEventListener('click', function(){ render(b.getAttribute('data-k')); });
  });
  render('bn');
})();
</script>

三个原因，一句话一个：

- **推理时 batch 经常是 1**：一个人问一句话，batch 里只有 1 个样本，BatchNorm 按列算均值，1 个数没有均值可算。
- **句子长短不一**：短句子要补 0（Padding），补的 0 会污染按列算的均值，LayerNorm 按行算，补 0 的位置根本不参与。
- **训练和推理不一致**：BatchNorm 训练用当前 batch 的均值，推理用训练时攒下来的全局均值，两边不一样；LayerNorm 训练推理完全一样，每个 token 自己算。

所以 Transformer 用 LayerNorm / RMSNorm，不用 BatchNorm。

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

先把一句话记下来：

**RoPE 不在输入上加位置向量，而是把 Q、K 的每两个数看成一对，转一个由位置决定的角度。**

注意力本身不知道词的先后顺序：“猫吃鱼”和“鱼吃猫”算出来完全一样，只是换了行。RoPE 解决的就是这个问题。

### 1. 公式描述：一个符号一个符号讲

位置 $i$ 处的查询向量 $\boldsymbol{q}^{(i)}$ 受到旋转变换：

$\boldsymbol{q}'^{(i)} = R^i \boldsymbol{q}^{(i)}$

旋转矩阵 $R^i$ 是**分块对角矩阵**，对角线上是一串 2×2 的小旋转块，其他位置都是 0：

<div style="overflow-x:auto;">
<table style="border-collapse:collapse;text-align:center;display:inline-table;">
<tr><td rowspan="5" style="border-left:2px solid #888;border-top:2px solid #888;border-bottom:2px solid #888;width:8px;"></td><td style="padding:8px 12px;"><i>R</i><sub>i,1</sub></td><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">⋯</td><td style="padding:8px 12px;">0</td><td rowspan="5" style="border-right:2px solid #888;border-top:2px solid #888;border-bottom:2px solid #888;width:8px;"></td></tr>
<tr><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;"><i>R</i><sub>i,2</sub></td><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">⋯</td><td style="padding:8px 12px;">0</td></tr>
<tr><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;"><i>R</i><sub>i,3</sub></td><td style="padding:8px 12px;">⋯</td><td style="padding:8px 12px;">0</td></tr>
<tr><td style="padding:8px 12px;">⋮</td><td style="padding:8px 12px;">⋮</td><td style="padding:8px 12px;">⋮</td><td style="padding:8px 12px;">⋱</td><td style="padding:8px 12px;">⋮</td></tr>
<tr><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">0</td><td style="padding:8px 12px;">⋯</td><td style="padding:8px 12px;"><i>R</i><sub>i,d/2</sub></td></tr>
</table>
</div>

$R^i$ = 上面这个大矩阵。对角线上的 $R_{i,1}$、$R_{i,2}$、$R_{i,3}$……一直到 $R_{i,d/2}$，每个都是一个 2×2 的小旋转块；对角线以外全是 0。

其中每个二维旋转子矩阵 $R_{i,k}$ 将元素两两分组进行旋转：

<div style="overflow-x:auto;">
<table style="border-collapse:collapse;text-align:center;display:inline-table;">
<tr><td rowspan="2" style="border-left:2px solid #888;border-top:2px solid #888;border-bottom:2px solid #888;width:8px;"></td><td style="padding:8px 12px;">cos θ<sub>i,k</sub></td><td style="padding:8px 12px;">sin θ<sub>i,k</sub></td><td rowspan="2" style="border-right:2px solid #888;border-top:2px solid #888;border-bottom:2px solid #888;width:8px;"></td></tr>
<tr><td style="padding:8px 12px;">−sin θ<sub>i,k</sub></td><td style="padding:8px 12px;">cos θ<sub>i,k</sub></td></tr>
</table>
</div>

对应的旋转角度 $\theta_{i,k}$ 计算公式：

$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$

| 符号 | 是什么 | 记住它 |
| --- | --- | --- |
| $i$ | 序列中的 token 下标（行） | 第几个词，0、1、2、3…… |
| $k$ | 特征维度两两分组后的组号（列） | 第几对，1、2…… |
| $d$ | 特征维度，这里是 4 | 一共几维 |
| $\Theta$ | 底数，通常取 10000 | 固定常数，不用学 |
| $\theta_{i,k}$ | 第 $i$ 个词、第 $k$ 对的旋转角度 | 位置越大，转得越多 |

### 2. 公式推导：4 维向量，一组一组转

以 4 维行向量 $\boldsymbol{q}^{(i)} = (q_0, q_1, q_2, q_3)$ 为例，分成两组：

**第 1 组（k=1）：** $(q_0, q_1)$ 转角度 $\theta_{i,1}$，行向量乘上面那个 2×2 矩阵：

<div style="overflow-x:auto;">
<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">
<tr><td style="padding:8px 12px;">(q<sub>0</sub>', q<sub>1</sub>')</td><td style="padding:8px 12px;">=</td><td style="padding:8px 12px;">(q<sub>0</sub>, q<sub>1</sub>)</td><td style="padding:8px 12px;">×</td>
</tr>
</table>
<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">
<tr><td rowspan="2" style="border-left:2px solid #888;border-top:2px solid #888;border-bottom:2px solid #888;width:8px;"></td><td style="padding:8px 12px;">cos θ<sub>i,1</sub></td><td style="padding:8px 12px;">sin θ<sub>i,1</sub></td><td rowspan="2" style="border-right:2px solid #888;border-top:2px solid #888;border-bottom:2px solid #888;width:8px;"></td></tr>
<tr><td style="padding:8px 12px;">−sin θ<sub>i,1</sub></td><td style="padding:8px 12px;">cos θ<sub>i,1</sub></td></tr>
</table>
<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">
<tr><td style="padding:8px 12px;">=</td><td style="padding:8px 12px;">(q<sub>0</sub> cos θ<sub>i,1</sub> − q<sub>1</sub> sin θ<sub>i,1</sub>, q<sub>0</sub> sin θ<sub>i,1</sub> + q<sub>1</sub> cos θ<sub>i,1</sub>)</td></tr>
</table>
</div>

**第 2 组（k=2）：** $(q_2, q_3)$ 转角度 $\theta_{i,2}$，算法一模一样，只是角度不同

$(q_2', q_3') = (q_2 \cos\theta_{i,2} - q_3 \sin\theta_{i,2},\; q_2 \sin\theta_{i,2} + q_3 \cos\theta_{i,2})$

最终拼回完整行向量：$\boldsymbol{q}'^{(i)} = (q_0', q_1', q_2', q_3')$。

**角度怎么算**：d=4 时，第 1 组 $\theta = i / 10000^0 = i$，第 2 组 $\theta = i / 10000^{0.5} = i / 100$。所以第 1 组转得快（位置 1 就转 1 弧度），第 2 组转得慢（位置 1 只转 0.01 弧度）。

**为什么这样就有相对位置**：两个词的 Q、K 各转了自己的角度，点积里的角度差就是它们的位置差。绝对位置同时加多少，角度差不变，所以点积只看相对位置，不看绝对位置。旋转只改方向不改长度，向量的模长不变。

### 3. 实例演示：一个 4×4 的张量走完全程

输入张量/矩阵 X，形状 (seq_len=4, d_k=4)，4 个 token，每个 4 维：

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| Token 0 (i=0) | 1 | 0 | 1 | 0 |
| Token 1 (i=1) | 0 | 2 | 1 | 1 |
| Token 2 (i=2) | 1 | 1 | 0 | 0 |
| Token 3 (i=3) | 2 | 1 | 1 | 0 |

先算好 cos 表和 sin 表，形状都是 (seq_len, d_k/2) = (4, 2)：

|  | cos 第 1 组 | cos 第 2 组 | sin 第 1 组 | sin 第 2 组 |
| --- | --- | --- | --- | --- |
| Token 0 | 1.00 | 1.00 | 0.00 | 0.00 |
| Token 1 | 0.54 | 1.00 | 0.84 | 0.01 |
| Token 2 | -0.42 | 1.00 | 0.91 | 0.02 |
| Token 3 | -0.99 | 1.00 | 0.14 | 0.03 |

Token 0 的角度是 0，cos=1、sin=0，所以 Token 0 完全不转。

### 动画：三步走完，分拆 → 旋转 → 拼回

点“播放”，一步一步看；也可以点“下一格”自己走。

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
  var X=[[1,0,1,0],[0,2,1,1],[1,1,0,0],[2,1,1,0]];
  var x1=[[1,1],[0,1],[1,0],[2,1]], x2=[[0,0],[2,1],[1,0],[1,0]];
  var r1=[[1,1],[-1.68,0.99],[-1.33,0],[ -2.12,1]];
  var r2=[[0,0],[1.08,1.01],[0.49,0],[ -0.71,0.03]];
  var out=[[1,0,1,0],[-1.68,1.08,0.99,1.01],[-1.33,0.49,0,0],[-2.12,-0.71,1,0.03]];
  var names=['Token 0','Token 1','Token 2','Token 3'];
  var stages=[
    {t:'Step 1：分拆。先通过 rearrange 把 (4, 4) 变成 (4, 2, 2)，再 unbind 拆成 x1 和 x2。x1 是偶数列（第 1、3 列），x2 是奇数列（第 2、4 列）。'},
    {t:'Step 2：旋转。结合 cos 表和 sin 表，分别求出偶数位和奇数位旋转后的结果：x1_rot = x1·cos - x2·sin，x2_rot = x1·sin + x2·cos。看 Token 1 那行：x1_rot = [0×0.54-2×0.84, 1×1-1×0.01] = [-1.68, 0.99]。'},
    {t:'Step 3：拼回。堆叠并展平恢复 (4, 4) 的原始形状：x_rot = stack([x1_rot, x2_rot])，再 rearrange 回来。Token 0 角度是 0，一点没变；Token 1 变成 [-1.68, 1.08, 0.99, 1.01]。'}
  ];
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl==='cur' ? '#000' : '#fff';
    return '<td style="border:1px solid #888;padding:8px 12px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function mat(M, title, on){
    var h='<div style="display:inline-block;vertical-align:top;margin:4px;"><div style="text-align:center;">'+title+'</div><table style="border-collapse:collapse;text-align:center;">';
    for(var r=0;r<M.length;r++){
      h+='<tr><td style="padding:4px;">'+names[r]+'</td>';
      for(var c=0;c<M[r].length;c++) h+=cell(on?M[r][c]:'?', on?'done':'');
      h+='</tr>';
    }
    return h+'</table></div>';
  }
  function render(){
    var html='';
    html+=mat(X, '输入 X (4, 4)', idx>=0);
    if(idx>=0) html+='<span style="margin:0 6px;">→</span>'+mat(x1, 'x1 偶数列 (4, 2)', true)+mat(x2, 'x2 奇数列 (4, 2)', true);
    if(idx>=1) html+='<br>'+mat(r1, 'x1_rot (4, 2)', true)+mat(r2, 'x2_rot (4, 2)', true);
    if(idx>=2) html+='<br>'+mat(out, '输出 X_rot (4, 4)', true);
    document.getElementById('rope-anim').innerHTML=html;
    document.getElementById('rope-explain').textContent = idx>=0 ? stages[idx].t : '点“播放”开始。';
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('rope-play').textContent='▶ 播放'; }
  function step(){ if(idx>=stages.length-1){ stop(); return; } idx++; render(); if(idx>=stages.length-1) stop(); }
  document.getElementById('rope-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=stages.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 1500);
  });
  document.getElementById('rope-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('rope-all').addEventListener('click', function(){ stop(); idx=stages.length-1; render(); });
  document.getElementById('rope-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

### 走完的结果

|  | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| Token 0 | 1.00 | 0.00 | 1.00 | 0.00 |
| Token 1 | -1.68 | 1.08 | 0.99 | 1.01 |
| Token 2 | -1.33 | 0.49 | 0.00 | 0.00 |
| Token 3 | -2.12 | -0.71 | 1.00 | 0.03 |

Token 0 一点没变（角度是 0），位置越往后转得越多，第 1 组转得快、第 2 组转得慢，形状还是 (4, 4)。

### 4. 代码对照与张量形状

每一步的代码和形状变化：

| 步骤 | 代码 | 张量变换 |
| --- | --- | --- |
| 预计算三角函数表 | cos_table, sin_table | (seq_len, d_k/2) = (4, 2)，旋转角度表给算出来 |
| 拆分提取 x1, x2 | x1, x2 = rearrange(x, "... (d_half xy) -> ... d_half xy").unbind(dim=-1) | x (4, 4) → (4, 2, 2) → x1 (4, 2)，x2 (4, 2) |
| 执行旋转 | x1_rot = x1 * cos - x2 * sin；x2_rot = x1 * sin + x2 * cos | (4, 2) * (4, 2) ± (4, 2) * (4, 2) → x1_rot (4, 2)，x2_rot (4, 2) |
| 拼接回原始形状 | x_rot = torch.stack([x1_rot, x2_rot], dim=-1)；x_rot = rearrange(x_rot, "... d_half xy -> ... (d_half xy)") | x1_rot (4, 2)，x2_rot (4, 2) → (4, 2, 2) → (4, 4) |

<details>
<summary>想看代码：RoPE 长什么样（点开）</summary>

```python
def apply_rope(x, cos_table, sin_table):
    # x: (batch, seq_len, d_k)
    # Step 1: 拆成 x1（偶数列）和 x2（奇数列）
    x1, x2 = rearrange(x, "... (d_half xy) -> ... d_half xy").unbind(dim=-1)
    # Step 2: 旋转
    x1_rot = x1 * cos_table - x2 * sin_table
    x2_rot = x1 * sin_table + x2 * cos_table
    # Step 3: 拼回
    x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
    return rearrange(x_rot, "... d_half xy -> ... (d_half xy)")
```

`rearrange` 是 einops 的写法，`... (d_half xy)` 的意思是把最后一维 4 拆成 2 组、每组 2 个数；`unbind(dim=-1)` 是把最后一维拆开，变成两个张量。

</details>

### 和原始位置编码的区别：先把原版算一遍

RoPE 之前，原始 Transformer 用的是**正弦位置编码**：不给 Q、K 转角度，而是在输入上直接加一个位置向量 P。

公式，一个符号一个符号讲：

$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$，$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$

| 符号 | 意思 |
| --- | --- |
| $pos$ | token 下标：第几个词，0、1、2…… |
| $i$ | 维度组号：把 d 个维度两两分组，第 0 组、第 1 组…… |
| $d$ | d_model，这里是 4 |
| $10000$ | 底数，论文定的，决定高频组和低频组的差距 |

偶数位用 sin，奇数位用 cos。分母 $10000^{2i/d}$ 越大，变化越慢：

- 第 0 组（i=0）：分母 $10000^0 = 1$，$\sin(pos)$、$\cos(pos)$，转得最快
- 第 1 组（i=1）：分母 $10000^{2/4} = 100$，$\sin(pos/100)$、$\cos(pos/100)$，转得很慢

### 动画：一格一格算出位置表

d=4，位置 0、1、2，点“播放”看每一格是怎么算出来的。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="pe-anim" style="overflow-x:auto;"></div>
<p id="pe-explain">点“播放”或“下一格”开始。</p>
<button id="pe-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="pe-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格 →</button>
<button id="pe-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="pe-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  // cells: [pos, dim, formula, value]
  var cells=[
    [0,0,'sin(0/1) = sin(0)','0'],[0,1,'cos(0/1) = cos(0)','1'],
    [0,2,'sin(0/100) = sin(0)','0'],[0,3,'cos(0/100) = cos(0)','1'],
    [1,0,'sin(1/1) = sin(1)','0.84'],[1,1,'cos(1/1) = cos(1)','0.54'],
    [1,2,'sin(1/100) = sin(0.01)','0.01'],[1,3,'cos(1/100) = cos(0.01)','1.00'],
    [2,0,'sin(2/1) = sin(2)','0.91'],[2,1,'cos(2/1) = cos(2)','-0.42'],
    [2,2,'sin(2/100) = sin(0.02)','0.02'],[2,3,'cos(2/100) = cos(0.02)','1.00']
  ];
  var done=0, timer=null;
  function render(){
    var html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;"></td><td style="border:1px solid #888;padding:8px 12px;">第 1 维<br><span style="font-size:12px;">sin(pos)</span></td><td style="border:1px solid #888;padding:8px 12px;">第 2 维<br><span style="font-size:12px;">cos(pos)</span></td><td style="border:1px solid #888;padding:8px 12px;">第 3 维<br><span style="font-size:12px;">sin(pos/100)</span></td><td style="border:1px solid #888;padding:8px 12px;">第 4 维<br><span style="font-size:12px;">cos(pos/100)</span></td></tr>';
    for(var p=0;p<3;p++){
      html+='<tr><td style="padding:6px;">位置 '+p+'</td>';
      for(var d=0;d<4;d++){
        var k=p*4+d;
        var c = k<done ? cells[k] : null;
        html+='<td style="border:1px solid #888;padding:8px 12px;background:'+(c?(k===done-1?'#2563eb':'#1f2937'):'#fff')+';color:'+(c?'#fff':'#000')+';">'+(c?c[3]:'?')+'</td>';
      }
      html+='</tr>';
    }
    html+='</table>';
    document.getElementById('pe-anim').innerHTML=html;
    document.getElementById('pe-explain').textContent = done===0 ? '点“播放”或“下一格”开始。' : done>=cells.length ? '算完了：位置 0 是 [0, 1, 0, 1]，位置 1 是 [0.84, 0.54, 0.01, 1.00]，位置 2 是 [0.91, -0.42, 0.02, 1.00]。注意第 3、4 维变化很慢，位置 0、1、2 几乎一样。' : '第 '+done+' / 12 格：'+cells[done-1][2]+' = '+cells[done-1][3];
  }
  function step(){ if(done<cells.length){ done++; render(); } else { clearInterval(timer); timer=null; } }
  document.getElementById('pe-next').addEventListener('click', function(){ if(timer){ clearInterval(timer); timer=null; } step(); });
  document.getElementById('pe-all').addEventListener('click', function(){ if(timer){ clearInterval(timer); timer=null; } done=cells.length; render(); });
  document.getElementById('pe-reset').addEventListener('click', function(){ if(timer){ clearInterval(timer); timer=null; } done=0; render(); });
  document.getElementById('pe-play').addEventListener('click', function(){
    if(timer){ clearInterval(timer); timer=null; return; }
    if(done>=cells.length){ done=0; }
    timer=setInterval(step, 700);
  });
  render();
})();
</script>

看两点：

- **第 1、2 维变化快**：位置 0、1、2 的 sin/cos 各不相同，能分清相邻的词。
- **第 3、4 维变化慢**：位置 0、1、2 几乎一样，$pos/100$ 很小，sin 接近 0、cos 接近 1。长句子里靠它分清远近。

算完位置向量 P，就加到嵌入向量上：$x = E + P$，再进 Transformer 块。RoPE 不加在输入上，而是算 Q、K 时旋转，好处在上面对比表里。

|  | 原始位置编码 | RoPE |
| --- | --- | --- |
| 在哪加 | 输入上加一个位置向量 P | 算 Q、K 时旋转 |
| 相对位置 | 靠模型自己学 | 天生就是相对的，点积只看位置差 |
| 现在谁用 | 原始 Transformer | LLaMA、GPT 等现代模型 |

### 可学习位置编码：位置向量让模型自己学

正弦位置编码是写死的公式，还有一种是**可学习位置编码**：位置向量本身就是参数，训练的时候学出来。

$x = E + P[pos]$，$P$ 是一张表，位置 0、1、2 各有一行，训练之前随机初始化，训练之后长成什么样，模型自己定。

| 位置 | 第 1 维 | 第 2 维 | 第 3 维 | 第 4 维 |
| --- | --- | --- | --- | --- |
| 0 | 0.1 | 0.2 | -0.1 | 0.3 |
| 1 | -0.2 | 0.1 | 0.4 | 0.0 |
| 2 | 0.3 | -0.3 | 0.1 | 0.2 |

点下面任意一个位置，看它加到嵌入上是什么。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="lpe-anim" style="overflow-x:auto;"></div>
<p id="lpe-explain">点下面任意一个位置。</p>
<button class="lpe-btn" data-p="0" style="padding:6px 12px;margin:2px;cursor:pointer;">位置 0（猫）</button>
<button class="lpe-btn" data-p="1" style="padding:6px 12px;margin:2px;cursor:pointer;">位置 1（吃）</button>
<button class="lpe-btn" data-p="2" style="padding:6px 12px;margin:2px;cursor:pointer;">位置 2（鱼）</button>
<button class="lpe-btn" data-p="513" style="padding:6px 12px;margin:2px;cursor:pointer;">位置 513</button>
</div>
<script>
(function(){
  var E=[[0.2,-0.5,0.8,0.1],[-0.3,0.6,0.2,0.9],[0.7,0.1,-0.4,0.3]];
  var P=[[0.1,0.2,-0.1,0.3],[-0.2,0.1,0.4,0.0],[0.3,-0.3,0.1,0.2]];
  var words=['猫','吃','鱼'];
  function render(p){
    var html;
    if(p>=3){
      html='<p>位置 513：这张表只学到了位置 511（训练时最大长度 512），位置 513 没有向量，加不上去。这就是可学习位置编码的硬伤：<strong>超过训练长度就没法用</strong>。</p>';
      document.getElementById('lpe-anim').innerHTML=html;
      document.getElementById('lpe-explain').textContent='位置 513 超出训练长度，表里没有这一行。';
      return;
    }
    var sum=E[p].map(function(v,i){return Math.round((v+P[p][i])*10)/10;});
    html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;">嵌入 E</td>'+E[p].map(function(v){return '<td style="border:1px solid #888;padding:8px 12px;">'+v+'</td>';}).join('')+'</tr>';
    html+='<tr><td style="padding:6px;">+ 位置 P</td>'+P[p].map(function(v){return '<td style="border:1px solid #888;padding:8px 12px;background:#fef3c7;">'+v+'</td>';}).join('')+'</tr>';
    html+='<tr><td style="padding:6px;">= 输入 x</td>'+sum.map(function(v){return '<td style="border:1px solid #888;padding:8px 12px;background:#2563eb;color:#fff;">'+v+'</td>';}).join('')+'</tr></table>';
    document.getElementById('lpe-anim').innerHTML=html;
    document.getElementById('lpe-explain').textContent='位置 '+p+'（'+words[p]+'）：嵌入加上学出来的位置向量，得到输入。位置向量是训练学出来的，不是公式算的。';
  }
  document.querySelectorAll('.lpe-btn').forEach(function(b){
    b.addEventListener('click', function(){ render(+b.getAttribute('data-p')); });
  });
  render(0);
})();
</script>

|  | 正弦位置编码 | 可学习位置编码 | RoPE |
| --- | --- | --- | --- |
| 位置信息从哪来 | 公式算的 | 训练学出来的 | 算 Q、K 时旋转 |
| 超过训练长度 | 能用，公式一直能算 | 不能用，表里没这一行 | 能用，但角度超出训练范围会变差 |
| 谁用 | 原始 Transformer | BERT、GPT-2 | LLaMA、GPT 等现代模型 |

### ALiBi 与长上下文：位置不够长怎么办

模型训练时最多看过 512 个词，推理时给它 2048 个词，位置编码怎么办？有三招：

**第一招：ALiBi，不用位置向量，直接在分数上扣分。**

$\text{分数} = q \cdot k - m \times |i - j|$

$i$、$j$ 是两个词的位置，$|i - j|$ 是它们隔了几个词，$m$ 是每个头的斜率。隔得越远，扣得越多，远的词天生就不受重视。位置信息不在向量里，在分数的扣分里。

拿头 1（斜率 $m = 0.5$）的 3 个词算一遍扣分表：

| $i \backslash j$ | 0（猫） | 1（吃） | 2（鱼） |
| --- | --- | --- | --- |
| 0（猫） | 0 | -0.5 | -1.0 |
| 1（吃） | -0.5 | 0 | -0.5 |
| 2（鱼） | -1.0 | -0.5 | 0 |

对角线是自己看自己，距离 0，不扣分；隔一个词扣 0.5，隔两个词扣 1.0。ALiBi 不用学位置向量，也不用公式算位置，长度随便多长都能用，BLOOM、MPT 用的就是它。

**第二招：滑窗注意力，只看附近的词。**

每个词只看前面 $w$ 个词，窗口外的直接不看。窗口 $w = 2$，3 个词的注意力是这样：

|  | 猫 | 吃 | 鱼 |
| --- | --- | --- | --- |
| 猫 | 看 | 不看 | 不看 |
| 吃 | 看 | 看 | 不看 |
| 鱼 | 不看 | 看 | 看 |

计算量从 $O(n^2)$ 降到 $O(n \times w)$，长文本算得动。代价是远的词永远看不到，Mistral 的滑窗注意力就是这招。

**第三招：位置插值，把长句子压缩回训练长度。**

训练长度 512，推理要 2048，位置 0–2047 直接除以 4，变成 0–511.75，塞进模型学过的范围里。RoPE 的角度也跟着缩小，模型看到的角度都在训练范围内。这叫位置插值（Position Interpolation），不用重新训练就能把上下文拉长 4 倍。

一句话记：**ALiBi 是扣分，滑窗是只看附近，位置插值是把位置压缩回去。** 三招都是为了解决同一个问题：训练时短，推理时长的句子怎么办。

### 自测：RoPE 过关了吗

**第 1 题**：RoPE 的公式是什么？$R^i$ 是什么样的矩阵？

<details>
<summary>点开看解析和答案</summary>

$\boldsymbol{q}'^{(i)} = R^i \boldsymbol{q}^{(i)}$，$R^i$ 是分块对角矩阵，对角线上是一串 2×2 的小旋转块，其他位置都是 0。每个小块就是上面第 1 节画出来的那个 2×2 大矩阵，把每两个数转一个角度。

</details>

**第 2 题**：角度 $\theta_{i,k} = i / \Theta^{(2k-2)/d}$ 里，$i$ 和 $k$ 分别是什么？d=4 时第 1 组和第 2 组的角度各是多少？

<details>
<summary>点开看解析和答案</summary>

$i$ 是 token 下标（第几个词），$k$ 是两两分组后的组号。d=4 时第 1 组 $\theta = i / 10000^0 = i$，第 2 组 $\theta = i / 10000^{0.5} = i / 100$。第 1 组转得快，第 2 组转得慢。

</details>

**第 3 题**：实例里 Token 1 的 x1_rot 第一格 -1.68 是怎么算出来的？

<details>
<summary>点开看解析和答案</summary>

x1_rot = x1·cos - x2·sin。Token 1 的 x1 第一格是 0，x2 第一格是 2，cos 是 0.54，sin 是 0.84，所以 0×0.54 - 2×0.84 = -1.68。

</details>

**第 4 题**：为什么 Token 0 旋转完一点没变？

<details>
<summary>点开看解析和答案</summary>

Token 0 的位置 $i=0$，角度 $\theta = 0$，cos 0 = 1、sin 0 = 0，旋转矩阵就是单位矩阵，转了等于没转。

</details>

**第 5 题**：为什么说 RoPE 的点积只和相对位置有关？

<details>
<summary>点开看解析和答案</summary>

两个词的 Q、K 各转了自己的角度，点积里的角度差就是它们的位置差。绝对位置同时加多少，角度差不变，所以点积只看相对位置。旋转只改方向不改长度，向量的模长不变。

</details>

全答得出来，4.2 就过了。

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

### 为什么除以 $\sqrt{d_k}$：不除，Softmax 就饱和了

Q、K 里的每个数都差不多是均值 0、方差 1 的随机数。两个向量点积，是 $d_k$ 个数相乘再相加：方差是 $d_k$ 个 1 加起来，等于 $d_k$，标准差是 $\sqrt{d_k}$。

$d_k$ 越大，分数的波动越大：

| $d_k$ | 分数标准差 | 分数大概范围 |
| --- | --- | --- |
| 2（玩具模型） | 1.41 | ±3 |
| 64（真实模型） | 8 | ±16 |

分数到了 ±16，Softmax 会怎么样？拿一组分数 [4, 0, -4] 算两遍：

### 动画：不除和除，Softmax 差多少

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="scale-anim" style="overflow-x:auto;"></div>
<p id="scale-explain">点下面两个按钮。</p>
<button class="scale-btn" data-k="raw" style="padding:6px 12px;margin:2px;cursor:pointer;">不除：分数 [4, 0, -4]</button>
<button class="scale-btn" data-k="scaled" style="padding:6px 12px;margin:2px;cursor:pointer;">除以 √4 = 2：分数 [2, 0, -2]</button>
</div>
<script>
(function(){
  var data={
    raw:{p:[0.98,0.02,0.00], t:'不除：Softmax 之后第一个词概率 0.98，第三个词 0.00。分布太尖了，几乎只看第一个词，其他词的梯度接近 0，学不动。这叫饱和。'},
    scaled:{p:[0.87,0.12,0.02], t:'除以 √d_k：第一个词概率 0.87，第二个 0.12，第三个 0.02。分布软下来了，每个词都有梯度，学得动。'}
  };
  function bar(p){
    var max=0.98;
    return '<div style="display:flex;align-items:flex-end;height:60px;gap:4px;">'+p.map(function(v,i){
      var h=Math.round(v/max*60);
      return '<div style="flex:1;text-align:center;"><div style="height:'+h+'px;background:'+(i===0?'#2563eb':'#94a3b8')+';"></div><div style="font-size:11px;">'+v+'</div></div>';
    }).join('')+'</div><div style="display:flex;gap:4px;">'+p.map(function(_,i){return '<div style="flex:1;text-align:center;font-size:11px;">词 '+(i+1)+'</div>';}).join('')+'</div>';
  }
  function render(k){
    document.getElementById('scale-anim').innerHTML=bar(data[k].p);
    document.getElementById('scale-explain').textContent=data[k].t;
  }
  document.querySelectorAll('.scale-btn').forEach(function(b){
    b.addEventListener('click', function(){ render(b.getAttribute('data-k')); });
  });
  render('raw');
})();
</script>

一句话记：**除以 $\sqrt{d_k}$ 是把分数的波动拉回 1 附近，Softmax 不饱和，梯度才传得下去。** 玩具模型 $d_k = 2$，除以 $\sqrt{2} = 1.41$，就是动画 4 里那个 1.41。

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

### Padding 掩码：短句子补的 0，不能参与注意力

因果掩码防的是“看未来”，Padding 掩码防的是“看空气”。一批句子长短不一，要拼成一个 batch 一起算，短的后面要补 0（PAD token）凑齐长度：

|  | token 1 | token 2 | token 3 |
| --- | --- | --- | --- |
| 句子 A | 猫 | 吃 | 鱼 |
| 句子 B | 猫 | 吃 | [PAD] |

[PAD] 不是词，是凑数的占位符。注意力不能看它，它也不能看别人。做法和因果掩码一样：[PAD] 所在的行和列，分数都改成 -inf。

拿句子 B 的分数表算一遍。遮罩之前：

|  | 猫 | 吃 | [PAD] |
| --- | --- | --- | --- |
| 猫 | 2 | 1 | 0.5 |
| 吃 | 1 | 2 | 0.3 |
| [PAD] | 0.5 | 0.3 | 1 |

### 动画：Padding 掩码怎么盖

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="pad-anim" style="overflow-x:auto;"></div>
<p id="pad-explain">点“下一步”开始。</p>
<button id="pad-prev" style="padding:6px 12px;margin:2px;cursor:pointer;">← 上一步</button>
<button id="pad-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="pad-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var names=['猫','吃','[PAD]'];
  var raw=[[2,1,0.5],[1,2,0.3],[0.5,0.3,1]];
  var stages=[
    {t:'第 0 步：遮罩之前。分数表 3 行 × 3 列，[PAD] 那一行一列都还在。', m:[[0,0,0],[0,0,0],[0,0,0]], w:null},
    {t:'第 1 步：盖掉 [PAD] 那一列。猫和吃不能看 [PAD]，分数改成 -inf。', m:[[0,0,1],[0,0,1],[0,0,1]], w:null},
    {t:'第 2 步：盖掉 [PAD] 那一行。[PAD] 自己也不能看别人，整行改成 -inf。', m:[[0,0,1],[0,0,1],[1,1,1]], w:null},
    {t:'第 3 步：Softmax 之后。猫那一行权重是 [0.73, 0.27, 0]，[PAD] 的权重是 0；[PAD] 那一行的输出直接置 0，不参与后面的计算。', m:[[0,0,1],[0,0,1],[1,1,1]], w:[[0.73,0.27,0],[0.27,0.73,0],[0,0,0]]}
  ];
  var idx=0;
  function render(){
    var s=stages[idx];
    var html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;"></td>'+names.map(function(n){return '<td style="padding:6px;">'+n+'</td>';}).join('')+'</tr>';
    for(var i=0;i<3;i++){
      html+='<tr><td style="padding:6px;">'+names[i]+'</td>';
      for(var j=0;j<3;j++){
        var masked=s.m[i][j]===1;
        var v = s.w ? s.w[i][j] : (masked ? '-inf' : raw[i][j]);
        html+='<td style="border:1px solid #888;padding:8px 12px;background:'+(masked?'#7f1d1d':s.w?'#2563eb':'#fff')+';color:'+(masked||s.w?'#fff':'#000')+';">'+v+'</td>';
      }
      html+='</tr>';
    }
    html+='</table>';
    document.getElementById('pad-anim').innerHTML=html;
    document.getElementById('pad-explain').textContent=s.t;
  }
  document.getElementById('pad-next').addEventListener('click', function(){ if(idx<stages.length-1){ idx++; render(); } });
  document.getElementById('pad-prev').addEventListener('click', function(){ if(idx>0){ idx--; render(); } });
  document.getElementById('pad-reset').addEventListener('click', function(){ idx=0; render(); });
  render();
})();
</script>

两点记住：

- **因果掩码和 Padding 掩码可以叠加**：一个防看未来，一个防看空气，两个都是把分数改成 -inf，Softmax 之后权重就是 0。
- **[PAD] 的输出直接置 0**：它那一行全是 -inf，Softmax 算不出来，干脆不要它的输出，后面算损失的时候也不算它。

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

## 4.3.5 多头注意力再往下：MHA / MQA / GQA

多头注意力里，Q 有多少个头，K、V 就有多少个头吗？不一定。**Q 的头数决定“问”的细度，K、V 的头数决定要存多少东西。** 三种分法：

|  | MHA（多头注意力） | GQA（分组查询注意力） | MQA（多查询注意力） |
| --- | --- | --- | --- |
| Q 头数 | 2 | 2 | 2 |
| K、V 头数 | 2 | 1（2 个 Q 头共用 1 个 K、V 头） | 1（所有 Q 头共用 1 个 K、V 头） |
| 一个 token 一层要存的 K、V 数 | 2 头 × 2 维 × 2（K+V）= 8 个数 | 1 头 × 2 维 × 2 = 4 个数 | 1 头 × 2 维 × 2 = 4 个数 |
| 谁用 | 原始 Transformer、BERT | LLaMA 2、LLaMA 3 | PaLM、Falcon |

### 动画：点一种，看 K、V 头怎么分

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="gqa-anim" style="overflow-x:auto;"></div>
<p id="gqa-explain">点下面三个按钮。</p>
<button class="gqa-btn" data-k="mha" style="padding:6px 12px;margin:2px;cursor:pointer;">MHA</button>
<button class="gqa-btn" data-k="gqa" style="padding:6px 12px;margin:2px;cursor:pointer;">GQA</button>
<button class="gqa-btn" data-k="mqa" style="padding:6px 12px;margin:2px;cursor:pointer;">MQA</button>
</div>
<script>
(function(){
  var data={
    mha:{t:'MHA：Q 头 1 看 K、V 头 1，Q 头 2 看 K、V 头 2，各看各的。效果最好，但 K、V 存得最多：一个 token 一层 8 个数。', kv:['K 头 1','K 头 2'], groups:[['Q 头 1'],['Q 头 2']]},
    gqa:{t:'GQA：Q 头 1 和 Q 头 2 共用同一个 K、V 头。效果接近 MHA，K、V 存一半：一个 token 一层 4 个数。LLaMA 2、3 用的就是它。', kv:['K 头 1'], groups:[['Q 头 1','Q 头 2']]},
    mqa:{t:'MQA：所有 Q 头共用同一个 K、V 头。K、V 存得最少：一个 token 一层 4 个数，但效果比 MHA 差一点。', kv:['K 头 1'], groups:[['Q 头 1','Q 头 2']]}
  };
  function render(k){
    var d=data[k];
    var html='<table style="border-collapse:collapse;text-align:center;"><tr>';
    d.groups.forEach(function(g,i){
      html+='<td style="padding:6px;vertical-align:top;"><table style="border-collapse:collapse;text-align:center;"><tr>'+g.map(function(q){return '<td style="border:1px solid #888;padding:8px 12px;background:#2563eb;color:#fff;">'+q+'</td>';}).join('')+'</tr><tr><td style="padding:4px;">↓ 共用</td></tr><tr><td style="border:1px solid #888;padding:8px 12px;background:#7f1d1d;color:#fff;">'+d.kv[i]+'</td></tr></table></td>';
    });
    html+='</tr></table>';
    document.getElementById('gqa-anim').innerHTML=html;
    document.getElementById('gqa-explain').textContent=d.t;
  }
  document.querySelectorAll('.gqa-btn').forEach(function(b){
    b.addEventListener('click', function(){ render(b.getAttribute('data-k')); });
  });
  render('mha');
})();
</script>

为什么 K、V 头数重要？推理时每个 token 的 K、V 都要存下来（KV Cache，第七步讲），句子越长、层数越多，存得越多。Q 头数不变，K、V 头数减半，KV Cache 直接减半，显存和速度都省一半。这就是 LLaMA 从 MHA 换成 GQA 的原因。

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

## 4.4.5 激活函数小结：ReLU、GELU、SiLU、SwiGLU

FFN 里用的是 SiLU 和 SwiGLU，但面试和论文里还会遇到 ReLU、GELU，一张表分清。点下面任意一个 x，看四个激活函数输出什么。

| 激活函数 | 公式 | 特点 | 谁用 |
| --- | --- | --- | --- |
| ReLU | $\max(0, x)$ | 小于 0 全变 0，大于 0 不变；简单但会“死掉” | 早期网络 |
| GELU | $x \cdot \Phi(x)$ | 小于 0 的数按概率放过去一点，平滑 | BERT、GPT-2 |
| SiLU | $x \cdot \text{sigmoid}(x)$ | 和 GELU 很像，但更简单 | LLaMA 的 FFN 里 |
| SwiGLU | $\text{SiLU}(xW_1) \odot (xW_3)$ | 两个输入相乘，门控，效果最好 | LLaMA 的 FFN 整体 |

$\Phi(x)$ 是标准正态分布的累积分布函数，$x$ 越大，$\Phi(x)$ 越接近 1。

### 动画：同一个 x，四个函数输出什么

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="act-anim" style="overflow-x:auto;"></div>
<p id="act-explain">点下面任意一个 x。</p>
<button class="act-btn" data-x="-2" style="padding:6px 12px;margin:2px;cursor:pointer;">x = -2</button>
<button class="act-btn" data-x="-1" style="padding:6px 12px;margin:2px;cursor:pointer;">x = -1</button>
<button class="act-btn" data-x="0" style="padding:6px 12px;margin:2px;cursor:pointer;">x = 0</button>
<button class="act-btn" data-x="1" style="padding:6px 12px;margin:2px;cursor:pointer;">x = 1</button>
<button class="act-btn" data-x="2" style="padding:6px 12px;margin:2px;cursor:pointer;">x = 2</button>
</div>
<script>
(function(){
  var table={'-2':[0,-0.05,-0.24],'-1':[0,-0.16,-0.27],'0':[0,0,0],'1':[1,0.84,0.73],'2':[2,1.95,1.76]};
  function render(x){
    var v=table[x];
    var html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;">x = '+x+'</td><td style="border:1px solid #888;padding:8px 12px;">ReLU</td><td style="border:1px solid #888;padding:8px 12px;">GELU</td><td style="border:1px solid #888;padding:8px 12px;">SiLU</td></tr><tr><td style="padding:6px;">输出</td>'+v.map(function(n){return '<td style="border:1px solid #888;padding:8px 12px;background:#2563eb;color:#fff;">'+n+'</td>';}).join('')+'</tr></table>';
    document.getElementById('act-anim').innerHTML=html;
    document.getElementById('act-explain').textContent='x = '+x+'：ReLU 小于 0 直接变 0；GELU 和 SiLU 小于 0 时留一点负数，平滑；大于 0 时三个都接近 x 本身。';
  }
  document.querySelectorAll('.act-btn').forEach(function(b){
    b.addEventListener('click', function(){ render(b.getAttribute('data-x')); });
  });
  render('1');
})();
</script>

一句话记：**ReLU 是硬截断，GELU 和 SiLU 是软截断，SwiGLU 是门控乘法。** 现代大模型用 SwiGLU，因为它多了一个门，能决定哪些信息放过去。

## 4.5 参数量总账：玩具模型一共有多少参数

把前面所有模块的参数加起来，看看一个模型到底“大”在哪。点“下一步”，一行一行加。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="param-anim" style="overflow-x:auto;"></div>
<p id="param-explain">点“下一步”开始加。</p>
<button id="param-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="param-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部加完</button>
<button id="param-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var rows=[
    ['词嵌入 Embedding', '8 个词 × 4 维', 32],
    ['块 1：两个 RMSNorm', '4 + 4', 8],
    ['块 1：注意力 W_Q、W_K、W_V、W_O', '4×4 × 4 个', 64],
    ['块 1：FFN W_1、W_3、W_2', '4×2 + 4×2 + 2×4', 24],
    ['块 2：和块 1 一模一样', '8 + 64 + 24', 96],
    ['最终 RMSNorm', '4', 4],
    ['输出头 LM Head', '8 个词 × 4 维', 32]
  ];
  var idx=0;
  function render(){
    var total=0;
    var html='<table style="border-collapse:collapse;"><tr><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">模块</td><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">算法</td><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">参数</td></tr>';
    for(var i=0;i<idx;i++){
      total+=rows[i][2];
      html+='<tr><td style="border:1px solid #888;padding:8px 12px;">'+rows[i][0]+'</td><td style="border:1px solid #888;padding:8px 12px;">'+rows[i][1]+'</td><td style="border:1px solid #888;padding:8px 12px;text-align:center;">'+rows[i][2]+'</td></tr>';
    }
    if(idx>=rows.length) html+='<tr><td colspan="2" style="border:1px solid #888;padding:8px 12px;font-weight:bold;">总计</td><td style="border:1px solid #888;padding:8px 12px;text-align:center;font-weight:bold;">'+total+'</td></tr>';
    html+='</table>';
    document.getElementById('param-anim').innerHTML=html;
    document.getElementById('param-explain').textContent = idx===0 ? '点“下一步”开始加。' : idx>=rows.length ? '总计 260 个参数。如果输出头和词嵌入共用同一张表（权重绑定），少 32 个，是 228 个。' : '已加 '+idx+' 行，当前总计 '+total+' 个参数。';
  }
  document.getElementById('param-next').addEventListener('click', function(){ if(idx<rows.length){ idx++; render(); } });
  document.getElementById('param-all').addEventListener('click', function(){ idx=rows.length; render(); });
  document.getElementById('param-reset').addEventListener('click', function(){ idx=0; render(); });
  render();
})();
</script>

两个结论：

- **玩具模型里注意力比 FFN 大**：注意力 64 个，FFN 24 个，是因为玩具的 d_ff 只设了 2。真实模型里 d_ff 是 d_model 的好几倍（LLaMA 是 2.7 倍左右），FFN 参数占三分之二，是真正的大户。
- **层数一乘，参数就翻倍**：2 层是 260 个，70 层就是几十亿。参数量 = 每层参数 × 层数 + 嵌入 + 输出头。

---

下一步：[第五步：拼成完整的模型](/posts/llm-study-notes-inference-full-model/)

回目录：[大模型推理](/posts/llm-study-notes-inference/)
