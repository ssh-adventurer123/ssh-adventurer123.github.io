---
title: "第三步：最基础的两个模块"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Linear", "Embedding"]
description: "大模型推理第三步：最基础的两个模块，Linear 线性层和 Embedding 嵌入层。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第三步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第二步：Transformer 整体长什么样](/posts/llm-study-notes-inference-transformer-overview/)。

Transformer 里最基础的两个模块，一个是算的，一个是查的：

| 模块 | 干什么 | 怎么来的 |
| --- | --- | --- |
| Linear 线性层 | 矩阵乘法，做线性变换 | 算出来的 |
| Embedding 嵌入层 | 查表，把 token ID 变成向量 | 查出来的 |

玩具模型还是第二步那个：d_model=4，词表 8 个词，“猫 吃 鱼”3 个 token。

## Linear 线性层：矩阵乘法

线性层是 Transformer 以及几乎所有网络中最基础的组件，它通过矩阵乘法，对输入执行线性变换：

$y = x W^T$

其中权重矩阵 $W \in \mathbb{R}^{d_{out} \times d_{in}}$。注意 W 的形状是 (d_out, d_in)，转置后才是 (d_in, d_out)，和 x 的最后一维对上。

### 演示 1：矩阵乘法动画，一格一格算

拿一个 token 的向量 x = [1, 2, 3, 4]，W 是 2 行 × 4 列，输出 y 是 2 个数。点“播放”看它一格一格算，也可以点“下一格”自己走。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p>x（1 行 × 4 列）× W<sup>T</sup>（4 行 × 2 列）= y（1 行 × 2 列）</p>
<div id="lin-anim" style="overflow-x:auto;"></div>
<p id="lin-explain">点“播放”开始。</p>
<button id="lin-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="lin-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="lin-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="lin-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var x=[1,2,3,4], W=[[1,0,1,0],[0,1,0,1]];
  // W^T is 4x2: WT[k][j] = W[j][k]
  var WT=[[1,0],[0,1],[1,0],[0,1]];
  var y=[4,6];
  // steps: for each y cell j, for each k: multiply x[k]*WT[k][j]; then sum step
  var steps=[];
  for(var j=0;j<2;j++){
    for(var k=0;k<4;k++) steps.push({type:'mul', j:j, k:k, prod:x[k]*WT[k][j]});
    steps.push({type:'sum', j:j});
  }
  var idx=-1, timer=null, playing=false;
  function render(){
    function cell(v, hl){
      var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
      var col = hl ? '#000' : '#fff';
      if(hl==='done') col='#fff';
      return '<td style="border:1px solid #888;padding:10px 14px;background:'+bg+';color:'+col+';">'+v+'</td>';
    }
    var html='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;"><tr><td style="padding:4px;">x =</td>';
    for(var k=0;k<4;k++){
      var hl='';
      if(idx>=0 && steps[idx] && steps[idx].j!==undefined){
        // highlight x[k] when current step uses it
        if(steps[idx].type==='mul' && steps[idx].k===k) hl='cur';
        else if(steps[idx].type==='sum') hl='';
      }
      html+=cell(x[k], hl);
    }
    html+='</tr></table>';
    html+='<span style="margin:0 8px;">×</span>';
    html+='<span style="display:inline-block;vertical-align:middle;">W<sup>T</sup> =<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;">';
    for(var r=0;r<4;r++){
      html+='<tr>';
      for(var c=0;c<2;c++){
        var hl2='';
        if(idx>=0 && steps[idx] && steps[idx].type==='mul' && steps[idx].k===r && steps[idx].j===c) hl2='cur';
        html+=cell(WT[r][c], hl2);
      }
      html+='</tr>';
    }
    html+='</table></span>';
    html+='<span style="margin:0 8px;">=</span>';
    html+='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;"><tr><td style="padding:4px;">y =</td>';
    for(var j2=0;j2<2;j2++){
      var done = idx>=0 && steps.slice(0, idx+1).some(function(s){return s.type==='sum' && s.j===j2;});
      var cur = idx>=0 && steps[idx] && steps[idx].j===j2;
      html+=cell(done?y[j2]:'?', done?'done':(cur?'cur':''));
    }
    html+='</tr></table>';
    document.getElementById('lin-anim').innerHTML=html;
    var ex=document.getElementById('lin-explain');
    if(idx<0){ ex.textContent='点“播放”开始。'; return; }
    var s=steps[idx];
    if(s.type==='mul'){
      var parts=[];
      for(var kk=0;kk<=s.k;kk++) parts.push(x[kk]+'×'+WT[kk][s.j]+'='+x[kk]*WT[kk][s.j]);
      ex.textContent='算 y['+s.j+']：'+parts.join('，')+'。当前这格是 '+x[s.k]+'×'+WT[s.k][s.j]+' = '+s.prod+'，黄色是正在算的格子。';
    } else {
      var all=[]; for(var kk2=0;kk2<4;kk2++) all.push(x[kk2]+'×'+WT[kk2][s.j]);
      ex.textContent='y['+s.j+'] = '+all.join(' + ')+' = '+y[s.j]+'。';
    }
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('lin-play').textContent='▶ 播放'; }
  function step(){
    if(idx>=steps.length-1){ stop(); return; }
    idx++; render();
    if(idx>=steps.length-1) stop();
  }
  document.getElementById('lin-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById('lin-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('lin-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('lin-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

### 形状：输入什么，输出什么

|  | 形状（行 × 列） | 说明 |
| --- | --- | --- |
| 输入 x | 3 × 4 | 3 个 token，每个 4 维（d_in=4） |
| 权重 W | 2 × 4 | d_out=2，d_in=4 |
| 输出 y | 3 × 2 | 3 个 token，每个 2 维（d_out=2） |

batch 和 seq 维不动，只变最后一维：d_in → d_out。`einsum` 写出来就是 `"... d_in, d_out d_in -> ... d_out"`，`...` 是任意 batch 维度，自动广播。

### 权重怎么初始化：截断正态分布

权重用截断正态分布 `nn.init.trunc_normal_` 初始化，均值 0，标准差 $\sqrt{2 / (d_{in} + d_{out})}$，截断范围 $[-3\sigma, 3\sigma]$。

拿 d_in=4、d_out=2 算一遍：$\sigma = \sqrt{2 / (4 + 2)} = \sqrt{1/3} \approx 0.58$，权重初始值大多落在 [-1.73, 1.73] 之间。太大的权重一开始就被截掉，训练一开始不会发散。

### 没有偏置

这里的 Linear **没有偏置（bias）参数**。Transformer 里很多线性层都不加偏置，少一组参数，也少一次加法。

<details>
<summary>想看代码：Linear 长什么样（点开）</summary>

```python
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 权重 W，形状 (out_features, in_features)
        self.w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        mean, std = 0, (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.w, mean=mean, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # einsum 的 ... 自动处理任意 batch 维度
        y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
        return y
```

`nn.Parameter` 的意思是：这个张量是模型的参数，要追踪梯度，训练时会被更新。推理时它就是固定的数。

</details>

## Embedding 嵌入层：查表

嵌入层是 Transformer 处理数据的第一步，它通过查表操作，将 token ID 映射为模型能够理解的词向量：

$y = E[\text{token id}]$

其中 $E \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}$ 为词向量矩阵，$|\mathcal{V}|$ 为词表大小。

### 演示 2：点一个 token，看查表查到哪一行

词表 8 个词，嵌入表是 8 行 × 4 列。点下面任意一个 token，看它查到第几行。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<button class="emb-btn" data-id="0" data-t="猫" style="padding:8px 14px;margin:4px;cursor:pointer;">猫（ID 0）</button>
<button class="emb-btn" data-id="1" data-t="吃" style="padding:8px 14px;margin:4px;cursor:pointer;">吃（ID 1）</button>
<button class="emb-btn" data-id="2" data-t="鱼" style="padding:8px 14px;margin:4px;cursor:pointer;">鱼（ID 2）</button>
<div id="emb-table"></div>
<p id="emb-explain">点一个 token。</p>
</div>
<script>
(function(){
  var E=[[0.2,-0.5,0.8,0.1],[-0.3,0.6,0.2,0.9],[0.7,0.1,-0.4,0.3],[0.1,0.1,0.1,0.1],[-0.2,0.3,0.5,-0.6],[0.4,-0.1,0.2,0.7],[0.9,0.0,-0.2,0.4],[-0.7,0.8,0.3,-0.3]];
  var names=['猫','吃','鱼','你','好','世','界','的'];
  function render(hi){
    var html='<table style="border-collapse:collapse;text-align:center;">';
    for(var i=0;i<8;i++){
      var on = (i===hi);
      html+='<tr style="background:'+(on?'#dbeafe':'#fff')+';"><td style="border:1px solid #888;padding:6px 10px;">ID '+i+'</td><td style="border:1px solid #888;padding:6px 10px;">'+names[i]+'</td>'+E[i].map(function(v){return '<td style="border:1px solid #888;padding:6px 10px;">'+v+'</td>';}).join('')+'</tr>';
    }
    html+='</table>';
    document.getElementById('emb-table').innerHTML=html;
  }
  render(-1);
  document.querySelectorAll('.emb-btn').forEach(function(b){
    b.addEventListener('click', function(){
      var id=+b.getAttribute('data-id');
      render(id);
      document.getElementById('emb-explain').textContent='“'+b.getAttribute('data-t')+'”的 ID 是 '+id+'，查表查到第 '+id+' 行，得到向量 ['+E[id].join(', ')+']。查表没有计算，就是按 ID 取一行。';
    });
  });
})();
</script>

### 形状：输入什么，输出什么

|  | 形状 | 说明 |
| --- | --- | --- |
| 输入 token ID | 1 × 3 | 1 个样本，3 个 token |
| 嵌入表 E | 8 × 4 | 词表 8 个词，每个 4 维 |
| 输出向量 | 3 × 4 | 3 个 token，每个 4 维 |

输入是整数，输出是向量。Embedding 就是一张表，行数是词表大小，列数是 d_model。

### 权重怎么初始化

权重用截断正态分布初始化，均值 0、标准差 1，截断范围 $[-3, 3]$。和 Linear 不同，这里标准差固定是 1，不是 $\sqrt{2/(d_{in}+d_{out})}$。

<details>
<summary>想看代码：Embedding 长什么样（点开）</summary>

```python
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # 嵌入表，形状 (num_embeddings, embedding_dim)
        self.embed = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embed, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 查表：按 ID 取行
        return self.embed[token_ids]
```

`self.embed[token_ids]` 就是查表：token_ids 是 [0, 1, 2]，就取出第 0、1、2 行，拼成 3 行 × 4 列。

</details>

## Linear 和 Embedding：一个算，一个查

|  | Linear | Embedding |
| --- | --- | --- |
| 本质 | 计算：矩阵乘法 | 查表：按 ID 取行 |
| 输入 | 向量（连续的数） | token ID（整数） |
| 输出 | 向量，最后一维变了 | 向量，最后一维是 d_model |
| 权重 | W，形状 (d_out, d_in) | E，形状 (vocab_size, d_model) |
| 在哪用 | 到处都是：QKV 投影、输出头、FFN 里 | 模型开头：把 ID 变成向量 |

一句话记：**Embedding 在开头把数字变成向量，Linear 在后面把向量变成另一个向量。**

## 自测：两个模块过关了吗

先自己答，答完再对答案。

### 第 1 题

x 是 3 行 × 4 列，W 是 2 行 × 4 列，y 是什么形状？怎么算的？

<details>
<summary>点开看解析和答案</summary>

y 是 3 行 × 2 列。$y = x W^T$，W 转置后是 4 行 × 2 列，3×4 乘 4×2 得到 3×2。batch 和 seq 维不动，只变最后一维：4 → 2。

</details>

### 第 2 题

Embedding 查表时，输入和输出分别是什么？

<details>
<summary>点开看解析和答案</summary>

输入是 token ID，整数，比如 [0, 1, 2]。输出是向量，查表取出对应的行，比如第 0 行是 [0.2, -0.5, 0.8, 0.1]。输入 1 行 × 3 列，输出 3 行 × 4 列。

</details>

### 第 3 题

Linear 的 W 为什么是 (d_out, d_in)，不是 (d_in, d_out)？

<details>
<summary>点开看解析和答案</summary>

因为公式是 $y = x W^T$，W 要先转置再乘。x 是 (…, d_in)，$W^T$ 是 (d_in, d_out)，乘完是 (…, d_out)。所以 W 本身是 (d_out, d_in)。PyTorch 的 `nn.Linear` 内部也是这个约定。

</details>

### 第 4 题

Linear 和 Embedding 的权重初始化有什么不同？

<details>
<summary>点开看解析和答案</summary>

Linear：截断正态，均值 0，标准差 $\sqrt{2/(d_{in}+d_{out})}$，截断 ±3σ。Embedding：截断正态，均值 0，标准差 1，截断 ±3。两个都用 `nn.Parameter` 存，训练时会被更新。

</details>

### 第 5 题

为什么说 Linear 是“算”，Embedding 是“查”？

<details>
<summary>点开看解析和答案</summary>

Linear 要做矩阵乘法，每个输出数都是输入数加权求和算出来的。Embedding 不做计算，按 ID 到表里取一行，直接拿走。Embedding 在模型开头只用一次，Linear 在模型里到处都是。

</details>

全答得出来，第三步就过了。

---

下一步：[第四步：Transformer 块，逐个拆](/posts/llm-study-notes-inference-transformer-block/)

回目录：[大模型推理](/posts/llm-study-notes-inference/)
