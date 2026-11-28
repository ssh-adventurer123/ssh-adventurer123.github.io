---
title: "第五步：拼成完整的模型"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Transformer"]
description: "大模型推理第五步：把 Transformer 块拼成完整模型，最终 RMSNorm、LM Head、logits 到概率分布。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第五步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第四步：Transformer 块，逐个拆](/posts/llm-study-notes-inference-transformer-block/)。

前面四步把零件都拆完了：分词、嵌入、块、注意力、FFN。现在把它们拼起来，从“猫 吃 鱼”一路算到下一个词的概率分布。数字全部接着前面走，能对上。

| 玩具模型 | 取值 |
| --- | --- |
| 输入 | “猫 吃 鱼”，3 个 token，ID [12, 45, 78] |
| d_model | 4 |
| 层数 | 2 |
| 词表 | 8 个词 |
| 块 2 输出 | 下面这张表 |

## 完整前向传播：七步走完

点“下一步”，看“猫 吃 鱼”一路变成下一个词的概率。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p id="fm-stage-label" style="font-weight:bold;">第 0 步 / 共 7 步：输入</p>
<div id="fm-progress" style="margin:8px 0;"></div>
<div id="fm-visual" style="min-height:110px;padding:10px;border:1px dashed #888;border-radius:6px;"></div>
<p id="fm-explain">还没进模型。</p>
<button id="fm-prev" style="padding:6px 12px;cursor:pointer;">← 上一步</button>
<button id="fm-next" style="padding:6px 12px;cursor:pointer;">下一步 →</button>
<button id="fm-reset" style="padding:6px 12px;cursor:pointer;">重来</button>
</div>
<script>
(function(){
  function table(rows, label){
    var h='<p style="margin:4px 0;">'+label+'</p><table style="border-collapse:collapse;text-align:center;">';
    rows.forEach(function(r){ h+='<tr>'+r.map(function(c){return '<td style="border:1px solid #888;padding:8px 12px;">'+c+'</td>';}).join('')+'</tr>'; });
    return h+'</table>';
  }
  var stages=[
    {title:'第 0 步 / 共 7 步：输入', visual:table([['<strong>12</strong><br>猫','<strong>45</strong><br>吃','<strong>78</strong><br>鱼']], '输入 ID 表：1 行 × 3 列'), explain:'分词的结果，3 个 token 的 ID。'},
    {title:'第 1 步 / 共 7 步：词嵌入 Embedding', visual:table([['0.2','-0.5','0.8','0.1'],['-0.3','0.6','0.2','0.9'],['0.7','0.1','-0.4','0.3']], '嵌入后：3 行 × 4 列'), explain:'查表：每个 ID 变成一个 4 维向量。'},
    {title:'第 2 步 / 共 7 步：Transformer 块 1', visual:table([['0.4','-0.2','0.9','0.3'],['-0.1','0.7','0.4','1.0'],['0.8','0.3','-0.1','0.5']], '块 1 输出：3 行 × 4 列，形状不变'), explain:'注意力让 3 个 token 互相看，FFN 再逐个加工。形状不变，内容变了。'},
    {title:'第 3 步 / 共 7 步：Transformer 块 2', visual:table([['0.5','-0.1','1.0','0.4'],['0.0','0.8','0.5','1.1'],['0.9','0.4','0.0','0.6']], '块 2 输出：3 行 × 4 列，形状不变'), explain:'同样的块再走一遍，参数不同。'},
    {title:'第 4 步 / 共 7 步：最终 RMSNorm', visual:table([['0.84','-0.17','1.68','0.67'],['0','1.10','0.69','1.52'],['1.56','0.69','0','1.04']], '归一化后：3 行 × 4 列，形状不变'), explain:'最后一层归一化，每行除以自己的均方根，数值稳定了再输出。下面会一格一格算。'},
    {title:'第 5 步 / 共 7 步：输出头 LM Head', visual:table([['0.61','-0.03','0.16','0.12','1.61','0.35','0.76','-0.68']], '最后一个位置的分数：1 行 × 8 列'), explain:'线性层把 4 维变成 8 维：词表里每个词一个分数。只看最后一个位置（鱼后面），分数最高的是第 5 个词。下面会一格一格算。'},
    {title:'第 6 步 / 共 7 步：Softmax → 概率', visual:table([['0.13','0.07','0.08','0.08','0.35','0.10','0.15','0.04']], '最后一个位置的概率：1 行 × 8 列，加起来是 1'), explain:'分数变成概率，第 5 个词概率 0.35，最高。解码就从这里选下一个词，第六步讲怎么选。'},
    {title:'第 7 步 / 共 7 步：走完了', visual:'<span style="font-size:18px;">猫 吃 鱼 → 下一个词（第 5 个词，概率 0.35）</span>', explain:'整个前向传播：输入 ID 表 → 嵌入 → N 个块 → 归一化 → 分数表 → 概率 → 下一个词。'}
  ];
  var i=0;
  var names=['输入','嵌入','块 1','块 2','归一化','分数','概率','走完'];
  function render(){
    document.getElementById('fm-stage-label').textContent=stages[i].title;
    document.getElementById('fm-visual').innerHTML=stages[i].visual;
    document.getElementById('fm-explain').textContent=stages[i].explain;
    document.getElementById('fm-progress').innerHTML=names.map(function(n,k){return '<span style="padding:3px 7px;margin:2px;border-radius:12px;background:'+(k===i?'#2563eb':'#ddd')+';color:'+(k===i?'#fff':'#000')+';font-size:12px;">'+n+'</span>';}).join('→');
  }
  document.getElementById('fm-next').addEventListener('click', function(){ if(i<stages.length-1){ i++; render(); } });
  document.getElementById('fm-prev').addEventListener('click', function(){ if(i>0){ i--; render(); } });
  document.getElementById('fm-reset').addEventListener('click', function(){ i=0; render(); });
  render();
})();
</script>

## 最终 RMSNorm：一格一格算

块 2 的输出是这张 3 行 × 4 列的表，最终 RMSNorm 对每一行单独做：除以这一行的均方根。公式和第四步 4.1 一样：

$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(\boldsymbol{a})} g_i$

这里 $g = 1$，所以就是除以均方根。拿第三行（鱼）算一遍：

- 平方和：$0.9^2 + 0.4^2 + 0^2 + 0.6^2 = 0.81 + 0.16 + 0 + 0.36 = 1.33$
- 均方根：$\sqrt{1.33 / 4} = \sqrt{0.33} = 0.58$
- 每个数除以 0.58：$0.9 / 0.58 = 1.56$，$0.4 / 0.58 = 0.69$，$0 / 0.58 = 0$，$0.6 / 0.58 = 1.04$

三行都算完：

|  | 1 | 2 | 3 | 4 | 均方根 |
| --- | --- | --- | --- | --- | --- |
| 猫（块 2 输出） | 0.5 | -0.1 | 1.0 | 0.4 | 0.60 |
| 猫（归一化后） | 0.84 | -0.17 | 1.68 | 0.67 | — |
| 吃（块 2 输出） | 0.0 | 0.8 | 0.5 | 1.1 | 0.72 |
| 吃（归一化后） | 0 | 1.10 | 0.69 | 1.52 | — |
| 鱼（块 2 输出） | 0.9 | 0.4 | 0.0 | 0.6 | 0.58 |
| 鱼（归一化后） | 1.56 | 0.69 | 0 | 1.04 | — |

归一化只改数值大小，不改形状，还是 3 行 × 4 列。为什么最后还要归一化一次？块 2 出来之后数值可能漂得很大，直接进输出头分数会失真，先缩放到稳定范围再算分数。

## LM Head：4 维变 8 维，一个数一个数算

LM Head 就是一个 Linear：$logits = h W_{head}^T$，$W_{head}$ 是 8 行 × 4 列（词表 8 个词，每个词一行）。只看最后一个位置，鱼的归一化向量 $h = [1.56, 0.69, 0, 1.04]$。

输出头的权重表（8 行 × 4 列）：

| 词 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 词 1 | 0.1 | 0.2 | -0.1 | 0.3 |
| 词 2 | -0.2 | 0.4 | 0.1 | 0.0 |
| 词 3 | 0.1 | -0.3 | 0.5 | 0.2 |
| 词 4 | 0.3 | 0.1 | 0.2 | -0.4 |
| 词 5 | 0.5 | 0.3 | 0.4 | 0.6 |
| 词 6 | 0.2 | -0.1 | 0.4 | 0.1 |
| 词 7 | 0.2 | 0.5 | -0.2 | 0.1 |
| 词 8 | -0.3 | 0.0 | 0.6 | -0.2 |

### 动画：词 5 的分数怎么算出来

点“播放”，看 $h$ 和词 5 那一行一个数一个数相乘再相加。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="lm-anim" style="overflow-x:auto;"></div>
<p id="lm-explain">点“播放”开始。</p>
<button id="lm-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="lm-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="lm-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="lm-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var h=[1.56,0.69,0,1.04], w=[0.5,0.3,0.4,0.6];
  var prods=[0.78,0.21,0,0.62];
  var idx=0, timer=null;
  function render(){
    var html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;">h（鱼）</td>'+h.map(function(v,i){return '<td style="border:1px solid #888;padding:8px 12px;background:'+(i<idx?'#2563eb':'#fff')+';color:'+(i<idx?'#fff':'#000')+';">'+v+'</td>';}).join('')+'</tr>';
    html+='<tr><td style="padding:6px;">× 词 5 的行</td>'+w.map(function(v,i){return '<td style="border:1px solid #888;padding:8px 12px;background:'+(i<idx?'#2563eb':'#fff')+';color:'+(i<idx?'#fff':'#000')+';">'+v+'</td>';}).join('')+'</tr>';
    html+='<tr><td style="padding:6px;">= 乘积</td>'+prods.map(function(v,i){return '<td style="border:1px solid #888;padding:8px 12px;background:'+(i<idx?'#2563eb':'#fff')+';color:'+(i<idx?'#fff':'#000')+';">'+(i<idx?v:'?')+'</td>';}).join('')+'</tr></table>';
    var sum=0; for(var i=0;i<idx;i++) sum+=prods[i];
    html+='<p>相加：'+(idx===0?'还没开始':prods.slice(0,idx).join(' + ')+' = '+Math.round(sum*100)/100)+'</p>';
    document.getElementById('lm-anim').innerHTML=html;
    document.getElementById('lm-explain').textContent = idx===0 ? '点“播放”开始。' : idx<4 ? '第 '+idx+' / 4 格：'+h[idx-1]+' × '+w[idx-1]+' = '+prods[idx-1] : '词 5 的分数 = 0.78 + 0.21 + 0 + 0.62 = 1.61，是 8 个词里最高的。';
  }
  function step(){ if(idx<4){ idx++; render(); } else { clearInterval(timer); timer=null; } }
  document.getElementById('lm-next').addEventListener('click', function(){ if(timer){ clearInterval(timer); timer=null; } step(); });
  document.getElementById('lm-all').addEventListener('click', function(){ if(timer){ clearInterval(timer); timer=null; } idx=4; render(); });
  document.getElementById('lm-reset').addEventListener('click', function(){ if(timer){ clearInterval(timer); timer=null; } idx=0; render(); });
  document.getElementById('lm-play').addEventListener('click', function(){
    if(timer){ clearInterval(timer); timer=null; return; }
    if(idx>=4){ idx=0; }
    timer=setInterval(step, 800);
  });
  render();
})();
</script>

8 个词的分数都算完：

| 词 1 | 词 2 | 词 3 | 词 4 | 词 5 | 词 6 | 词 7 | 词 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.61 | -0.03 | 0.16 | 0.12 | **1.61** | 0.35 | 0.76 | -0.68 |

分数最高的是词 5（1.61）。分数本身还不是概率，还要过 Softmax。

### 权重绑定：输出头和词嵌入共用一张表

输出头的权重是 8 行 × 4 列，词嵌入表也是 8 行 × 4 列，形状一模一样。很多模型（比如 LLaMA）让它们**共用同一张表**，这叫权重绑定（tied embeddings）：

- 省参数：少了 32 个参数（玩具模型），真实模型省几亿个
- 有道理：一个词的“输入表示”和“输出表示”本来就该是一回事

不绑定的模型（比如 GPT-2 早期）就是两张独立的表，参数更多。

## logits → Softmax → 概率分布

分数（logits）有正有负、加起来不是 1，不能直接当概率。Softmax 把它变成概率：

$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

分三步：每个分数取 $e$ 的幂，加起来，再每个除以总和。拿最后一个位置的 8 个分数算一遍：

### 动画：分数变概率

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="sm-anim" style="overflow-x:auto;"></div>
<p id="sm-explain">点“下一步”开始。</p>
<button id="sm-prev" style="padding:6px 12px;margin:2px;cursor:pointer;">← 上一步</button>
<button id="sm-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="sm-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var logits=[0.61,-0.03,0.16,0.12,1.61,0.35,0.76,-0.68];
  var exps=[1.84,0.97,1.17,1.13,5.02,1.42,2.15,0.51];
  var probs=[0.13,0.07,0.08,0.08,0.35,0.10,0.15,0.04];
  var stages=[
    {t:'第 0 步：分数（logits）。8 个词的分数，有正有负，加起来不是 1。', row:logits, label:'分数'},
    {t:'第 1 步：每个分数取 e 的幂。负数变小，正数变大，词 5 的 1.61 变成 5.02，拉开差距。', row:exps, label:'e 的幂'},
    {t:'第 2 步：加起来。总和 = 1.84 + 0.97 + 1.17 + 1.13 + 5.02 + 1.42 + 2.15 + 0.51 = 14.19。', row:exps, label:'e 的幂（总和 14.19）'},
    {t:'第 3 步：每个除以总和，变成概率。8 个概率加起来是 1，词 5 概率 0.35，最高。', row:probs, label:'概率'}
  ];
  var idx=0;
  function render(){
    var s=stages[idx];
    var html='<table style="border-collapse:collapse;text-align:center;"><tr><td style="padding:6px;">'+s.label+'</td>'+s.row.map(function(v,i){return '<td style="border:1px solid #888;padding:8px 12px;background:'+(idx===3&&i===4?'#2563eb':'#fff')+';color:'+(idx===3&&i===4?'#fff':'#000')+';">'+v+'</td>';}).join('')+'</tr></table>';
    document.getElementById('sm-anim').innerHTML=html;
    document.getElementById('sm-explain').textContent=s.t;
  }
  document.getElementById('sm-next').addEventListener('click', function(){ if(idx<stages.length-1){ idx++; render(); } });
  document.getElementById('sm-prev').addEventListener('click', function(){ if(idx>0){ idx--; render(); } });
  document.getElementById('sm-reset').addEventListener('click', function(){ idx=0; render(); });
  render();
})();
</script>

概率分布就是模型对“下一个词是什么”的完整回答：词 5 概率 0.35，词 7 概率 0.15，词 1 概率 0.13……怎么从分布里选出下一个词，就是第六步解码的事。

## 自测：完整模型过关了吗

先自己答，答完再对答案。

### 第 1 题

完整的前向传播一共几步？每一步的形状是多少？

<details>
<summary>点开看解析和答案</summary>

7 步：输入 1 行 × 3 列 → 嵌入 3 行 × 4 列 → 块 1、块 2、最终归一化都是 3 行 × 4 列 → 最后一个位置的分数 1 行 × 8 列 → 概率 1 行 × 8 列。中间形状不变，只有嵌入和输出头改变形状。

</details>

### 第 2 题

最终 RMSNorm 是对哪一行算的？为什么最后还要归一化一次？

<details>
<summary>点开看解析和答案</summary>

对每一行单独算，每行除以自己的均方根。块 2 出来之后数值可能漂得很大，直接进输出头分数会失真，先缩放到稳定范围再算分数。

</details>

### 第 3 题

词 5 的分数 1.61 是怎么算出来的？

<details>
<summary>点开看解析和答案</summary>

鱼的归一化向量 [1.56, 0.69, 0, 1.04] 和词 5 的权重行 [0.5, 0.3, 0.4, 0.6] 逐个相乘再相加：1.56×0.5 + 0.69×0.3 + 0×0.4 + 1.04×0.6 = 0.78 + 0.21 + 0 + 0.62 = 1.61。

</details>

### 第 4 题

权重绑定是什么意思？省了多少参数？

<details>
<summary>点开看解析和答案</summary>

输出头的权重表和词嵌入表共用同一张表。玩具模型省 32 个参数，真实模型省几亿个。一个词的输入表示和输出表示本来就该是一回事。

</details>

### 第 5 题

Softmax 的三步是什么？概率加起来是多少？

<details>
<summary>点开看解析和答案</summary>

每个分数取 e 的幂，加起来，再每个除以总和。8 个概率加起来是 1，词 5 概率 0.35，最高。

</details>

全答得出来，第五步就过了。

---

下一步：[第六步：解码，下一个词怎么出来](/posts/llm-study-notes-inference-decoding/)

回目录：[大模型推理](/posts/llm-study-notes-inference/)
