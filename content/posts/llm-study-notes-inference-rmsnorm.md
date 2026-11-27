---
title: "4.1 RMSNorm：归一化"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "RMSNorm", "归一化"]
description: "大模型推理第四步 4.1：RMSNorm 归一化，逐格动画算一遍。"
---

这是 [第四步：Transformer 块，逐个拆](/posts/llm-study-notes-inference-transformer-block/) 里的 4.1，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第三步：最基础的两个模块](/posts/llm-study-notes-inference-linear-embedding/)。

## RMSNorm 是干什么的

RMSNorm 是归一化：把每个 token 的向量缩放到合适大小，让数值不要太大也不要太小，后面的注意力算起来才稳定。

公式是：

$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \odot g$

- $\text{mean}(x^2)$：一个 token 的 4 个数，各自平方，再求平均
- $\sqrt{\text{mean}(x^2) + \epsilon}$：均方根，$\epsilon$ 是个很小的数（比如 $10^{-6}$），防止除以 0
- $\odot g$：逐元素乘一个可学习的参数 $g$，形状和 x 一样

关键点：**RMSNorm 是按行算的**，每个 token 独立归一化，token 之间互不影响。形状不变，输入 3 行 × 4 列，输出还是 3 行 × 4 列。

和 LayerNorm 的区别：LayerNorm 要减均值再除以标准差，RMSNorm 不减均值，只除以均方根，少一次计算，效果差不多，所以 LLaMA 这些模型都用 RMSNorm。

## 演示：逐格动画算一遍

拿一个 token 的向量 x = [1, 2, 0.5, -1]，$g$ = [1, 1, 1, 1]，$\epsilon$ 很小忽略不计。点“播放”看它一格一格算。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p>x（1 行 × 4 列）→ 均方根 → 除 → × g → 输出（1 行 × 4 列）</p>
<div id="rms-anim" style="overflow-x:auto;"></div>
<p id="rms-explain">点“播放”开始。</p>
<button id="rms-play" style="padding:6px 12px;margin:2px;cursor:pointer;">▶ 播放</button>
<button id="rms-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一格</button>
<button id="rms-all" style="padding:6px 12px;margin:2px;cursor:pointer;">全部算完</button>
<button id="rms-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var x=[1,2,0.5,-1], g=[1,1,1,1];
  var sq=x.map(function(v){return v*v;});
  var mean=(sq[0]+sq[1]+sq[2]+sq[3])/4;
  var rms=Math.sqrt(mean);
  var out=x.map(function(v){return Math.round(v/rms*100)/100;});
  // steps: 4 squares, mean, rms, 4 divides
  var steps=[];
  for(var k=0;k<4;k++) steps.push({type:'sq', k:k});
  steps.push({type:'mean'});
  steps.push({type:'rms'});
  for(var k2=0;k2<4;k2++) steps.push({type:'div', k:k2});
  var idx=-1, timer=null, playing=false;
  function cell(v, hl){
    var bg = hl==='cur' ? '#facc15' : hl==='done' ? '#2563eb' : '#1f2937';
    var col = hl ? '#fff' : '#fff';
    if(hl==='cur') col='#000';
    return '<td style="border:1px solid #888;padding:10px 14px;background:'+bg+';color:'+col+';">'+v+'</td>';
  }
  function render(){
    var html='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;"><tr><td style="padding:4px;">x =</td>';
    for(var k=0;k<4;k++){
      var hl='';
      if(idx>=0 && steps[idx] && ((steps[idx].type==='sq'&&steps[idx].k===k)||(steps[idx].type==='div'&&steps[idx].k===k))) hl='cur';
      html+=cell(x[k], hl);
    }
    html+='</tr><tr><td style="padding:4px;">x² =</td>';
    for(var k3=0;k3<4;k3++){
      var done = idx>=0 && steps.slice(0,idx+1).some(function(s){return s.type==='sq'&&s.k===k3;});
      html+=cell(done?sq[k3]:'?', done?'done':'');
    }
    html+='</tr></table>';
    html+='<span style="margin:0 8px;">→</span>';
    html+='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;"><tr><td style="padding:4px;">均值 =</td>';
    var meanDone = idx>=0 && steps.slice(0,idx+1).some(function(s){return s.type==='mean';});
    html+=cell(meanDone?Math.round(mean*100)/100:'?', meanDone?'done':'');
    html+='</tr><tr><td style="padding:4px;">均方根 =</td>';
    var rmsDone = idx>=0 && steps.slice(0,idx+1).some(function(s){return s.type==='rms';});
    html+=cell(rmsDone?Math.round(rms*100)/100:'?', rmsDone?'done':'');
    html+='</tr></table>';
    html+='<span style="margin:0 8px;">→</span>';
    html+='<table style="border-collapse:collapse;text-align:center;display:inline-table;vertical-align:middle;"><tr><td style="padding:4px;">输出 =</td>';
    for(var k4=0;k4<4;k4++){
      var done2 = idx>=0 && steps.slice(0,idx+1).some(function(s){return s.type==='div'&&s.k===k4;});
      html+=cell(done2?out[k4]:'?', done2?'done':'');
    }
    html+='</tr></table>';
    document.getElementById('rms-anim').innerHTML=html;
    var ex=document.getElementById('rms-explain');
    if(idx<0){ ex.textContent='点“播放”开始。'; return; }
    var s=steps[idx];
    if(s.type==='sq') ex.textContent='第 '+(s.k+1)+' 格平方：'+x[s.k]+'² = '+sq[s.k]+'。';
    else if(s.type==='mean') ex.textContent='均值 = (1 + 4 + 0.25 + 1) / 4 = '+Math.round(mean*100)/100+'。';
    else if(s.type==='rms') ex.textContent='均方根 = √'+Math.round(mean*100)/100+' ≈ '+Math.round(rms*100)/100+'。';
    else ex.textContent='输出第 '+(s.k+1)+' 格：'+x[s.k]+' / '+Math.round(rms*100)/100+' = '+out[s.k]+'。';
  }
  function stop(){ if(timer){ clearInterval(timer); timer=null; } playing=false; document.getElementById('rms-play').textContent='▶ 播放'; }
  function step(){ if(idx>=steps.length-1){ stop(); return; } idx++; render(); if(idx>=steps.length-1) stop(); }
  document.getElementById('rms-play').addEventListener('click', function(){
    if(playing){ stop(); return; }
    if(idx>=steps.length-1){ idx=-1; render(); }
    playing=true; this.textContent='⏸ 暂停';
    timer=setInterval(step, 900);
  });
  document.getElementById('rms-next').addEventListener('click', function(){ stop(); step(); });
  document.getElementById('rms-all').addEventListener('click', function(){ stop(); idx=steps.length-1; render(); });
  document.getElementById('rms-reset').addEventListener('click', function(){ stop(); idx=-1; render(); });
  render();
})();
</script>

算一遍：$x^2$ = [1, 4, 0.25, 1]，均值 = 1.5625，均方根 ≈ 1.25，输出 = [0.8, 1.6, 0.4, -0.8]。形状不变，还是 1 行 × 4 列。

### 为什么要归一化

不归一化，x 里的数有大有小，注意力里 QKᵀ 一乘就更大，Softmax 直接饱和，梯度传不下去。归一化把每个 token 的向量拉到同一量级，训练才稳。

### g 是什么

$g$ 是可学习的参数，形状和 x 的最后一维一样（这里是 4 个数）。归一化之后再乘 $g$，模型可以自己决定每个维度放大还是缩小。推理时 $g$ 是固定的数。

<details>
<summary>想看代码：RMSNorm 长什么样（点开）</summary>

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))  # g
        self.eps = eps

    def forward(self, x):
        # 按最后一维算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

`keepdim=True` 的意思是：算完均值还保留最后一维是 1，这样 x / rms 才能广播对齐。

</details>

## 自测：RMSNorm 过关了吗

### 第 1 题

x = [1, 2, 0.5, -1]，均方根是多少？输出第一格是多少？

<details>
<summary>点开看解析和答案</summary>

$x^2$ = [1, 4, 0.25, 1]，均值 = 6.25 / 4 = 1.5625，均方根 = √1.5625 = 1.25。输出第一格 = 1 / 1.25 = 0.8。

</details>

### 第 2 题

RMSNorm 是按行算还是按列算？token 之间会互相影响吗？

<details>
<summary>点开看解析和答案</summary>

按行算，每个 token 独立归一化。token 之间互不影响，交换信息是注意力干的事，不是归一化。

</details>

### 第 3 题

RMSNorm 和 LayerNorm 有什么区别？

<details>
<summary>点开看解析和答案</summary>

LayerNorm 先减均值再除以标准差，RMSNorm 不减均值，只除以均方根。RMSNorm 少一次计算，效果差不多，LLaMA 这些模型都用 RMSNorm。

</details>

### 第 4 题

归一化之后形状变了吗？

<details>
<summary>点开看解析和答案</summary>

没变。输入 3 行 × 4 列，输出还是 3 行 × 4 列。归一化只改数值大小，不改形状。

</details>

### 第 5 题

$g$ 是干什么的？推理时它会变吗？

<details>
<summary>点开看解析和答案</summary>

$g$ 是可学习的参数，归一化之后逐元素乘它，模型可以自己决定每个维度放大还是缩小。推理时它是固定的数，不会变。

</details>

全答得出来，4.1 就过了。

---

下一步：[4.2 RoPE：旋转位置编码](/posts/llm-study-notes-inference-transformer-block/)（还没写，先回第四步目录）

回目录：[第四步：Transformer 块，逐个拆](/posts/llm-study-notes-inference-transformer-block/)
