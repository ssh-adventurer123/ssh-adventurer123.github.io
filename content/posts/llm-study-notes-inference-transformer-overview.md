---
title: "第二步：Transformer 整体长什么样"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Transformer"]
description: "大模型推理第二步：Transformer 整体长什么样，decoder-only 结构，前向传播全流程。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第二步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第一步：分词](/posts/llm-study-notes-inference-tokenization/)。

这一页全是图。先把玩具模型定死，后面所有图都用它，数字能对上：

| 玩具模型 | 取值 |
| --- | --- |
| 输入 | “猫 吃 鱼”，3 个 token |
| d_model | 4（一个 token 变成 4 个数） |
| 层数 | 2（两个 Transformer 块） |
| 词表 | 8 个词（演示用，真实模型是几万个） |

## 先看全图：前向传播一步一步走

点“下一步”，看“猫 吃 鱼”一路变成下一个词。形状每一步都在图里，不写方括号。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p id="ov-stage-label" style="font-weight:bold;">第 0 步 / 共 6 步：输入</p>
<div id="ov-progress" style="margin:8px 0;"></div>
<div id="ov-visual" style="min-height:110px;padding:10px;border:1px dashed #888;border-radius:6px;"></div>
<p id="ov-explain">还没进模型。</p>
<button id="ov-prev" style="padding:6px 12px;cursor:pointer;">← 上一步</button>
<button id="ov-next" style="padding:6px 12px;cursor:pointer;">下一步 →</button>
<button id="ov-reset" style="padding:6px 12px;cursor:pointer;">重来</button>
</div>
<script>
(function(){
  function table(rows, label){
    var h='<p style="margin:4px 0;">'+label+'</p><table style="border-collapse:collapse;text-align:center;">';
    rows.forEach(function(r){ h+='<tr>'+r.map(function(c){return '<td style="border:1px solid #888;padding:8px 12px;">'+c+'</td>';}).join('')+'</tr>'; });
    return h+'</table>';
  }
  var stages=[
    {title:'第 0 步 / 共 6 步：输入', visual:table([['<strong>12</strong><br>猫','<strong>45</strong><br>吃','<strong>78</strong><br>鱼']], '输入 ID 表：1 行 × 3 列'), explain:'分词的结果，3 个 token 的 ID。模型只认识这张表。'},
    {title:'第 1 步 / 共 6 步：词嵌入 Embedding', visual:table([['0.2','-0.5','0.8','0.1'],['-0.3','0.6','0.2','0.9'],['0.7','0.1','-0.4','0.3']], '嵌入后：3 行 × 4 列（3 个 token，每个 4 个数）'), explain:'查表：每个 ID 变成一个 4 维向量。猫是第一行，吃是第二行，鱼是第三行。形状从 1 行 × 3 列变成 3 行 × 4 列。'},
    {title:'第 2 步 / 共 6 步：Transformer 块 1', visual:table([['0.4','-0.2','0.9','0.3'],['-0.1','0.7','0.4','1.0'],['0.8','0.3','-0.1','0.5']], '块 1 输出：3 行 × 4 列，形状不变'), explain:'第一个块：注意力让 3 个 token 互相看，FFN 再逐个加工。形状不变，还是 3 行 × 4 列，内容变了。'},
    {title:'第 3 步 / 共 6 步：Transformer 块 2', visual:table([['0.5','-0.1','1.0','0.4'],['0.0','0.8','0.5','1.1'],['0.9','0.4','0.0','0.6']], '块 2 输出：3 行 × 4 列，形状不变'), explain:'第二个块，和第一个块结构一模一样，参数不同。再走一遍注意力 + FFN。层数就是这样叠出来的。'},
    {title:'第 4 步 / 共 6 步：最终 RMSNorm', visual:table([['0.5','-0.1','1.0','0.4'],['0.0','0.8','0.5','1.1'],['0.9','0.4','0.0','0.6']], '归一化后：3 行 × 4 列，形状不变'), explain:'最后一层归一化，把数值缩放到稳定范围，准备输出。形状还是 3 行 × 4 列。'},
    {title:'第 5 步 / 共 6 步：输出头 Logits', visual:table([['0.1','0.2','0.0','0.3','0.9','0.1','0.0','0.2']], '最后一个位置的分数：1 行 × 8 列（词表 8 个词，每个一个分数）'), explain:'线性层把 4 维变成 8 维：词表里每个词一个分数。只看最后一个位置（鱼后面），分数最高的是第 5 个词，解码就选它做下一个词。'},
    {title:'第 6 步 / 共 6 步：走完了', visual:'<span style="font-size:18px;">猫 吃 鱼 → 下一个词</span>', explain:'整个前向传播：输入 ID 表 → 嵌入 → N 个块 → 归一化 → 分数表 → 下一个词。'}
  ];
  var i=0;
  var names=['输入','嵌入','块 1','块 2','归一化','分数','走完'];
  function render(){
    document.getElementById('ov-stage-label').textContent=stages[i].title;
    document.getElementById('ov-visual').innerHTML=stages[i].visual;
    document.getElementById('ov-explain').textContent=stages[i].explain;
    document.getElementById('ov-progress').innerHTML=names.map(function(n,k){return '<span style="padding:3px 7px;margin:2px;border-radius:12px;background:'+(k===i?'#2563eb':'#ddd')+';color:'+(k===i?'#fff':'#000')+';font-size:12px;">'+n+'</span>';}).join('→');
  }
  document.getElementById('ov-next').addEventListener('click', function(){ if(i<stages.length-1){ i++; render(); } });
  document.getElementById('ov-prev').addEventListener('click', function(){ if(i>0){ i--; render(); } });
  document.getElementById('ov-reset').addEventListener('click', function(){ i=0; render(); });
  render();
})();
</script>

记住一条：**除了嵌入和输出头，中间所有块的形状都不变**，都是 3 行 × 4 列。变的是内容，不是形状。

## 整体结构图：点每一个方块

<style>
.archbox{border:1px solid #888;border-radius:8px;padding:10px;margin:4px auto;cursor:pointer;max-width:420px;text-align:center;}
.archbox.on{background:#2563eb;color:#fff;}
</style>
<div>
<div class="archbox on" data-stage="0">输出：下一个词的分数（1 行 × 8 列）</div>
<div class="archbox" data-stage="1">线性层 LM Head：4 维 → 8 维</div>
<div class="archbox" data-stage="2">最终 RMSNorm</div>
<div class="archbox" data-stage="3">Transformer 块 × 2（结构相同，参数不同）</div>
<div class="archbox" data-stage="4">词嵌入 Embedding：ID → 4 维向量</div>
<div class="archbox" data-stage="5">输入：token ID 表（1 行 × 3 列）</div>
</div>
<p id="arch-explain"><strong>输出：</strong>每个位置对词表里每个词打一个分数，推理只看最后一个位置。</p>
<script>
(function(){
  var texts=[
    "<strong>输出：</strong>每个位置对词表里每个词打一个分数，推理只看最后一个位置。",
    "<strong>线性层 LM Head：</strong>把 4 维向量变成 8 维分数，8 是词表大小。权重形状是 (8, 4)。",
    "<strong>最终 RMSNorm：</strong>最后一层归一化，数值稳定了再输出，形状不变。",
    "<strong>Transformer 块 × 2：</strong>模型的主体，两个块结构一模一样，只是参数不同。每个块里有注意力 + FFN，第三步、第四步会逐个拆。",
    "<strong>词嵌入 Embedding：</strong>查表，ID 12 变成 [0.2, -0.5, 0.8, 0.1] 这样的向量。词表 8 个词，嵌入表就是 8 行 × 4 列。",
    "<strong>输入：</strong>分词得到的 ID 表，1 行 × 3 列。"
  ];
  document.querySelectorAll('.archbox').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('.archbox').forEach(function(x){x.classList.remove('on');});
      b.classList.add('on');
      document.getElementById('arch-explain').innerHTML=texts[+b.getAttribute('data-stage')];
    });
  });
})();
</script>

从下往上读就是前向传播的方向：输入在最下面，输出在最上面。

## Transformer 块里面长什么样：先看，不拆

一个块分两层子层，每层都是“先归一化、再子层、再残差”，这叫 **Pre-Norm**。下面是真的结构图，不是文字：主路在中间走，残差走右边的捷径，绕过归一化和子层，最后加回来。点图里任意一个方块。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div style="display:flex;gap:16px;">
<div style="flex:1;">
<div style="text-align:center;font-weight:bold;">主路（数据怎么走）</div>
<div class="blk-box" data-k="x" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;">x<br><span style="font-size:12px;">3 行 × 4 列</span></div>
<div style="text-align:center;">↓</div>
<div class="blk-box" data-k="norm" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#fef3c7;">RMSNorm</div>
<div style="text-align:center;">↓</div>
<div class="blk-box" data-k="attn" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#dbeafe;">多头注意力</div>
<div style="text-align:center;">↓</div>
<div class="blk-box" data-k="add1" style="border:2px solid #2563eb;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;">＋ x（残差加回来）</div>
<div style="text-align:center;">↓</div>
<div class="blk-box" data-k="norm2" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#fef3c7;">RMSNorm</div>
<div style="text-align:center;">↓</div>
<div class="blk-box" data-k="ffn" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#dcfce7;">FFN</div>
<div style="text-align:center;">↓</div>
<div class="blk-box" data-k="add2" style="border:2px solid #2563eb;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;">＋ x（残差加回来）</div>
<div style="text-align:center;">↓</div>
<div style="text-align:center;font-weight:bold;">下一块</div>
</div>
<div style="flex:1;">
<div style="text-align:center;font-weight:bold;">残差捷径（x 走的路）</div>
<div style="border:2px dashed #2563eb;border-radius:8px;padding:10px;margin:6px;">x 从上面直接下来，<strong>不经过</strong> RMSNorm，也不经过注意力 / FFN，在“＋ x”那里加回来。</div>
<div style="text-align:center;font-size:24px;">↓</div>
<div style="border:2px dashed #2563eb;border-radius:8px;padding:10px;margin:6px;">第一层加完，新的 x 再走一次同样的捷径，在第二层“＋ x”那里加回来。</div>
<p style="font-size:13px;">捷径上没有任何计算，所以梯度可以沿着它从后一层直接传到前一层，不经过归一化和子层。</p>
</div>
</div>
<p id="blk-explain">点图里任意一个方块。</p>
</div>
<script>
(function(){
  var texts={
    x:'<strong>x：</strong>块的输入，3 行 × 4 列。3 是 token 数，4 是 d_model。',
    norm:'<strong>RMSNorm：</strong>归一化，把每一行（一个 token）的数值缩放到稳定范围，形状不变，还是 3 行 × 4 列。',
    attn:'<strong>多头注意力：</strong>让 3 个 token 互相看。猫看吃、看鱼，吃看猫、看鱼，每个 token 都带上上下文。输入输出都是 3 行 × 4 列。',
    add1:'<strong>＋ x：</strong>残差加法。注意力算完，把最开始的 x 加回来，逐个位置、逐个数字相加，形状不变。',
    norm2:'<strong>RMSNorm：</strong>第二层子层前的归一化，和上面那个是两个不同的归一化层，参数不同。',
    ffn:'<strong>FFN：</strong>逐个 token 独立加工，不看别的 token。把注意力收集到的信息整合吸收。输入输出都是 3 行 × 4 列。',
    add2:'<strong>＋ x：</strong>第二次残差加法。FFN 算完，再把上一层的 x 加回来，形状不变，送进下一块。'
  };
  document.querySelectorAll('.blk-box').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('.blk-box').forEach(function(x){x.style.outline='';});
      b.style.outline='3px solid #2563eb';
      document.getElementById('blk-explain').innerHTML=texts[b.getAttribute('data-k')];
    });
  });
})();
</script>

### 残差是怎么算的：逐个数字相加

残差就是“子层算完，加回原来的 x”，加法是逐个位置、逐个数字做的。拿一个 token 的 4 个数算一遍，点“下一步”。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p id="res-stage-label" style="font-weight:bold;">第 0 步：原来的 x</p>
<div id="res-visual" style="min-height:80px;padding:10px;border:1px dashed #888;border-radius:6px;"></div>
<p id="res-explain">一个 token 的向量，4 个数。</p>
<button id="res-prev" style="padding:6px 12px;cursor:pointer;">← 上一步</button>
<button id="res-next" style="padding:6px 12px;cursor:pointer;">下一步 →</button>
<button id="res-reset" style="padding:6px 12px;cursor:pointer;">重来</button>
</div>
<script>
(function(){
  function row(label, vals, color){
    return '<p style="margin:6px 0;">'+label+'</p><table style="border-collapse:collapse;text-align:center;"><tr>'+vals.map(function(v){return '<td style="border:1px solid #888;padding:8px 12px;background:'+color+';">'+v+'</td>';}).join('')+'</tr></table>';
  }
  var stages=[
    {title:'第 0 步：原来的 x', visual:row('x', ['1.0','2.0','0.5','-1.0'], '#dbeafe'), explain:'一个 token 的向量，4 个数。这是子层算之前的值，也是残差捷径上走的值。'},
    {title:'第 1 步：子层输出', visual:row('x', ['1.0','2.0','0.5','-1.0'], '#dbeafe')+'<p style="margin:6px 0;">＋</p>'+row('子层输出', ['0.5','-0.5','1.0','0.5'], '#fef3c7'), explain:'子层（注意力或 FFN）算出来的修正量。注意：它不是替换 x，而是要加到 x 上。'},
    {title:'第 2 步：逐个相加', visual:row('x', ['1.0','2.0','0.5','-1.0'], '#dbeafe')+'<p style="margin:6px 0;">＋</p>'+row('子层输出', ['0.5','-0.5','1.0','0.5'], '#fef3c7')+'<p style="margin:6px 0;">＝ 每个格子单独算：1.0+0.5，2.0+(-0.5)，0.5+1.0，-1.0+0.5</p>', explain:'加法是逐个格子做的：第 1 个数加第 1 个数，第 2 个加第 2 个，以此类推。位置不能错。'},
    {title:'第 3 步：结果', visual:row('x + 子层输出', ['1.5','1.5','1.5','-0.5'], '#dcfce7'), explain:'结果是 [1.5, 1.5, 1.5, -0.5]。原来的信息没丢，子层只是在上面加了一点修正。形状还是 4 个数，不变。'}
  ];
  var i=0;
  function render(){
    document.getElementById('res-stage-label').textContent=stages[i].title;
    document.getElementById('res-visual').innerHTML=stages[i].visual;
    document.getElementById('res-explain').textContent=stages[i].explain;
  }
  document.getElementById('res-next').addEventListener('click', function(){ if(i<stages.length-1){ i++; render(); } });
  document.getElementById('res-prev').addEventListener('click', function(){ if(i>0){ i--; render(); } });
  document.getElementById('res-reset').addEventListener('click', function(){ i=0; render(); });
  render();
})();
</script>

两个好处，一句话一个：

- **信息不丢**：子层算坏了，原来的 x 还在，不会越传越偏。
- **梯度好传**：Pre-Norm 里残差路径上没有归一化，梯度可以从后一层无损地传到前一层，训练更稳定。这是 LLaMA、GPT-3 等现代大模型的标准选择。

### Pre-Norm 和 Post-Norm：并排对比图

两边都是一个子层，唯一的区别是**归一化放在哪里**。点下面任意一个方块，看它在残差路径上还是不在。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div style="display:flex;gap:16px;">
<div style="flex:1;border:1px solid #888;border-radius:8px;padding:10px;">
<div style="text-align:center;font-weight:bold;">Pre-Norm（现代大模型）</div>
<div class="pn-box" data-k="pre-norm" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#fef3c7;">① RMSNorm</div>
<div style="text-align:center;">↓</div>
<div class="pn-box" data-k="pre-sub" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#dbeafe;">② 子层（注意力 / FFN）</div>
<div style="text-align:center;">↓</div>
<div class="pn-box" data-k="pre-add" style="border:2px solid #2563eb;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;">③ ＋ x（残差）</div>
<p style="font-size:13px;color:#059669;">✓ 残差捷径上没有归一化，梯度直通。</p>
</div>
<div style="flex:1;border:1px solid #888;border-radius:8px;padding:10px;">
<div style="text-align:center;font-weight:bold;">Post-Norm（原始 Transformer）</div>
<div class="pn-box" data-k="post-sub" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#dbeafe;">① 子层（注意力 / FFN）</div>
<div style="text-align:center;">↓</div>
<div class="pn-box" data-k="post-add" style="border:2px solid #2563eb;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;">② ＋ x（残差）</div>
<div style="text-align:center;">↓</div>
<div class="pn-box" data-k="post-norm" style="border:1px solid #888;border-radius:8px;padding:10px;margin:6px;text-align:center;cursor:pointer;background:#fee2e2;">③ RMSNorm</div>
<p style="font-size:13px;color:#dc2626;">✗ 残差加完才归一化，梯度要过归一化才能回去。</p>
</div>
</div>
<p id="pn-explain">点上面任意一个方块。</p>
</div>
<script>
(function(){
  var texts={
    'pre-norm':'<strong>Pre-Norm ①：</strong>归一化在子层前面。x 先归一化，再进子层。残差捷径从 x 直接走，不经过归一化。',
    'pre-sub':'<strong>Pre-Norm ②：</strong>子层算的是归一化后的 x，输出再加回原来的 x。',
    'pre-add':'<strong>Pre-Norm ③：</strong>加完就是这一层的输出，直接进下一层。没有归一化挡在残差路上。',
    'post-sub':'<strong>Post-Norm ①：</strong>子层直接算 x，没有先归一化。',
    'post-add':'<strong>Post-Norm ②：</strong>算完加回 x，得到的结果还要再归一化一次。',
    'post-norm':'<strong>Post-Norm ③：</strong>归一化在残差后面。梯度从下一层回来时，要先过这个归一化，才能回到 x。层数深了，梯度就不好传，容易不稳定。'
  };
  document.querySelectorAll('.pn-box').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('.pn-box').forEach(function(x){x.style.outline='';});
      b.style.outline='3px solid #2563eb';
      document.getElementById('pn-explain').innerHTML=texts[b.getAttribute('data-k')];
    });
  });
})();
</script>

一句话记区别：**Pre-Norm 是“先归一化再算”，Post-Norm 是“算完再归一化”。** 归一化在残差路径上还是在外面，决定了梯度好不好传，所以现代大模型都用 Pre-Norm。

## Decoder-only 和 Encoder-Decoder：为什么现代大模型只用前者

原始 Transformer 是编码器-解码器结构：编码器读输入，解码器生成输出，中间还有交叉注意力。

现代大语言模型（GPT、LLaMA 等）统一用 **decoder-only**：只有一个解码器堆叠，自己读输入、自己生成输出，不需要编码器。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<p><strong>Encoder-Decoder（原始）</strong>：编码器 → 解码器 → 输出，两个塔，中间交叉注意力连起来。</p>
<p><strong>Decoder-only（现在）</strong>：一个塔，输入进来，输出出去，塔里全是自注意力 + FFN。</p>
<p>为什么只用后者？语言模型要做的事就是“给定前面的词，预测下一个词”，一个塔就够了，两个塔是多余的。而且 decoder-only 可以直接拿海量文本预训练，不需要成对的输入输出。</p>
</div>

## 形状总览：一张表看完

| 阶段 | 形状（行 × 列） | 说明 |
| --- | --- | --- |
| 输入 ID | 1 × 3 | 1 个样本，3 个 token |
| 嵌入后 | 3 × 4 | 3 个 token，每个 4 维 |
| 块 1 输出 | 3 × 4 | 形状不变 |
| 块 2 输出 | 3 × 4 | 形状不变 |
| 最终归一化 | 3 × 4 | 形状不变 |
| 输出分数 | 3 × 8 | 3 个位置，每个位置 8 个词的分数 |

真实模型只是数字变大：d_model 不是 4 而是几千，层数不是 2 而是几十，词表不是 8 而是几万。结构和这张表一模一样。

<details>
<summary>想看代码：TransformerLM 长什么样（点开）</summary>

```python
def transformer_lm(token_ids):
    x = token_embedding(token_ids)  # (batch, seq, d_model)
    for _ in range(num_layers):
        # 子层 1：先 RMSNorm，再注意力，最后残差
        norm_x = rmsnorm(x)
        q, k, v = linear(norm_x)  # 映射出查询、键、值
        x = x + multi_head_self_attention(rope(q), rope(k), v)
        # 子层 2：先 RMSNorm，再 FFN，最后残差
        x = x + ffn(rmsnorm(x))
    logits = linear(rmsnorm(x))  # (batch, seq, vocab_size)
    return logits
```

上面是简化版，真实代码里每个子层前都有 RMSNorm，Q、K、V 是从归一化后的 x 线性映射出来的。第三步、第四步会把每个模块逐个拆开实现。

</details>

## 自测：整体过关了吗

先自己答，答完再对答案。

### 第 1 题

输入 ID 表、嵌入后、块输出、最终分数，形状各是多少？

<details>
<summary>点开看解析和答案</summary>

输入 ID：1 行 × 3 列。嵌入后：3 行 × 4 列。块 1、块 2、最终归一化：都是 3 行 × 4 列，形状不变。输出分数：3 行 × 8 列，每个位置 8 个词的分数。

</details>

### 第 2 题

Transformer 块里有两个子层，各是干什么的？

<details>
<summary>点开看解析和答案</summary>

第一个是多头注意力：让 token 互相看，带上上下文。第二个是 FFN：逐个 token 独立加工，整合信息。两个子层前都有 RMSNorm，算完都加回残差。

</details>

### 第 3 题

残差连接为什么重要？

<details>
<summary>点开看解析和答案</summary>

子层算完加回原来的 x，信息不丢，梯度也能从后一层直通前一层。Pre-Norm 里残差路径上没有归一化，所以更稳定，层数可以叠很深。

</details>

### 第 4 题

现代大模型为什么用 decoder-only，不用 encoder-decoder？

<details>
<summary>点开看解析和答案</summary>

语言模型要做的事就是“给定前面的词，预测下一个词”，一个解码器塔就够了，不需要编码器读输入、解码器生成输出两个塔。而且 decoder-only 可以直接拿海量文本预训练。

</details>

### 第 5 题

推理时，输出分数表里看哪一个位置？

<details>
<summary>点开看解析和答案</summary>

只看最后一个位置（鱼后面那个位置）。它的 8 个分数里最高的是下一个词，选出来拼回序列，再跑一遍，生成下下个词。

</details>

全答得出来，整体就过了。

---

下一步：[第三步：最基础的两个模块](/posts/llm-study-notes-inference/)（还没写，先回目录）

回目录：[大模型推理](/posts/llm-study-notes-inference/)
