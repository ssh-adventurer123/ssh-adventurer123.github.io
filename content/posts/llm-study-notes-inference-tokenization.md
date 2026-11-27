---
title: "第一步：分词，文本怎么变成数字"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "分词"]
description: "大模型推理第一步：分词。文本怎么变成 token，再变成 token ID。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第一步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[先修](/posts/llm-study-notes-inference-prerequisites/)。

这一页不堆文字：每个概念先看图、点一点，文字只用来解释图里发生了什么。

## 先看全图：文本怎么变成数字

点下面任意一个 token，看它一路变成什么。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p style="font-size:18px;">猫 &nbsp;吃 &nbsp;鱼</p>
<div id="flow-tokens"></div>
<p id="flow-explain">点一个 token 试试。</p>
</div>
<script>
(function(){
  var toks=[{t:'猫',id:12,vec:[0.2,-0.5,0.8,0.1]},{t:'吃',id:45,vec:[-0.3,0.6,0.2,0.9]},{t:'鱼',id:78,vec:[0.7,0.1,-0.4,0.3]}];
  var el=document.getElementById('flow-tokens');
  el.innerHTML=toks.map(function(x,i){return '<button data-i="'+i+'" style="padding:10px 18px;margin:4px;font-size:16px;cursor:pointer;border-radius:8px;">'+x.t+'</button>';}).join('');
  el.querySelectorAll('button').forEach(function(b){
    b.addEventListener('click', function(){
      var x=toks[+b.getAttribute('data-i')];
      document.getElementById('flow-explain').innerHTML='“'+x.t+'” → token “'+x.t+'” → token ID <strong>'+x.id+'</strong> → 词向量 ['+x.vec.join(', ')+']（4 维，演示用）。3 个 token，输入形状是 (1, 3)。';
    });
  });
})();
</script>

一句话：模型只认识数字。分词把文字变成 token，再把 token 变成整数 ID，模型才能处理。没有这一步，文本进不了模型。

## Token 是什么：不是字，也不是词

Token 是**子词**：比字大、比词小的一段文本。

- 常见词整个是一个 token，比如“吃”
- 罕见词会被拆成几个 token
- 所有 token 的集合叫**词表（vocab）**，词表大小 vocab_size 是词表里 token 的个数

### 演示：词表越大，序列越短

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<label>词表大小 <input id="vocab-size" type="range" min="100" max="32000" step="100" value="10000" style="width:220px;"> <span id="vocab-size-val">10000</span></label>
<p id="vocab-out"></p>
</div>
<script>
(function(){
  var s=document.getElementById('vocab-size');
  function render(){
    var v=+s.value;
    document.getElementById('vocab-size-val').textContent=v;
    var seq = v<1000 ? 18 : v<5000 ? 12 : v<15000 ? 8 : 6;
    var bar='█'.repeat(seq);
    document.getElementById('vocab-out').textContent='同一段话，词表 '+v+' 时大约需要 '+seq+' 个 token：'+bar+'。词表越大，能整块表示的词越多，序列越短。TinyStories 用 10,000，OpenWebText 用 32,000。';
  }
  s.addEventListener('input', render); render();
})();
</script>

## 从字符到字节：一个字占几个字节

文本进分词器前有三层：字符 → Unicode 码点 → UTF-8 字节。BBPE 不直接处理字符，在字节上学合并规则。

点下面三个字，看它们的字节。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<button class="utf-btn" data-ch="A" style="padding:10px 16px;margin:4px;cursor:pointer;">A</button>
<button class="utf-btn" data-ch="你" style="padding:10px 16px;margin:4px;cursor:pointer;">你</button>
<button class="utf-btn" data-ch="😀" style="padding:10px 16px;margin:4px;cursor:pointer;">😀</button>
<p id="utf-out">点一个字。</p>
</div>
<script>
(function(){
  var data={'A':{code:65,bytes:[65]},'你':{code:20320,bytes:[228,189,160]},'😀':{code:128512,bytes:[240,159,152,128]}};
  document.querySelectorAll('.utf-btn').forEach(function(b){
    b.addEventListener('click', function(){
      var d=data[b.getAttribute('data-ch')];
      var boxes=d.bytes.map(function(x){return '<span style="display:inline-block;border:1px solid #888;border-radius:4px;padding:4px 8px;margin:2px;">'+x+'</span>';}).join('');
      document.getElementById('utf-out').innerHTML='“'+b.getAttribute('data-ch')+'”的 Unicode 码点是 '+d.code+'，UTF-8 占 '+d.bytes.length+' 个字节：'+boxes;
    });
  });
})();
</script>

记住这张图：ASCII 1 个字节，中文 3 个，emoji 4 个。字节级分词的基础单位是字节，只有 256 种可能，初始词表就是 0~255，任意文本都能覆盖，不会出现未知字符。这就是为什么用 UTF-8，不用 UTF-16 / UTF-32：后者基础单位更大、初始词表大得多，还浪费。

### 逐字节解码为什么是错的

一个中文占 3 个字节，单独一个字节解不出来。`b = "你".encode("utf-8")` 是 `[228, 189, 160]`，逐字节解码时第一个字节 228 就失败。正确做法是整个字节串一起解码：`b.decode("utf-8")`。一个怎么解都错的两字节序列：`b"\xff\xff"`，0xff 不是合法的 UTF-8 起始字节。

## BPE：高频的先合成一个 token

字节级分词能覆盖任意文本，但序列太长。BPE 反复合并最高频的相邻一对，用更大的词表换更短的序列。

### 演示：合并一步，看序列怎么变短

语料是 low、low、lower。点“合并一步”，出现次数最多的一对合成一个新 token。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<button id="bpe-step" style="padding:6px 12px;cursor:pointer;">合并一步</button>
<button id="bpe-reset" style="padding:6px 12px;cursor:pointer;">重来</button>
<p id="bpe-out">当前：l o w | l o w | l o w e r</p>
</div>
<script>
(function(){
  var words=[['l','o','w'],['l','o','w'],['l','o','w','e','r']];
  var step=0;
  function render(){
    document.getElementById('bpe-out').textContent='第 '+step+' 轮：'+words.map(function(w){return w.join(' ');}).join(' | ');
  }
  function countPairs(){
    var c={};
    words.forEach(function(w){ for(var i=0;i<w.length-1;i++){ var p=w[i]+'+'+w[i+1]; c[p]=(c[p]||0)+1; } });
    var best=null, bestN=0;
    Object.keys(c).forEach(function(p){ if(c[p]>bestN || (c[p]===bestN && best!==null && p>best)){ bestN=c[p]; best=p; } });
    return {pair:best, n:bestN};
  }
  document.getElementById('bpe-step').addEventListener('click', function(){
    var r=countPairs();
    if(!r.pair){ document.getElementById('bpe-out').textContent='没有可合并的 pair 了。'; return; }
    var parts=r.pair.split('+');
    words=words.map(function(w){
      var out=[];
      for(var i=0;i<w.length;i++){
        if(i<w.length-1 && w[i]===parts[0] && w[i+1]===parts[1]){ out.push(parts[0]+parts[1]); i++; }
        else out.push(w[i]);
      }
      return out;
    });
    step++;
    document.getElementById('bpe-out').textContent='第 '+step+' 轮：合并了 "'+parts[0]+'" + "'+parts[1]+'"（出现 '+r.n+' 次），现在是 '+words.map(function(w){return w.join(' ');}).join(' | ');
  });
  document.getElementById('bpe-reset').addEventListener('click', function(){ words=[['l','o','w'],['l','o','w'],['l','o','w','e','r']]; step=0; render(); });
  render();
})();
</script>

第一轮 `l o` 和 `o w` 都出现 3 次，平局取字节序更大的 pair，所以先合并 `o w`。合并完再完整遍历所有片段、重新统计、再选、再合并，重复多轮，直到词表达到目标大小。

### 合并前先预分词：GPT-2 正则怎么切

BPE 只在每一段内部合并，绝不跨段。下面这句话，颜色不同就是不同的段。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<p style="font-size:16px;"><span style="background:#dbeafe;padding:2px 4px;">some</span><span style="background:#fee2e2;padding:2px 4px;"> text</span><span style="background:#dcfce7;padding:2px 4px;"> that</span><span style="background:#fef3c7;padding:2px 4px;"> i</span><span style="background:#ede9fe;padding:2px 4px;">'ll</span><span style="background:#dbeafe;padding:2px 4px;"> pre</span><span style="background:#fee2e2;padding:2px 4px;">-</span><span style="background:#dcfce7;padding:2px 4px;">tokenize</span></p>
<p>规律：字母连在一起、数字连在一起、空格跟着后面的词走、缩写 <code>'ll</code> 单独成段。特殊 token 比如 <code>&lt;|endoftext|&gt;</code> 更早一步就被切出来，整体保留，不参与合并。</p>
</div>

<details>
<summary>想看代码：train_bpe 长什么样（点开）</summary>

```python
def train_bpe(input_path, vocab_size, special_tokens):
    # 1. 初始化词表：256 个基础字节 + 特殊 token
    # 2. 预分词：特殊 token 切分，再用 GPT-2 正则
    # 3. 合并循环：统计 pair → 选最高频 → 合并 → 再遍历替换
    # 返回 (vocab, merges)
    pass
```

TinyStories：`train_bpe("data/TinyStories/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])`
OpenWebText：`train_bpe("data/openwebtext/owt_train.txt", 32000, ["<|endoftext|>"])`
产物是 `vocab.json` 和 `merge.txt`，下一节载入它们做编码和解码。

</details>

## 编码和解码：来回走一遍

编码是文本 → token ID，解码是 token ID → 文本，是一条路的两个方向。下面用“猫 吃 鱼”一步一步走，点“下一步”看每一步发生了什么。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<p id="ed-stage-label" style="font-weight:bold;">第 0 步 / 共 6 步：原始文本</p>
<div id="ed-progress" style="margin:8px 0;"></div>
<div id="ed-visual" style="min-height:90px;padding:10px;border:1px dashed #888;border-radius:6px;"></div>
<p id="ed-explain">一句话“猫 吃 鱼”，还没进分词器。</p>
<button id="ed-prev" style="padding:6px 12px;cursor:pointer;">← 上一步</button>
<button id="ed-next" style="padding:6px 12px;cursor:pointer;">下一步 →</button>
<button id="ed-reset" style="padding:6px 12px;cursor:pointer;">重来</button>
</div>
<script>
(function(){
  function chip(t, color, sub){
    return '<span style="display:inline-block;padding:8px 12px;margin:3px;border-radius:8px;background:'+color+';color:#fff;">'+t+(sub?'<br><span style="font-size:12px;">'+sub+'</span>':'')+'</span>';
  }
  function bytebox(n){
    return '<span style="display:inline-block;border:1px solid #888;border-radius:4px;padding:4px 6px;margin:2px;">'+n+'</span>';
  }
  var stages=[
    {title:'第 0 步 / 共 6 步：原始文本', visual:'<span style="font-size:20px;">猫 吃 鱼</span>', explain:'一句话“猫 吃 鱼”，还没进分词器。'},
    {title:'第 1 步 / 共 6 步：预分词', visual:chip('猫','#2563eb')+chip('吃','#dc2626')+chip('鱼','#059669'), explain:'预分词先把文本切成段：没有特殊 token，GPT-2 正则按字切开，得到 3 个段：猫、吃、鱼。BPE 只在每一段内部合并，绝不跨段。'},
    {title:'第 2 步 / 共 6 步：字节映射', visual:chip('猫','#2563eb','字节')+bytebox(231)+bytebox(140)+bytebox(171)+'<br>'+chip('吃','#dc2626','字节')+bytebox(229)+bytebox(144)+bytebox(131)+'<br>'+chip('鱼','#059669','字节')+bytebox(233)+bytebox(177)+bytebox(188), explain:'每个字变成 UTF-8 字节，再逐字节映射为基础 token ID。猫是 3 个字节 [231, 140, 171]，吃是 [229, 144, 131]，鱼是 [233, 177, 188]。一共 9 个字节。'},
    {title:'第 3 步 / 共 6 步：BPE 合并', visual:chip('猫','#2563eb','ID 12')+chip('吃','#dc2626','ID 45')+chip('鱼','#059669','ID 78'), explain:'在每一段里反复选 merges 中 rank 最小的相邻 pair 合并：猫的 3 个字节合成一个 token，ID 是 12；吃合成 45；鱼合成 78。9 个字节变成了 3 个 token。'},
    {title:'第 4 步 / 共 6 步：拼接成 ID 序列', visual:chip('12','#2563eb','猫')+chip('45','#dc2626','吃')+chip('78','#059669','鱼')+'<br><span>形状 (1, 3)，这就是模型收到的输入。</span>', explain:'把 3 段的 ID 拼在一起，得到完整编码 [12, 45, 78]。模型拿到的是这个整数序列，不是文字。'},
    {title:'第 5 步 / 共 6 步：解码，走回去', visual:chip('12','#2563eb')+'→'+bytebox(231)+bytebox(140)+bytebox(171)+'<br>'+chip('45','#dc2626')+'→'+bytebox(229)+bytebox(144)+bytebox(131)+'<br>'+chip('78','#059669')+'→'+bytebox(233)+bytebox(177)+bytebox(188)+'<br><span style="font-size:18px;">→ 猫 吃 鱼</span>', explain:'解码反过来走：查词表把每个 ID 变回字节串，拼在一起，再用 UTF-8 解码成文本，又回到了“猫 吃 鱼”。非法字节用 errors="replace" 兜底。'},
    {title:'第 6 步 / 共 6 步：走完了', visual:'<span style="font-size:18px;">猫 吃 鱼 → [12, 45, 78] → 猫 吃 鱼</span>', explain:'编码和解码是一条路的两个方向：去的时候文字变数字，回的时候数字变文字。'}
  ];
  var i=0;
  var names=['原始','预分词','字节','合并','拼接','解码','走完'];
  function render(){
    document.getElementById('ed-stage-label').textContent=stages[i].title;
    document.getElementById('ed-visual').innerHTML=stages[i].visual;
    document.getElementById('ed-explain').textContent=stages[i].explain;
    document.getElementById('ed-progress').innerHTML=names.map(function(n,k){return '<span style="padding:3px 7px;margin:2px;border-radius:12px;background:'+(k===i?'#2563eb':'#ddd')+';color:'+(k===i?'#fff':'#000')+';font-size:12px;">'+n+'</span>';}).join('→');
  }
  document.getElementById('ed-next').addEventListener('click', function(){ if(i<stages.length-1){ i++; render(); } });
  document.getElementById('ed-prev').addEventListener('click', function(){ if(i>0){ i--; render(); } });
  document.getElementById('ed-reset').addEventListener('click', function(){ i=0; render(); });
  render();
})();
</script>

编码四步：预分词（特殊 token 先切出来）→ 字节映射（逐字节变基础 token ID）→ BPE 合并（选 merges 里 rank 最小的 pair 反复合）→ 拼接成完整 ID 序列。解码三步：查词表变回字节串 → 拼在一起 → UTF-8 解码成文本，非法字节用 `errors="replace"` 兜底。

<details>
<summary>想看代码：Tokenizer 类长什么样（点开）</summary>

```python
class Tokenizer:
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        pass  # vocab：ID → 字节串；merges：按创建顺序的合并规则
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        pass  # 载入 vocab.json 和 merge.txt
    def encode(self, text: str) -> list[int]:
        pass  # 文本 → token ID 列表
    def decode(self, ids: list[int]) -> str:
        pass  # token ID 列表 → 文本
```

</details>

## 自测：分词过关了吗

先自己答，答完再对答案。

### 第 1 题

“猫 吃 鱼”分词后，输入形状是多少？

<details>
<summary>点开看解析和答案</summary>

3 个 token，形状是 (1, 3)。1 是 batch_size，3 是 sequence_length。模型拿到的是整数 token ID 序列，不是文字。

</details>

### 第 2 题

`A`、`你`、`😀` 各占几个 UTF-8 字节？

<details>
<summary>点开看解析和答案</summary>

`A` 占 1 个字节（65），`你` 占 3 个字节（228, 189, 160），`😀` 占 4 个字节（240, 159, 152, 128）。ASCII 1 个字节，中文 3 个，emoji 4 个。

</details>

### 第 3 题

BPE 合并时，频次最高的 pair 被合并后，下一步干什么？

<details>
<summary>点开看解析和答案</summary>

再次完整遍历所有片段，替换本轮选中的 pair，然后重新统计、再选、再合并，重复多轮，直到词表大小达到目标值。不是合并一次就完事。

</details>

### 第 4 题

特殊 token `<|endoftext|>` 会参与 BPE 合并吗？

<details>
<summary>点开看解析和答案</summary>

不会。它在预分词第一步就被切出来，整体保留、直接映射为对应 ID，不进入普通片段和 pair 统计，也不参与合并。

</details>

### 第 5 题

`decode` 时遇到非法 UTF-8 字节怎么办？

<details>
<summary>点开看解析和答案</summary>

用 `bytes.decode("utf-8", errors="replace")` 兜底，把解不出来的字节替换掉，而不是直接报错。编码时不会遇到这个问题，解码模型输出时才可能遇到。

</details>

全答得出来，分词就过了。

---

下一步：[第二步：Transformer 整体长什么样](/posts/llm-study-notes-inference/)（还没写，先回目录）

回目录：[大模型推理](/posts/llm-study-notes-inference/)
