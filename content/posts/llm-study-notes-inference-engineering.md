---
title: "第七步：推理工程进阶"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Transformer"]
description: "大模型推理第七步：推理工程进阶，KV Cache、批处理、量化、vLLM、Ollama。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第七步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第六步：解码，下一个词怎么出来](/posts/llm-study-notes-inference-decoding/)。

前面六步讲的是“模型怎么算”。这一步讲的是“怎么算得快、算得省”——真实模型几十亿参数，一句话生成几十个词，每次都从头算一遍是算不起的。

## KV Cache：不重复算已经算过的

第六步的生成循环有个浪费：每生成一个新词，都要把整句话重新跑一遍。输入“猫 吃 鱼”生成词 5，再输入“猫 吃 鱼 词5”生成词 3，词 5 之前的 K、V 其实已经算过了，又算了一遍。

KV Cache 就是把算过的 K、V 存下来，下次直接用。

### 动画：Cache 怎么一步步长出来

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="kv-anim" style="overflow-x:auto;"></div>
<p id="kv-explain">点“下一步”开始。</p>
<button id="kv-prev" style="padding:6px 12px;margin:2px;cursor:pointer;">← 上一步</button>
<button id="kv-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="kv-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var stages=[
    {t:'第 0 步：Prefill（预填充）。输入“猫 吃 鱼”，一次算出 3 个 token 的 K、V，全部存进 Cache。', rows:[['猫','K、V 已存'],['吃','K、V 已存'],['鱼','K、V 已存']], note:'这一步算 3 个 token，是并行的。'},
    {t:'第 1 步：Decode（解码）。只算新词 5 的 Q、K、V，猫、吃、鱼的 K、V 从 Cache 里直接拿，不用重算。', rows:[['猫','Cache 里拿'],['吃','Cache 里拿'],['鱼','Cache 里拿'],['词 5','新算，存进 Cache']], note:'这一步只算 1 个 token。'},
    {t:'第 2 步：再生成词 3。还是只算词 3 一个，前面 4 个 token 的 K、V 都是 Cache 里拿。', rows:[['猫','Cache 里拿'],['吃','Cache 里拿'],['鱼','Cache 里拿'],['词 5','Cache 里拿'],['词 3','新算，存进 Cache']], note:'Cache 每生成一个词长一格。'}
  ];
  var idx=0;
  function render(){
    var s=stages[idx];
    var html='<table style="border-collapse:collapse;"><tr><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">Token</td><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">K、V 从哪来</td></tr>';
    s.rows.forEach(function(r){ html+='<tr>'+r.map(function(c){return '<td style="border:1px solid #888;padding:8px 12px;">'+c+'</td>';}).join('')+'</tr>'; });
    html+='</table><p>'+s.note+'</p>';
    document.getElementById('kv-anim').innerHTML=html;
    document.getElementById('kv-explain').textContent=s.t;
  }
  document.getElementById('kv-next').addEventListener('click', function(){ if(idx<stages.length-1){ idx++; render(); } });
  document.getElementById('kv-prev').addEventListener('click', function(){ if(idx>0){ idx--; render(); } });
  document.getElementById('kv-reset').addEventListener('click', function(){ idx=0; render(); });
  render();
})();
</script>

KV Cache 有多大？玩具模型：一个 token 一层存 K 4 个数 + V 4 个数 = 8 个数，2 层就是 16 个数。3 个 token 是 48 个数，生成 10 个词就是 160 个数。真实模型：70 亿参数，80 层，d_model 8192，一个 token 一层存 K、V 各 8192 个数，80 层就是 130 万个数，FP16 一个数 2 字节，一个 token 2.6MB，1000 个 token 就是 2.6GB。这就是为什么第四步的 MQA、GQA 重要：K、V 头数减半，KV Cache 直接减半。

没有 KV Cache，每生成一个词都要把整句话重算一遍，越生成越慢；有了 KV Cache，每步只算一个新词，速度基本不变。

## 批处理：一次算好几个人的请求

一个请求来的时候，GPU 一次只能算一个序列，很浪费。批处理就是把多个请求拼在一起算。

|  | 静态批处理 | 连续批处理 |
| --- | --- | --- |
| 怎么拼 | 等几个请求凑齐，一起算，一起结束 | 请求随时进、随时出，算完一个马上补一个 |
| 问题 | 有人先写完，有人还在写，先写完的要等 | 没有等待，GPU 一直满载 |
| 谁用 | 早期 | vLLM、TensorRT-LLM |

连续批处理里，GPU 里永远有固定数量的序列在算，算完一个马上补一个新的，吞吐量高好几倍。代价是实现复杂，要管理每个序列自己的 KV Cache（vLLM 的 PagedAttention 就是干这个的：把 KV Cache 按页分块，像操作系统管理内存一样管理）。

## 量化：参数用更少的字节存

参数存在显存里，一个数占多少字节，直接决定模型能不能装进显卡。

| 精度 | 一个数占多少字节 | 70 亿参数占多少显存 | 精度损失 |
| --- | --- | --- | --- |
| FP32 | 4 字节 | 28 GB | 没有 |
| FP16 / BF16 | 2 字节 | 14 GB | 很小 |
| INT8 | 1 字节 | 7 GB | 小 |
| INT4 | 0.5 字节 | 3.5 GB | 有，但能接受 |

量化就是把 FP16 的参数映射成 INT8 或 INT4 的整数，算的时候再反量化回来。INT4 的 70 亿模型只要 3.5GB，一张消费级显卡就能跑。代价是精度掉一点，但对聊天、写代码这种任务，INT4 够用了。

## 推理框架：vLLM、Ollama、TensorRT-LLM

| 框架 | 定位 | 适合谁 |
| --- | --- | --- |
| vLLM | 高性能推理服务，连续批处理 + PagedAttention | 服务器上给很多人同时用 |
| TensorRT-LLM | 英伟达官方，针对 GPU 深度优化 | 英伟达显卡上的生产环境 |
| Ollama | 一行命令在本地跑模型，自动下载、量化 | 自己电脑上试着玩 |
| llama.cpp | C++ 写的，能在 CPU 上跑 | 没有显卡的电脑 |

一句话记：**自己玩用 Ollama，给很多人用用 vLLM，英伟达显卡上追求极致用 TensorRT-LLM。**

## 推测解码：小模型先猜，大模型再验

生成一个词要跑一遍大模型，很慢。推测解码是：让一个小模型一次猜好几个词，大模型一次验证这几个词对不对，对的留下，错的从错的地方重算。小模型猜对的概率很高，大模型一次验证好几个词，等于一次生成了好几个词，速度能快 2-3 倍。

## 自测：推理工程过关了吗

先自己答，答完再对答案。

### 第 1 题

KV Cache 存的是什么？没有它会怎样？

<details>
<summary>点开看解析和答案</summary>

存的是每个 token 每层的 K 和 V。没有它，每生成一个新词都要把整句话重算一遍，越生成越慢；有了它，每步只算一个新词。

</details>

### 第 2 题

玩具模型里，一个 token 一层要存几个数？真实 70 亿模型一个 token 要存多少？

<details>
<summary>点开看解析和答案</summary>

玩具模型一个 token 一层存 K 4 个数 + V 4 个数 = 8 个数，2 层 16 个数。真实 70 亿模型一个 token 一层存 K、V 各 8192 个数，80 层 130 万个数，FP16 约 2.6MB。

</details>

### 第 3 题

连续批处理比静态批处理好在哪？

<details>
<summary>点开看解析和答案</summary>

静态批处理要等所有请求一起结束，先写完的要等后写完的；连续批处理请求随时进、随时出，算完一个马上补一个，GPU 一直满载。

</details>

### 第 4 题

70 亿参数的模型，FP16、INT8、INT4 各占多少显存？

<details>
<summary>点开看解析和答案</summary>

FP16 每个数 2 字节，14GB；INT8 每个数 1 字节，7GB；INT4 每个数 0.5 字节，3.5GB。量化越狠显存越小，精度损失越大。

</details>

### 第 5 题

自己电脑上试着玩用什么框架？服务器上给很多人同时用用什么？

<details>
<summary>点开看解析和答案</summary>

自己玩用 Ollama，一行命令就能跑；服务器上给很多人同时用用 vLLM，连续批处理 + PagedAttention 吞吐量高。

</details>

全答得出来，第七步就过了，整个大模型推理就通了。

---

回目录：[大模型推理](/posts/llm-study-notes-inference/)
