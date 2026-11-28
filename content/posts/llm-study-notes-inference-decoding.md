---
title: "第六步：解码，下一个词怎么出来"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Transformer"]
description: "大模型推理第六步：解码，贪心、温度、top-k、top-p、beam search、EOS 停止。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第六步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第五步：拼成完整的模型](/posts/llm-study-notes-inference-full-model/)。

第五步算完了概率分布：8 个词的概率是下面这张表。解码就是从这张表里选出下一个词，再拼回序列，重复。

| 词 1 | 词 2 | 词 3 | 词 4 | 词 5 | 词 6 | 词 7 | 词 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.13 | 0.07 | 0.08 | 0.08 | **0.35** | 0.10 | 0.15 | 0.04 |

## 贪心解码：永远选概率最高的

最简单的办法：每次选概率最高的词。

$\text{下一个词} = \arg\max_i P(w_i)$

这里选词 5，概率 0.35。拼回序列变成“猫 吃 鱼 词5”，再跑一遍模型，生成下下个词。

优点是确定、可复现；缺点是容易重复、死板。写文章、聊天不会只用贪心。

## 温度：一个滑杆，控制随机程度

温度 $T$ 在 Softmax 之前除一下分数：

$\text{Softmax}\left(\frac{z_i}{T}\right)$

- $T = 1$：原样，不变
- $T < 1$：分数拉大，分布变尖，更确定
- $T > 1$：分数压小，分布变平，更随机

### 动画：拉动温度，看分布怎么变

点下面三个温度，看同一组分数变成什么分布。

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="temp-anim" style="overflow-x:auto;"></div>
<p id="temp-explain">点下面三个温度。</p>
<button class="temp-btn" data-t="0.5" style="padding:6px 12px;margin:2px;cursor:pointer;">T = 0.5（更确定）</button>
<button class="temp-btn" data-t="1" style="padding:6px 12px;margin:2px;cursor:pointer;">T = 1（原样）</button>
<button class="temp-btn" data-t="2" style="padding:6px 12px;margin:2px;cursor:pointer;">T = 2（更随机）</button>
</div>
<script>
(function(){
  var data={
    '0.5':{p:[0.09,0.02,0.04,0.03,0.64,0.05,0.12,0.01], t:'T = 0.5：词 5 的概率从 0.35 涨到 0.64，几乎一定选它。写代码、做数学题用低温。'},
    '1':{p:[0.13,0.07,0.08,0.08,0.35,0.10,0.15,0.04], t:'T = 1：原样，词 5 概率 0.35。'},
    '2':{p:[0.13,0.10,0.11,0.11,0.22,0.12,0.15,0.07], t:'T = 2：词 5 的概率从 0.35 掉到 0.22，词 7、词 1 都有机会。写故事、头脑风暴用高温。'}
  };
  function bar(p){
    var max=0.64;
    return '<div style="display:flex;align-items:flex-end;height:60px;gap:4px;">'+p.map(function(v,i){
      var h=Math.round(v/max*60);
      return '<div style="flex:1;text-align:center;"><div style="height:'+h+'px;background:'+(i===4?'#2563eb':'#94a3b8')+';"></div><div style="font-size:11px;">'+v+'</div></div>';
    }).join('')+'</div><div style="display:flex;gap:4px;">'+p.map(function(_,i){return '<div style="flex:1;text-align:center;font-size:11px;">词 '+(i+1)+'</div>';}).join('')+'</div>';
  }
  function render(t){
    document.getElementById('temp-anim').innerHTML=bar(data[t].p);
    document.getElementById('temp-explain').textContent=data[t].t;
  }
  document.querySelectorAll('.temp-btn').forEach(function(b){
    b.addEventListener('click', function(){ render(b.getAttribute('data-t')); });
  });
  render('1');
})();
</script>

一句话记：**温度越低越确定，温度越高越随机。** 温度 0 就是贪心。

## 采样：不直接选，而是按概率抽

采样是从概率分布里随机抽一个词，概率高的更容易被抽到，但不是一定被抽到。直接采样的问题是：概率很小的词偶尔也会被抽到，抽到一个烂词，后面全歪了。所以要先把分布切掉一块，再采样。

### top-k：只留前 k 个

k=3，只留概率最高的 3 个词：词 5（0.35）、词 7（0.15）、词 1（0.13）。剩下的 5 个词概率变 0。再重新归一化：

|  | 词 5 | 词 7 | 词 1 | 其他 5 个词 |
| --- | --- | --- | --- | --- |
| 切之前 | 0.35 | 0.15 | 0.13 | 0.37 |
| 切之后 | 0.56 | 0.24 | 0.21 | 0 |

切之后总和是 0.35 + 0.15 + 0.13 = 0.63，每个除以 0.63：0.35/0.63 = 0.56，0.15/0.63 = 0.24，0.13/0.63 = 0.21。再从这 3 个词里按概率抽。

### top-p：按累计概率切

top-p（核采样）不固定个数，按累计概率切。p=0.9：把词按概率从大到小排，累计加到 0.9 就停。

| 排名 | 词 | 概率 | 累计 |
| --- | --- | --- | --- |
| 1 | 词 5 | 0.35 | 0.35 |
| 2 | 词 7 | 0.15 | 0.50 |
| 3 | 词 1 | 0.13 | 0.63 |
| 4 | 词 6 | 0.10 | 0.73 |
| 5 | 词 3 | 0.08 | 0.81 |
| 6 | 词 4 | 0.08 | 0.89 |
| 7 | 词 2 | 0.07 | 0.96 ← 到 0.9 了，停 |
| 8 | 词 8 | 0.04 | 1.00，扔掉 |

留 7 个词，扔掉词 8，再重新归一化（每个除以 0.96），从这 7 个词里抽。

top-k 和 top-p 的区别：**top-k 固定个数，top-p 固定累计概率。** 分布很尖的时候 top-p 只留 1、2 个词，分布很平的时候留很多个，自适应。实际用的时候两个经常一起开：先 top-k 切到 50 个，再 top-p 切到 0.95。

## beam search：留几条路，不只留一个词

贪心只留一个词，beam search 留几条路（beam 宽）。beam=2 的过程：

### 动画：beam=2，两步走完

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="beam-anim" style="overflow-x:auto;"></div>
<p id="beam-explain">点“下一步”开始。</p>
<button id="beam-prev" style="padding:6px 12px;margin:2px;cursor:pointer;">← 上一步</button>
<button id="beam-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="beam-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var stages=[
    {t:'第 0 步：起点。“猫 吃 鱼”，还没选词。', rows:[['猫 吃 鱼','—','0']]},
    {t:'第 1 步：第一个词。词 5 概率 0.35（对数 -1.05），词 7 概率 0.15（对数 -1.90），留这两条路。', rows:[['猫 吃 鱼 词5','0.35','-1.05'],['猫 吃 鱼 词7','0.15','-1.90']]},
    {t:'第 2 步：每条路再展开。词 5 后面词 3 概率 0.4、词 1 概率 0.3；词 7 后面词 5 概率 0.5、词 2 概率 0.2。四个组合的对数概率：-1.97、-2.25、-2.59、-3.51，留前两个。', rows:[['猫 吃 鱼 词5 词3','0.35×0.4','-1.97'],['猫 吃 鱼 词5 词1','0.35×0.3','-2.25']]},
    {t:'第 3 步：继续，直到遇到 EOS 或到最大长度。最后在所有走完的路里选对数概率最高的那条。', rows:[['猫 吃 鱼 词5 词3','…','-1.97'],['猫 吃 鱼 词5 词1','…','-2.25']]}
  ];
  var idx=0;
  function render(){
    var s=stages[idx];
    var html='<table style="border-collapse:collapse;"><tr><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">序列</td><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">概率</td><td style="border:1px solid #888;padding:8px 12px;font-weight:bold;">对数概率</td></tr>';
    s.rows.forEach(function(r){ html+='<tr>'+r.map(function(c){return '<td style="border:1px solid #888;padding:8px 12px;">'+c+'</td>';}).join('')+'</tr>'; });
    html+='</table>';
    document.getElementById('beam-anim').innerHTML=html;
    document.getElementById('beam-explain').textContent=s.t;
  }
  document.getElementById('beam-next').addEventListener('click', function(){ if(idx<stages.length-1){ idx++; render(); } });
  document.getElementById('beam-prev').addEventListener('click', function(){ if(idx>0){ idx--; render(); } });
  document.getElementById('beam-reset').addEventListener('click', function(){ idx=0; render(); });
  render();
})();
</script>

为什么用对数概率？概率是乘起来的，越乘越小，下溢；取对数变成加法，数值稳定。beam search 适合翻译这种“要全局最好”的任务，聊天一般不用，因为它太确定、太死板。

## EOS：什么时候停

词表里有一个特殊的词叫 EOS（End Of Sequence，结束符）。模型生成 EOS 就停了。生成循环是这样：

<div style="border:1px solid #888;padding:14px;border-radius:8px;">
<div id="loop-anim" style="overflow-x:auto;"></div>
<p id="loop-explain">点“下一步”看循环怎么走。</p>
<button id="loop-next" style="padding:6px 12px;margin:2px;cursor:pointer;">下一步 →</button>
<button id="loop-reset" style="padding:6px 12px;margin:2px;cursor:pointer;">重置</button>
</div>
<script>
(function(){
  var stages=[
    {t:'输入“猫 吃 鱼”，模型算出概率，选出词 5。', seq:'猫 吃 鱼', next:'词 5'},
    {t:'把词 5 拼回去，输入变成“猫 吃 鱼 词5”，再跑一遍，选出词 3。', seq:'猫 吃 鱼 词5', next:'词 3'},
    {t:'把词 3 拼回去，输入变成“猫 吃 鱼 词5 词3”，再跑一遍，选出 EOS。', seq:'猫 吃 鱼 词5 词3', next:'EOS'},
    {t:'遇到 EOS，停。输出就是“词5 词3”。', seq:'猫 吃 鱼 词5 词3', next:'停'}
  ];
  var idx=0;
  function render(){
    var s=stages[idx];
    document.getElementById('loop-anim').innerHTML='<p style="font-size:18px;">'+s.seq+' → <strong>'+s.next+'</strong></p>';
    document.getElementById('loop-explain').textContent=s.t;
  }
  document.getElementById('loop-next').addEventListener('click', function(){ if(idx<stages.length-1){ idx++; render(); } });
  document.getElementById('loop-reset').addEventListener('click', function(){ idx=0; render(); });
  render();
})();
</script>

除了 EOS，还有两个硬限制：**最大长度**（生成到比如 2048 个词就停，不管有没有 EOS）和**重复惩罚**（已经出现过的词，分数扣一点，防止一直重复同一句话）。

## 自测：解码过关了吗

先自己答，答完再对答案。

### 第 1 题

贪心解码选哪个词？为什么聊天一般不用纯贪心？

<details>
<summary>点开看解析和答案</summary>

选概率最高的词 5（0.35）。纯贪心确定、可复现，但容易重复、死板，写出来的东西没变化。

</details>

### 第 2 题

温度 0.5 和温度 2，词 5 的概率各变成多少？温度越低越确定还是越随机？

<details>
<summary>点开看解析和答案</summary>

温度 0.5 时词 5 概率 0.64，温度 2 时 0.22。温度越低越确定，温度越高越随机。温度 0 就是贪心。

</details>

### 第 3 题

top-k（k=3）切完之后，词 5、词 7、词 1 的概率各是多少？

<details>
<summary>点开看解析和答案</summary>

切之前三个概率是 0.35、0.15、0.13，总和 0.63，重新归一化：0.56、0.24、0.21。其他 5 个词概率变 0。

</details>

### 第 4 题

top-p（p=0.9）留了几个词？扔掉了谁？

<details>
<summary>点开看解析和答案</summary>

留 7 个词，扔掉词 8。按概率从大到小累计：0.35、0.50、0.63、0.73、0.81、0.89、0.96，到 0.9 就停，词 8 没进来。

</details>

### 第 5 题

beam search 为什么用对数概率？生成什么时候停？

<details>
<summary>点开看解析和答案</summary>

概率是乘起来的，越乘越小会下溢，取对数变成加法，数值稳定。生成到 EOS 就停，或者到最大长度就停。

</details>

全答得出来，第六步就过了。

---

下一步：[第七步：推理工程进阶](/posts/llm-study-notes-inference-engineering/)

回目录：[大模型推理](/posts/llm-study-notes-inference/)
