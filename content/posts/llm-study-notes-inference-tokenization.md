---
title: "第一步：分词，文本怎么变成数字"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "分词"]
description: "大模型推理第一步：分词。文本怎么变成 token，再变成 token ID。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第一步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[先修](/posts/llm-study-notes-inference-prerequisites/)。

## 为什么要分词

模型只认识数字，不认识文字。分词是推理的第一步：把原始字符文本转换成 token ID 序列，模型才能处理。

一句话走完分词：

> 文本 → token → token ID → 词向量

比如“猫 吃 鱼”：

| 阶段 | 结果 |
| --- | --- |
| 文本 | 猫 吃 鱼 |
| token | 猫、吃、鱼（3 个 token） |
| token ID | [12, 45, 78]（举例，真实 ID 由词表决定） |
| 输入形状 | (batch_size, sequence_length) = (1, 3) |

分词器没什么存在感，但不可或缺。没有它，文本进不了模型。

### Token 是什么

Token 不是字，也不是词，是**子词（subword）**：比字大、比词小的一段文本。

- 常见词整个是一个 token，比如“吃”
- 罕见词会被拆成几个 token，比如一个生僻词拆成两三个 token
- 词表（vocab）就是所有 token 的集合，词表大小 vocab_size 是词表里 token 的个数
- 每个 token 在词表里有一个编号，就是 token ID

词表越大，能整块表示的词越多，序列越短；词表越小，拆得越碎，序列越长。TinyStories 上训练的分词器词表是 10,000，OpenWebText 上是 32,000。

#### 演示 1：文本怎么变成 token 再变成 ID，输入一段话试试

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<input id="tok-input" type="text" value="猫 吃 鱼" style="width:220px;padding:6px;">
<button id="tok-run" style="padding:6px 12px;cursor:pointer;">分词</button>
<p id="tok-out">点“分词”试试，也可以把输入改成 hello world 或你好世界。</p>
</div>
<script>
(function(){
  // 演示用的小词表，不是真实词表：ID 只是为了让你看到“一个 token 对应一个整数”
  var vocab={"猫":12,"吃":45,"鱼":78,"你":203,"好":88,"世":91,"界":92,"hello":101,"world":102," ":0};
  function run(){
    var text=document.getElementById('tok-input').value;
    var tokens=text.split(/(\s+)/).filter(function(x){return x!=='';});
    // 英文按空格切，中文按字切，演示用
    var flat=[];
    tokens.forEach(function(t){
      if(/^\s+$/.test(t)) return;
      if(/^[a-zA-Z]+$/.test(t)) flat.push(t);
      else { for(var i=0;i<t.length;i++) flat.push(t[i]); }
    });
    var ids=flat.map(function(t){ return vocab[t]!==undefined ? vocab[t] : ('?' ); });
    document.getElementById('tok-out').textContent='token：['+flat.join(', ')+']，共 '+flat.length+' 个；token ID：['+ids.join(', ')+']；输入形状 (1, '+flat.length+')。';
  }
  document.getElementById('tok-run').addEventListener('click', run);
  run();
})();
</script>

## 从 Unicode 到 UTF-8 字节

文本进入分词器前，通常要经过三层表示：字符、Unicode 码点和 UTF-8 字节。BBPE 不直接处理字符，而是在字节序列上学习合并规则。

- **Unicode 码点**：Unicode 为每个字符分配一个整数编号，即码点。例如，`A` 的码点是 `65`，`你` 的码点是 `20320`。Python 用 `ord()` 查询码点，用 `chr()` 从码点还原字符。
- **UTF-8** 是 Unicode 的变长编码（占 1~4 个字节）：它把字符串转换成字节序列，再通过解码还原；ASCII 通常占 1 个字节，中文占 3 个字节，emoji 占 4 个字节。

```python
# ord：字符 -> Unicode 码点
print(ord("A"))                 # -> 65
print(ord("你"))                 # -> 20320

# chr：Unicode 码点 -> 字符
print(chr(65), chr(20320))       # -> A 你

text = "A你😀"
encoded = text.encode("utf-8")   # 字符串 -> bytes（字节序列）
print(repr(encoded))             # -> b'A\xe4\xbd\xa0\xf0\x9f\x98\x80'
print(list(encoded))             # -> [65, 228, 189, 160, 240, 159, 152, 128]
print(encoded.decode("utf-8"))   # bytes -> 字符串
                                  # -> A你😀
```

数一下：`A` 占 1 个字节（65），`你` 占 3 个字节（228, 189, 160），`😀` 占 4 个字节（240, 159, 152, 128）。一共 8 个字节。

#### 演示 2：一个字占几个字节，输入一个字试试

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<input id="utf-input" type="text" value="你" maxlength="2" style="width:80px;padding:6px;">
<button id="utf-run" style="padding:6px 12px;cursor:pointer;">看字节</button>
<p id="utf-out"></p>
</div>
<script>
(function(){
  function run(){
    var ch=document.getElementById('utf-input').value;
    if(!ch){ document.getElementById('utf-out').textContent='先输入一个字。'; return; }
    var c=ch[0];
    var code=c.codePointAt(0);
    var bytes=new TextEncoder().encode(c);
    document.getElementById('utf-out').textContent='“'+c+'”的 Unicode 码点是 '+code+'，UTF-8 占 '+bytes.length+' 个字节：['+Array.from(bytes).join(', ')+']。';
  }
  document.getElementById('utf-run').addEventListener('click', run);
  run();
})();
</script>

### 为什么在 UTF-8 字节上训练，而不是 UTF-16 / UTF-32

- UTF-8 的基础单位是字节，只有 256 种可能，初始词表就是 0~255 这 256 个字节 token，任意文本都能覆盖，不会出现未知字符。
- UTF-16 / UTF-32 的基础单位更大，初始词表要大得多，而且一个字符的字节数固定，浪费也多。
- 字节级分词能覆盖任意文本，但序列会比较长。BPE 通过反复合并高频的相邻 token pair，用更大的词表换取更短的序列。

### 逐字节解码为什么是错的

`decode_utf8_bytes_to_str_wrong(b) = "".join(bytes([x]).decode("utf-8") for x in b)`，错在把每个字节单独解码。中文一个字占 3 个字节，单独一个字节解不出来，会报错或者变成乱码。

比如 `b = "你".encode("utf-8")`，也就是 `[228, 189, 160]`，逐字节解码时第一个字节 228 单独解码就失败。正确做法是把整个字节串一起解码：`b.decode("utf-8")`。

一个无法解码成任何字符的两字节序列：`b"\xff\xff"`，0xff 不是合法的 UTF-8 起始字节，怎么解都是错的。

## BPE 训练流程：字节级字节对编码（BBPE）

BBPE 是字节对编码（BPE）的改进变体，是 GPT、千问等系列模型正在使用的分词算法。

**拿到原始输入语料与特殊字符列表后，BPE 训练的工作过程如下：**

1. **初始化词表**：加入 256 个基础字节 token（0~255）与所有特殊 token。
2. **预分词**：以特殊 token 为分隔符进行切分；再对普通片段应用 GPT-2 正则。

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(PAT, "some text that i'll pre-tokenize")
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

GPT-2 正则的意思：字母连在一起、数字连在一起、空格跟着后面的词走、缩写 `'ll` 单独成段。预分词先把文本切成小段，BPE 只在每一段内部合并，绝不跨段。

3. **BPE 合并循环**（重复多轮，直到词表大小达到目标值）：
   1. 统计所有片段相邻 token pair 的出现次数；
   2. 选择频次最高的 pair，平局时取字节序更大的 pair；
   3. 将 pair 合并成新 token，加入词表并记录 merge；
   4. 再次完整遍历所有片段，替换本轮选中的 pair。

#### 演示 3：BPE 合并，点一步合并一步

语料是 low、low、lower 三个词。一开始每个字母都是一个 token，点“合并一步”，把出现次数最多的相邻一对合成一个新 token。

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

第一轮会合并 `l` + `o` 还是 `o` + `w`？数一下：`l o` 出现 3 次，`o w` 也出现 3 次，平局取字节序更大的 pair，`o w` 的字节序更大，所以先合并 `o w`。这就是“平局时取字节序更大的 pair”的意思，保证合并顺序是确定的。

### train_bpe：训练函数的样子

```python
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    input_path: str  训练语料文件路径
    vocab_size: int  目标词表大小
    special_tokens: list[str]  特殊 token 列表
    return: (vocab, merges)
        vocab: dict[int, bytes]  token ID 到字节串的映射
        merges: list[tuple[bytes, bytes]]  按创建顺序排列的合并规则
    """
    pass
```

注意三点：预分词分两步，特殊 token 不进入普通片段和 pair 统计；频次相同时取字节序更大的 pair；可选优化是维护 `pair_to_ids` 反向索引，选中 pair 后只处理受影响片段。

### 在真实数据集上训练 BPE

- **TinyStories** 是用 GPT-4 生成的短篇儿童故事集，语言简单、干净、格式统一。训练入口：`train_bpe("data/TinyStories/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])`，词表 10,000。
- **OpenWebText** 是 GPT-2 所用 WebText 数据集的开源复现，由 Reddit 高赞外链网页正文抓取而来，规模更大、覆盖面更广。训练入口：`train_bpe("data/openwebtext/owt_train.txt", 32000, ["<|endoftext|>"])`，词表 32,000。

训练产物保存在 `results/tokenizer/<数据集名>/`，包括 `vocab.json` 和 `merge.txt`，下一节载入它们做编码和解码。

最长的 BPE token 只统计 BPE 合并产生的 token，不考虑特殊 token。OpenWebText 的语料更杂，最长 token 通常比 TinyStories 的更长、更不像一个完整词，因为它要覆盖更多罕见组合。

## 分词器的编码与解码：Tokenizer

训练得到的 `vocab` 和 `merges` 需要封装成一个分词器类：`encode` 方法把文本转换为 token ID，`decode` 方法把 token ID 还原为文本。

### 编码（encode）：四步

1. **两步预分词**（与 BPE 训练保持一致，需要额外保留特殊 token）：
   1. **特殊 token 切分**：以特殊 token 为分隔符切分文本，特殊 token 本身整体保留、直接映射为对应 ID，不参与后续合并。
   2. **GPT-2 正则匹配**：对每个普通片段应用 GPT-2 正则，切成更小的 piece。
2. **字节映射**：把每个 piece 编码为 UTF-8 字节，再通过反向词表（字节序列 -> ID）逐字节映射为基础 token ID。
3. **BPE 合并**：在片段内反复选出合并优先级最高的相邻 pair（`merges` 中 rank 最小者）并合并，直到没有可合并的 pair 为止。
4. **拼接成完整编码序列**：拼接所有预分词段的编码序列，输出完整的 Token ID 序列。

### 解码（decode）：三步

`[1, 2, 3, 4]` 查词表将每个 token ID 转换成字节串，把所有字节串拼接在一起，再用 UTF-8 解码为文本。解码遇到非法的 UTF-8 字节时使用 `bytes.decode("utf-8", errors="replace")` 兜底。

### 特殊 token

`<|endoftext|>` 是文档结束符。原始语料由许多独立文档组成，通常把它们拼接成一条序列，只在文档之间插入分隔符，这些符号有自己的 token ID。特殊 token 永远整体作为一个 token，不参与合并，不会被拆开。

### Tokenizer 类的样子

```python
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        vocab: dict[int, bytes]  词表，token ID 到该 token 对应字节串的映射
        merges: list[tuple[bytes, bytes]]  按创建顺序排列的合并规则
        special_tokens: list[str] | None  特殊 token（如 "<|endoftext|>"），
            永远整体作为一个 token，不参与合并
        """
        pass

    def from_files(
        self,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        vocab_filepath: str  vocab.json 路径（保存时 bytes 被转成十六进制字符串）
        merges_filepath: str  merge.txt 路径（每行形如 [b' ', b't']）
        special_tokens: list[str] | None  特殊 token 列表
        """
        pass

    def encode(self, text: str) -> list[int]:
        """
        text: str  待编码的原始文本
        return: list[int]  token ID 列表
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        ids: list[int]  token ID 列表
        return: str  还原出的文本
        """
        pass
```

注意：通过反向词表将基础字节映射到 token ID（基础字节可能被映射到 0~255 以外的范围）；编码时按 `merges` 的创建顺序选择优先级最高的可合并 pair。

#### 演示 4：编码再解码，绕一圈回到原文

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<input id="ed-input" type="text" value="猫 吃 鱼" style="width:220px;padding:6px;">
<button id="ed-run" style="padding:6px 12px;cursor:pointer;">编码 → 解码</button>
<p id="ed-out"></p>
</div>
<script>
(function(){
  var vocab={"猫":12,"吃":45,"鱼":78,"你":203,"好":88,"世":91,"界":92,"hello":101,"world":102};
  var rev={}; Object.keys(vocab).forEach(function(k){ rev[vocab[k]]=k; });
  function run(){
    var text=document.getElementById('ed-input').value;
    var tokens=[];
    text.split(/(\s+)/).filter(function(x){return x!=='';}).forEach(function(t){
      if(/^\s+$/.test(t)) return;
      if(/^[a-zA-Z]+$/.test(t)) tokens.push(t);
      else { for(var i=0;i<t.length;i++) tokens.push(t[i]); }
    });
    var ids=tokens.map(function(t){ return vocab[t]!==undefined ? vocab[t] : '?'; });
    var back=ids.map(function(id){ return rev[id]!==undefined ? rev[id] : '?'; }).join(' ');
    document.getElementById('ed-out').textContent='encode：['+tokens.join(', ')+'] → ['+ids.join(', ')+']；decode：['+ids.join(', ')+'] → "'+back+'"（词表里没有的字会变成 ?，真实词表不会有这个问题）。';
  }
  document.getElementById('ed-run').addEventListener('click', run);
  run();
})();
</script>

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
