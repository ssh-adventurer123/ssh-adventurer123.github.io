---
title: "先修"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "先修"]
description: "大模型推理的先修：数学、Python、PyTorch 与 Tensor 运算。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的先修，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。只讲推理要用的那部分，不讲训练。

## 数学基础

### 函数

函数是一组输入到输出的映射。大模型本质就是一个超大规模的嵌套函数：输入 token，输出下一个 token 的概率。

### 导数与梯度

导数是单变量函数在某点的变化率。梯度是多元函数对每个变量求偏导组成的向量，指向函数值上升最快的方向。推理时我们不更新参数，这里先知道梯度是什么就够了，训练时再用它。

### 向量与矩阵

向量是一维数字序列，对应一个 token 的词向量。矩阵是二维数字表格，对应模型权重。数据通过矩阵乘法完成特征变换，是 Transformer 的运算基础。

## Python 基础

### 变量与函数、分支与循环、列表分片、类

```python
# 变量与基础类型
num = 10
arr = [1, 2, 3, 4, 5]

# 函数定义
def add(x, y):
    return x + y

# 分支 + 循环
for i in range(3):
    if num > 3:
        print(f"第{i}轮: {i}")

# 列表分片（左闭右开）
sub_arr = arr[1:4]  # 得到 [2, 3, 4]

# 类 = 属性 + 方法
class Module:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        return x * self.dim

net = Module(dim=4)
print(net.forward(3))  # 12
```

## PyTorch 基础

PyTorch 是基于 Python 的深度学习框架，用来搭建和训练神经网络，自动完成张量运算与梯度求导。

```python
import torch
import torch.nn as nn

# 张量：多维数组，支持梯度自动计算，可以搬到 GPU 上
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.shape)  # torch.Size([3])

# 矩阵乘法：Transformer 最核心的基础运算
w = torch.ones(3, 3)
y = x @ w  # 等价于 torch.matmul(x, w)

# 网络都继承 nn.Module
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)  # 输入 10 维，输出 2 维

    def forward(self, x):
        return self.fc(x)  # (1, 10) * (10, 2) -> (1, 2)
```

## Tensor 基本运算

### Batch、Sequence、Head 三个维度

Transformer 里很多计算是对 batch-like 维度的批量操作：

- **Batch 维度**：对 batch 中每个样本独立执行相同的 Transformer 前向操作
- **Sequence 维度**：RMSNorm、FFN 等逐位置操作，在序列的每个 token 上独立执行
- **Head 维度**：多头注意力中，注意力计算在每个头上独立批量执行

### 矩阵乘法：点一下结果格子，看它怎么算出来的

<div id="matmul-demo" style="border:1px solid #555;padding:12px;border-radius:8px;">
<p>A (2×3) × B (3×2) = C (2×2)。点 C 里任意一格，看它是怎么算的。</p>
<div id="matmul-grid"></div>
<p id="matmul-explain">点一格试试。</p>
</div>
<script>
(function(){
  var A=[[1,2,3],[4,5,6]], B=[[1,0],[0,1],[1,1]];
  var C=[[0,0],[0,0]];
  for(var i=0;i<2;i++) for(var j=0;j<2;j++){ var s=0; for(var k=0;k<3;k++) s+=A[i][k]*B[k][j]; C[i][j]=s; }
  var el=document.getElementById('matmul-grid');
  var html='<table style="border-collapse:collapse;text-align:center;">';
  html+='<tr><td>A =</td>'+A.map(function(r){return '<td style="border:1px solid #888;padding:4px 8px;">'+r.join(', ')+'</td>';}).join('')+'</tr>';
  html+='<tr><td>B =</td>'+B.map(function(r){return '<td style="border:1px solid #888;padding:4px 8px;">'+r.join(', ')+'</td>';}).join('')+'</tr></table>';
  html+='<p>C =</p><table style="border-collapse:collapse;text-align:center;">';
  for(var i=0;i<2;i++){ html+='<tr>'; for(var j=0;j<2;j++){ html+='<td><button data-i="'+i+'" data-j="'+j+'" style="padding:8px 14px;margin:2px;cursor:pointer;">'+C[i][j]+'</button></td>'; } html+='</tr>'; }
  html+='</table>';
  el.innerHTML=html;
  el.querySelectorAll('button').forEach(function(b){
    b.addEventListener('click', function(){
      var i=+b.getAttribute('data-i'), j=+b.getAttribute('data-j');
      var parts=[]; for(var k=0;k<3;k++) parts.push(A[i][k]+'×'+B[k][j]);
      document.getElementById('matmul-explain').textContent='C['+i+']['+j+'] = '+parts.join(' + ')+' = '+C[i][j]+'。行 i 和列 j 逐项相乘再相加。';
    });
  });
})();
</script>

### 广播

```python
m = torch.tensor([[1, 2],
                  [3, 4]])  # (2, 2)
v = torch.tensor([[2, 3]])   # (1, 2)，触发广播
out = m * v
# (1, 2) 的 v 会被复制扩展为 (2, 2)，再对应位置元素相乘
# 结果 [[2, 6], [6, 12]]
```

### 沿最后一维求和

```python
t = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])
t.sum(dim=-1)  # 沿最后一维求和，该维度会被消除，结果 [3, 7, 11]
```

### 维度拆分与重排：einsum 和 rearrange

```python
from einops import rearrange, einsum

# (batch, seq, d_model)，d_model=8 拆分为 h=2, d=4
x = torch.randn(1, 2, 8)
x_rearr = rearrange(x, "b s (h d) -> b h s d", h=2, d=4)
# 形状 (1, 2, 8) -> (1, 2, 2, 4)，显式声明维度拆分与顺序变换

# einsum：输入输出维度映射清清楚楚，自动处理任意前置 batch 维度的广播
out = einsum(batch_a, w, "b n k, k m -> b n m")
```

### 行向量 vs 列向量

传统数学习惯用列向量，线性变换写成 $y = Wx$，批处理时 Batch 维度要放在最后。

PyTorch 默认用行向量，线性变换写成 $y = xW^T$。批处理只需要在最外层增加维度，输入直接是 $X \in \mathbb{R}^{batch \times d_{in}}$，这就是 Transformer 代码里到处是 `x @ W.T` 的原因。
