---
title: "先修"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "先修"]
description: "大模型推理的先修：数学、Python、PyTorch 与 Tensor 运算。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的先修，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。只讲推理要用的那部分，不讲训练。

先修要过五关，每关都有图和能点的演示，全部过完再进第一步分词：

| 关 | 内容 | 过关标准 |
| --- | --- | --- |
| 0.0 | 大模型基本工作原理 | 能说出输入怎么变成下一个词 |
| 0.1 | 数学基础 | 能算导数、梯度、向量点积、矩阵乘法 |
| 0.2 | Python 基础 | 能看懂类、self、分片、循环 |
| 0.3 | PyTorch 基础 | 能看懂 tensor、shape、device、nn.Module |
| 1.2 | Tensor 基本运算 | 能说清 batch、sequence、head 各自管什么 |

## 0.0 大模型基本工作原理

大模型做一件事：输入一段文字，预测下一个词是什么。重复这个过程，就生成一整段话。

### 流程图：点每一个方块，看它在干什么

<style>
.flowbox{border:1px solid #888;border-radius:8px;padding:10px;margin:4px;display:inline-block;cursor:pointer;min-width:90px;text-align:center;}
.flowbox.on{background:#2563eb;color:#fff;}
</style>
<div>
<span class="flowbox on" data-stage="0">输入文本<br>猫 吃 鱼</span> →
<span class="flowbox" data-stage="1">分词<br>猫 / 吃 / 鱼</span> →
<span class="flowbox" data-stage="2">Token ID<br>[12, 45, 78]</span> →
<span class="flowbox" data-stage="3">Transformer<br>N 层</span> →
<span class="flowbox" data-stage="4">Logits<br>每个词一个分数</span> →
<span class="flowbox" data-stage="5">解码<br>选下一个词</span>
</div>
<p id="flow-explain"><strong>输入文本：</strong>用户给一句话，比如“猫 吃 鱼”。模型一次只处理固定长度的序列，不是一整篇文章。</p>
<script>
(function(){
  var texts=[
    "<strong>输入文本：</strong>用户给一句话，比如“猫 吃 鱼”。模型一次只处理固定长度的序列，不是一整篇文章。",
    "<strong>分词：</strong>把文本切成 token。中文一个字常是一个 token，“猫 吃 鱼”是 3 个 token，序列长度 seq_len = 3。",
    "<strong>Token ID：</strong>每个 token 在词表里有一个编号，变成整数序列。输入形状是 (batch_size, sequence_length)，比如 (1, 3)。",
    "<strong>Transformer：</strong>模型主体，由 N 个 Transformer 块组成。每个块里有注意力（让 token 互相看）和 FFN（逐个 token 加工）。输出形状是 (batch_size, sequence_length, vocab_size)。",
    "<strong>Logits：</strong>每个位置对词表里每个词都打一个分数。推理时只看最后一个位置的分数分布。",
    "<strong>解码：</strong>从最后一个位置的分布里选出下一个词，拼回序列，再跑一遍，生成下下个词。重复直到生成结束符或达到长度上限。"
  ];
  document.querySelectorAll('.flowbox').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('.flowbox').forEach(function(x){x.classList.remove('on');});
      b.classList.add('on');
      document.getElementById('flow-explain').innerHTML=texts[+b.getAttribute('data-stage')];
    });
  });
})();
</script>

### 形状总览：输入什么形状，输出什么形状

- 输入：`(batch_size, sequence_length)`，整数 token ID
- 输出：`(batch_size, sequence_length, vocab_size)`，每个位置的 next-token logits
- 推理：取最后一个位置的分数分布，采样生成下一个 token，拼接回序列后重复
- 结构：现代大模型统一用 decoder-only，CS336 版也是。每个 Transformer 块由两个子层组成，均采用 Pre-Norm（先归一化、再子层、再残差）

## 0.1 数学基础

### 函数

函数是一组输入到输出的映射关系。大模型本质就是一个超大规模的嵌套函数：输入 token，输出下一个 token 的概率。

例子：$f(x) = x^2$，输入 2 输出 4，输入 -2 输出 4。输入变，输出跟着变，这就是映射。

### 导数与梯度：能算出来才算懂

导数是单变量函数在某点的变化率。梯度是多元函数对每个变量求偏导组成的向量，指向函数值上升最快的方向。模型训练靠梯度反向传播更新参数，推理时不更新参数，但要看懂公式必须先懂它。

#### 演示 1：导数，拖动滑块看变化率

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<label>x = <span id="deriv-x-val">2</span> <input id="deriv-x" type="range" min="-3" max="3" step="0.5" value="2" style="width:200px;"></label>
<p id="deriv-out">f(x) = x² = 4，f'(x) = 2x = 4。x 每增加一点点，f 大约增加 4 倍那么多。</p>
</div>
<script>
(function(){
  var s=document.getElementById('deriv-x');
  s.addEventListener('input', function(){
    var x=parseFloat(s.value);
    document.getElementById('deriv-x-val').textContent=x;
    document.getElementById('deriv-out').textContent='f(x) = x² = '+(x*x)+ '，f\'(x) = 2x = '+(2*x)+'。x 每增加一点点，f 大约增加 '+(2*x)+' 倍那么多。';
  });
})();
</script>

#### 演示 2：梯度下降，点一步走一步

损失 $L(w) = w^2$，起点 $w = 5$。梯度是 $2w$，更新是 $w \leftarrow w - lr \times 2w$。学习率太大就发散，太小就走得慢，点一点试试。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<label>学习率 lr = <span id="gd-lr-val">0.1</span> <input id="gd-lr" type="range" min="0.01" max="1.1" step="0.01" value="0.1" style="width:200px;"></label>
<button id="gd-step" style="padding:6px 12px;cursor:pointer;">走一步</button>
<button id="gd-reset" style="padding:6px 12px;cursor:pointer;">回到 w=5</button>
<p id="gd-out">w = 5，损失 L = 25</p>
</div>
<script>
(function(){
  var w=5, lr=0.1;
  var out=document.getElementById('gd-out');
  function render(){ out.textContent='w = '+w.toFixed(3)+'，损失 L = w² = '+(w*w).toFixed(3)+'，梯度 2w = '+(2*w).toFixed(3); }
  document.getElementById('gd-lr').addEventListener('input', function(){ lr=parseFloat(this.value); document.getElementById('gd-lr-val').textContent=lr; });
  document.getElementById('gd-step').addEventListener('click', function(){ w=w-lr*2*w; render(); });
  document.getElementById('gd-reset').addEventListener('click', function(){ w=5; render(); });
  render();
})();
</script>

### 向量与矩阵

向量是一维数字序列，对应一个 token 的词向量。矩阵是二维数字表格，对应模型权重。数据通过矩阵乘法完成特征变换，是 Transformer 的运算基础。

#### 演示 3：向量点积，一项一项算

$a = [1, 2, 3]$，$b = [4, 5, 6]$，点积 $a \cdot b = 1\times4 + 2\times5 + 3\times6 = 32$。点积越大，两个向量越“像”，这就是注意力分数的雏形。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<p>a = [1, 2, 3]，b = [4, 5, 6]。点一项，算一项。</p>
<button class="dot-term" data-v="4" style="padding:8px 12px;margin:2px;cursor:pointer;">1×4 = 4</button>
<button class="dot-term" data-v="10" style="padding:8px 12px;margin:2px;cursor:pointer;">2×5 = 10</button>
<button class="dot-term" data-v="18" style="padding:8px 12px;margin:2px;cursor:pointer;">3×6 = 18</button>
<p id="dot-out">还没点。点积是三项加起来。</p>
</div>
<script>
(function(){
  var sum=0, n=0;
  document.querySelectorAll('.dot-term').forEach(function(b){
    b.addEventListener('click', function(){
      if(b.disabled) return;
      b.disabled=true; b.style.opacity=0.5;
      sum+=+b.getAttribute('data-v'); n++;
      document.getElementById('dot-out').textContent = n<3 ? '已算 '+n+' 项，累计 '+sum+'，还差 '+(3-n)+' 项。' : 'a · b = 4 + 10 + 18 = 32。点积越大，两个向量越“像”，这就是注意力分数的雏形。';
    });
  });
})();
</script>

#### 演示 4：矩阵乘法，点结果格子看它怎么来

<div id="matmul-demo" style="border:1px solid #888;padding:12px;border-radius:8px;">
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

## 0.2 Python 基础

Python 是一门计算机编程语言。先看一个能点的演示，再看要点，完整代码折在下面，输出在注释里。

#### 演示 5：列表分片，拖动看取到谁

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<p>arr = [1, 2, 3, 4, 5]</p>
<label>起点 <input id="slice-s" type="range" min="0" max="4" step="1" value="1" style="width:150px;"> <span id="slice-s-val">1</span></label>
<label>终点 <input id="slice-e" type="range" min="1" max="5" step="1" value="4" style="width:150px;"> <span id="slice-e-val">4</span></label>
<p id="slice-out">arr[1:4] = [2, 3, 4]，取下标 1、2、3，不取 4。</p>
</div>
<script>
(function(){
  var arr=[1,2,3,4,5];
  function render(){
    var s=+document.getElementById('slice-s').value, e=+document.getElementById('slice-e').value;
    document.getElementById('slice-s-val').textContent=s;
    document.getElementById('slice-e-val').textContent=e;
    if(e<=s){ document.getElementById('slice-out').textContent='arr['+s+':'+e+'] = []，终点必须大于起点。'; return; }
    document.getElementById('slice-out').textContent='arr['+s+':'+e+'] = ['+arr.slice(s,e).join(', ')+']，取下标 '+s+' 到 '+(e-1)+'，不取 '+e+'。';
  }
  document.getElementById('slice-s').addEventListener('input', render);
  document.getElementById('slice-e').addEventListener('input', render);
  render();
})();
</script>

### 关键点，一句话一个

- `self` 就是实例自己。`self.dim` 是这个实例的属性，`self.forward()` 是这个实例的方法。
- `__init__` 在 `Module(dim=4)` 那一刻自动跑，用来把参数存下来。
- 分片左闭右开：`arr[1:4]` 取下标 1、2、3，不取 4。
- `f"..."` 是 f-string，大括号里的变量会被算出来再拼进字符串。
- Transformer 里所有模块都是类，都是这个套路：`__init__` 里建参数，`forward` 里写计算流程。

<details>
<summary>展开看完整代码：先修的全部 Python 要点（输出在注释里）</summary>

```python
# 1. 变量与基础类型
num = 10          # 整数
arr = [1, 2, 3, 4, 5]  # 其他语言中叫数组（array），Python 中叫列表（list）

# 2. 函数定义
def add(x, y):
    return x + y

res = add(arr[0], arr[1])
print(f"arr[0] 值: {arr[0]}")  # arr[0] 值: 1
print(f"arr[1] 值: {arr[1]}")  # arr[1] 值: 2
print(f"add(arr[0], arr[1]) 的结果：{add(arr[0], arr[1])}")  # 结果：3

# 3. 分支 + 循环：for 和 while 是一回事，for 更常用
for i in range(3):
    if num > 3:
        print(f"第{i}轮: {i}")  # 第0轮: 0，第1轮: 1，第2轮: 2

i = 0
while i < 3:
    if num > 3:
        print(f"第{i}轮: {i}")
    i += 1

# 4. 列表分片（左闭右开）
sub_arr = arr[1:4]  # 得到 [2, 3, 4]，下标范围为 [1, 4)
print(f"原始数组 arr：{arr}")       # [1, 2, 3, 4, 5]
print(f"arr[1:4]：{sub_arr}")      # [2, 3, 4]

# 5. 一个简单的类；简单来说，类 = 属性 + 方法（类内部的函数）
class Module:
    def __init__(self, dim):
        # __init__ 是构造函数，创建实例时自动执行，用于初始化属性
        self.dim = dim  # 实例属性

    # 定义一个方法
    def forward(self, x):
        return x * self.dim  # 通过 self.xxx 访问实例的属性和方法

# 初始化实例并使用
net = Module(dim=4)
print(f"net.dim: {net.dim}")  # net.dim: 4
print(f"net.forward(3) 的结果: {net.forward(3)}")  # 结果为 12
```

</details>

## 0.3 PyTorch 基础

PyTorch 是基于 Python 的深度学习框架 / 代码库，用来搭建和训练神经网络，自动完成张量运算与梯度求导。

### 先看形状怎么走

| 步骤 | 形状 | 发生了什么 |
| --- | --- | --- |
| 创建 x | (3,) | 一维张量，开启梯度追踪 |
| 创建 y | (3, 3) | 二维张量 |
| y 搬到设备 | (3, 3) | `.to(device)`，CPU 或 GPU |
| 矩阵乘法 x @ w | (3,) | 向量乘矩阵，还是 (3,) |
| 网络输入 → 输出 | (3, 10) → (3, 2) | nn.Linear(10, 2)，每行独立算 |

### 关键点，一句话一个

- `torch.tensor` 是多维数组，`shape` 告诉你每一维多大，`requires_grad=True` 才追踪梯度。
- `.to(device)` 把张量搬到 GPU 或 CPU，模型和数据必须在同一个设备上，否则报错。
- `x @ w` 就是矩阵乘法，`@` 是 Python 的矩阵乘法运算符。
- 所有网络都继承 `nn.Module`，`__init__` 里建层，`forward` 里写流程，调用 `net(x)` 时自动走 `forward`。
- `nn.Linear(10, 2)`：输入 10 维，输出 2 维，内部就是 `y = x @ W.T`，W 的形状是 (2, 10)。

<details>
<summary>展开看完整代码：创建张量、矩阵乘法、定义一个网络</summary>

```python
import torch
import torch.nn as nn

# 1. 创建张量 Tensor（深度学习的基础数据结构，本质上是多维数组，
#    但支持梯度自动计算，以及搬到 GPU 上进行并行计算）
# 一维常量张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"张量 x 的值：{x}")            # tensor([1., 2., 3.], requires_grad=True)
print(f"张量 x 的形状：{x.shape}")    # torch.Size([3])
print(f"张量 x 是否开启梯度追踪：{x.requires_grad}")  # True

# 二维常量张量
y = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
print(f"张量 y 的形状：{y.shape}")    # torch.Size([3, 3])
print(f"张量 y 是否开启梯度追踪：{y.requires_grad}")  # False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将张量迁移到目标设备
y_gpu = y.to(device)
print(f"迁移后张量 y_gpu 所在的设备：{y_gpu.device}")

# 2. 矩阵乘法（Transformer 大模型最核心的基础运算）
w = torch.ones(3, 3)
y = x @ w  # 写法等价于 torch.matmul(x, w)
print(f"矩阵乘法结果 y：{y}")  # tensor([3., 3., 3.], grad_fn=<MmBackward0>)

# 3. 一个简单网络的定义（PyTorch 中所有网络都继承 nn.Module 基类）
class Net(nn.Module):
    def __init__(self):
        super().__init__()  # 固定写法，必须调用父类的构造函数
        self.fc = nn.Linear(10, 2)  # 全连接层：输入维度 10，输出维度 2

    # 前向传播函数：定义数据在网络里的计算流程
    def forward(self, x):
        return self.fc(x)  # x * W_fc, (1, 10) * (10, 2) -> (1, 2)

# 初始化网络实例并运行
net = Net()
input_x = torch.randn(3, 10)  # (3, 10)
output = net(input_x)  # 把实例名 net 当作函数来用，相当于 net.forward(input_x)
print(f"网络输入形状：{input_x.shape}")   # torch.Size([3, 10])
print(f"网络输出形状：{output.shape}")   # torch.Size([3, 2])
```

</details>

## 1.2 Tensor 基本运算

在 Transformer 中，很多计算是对 "batch-like" 维度的批量操作。这一节是先修里最重要的一节，每一个概念都有图。

### Batch、Sequence、Head：三个维度各管什么

先把形状记死：一个 Transformer 输入的形状是 `(batch, seq, d_model)`，比如 `(2, 3, 4)`。

- **Batch 维度**：对 batch 中每个样本独立执行相同的 Transformer 前向操作。batch 0 和 batch 1 互不干扰，只是拼在一起算，省时间。
- **Sequence 维度**：RMSNorm、FFN 等“逐位置”操作，在序列的每个 token 上独立执行。一个 token 算自己的，不看别的 token。
- **Head 维度**：多头注意力中，注意力计算在每个头上独立批量执行。每个头看关系的角度不同。

#### 演示 6：Tensor 三维图，点 batch / seq / head 看它管哪一块

形状是 (batch=2, seq=3, d_model=4)，d_model=4 再拆成 head=2、每头 d=2。下面点一个模式，看高亮的是哪一块。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<button id="t-mode-batch" style="padding:6px 12px;cursor:pointer;">按 Batch 看</button>
<button id="t-mode-seq" style="padding:6px 12px;cursor:pointer;">按 Sequence 看</button>
<button id="t-mode-head" style="padding:6px 12px;cursor:pointer;">按 Head 看</button>
<button id="t-next" style="padding:6px 12px;cursor:pointer;">换一个看</button>
<p id="t-explain">按 Batch 看：第 0 个样本。batch 0 和 batch 1 各算各的，互不干扰。</p>
<div id="t-grid"></div>
</div>
<script>
(function(){
  var mode='batch', idx=0;
  var tokens=[['猫','吃','鱼'],['我','爱','你']];
  function render(){
    var html='';
    for(var b=0;b<2;b++){
      html+='<div style="display:inline-block;vertical-align:top;margin:6px;padding:6px;border:1px solid #888;border-radius:6px;">';
      html+='<div><strong>batch '+b+'</strong>（第 '+(b+1)+' 个样本）</div>';
      for(var s=0;s<3;s++){
        var rowOn = (mode==='batch' && b===idx) || (mode==='seq' && s===idx);
        html+='<div style="margin:4px 0;">'+tokens[b][s]+'：';
        for(var h=0;h<2;h++){
          var headOn = mode==='head' && h===idx;
          var color = h===0 ? '#2563eb' : '#dc2626';
          var on = rowOn || headOn;
          for(var d=0;d<2;d++){
            var v=(b*12+s*4+h*2+d+1);
            html+='<span style="display:inline-block;width:28px;text-align:center;padding:4px 0;margin:1px;border:1px solid '+(on?color:'#888')+';border-radius:4px;background:'+(on?color:'#fff')+';color:'+(on?'#fff':'#000')+';">'+v+'</span>';
          }
          html+='<span style="margin-right:6px;color:'+color+';"> 头'+h+'</span>';
        }
        html+='</div>';
      }
      html+='</div>';
    }
    document.getElementById('t-grid').innerHTML=html;
    var max = mode==='head'?2:(mode==='seq'?3:2);
    var label = mode==='batch' ? '按 Batch 看：第 '+idx+' 个样本。batch 0 和 batch 1 各算各的，互不干扰。'
      : mode==='seq' ? '按 Sequence 看：第 '+idx+' 个 token。RMSNorm、FFN 都是逐个 token 独立算的。'
      : '按 Head 看：第 '+idx+' 个头。d_model=4 拆成 2 个头，每头 d=2，两个头各算各的注意力。';
    document.getElementById('t-explain').textContent=label+'（一共 '+max+' 个，点“换一个看”切换）';
  }
  function setMode(m){ mode=m; idx=0; render(); }
  document.getElementById('t-mode-batch').addEventListener('click', function(){ setMode('batch'); });
  document.getElementById('t-mode-seq').addEventListener('click', function(){ setMode('seq'); });
  document.getElementById('t-mode-head').addEventListener('click', function(){ setMode('head'); });
  document.getElementById('t-next').addEventListener('click', function(){ var max=mode==='head'?2:(mode==='seq'?3:2); idx=(idx+1)%max; render(); });
  render();
})();
</script>

记住这张图的读法：batch 是“第几个样本”，seq 是“第几个 token”，head 是“第几个角度”。后面的注意力、FFN、RMSNorm，都是在这三个维度上分工的。

### 场景 1：矩阵乘法，含 batch 广播

先看演示：选 batch，再点结果格子，看每个 batch 怎么独立算。

#### 演示 7：带 batch 的矩阵乘法，选 batch 再点结果格子

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<button id="bm-batch0" style="padding:6px 12px;cursor:pointer;">batch 0</button>
<button id="bm-batch1" style="padding:6px 12px;cursor:pointer;">batch 1</button>
<div id="bm-grid"></div>
<p id="bm-explain">选 batch，再点结果格子。</p>
</div>
<script>
(function(){
  var batchA=[[[1,2],[3,4]],[[5,6],[7,8]]], W=[[1,1],[0,1]];
  var b=0;
  function render(){
    var A=batchA[b], C=[[0,0],[0,0]];
    for(var i=0;i<2;i++) for(var j=0;j<2;j++) C[i][j]=A[i][0]*W[0][j]+A[i][1]*W[1][j];
    var html='<p>A (batch '+b+') = [['+A[0].join(', ')+'], ['+A[1].join(', ')+']]，W = [[1, 1], [0, 1]]</p><table style="border-collapse:collapse;">';
    for(var i=0;i<2;i++){ html+='<tr>'; for(var j=0;j<2;j++){ html+='<td><button data-i="'+i+'" data-j="'+j+'" style="padding:8px 14px;margin:2px;cursor:pointer;">'+C[i][j]+'</button></td>'; } html+='</tr>'; }
    html+='</table>';
    document.getElementById('bm-grid').innerHTML=html;
    document.querySelectorAll('#bm-grid button').forEach(function(btn){
      btn.addEventListener('click', function(){
        var i=+btn.getAttribute('data-i'), j=+btn.getAttribute('data-j');
        document.getElementById('bm-explain').textContent='batch '+b+'，C['+i+']['+j+'] = '+A[i][0]+'×'+W[0][j]+' + '+A[i][1]+'×'+W[1][j]+' = '+C[i][j]+'。这个 batch 只和 W 算，和另一个 batch 无关。';
      });
    });
  }
  document.getElementById('bm-batch0').addEventListener('click', function(){ b=0; render(); });
  document.getElementById('bm-batch1').addEventListener('click', function(){ b=1; render(); });
  render();
})();
</script>

<details>
<summary>展开看完整代码：带 batch 的矩阵乘法与 einsum 写法</summary>

```python
import torch
from einops import einsum

# 1.1 普通二维矩阵乘法 (2, 2) @ (2, 2)
a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([[1, 0],
                  [2, 1]])
out_mat = a @ b  # [[5, 2], [11, 4]]，形状 (2, 2)

# 1.2 带 batch 的矩阵乘法广播 (batch=2, 2, 2) @ (2, 2)
batch_a = torch.tensor([
    [[1, 2],
     [3, 4]],
    [[5, 6],
     [7, 8]]
])  # (2, 2, 2)
w = torch.tensor([[1, 1],
                  [0, 1]])  # (2, 2)
# 矩阵乘法会被自动广播到前面的所有维度上，
# batch 中每一个 (2, 2) 矩阵独立和 w 做矩阵乘法
out_batch = batch_a @ w
# batch 0: [[1, 3], [3, 7]]；batch 1: [[5, 11], [7, 15]]，形状 (2, 2, 2)

# einsum 优雅写法：输入输出维度映射清清楚楚，
# 自动处理任意前置 batch 维度的广播
out_ein = einsum(batch_a, w, "b n k, k m -> b n m")
```

</details>

### 场景 2：逐点乘法，含广播

先看演示：点一下，看 (1, 2) 怎么变成 (2, 2)。

#### 演示 8：广播，点一下看 (1, 2) 怎么变成 (2, 2)

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<p>m = [[1, 2], [3, 4]]，v = [[2, 3]]</p>
<button id="bc-show" style="padding:6px 12px;cursor:pointer;">展开广播</button>
<p id="bc-out">v 只有一行，形状 (1, 2)。</p>
</div>
<script>
(function(){
  var shown=false;
  document.getElementById('bc-show').addEventListener('click', function(){
    shown=!shown;
    document.getElementById('bc-out').textContent = shown
      ? '广播后 v 变成 [[2, 3], [2, 3]]，形状 (2, 2)。再逐个位置相乘：1×2=2，2×3=6，3×2=6，4×3=12，结果 [[2, 6], [6, 12]]。'
      : 'v 只有一行，形状 (1, 2)。';
    this.textContent = shown ? '收起' : '展开广播';
  });
})();
</script>

广播规则，一句话：缺的那一维会被复制，直到形状对齐。只在维度是 1 或者缺失时才能广播，否则报错。

<details>
<summary>展开看完整代码：逐点乘法与广播</summary>

```python
m = torch.tensor([[1, 2],
                  [3, 4]])  # (2, 2)
v = torch.tensor([[2, 3]])   # (1, 2)，触发广播机制
out_elem = m * v
# (1, 2) 的 v 会被复制扩展为 (2, 2)，再对应位置元素相乘
# 结果 [[2, 6], [6, 12]]，形状 (2, 2)
```

</details>

### 场景 3：归约求和，沿最后一维

先看演示：点一行，看它怎么变成一个数。

#### 演示 9：归约求和，点一行算一行

t = [[1, 2], [3, 4], [5, 6]]，沿最后一维求和。点一行，看它变成一个数。

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<button class="sum-row" data-row="0" style="padding:8px 12px;margin:2px;cursor:pointer;">[1, 2]</button>
<button class="sum-row" data-row="1" style="padding:8px 12px;margin:2px;cursor:pointer;">[3, 4]</button>
<button class="sum-row" data-row="2" style="padding:8px 12px;margin:2px;cursor:pointer;">[5, 6]</button>
<p id="sum-out">还没点。求和后最后一维消失了。</p>
</div>
<script>
(function(){
  var rows=[[1,2],[3,4],[5,6]];
  var done=[false,false,false];
  document.querySelectorAll('.sum-row').forEach(function(b){
    b.addEventListener('click', function(){
      var i=+b.getAttribute('data-row');
      if(done[i]) return;
      done[i]=true; b.style.opacity=0.5;
      var parts=done.map(function(d,k){return d?('第'+(k+1)+'行：'+rows[k][0]+'+'+rows[k][1]+'='+(rows[k][0]+rows[k][1])):null;}).filter(Boolean);
      document.getElementById('sum-out').textContent = parts.join('；')+'。'+(done.every(Boolean)?'三行都算完：[3, 7, 11]，形状从 (3, 2) 变成 (3,)，最后一维被消掉了。':'求和后那一维消失，(3, 2) 变成 (3,)。');
    });
  });
})();
</script>

<details>
<summary>展开看完整代码：沿最后一维归约求和</summary>

```python
t = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])
sum_last = t.sum(dim=-1)
# dim=-1 代表沿着最后一个维度做归约求和，该维度会被消除
# 结果 [3, 7, 11]，形状 (3,)
```

</details>

`dim=-1` 就是最后一维。求和后那一维消失了，(3, 2) 变成 (3,)。这个操作在注意力里到处都是：分数加权求和、归一化，都是沿某一维消掉它。

### 场景 4：维度拆分与重排 rearrange

先看演示：点一下，看 8 维怎么拆成 2 个头。

#### 演示 10：rearrange，8 维怎么拆成 2 个头 × 4 维

<div style="border:1px solid #888;padding:12px;border-radius:8px;">
<p>一个 token 的 8 维向量：[1, 2, 3, 4, 5, 6, 7, 8]</p>
<button id="rg-split" style="padding:6px 12px;cursor:pointer;">拆成 2 个头</button>
<p id="rg-out">还没拆。d_model=8，h=2，每头 d=4。</p>
</div>
<script>
(function(){
  var split=false;
  document.getElementById('rg-split').addEventListener('click', function(){
    split=!split;
    document.getElementById('rg-out').innerHTML = split
      ? '头 0：[1, 2, 3, 4]（蓝色）&nbsp;&nbsp;头 1：[5, 6, 7, 8]（红色）。rearrange 写成 "b s (h d) -&gt; b h s d"，就是把 8 拆成 2×4，再把 head 维提到 seq 前面。'
      : '还没拆。d_model=8，h=2，每头 d=4。';
    this.textContent = split ? '合回去' : '拆成 2 个头';
  });
})();
</script>

<details>
<summary>展开看完整代码：view + transpose 与 rearrange 对照</summary>

```python
from einops import rearrange

# (batch, seq, d_model)，d_model=8 拆分为 h=2, d=4
x = torch.tensor([
    [[1, 2, 3, 4, 5, 6, 7, 8],
     [9, 10, 11, 12, 13, 14, 15, 16]]
])  # shape (batch=1, seq=2, d_model=8)

# 传统抽象写法：view 做维度拆分，再 transpose 交换维度，难读且易错
x_trad = x.view(1, 2, 2, 4).transpose(1, 2)
# 形状 (1, 2, 2, 4)

# rearrange 优雅写法：显式声明维度拆分与顺序变换，替代 view + transpose
x_rearr = rearrange(x, "b s (h d) -> b h s d", h=2, d=4)
# 形状 (1, 2, 2, 4)，和上面结果一样，但一眼就能看懂
```

</details>

### 1.2.1 数学记号与代码实现的差异：行向量 vs 列向量

在机器学习中，传统数学推导与 PyTorch 代码实现的核心差异在于向量排布方式：数学习惯使用列向量，而 PyTorch 默认使用行优先内存排布。

- **传统数学（列向量）**：线性变换通常表示为 $y = Wx$，若要进行批处理（Batching），Batch 维度需放在最后，即 $\tilde{X} \in \mathbb{R}^{d_{in} \times batch}$。
- **PyTorch（行向量）**：线性变换表示为 $y = xW^T$。这种方式使得批处理极其自然，只需在最外层增加维度，输入即可直接转化为符合直觉的 $X \in \mathbb{R}^{batch \times d_{in}}$ 格式。

对照着看：

|  | 数学（列向量） | PyTorch（行向量） |
| --- | --- | --- |
| 一个样本 | $y = Wx$，$x$ 是列向量 | $y = xW^T$，$x$ 是行向量 |
| 一批样本 | $\tilde{X} \in \mathbb{R}^{d_{in} \times batch}$，batch 在最后 | $X \in \mathbb{R}^{batch \times d_{in}}$，batch 在最外层 |
| 代码里 | 很少这样写 | `x @ W.T`，到处都是这个写法 |

记住结论就行：看到 `x @ W.T`，就是行向量的线性层，输入是 (batch, d_in)，输出是 (batch, d_out)。

## 自测：先修过关了吗

先自己答，答完再对答案。

### 第 1 题

不看上面，能说出 (batch, seq, d_model) 三个维度各管什么吗？

<details>
<summary>点开看解析和答案</summary>

- batch：第几个样本。batch 0 和 batch 1 各算各的，互不干扰，只是拼在一起算。
- seq：第几个 token。RMSNorm、FFN 都是逐个 token 独立算的。
- d_model：一个 token 的向量有多长，比如 4 维。注意力里它再拆成 head × 每头维度。

</details>

### 第 2 题

(2, 2, 2) @ (2, 2) 的结果是什么形状？每个 batch 怎么算？

<details>
<summary>点开看解析和答案</summary>

结果形状是 (2, 2, 2)。前面那个 2 是 batch 数，算的时候 batch 维不动，batch 里的每个 (2, 2) 矩阵各自和后面的 (2, 2) 矩阵做普通矩阵乘法。

比如 batch_a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]，w = [[1, 1], [0, 1]]，结果 batch 0 是 [[1, 3], [3, 7]]，batch 1 是 [[5, 11], [7, 15]]。两个 batch 互不干扰。

</details>

### 第 3 题

(1, 2) 和 (2, 2) 相乘，会广播成什么？结果是多少？

<details>
<summary>点开看解析和答案</summary>

拿 m = [[1, 2], [3, 4]]，v = [[2, 3]] 来说。v 形状 (1, 2)，第一维是 1，会被复制扩展成 (2, 2)，变成 [[2, 3], [2, 3]]，再逐个位置相乘。

结果是 [[2, 6], [6, 12]]：1×2=2，2×3=6，3×2=6，4×3=12。

广播规则：缺的那一维会被复制，直到形状对齐，只在维度是 1 或者缺失时才能广播，否则报错。

</details>

### 第 4 题

`rearrange(x, "b s (h d) -> b h s d", h=2, d=4)` 里，(h d) 是什么意思？

<details>
<summary>点开看解析和答案</summary>

(h d) 是把原来的一维拆成两维。x 的形状是 (1, 2, 8)，最后一维 d_model=8，拆成 h=2 个头、每头 d=4，变成 (1, 2, 2, 4)，再把 head 维提到 seq 前面，变成 (1, 2, 2, 4)，也就是 (batch, head, seq, d)。

括号的意思就是“这一维是拼出来的，拆开它”。不写括号，rearrange 不知道 8 是 2×4 还是 4×2。

</details>

### 第 5 题

`x @ W.T` 里，W 的形状为什么是 (d_out, d_in)？

<details>
<summary>点开看解析和答案</summary>

因为 PyTorch 用行向量，线性层是 $y = xW^T$。x 的形状是 (batch, d_in)，要输出 (batch, d_out)，就得让 $W^T$ 是 (d_in, d_out)，所以 W 本身是 (d_out, d_in)。

对照数学的列向量写法 $y = Wx$：W 也是 (d_out, d_in)，只是 x 是列向量。W 的形状没变，变的是 x 摆成行还是列。

</details>

全答得出来，先修就过了。

---

下一步：[第一步：分词，文本怎么变成数字](/posts/llm-study-notes-inference-tokenization/)
