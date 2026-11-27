---
title: "第四步：Transformer 块，逐个拆"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理", "Transformer", "RMSNorm", "RoPE", "注意力", "FFN"]
description: "大模型推理第四步：Transformer 块逐个拆，RMSNorm、RoPE、注意力、FFN。"
---

这是 [大模型推理](/posts/llm-study-notes-inference/) 的第四步，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。上一页是[第三步：最基础的两个模块](/posts/llm-study-notes-inference-linear-embedding/)。

第二步我们看了 Transformer 整体长什么样，现在把一个块拆开，一个模块一个模块讲透。

## 一个块里有什么

一个 Transformer 块（Pre-Norm 结构）是两条子层，每条子层前面一个 RMSNorm，后面一个残差相加：

| 顺序 | 模块 | 干什么 |
| --- | --- | --- |
| ① | RMSNorm | 归一化，把每个 token 的向量缩放到合适大小 |
| ② | 多头自注意力 | token 之间互相看，交换信息 |
| ③ | ＋x（残差） | 把输入加回来，形状不变 |
| ④ | RMSNorm | 再归一化一次 |
| ⑤ | FFN（SwiGLU） | 逐个 token 独立加工，升维再降维 |
| ⑥ | ＋x（残差） | 再把输入加回来，形状不变 |

形状全程不变：输入 3 行 × 4 列，输出还是 3 行 × 4 列。

## 拆开讲，每块单独一页

- [4.1 RMSNorm：归一化](/posts/llm-study-notes-inference-rmsnorm/)
- 4.2 RoPE：旋转位置编码（还没写）
- 4.3 多头自注意力：QKV、注意力分数、遮罩 + Softmax、加权求和、拼接 + 投影（还没写）
- 4.4 FFN（SwiGLU）：前馈网络（还没写）

其中 4.3 注意力最大，会按参考站的 03 到 07 一步步拆：QKV → 注意力分数 → 遮罩 + Softmax → 加权求和 → 拼接 + 投影，每一步都有动画。

---

回目录：[大模型推理](/posts/llm-study-notes-inference/)
