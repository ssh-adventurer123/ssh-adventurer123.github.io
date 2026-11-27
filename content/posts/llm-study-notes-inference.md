---
title: "大模型推理"
date: 2026-11-27
categories: ["llm-study-notes"]
tags: ["LLM", "大模型", "学习笔记", "推理"]
description: "LLM 学习笔记的子项目：大模型推理。"
---

这是 LLM 学习笔记的子项目：**大模型推理**，隶属于 [LLM 学习笔记](/posts/llm-study-notes/)。

这里讲模型拿到权重之后，是怎么一步步算出下一个词的。只讲推理，不讲训练。

## 目录（七步，不算先修）

### 第一步：分词，文本怎么变成数字

- Unicode 到 UTF-8 字节
- BPE 怎么从 256 个字节出发，一次次合并高频 pair
- encode 和 decode
- 可视化：一段话被切成 token，词表一次次长大

### 第二步：Transformer 整体长什么样

- 输入 (batch, seq) 的 token ID，输出 (batch, seq, vocab) 的 logits
- decoder-only；Pre-Norm；一个块里两个子层，残差在哪
- 可视化：全流程播放，数据从哪进、从哪出，形状每一步怎么变

### 第三步：最基础的两个模块

- Linear：y = xWᵀ，形状、权重初始化；可视化：矩阵乘法逐格算
- Embedding：查表，token ID 对应词表里哪一行；可视化：点一个 ID，高亮一行向量

### 第四步：Transformer 块，逐个拆

- RMSNorm：RMS 怎么算、为什么转 float32、乘可学习增益 g；可视化：逐向量算一遍
- RoPE：位置 i 的 q/k 怎么旋转；角度怎么随位置和分组变；为什么只转 Q/K 不转 V；可视化：二维旋转动画
- Softmax：减 max 防溢出，再 exp 再归一化；可视化：分数变概率
- 缩放点积注意力：Scores = QKᵀ/√d，点积为什么要除 √d；因果掩码把未来盖掉；加权求和；可视化：热力图 + 连线 + 点格子溯源
- 多头自注意力：分头、并行、拼接、再过 Wᴼ；QKV 合并成一个大矩阵一次算完；可视化：多头并排跑
- FFN SwiGLU：SiLU 曲线、GLU 门控相乘、W₁/W₃/W₂ 三路、d_ff 为什么取 8/3 d_model 还凑 64 的倍数；可视化：三路怎么汇成一路

### 第五步：拼成完整的模型

- TransformerBlock：先 RMSNorm、再子层、再残差，两次
- TransformerLM：Embedding → N 个块 → 最后 RMSNorm → LM Head 出 logits
- 可视化：一个 token 从进到出，每层发生了什么

### 第六步：解码，下一个词怎么出来

- 取最后一个位置的 logits；温度、top-p 怎么改分布；贪心和采样
- 生成到 <|endoftext|> 停；KV Cache 为什么能省重算
- 可视化：温度滑杆、top-p 滑杆、KV Cache 随生成一步步长出来

### 第七步：推理工程进阶

- KV Cache、PagedAttention、连续批处理
- 量化 INT8/FP8/AWQ/GPTQ
- vLLM、TGI、TensorRT-LLM、llama.cpp 各适合什么；本地跑起来对照

## 先修（不计入七步）

- 数学：向量、矩阵，只讲矩阵乘法怎么算
- Tensor 和 einsum：batch、sequence、head 三个维度怎么广播；行向量 vs 列向量，为什么代码里是 xWᵀ

## 每篇结构

是什么、为什么、一步步算、代码复现、常见误区、自测题。先看动画跑一遍，再看公式，最后看代码。

## 状态

目录已定，第一步筹备中。
