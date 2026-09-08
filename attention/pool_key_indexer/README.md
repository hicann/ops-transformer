# PoolKeyIndexer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`pool_key_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。与`lightning_indexer`不同，`pool_key_indexer`在Key侧引入了**池化（Pooling）**机制，将原始Key序列按`pool_size`个token为一组进行聚合（每个pool为一个token），先在pool级别做TopK选取，再将选中的pool展开为原始token索引。

- 计算公式：

    $$
    Top\text{-}k \leftarrow Expand\left(Top\text{-}\frac{k}{p}\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}^{pool}}\right)\odot ReLU\left(\left(Q_{index}@{K_{index}^{Pool}}^T\right)\odot\frac{1}{\sqrt{d}}\right)\right]\right\}\right) + AppendTail
    $$

    其中$p = pool\_size$，$S_{k}^{pool} = \lfloor S_{k} / pool\_size \rfloor$，$d$为注意力头维度（headDim），缩放系数$\frac{1}{\sqrt{d}}$防止点积过大；$Expand$将每个pool索引展开为$p$个连续token索引，$AppendTail$追加尾部pool中`pool_tail_k`个有效token。

    主要计算过程为：

    1. 将某个token对应的输入参数`query`（$Q_{index}\in\R^{g\times d}$）乘以池化后的上下文`pool_key`（$K_{index}^{Pool}\in\R^{S_{k}^{pool}\times d}$），再乘以缩放系数$\frac{1}{\sqrt{d}}$，得到相关性分数。
    2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的pool级相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向聚合。
    4. 在pool空间选取前$Top\text{-}k / pool\_size$个pool索引，展开为$Top\text{-}k$个token索引。
    5. 追加尾部pool中未被完整pool_size覆盖的有效token（由`pool_tail_k`指定），无效位置填-1。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、S1和S2分别表示query和pool_key的Sequence Length（S2为pool级别）、N1和N2分别表示query和pool_key的Head Num、D表示Head Dim（仅支持128）、T1和T2分别表示query和pool_key的Total Tokens、block_num和block_size分别表示PageAttention场景下的block总数和每个block包含的pool数。N2仅支持1。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| query | 输入 | 公式中的输入Q。不支持空tensor。layout_q为BSND时，shape为[B, S1, N1, D]；layout_q为TND时，shape为[T1, N1, D]。 | FLOAT16、BFLOAT16、FLOAT8_E4M3FN | ND |
| pool_key | 输入 | 公式中的池化后输入K。不支持空tensor。layout_k为PA_BBND时，shape为[block_num, block_size, N2, D]；layout_k为BSND时，shape为[B, S2, N2, D]；layout_k为TND时，shape为[T2, N2, D]。数据类型需与query保持一致。 | FLOAT16、BFLOAT16、FLOAT8_E4M3FN | ND |
| weights | 输入 | 公式中的输入W。不支持空tensor。layout_q为BSND时，shape为[B, S1, N1]；layout_q为TND时，shape为[T1, N1]。 | FLOAT16、BFLOAT16 | ND |
| pool_tail_k | 输入 | 每个Batch中尾部不完整pool的有效token数。取值范围[0, pool_size-1]，0表示无尾部有效token。shape为[B, ]。 | INT64 | ND |
| actual_seq_q | 输入 | 每个Batch中Query的有效token数。layout_q为TND时必传，每个元素的值表示当前batch与之前所有batch的token数总和（前缀和）。shape为[B, ]。 | INT64 | ND |
| actual_seq_k | 输入 | 每个Batch中pool_key的有效pool数。layout_k为TND时必传（前缀和）；layout_k为PA_BBND时必传（非前缀和，表示当前batch的pool数）。shape为[B, ]。 | INT64 | ND |
| block_table | 输入 | 表示PageAttention中KV存储使用的block映射表。layout_k为PA_BBND时必传，shape为[B, maxBlockNumPerSeq]。 | INT32 | ND |
| q_descale | 输入 | Query的反量化系数。仅quant_mode>=0时有效：quant_mode为0时shape为[B, S1, N1]（BSND）或[T1, N1]（TND）；quant_mode为1时shape为[B, S1, N1, D/64, 2]（BSND）或[T1, N1, D/64, 2]（TND）。 | FLOAT、FLOAT8_E8M0 | ND |
| k_descale | 输入 | pool_key的反量化系数。仅quant_mode>=0时有效：quant_mode为0时shape为[B, S2, N2]（BSND）、[T2, N2]（TND）或[block_num, block_size, N2]（PA_BBND）；quant_mode为1时在最后一维追加[D/64, 2]。仅PA_BBND布局下支持0轴非连续。 | FLOAT、FLOAT8_E8M0 | ND |
| layout_q | 属性 | 用于标识输入Query的数据排布格式。支持BSND、TND，默认值为BSND。 | STRING | - |
| layout_k | 属性 | 用于标识输入pool_key的数据排布格式。支持PA_BBND、BSND、TND，默认值为BSND。 | STRING | - |
| topk | 属性 | 展开后需要保留的token数量。支持[1, 2048]以及3072、4096、5120、6144、7168、8192，需满足topk % pool_size == 0。默认值为2048。 | INT64 | - |
| pool_size | 属性 | 每个pool包含的token数量。支持[1, 128]，默认值为16。取值为1时表示无池化，退化为lightning_indexer行为。 | INT64 | - |
| mask_mode | 属性 | 表示mask的模式。0表示defaultMask模式，3表示rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。默认值为3。 | INT64 | - |
| quant_mode | 属性 | 表示Query/Key的量化模式。-1表示不量化，0表示FP8 per-token-head量化（反量化系数为FLOAT），1表示mxFP8量化（反量化系数为FLOAT8_E8M0）。默认值为-1。 | INT64 | - |
| return_value | 属性 | 表示是否输出sparse_values。True表示输出，False表示不输出（输出shape为(0,)的空tensor）。默认值为False。 | BOOL | - |
| key_stride0 | 属性 | pool_key第0轴的stride（元素单位），用于layout_k为PA_BBND场景下0轴非连续pool_key的寻址。-1表示未指定，按连续输入由shape推导stride。默认值为-1。 | INT64 | - |
| k_descale_stride0 | 属性 | k_descale第0轴的stride（元素单位），用于layout_k为PA_BBND场景下0轴非连续k_descale的寻址，仅量化场景有效。-1表示未指定，按连续输入由shape推导stride。默认值为-1。 | INT64 | - |
| sparse_indices | 输出 | 公式中的Indices输出，展开后的稀疏token索引，无效部分填-1。layout_q为BSND时，shape为[B, S1, topk + pool_size - 1]；layout_q为TND时，shape为[T1, topk + pool_size - 1]。 | INT32 | ND |
| sparse_values | 输出 | 公式中的Indices对应的Values输出，即选中pool的分数值，无效部分填-inf。return_value为True时，layout_q为BSND时shape为[B, S1, topk // pool_size]，layout_q为TND时shape为[T1, topk // pool_size]；return_value为False时，shape为(0,)。 | FLOAT | ND |

## 约束说明

- 该接口支持推理场景下使用，支持单算子模式、TorchAir（GE图）模式和TorchAir（aclgraph）图模式调用。
- query的N1支持小于等于64，pool_key的N2仅支持1，D（HeadDim）仅支持128。
- 参数query、pool_key的数据类型应保持一致；非量化场景（quant_mode为-1）下，参数weights的数据类型应与query、pool_key保持一致；量化场景（quant_mode为0或1）下，参数weights为FLOAT16或BFLOAT16。
- topk需满足topk % pool_size == 0；topk支持[1, 2048]以及3072、4096、5120、6144、7168、8192；pool_size支持[1, 128]。
- 当pool_size=1且pool_tail_k=0时，退化为lightning_indexer行为，输出维度为topk。
- 当layout_k为PA_BBND时，必须传入block_table和actual_seq_k；当layout_k不为PA_BBND时，不支持传入block_table。
- 当layout_q为BSND时，不支持传入actual_seq_q；当layout_q为TND时，必须传入actual_seq_q。当layout_k为BSND时，不支持传入actual_seq_k；当layout_k为TND时，必须传入actual_seq_k。
- PA_BBND场景下，block_size取值为16的倍数，最大支持1024。
- pool_key、k_descale仅支持在PA_BBND布局下0轴非连续（需配合key_stride0/k_descale_stride0属性），BSND/TND布局要求连续输入。
- 当quant_mode=-1时，不支持传入q_descale和k_descale。
- pool_tail_k取值校验仅对CPU输入在host侧执行；NPU输入经aclnn Tensor变体直传device数据，host侧不做值校验，调用方须自行保证取值合法，否则kernel行为未定义。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../docs/zh/context/sparse_mode_introduction.md)。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - query、pool_key支持FLOAT8_E4M3FN数据类型。
  - quant_mode支持-1（不量化）、0（FP8 per-token-head量化）、1（mxFP8量化）。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 不支持FLOAT8_E4M3FN与FLOAT8_E8M0数据类型。
  - quant_mode仅支持-1（不量化），quant_mode为0/1暂不支持。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn接口 | [test_aclnn_pool_key_indexer](./examples/test_aclnn_pool_key_indexer.cpp) | 通过[aclnnPoolKeyIndexer](./docs/aclnnPoolKeyIndexer.md)接口方式调用算子 |
