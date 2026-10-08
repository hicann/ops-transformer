# QuantLightningIndexerV2

本算子是两级 TopK（two-level TopK，先按块粗筛再在块内精筛）方案的**第一级**（source，候选块筛选级）：`candidate_topk_blocks` 取 2048（当前唯一有效值）时开启 source，按块粒度选出候选块并输出块级索引 `candidate_topk_index_out`；取 -1（默认）时关闭，为现网行为。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantLightningIndexerV2是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| query                     | 输入      | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。| INT8 | ND         |
| key                   | 输入      | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，`k`/`k_descale` 支持0轴padding型非连续（仅PA_BBND布局，其余轴必须连续），stride由输入描述符携带，kernel按块号×stride寻址。| INT8 | ND |
| weights                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。 | FLOAT16 | ND |
| query_dequant_scale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续。shape与`weights`一致 | FLOAT16     | ND         |
| key_dequant_scale            | 输入      | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续。shape为移除`key`的D轴 | FLOAT16       | ND         |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`query`的有效token数前缀和。；layout_q为BSND时不能传入 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为TND时必须传入，表示每个Batch中`key`的有效token数前缀和；layout_k为PA_BSND或BSND时不能传入 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`query`的有效token数 | INT32       | ND         |
| seqused_k                    | 可选输入      | layout_k为PA_BSND或BSND时使用，表示每个Batch中`key`的有效token数。| INT32       | ND         |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。| INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表。 | INT32       | ND         |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量 | INT32       | ND         |
| metadata                    | 可选输入      | QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。 | INT32       | ND         |
| quant_mode                 | 属性      | 用于标识输入的量化模式，仅支持2（INT8量化）。 | INT32          | -         |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`query`的数据排布格式，默认值"BSND"。 | STRING | -         |
| layout_k      | 可选属性      | 用于标识输入`key`的数据排布格式，默认值"BSND"。| STRING          | -         |
| topk  | 属性      | 代表topK阶段需要保留的索引数量，默认值2048。 | INT32          | -         |
| mask_mode | 可选属性      | 表示mask的模式，默认值0。 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数，默认值1。 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`，默认值0。 | INT32          | -         |
| candidate_topk_blocks      |  可选属性     | 两级TopK第一级的**开关兼候选块个数**：-1=关闭（现网行为，默认）；**2048**=开启source模式（输出块级候选索引`candidate_topk_index_out`）。当前仅支持 -1 / 2048 | INT32          | -         |
| candidate_block_size      |  可选属性     | 候选块大小（每个块包含的位置数）：`candidate_topk_blocks=-1`（关闭）时**必须为 -1**；开启时必须 **> 0**（当前仅支持 8） | INT32          | -         |
| sparse_indices     | 输出      | 公式中的输出Out，参与稀疏attention计算的token索引值。 | INT32          | ND         |
| sparse_values           | 输出      | 公式中的Indices输出对应的value值。`return_value`为1时shape与`sparse_indices`一致，`return_value`为0时shape为(0,) | BFLOAT16         | ND          |
| candidate_block_length           | 输出（可选）      | **预留接口**，恒输出空 tensor（shape (0,)），本期不产出实际值 | INT32         | ND          |
| candidate_topk_index_out           | 输出      | 两级TopK第一级的输出：块级候选索引（无序，槽位语义为块号或-1无效槽），供QuantSparseLightningIndexer消费。`candidate_topk_blocks=2048`（开启）时shape为(B,S1,N2,candidate_topk_blocks)（BSND）或(T1,N2,candidate_topk_blocks)（TND）；为-1（默认，关闭）时shape为(0,) | INT32         | ND          |

## 约束说明

> 本算子当前仅支持 arch22 (910B/910_93)，Ascend 950 的 arch35 实现已移除；下表 950 列为 ×。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - `quant_mode`仅支持2。
  - `query`、`key`支持INT8，不支持FLOAT8_e4m3fn、HIFLOAT8和FLOAT4_e2m1。
  - `query_dequant_scale`和`key_dequant_scale`支持FLOAT16，不支持FLOAT32和FLOAT8_e8m0。
  - `weights`支持FLOAT16，不支持FLOAT32。
  - 不支持`output_idx_offset`和`return_value`。
  - `key`的N仅支持1；`query`的N与`key`的N之比仅支持64或32（即`query`的N支持64或32）。
  - `topk`支持[1, 2048]。
  - 两级TopK第一级（`candidate_topk_blocks != -1`）当前仅在此两类产品（arch22）上实现：
    - 开启时输出`candidate_topk_index_out`；`layout_q`支持BSND与TND（TND需传入`cu_seqlens_q`，且`layout_k`必须为`PA_BBND`）。
    - `candidate_topk_blocks`当前仅支持 -1（关闭）与 2048（开启）；`candidate_block_size`在关闭时必须为 -1、开启时必须 > 0（当前仅支持 8）。

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| PyTorch API | [quant_lightning_indexer.py](torch_extension/quant_lightning_indexer.py) | 通过`torch.ops.cann_ops_transformer.quant_lightning_indexer`（基础接口）或`torch.ops.cann_ops_transformer.quant_lightning_indexer_candidate`（两级TopK第一级接口，以`candidate_topk_blocks`有效值开启 source）调用本算子。 |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2](examples/test_aclnn_quant_lightning_indexer_v2.cpp) | 通过[aclnnQuantLightningIndexerV2](docs/aclnnQuantLightningIndexerV2.md)两段式接口调用QuantLightningIndexerV2算子。 |
