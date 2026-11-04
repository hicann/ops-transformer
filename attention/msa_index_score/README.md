# MsaIndexScore

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- **算子功能**：计算MSA（MiniMax Sparse Attention）Index Branch的block score。对每个query token与每个KV sparse block，取该block内因果可见token的$Q_{idx}$与$K_{idx}$做matmul，再按block做Maxpool，得到逐block的重要性分数。本算子不包含TopK。Prefill与Decode共用本接口。

- **计算公式**：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    非量化时没有$scale$。`sparse_mode=3`时$atten\_mask$为右下因果；`sparse_mode=0`时不叠加。$local\_mask$由`start_loc`、`init_blocks`、`local_blocks`生成，对强制保留的block写入高分。

- **主要计算过程**：

    1. 将每个query token的$Q_{idx}$与KV sparse block内因果可见的$K_{idx}$做矩阵乘，得到token级相关性；
    2. 按`sparse_mode`叠加$atten\_mask$：`sparse_mode=0`不做因果截断，`sparse_mode=3`按rightDownCausal屏蔽不可见位置。
    3. 在每个长度为$block\_size$的sparse block上做Maxpool，得到逐block的重要性分数。
    4. 用`start_loc`、`init_blocks`、`local_blocks`生成$local\_mask$，对强制保留的block写入高分，得到输出`score`。本算子不包含TopK。

## 参数说明

> **说明：**
>
> - B为batch，S为序列长度（$S1$ query，$S2$ key），T为各batch长度累加（$T1$ / $T2$）。
> - N为头数（$N1$ query，$N2$ key），D为head dim。
> - PageAttention下$block\_num$为物理page数，$block\_size$为每页token数。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| query | 输入 | $Q_{idx}$。非V2固定TND $[T1, N1, D]$。V2还可为BSND $[B, S1, N1, D]$、BNSD $[B, N1, S1, D]$，这两种仅<term>Ascend 950PR&950DT系列产品</term>。 | BFLOAT16、FLOAT16；<term>Ascend 950PR&950DT系列产品</term>另支持FLOAT8_E5M2、FLOAT8_E4M3FN。 | ND |
| key | 输入 | $K_{idx}$。TND $[T2, N2, D]$；BSND $[B, S2, N2, D]$；BNSD $[B, N2, S2, D]$；BBND $[block\_num, block\_size, N2, D]$；BNBD $[block\_num, N2, block\_size, D]$。V2的非分页key必须与query同布局，且BSND/BNSD仅<term>Ascend 950PR&950DT系列产品</term>、逻辑块长16。PA允许首轴非连续。 | 与query同型。 | ND |
| block_table | 可选输入 | PA的逻辑block到物理page映射，shape $[B, maxBlockNumPerSeq]$。TND / BSND / BNSD不传。 | INT32 | ND |
| scale | 可选输入 | 非量化与FP8必须为空。 | FLOAT | ND |
| atten_mask | 可选输入 | `sparse_mode=3`时必传，shape $[2048, 2048]$。1表示不参与计算，0表示参与。 | INT8 | ND |
| actual_seq_qlen | 可选输入 | query前缀和，shape $[B+1]$。 | INT32 | ND |
| actual_seq_klen | 可选输入 | TND / BSND / BNSD为key前缀和$[B+1]$。PA为各请求可见长度$[B]$。BSND/BNSD每个请求的长度差不超过$S2$。 | INT32 | ND |
| start_loc | 输入 | 当前query所在逻辑block索引，shape $[B]$，用于$local\_mask$。 | INT32 | ND |
| layout_key | 属性 | `TND` / `BSND` / `BNSD` / `BBND` / `BNBD`。默认`BBND`。 | STRING | - |
| sparse_mode | 属性 | 0：defaultMask；3：rightDownCausal。 | INT64 | - |
| init_blocks | 属性 | 头部强制保留的block数，默认0。 | INT64 | - |
| local_blocks | 属性 | 当前query附近强制保留的窗口长度，默认1。与raw score对齐时置0。 | INT64 | - |
| layout_query | 属性 | `TND` / `BSND` / `BNSD`。默认`TND`。（aclnnMsaIndexScore接口不支持设置，固定为`TND`；aclnnMsaIndexScoreV2接口支持设置） | STRING | - |
| pool_block_size | 属性 | 每个score归约的token数，只允许16 / 128，默认128。（aclnnMsaIndexScore接口不支持设置，固定为128；aclnnMsaIndexScoreV2接口支持设置） | INT64 | - |
| score | 输出 | 逐block重要性分数。块长128的shape为$[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$；块长16做完head间max后head维为1，shape为$[1, T1, RoundUp(maxBlockNumPerSeq, 16)]$。 | FLOAT | ND |

`layout_query`与`pool_block_size`由本接口固定为`TND`与128，不在`aclnnMsaIndexScore`参数表中。可配置这两项的是[aclnnMsaIndexScoreV2](./docs/aclnnMsaIndexScoreV2.md)。

## 约束说明

- 非V2 aclnn接口（`aclnnMsaIndexScore`）：query固定TND，逻辑块长128。key可为`BBND` / `BNBD` / `TND` / `BSND` / `BNSD`，<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>与<term>Ascend 950PR&950DT系列产品</term>都支持。BSND / BNSD不传`block_table`，`actual_seq_klen`为$[B+1]$前缀和，每个请求的长度差不超过key的$S2$。
- V2 aclnn接口（`aclnnMsaIndexScoreV2`）：query可为TND / BSND / BNSD。非分页key必须与query同布局，仅支持<term>Ascend 950PR&950DT系列产品</term>，`pool_block_size`为16，`actual_seq_klen`为$[B+1]$前缀和。
- PageAttention必须传`block_table`，`actual_seq_klen`为$[B]$可见长度。TND / BSND / BNSD不得传`block_table`，`actual_seq_klen`为$[B+1]$前缀和。
- 非量化时key与query同dtype，`scale`为空。<term>Ascend 950PR&950DT系列产品</term>支持FLOAT8_E5M2与FLOAT8_E4M3FN。
- `sparse_mode`仅0或3。为3时`atten_mask`必须是$[2048, 2048]$。
- `init_blocks`、`local_blocks`必须$\ge 0$，且不超过逻辑block数。两者都为0时不写$local\_mask$。
- `q_len` / `kv_len`允许为0。空KV的score填`-inf`。
- 本算子不包含TopK。
- 确定性计算：相同输入与属性下，输出与参考实现一致。

## 调用说明

| 调用方式 | 样例 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_msa_index_score](./examples/test_aclnn_msa_index_score.cpp) | 通过[aclnnMsaIndexScore](./docs/aclnnMsaIndexScore.md)接口方式调用算子 |
| PyTorch API | - | 通过[msa_index_score](./docs/torchapi_msa_index_score.md)接口调用msa_index_score算子 |
