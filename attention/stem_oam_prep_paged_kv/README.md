# StemOamPrepPagedKv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- 算子功能：Stem OAM (Output-Aware Metric) 大模型推理动态稀疏注意力机制的前置评分模块。从 Paged KV Cache 中提取 K/V 数据，经 K Processing (per-token K-scale × group sum + anti-diagonal flip) 和 V Processing (per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average) 计算，输出 kFlat 和 vBias 供 Stem OAM score computation 使用。

- 输入输出支持以下数据场景：

    ```
    kCache:[total_blocks, kvBlockSize, H_kv, 128] 或 [total_blocks, H_kv, kvBlockSize, 128]
    vCache:[total_blocks, kvBlockSize, H_kv, 128] 或 [total_blocks, H_kv, kvBlockSize, 128]
    kvIndices:[batch, max_kv_blocks]
    kvSeqLens:[batch]
    kScaleCache:[total_blocks, kvBlockSize, H_kv, 1] 或 [total_blocks, H_kv, kvBlockSize, 1]
    vScale:[H_kv]
    kFlat:[batch, H_kv, max_Kb, stemStride * 128]
    vBias:[batch, H_kv, max_Kb]
    ```

- kCache/vCache 支持两种 layout，由 cacheLayout 属性指定：0=Layout A (interleaved), 1=Layout B (contiguous)。
- kScaleCache 布局随 cacheLayout 变化，与 kCache/vCache 布局一致（仅最后一维 D=1 代替 D=128）。
- 仅支持 FP8_E4M3FN 输入路径。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 240px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>kCache</td>
      <td>输入</td>
      <td>Paged K cache。cacheLayout=0时shape为[total_blocks, kvBlockSize, H_kv, 128]，cacheLayout=1时shape为[total_blocks, H_kv, kvBlockSize, 128]。支持非连续（stride>shape），最后一维HeadDim 128连续。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vCache</td>
      <td>输入</td>
      <td>Paged V cache。shape与kCache一致。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kvIndices</td>
      <td>输入</td>
      <td>Block index 数组，每个batch的physical block索引。shape[1]=max_kv_blocks，由kvSeqLens决定：max_kv_blocks = max(ceil(kvSeqLens[b] / kvBlockSize))。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kvSeqLens</td>
      <td>输入</td>
      <td>每batch KV序列长度。该值用于派生kvIndices第二维max_kv_blocks及输出shape中max_Kb。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kScaleCache</td>
      <td>输入</td>
      <td>Per-token per-head K scale。随cacheLayout变化：cacheLayout=0: [total_blocks, kvBlockSize, H_kv, 1]，cacheLayout=1: [total_blocks, H_kv, kvBlockSize, 1]。支持非连续（stride>shape），最后一维连续。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vScale</td>
      <td>输入</td>
      <td>Per-head V scale。shape为[H_kv]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambdaMag</td>
      <td>属性</td>
      <td>V bias 乘数，默认 0.3。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheLayout</td>
      <td>属性</td>
      <td>KV Cache布局，0=Layout A (interleaved), 1=Layout B (contiguous)，默认 0。当前仅支持Layout B</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvBlockSize</td>
      <td>属性</td>
      <td>Paged KV block size，64或128，默认 64。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemBlockSize</td>
      <td>属性</td>
      <td>Stem block大小，%32==0，≤256，推荐128，默认 128。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemStride</td>
      <td>属性</td>
      <td>Stride大小，%16==0，≤64，≤stemBlockSize，推荐16，默认 16。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kFlat</td>
      <td>输出</td>
      <td>K group sum + anti-diag flip 结果。shape为[batch, H_kv, max_Kb, stemStride*128]。</td>
      <td>BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vBias</td>
      <td>输出</td>
      <td>V block bias结果。shape为[batch, H_kv, max_Kb]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- kCache/vCache dtype必须为FLOAT8_E4M3FN（仅FP8路径）。
- kScaleCache/vScale必填，不允许nullptr。
- kScaleCache shape随cacheLayout变化：Layout A `[total_blocks, kvBlockSize, H_kv, 1]`，Layout B `[total_blocks, H_kv, kvBlockSize, 1]`。当前仅支持Layout B。
- vScale shape：`[H_kv]`。
- kvBlockSize ∈ {64, 128}。
- stemBlockSize % 32 == 0，≤256；stemStride % 16 == 0，≤64，且stemStride ≤ stemBlockSize，stemBlockSize必须是stemStride的整数倍。
- 派生值：R = stemBlockSize / stemStride，kflat_dim = stemStride × 128，k_down_len = num_Kb × R。
- 边界：kvSeqLens[b]=0时该batch对应的kFlat/vBias输出全零；kScaleCache padding rows（beyond kv_len）→ zero。
- 仅支持arch35架构（Ascend 950PR/Ascend 950DT），不支持arch22。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_stem_oam_prep_paged_kv](./examples/test_aclnn_stem_oam_prep_paged_kv.cpp) | 通过[aclnnStemOamPrepPagedKv](./docs/aclnnStemOamPrepPagedKv.md)调用StemOamPrepPagedKv算子 |
