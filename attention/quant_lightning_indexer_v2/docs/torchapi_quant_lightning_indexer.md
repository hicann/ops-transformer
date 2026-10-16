# quant\_lightning\_indexer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id8 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id9 -->

## 功能说明

- **接口功能**：

  `quant_lightning_indexer_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的batch、head、以及Q和K的分块的索引，供后续`quant_lightning_indexer`算子使用。

  `quant_lightning_indexer`接口基于一系列操作得到每一个token对应的top-k个位置。主要计算过程为：

  1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
  2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparse\_indices$，并输出对应的$sparse\_values$，作为Attention的输入。

- **计算公式**：

  $$
  out = Top\text{-}k\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}}\right)\odot ReLU\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
  $$

## 函数原型

调用quant_lightning_indexer接口之前，先调用前置接口quant_lightning_indexer_metadata，完成quant_lightning_indexer负载均衡的计算。

```python
cann_ops_transformer.quant_lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  batch_size=None,
  max_seqlen_q=None,
  max_seqlen_k=None,
  layout_q=None,
  layout_k=None,
  mask_mode=None,
  cmp_ratio=None
) -> Tensor
```

```python
cann_ops_transformer.quant_lightning_indexer(
  query,
  key,
  weights,
  query_dequant_scale,
  key_dequant_scale,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  block_table=None,
  output_idx_offset=None,
  metadata=None,
  max_seqlen_q=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1,
  return_value=0
) -> (Tensor, Tensor)
```

## 参数说明

>**说明：**
>
> query、key、weights、query_dequant_scale、key_dequant_scale参数维度含义：b（batch Size）表示输入样本批量大小、q_s表示query的输入样本序列长度、k_s表示key的输入样本序列长度、q_n表示query的多头数、k_n表示key的多头数、d（head dim）表示注意力头的维度、q_t表示query的输入样本序列长度的累加和、k_t表示key的输入样本序列长度的累加和、block_num表示PageAttention场景下的block总数、block_size表示PageAttention场景下每个block的token数、g表示GQA的group size（g = q_n / k_n）。参数query中的d和参数key中的d值相等，当前仅支持128。

### quant_lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示query的head个数。 | int64 | - |
| num_heads_k | int | 必选 | 表示key的head个数，当前仅支持1。 | int64 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持128。 | int64 | - |
| topk | int | 必选 | 表示topK阶段需要保留的key token索引数量，当前仅支持[1, 8192]。 | int64 | - |
| quant_mode | int | 必选 | 表示量化模式，当前支持1/2/3/4/5。1表示qk: fp8(e4m3fn) per-token-head, scale: fp32；2表示qk: int8 per-token-head, scale: fp16, weights: fp16；3表示qk: mxfp8(e4m3fn), scale: fp8(e8m0)；4表示qk: hifloat8 per-tensor, scale: fp32；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。 | int64 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同batch中query的有效Sequence Length的累加前缀和，仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32 | (b+1, ) |
| cu_seqlens_k | Tensor | 可选 | 表示不同batch中key的有效Sequence Length的累加前缀和，仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32 | (b+1, ) |
| seqused_q | Tensor | 可选 | 表示不同batch中query实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (b, ) |
| seqused_k | Tensor | 可选 | 表示不同batch中key实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (b, ) |
| cmp_residual_k | Tensor | 可选 | 表示不同batch中key压缩前Sequence Length除以cmp_ratio的余数，配合cmp_ratio实现key部分的mask和负载计算。cmp_ratio不为1且mask_mode为3场景下必传。需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。数据格式为ND，支持非连续的Tensor。 | int32 | (b, ) |
| batch_size | int | 可选 | 表示batch数量，默认值为None。 | int64 | - |
| max_seqlen_q | int | 可选 | 表示query的最长Sequence Length，-1表示任意可能长度，默认值为None。 | int64 | - |
| max_seqlen_k | int | 可选 | 表示key的最长Sequence Length，-1表示任意可能长度，默认值为None。 | int64 | - |
| layout_q | str | 可选 | 表示query的排列格式，支持BSND、TND，默认值为None。 | string | - |
| layout_k | str | 可选 | 表示key的排列格式，支持BSND、TND、PA_BBND，默认值为None。 | string | - |
| mask_mode | int | 可选 | 表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为None。 | int64 | - |
| cmp_ratio | int | 可选 | 表示key的压缩率，取值范围[1, 128]，1表示无压缩，默认值为None。 | int64 | - |

### quant_lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | 公式中的量化输入$Q_{index}^{Quant}$。不支持空tensor。数据格式为ND。 | int8、float8_e4m3fn、HIfloat8、float4_e2m1 | layout_q为BSND时shape为(b,q_s,q_n,d)；layout_q为TND时shape为(q_t,q_n,d) |
| key | Tensor | 必选 | 公式中的量化输入$K_{index}^{Quant}$。不支持空tensor。数据格式为ND，仅PA_BBND场景下0轴支持非连续。PA_BBND场景下block_size取值为16的倍数，最大支持1024。 | int8、float8_e4m3fn、HIfloat8、float4_e2m1 | layout_k为BSND时shape为(b,k_s,k_n,d)；layout_k为TND时shape为(k_t,k_n,d)；layout_k为PA_BBND时shape为(block_num,block_size,k_n,d) |
| weights | Tensor | 必选 | 公式中的权重系数$W$。不支持空tensor。数据格式为ND。 | float16、float32 | layout_q为BSND时shape为(b,q_s,q_n)；layout_q为TND时shape为(q_t,q_n) |
| query_dequant_scale | Tensor | 必选 | 公式中的$Scale_Q$，表示Index query的反量化系数。不支持空tensor。数据格式为ND。 | float16、float32、float8_e8m0 | quant_mode为3/5时，layout_q为BSND时shape为(b,q_s,q_n,d/64,2)，layout_q为TND时shape为(q_t,q_n,d/64,2)；quant_mode为4时shape必须为(1,)；其他场景shape与weights保持一致 |
| key_dequant_scale | Tensor | 必选 | 公式中的$Scale_K$，表示Index key的反量化系数。不支持空tensor。数据格式为ND，仅PA_BBND场景下0轴支持非连续。 | float16、float32、float8_e8m0 | quant_mode为3/5时，layout_k为PA_BBND时shape为(block_num,block_size,k_n,d/64,2)，layout_k为BSND时shape为(b,k_s,k_n,d/64,2)，layout_k为TND时shape为(k_t,k_n,d/64,2)；quant_mode为4时shape必须为(1,)；其他场景分别为(block_num,block_size,k_n)、(b,k_s,k_n)或(k_t,k_n) |
| topk | int | 必选 | 表示topK阶段需要保留的key token索引数量，当前支持[1, 8192]。 | int64 | - |
| quant_mode | int | 必选 | 表示量化模式，当前支持1/2/3/4/5。1表示qk: fp8(e4m3fn) per-token-head, scale: fp32；2表示qk: int8 per-token-head, scale: fp16, weights: fp16；3表示qk: mxfp8(e4m3fn), scale: fp8(e8m0)；4表示qk: hifloat8 per-tensor, scale: fp32；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。 | int64 | - |
| cu_seqlens_q | Tensor | 可选 | 当前batch及前序batch中query的有效token数的累加和，后一个元素的值必须大于等于前一个元素的值。仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND。 | int32 | (b+1,) |
| cu_seqlens_k | Tensor | 可选 | 当前batch及前序batch中key的有效token数的累加和，后一个元素的值必须大于等于前一个元素的值。仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND。 | int32 | (b+1,) |
| seqused_q | Tensor | 可选 | 不同batch中query的真实使用长度，每个batch的有效token数不超过query中的维度S大小且不小于0。数据格式为ND。 | int32 | (b,) |
| seqused_k | Tensor | 可选 | 不同batch中key的真实使用长度，每个batch的有效token数不超过key中的维度S大小且不小于0。数据格式为ND。layout_k为PA_BBND时必须传入。 | int32 | (b,) |
| cmp_residual_k | Tensor | 可选 | 表示key压缩前token数量除以cmp_ratio的余数，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。需要在mask_mode等于3、cmp_ratio不等于1的场景下使用。数据格式为ND。 | int32 | (b,) |
| block_table | Tensor | 可选 | 表示PageAttention中KV存储使用的block映射表。不支持空tensor。layout_k为PA_BBND时必须传入。数据格式为ND。 | int32 | (b, k_s_max/block_size) |
| output_idx_offset | Tensor | 可选 | 表示topK结果输出索引所需要加上的偏移。值必须大于等于0，加上偏移后topk index不能超过int32最大值。数据格式为ND。 | int32 | layout_q为BSND时shape为(b,q_s,k_n)；layout_q为TND时shape为(q_t,k_n) |
| metadata | Tensor | 可选 | quant_lightning_indexer_metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。不支持空tensor。数据格式为ND。 | int32 | (1024,) |
| max_seqlen_q | int | 可选 | query的最大序列长度。-1表示任意可能长度，默认值为-1。 | int64 | - |
| layout_q | str | 可选 | 用于标识输入query的数据排布格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入key的数据排布格式，支持BSND、TND、PA_BBND，默认值为BSND。 | string | - |
| mask_mode | int | 可选 | 表示mask的模式，0代表defaultMask模式，3代表rightDownCausal模式，默认值为0。 | int64 | - |
| cmp_ratio | int | 可选 | 用于稀疏计算，表示key的压缩倍数。默认值为1，表示无压缩。 | int64 | - |
| return_value | int | 可选 | 代表是否需要返回sparse_indices对应的sparse_values值。0代表不返回，1代表返回值，默认值为0。 | int64 | - |

## 返回值说明

### quant_lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的batch、head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | (1024, )  |

### quant_lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | 必选 | 公式中的Indices输出。不支持空tensor。无效部分填-1。数据格式为ND。 | int32 | layout_q为BSND时shape为(b,q_s,k_n,topk)；layout_q为TND时shape为(q_t,k_n,topk) |
| sparse_values | Tensor | 条件输出 | 公式中的Indices对应的Values输出。当return_value为1时输出对应值；当return_value为0时输出shape为[0]的空tensor。无效部分填-inf。数据格式为ND。 | bfloat16 | layout_q为BSND时shape为(b,q_s,k_n,topk)；layout_q为TND时shape为(q_t,k_n,topk)；return_value为0时shape为(0,) |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir（aclgraph）图模式调用。
- quant_lightning_indexer_metadata接口需与quant_lightning_indexer算子配套使用。
- b（batch）表示输入样本批量大小。
- 参数cu_seqlens_q、cu_seqlens_k要求其值为当前batch与前序batch有效token数的累加值，第一个元素必须为0，且后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_k要求其值表示每个batch中的有效token数。
- 参数cmp_residual_k需满足0 <= cmp_residual_k\[i\] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../docs/zh/context/sparse_mode_introduction.md)。
- pa_kv_cache支持0轴非连续；pa_block_size支持[16, 1024]，且是16的倍数。
- 参数query、key的数据类型应保持一致。
- 该接口的TopK排序过程对NaN排序是未定义行为。
- 当layout_q为BSND时，不支持传入cu_seqlens_q；当layout_k为BSND或PA_BBND时，不支持传入cu_seqlens_k。
- 当传入的cmp_ratio > 1且mask_mode = 3时，必须传入cmp_residual_k，其余情况不传入。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf。
<!-- npu="A3,910b" id7 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - topk取值范围当前仅支持[1, 2048]。
  - 当layout_k为TND时，不支持传入seqused_k。
  - 不支持output_idx_offset。
  - 仅支持num_heads_q = 64、q_n = 64。
  - 仅支持quant_mode = 2。
  - query、key仅支持int8，不支持float8_e4m3fn、HIfloat8和float4_e2m1。
  - query_dequant_scale和key_dequant_scale支持float16，不支持float32和float8_e8m0。
  - weights支持float16，不支持float32。
  - layout_k仅支持PA_BBND，必须传入seqused_k。
  - cmp_ratio仅支持2的幂次方值：1/2/4/8/16/32/64/128。
  - 不支持return_value功能，不建议传入该参数。
<!-- end id7 -->
<!-- npu="950" id5 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - topk取值范围当前仅支持[1, 8192]。
  - 支持num_heads_q = 1~64、q_n = 1~64。
  - cmp_ratio支持[1, 128]。
  - 当传入output_idx_offset时，支持大于等于0的索引偏移值；且应满足约束：加上传入的索引偏移值后，得到的sparse_indices值不超过INT32的最大值。
  - 当layout_q为TND时，必须传入cu_seqlens_q，如果也传入seqused_q，应保证由seqused_q传入的各个batch的query长度不超过根据cu_seqlens_q计算出的各个batch的query序列长度。当某个batch由seqused_q传入的query序列长度seqlen1小于由cu_seqlens_q计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1的差值部分的query输出的sparse_indices和sparse_values全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。
  - 参数metadata必须传入，shape为(1024,)。
  - query、key在quant_mode为1/3时支持float8_e4m3fn，quant_mode为4时支持HIfloat8，quant_mode为5时支持float4_e2m1，quant_mode为2时支持int8。
  - query_dequant_scale和key_dequant_scale在quant_mode为3/5时支持float8_e8m0，quant_mode为1/4时支持float32，quant_mode为2时支持float16。
  - weights在quant_mode为2时支持float16，quant_mode为1/3/4/5时支持float32。
<!-- end id5 -->

### 特性参数组

|      特性参数组      |     参数字段名称     |
| :-------------------: | :-------------------: |
|      公共参数组      | query、key、weights、query_dequant_scale、key_dequant_scale、metadata、output_idx_offset、topk、quant_mode、layout_q、layout_k、sparse_indices、sparse_values |
|      Mask参数组      | mask_mode |
|   SeqLens参数组   | cu_seqlens_q、cu_seqlens_k、seqused_q、seqused_k、max_seqlen_q |
|   稀疏压缩参数组    | cmp_ratio、cmp_residual_k |
| Paged Attention参数组 | block_table |

### 基准信息说明

#### 公共参数组
- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。

- query、key、sparse_indices、sparse_values校验
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 232px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
</colgroup>
<thead>
<tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>query</td>
        <td>
            <ul>
                <li>tensor_type支持INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1fn</li>
                <li>BSND -> (b, q_s, q_n, d)</li>
                <li>TND -> (q_t, q_n, d)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>query、key的数据类型需相同</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>65536 > b > 0</li>
                <li>q_t > 0</li>
                <li>k_t > 0</li>
                <li>q_n > 0</li>
                <li>k_n = 1</li>
                <li>q_s > 0</li>
                <li>k_s > 0</li>
                <li>d = 128</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>key</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1fn</li>
                <li>BSND -> (b, k_s, k_n, d)</li>
                <li>TND -> (k_t, k_n, d)</li>
                <li>PA_BBND -> (num_blocks, block_size, k_n, d)</li>
                <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_indices</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持INT32</li>
                <li>layout_q为BSND时，sparse_indices的shape为(b, q_s, k_n, topk)</li>
                <li>layout_q为TND时，sparse_indices的shape为(q_t, k_n, topk)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_values</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持BFLOAT16</li>
                <li>layout_q为BSND时，sparse_indices的shape为(b, q_s, k_n, topk)</li>
                <li>layout_q为TND时，sparse_indices的shape为(q_t, k_n, topk)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>


layout匹配关系表：
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 247px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 119px">
<col style="width: 272px">
<col style="width: 145px">
</colgroup>
<thead>
<tr>
    <th>layout_q</th>
    <th>layout_k</th>
    <th>layout_out</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>
          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
        <td>BSND</td>
    </tr>
    <tr>
        <td>TND</td>
        <td>
          <li>TND</li>
          <li>PA_BBND</li>
        </td>
        <td>TND</td>
    </tr>
</tbody>
</table>

metadata校验
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadata</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>shape由quant_lightning_indexer_v2_metadata动态计算</li>
                    <li>当前不支持不传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与quant_lightning_indexer_v2_metadata生成的结果一致</td>
        </tr>
    </tbody>
</table>

mask_mode参数解释
<ul>
    <li>mask_mode=0，全计算模式（默认值）</li>
    <li>mask_mode=3，Causal模式</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为0、3，默认值为0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
        </tr>
    </tbody>
</table>

#### SeqLengths参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(b,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于q_s</li>
                    <li>seqused_k中的值需小于等于k_s</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">无</td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>seqused_k</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(b+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于q_t</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，必须传入</li>
                    <li>当layout_q不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_k</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(b+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于k_t</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_k为TND时，必须传入</li>
                    <li>当layout_k不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>取值应大于等于-1</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>无</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组
当block_table不为空时，开启Paged Attention
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>tensor_shape为(b, max_num_blocks_per_seq)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>PagedAttention开启情况下，必须传入seqused_k</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

## 确定性计算
默认支持确定性计算

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import quant_lightning_indexer, quant_lightning_indexer_metadata

  B = 2
  S1 = 64
  N1 = 64
  N2 = 1
  D = 128
  blockNum = 128
  blockSize = 128
  topk = 32

  # 构造输入
  query = torch.randn(B, S1, N1, D).to(torch.float8_e4m3fn).npu()
  key = torch.randn(blockNum, blockSize, N2, D).to(torch.float8_e4m3fn).npu()
  weights = torch.randn(B, S1, N1).npu()
  query_dequant_scale = torch.randn(B, S1, N1).npu()
  key_dequant_scale = torch.randn(blockNum, blockSize, N2).npu()
  block_table = torch.arange(blockNum, dtype=torch.int32).expand(B, -1).npu()
  seqused_k = torch.full((B,), blockNum * blockSize, dtype=torch.int32).npu()

  # 生成metadata
  metadata = quant_lightning_indexer_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      topk=topk,
      quant_mode=1,
      seqused_k=seqused_k,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=blockNum * blockSize,
      layout_q="BSND",
      layout_k="PA_BBND",
      mask_mode=0,
      cmp_ratio=1
  )

  # 执行quant_lightning_indexer
  sparse_indices, sparse_values = quant_lightning_indexer(
      query, key, weights, query_dequant_scale, key_dequant_scale,
      topk=topk,
      quant_mode=1,
      block_table=block_table,
      metadata=metadata,
      seqused_k=seqused_k,
      max_seqlen_q=-1,
      layout_q="BSND",
      layout_k="PA_BBND",
      mask_mode=0,
      cmp_ratio=1,
      return_value=1
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")
  print(f"sparse_values shape: {sparse_values.shape}")
  ```

- aclgraph图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import quant_lightning_indexer, quant_lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 128
  N1 = 64
  N2 = 1
  D = 128
  topk = 32

  query = torch.randn(B, S1, N1, D).to(torch.float8_e4m3fn).npu()
  key = torch.randn(B, S2, N2, D).to(torch.float8_e4m3fn).npu()
  weights = torch.randn(B, S1, N1).npu()
  query_dequant_scale = torch.randn(B, S1, N1).npu()
  key_dequant_scale = torch.randn(B, S2, N2).npu()

  class QuantLightningIndexerNetwork(torch.nn.Module):
      def __init__(self):
          super(QuantLightningIndexerNetwork, self).__init__()

      def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale):
          metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              topk=topk,
              quant_mode=1,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )

          return torch.ops.cann_ops_transformer.quant_lightning_indexer(
              query, key, weights, query_dequant_scale, key_dequant_scale,
              topk=topk,
              quant_mode=1,
              metadata=metadata,
              max_seqlen_q=-1,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1,
              return_value=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(QuantLightningIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  print(f"sparse_values shape: {sparse_values.shape}")
  ```
