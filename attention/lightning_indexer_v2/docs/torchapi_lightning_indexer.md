# lightning\_indexer

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
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `lightning_indexer_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的batch、head、以及Q和K的分块的索引，供后续`lightning_indexer`算子使用。

  `lightning_indexer`接口基于一系列操作得到每一个token对应的top-k个位置。主要计算过程为：

  1. 将某个token对应的输入参数`q`（$Q_{index}\in\R^{g\times d}$）乘以给定上下文`k`（$K_{index}\in\R^{S_{k}\times d}$），得到相关性。
  2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`w`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparse\_indices$，并输出对应的$sparse\_values$，作为Attention的输入。

  `lightning_indexer`接口同时承载两级TopK候选源（source）模式：当可选参数`candidate_topk_blocks`传入非None（含显式-1关闭）时，接口额外返回候选块索引`candidate_block_indices`（batch内相对块号，无效槽填-1，供`sparse_lightning_indexer`消费）与预留输出`candidate_block_length`（恒空），即返回4元组；未传入`candidate_topk_blocks`时保持返回2元组。`return_value`在candidate路径下已放开（值域{0,1}）：rv=1时`sparse_values`输出位置级topk真实分数（shape与`sparse_indices`相同），rv=0时恒为空tensor (0,)。candidate模式仅Atlas A2系列产品、Atlas A3系列产品支持开启，Ascend 950PR&950DT系列产品上开启会被host侧校验拒绝。

- **计算公式**：

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

## 函数原型

调用lightning_indexer接口之前，先调用前置接口lightning_indexer_metadata，完成lightning_indexer负载均衡的计算。

```python
cann_ops_transformer.lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
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
cann_ops_transformer.lightning_indexer(
  q,
  k,
  w,
  topk,
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
  return_value=0,
  candidate_topk_blocks=None,
  candidate_block_size=8
) -> (Tensor, Tensor) 或 (Tensor, Tensor, Tensor, Tensor)
```

> **说明：** `candidate_topk_blocks`为两级TopK候选源（source）模式开关：未传入（None）时返回2元组`(sparse_indices, sparse_values)`，与既有版本对外保持不变；传入（含显式-1关闭）时返回4元组`(sparse_indices, sparse_values, candidate_block_indices, candidate_block_length)`。`candidate_topk_blocks`开启时取值需合法（(0, 2048]且64的倍数），取值校验由host侧完成。`return_value`值域{0,1}：rv=1时`sparse_values`输出位置级topk真实分数（shape与`sparse_indices`相同），rv=0时恒为空tensor (0,)。`candidate_block_size`仅candidate路径下生效，[2, 64]内的2的幂，默认值为8。

## 参数说明

>**说明：**<br>
>
>- q、k、w参数维度含义：b（batch Size）表示输入样本批量大小、q_s表示q的输入样本序列长度、k_s表示k的输入样本序列长度、q_n表示q的多头数、k_n表示k的多头数、d（head dim）表示注意力头的维度、q_t表示q的输入样本序列长度的累加和、k_t表示k的输入样本序列长度的累加和。参数q中的d和参数k中的d值相等，当前仅支持128。

### lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示q的head个数。 | int64 | - |
| num_heads_k | int | 必选 | 表示k的head个数，当前仅支持1。 | int64 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持128。 | int64 | - |
| topk | int | 必选 | 表示为每个q token保留的Key token索引个数，当前支持[1, 8192]。 | int64 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同batch中q的有效Sequence Length的累加前缀和，仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 shape为(b+1, )。 | int32 | 1 |
| cu_seqlens_k | Tensor | 可选 | 表示不同batch中k的有效Sequence Length的累加前缀和，仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 shape为(b+1, )。 | int32 | 1 |
| seqused_q | Tensor | 可选 | 表示不同batch中q实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 shape为(b, )。 | int32 | 1 |
| seqused_k | Tensor | 可选 | 表示不同batch中k实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 shape为(b, )。 | int32 | 1 |
| cmp_residual_k | Tensor | 可选 | 表示不同batch中k压缩前Sequence Length除以cmp_ratio的余数，配合cmp_ratio实现k部分的mask和负载计算。cmp_ratio不为1且mask_mode为3场景下必传。数据格式为ND，支持非连续的Tensor。 shape为(b, )。 | int32 | 1 |
| batch_size | int | 可选 | 表示batch数量，默认值为None。 | int64 | - |
| max_seqlen_q | int | 可选 | 表示q的最长Sequence Length，-1表示任意可能长度，默认值为None。 | int64 | - |
| max_seqlen_k | int | 可选 | 表示k的最长Sequence Length，-1表示任意可能长度，默认值为None。 | int64 | - |
| layout_q | str | 可选 | 表示q的排列格式，支持BSND、TND，默认值为None。 | string | - |
| layout_k | str | 可选 | 表示k的排列格式，支持BSND、TND、PA_BBND，默认值为None。 | string | - |
| mask_mode | int | 可选 | 表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为None。 | int64 | - |
| cmp_ratio | int | 可选 | 表示k的压缩率，取值范围[1, 128]，1表示无压缩，默认值为None。 | int64 | - |

### lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | 必选 | 公式中的输入Q。不支持空tensor。数据格式为ND。 支持的shape为：layout_q为BSND时shape为(b,q_s,q_n,d)；layout_q为TND时shape为(q_t,q_n,d)。 | bfloat16、float16 | 3-4 |
| k | Tensor | 必选 | 公式中的输入K。不支持空tensor。数据格式为ND，支持非连续的Tensor（仅PA_BBND场景下0轴支持非连续）。 支持的shape为：layout_k为BSND时shape为(b,k_s,k_n,d)；layout_k为TND时shape为(k_t,k_n,d)；layout_k为PA_BBND时shape为(block_num,block_size,k_n,d)。 | bfloat16、float16 | 3-4 |
| w | Tensor | 必选 | 公式中的输入W。不支持空tensor。数据格式为ND。 支持的shape为：layout_q为BSND时shape为(b,q_s,q_n)；layout_q为TND时shape为(q_t,q_n)。 | float | 2-3 |
| topk | int | 必选 | topK阶段需要保留的Key token索引数量，当前支持[1, 8192]。 | int64 | - |
| cu_seqlens_q | Tensor | 可选 | layout_q为TND时为各batch有效token数的累加和（首元素为0，非递减）；layout_q为BSND时为[0, 各样本有效Q长度]，长度各自不超过q_s。仅TND场景下必传。数据格式为ND。 shape为(b+1,)。 | int32 | 1 |
| cu_seqlens_k | Tensor | 可选 | layout_k为TND时为各batch有效token数的累加和（首元素为0，非递减）；layout_k为BSND时不支持传入。仅TND场景下必传。数据格式为ND。 shape为(b+1,)。 | int32 | 1 |
| seqused_q | Tensor | 可选 | 不同batch中q的真实使用长度。数据格式为ND。 shape为(b,)。 | int32 | 1 |
| seqused_k | Tensor | 可选 | 不同batch中k的真实使用长度。数据格式为ND。 shape为(b,)。 | int32 | 1 |
| cmp_residual_k | Tensor | 可选 | 表示k压缩前token数量除以cmp_ratio的余数。需要在mask_mode等于3、cmp_ratio不等于1的场景下使用。数据格式为ND。 shape为(b,)。 | int32 | 1 |
| block_table | Tensor | 可选 | 表示PageAttention中KV存储使用的block映射表。不支持空tensor。数据格式为ND。 shape为(b, k_s_max/block_size)。 | int32 | 2 |
| output_idx_offset | Tensor | 可选 | 表示topK结果输出索引所需要加上的偏移。值必须大于等于0，加上偏移后topK index不能超过int32最大值。数据格式为ND。 支持的shape为：layout_q为BSND时shape为(b,q_s,k_n)；layout_q为TND时shape为(q_t,k_n)。 | int32 | 2-3 |
| metadata | Tensor | 可选 | 由lightning_indexer_metadata得到的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。不支持空tensor。数据格式为ND。 shape为(2048,)。 | int32 | 1 |
| max_seqlen_q | int | 可选 | q的最大序列长度。-1表示任意可能长度，默认值为-1。 | int64 | - |
| layout_q | str | 可选 | 用于标识输入q的数据排布格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入k的数据排布格式，支持BSND、TND、PA_BBND，默认值为BSND。 | string | - |
| mask_mode | int | 可选 | 表示mask的模式，0代表defaultMask模式，3代表rightDownCausal模式，默认值为0。 | int64 | - |
| cmp_ratio | int | 可选 | 用于稀疏计算，表示k的压缩倍数。支持1-128，默认值为1。 | int64 | - |
| return_value | int | 可选 | 代表是否需要返回Indices对应的Values值。0代表不返回，1代表返回值（rv=1时输出位置级topk真实分数），默认值为0。值域{0,1}。 | int64 | - |
| candidate_topk_blocks | int | 可选 | 两级TopK候选源模式的候选块数量。未传入（None）时走兼容路径返回2元组；传入-1（关闭）时返回4元组且后两个输出为空占位；开启时支持(0, 2048]且必须是64的倍数，并要求topk≤2048 | int64 | - |
| candidate_block_size | int | 可选 | 候选块大小，即压缩后K空间中每个候选块包含的位置数。支持[2, 64]内的2的幂，默认值为8；candidate关闭时同样校验。需与下游消费算子sparse_lightning_indexer的配置保持一致。 | int64 | - |

## 返回值说明

### lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的batch、head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 支持的shape为(2048,)。 | int32 | 1 |

### lightning_indexer

未传入`candidate_topk_blocks`时返回2元组`(sparse_indices, sparse_values)`：

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | 必选 | 公式中的Indices输出。不支持空tensor。无效部分填-1。数据格式为ND。 支持的shape为：layout_q为BSND时shape为(b,q_s,k_n,topk)；layout_q为TND时shape为(q_t,k_n,topk)。 | int32 | 3/4 |
| sparse_values | Tensor | 条件输出 | 公式中的Indices对应的Values输出。return_value为1时输出位置级topk真实分数（与sparse_indices按槽位一一配对）；return_value为0时输出shape为(0,)的空tensor。无效部分填-inf。数据格式为ND。 支持的shape为：layout_q为BSND且return_value为1时shape为(b,q_s,k_n,topk)；layout_q为TND且return_value为1时shape为(q_t,k_n,topk)；return_value为0时shape为(0,)。 | float | 3-4 |

传入`candidate_topk_blocks`（含显式-1关闭）时返回4元组`(sparse_indices, sparse_values, candidate_block_indices, candidate_block_length)`：

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | 必选 | 同2元组返回，candidate模式不影响其结果。 | int32 | 3-4 |
| sparse_values | Tensor | 条件输出 | return_value已放开（值域{0,1}）：rv=1时输出位置级topk真实分数（与sparse_indices按槽位一一配对，FLOAT32）；rv=0时恒为shape(0,)的空tensor。无效部分填-inf。 支持的shape为：rv=1且layout_q为BSND时shape为(b,q_s,k_n,topk)；rv=1且layout_q为TND时shape为(q_t,k_n,topk)；rv=0时shape为(0,)。 | float | 3-4 |
| candidate_block_indices | Tensor | 条件输出 | candidate_topk_blocks开启时输出候选块索引：batch内压缩K位置空间的相对块号（块j覆盖[j×candidate_block_size,(j+1)×candidate_block_size)），无效槽填-1，不叠加output_idx_offset，槽位顺序不作为消费约束；该输出不受return_value影响。 支持的shape为：开启且layout_q为BSND时shape为(b,q_s,k_n,candidate_topk_blocks)；开启且layout_q为TND时shape为(q_t,k_n,candidate_topk_blocks)；关闭时shape为(0,)。 | int32 | 3-4 |
| candidate_block_length | Tensor | 必选 | 预留接口，恒为空tensor，kernel不消费。 | int32 | 1 |

## 约束说明

- 确定性计算与batch一致性：
  - 默认确定性实现，相同输入多次调用结果一致。
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：batch一致性：通过`torch_npu.npu.set_deterministic_level(3)`开启batch一致性，开启后可以满足计算结果和所在批次大小、位置无关。
  <!-- end id9 -->
- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir（aclgraph）图模式调用。
- lightning_indexer_metadata接口需与lightning_indexer算子配套使用。
- b（batch）表示输入样本批量大小。
- 参数cu_seqlens_q在layout_q为TND时要求值为累计数组（首元素为0，非递减）；在layout_q为BSND时要求值为[0, 各样本有效Q长度]，各长度不超过q_s。参数cu_seqlens_k在layout_k为TND时要求值为累计数组（首元素为0，非递减）；在layout_k为BSND时不支持传入。
- 参数seqused_q、seqused_k要求其值表示每个batch中的有效token数。
- 参数cmp_residual_k需满足cmp_residual_k\[i\] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../docs/zh/context/sparse_mode_introduction.md)。
- pa_kv_cache支持0轴非连续；pa_block_size支持[16, 1024]，且是16的倍数。
- 参数q、k的数据类型应保持一致。
- 该接口的TopK排序过程对NaN排序是未定义行为。
- 当layout_q为BSND时，<term>Ascend 950PR&950DT系列产品</term>不支持传入cu_seqlens_q；当layout_k为BSND或PA_BBND时，<term>Ascend 950PR&950DT系列产品</term>不支持传入cu_seqlens_k。
- 当传入的cmp_ratio > 1且mask_mode = 3时，必须传入cmp_residual_k，其余情况不传入。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf；candidate_block_indices无效槽位填-1。
<!-- npu="A3,910b" id7 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>:
  - topk取值范围当前仅支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。
  - 当前lightning_indexer接口不支持seqused_q、output_idx_offset、max_seqlen_q功能，不建议传入这些参数。
  - 支持num_heads_q = 1~64、q_n = 1~64。
  - layout_k支持BSND、TND、PA_BBND；非PA_BBND场景下layout_q和layout_k必须一致。
  - layout_k为PA_BBND时必须传入seqused_k；为BSND时可选，不传使用S2，传入使用各样本有效K长度。
  - Q为BSND时，cu_seqlens_q可选；不传使用S1，传入格式为[0, 各样本有效Q长度…]，长度各自不超过S1。
  - Q为TND时，cu_seqlens_q必传，使用累计长度，相邻值之差为样本长度。
  - K为BSND时，cu_seqlens_k不传。
  - K为TND时，cu_seqlens_k必传，使用累计长度，相邻值之差为样本长度。
  - cmp_ratio支持[1, 128]。
  - 支持return_value功能，0代表不返回对应值，1代表返回FLOAT32类型的sparse_values。
  - 两级TopK候选源（source）模式仅在本系列产品上支持开启：`lightning_indexer`传入`candidate_topk_blocks`非-1即开启（返回4元组）；开启时topk仅支持[1, 2048]；return_value已放开（值域{0,1}，rv=1时sparse_values输出位置级topk真实分数）；未传入`candidate_topk_blocks`时行为与既有版本完全一致（返回2元组）。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>:
  - topk取值范围当前仅支持[1, 8192]。
  - 支持num_heads_q = 1~64、q_n = 1~64。
  - cmp_ratio支持[1, 128]。
  - 当传入output_idx_offset时，支持大于等于0的索引偏移值；且应满足约束：加上传入的索引偏移值后，得到的sparse_indices值不超过INT32的最大值。
  - 当layout_q为TND时，必须传入cu_seqlens_q，如果也传入seqused_q，应保证由seqused_q传入的各个batch的q长度不超过根据cu_seqlens_q计算出的各个batch的q序列长度。当某个batch由seqused_q传入的q序列长度seqlen1小于由cu_seqlens_q计算出的q长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1的差值部分的q输出的sparse_indices和sparse_values全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。
  - 参数metadata必须传入，shape为(2048,)。
  - 两级TopK候选源（source）模式在本系列产品上不支持开启：`lightning_indexer`传入`candidate_topk_blocks`非-1会被host侧校验拒绝报错。
<!-- end id8 -->

### 特性参数组

|      特性参数组      |     参数字段名称     |
| :-------------------: | :-------------------: |
|      公共参数组      | q、k、w、metadata、output_idx_offset、topk、layout_q、layout_k、sparse_indices、sparse_values |
|      Mask参数组      | mask_mode |
|   SeqLens参数组   | cu_seqlens_q、cu_seqlens_k、seqused_q、seqused_k、max_seqlen_q |
|   稀疏压缩参数组    | cmp_ratio、cmp_residual_k |
| Paged Attention参数组 | block_table |
|   Candidate参数组    | candidate_topk_blocks、candidate_block_size、candidate_block_indices、candidate_block_length |

### 基准信息说明

#### 公共参数组

- 入参为空的场景处理：

    - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。

- q、k、sparse_indices、sparse_values校验

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
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>BSND -> (b, q_s, q_n, d)</li>
                <li>TND -> (q_t, q_n, d)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、k的数据类型需相同</li>
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
        <td>k</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
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
                <li>tensor_type支持FLOAT32</li>
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
                    <li>shape由lightning_indexer_v2_metadata动态计算，固定为(2048,)</li>
                    <li><term>Ascend 950PR&950DT系列产品</term>：必须传入，未传入将被host侧校验拦截</li>
                    <li><term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：可选传入，kernel不消费，传None即可（如传入需与lightning_indexer_v2_metadata生成的结果一致）</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与lightning_indexer_v2_metadata生成的结果一致</td>
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
                    <li>暂不生效，仅支持-1</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>暂不生效，仅支持传入-1</li>
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
  import numpy as np
  from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 32
  cmp_ratio = 4

  # 计算压缩后S2长度及余数
  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  # 构造输入
  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  # 生成metadata
  metadata = lightning_indexer_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      topk=topk,
      cmp_residual_k=cmp_residual_k,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2_compressed,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=3,
      cmp_ratio=cmp_ratio
  )

  # 执行lightning_indexer
  sparse_indices, sparse_values = lightning_indexer(
      q, k, w, topk,
      cmp_residual_k=cmp_residual_k,
      metadata=metadata,
      max_seqlen_q=-1,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=3,
      cmp_ratio=cmp_ratio,
      return_value=0
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```

- 两级TopK候选源（candidate source）模式调用（仅Atlas A2系列产品、Atlas A3系列产品支持开启，传入`candidate_topk_blocks`即可在同一`lightning_indexer`接口上启用）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import lightning_indexer

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 128
  cmp_ratio = 4
  candidate_topk_blocks = 2048  # (0, 2048]且64的倍数；不传该参数则返回2元组（兼容），传-1关闭candidate
  candidate_block_size = 8      # [2, 64]内的2的幂

  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  # 传入candidate_topk_blocks返回4元组：sparse_indices, sparse_values(rv=0恒空(0,)；rv=1为位置级topk真实分), candidate_block_indices, candidate_block_length(恒空)
  sparse_indices, sparse_values, candidate_block_indices, candidate_block_length = \
      lightning_indexer(
          q, k, w, topk,
          cmp_residual_k=cmp_residual_k,
          layout_q="BSND",
          layout_k="BSND",
          mask_mode=3,
          cmp_ratio=cmp_ratio,
          return_value=0,
          candidate_topk_blocks=candidate_topk_blocks,
          candidate_block_size=candidate_block_size
      )
  print(f"candidate_block_indices shape: {candidate_block_indices.shape}")  # (B, S1, N2, candidate_topk_blocks)
  ```

- TorchAir（aclgraph）图模式调用：

  ```python
  import torch
  import torch_npu
  import numpy as np
  import torchair
  from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 32
  cmp_ratio = 4

  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  class LightningIndexerNetwork(torch.nn.Module):
      def __init__(self):
          super(LightningIndexerNetwork, self).__init__()

      def forward(self, q, k, w, cmp_residual_k):
          metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              topk=topk,
              cmp_residual_k=cmp_residual_k,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2_compressed,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=3,
              cmp_ratio=cmp_ratio
          )

          return torch.ops.cann_ops_transformer.lightning_indexer(
              q, k, w, topk,
              cmp_residual_k=cmp_residual_k,
              metadata=metadata,
              max_seqlen_q=-1,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=3,
              cmp_ratio=cmp_ratio,
              return_value=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(LightningIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(q, k, w, cmp_residual_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```
