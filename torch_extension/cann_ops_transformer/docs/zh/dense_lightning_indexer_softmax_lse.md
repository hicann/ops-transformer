# dense_lightning_indexer_softmax_lse / dense_lightning_indexer_softmax_lse_metadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- API功能：

    * `dense_lightning_indexer_softmax_lse`：计算 Dense 打分的 Lightning Indexer 对应的 Softmax Log-Sum-Exp（LSE）值。主要计算过程为：

        1. 将输入的 Query 张量`q`与 Key 张量`k`进行矩阵乘法，得到每个 Query Token 与所有 Key Token 之间的相关性分数矩阵。
        2. 通过 ReLU 激活函数过滤负相关信号后，将结果与加权系数`w`逐元素相乘。
        3. 沿 Query Head 维度（N1）求和，合并各 Head 的加权分数。
        4. 在 Key 序列维度（S2）上进行 Softmax Log-Sum-Exp 运算，输出每个 Query Token 的 LSE 值。

        该算子常与 `lightning_indexer` / `quant_lightning_indexer` 的 Dense 打分模式配合使用，用于后续的 Softmax 归一化计算。

    * `dense_lightning_indexer_softmax_lse_metadata`：用于在主算子前生成分核 metadata 信息，输出结果需要作为 `dense_lightning_indexer_softmax_lse` 的 `metadata` 输入传入。

- 计算公式：

    1. 计算相关性分数矩阵：

    $$
    Score = Q_{index} @ K_{index}^T
    $$

    2. ReLU 激活与加权：

    $$
    Weighted = ReLU(Score) \odot W
    $$

    3. N1 维度求和：

    $$
    X = \sum_{N1} Weighted
    $$

    4. S2 维度的 Softmax LSE：

    $$
    LSE = \max_{S2}(X) + \log\left(\sum_{S2}\exp(X - \max_{S2}(X))\right)
    $$

    其中，$Q_{index} \in \mathbb{R}^{B \times S1 \times N1 \times D}$，$K_{index} \in \mathbb{R}^{B \times S2 \times N2 \times D}$，$W \in \mathbb{R}^{B \times S1 \times N1}$，输出 $LSE \in \mathbb{R}^{B \times N2 \times S1}$。

> [!NOTE]
>
> `cmp_residual_k` 同时是 `dense_lightning_indexer_softmax_lse` 和 `dense_lightning_indexer_softmax_lse_metadata` 的可选输入。压缩 KV 且 `mask_mode=3` 时，该参数用于恢复压缩前 key 长度：`pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k[b]`，从而确定 causal 有效范围。

## 函数原型

```python
from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
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
from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse

dense_lightning_indexer_softmax_lse(
    q,
    k,
    w,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1
) -> Tensor
```

## 参数说明

>**说明：**<br>
>
>- q、k、w 参数维度含义：B（Batch Size）表示输入样本批量大小、S1 表示 q 的输入样本序列长度、S2 表示 k 的输入样本序列长度、N1 表示 q 的多头数、N2 表示 k 的多头数、D（Head Dim）表示注意力头的维度。参数 q 中的 D 和参数 k 中的 D 值相等，当前仅支持 128。

### dense_lightning_indexer_softmax_lse

| 参数名 | 参数类型 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选输入 | 公式中的输入 $Q_{index}$。支持空 tensor。数据格式为 ND。 | `float16`、`bfloat16` | `layout_q="BSND"` 时为 `[B, S1, N1, D]`；`layout_q="TND"` 时为 `[T1, N1, D]`。 |
| k | Tensor | 必选输入 | 公式中的输入 $K_{index}$。支持空 tensor。数据格式为 ND。N2 仅支持 1。D：固定为 128。 | 与 `q` 一致 | `layout_k="BSND"` 时为 `[B, S2, N2, D]`；`layout_k="TND"` 时为 `[T2, N2, D]`。 |
| w | Tensor | 必选输入 | 公式中的加权系数 $W$。支持空 tensor。数据格式为 ND。 | `float32` | `layout_q="BSND"` 时为 `[B, S1, N1]`；`layout_q="TND"` 时为 `[T1, N1]`。 |
| cu_seqlens_q | Tensor | 可选输入 | 当前 Batch 及前序 Batch 中 q 的有效 token 数的累加和。仅 `layout_q="TND"` 场景下必传，第一个值固定为 0。数据格式为 ND。 | `int32` | `[B+1]`。 |
| cu_seqlens_k | Tensor | 可选输入 | 当前 Batch 及前序 Batch 中 k 的有效 token 数的累加和。仅 `layout_k="TND"` 场景下必传，第一个值固定为 0。数据格式为 ND。 | `int32` | `[B+1]`。 |
| seqused_q | Tensor | 可选输入 | 不同 Batch 中 q 的实际使用长度。数据格式为 ND。 | `int32` | `[B]`。 |
| seqused_k | Tensor | 可选输入 | 不同 Batch 中 k 的实际使用长度。数据格式为 ND。 | `int32` | `[B]`。 |
| cmp_residual_k | Tensor | 可选输入 | 表示 k 压缩前 token 数量除以 cmp_ratio 的余数。需在 `mask_mode=3` 且 `cmp_ratio≠1` 时传入。数据格式为 ND。 | `int32` | `[B]`。 |
| metadata | Tensor | 可选输入 | `dense_lightning_indexer_softmax_lse_metadata` 生成的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。数据格式为 ND。 | `int32` | `[1024]`。 |
| layout_q | str | 可选属性 | 用于标识输入 q 的数据排布格式，支持 `BSND`、`TND`，默认值为 `BSND`。 | `str` | - |
| layout_k | str | 可选属性 | 用于标识输入 k 的数据排布格式，支持 `BSND`、`TND`，默认值为 `BSND`。`layout_q` 与 `layout_k` 必须一致。 | `str` | - |
| mask_mode | int | 可选属性 | 表示 mask 的模式，`0` 代表 defaultMask 模式，`3` 代表 rightDownCausal 模式，默认值为 `0`。 | `int` | - |
| cmp_ratio | int | 可选属性 | 表示 k 的压缩倍数。取值范围 `[1, 128]`，默认值为 `1`，表示无压缩。 | `int` | - |

### dense_lightning_indexer_softmax_lse_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | `q` 的 head 数，即主接口 `q` 的 `N1`，当前支持[1, 128]。 | int32 | - |
| num_heads_k | int | 必选 | `k` 的 head 数，即主接口 `k` 的 `N2`。当前仅支持 1。 | int32 | - |
| head_dim | int | 必选 | `q`、`k` 的最后一维 `D`，当前仅支持128。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | TND 场景中 query 的前缀和序列长度，首元素为 0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | 可选 | TND 场景中 key 的前缀和序列长度，首元素为 0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示每个 batch 实际使用的 query 长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| seqused_k | Tensor | 可选 | 表示每个 batch 实际使用的 key 长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| cmp_residual_k | Tensor | 可选 | 压缩 key 场景下的残差长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| batch_size | int | 可选 | batch 大小。BSND 场景建议显式传入，TND 场景也可由 `cu_seqlens_q` 推导，默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 单个 batch 中最大的 query 序列长度，默认值为0。 | int32 | - |
| max_seqlen_k | int | 可选 | 单个 batch 中最大的压缩后 key 序列长度，默认值为0。 | int32 | - |
| layout_q | str | 可选 | `q` 侧 layout，支持 `"BSND"`、`"TND"`，默认值为 `"BSND"` 。 | string | - |
| layout_k | str | 可选 | `k` 侧 layout，支持 `"BSND"`、`"TND"`，默认值为 `"BSND"` 。 | string | - |
| mask_mode | int | 可选 | sparse mask 模式。0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | key 压缩比例，当前支持[1, 128]，默认值为 1。 | int32 | - |

## 返回值说明

### dense_lightning_indexer_softmax_lse

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| softmax_lse | Tensor | 必选 | Softmax Log-Sum-Exp 计算结果。 | float32 | `layout_q="BSND"` 时为 `[B, N2, S1]`；`layout_q="TND"` 时为 `[N2, T1]`。 |

### dense_lightning_indexer_softmax_lse_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的Batch、Head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | shape为(1024, )  |

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持单算子模式和 aclgraph 图模式调用。
- 所有输入 Tensor 默认支持空 tensor。
- `q`、`k` 的数据类型必须保持一致，支持 `float16` 和 `bfloat16`。
- `w`、`softmax_lse` 的数据类型应为 `float32`。
- B（Batch）表示输入样本批量大小，支持泛化。
- S1、S2 表示序列长度，支持泛化。对于超长序列场景，如果计算量过大会导致 NPU 内存超限。
- 参数 q 的 N1 支持 1~128，k 的 N2 固定为 1。
- 参数 D 固定为 128。
- `layout_q` 和 `layout_k` 支持 `BSND` 和 `TND`，且二者必须一致。
- `layout_q="TND"` 时需要传入 `cu_seqlens_q`；`layout_k="TND"` 时需要传入 `cu_seqlens_k`。
- `mask_mode` 当前支持 `0` 和 `3`。`mask_mode` 的具体含义参见[sparse_mode 参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。
- `cmp_ratio` 取值范围为 `[1, 128]`。
- 参数 `cu_seqlens_q`、`cu_seqlens_k` 要求其值为当前 Batch 与前序 Batch 有效 token 数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数 `seqused_q`、`seqused_k` 要求其值不大于各 Batch 的实际序列长度。
  - BSND 场景下：seqused_q ≤ S1，seqused_k ≤ S2。
  - TND 场景下：seqused_q ≤ cu_seqlens_q[i+1] - cu_seqlens_q[i]，seqused_k ≤ cu_seqlens_k[i+1] - cu_seqlens_k[i]。
- 参数 `cmp_residual_k` 需满足 cmp_residual_k[i] < cmp_ratio。
- 当 cmp_ratio > 1 且 mask_mode = 3 时，必须传入 cmp_residual_k；其余情况不需要传入。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 当前不支持 seqused_q、seqused_k、metadata 功能，不建议传入这些参数。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 当 layout_q 为 BSND 时，不支持传入 cu_seqlens_q；当 layout_k 为 BSND 时，不支持传入 cu_seqlens_k。
  - 当 layout_q 为 TND 时，必须传入 cu_seqlens_q；如果同时传入 seqused_q，应保证由 seqused_q 传入的各 batch query 长度不超过根据 cu_seqlens_q 计算出的各 batch query 长度。当某个 batch 的 seqused_q 小于实际长度时，启用 TND Padding 功能，该 batch 超出部分的输出填充无效值。
  - 当 cmp_ratio > 1 且 mask_mode = 3 时，必须传入 cmp_residual_k。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import numpy as np
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  B = 2
  S1 = 4096
  S2 = 1024
  N1 = 32
  N2 = 1
  D = 128

  torch.npu.set_device(0)
  device = torch.device("npu:0")

  # 构造输入
  q = torch.randn(B, S1, N1, D, dtype=torch.float16, device=device)
  k = torch.randn(B, S2, N2, D, dtype=torch.float16, device=device)
  w = torch.randn(B, S1, N1, dtype=torch.float32, device=device)

  # 生成 metadata
  metadata = dense_lightning_indexer_softmax_lse_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )

  # 执行算子
  lse = dense_lightning_indexer_softmax_lse(
      q,
      k,
      w,
      metadata=metadata,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )
  print(f"lse shape: {lse.shape}")  # (B, 1, S1)
  ```

- aclgraph 图模式调用：

  ```python
  import torch
  import torch_npu
  import numpy as np
  import torchair
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  B = 2
  S1 = 4096
  S2 = 1024
  N1 = 32
  N2 = 1
  D = 128

  torch.npu.set_device(0)
  device = torch.device("npu:0")

  q = torch.randn(B, S1, N1, D, dtype=torch.float16, device=device)
  k = torch.randn(B, S2, N2, D, dtype=torch.float16, device=device)
  w = torch.randn(B, S1, N1, dtype=torch.float32, device=device)

  class DenseLightningIndexerLSENetwork(torch.nn.Module):
      def __init__(self):
          super(DenseLightningIndexerLSENetwork, self).__init__()

      def forward(self, q, k, w):
          metadata = torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )
          return torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse(
              q,
              k,
              w,
              metadata=metadata,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(
      DenseLightningIndexerLSENetwork(), fullgraph=True, backend=npu_backend, dynamic=False
  )
  lse = npu_mode(q, k, w)
  print(f"lse shape: {lse.shape}")
  ```