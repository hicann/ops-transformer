# dense_lightning_indexer_kl_loss_grad / dense_lightning_indexer_kl_loss_grad_metadata

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

    * `dense_lightning_indexer_kl_loss_grad`：计算 Lightning Indexer KL Loss 训练场景下的反向输出。该接口接收 Lightning Indexer 分支的 `q`、`k`、`w`，以及由主 Attention 分支预先计算得到的目标分布 `attn_softmax_l1_norm`和`softmax_lse`，输出 `dq`、`dk`、`dw` 和 Lightning Indexer 分支的 `softmax_out`。
    * `dense_lightning_indexer_kl_loss_grad_metadata`：用于在主算子前生成分核 metadata，输出结果需要作为 `dense_lightning_indexer_kl_loss_grad` 的 `metadata` 输入传入。

- 计算公式：

    Indexer logits可表示为：

   $$
   S_{t,:}=q_{t,:}@K^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   其中，$q$和$K$分别对应本接口的q和k，$W$对应本接口的w。Indexer分支的softmax输出为：

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   本接口将$y$写出到softmaxOut。目标分布$p$由attnSoftmaxL1Norm输入提供，等价于旧版kernel内部由main attention score经head求和和L1归一化得到的结果。若后续继续计算KL Loss，其形式与旧版保持一致：

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   通过求导可得Loss的梯度表达式：

   $$
   dI_{t,:}=\operatorname{Softmax}(I_{t,:})-p_{t,:}
   $$

   利用链式法则可以进行w、q和k矩阵的梯度计算：

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K
   $$

   $$
   dK=\sum_t\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK直接累积到对应key位置上，无需scatter-add。

> [!NOTE]
>
> `cmp_residual_k` 同时是 `dense_lightning_indexer_kl_loss_grad` 和 `dense_lightning_indexer_kl_loss_grad_metadata` 的可选输入。压缩 KV 且 `mask_mode=3` 时，该参数用于恢复压缩前 key 长度：`pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k[b]`，从而确定 causal 有效范围。

## 函数原型

```python
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad_metadata

dense_lightning_indexer_kl_loss_grad_metadata(
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
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad

dense_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    attn_softmax_l1_norm,
    softmax_lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="TND",
    layout_k="TND",
    mask_mode=3,
    cmp_ratio=1
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

### dense_lightning_indexer_kl_loss_grad

| 参数名 | 参数类型 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选输入 | Lightning Indexer 分支的 query 输入。D：固定为 128。 | `float16`、`bfloat16` | `layout_q="BSND"` 时为 `[B, S1, N1, D]`；`layout_q="TND"` 时为 `[T1, N1, D]`。 |
| k | Tensor | 必选输入 | Lightning Indexer 分支的 key 输入。N2 仅支持 1。D：固定为 128。 | 与 `q` 一致 | `layout_k="BSND"` 时为 `[B, S2, N2, D]`；`layout_k="TND"` 时为 `[T2, N2, D]`。 |
| w | Tensor | 必选输入 | Indexer logits 的 head 权重。 | `float32` | `layout_q="BSND"` 时为 `[B, S1, N1]`；`layout_q="TND"` 时为 `[T1, N1]`。 |
| attn_softmax_l1_norm | Tensor | 必选输入 | 主 Attention 分支预先计算得到的目标分布 `p`。 | `float32` | `layout_k="BSND"` 时为 `[B, S1, N2, S2]`；`layout_k="TND"` 时为 `[T1, N2, MAX_SEQ_K]`。 |
| softmax_lse | Tensor | 必选输入 | 主 Attention 计算的中间输出。 | `float32` | `layout_q="BSND"` 时为 `[B, N2, S1]`；`layout_q="TND"` 时为 `[N2, T1]`。 |
| cu_seqlens_q | Tensor | 可选输入 | TND 场景中 query 的前缀和序列长度，首元素为 0。 | `int32` | `[B + 1]`。 |
| cu_seqlens_k | Tensor | 可选输入 | TND 场景中 key 的前缀和序列长度，首元素为 0。 | `int32` | `[B + 1]`。 |
| seqused_q | Tensor | 可选输入 | 表示每个 batch 实际使用的 query 长度。 | `int32` | `[B]`。 |
| seqused_k | Tensor | 可选输入 | 表示每个 batch 实际使用的 key 长度。 | `int32` | `[B]`。 |
| cmp_residual_k | Tensor | 可选输入 | 压缩 key 场景下的残差长度。`mask_mode=3` 且 `cmp_ratio!=1` 时，用于还原压缩前 key 长度。 | `int32` | `[B]`。 |
| metadata | Tensor | 可选输入 | `dense_lightning_indexer_kl_loss_grad_metadata` 生成的任务切分结果，建议传入。 | `int32` | `[64]`。 |
| layout_q | str | 可选属性 | `q` 侧 layout。 | `str` | 支持 `"BSND"`、`"TND"`，默认 `"BSND"`。 |
| layout_k | str | 可选属性 | `k` 侧 layout。 | `str` | 支持 `"BSND"`、`"TND"`，默认 `"BSND"`。 |
| mask_mode | int | 可选属性 | mask 模式。 | `int` | 当前支持 `0` 和 `3`，默认 `0`。 |
| cmp_ratio | int | 可选属性 | key 压缩比例。 | `int` | 取值范围 `[1, 128]`，默认 `1`。 |



### dense_lightning_indexer_kl_loss_grad_metadata

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
| mask_mode | int | 可选 | mask 模式。0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | key 压缩比例，当前支持[1, 128]，默认值为 1。 | int32 | - |


## 返回值说明

### dense_lightning_indexer_kl_loss_grad

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| dq | Tensor | 必选 | `q` 的梯度。 | 与 `q` 一致 | 与 `q` 一致 |
| dk | Tensor | 必选 | `k` 的梯度。 | 与 `k` 一致 | 与 `k` 一致 |
| dw | Tensor | 必选 | `w` 的梯度。 | float32 | 与 `w` 一致 |
| softmax_out | Tensor | 必选 | Lightning Indexer 分支的 softmax 输出。 | float32 | 与 `attn_softmax_l1_norm` 一致 |

### dense_lightning_indexer_kl_loss_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的Batch、Head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | shape为(64, )  |

## 约束说明

- 该接口支持训练场景。
- 该接口支持单算子模式和aclgraph模式。
- `q`、`k` 的数据类型必须保持一致，支持 `float16` 和 `bfloat16`。
- `w`、`attn_softmax_l1_norm`、`dw`、`softmax_out` 的数据类型应为 `float32`。
- `cu_seqlens_q`、`cu_seqlens_k`、`seqused_q`、`seqused_k`、`cmp_residual_k`、`metadata` 的数据类型应为 `int32`。
- `layout_q` 和 `layout_k` 支持 `BSND` 和 `TND`。
- `layout_q="TND"` 时需要传入 `cu_seqlens_q`；`layout_k="TND"` 时需要传入 `cu_seqlens_k`。
- 当前 `num_heads_k` 仅支持 1。
- `attn_softmax_l1_norm` 在TND场景下最后一维是max_seqlen_k，所有batch中最长的seqlens_k。
- `mask_mode` 当前支持 `0` 和 `3`。
- `cmp_ratio` 取值范围为 `[1, 128]`。
- 压缩 key 且 `mask_mode=3` 时，压缩前 key 长度通过 `compressed_k_len * cmp_ratio + cmp_residual_k[b]` 计算。CP 切分场景中，每个 CP shard 单独调用时，`q` 传入当前 shard 的 query 长度，`k` 传入该 shard 对应的压缩后 key 前缀长度，并按该公式传入对应的 `cmp_residual_k`。
- **规格约束**：

    | 规格项 | 规格 | 规格说明 |
    | :--- | :--- | :--- |
    | B | - | 支持泛化 |
    | S1 | - | 支持泛化 |
    | S2 | - | 支持泛化 |
    | N1 | 1~128 | - |
    | N2 | 1 | 当前仅支持N2=1。 |
    | D | 128 | q和k最后一维需保持一致。 |
    | cmp_ratio | 1~128 | - |
    | mask_mode | 0、3 | - |

## 确定性计算

- 默认不支持确定性计算，可通过PyTorch开关（torch.use_deterministic_algorithms）支持。

## 调用示例

- 单算子模式调用：
```python
import torch
import torch_npu
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad_metadata

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

q_len = 32
compressed_k_len = 128
cmp_ratio = 4
cmp_residual_k = 0
batch_size = 1
num_heads_q = 64
num_heads_k = 1
head_dim = 128
layout = "TND"
mask_mode = 3
dtype = torch.bfloat16

q = torch.randn(q_len, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(compressed_k_len, num_heads_k, head_dim, dtype=dtype, device=device)
w = torch.randn(q_len, num_heads_q, dtype=torch.float32, device=device) * (0.1 / 6.0)
softmax_lse = torch.randn((num_heads_k, q_len), dtype=torch.float32, device=device)
attn_l1_cpu = torch.zeros((q_len, 1, compressed_k_len), dtype=torch.float32)

attn_softmax_l1_norm = attn_l1_cpu.to(device)

cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, compressed_k_len], dtype=torch.int32, device=device)
cmp_residual_k_tensor = torch.tensor([cmp_residual_k], dtype=torch.int32, device=device)

metadata = dense_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_k=compressed_k_len,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

dq, dk, dw, softmax_out = dense_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    attn_softmax_l1_norm,
    softmax_lse,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    metadata=metadata,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

torch.npu.synchronize()
print(dq.shape, dk.shape, dw.shape, softmax_out.shape)
```
