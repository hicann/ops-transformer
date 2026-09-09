# sparse_lightning_indexer_kl_loss_grad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id8 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id9 -->

## 功能说明

- **接口功能**：

  - `sparse_lightning_indexer_kl_loss_grad`：计算Lightning Indexer KL Loss训练场景下的反向输出。该接口接收Lightning Indexer分支的 `q`、`k`、`w`、`sparse_indices`，以及由主Attention分支预先计算得到的目标分布 `attn_softmax_l1_norm`，输出 `dq`、`dk`、`dw` 和Lightning Indexer分支的 `softmax_out`。
  - `sparse_lightning_indexer_kl_loss_grad_metadata`：用于在主算子前生成分核metadata，输出结果需要作为 `sparse_lightning_indexer_kl_loss_grad` 的 `metadata` 输入传入。

- **计算公式**：

    对每个query token，根据 `sparse_indices` 选取top-k key后，Indexer logits可表示为：

    $$
    S_{t,:}=q_{t,:}@K_{\operatorname{topk}(t),:}^{T}
    $$

    $$
    I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
    $$

    Indexer分支softmax输出为：

    $$
    y_{t,:}=\operatorname{Softmax}(I_{t,:})
    $$

    目标分布 `p` 由 `attn_softmax_l1_norm` 输入提供，主算子将 `y` 写出到 `softmax_out`。当前KL Loss反向中用于回传的梯度为：

    $$
    p_{\mathrm{reduce}}=\operatorname{ReduceSum}(p,\operatorname{axis}=-1,\operatorname{keepdim}=true)
    $$

    $$
    dI_{t,:}=y_{t,:} * p_{\mathrm{reduce}} - p_{t,:}
    $$

    进而通过链式法则计算：

    $$
    dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
    $$

    $$
    dq_{t,:}=dS_{t,:}@K_{\operatorname{topk}(t),:}
    $$

    $$
    dK_{\operatorname{topk}(t),:}=\left(dS_{t,:}\right)^{T}@q_{t,:}
    $$

    `dk` 写回时会按 `sparse_indices` 指向的key位置执行scatter-add，无效top-k位置不参与计算。

> [!NOTE]
>
> `cmp_residual_k` 同时是 `sparse_lightning_indexer_kl_loss_grad` 和 `sparse_lightning_indexer_kl_loss_grad_metadata` 的可选输入。压缩KV且 `mask_mode=3` 时，该参数用于恢复压缩前key长度：`pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k[b]`，从而确定causal有效范围。

## 函数原型

```python
cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
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
    topk=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None
) -> Tensor
```

```python
cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
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

### sparse_lightning_indexer_kl_loss_grad

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| q | 必选输入 | Lightning Indexer分支的query输入。D：固定为128。 | `float16`、`bfloat16` | `layout_q="BSND"` 时为 `[B, S1, N1, D]`；`layout_q="TND"` 时为 `[T1, N1, D]`。 |
| k | 必选输入 | Lightning Indexer分支的key输入。N2 仅支持1。D：固定为128。 | 与 `q` 一致 | `layout_k="BSND"` 时为 `[B, S2, N2, D]`；`layout_k="TND"` 时为 `[T2, N2, D]`。 |
| w | 必选输入 | Indexer logits的head权重。 | `float32` | `layout_q="BSND"` 时为 `[B, S1, N1]`；`layout_q="TND"` 时为 `[T1, N1]`。 |
| sparse_indices | 必选输入 | 每个query对应的top-k key下标。有效位置填key下标，无效位置填 `-1`。 | `int32` | `layout_q="BSND"` 时为 `[B, S1, N2, K]`；`layout_q="TND"` 时为 `[T1, N2, K]`。 |
| attn_softmax_l1_norm | 必选输入 | 主Attention分支预先计算得到的目标分布 `p`，无效top-k位置建议置0。 | `float32` | 与 `sparse_indices` 一致。 |
| cu_seqlens_q | 可选输入 | TND场景中query的前缀和序列长度，首元素为0。 | `int32` | `[B + 1]`。 |
| cu_seqlens_k | 可选输入 | TND场景中key的前缀和序列长度，首元素为0。 | `int32` | `[B + 1]`。 |
| seqused_q | 可选输入 | 表示每个batch实际使用的query长度。 | `int32` | `[B]`。 |
| seqused_k | 可选输入 | 表示每个batch实际使用的key长度。 | `int32` | `[B]`。 |
| cmp_residual_k | 可选输入 | 压缩key场景下的残差长度。`mask_mode=3` 且 `cmp_ratio!=1` 时，用于还原压缩前key长度。 | `int32` | `[B]`。 |
| metadata | 可选输入 | `sparse_lightning_indexer_kl_loss_grad_metadata` 生成的任务切分结果，建议传入。 | `int32` | `[64]`。 |
| layout_q | 可选属性 | `q` 侧layout。 | `str` | 支持 `"BSND"`、`"TND"`，默认 `"TND"`。 |
| layout_k | 可选属性 | `k` 侧layout。 | `str` | 支持 `"BSND"`、`"TND"`，默认 `"TND"`。 |
| mask_mode | 可选属性 | sparse mask模式。 | `int` | 当前支持 `0` 和 `3`，默认 `3`。 |
| cmp_ratio | 可选属性 | key压缩比例。 | `int` | 取值范围 `[1, 128]`，默认 `1`。 |

<!-- npu="A3,910b" id5 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持`seqused_q`、`seqused_k`参数。
<!-- end id5 -->

### sparse_lightning_indexer_kl_loss_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | `q` 的head数，即主接口 `q` 的 `N1`，当前支持[1, 128]。 | int32 | - |
| num_heads_k | int | 必选 | `k` 的head数，即主接口 `k` 的 `N2`。当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | `q`、`k` 的最后一维 `D`，当前仅支持128。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | TND场景中query的前缀和序列长度，首元素为0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | 可选 | TND场景中key的前缀和序列长度，首元素为0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示每个batch实际使用的query长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| seqused_k | Tensor | 可选 | 表示每个batch实际使用的key长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| cmp_residual_k | Tensor | 可选 | 压缩key场景下的残差长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| batch_size | int | 可选 | batch大小。BSND场景建议显式传入，TND场景也可由 `cu_seqlens_q` 推导，默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 单个batch中最大的query序列长度，默认值为0。 | int32 | - |
| max_seqlen_k | int | 可选 | 单个batch中最大的压缩后key序列长度，默认值为0。 | int32 | - |
| topk | int | 可选 | top-k大小，即 `sparse_indices` 最后一维 `K`，当前支持[1, 2048]和4096、8192。 | int32 | - |
| layout_q | str | 可选 | `q` 侧layout，支持 `"BSND"`、`"TND"`，默认值为 `"BSND"` 。 | string | - |
| layout_k | str | 可选 | `k` 侧layout，支持 `"BSND"`、`"TND"`，默认值为 `"BSND"` 。 | string | - |
| mask_mode | int | 可选 | sparse mask模式。0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | key压缩比例，当前支持[1, 128]，默认值为1。 | int32 | - |

<!-- npu="950" id6 -->
- <term>Ascend 950PR/Ascend 950DT</term> ：`topk`仅支持[1, 2048]。
<!-- end id6 -->
<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持`seqused_q`、`seqused_k`、`cmp_residual_k`，`num_heads_q`仅支持8/16/32/64，`topk`仅支持512/1024/2048/4096/8192。
<!-- end id7 -->

## 返回值说明

### sparse_lightning_indexer_kl_loss_grad

- **dq**：`q` 的梯度，shape和数据类型与 `q` 一致。
- **dk**：`k` 的梯度，shape和数据类型与 `k` 一致。
- **dw**：`w` 的梯度，shape与 `w` 一致，数据类型为 `float32`。
- **softmax_out**：Lightning Indexer分支的softmax输出，shape与 `attn_softmax_l1_norm` 一致，数据类型为 `float32`。

### sparse_lightning_indexer_kl_loss_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的Batch、Head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | shape为(64, )  |

## 约束说明

- 该接口支持训练场景。
- 该接口支持单算子模式和aclgraph模式。
- `q`、`k` 的数据类型必须保持一致，支持 `float16` 和 `bfloat16`。
- `w`、`attn_softmax_l1_norm`、`dw`、`softmax_out` 的数据类型应为 `float32`。
- `sparse_indices`、`cu_seqlens_q`、`cu_seqlens_k`、`seqused_q`、`seqused_k`、`cmp_residual_k`、`metadata` 的数据类型应为 `int32`。
- `layout_q` 和 `layout_k` 支持 `BSND` 和 `TND`。
- `layout_q="TND"` 时需要传入 `cu_seqlens_q`；`layout_k="TND"` 时需要传入 `cu_seqlens_k`。
- 当前 `num_heads_k` 仅支持1。
- `sparse_indices` 中有效位置必须位于当前batch的key序列范围内，无效位置使用 `-1` 填充。
- `attn_softmax_l1_norm` 的无效top-k位置建议置0，并与 `sparse_indices` 的有效位置保持一致。
- `mask_mode` 当前支持 `0` 和 `3`。
- `cmp_ratio` 取值范围为 `[1, 128]`。
- 压缩key且 `mask_mode=3` 时，压缩前key长度通过 `compressed_k_len * cmp_ratio + cmp_residual_k[b]` 计算。CP切分场景中，每个CP shard单独调用时，`q` 传入当前shard的query长度，`k` 传入该shard对应的压缩后key前缀长度，并按该公式传入对应的 `cmp_residual_k`。

- **规格约束**：

    | 规格项 | 规格 | 规格说明 |
    | :--- | :--- | :--- |
    | N2 | 1 | 当前仅支持N2=1。 |
    | D | 128 | q和k最后一维需保持一致。 |
    | cmp_ratio | 1~128 | - |
    | mask_mode | 0、3 | - |

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - B：支持1~256。
    - S1、S2：S1支持1~8192，S2支持1~524288。
    - N1：支持8、16、32、64。
    - K：支持512、1024、2048、4096、8192。
  <!-- end id10 -->

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：

    - B：B>0。
    - S1、S2：S1>0，S2>0。
    - N1：支持1~128。
    - K：支持1~2048。
  <!-- end id11 -->

## 确定性计算

默认支持确定性计算

## 调用示例

### TND单batch，压缩key场景

```python
import torch
import torch_npu
from cann_ops_transformer.ops import sparse_lightning_indexer_kl_loss_grad
from cann_ops_transformer.ops import sparse_lightning_indexer_kl_loss_grad_metadata


def valid_k_count(local_q_idx, q_len, compressed_k_len, cmp_ratio, cmp_residual_k):
    pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k
    numerator = pre_compress_k_len - q_len + local_q_idx + 1
    return numerator // cmp_ratio if numerator >= 0 else -((-numerator) // cmp_ratio)


def make_sparse_indices_and_target(q_len, compressed_k_len, topk, cmp_ratio, cmp_residual_k):
    sparse_indices = torch.full((q_len, 1, topk), -1, dtype=torch.int32)
    attn_softmax_l1_norm = torch.zeros((q_len, 1, topk), dtype=torch.float32)

    for q_idx in range(q_len):
        real_k = valid_k_count(q_idx, q_len, compressed_k_len, cmp_ratio, cmp_residual_k)
        real_k = max(0, min(real_k, compressed_k_len, topk))
        if real_k == 0:
            continue

        sparse_indices[q_idx, 0, :real_k] = torch.randperm(real_k, dtype=torch.int32)
        attn_softmax_l1_norm[q_idx, 0, :real_k] = 1.0 / real_k

    return sparse_indices, attn_softmax_l1_norm


torch_npu.npu.set_device(0)
device = torch.device("npu:0")

q_len = 512
compressed_k_len = 128
cmp_ratio = 4
cmp_residual_k = 0
batch_size = 1
num_heads_q = 64
num_heads_k = 1
head_dim = 128
topk = 512
layout = "TND"
mask_mode = 3
dtype = torch.bfloat16

q = torch.randn(q_len, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(compressed_k_len, num_heads_k, head_dim, dtype=dtype, device=device)
w = torch.randn(q_len, num_heads_q, dtype=torch.float32, device=device) * (0.1 / 6.0)

sparse_indices_cpu, attn_l1_cpu = make_sparse_indices_and_target(
    q_len, compressed_k_len, topk, cmp_ratio, cmp_residual_k
)
sparse_indices = sparse_indices_cpu.to(device)
attn_softmax_l1_norm = attn_l1_cpu.to(device)

cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, compressed_k_len], dtype=torch.int32, device=device)
cmp_residual_k_tensor = torch.tensor([cmp_residual_k], dtype=torch.int32, device=device)

metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_k=compressed_k_len,
    topk=topk,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

dq, dk, dw, softmax_out = sparse_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
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
