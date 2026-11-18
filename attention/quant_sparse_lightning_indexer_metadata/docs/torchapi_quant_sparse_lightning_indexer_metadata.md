# quant_sparse_lightning_indexer_metadata

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

- 接口功能：本接口为[quant_sparse_lightning_indexer](../../quant_sparse_lightning_indexer/docs/torchapi_quant_sparse_lightning_indexer.md)生成负载均衡所需的分核信息，不执行候选范围内的打分和TopK计算。
- 调用流程：根据序列信息、候选有效长度及计算属性生成metadata，再将其传入quant_sparse_lightning_indexer。候选有效长度的传入规则见[约束说明](#约束说明)。

## 函数原型

```python
cann_ops_transformer.quant_sparse_lightning_indexer_metadata(
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
    cmp_ratio=None,
    candidate_block_length=None,
    candidate_block_size=8,
) -> Tensor
```

## 参数说明

b表示batch数量，q_s表示BSND布局下Query的物理序列长度，q_t表示TND布局下Query的token总数。参数须与后续计算接口保持一致。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示query的head个数。 | int64 | - |
| num_heads_k | int | 必选 | 表示key的head个数，当前仅支持1。 | int64 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持128。 | int64 | - |
| topk | int | 必选 | 表示topK阶段需要保留的key token索引数量，当前仅支持[1, 8192]。 | int64 | - |
| quant_mode | int | 必选 | 表示量化模式，当前支持2/5。2表示qk: int8 per-token-head, scale: fp16, weights: fp16；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。 | int64 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同batch中query的有效Sequence Length的累加前缀和，仅layout_q为TND场景下必传，第一个值固定为0；layout_q为BSND时不传入。默认值为None。数据格式为ND。 | int32 | (b+1,) |
| cu_seqlens_k | Tensor | 可选 | 表示不同batch中key的有效Sequence Length的累加前缀和，仅layout_k为TND场景下必传，第一个值固定为0；其余布局不传入。默认值为None。数据格式为ND。 | int32 | (b+1,) |
| seqused_q | Tensor | 可选 | 表示不同batch中query实际参与运算的Sequence Length。默认值为None，未传入时按query布局和序列信息确定有效长度。数据格式为ND。 | int32 | (b,) |
| seqused_k | Tensor | 可选 | 表示不同batch中key实际参与运算的Sequence Length。默认值为None。数据格式为ND，分页场景的传入规则见约束说明。 | int32 | (b,) |
| cmp_residual_k | Tensor | 可选 | 表示不同batch中key压缩前Sequence Length除以cmp_ratio的余数，配合cmp_ratio实现key部分的mask和负载计算。cmp_ratio不为1且mask_mode为3场景下必传。需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。数据格式为ND，默认值为None。 | int32 | (b,) |
| batch_size | int | 可选 | 表示batch数量，默认值为None。推导规则见约束说明。 | int64 | - |
| max_seqlen_q | int | 可选 | 表示query的最长Sequence Length，-1表示任意可能长度，默认值为None。未传入时按-1处理。 | int64 | - |
| max_seqlen_k | int | 可选 | 表示key的最长Sequence Length，-1表示任意可能长度，默认值为None。未传入时按-1处理。 | int64 | - |
| layout_q | str | 可选 | 表示query的排列格式，支持BSND、TND，默认值为None。未传入时按BSND处理。 | string | - |
| layout_k | str | 可选 | 表示key的排列格式，支持BSND、TND、PA_BBND，默认值为None。未传入时按BSND处理，非分页场景须与layout_q相同。 | string | - |
| mask_mode | int | 可选 | 表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为None。未传入时按0处理。 | int64 | - |
| cmp_ratio | int | 可选 | 表示key的压缩率，取值范围[1, 128]，1表示无压缩，默认值为None。未传入时按1处理。 | int64 | - |
| candidate_block_length | Tensor | 可选 | 每行实际有效候选块数，取值范围为[0,2048]，须与后续计算使用的候选索引逐行对应。默认值为None。数据格式为ND，传入规则见约束说明。 | int32 | TND为(q_t,1)，BSND为(b,q_s,1)，或空Tensor |
| candidate_block_size | int | 可选 | 每个候选块包含的逻辑Key位置数，必须与第一级算子及后续计算使用相同配置。默认值为8，当前仅支持8。 | int64 | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AI Core计算任务的分核信息，供后续quant_sparse_lightning_indexer使用。数据格式为ND，使用连续存储。 | int32 | (1024,) |

## 约束说明

- 该接口支持推理场景下的单算子模式调用。
- 本接口需与[quant_sparse_lightning_indexer](../../quant_sparse_lightning_indexer/docs/torchapi_quant_sparse_lightning_indexer.md)配套使用，生成的metadata不能与第一级算子的metadata混用。
- `num_heads_q`、`num_heads_k`、`head_dim`、`topk`、`quant_mode`、布局、`mask_mode`、`cmp_ratio`、`candidate_block_size`及序列信息必须与计算接口一致；相关信息变化后需重新生成metadata。
- `mask_mode`为3且`cmp_ratio`不为1时，必须传入`cmp_residual_k`。

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - `quant_mode`仅支持5，`num_heads_q`仅支持[1,32]。
  - `candidate_block_length`必须传入，不能以候选容量代替实际有效候选数；候选有效长度变化后需重新生成metadata。
  - 所有输入Tensor必须位于同一NPU设备并连续，输出metadata与输入位于同一设备。
  - TND/TND布局下，`batch_size`必须保持None，由Query和Key的前缀和推导batch数量；BSND布局下由`candidate_block_length`的shape推导，显式传入时须与其一致；TND/PA_BBND布局下由Query的前缀和推导。
  - `layout_k`为BSND时，`max_seqlen_k`必须与Key的物理序列长度一致。
  - `layout_k`为PA_BBND时，`seqused_k`与`max_seqlen_k`至少传入一个；未传入`seqused_k`时，`max_seqlen_k`必须与计算接口的逻辑Key容量一致。
  - 仅在`mask_mode`为3且`cmp_ratio`不为1时传入`cmp_residual_k`，其余情况不传入。
  - 支持aclgraph图模式调用及Meta/Fake模式下的输出shape和数据类型推导。
<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - `quant_mode`仅支持2，`num_heads_q`仅支持32、64，`topk`仅支持[1,2048]。
  - `layout_k`仅支持PA_BBND，`seqused_k`必须传入。
  - `candidate_block_length`为预留参数，应保持默认值None或传入空Tensor，不支持非空Tensor。
  - `cmp_ratio`仅支持1、2、4、8、16、32、64、128。
  - `batch_size`为None时按0处理。`layout_q`为BSND时，优先通过`seqused_q`的shape推导batch数量，未传入时使用`batch_size`；`layout_q`为TND时，优先通过`seqused_q`的shape推导，未传入时使用`cu_seqlens_q`的shape。
  - `layout_q`为BSND时，`max_seqlen_q`须与Query的物理序列长度一致。
  - 图模式暂不支持。
<!-- end id8 -->

## 确定性计算

默认支持确定性计算。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer as cot

torch.npu.set_device(0)
device = "npu:0"
batch, key_length = 2, 2048
# 每个batch包含一个Query，每行有256个有效候选逻辑块，块大小为8。
lengths = torch.full((batch, 1), 256, dtype=torch.int32, device=device)
cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
used_k = torch.full((batch,), key_length, dtype=torch.int32, device=device)
metadata = cot.quant_sparse_lightning_indexer_metadata(
    32, 1, 128, 512, 5,
    cu_seqlens_q=cu_q, seqused_k=used_k,
    batch_size=batch, max_seqlen_q=1, max_seqlen_k=key_length,
    candidate_block_length=lengths, candidate_block_size=8,
    layout_q="TND", layout_k="PA_BBND", mask_mode=0, cmp_ratio=1,
)
torch.npu.synchronize()
print(metadata.shape)  # (1024,)
```


metadata与计算的完整配套调用见[QSLI调用示例](../../quant_sparse_lightning_indexer/docs/torchapi_quant_sparse_lightning_indexer.md#调用示例)。
