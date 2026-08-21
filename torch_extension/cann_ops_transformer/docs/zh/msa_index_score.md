# msa\_index\_score

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id3 -->

## 功能说明

- **接口功能**：封装 `aclnnMsaIndexScore`，计算 MSA Index Branch 的 block score。
- **公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- `sparse_mode=0`：无因果；`sparse_mode=3`：rightDownCausal（须传 `[2048,2048]` `atten_mask`）。
- `start_loc[B]`：当前 query 所在逻辑 block 索引，用于 `local_mask`。

## 函数原型

```python
cann_ops_transformer.msa_index_score(
    query,
    key,
    start_loc,
    *,
    block_table=None,
    scale=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_key="BBND",
    sparse_mode=3,
) -> Tensor
```

## 参数说明

| 参数名 | 可选/必选 | 描述 | dtype | shape |
|--------|-----------|------|-------|-------|
| query | 必选 | $Q_{idx}$ TND | fp16/bf16 | `[T1,N1,D]` |
| key | 必选 | $K_{idx}$ PA BBND/BNBD 或 TND | 同 query 或 int8 | `[NP,P,N2,D]` / `[NP,N2,P,D]` / `[T2,N2,D]` |
| start_loc | 必选 | query 所在逻辑 block | int32 | `[B]` |
| block_table | PA 必选 | 逻辑→物理 page；TND 不传 | int32 | `[B,MB]` |
| scale | 量化必选 | 反量化 | float32 | PA `[NP,N2,P]`；TND `[T2,N2]` |
| atten_mask | mode=3 必选 | 压缩下三角模板 | int8 | `[2048,2048]` |
| actual_seq_qlen | TND query 必选 | query 前缀和 | int32 | `[B+1]` |
| actual_seq_klen | 必选 | PA 为各请求 S2；TND 为 key 前缀和 | int32 | `[B]` / `[B+1]` |
| layout_key | 可选 | key 布局：`TND` / `BBND` / `BNBD`，默认 `BBND` | str | - |
| sparse_mode | 可选 | 0 / 3，默认 3 | int | - |

## 输出

`[N1, T1, RoundUp(MB,16)]` float32

## 调用示例

- PageAttention BBND / BNBD：

```python
import torch
import torch_npu
import cann_ops_transformer

T1, N1, N2, D, P = 32, 8, 1, 128, 128
B, NP, MB = 1, 8, 2
query = torch.randn(T1, N1, D, dtype=torch.float16).npu()
key_bbnd = torch.randn(NP, P, N2, D, dtype=torch.float16).npu()
# BNBD 与 BBND 仅 N/P 轴对调：key_bnbd = key_bbnd.permute(0, 2, 1, 3).contiguous()
block_table = torch.arange(B * MB, dtype=torch.int32).view(B, MB).npu()
actual_seq_qlen = torch.tensor([0, T1], dtype=torch.int32).npu()
actual_seq_klen = torch.tensor([256], dtype=torch.int32).npu()
start_loc = torch.tensor([1], dtype=torch.int32).npu()
atten_mask = torch.zeros(2048, 2048, dtype=torch.int8).npu()
score = cann_ops_transformer.msa_index_score(
    query, key_bbnd, start_loc,
    block_table=block_table, atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    layout_key="BBND")
```

- TND packed key（`layout_key="TND"`，不传 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和）：

```python
T2 = 256
key_tnd = torch.randn(T2, N2, D, dtype=torch.float16).npu()
actual_seq_klen_tnd = torch.tensor([0, T2], dtype=torch.int32).npu()
score_tnd = cann_ops_transformer.msa_index_score(
    query, key_tnd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_tnd,
    layout_key="TND")
```
