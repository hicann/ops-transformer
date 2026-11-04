# msa\_index\_score

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：支持
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

- **接口功能**：封装`aclnnMsaIndexScore`，计算MSA Index Branch的block score。对每个query token与每个KV sparse block，取该block内因果可见token的$Q_{idx}$与$K_{idx}$做matmul后按block Maxpool，得到逐block重要性分数，作为后续TopK的输入。
- **计算公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- **主要计算过程**：

  1. 将每个query token的$Q_{idx}$与KV sparse block内因果可见的$K_{idx}$做矩阵乘，得到token级相关性。
  2. 按`sparse_mode`叠加$atten\_mask$：`sparse_mode=0`不做因果截断，`sparse_mode=3`按rightDownCausal屏蔽不可见位置。
  3. 在每个长度为$block\_size$的sparse block上做Maxpool，得到逐block的重要性分数。
  4. 用`start_loc`、`init_blocks`、`local_blocks`生成$local\_mask$，对强制保留的block写入高分，得到输出`score`。本接口不包含TopK。

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
    init_blocks=0,
    local_blocks=1,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 描述 | dtype | shape |
| --- | --- | --- | --- | --- |
| query | 必选 | $Q_{idx}$，TND | fp16 / bf16 | `[T1,N1,D]` |
| key | 必选 | $K_{idx}$。PA允许dim0非连续 | 与query同型。 | `[NP,P,N2,D]` / `[NP,N2,P,D]` / `[T2,N2,D]` / `[B,S2,N2,D]` / `[B,N2,S2,D]` |
| start_loc | 必选 | query所在逻辑block | int32 | `[B]` |
| block_table | PA必选 | 逻辑block到物理page。TND / BSND / BNSD不传 | int32 | `[B,MB]` |
| scale | 可选 | 非量化与FP8不传 | float32 | - |
| atten_mask | mode=3必选 | 因果模板 | int8 | `[2048,2048]` |
| actual_seq_qlen | 必选 | query前缀和 | int32 | `[B+1]` |
| actual_seq_klen | 必选 | PA为各请求可见长度；TND / BSND / BNSD为前缀和 | int32 | `[B]` / `[B+1]` |
| layout_key | 可选 | `TND` / `BSND` / `BNSD` / `BBND` / `BNBD`，默认`BBND` | str | - |
| sparse_mode | 可选 | 0或3，默认3 | int | - |
| init_blocks | 可选 | `local_mask`头部强制块数，默认0 | int | - |
| local_blocks | 可选 | `local_mask`局部窗口，默认1 | int | - |

<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - query、key：数据类型支持BFLOAT16、FLOAT16。
<!-- end id8 -->
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - query、key：数据类型除fp16/bf16外，还支持float8_e4m3fn / float8_e5m2。
<!-- end id7 -->

## 返回值说明

| 返回值 | dtype | shape |
| --- | --- | --- |
| score | float32 | `[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]` |

## 约束说明

- query固定TND，逻辑块长128。BSND / BNSD的$S2$是定长槽，`actual_seq_klen[b+1] - actual_seq_klen[b]`不超过$S2$，且不传`block_table`。
- `q_len` / `kv_len`允许为0。空KV的score为`-inf`。
- 本接口不包含TopK。

## 确定性计算

默认支持确定性计算。

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

- TND packed key（`layout_key="TND"`，不传`block_table`，`actual_seq_klen`为`[B+1]`前缀和）：

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

- BSND（`actual_seq_klen`为`[B+1]`前缀和，不传`block_table`。BNSD把key换成`[B, N2, S2, D]`）：

```python
S2 = 256
key_bsnd = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
actual_seq_klen_bsnd = torch.tensor([0, S2], dtype=torch.int32).npu()
score_bsnd = cann_ops_transformer.msa_index_score(
    query, key_bsnd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_bsnd,
    layout_key="BSND")
```
