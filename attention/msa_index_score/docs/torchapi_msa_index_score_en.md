# msa\_index\_score

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:54:15.798Z -->

## Applicable Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**: Wraps `aclnnMsaIndexScore`. Query stays TND. The API does not include TopK.
- **Formula**:

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- Computation: load visible $K$ by `layout_key`, compute $QK^{T}$, apply rightDownCausal when `sparse_mode=3`, Maxpool over 128-token blocks, then add `local_mask`.
- `layout_key` is `TND`, `BSND`, `BNSD`, `BBND`, or `BNBD`. BSND / BNSD omit `block_table` and pass the `[B+1]` key prefix sum.

## Function Prototype

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

## Parameters

<table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 360px">
<col style="width: 280px">
<col style="width: 260px">
</colgroup>
<thead>
  <tr>
    <th>Parameter</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>dtype</th>
    <th>shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>Required</td>
    <td>$Q_{idx}$ TND</td>
    <td>fp16/bf16</td>
    <td><code>[T1,N1,D]</code></td>
  </tr>
  <tr>
    <td>key</td>
    <td>Required</td>
    <td>$K_{idx}$. TND, BSND, BNSD, BBND, or BNBD. Query stays TND. BSND / BNSD do not pass <code>block_table</code>. PA may be non-contiguous on dim0.</td>
    <td>Same dtype as query</td>
    <td><code>[T2,N2,D]</code> / <code>[B,S2,N2,D]</code> / <code>[B,N2,S2,D]</code> / <code>[NP,P,N2,D]</code> / <code>[NP,N2,P,D]</code></td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>Required</td>
    <td>Logical block where the query is located.</td>
    <td>int32</td>
    <td><code>[B]</code></td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>Required for PA</td>
    <td>Logical to physical page. Not passed for TND / BSND / BNSD.</td>
    <td>int32</td>
    <td><code>[B,MB]</code></td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Optional</td>
    <td>Omitted for non-quantized and FP8 inputs.</td>
    <td>float32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>Required for mode=3</td>
    <td>Compressed lower triangular template.</td>
    <td>int8</td>
    <td><code>[2048,2048]</code></td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>Required for TND query</td>
    <td>Query prefix sum.</td>
    <td>int32</td>
    <td><code>[B+1]</code></td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>Required</td>
    <td>Visible length <code>[B]</code> for PA. Prefix sum <code>[B+1]</code> for TND / BSND / BNSD.</td>
    <td>int32</td>
    <td><code>[B]</code> / <code>[B+1]</code></td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>optional</td>
    <td>Key layout: <code>TND</code> / <code>BSND</code> / <code>BNSD</code> / <code>BBND</code> / <code>BNBD</code>, default <code>BBND</code>.</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>optional</td>
    <td>0 / 3, default 3.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>optional</td>
    <td>Number of forced head blocks of <code>local_mask</code>. Defaults to 0</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>optional</td>
    <td>Local window of <code>local_mask</code>. Defaults to 1</td>
    <td>int</td>
    <td>-</td>
  </tr>
</tbody>
</table>

<!-- npu="950" id8 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - query and key: BFLOAT16 and FLOAT16 are supported.
<!-- end id8 -->
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
    - query and key: In addition to fp16/bf16, fp8_e4m3fn / fp8_e5m2 are supported.
<!-- end id7 -->

## Output

`[N1, T1, RoundUp(MB,16)]` float32

## Example

- PageAttention BBND / BNBD:

```python
import torch
import torch_npu
import cann_ops_transformer

T1, N1, N2, D, P = 32, 8, 1, 128, 128
B, NP, MB = 1, 8, 2
query = torch.randn(T1, N1, D, dtype=torch.float16).npu()
key_bbnd = torch.randn(NP, P, N2, D, dtype=torch.float16).npu()
# BNBD and BBND differ only in the N/P axis swap: key_bnbd = key_bbnd.permute(0, 2, 1, 3).contiguous()
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

- TND packed key (`layout_key="TND"`, `block_table` not passed, `actual_seq_klen` is the `[B+1]` prefix sum):

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
