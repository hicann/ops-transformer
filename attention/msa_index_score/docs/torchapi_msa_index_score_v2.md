# msa\_index\_score\_v2

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

## 功能说明

- **接口功能**：封装`aclnnMsaIndexScoreV2`，计算MSA Index Branch的block score, 支持指定layout_query和pool_block_size。
- **公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- `sparse_mode=0`：无因果；`sparse_mode=3`：rightDownCausal（须传`[2048,2048]` `atten_mask`）。
- `start_loc[B]`：当前query所在逻辑block索引，用于`local_mask`。
- `init_blocks` / `local_blocks`：Maxpool之后的强制高分块；与Triton raw score对齐时置0。默认`0` / `1`。
- `q_len` / `kv_len`允许为0（含整batch）。对应请求跳过QK；空KV的score为`-inf`。
- PageAttention `key`允许dim0（物理page）非连续，须保持为view（不要`.contiguous()`）；size>1的非首轴必须连续，否则tiling拒绝。TND / BSND / BNSD不允许非连续。
- PageAttention `block_table`第二维可以大于实际KV逻辑block数（例如vLLM预分配宽表）。

<!-- npu="A3, 910b" id5 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 短decode按估计M-task启动MIX，不打满空核。
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - query/key额外支持同型`torch.float8_e4m3fn` / `torch.float8_e5m2`（无`scale`）。
    - 短decode按估计M-task启动MIX，不打满空核；短M长S再按KV stile切`kvChunks`。
    - C2UB对score末维超过256列按窗flush。
<!-- end id6 -->

## 函数原型

```python
cann_ops_transformer.msa_index_score_v2(
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
    layout_query="TND",
    pool_block_size=128,
) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 360px">
<col style="width: 280px">
<col style="width: 260px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>dtype</th>
    <th>shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>必选</td>
    <td>$Q_{idx}$。由<code>layout_query</code>指定：TND/BSND/BNSD，默认TND。</td>
    <td>fp16/bf16</td>
    <td><code>[T1,N1,D]</code> / <code>[B,S1,N1,D]</code> / <code>[B,N1,S1,D]</code></td>
  </tr>
  <tr>
    <td>key</td>
    <td>必选</td>
    <td>$K_{idx}$。非分页TND/BSND/ BNSD必须与query同布局；分页BBND/BNBD可配任意query。PA允许dim0非连续。TND缺省逻辑块长128，每head一次块内max。</td>
    <td>与query同型</td>
    <td><code>[T2,N2,D]</code> / <code>[B,S2,N2,D]</code> / <code>[B,N2,S2,D]</code> / <code>[NP,P,N2,D]</code> / <code>[NP,N2,P,D]</code></td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>必选</td>
    <td>query所在逻辑block</td>
    <td>int32</td>
    <td><code>[B]</code></td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>PA必选</td>
    <td>逻辑→物理page；TND/BSND/BNSD不传</td>
    <td>int32</td>
    <td><code>[B,MB]</code></td>
  </tr>
  <tr>
    <td>scale</td>
    <td>可选</td>
    <td>非量化与FP8不传</td>
    <td>float32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>mode=3必选</td>
    <td>压缩下三角模板</td>
    <td>int8</td>
    <td><code>[2048,2048]</code></td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>必选</td>
    <td>query前缀和。BSND/BNSD时每个请求长度不超过$S1$</td>
    <td>int32</td>
    <td><code>[B+1]</code></td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>必选</td>
    <td>PA为各请求S2；TND/BSND/BNSD为key前缀和</td>
    <td>int32</td>
    <td><code>[B]</code> / <code>[B+1]</code></td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>可选</td>
    <td>key布局：<code>TND</code>/<code>BSND</code>/<code>BNSD</code>/<code>BBND</code>/<code>BNBD</code>，默认<code>BBND</code>。非分页必须与<code>layout_query</code>相同</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>可选</td>
    <td>0 / 3，默认3</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>可选</td>
    <td><code>local_mask</code>头部强制块数，默认0</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>可选</td>
    <td><code>local_mask</code>局部窗口，默认1</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layout_query</td>
    <td>可选</td>
    <td>query布局：<code>TND</code>/<code>BSND</code>/<code>BNSD</code>，默认<code>TND</code>。分页key可配任意query；非分页key必须相同</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pool_block_size</td>
    <td>可选</td>
    <td>打分池化块长：每个score归约的token数。<code>16</code>/<code>128</code>，默认128。与BBND/BNBD的存储页长不是同一个量；分页时不能超过页长。128走原单次块内maxpool；16走两级maxpool，且<code>N1</code>整除128。BSND/BNSD固定16</td>
    <td>int</td>
    <td>-</td>
  </tr>
</tbody>
</table>

<!-- npu="950" id4 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - query、key：数据类型除fp16/bf16外，还支持float8_e4m3fn / float8_e5m2。
    - query：BSND、BNSD布局仅支持fp16/bf16。
    - key：BSND、BNSD的逻辑块长为16，且仅支持fp16/bf16。
    - layout_query：支持BSND、BNSD。
    - pool_block_size：取值为16时，仅支持fp16/bf16。
<!-- end id4 -->

## 输出

块长16时`[1, T1, RoundUp(MB,16)]`，块长128时`[N1, T1, RoundUp(MB,16)]`，float32。`T1`是`actual_seq_qlen`的有效token总数。逻辑块长16时，先在每个head内对block取max，再对每个block取head间max，结果放在这一个head上；query的head数必须整除128。块长128时每个head只做一次块内max，不做head间max。

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
score = cann_ops_transformer.msa_index_score_v2(
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
score_tnd = cann_ops_transformer.msa_index_score_v2(
    query, key_tnd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_tnd,
    layout_key="TND")
```

- query BSND、key仍为BBND（`layout_query`与`layout_key`可以不同）：

```python
S1 = 32
query_bsnd = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
score_bsnd = cann_ops_transformer.msa_index_score_v2(
    query_bsnd, key_bbnd, start_loc,
    block_table=block_table, atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    layout_key="BBND", layout_query="BSND")
```

- query与key同为BNSD（仅<term>Ascend 950PR&950DT系列产品</term>，fp16/bf16，不传`block_table`，逻辑块长16）：

```python
S2 = 256
query_bnsd = torch.randn(B, N1, S1, D, dtype=torch.float16).npu()
key_bnsd = torch.randn(B, N2, S2, D, dtype=torch.float16).npu()
actual_seq_klen_bnsd = torch.tensor([0, S2], dtype=torch.int32).npu()
score_bnsd = cann_ops_transformer.msa_index_score_v2(
    query_bnsd, key_bnsd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_bnsd,
    layout_key="BNSD", layout_query="BNSD")
```
