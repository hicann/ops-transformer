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

`return_choice=0`时只返回上面的`score`。`return_choice`为1或2时，对每个$(h,t)$在有效列上继续取TopK。`sparse_mode=3`时$visibleEnd=\mathrm{clamp}(kv-q+t_{off}+1,0,kv)$，`sparse_mode=0`时$visibleEnd=kv$。$BlockCap$在PageAttention下取`block_table`宽度与逻辑块上限的较小值，否则为逻辑块上限。

$$
validBlocks=\min(\lceil visibleEnd/128\rceil,BlockCap),\quad m=\min(topk,validBlocks),\quad tailStart=\lfloor validBlocks/1024\rfloor\times 1024
$$

在列号$[0,validBlocks)$中取$m$个下标。分数更高者优先，分数相同则列号更小者优先。这只决定入选集合。写出时不按分数排序：列号小于$tailStart$的入选列按列号从小到大，列号不小于$tailStart$的入选列接在后面并按列号从小到大。记排好的第$j$个列号为$I_j$。

$$
sparse\_indices[h,t,j]=\begin{cases}I_j&j<m\\-1&j\ge m\end{cases},\quad sparse\_values[h,t,j]=\begin{cases}score[h,t,I_j]&j<m\\-\infty&j\ge m\end{cases}
$$

`return_choice=1`时不返回`sparse_values`。`validBlocks=0`时各槽为$-1$和$-\infty$。

- **主要计算过程**：

  1. 将每个query token的$Q_{idx}$与KV sparse block内因果可见的$K_{idx}$做矩阵乘，得到token级相关性。
  2. 按`sparse_mode`叠加$atten\_mask$：`sparse_mode=0`不做因果截断，`sparse_mode=3`按rightDownCausal屏蔽不可见位置。
  3. 在每个长度为$block\_size$的sparse block上做Maxpool，得到逐block的重要性分数。
  4. 用`start_loc`、`init_blocks`、`local_blocks`生成$local\_mask$，对强制保留的block写入高分，得到输出`score`。
  5. `return_choice`为1或2时做TopK。前缀列号从小到大写出，尾块列号接在后面。尾块是有效列数按1024取余。不足k的索引为-1，值为`-inf`。`pool_block_size=128`时支持输出TopK：Ascend 950PR&950DT系列产品支持fp16、bf16、fp8与MXFP4；Atlas A2/A3系列产品只支持fp16和bf16。

## 函数原型

```python
cann_ops_transformer.msa_index_score(
    query,
    key,
    start_loc,
    *,
    block_table=None,
    query_scale=None,
    key_scale=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_key="BBND",
    sparse_mode=3,
    init_blocks=0,
    local_blocks=1,
    return_choice=0,
    topk=16,
    compress_factor=1,
)
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
    <th>参数类型</th>
    <th>描述</th>
    <th>dtype</th>
    <th>shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>必选</td>
    <td>Q_idx，TND</td>
    <td>fp16 / bf16</td>
    <td>[T1,N1,D]</td>
  </tr>
  <tr>
    <td>key</td>
    <td>必选</td>
    <td>K_idx。PA允许dim0非连续</td>
    <td>与query同型。</td>
    <td>[NP,P,N2,D] / [NP,N2,P,D] / [T2,N2,D] / [B,S2,N2,D] / [B,N2,S2,D]</td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>必选</td>
    <td>query所在逻辑block</td>
    <td>int32</td>
    <td>[B]</td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>PA必选</td>
    <td>逻辑block到物理page。TND / BSND / BNSD不传</td>
    <td>int32</td>
    <td>[B,MB]</td>
  </tr>
  <tr>
    <td>query_scale</td>
    <td>可选</td>
    <td>MXFP4的Q逐组系数，其他类型为None</td>
    <td>float8_e8m0fnu</td>
    <td>Q形状前缀相同，末维D/32</td>
  </tr>
  <tr>
    <td>key_scale</td>
    <td>可选</td>
    <td>MXFP4的K逐组系数，原生浮点不传</td>
    <td>float32 / float8_e8m0fnu</td>
    <td>MXFP4与K形状前缀相同，末维D/32</td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>mode=3必选</td>
    <td>因果模板</td>
    <td>int8</td>
    <td>[2048,2048]</td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>必选</td>
    <td>query前缀和</td>
    <td>int32</td>
    <td>[B+1]</td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>必选</td>
    <td>PA为各请求可见长度；TND / BSND / BNSD为前缀和</td>
    <td>int32</td>
    <td>[B] / [B+1]</td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>可选</td>
    <td>TND / BSND / BNSD / BBND / BNBD，默认BBND</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>可选</td>
    <td>0或3，默认3</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>可选</td>
    <td>local_mask头部强制块数，默认0</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>可选</td>
    <td>local_mask局部窗口，默认1</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>compress_factor</td>
    <td>可选</td>
    <td>key压缩倍数，默认1。当前只支持1</td>
    <td>int</td>
    <td>-</td>
  </tr>
</tbody>
</table>

<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - query、key：数据类型支持BFLOAT16、FLOAT16。
<!-- end id8 -->
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - query、key：数据类型除fp16/bf16外，还支持float8_e4m3fn / float8_e5m2。
- 原型预留但当前会拒绝：A2/A3的float16 score/values、int8 query+int8 key；bf16/fp16的float16输出，以及float8_e4m3fn query配float4_e2m1 key。
<!-- end id7 -->

## 返回值说明

| 返回值 | dtype | shape |
| --- | --- | --- |
| score（`return_choice=0`） | float32 | `[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]` |
| sparse_indices（`return_choice=1`或`2`） | int32 | `[N1, T1, topk]` |
| sparse_values（`return_choice=2`） | float32 | `[N1, T1, topk]` |

## 约束说明

- `actual_seq_qlen` 和 `actual_seq_klen` 可在 CPU 或 query 所在 NPU；CPU 长度会复制到设备，并在调用返回前等待当前 NPU 流完成；该路径不支持外部流或 NPU 图捕获。其他已传入的 Tensor 必须位于 query 所在 NPU。

- query固定TND，逻辑块长128。BSND / BNSD的$S2$是定长槽，`actual_seq_klen[b+1] - actual_seq_klen[b]`不超过$S2$，且不传`block_table`。
- `q_len` / `kv_len`允许为0。空KV的score为`-inf`。
- `return_choice`默认0，返回score。为1时返回`sparse_indices`，为2时返回`(sparse_indices, sparse_values)`。`topk`默认16，范围1到128。`pool_block_size=128`时支持输出TopK。Ascend 950PR&950DT系列产品支持fp16、bf16、fp8与MXFP4；Atlas A2/A3系列产品只支持fp16和bf16。

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

### MXFP4（A4C4）

仅Ascend 950支持Q/K均为`float4_e2m1fn_x2`，逻辑D为64或128、N2=1。Torch每个元素打包两个FP4值，因此输入末维为D/2；两份独立`float8_e8m0fnu`系数的形状前缀与对应Q/K相同、末维D/32，按每32个元素分别缩放后再点积。
