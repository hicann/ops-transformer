# msa\_index\_score

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id3 -->

## 功能说明

- **接口功能**：封装 `aclnnMsaIndexScore`，计算 MSA Index Branch 的 block score。
- **公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- `sparse_mode=0`：无因果；`sparse_mode=3`：rightDownCausal（须传 `[2048,2048]` `atten_mask`）。
- `start_loc[B]`：当前 query 所在逻辑 block 索引，用于 `local_mask`。
- `init_blocks` / `local_blocks`：Maxpool 之后的强制高分块；与 Triton raw score 对齐时置 0。默认 `0` / `1`。
- Ascend 950 额外支持 query/key 同型 `torch.float8_e4m3fn` / `torch.float8_e5m2` / `torch_npu.hifloat8`（无 `scale`）。`hifloat8.npu()` 当前不安全，测试脚本跳过；kernel 已注册。
- `q_len` / `kv_len` 允许为 0（含整 batch）。对应请求跳过 QK；空 KV 的 score 为 `-inf`。
- A2/A3 与 Ascend 950 短 decode 按估计 M-task 启动 MIX，不打满空核。
- PageAttention `key` 允许 dim0（物理 page）非连续，须保持为 view（不要 `.contiguous()`）；TND 不允许。
- PageAttention `block_table` 第二维可以大于实际 KV 逻辑 block 数（例如 vLLM 预分配宽表）。950 C2UB 对 score 末维超过 256 列按窗 flush。

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
    <td>$Q_{idx}$ TND</td>
    <td>fp16/bf16；950 另支持 fp8_e4m3fn / fp8_e5m2 / hifloat8</td>
    <td><code>[T1,N1,D]</code></td>
  </tr>
  <tr>
    <td>key</td>
    <td>必选</td>
    <td>$K_{idx}$ PA BBND/BNBD 或 TND。A2/A3 与 950 上 PA 允许 dim0 非连续。</td>
    <td>同 query 或 int8（int8 配 fp16 query）</td>
    <td><code>[NP,P,N2,D]</code> / <code>[NP,N2,P,D]</code> / <code>[T2,N2,D]</code></td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>必选</td>
    <td>query 所在逻辑 block</td>
    <td>int32</td>
    <td><code>[B]</code></td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>PA 必选</td>
    <td>逻辑→物理 page；TND 不传</td>
    <td>int32</td>
    <td><code>[B,MB]</code></td>
  </tr>
  <tr>
    <td>scale</td>
    <td>量化必选</td>
    <td>反量化</td>
    <td>float32</td>
    <td>PA <code>[NP,N2,P]</code>；TND <code>[T2,N2]</code></td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>mode=3 必选</td>
    <td>压缩下三角模板</td>
    <td>int8</td>
    <td><code>[2048,2048]</code></td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>TND query 必选</td>
    <td>query 前缀和</td>
    <td>int32</td>
    <td><code>[B+1]</code></td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>必选</td>
    <td>PA 为各请求 S2；TND 为 key 前缀和</td>
    <td>int32</td>
    <td><code>[B]</code> / <code>[B+1]</code></td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>可选</td>
    <td>key 布局：<code>TND</code> / <code>BBND</code> / <code>BNBD</code>，默认 <code>BBND</code></td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>可选</td>
    <td>0 / 3，默认 3</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>可选</td>
    <td><code>local_mask</code> 头部强制块数，默认 0</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>可选</td>
    <td><code>local_mask</code> 局部窗口，默认 1</td>
    <td>int</td>
    <td>-</td>
  </tr>
</tbody>
</table>

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
