# k2q\_csr

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

- **接口功能**：

  将每条 query 的 KV block 索引 `q2k` 转为按 KV 行为主序的 CSR（Compressed Sparse Row）。接口在 Host 侧编排五阶段 aclnn 算子：

  ```text
  aclnnK2qCsrMeta → aclnnK2qCsrHist → aclnnK2qCsrRowPrefix
    → aclnnK2qCsrTilePrefix → aclnnK2qCsrScatter
  ```

  输出 `row_ptr`（行指针）、`q_ind`（query 下标）与 `slot`（topk 槽位）。未命中位置为 `-1`。

- **计算过程**：

  1. Meta：按 `order_method` 生成 `row_map` 与 `token_batch`。
  2. Hist：统计每个 tile / 行上的非空命中次数。
  3. RowPrefix：对行计数做 exclusive scan，写出 `row_ptr`。
  4. TilePrefix：生成各 tile 的绝对写偏移。
  5. Scatter：按 CSR 写出 `q_ind` 与 `slot`。

## 函数原型

```python
cann_ops_transformer.k2q_csr(
    q2k,
    cu_seqlens,
    cu_block_lens,
    *,
    order_method=0,
    total_rows=None,
    max_kv=None,
    use_simt=False,
    q_global_offset=False,
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 360px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q2k</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每条 query 的 KV block 索引。</td>
        <td>int32</td>
        <td>(H, T, topk)</td>
    </tr>
    <tr>
        <td>cu_seqlens</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>query 侧前缀和。</td>
        <td>int32</td>
        <td>(B+1)</td>
    </tr>
    <tr>
        <td>cu_block_lens</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>KV block 前缀和。末元素即 total_rows。</td>
        <td>int32</td>
        <td>(B+1)</td>
    </tr>
    <tr>
        <td>order_method</td>
        <td>int</td>
        <td>可选</td>
        <td>0：按 batch 内 block 顺序建 row_map；1：round-robin 交错。默认 0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>total_rows</td>
        <td>int</td>
        <td>可选</td>
        <td>KV 总行数。传入非负整数则 Host 直用；None 时从 cu_block_lens 末元素推导。默认 None。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>max_kv</td>
        <td>int</td>
        <td>可选</td>
        <td>单 batch 最大 KV block 数。None 时从 cu_block_lens 差分最大值推导。默认 None。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>use_simt</td>
        <td>bool</td>
        <td>可选</td>
        <td>True 时 Hist/Scatter 走 SIMT，仅 Ascend 950 生效。默认 False。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>q_global_offset</td>
        <td>bool</td>
        <td>可选</td>
        <td>True 时 q_ind 为全局 Q token 下标；False 为 batch-local。默认 False。</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 输出说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 360px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>row_ptr</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>CSR 行指针。</td>
        <td>int32</td>
        <td>(H, total_rows+1)</td>
    </tr>
    <tr>
        <td>q_ind</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>非空位为 query 下标；未写入位为 -1。</td>
        <td>int32</td>
        <td>(H, T*topk)</td>
    </tr>
    <tr>
        <td>slot</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>对应 topk slot；未写入位为 -1。</td>
        <td>int32</td>
        <td>(H, T*topk)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式调用。图模式调用暂不支持。
- `q2k`、`cu_seqlens`、`cu_block_lens` 必须位于 NPU，dtype 为 int32，不支持空 Tensor。
- `q2k` 必须为 3 维 `[H, T, topk]`。
- `order_method` 仅支持 0 或 1。
- 建议在已知 shape 时显式传入 `total_rows` / `max_kv`，避免 Host D2H。
- SoC 与实现路径：
  - Atlas A2 / A3：Hist / Scatter 仅 MC，`use_simt` 无效。
  - Ascend 950：`use_simt=False` 走 MC；`True` 走 SIMT。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import k2q_csr

  H, T, topk = 8, 128, 8
  q2k = torch.randint(0, 16, (H, T, topk), dtype=torch.int32, device="npu")
  cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="npu")
  cu_block_lens = torch.tensor([0, 16], dtype=torch.int32, device="npu")

  row_ptr, q_ind, slot = k2q_csr(
      q2k,
      cu_seqlens,
      cu_block_lens,
      order_method=0,
      total_rows=16,
      max_kv=16,
      use_simt=True,
  )
  ```

- 图模式调用（暂不支持）
