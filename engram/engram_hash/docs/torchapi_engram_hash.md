# engram_hash

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

  Engram 门控的 n-gram 哈希索引计算接口。对每层 `l`、每个 token `t`、每个输出列 `c`，按 n-gram 长度逐级计算 int64 多项式 hash 并取模 vocab，加上 embedding 表偏移，输出 INT32 索引。

- **计算公式**（对每个输出元素 `[l, t, c]`，`last = c / tables + 1`，`tb = c % tables`）：

  $$
  \begin{aligned}
  hash &= \mathrm{int64}(ids[t, 0]) \times mult[l, 0] \\
  hash &\mathrel{\oplus}= \mathrm{int64}(ids[t, k]) \times mult[l, k], \quad k = 1 \dots last \\
  r &= hash \bmod \mathrm{int64}(vocab[l, last-1, tb]) \\
  out[l, t, c] &= \mathrm{int32}(r) + offs[l, c]
  \end{aligned}
  $$

  hash 中间量为 int64（乘积可达 $2^{62}$），`%` 为截断取模（truncmod），余数为负时加 `V` 修正。

## 函数原型

```python
cann_ops_transformer.engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 480px">
<col style="width: 140px">
<col style="width: 240px">
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
        <td>ngram_token_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>N-gram token ids。</td>
        <td>int32</td>
        <td>(T, N)</td>
    </tr>
    <tr>
        <td>multipliers</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每层的哈希乘子。对应公式中$mult[l, k]$。</td>
        <td>int64</td>
        <td>(L, N)</td>
    </tr>
    <tr>
        <td>vocab_sizes</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每层每个 n-gram 长度每张 embedding 表的大小（素数桶）。对应公式中$vocab[l, last-1, tb]$。</td>
        <td>int32</td>
        <td>(L, N-1, tables)</td>
    </tr>
    <tr>
        <td>offsets</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每层各哈希桶的 embedding 表偏移（词表大小前缀和）。对应公式中$offsets[l, c]$。</td>
        <td>int32</td>
        <td>(L, C)</td>
    </tr>
    </tbody>
</table>

## 输出说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 480px">
<col style="width: 140px">
<col style="width: 240px">
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
        <td>hash_indices</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>哈希后的 embedding 索引，为新分配张量。对应公式中$out[l, t, c]$。</td>
        <td>int32</td>
        <td>(L, T, C)</td>
    </tr>
    </tbody>
</table>

> 说明：T表示token数（num_tokens），N表示最大n-gram长度（max_ngram_size，不小于2），L表示哈希层数（num_ngram_layers），tables表示每个n-gram的哈希表数（num_embed_table_per_ngram），C = (N - 1) × tables表示输出列数（num_out_cols）。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 参数约束：输入输出支持空Tensor。不为空Tensor时，各个维度需要按照规格约束执行。
- 规格约束：
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - ngram_token_ids仅支持2维格式，N不能小于2。
    - vocab_sizes仅支持3维格式，且第2维必须为N-1，L与tables不能小于1。
    - multipliers的shape必须为(L, N)，offsets的shape必须为(L, C)。
    - N有上限，需要能满足全载 SIMT 的 UB 窗口。
    - T无规模限制。
  <!-- end id7 -->

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import engram_hash

  torch_npu.npu.set_device(0)

  T = 4096
  N = 3
  L = 2
  tables = 8

  ngram_token_ids = torch.randint(0, 100000, (T, N), dtype=torch.int32, device="npu")
  multipliers = torch.randint(0, 100000, (L, N), dtype=torch.int64, device="npu")
  vocab_sizes = torch.randint(100000, 1000000, (L, N - 1, tables), dtype=torch.int32, device="npu")
  offsets = torch.randint(0, 1000, (L, (N - 1) * tables), dtype=torch.int32, device="npu")

  hash_indices = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)

  print(f"Output shape: {hash_indices.shape}")
  ```

- 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import engram_hash

  torch_npu.npu.set_device(0)

  T = 4096
  N = 3
  L = 2
  tables = 8

  class EngramHashModel(torch.nn.Module):
      def forward(self, ngram_token_ids, multipliers, vocab_sizes, offsets):
          return engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)

  model = EngramHashModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  ngram_token_ids = torch.randint(0, 100000, (T, N), dtype=torch.int32, device="npu")
  multipliers = torch.randint(0, 100000, (L, N), dtype=torch.int64, device="npu")
  vocab_sizes = torch.randint(100000, 1000000, (L, N - 1, tables), dtype=torch.int32, device="npu")
  offsets = torch.randint(0, 1000, (L, (N - 1) * tables), dtype=torch.int32, device="npu")

  hash_indices = model(ngram_token_ids, multipliers, vocab_sizes, offsets)
  ```
