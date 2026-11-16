# EngramHash

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                   |    √    |
| <term>Atlas A3系列产品</term>                             |    ×    |
| <term>Atlas A2系列产品</term>                             |    ×    |
| <term>Atlas 200I/500 A2推理产品</term>                    |    ×    |
| <term>Atlas推理系列产品</term>                            |    ×    |
| <term>Atlas训练系列产品</term>                            |    ×    |

## 功能说明

- **算子功能**：Engram 门控的 n-gram 哈希索引计算算子。对每层 `l`、每个 token `t`、每个输出列 `c`，按 n-gram 长度逐级计算 int64 多项式 hash 并取模 vocab，加上 embedding 表偏移，输出 INT32 索引。

- **计算公式**（对每个输出元素 `[l, t, c]`，`last = c / tables + 1`，`tb = c % tables`）：

  $$
  \begin{aligned}
  hash &= \mathrm{int64}(ids[t, 0]) \times mult[l, 0] \\
  hash &\mathrel{\oplus}= \mathrm{int64}(ids[t, k]) \times mult[l, k], \quad k = 1 \dots last \\
  r &= hash \bmod \mathrm{int64}(vocab[l, last-1, tb]) \\
  out[l, t, c] &= \mathrm{int32}(r) + offs[l, c]
  \end{aligned}
  $$

  - hash 中间量为 int64（乘积可达 $2^{62}$），`%` 为截断取模（truncmod），余数为负时加 `V` 修正。
  - 输出 shape 为 $[L, T, C]$，其中 $C = (max\_ngram\_size - 1) \times tables$。

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ngram_token_ids</td>
        <td>输入</td>
        <td>N-gram token ids，shape为(num_tokens, max_ngram_size)。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>multipliers</td>
        <td>输入</td>
        <td>每层的哈希乘子，shape为(num_ngram_layers, max_ngram_size)。</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>vocab_sizes</td>
        <td>输入</td>
        <td>每层每个 n-gram 长度每张 embedding 表的大小（素数桶），shape为(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram)。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>offsets</td>
        <td>输入</td>
        <td>每层各哈希桶的 embedding 表偏移（词表大小前缀和），shape为(num_ngram_layers, num_out_cols)，其中num_out_cols = (max_ngram_size - 1) × num_embed_table_per_ngram。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hash_indices</td>
        <td>输出</td>
        <td>哈希后的 embedding 索引，shape为(num_ngram_layers, num_tokens, num_out_cols)。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- <term>Ascend 950PR&950DT系列产品</term>：
  - 输入ngram_token_ids仅支持2维格式，shape为(num_tokens, max_ngram_size)，max_ngram_size不能小于2。
  - 输入vocab_sizes仅支持3维格式，shape为(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram)，num_ngram_layers与num_embed_table_per_ngram不能小于1。
  - 输入multipliers的shape须为(num_ngram_layers, max_ngram_size)，输入offsets的shape须为(num_ngram_layers, num_out_cols)，num_out_cols = (max_ngram_size - 1) × num_embed_table_per_ngram。
  - max_ngram_size有上限，需要能满足全载 SIMT 的 UB 窗口。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_engram_hash](examples/test_aclnn_engram_hash.cpp) | 通过[aclnnEngramHash](docs/aclnnEngramHash.md)接口方式调用EngramHash算子。 |
| PyTorch API | - | 通过[torchapi_engram_hash](docs/torchapi_engram_hash.md)接口方式调用EngramHash算子。 |
