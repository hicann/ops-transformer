# EngramHash

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √    |
| <term>Atlas A3 products</term>                             |    ×    |
| <term>Atlas A2 products</term>                             |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                    |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                            |    ×    |

## Description

- **Operator function**: computes n-gram hash indices for Engram gating. For each layer `l`, each token `t`, and each output column `c`, the operator progressively computes an int64 polynomial hash over the n-gram length. It then takes the modulo of the vocab size, adds the embedding table offset, and outputs an INT32 index.

- **Computation formula** (for each output element `[l, t, c]`, where `last = c / tables + 1` and `tb = c % tables`):

  $$
  \begin{aligned}
  hash &= \mathrm{int64}(ids[t, 0]) \times mult[l, 0] \\
  hash &\mathrel{\oplus}= \mathrm{int64}(ids[t, k]) \times mult[l, k], \quad k = 1 \dots last \\
  r &= hash \bmod \mathrm{int64}(vocab[l, last-1, tb]) \\
  out[l, t, c] &= \mathrm{int32}(r) + offs[l, c]
  \end{aligned}
  $$

  - The hash intermediate value is int64 (the product can reach $2^{62}$); `%` is the truncation modulo (truncmod); and `V` is added as a correction when the remainder is negative.
  - The output shape is $[L, T, C]$, where $C = (max\_ngram\_size - 1) \times tables$.

## Parameters

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">Parameter</th>
        <th style="white-space: nowrap">Input/Output/Attribute</th>
        <th style="white-space: nowrap">Description</th>
        <th style="white-space: nowrap">Data Type</th>
        <th style="white-space: nowrap">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ngram_token_ids</td>
        <td>Input</td>
        <td>N-gram token IDs, with shape (num_tokens, max_ngram_size).</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>multipliers</td>
        <td>Input</td>
        <td>Hash multiplier of each layer, with shape (num_ngram_layers, max_ngram_size).</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>vocab_sizes</td>
        <td>Input</td>
        <td>Size of each embedding table (prime number of buckets) for each layer and each n-gram length, with shape (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram).</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>offsets</td>
        <td>Input</td>
        <td>Embedding table offset of each hash bucket of each layer (prefix sum of vocab sizes), with shape (num_ngram_layers, num_out_cols), where num_out_cols = (max_ngram_size - 1) × num_embed_table_per_ngram.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hash_indices</td>
        <td>Output</td>
        <td>Embedding indices after hashing, with shape (num_ngram_layers, num_tokens, num_out_cols).</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - The input ngram_token_ids supports only the 2-D format with shape (num_tokens, max_ngram_size), and max_ngram_size must be 2 or greater.
  - The input vocab_sizes supports only the 3-D format with shape (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), and num_ngram_layers and num_embed_table_per_ngram must be 1 or greater.
  - The shape of the input multipliers must be (num_ngram_layers, max_ngram_size). The shape of the input offsets must be (num_ngram_layers, num_out_cols), where num_out_cols = (max_ngram_size - 1) × num_embed_table_per_ngram.
  - max_ngram_size has an upper limit and must fit the UB window in full-load SIMT mode.

## Calling Instructions

| Calling Method      | Sample Code                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_engram_hash](examples/test_aclnn_engram_hash.cpp) | Calls the EngramHash operator through the [aclnnEngramHash](docs/aclnnEngramHash.md) API. |
| PyTorch API | - | Calls the EngramHash operator through the [torchapi_engram_hash](docs/torchapi_engram_hash.md) API. |
