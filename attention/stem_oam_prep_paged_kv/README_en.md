# StemOamPrepPagedKv

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported or Not |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                 |    √     |
| <term>Atlas A3 products</term> |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term> |      ×     |
| <term>Atlas inference products</term> |      ×     |
| <term>Atlas training products</term> |      ×     |

## Function Description

- Operator function: The Stem OAM (Output-Aware Metric) is the preceding scoring module of the dynamic sparse attention mechanism for large model inference. It extracts K/V data from the Paged KV Cache, performs K Processing (per-token K-scale × group sum + anti-diagonal flip) and V Processing (per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average), and outputs kFlat and vBias for Stem OAM score computation.

- The inputs and outputs support the following data scenarios:

    ```
    kCache:[total_blocks, kvBlockSize, H_kv, 128] or [total_blocks, H_kv, kvBlockSize, 128]
    vCache:[total_blocks, kvBlockSize, H_kv, 128] or [total_blocks, H_kv, kvBlockSize, 128]
    kvIndices:[batch, max_kv_blocks]
    kvSeqLens:[batch]
    kScaleCache:[total_blocks, kvBlockSize, H_kv, 1] or [total_blocks, H_kv, kvBlockSize, 1]
    vScale:[H_kv]
    kFlat:[batch, H_kv, max_Kb, stemStride * 128]
    vBias:[batch, H_kv, max_Kb]
    ```

- kCache/vCache supports two layouts, specified by the kvLayout attribute: "BBND" indicates [total_blocks, kvBlockSize, H_kv, 128], and "BNBD" indicates [total_blocks, H_kv, kvBlockSize, 128]. Currently, only "BNBD" is supported.
- The layout of kScaleCache changes with kvLayout and is consistent with the layout of kCache/vCache (only the last dimension is D=1 instead of D=128).
- Only the FP8_E4M3FN input path is supported.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 240px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>kCache</td>
      <td>Input</td>
      <td>Paged K cache. Empty tensors are not supported. Two layouts are available, specified by kvLayout. Currently, only "BNBD" is supported: "BBND" indicates [total_blocks, kvBlockSize, H_kv, 128], and "BNBD" indicates [total_blocks, H_kv, kvBlockSize, 128]. The first three dimensions support non-contiguity, and the last dimension must be contiguous.</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vCache</td>
      <td>Input</td>
      <td>Paged V cache. Empty tensors are not supported. The layout must be consistent with kCache.</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kvIndices</td>
      <td>Input</td>
      <td>Array of KV block indices for each batch. Empty tensors are not supported. The shape is [batch, max_kv_blocks], and the maximum value of max_kv_blocks is 2048.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kvSeqLens</td>
      <td>Input</td>
      <td>KV sequence length of each batch. Empty lists are not supported. The shape is [batch], and the maximum KV sequence length is 262144.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kScaleCache</td>
      <td>Input</td>
      <td>Per-token per-head K scale. This parameter is required when the data type of kCache is FP8 and can be omitted for other types. Two layouts are available, specified by kvLayout. Currently, only "BNBD" is supported: "BBND" indicates [total_blocks, kvBlockSize, H_kv, 1], and "BNBD" indicates [total_blocks, H_kv, kvBlockSize, 1]. The first three dimensions support non-contiguity, and the last dimension must be contiguous.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vScale</td>
      <td>Input</td>
      <td>Per-head V scale. This parameter is required when the data type of kCache is FP8 and can be omitted for other types. The shape is [H_kv].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambdaMag</td>
      <td>Attribute</td>
      <td>V bias multiplier. Value range: (0,1]. Default value: 0.3.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvLayout</td>
      <td>Attribute</td>
      <td>KV Cache layout. Value range: "BBND" and "BNBD". Currently, only "BNBD" is supported, and the default value is "BNBD".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemBlockSize</td>
      <td>Attribute</td>
      <td>Stem block size. Value range: %32==0, ≤256, and stemBlockSize must be a multiple of stemStride. The recommended value is 128. Default value: 128.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemStride</td>
      <td>Attribute</td>
      <td>Stride size. Value range: %16==0, ≤64, and ≤stemBlockSize. The recommended value is 16. Default value: 16.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kFlat</td>
      <td>Output</td>
      <td>Result of K group sum + anti-diag flip. Empty tensors are not supported. The shape is [batch, H_kv, max_Kb, kflat_dim], where kflat_dim=stemStride×128, and max_Kb is computed based on the kvSeqLens input.</td>
      <td>BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vBias</td>
      <td>Output</td>
      <td>V block bias result. Empty tensors are not supported. The shape is [batch, H_kv, max_Kb], and max_Kb is computed based on the kvSeqLens input.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- kvBlockSize ∈ {64, 128}.

## Calling Instructions

| Calling Method | Sample Code                                                  | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_stem_oam_prep_paged_kv](./examples/test_aclnn_stem_oam_prep_paged_kv.cpp) | The StemOamPrepPagedKv operator is called through [aclnnStemOamPrepPagedKv](docs/aclnnStemOamPrepPagedKv_en.md) |
| PyTorch API | - | The stem_oam_prep_paged_kv operator is called through the [stem_oam_prep_paged_kv](docs/torchapi_stem_oam_prep_paged_kv_en.md) API. |
