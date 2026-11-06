# GenericBlockSparseAttentionMetadata

English | [简体中文](./README.md)

## Product Support

- <term>Ascend 950PR&950DT products</term>: supported
- <term>Atlas A3 products</term>: supported
- <term>Atlas A2 products</term>: supported
- <term>Atlas 200I/500 A2 inference products</term>: not supported
- <term>Atlas inference products</term>: not supported
- <term>Atlas training products</term>: not supported

## Description

- Operator function:

  `GenericBlockSparseAttentionMetadata` is the preceding operator of `GenericBlockSparseAttention`. It generates the core allocation result `metadataOptional` used by the main operator based on the Query sequence lengths, Q/KV head counts, and sparse block information. This operator does not perform the Attention calculation.

  The typical calling workflow is as follows:

  1. Prepare inputs such as `sparseBlockIdx`, `sparseBlockCount`, sequence lengths, and attributes.
  2. Call `aclnnGenericBlockSparseAttentionMetadata` to generate `metadataOptional`.
  3. Call `aclnnGenericBlockSparseAttention` and pass the `metadataOptional` obtained in the previous step as the input of the main operator.

## Parameter Description

> **Note:**<br>
> B indicates the batch size, `totalQBlocks` indicates the total number of Q blocks after division by the Query storage length of each batch, and `maxSparseBlockCount` indicates the last dimension of `sparseBlockIdx`.

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 220px">
  <col style="width: 130px">
  <col style="width: 550px">
  <col style="width: 180px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sparseBlockIdx</td>
      <td>Input</td>
      <td>Sparse block index. The current shape is [numKeyValueHeads, totalQBlocks, maxSparseBlockCount].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>Input</td>
      <td>Number of KV blocks actually selected by each Q block. The current shape is [numKeyValueHeads, totalQBlocks].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>Input</td>
      <td>Prefix sum of the Query sequence storage lengths in each batch. This parameter must be passed in for TND; it is not used for BSND/BNSD, and it is recommended to pass in nullptr. The shape is (B+1,).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>Input</td>
      <td>Prefix sum of the Key/Value sequence storage lengths in each batch. This parameter must be passed in when layoutKv is "TND" and must not be passed in for PA_BBND/PA_BNBD; it is not used for BSND/BNSD, and it is recommended to pass in nullptr. The shape is (B+1,).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>Conditionally Required Input</td>
      <td>Actual sequence length of Query in each batch. The shape is (B,). This parameter is required for BSND/BNSD and optional for TND; when it is not passed in for TND, the Query storage length is used.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>Input</td>
      <td>Actual sequence length of Key/Value in each batch. The shape is (B,). This parameter is required for PA_BBND/PA_BNBD and BSND/BNSD KV, and optional for TND KV.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>Attribute</td>
      <td>Maximum sequence length of Query. For TND, -1 (unspecified) or a non-negative value is allowed; for BSND/BNSD, it must be equal to the shape capacity Sq and greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>Attribute</td>
      <td>Maximum sequence length of Key/Value. -1 (unspecified) or a non-negative value is allowed. It is not used to derive the actual KV length for BSND/BNSD.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>Attribute</td>
      <td>Number of heads of the Query of the main operator.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>Attribute</td>
      <td>Number of heads of the Key/Value of the main operator.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>Attribute</td>
      <td>Feature dimension of each head of the Query, Key, and Value of the main operator. For the value range of headDim, see <a href="#product-specific-constraints">Product-Specific Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>Attribute</td>
      <td>Sparse block shape [blockShapeX, blockShapeY]. blockShapeX is 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>Attribute</td>
      <td>Sparse pattern layout. Currently only 4 is supported; the Q heads corresponding to the same KV head share the sparse pattern.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>Attribute</td>
      <td>Data layout format of the Query of the main operator. "TND", "BSND", and "BNSD" are supported. For the application scope, see <a href="#product-specific-constraints">Product-Specific Constraints</a>.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>Attribute</td>
      <td>Data layout format of the Key/Value of the main operator. "PA_BBND", "PA_BNBD", "TND", "BSND", or "BNSD" is supported. For the application scope, see <a href="#product-specific-constraints">Product-Specific Constraints</a>.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>Attribute</td>
      <td>Mask type of the Attention calculation. 0 (non-causal) or 1 (causal) is supported. For the application scope, see <a href="#product-specific-constraints">Product-Specific Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>Attribute</td>
      <td>Quantization type. 0 or 5 is supported. For the application scope, see <a href="#product-specific-constraints">Product-Specific Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>Attribute</td>
      <td>Precision level of the Softmax calculation. The value is 0 or 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>Attribute</td>
      <td>Number of tokens included forward in the sliding window Attention. Currently only -1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>Attribute</td>
      <td>Number of tokens included backward in the sliding window Attention. Currently only -1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualBlockMode</td>
      <td>Attribute</td>
      <td>0: the trailing incomplete block is determined by the explicit sparse list; 1: the trailing incomplete block is implicitly included and is not contained in the explicit list. For the application scope of the values, see <a href="#product-specific-constraints">Product-Specific Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isConsistentTopk</td>
      <td>Attribute</td>
      <td>Indicates whether the maximum number of KV blocks selected by the Q blocks within the same batch and the same head is consistent. The value is false or true and must be the same as that of the main operator/Metadata.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>Output</td>
      <td>Core allocation result used by the main operator. The shape is fixed to (1024,).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- Sparse block shape [blockShapeX, blockShapeY]. blockShapeX is 1. For <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, Y supports 1, 2, 4, 8, 16, 32, 64, and 128.
- For <term>Ascend 950PR&950DT products</term>, Y in the non-quantized scenario supports any integer from 1 to 128; in the quantized scenario, only [1,128] is supported. The value must be the same as that of the main operator.


## Constraints

- This operator must be used together with `GenericBlockSparseAttention`. All corresponding inputs and attributes must be the same as those of the main operator called later, and `metadataOptional` must be regenerated before each call to the main operator.
- For the layout combinations and their application scopes, see [Product-Specific Constraints](#product-specific-constraints). `layoutSparsePattern=4` and `blockShapeX=1`. For the range of headDim, see [Product-Specific Constraints](#product-specific-constraints).
- `numQHeads >= numKvHeads` and `numQHeads % numKvHeads == 0`, and `numQHeads / numKvHeads` does not exceed 128.
- The last dimension of `sparseBlockIdx` must not be less than the maximum value of all elements in `sparseBlockCount` and must be greater than 0. Capacity upper limits: INT32_MAX for <term>Atlas A2 products</term> and <term>Atlas A3 products</term> in the non-quantized scenario, INT32_MAX for <term>Ascend 950PR&950DT products</term> in the non-quantized scenario, and 256 for <term>Ascend 950PR&950DT products</term> in the FP8 quantized scenario.
- Each group of valid sparse indices must be non-negative, strictly ascending, and duplicate-free, located in the first `sparseBlockCount` positions. Random ascending order is acceptable; equal strides or physical page continuity is not required.
- When `residualBlockMode=1`, the explicit list for <term>Atlas A2 products</term> and <term>Atlas A3 products</term> contains only complete KV blocks within the current visible range and does not contain implicit trailing blocks.
- The output `metadataOptional` is opaque data fixed to INT32 with the shape (1024,). It must not be parsed, modified, or reused across different inputs and attributes.
- For the complete constraints, see [aclnnGenericBlockSparseAttentionMetadata](docs/aclnnGenericBlockSparseAttentionMetadata_en.md).

### Product-Specific Constraints

- Non-quantized FLOAT16/BFLOAT16 scenario of <term>Ascend 950PR&950DT products</term>: `headDim` supports any value from 1 to 256; `layoutQ/layoutKv` supports `TND/PA_BBND`, `TND/PA_BNBD`, `TND/TND`, `BSND/BSND`, `BNSD/BNSD`, `BSND/PA_BBND`, `BNSD/PA_BBND`, `BSND/PA_BNBD`, and `BNSD/PA_BNBD`, and other combinations are not supported; `maskType` supports 0 (non-causal) and 1 (causal); `residualBlockMode` supports 0 or 1.
- Non-quantized FLOAT16/BFLOAT16 scenario of <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `headDim` supports any value from 1 to 256; `layoutQ/layoutKv` supports `TND/PA_BBND` and `TND/TND`; `blockShape=[1,Y]`, where Y supports 1, 2, 4, 8, 16, 32, 64, and 128; `maskType` supports 0 (non-causal) and 1 (causal); `residualBlockMode` supports 0 or 1. For TND KV, `cuSeqLengthsKvOptional` must be passed in; in causal mode, each batch must satisfy that the actual KV length is not less than the actual Q length.
- FP8 quantized scenario of <term>Ascend 950PR&950DT products</term>: `headDim` supports only 128, `layoutQ/layoutKv` supports only `TND/PA_BBND`, `maskType=1`, `residualBlockMode=0`, and `blockShape=[1,128]`.
- Capacity of the last dimension of the sparse index: from 1 to INT32_MAX for <term>Atlas A2 products</term> and <term>Atlas A3 products</term> in the non-quantized scenario, from 1 to INT32_MAX for <term>Ascend 950PR&950DT products</term> in the non-quantized FLOAT16/BFLOAT16 scenario, and from 1 to 256 for <term>Ascend 950PR&950DT products</term> in the FP8 quantized scenario. The capacities must all be not less than the maximum value of the valid sparse block counts.
- `quantType`: 0 (non-quantized) or 5 (FP8 quantized) is supported for <term>Ascend 950PR&950DT products</term>; only 0 is supported for <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.
- `softmaxPrecision`: 0 or 1 is supported for <term>Ascend 950PR&950DT products</term> in the non-quantized FLOAT16/BFLOAT16 scenario, and only 1 is supported in the FP8 quantized scenario; on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, 0 or 1 is supported for FLOAT16, and only 0 is supported for BFLOAT16.

## Calling Instructions

| Calling Method | Sample Code | Description |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_generic_block_sparse_attention_metadata](./examples/test_aclnn_generic_block_sparse_attention_metadata.cpp) | Calls the GenericBlockSparseAttentionMetadata operator by using the [aclnnGenericBlockSparseAttentionMetadata](docs/aclnnGenericBlockSparseAttentionMetadata_en.md) two-phase API. |
