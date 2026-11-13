# aclnnGenericBlockSparseAttentionMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:22:58.323Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- **Description**: Generates the `metadataOptional` required for the `aclnnGenericBlockSparseAttention` computation.
- The output `metadataOptional` must be used as the `metadataOptional` input of the primary operator `aclnnGenericBlockSparseAttention` and must be regenerated before each call to the primary operator.
- `metadataOptional` is opaque data. Callers must not parse or modify its content.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize` API to obtain the workspace size and the executor containing the operator computation process, and then call the `aclnnGenericBlockSparseAttentionMetadata` API to execute the computation.

```c++
aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
  const aclTensor   *sparseBlockIdx,
  const aclTensor   *sparseBlockCount,
  const aclTensor   *cuSeqLengthsQOptional,
  const aclTensor   *cuSeqLengthsKvOptional,
  const aclTensor   *sequsedQOptional,
  const aclTensor   *sequsedKvOptional,
  int64_t            maxQSeqLen,
  int64_t            maxKvSeqLen,
  int64_t            numQHeads,
  int64_t            numKvHeads,
  int64_t            headDim,
  const aclIntArray *blockShape,
  const char         *layoutQ,
  const char         *layoutKv,
  int64_t            layoutSparsePattern,
  int64_t            maskType,
  int64_t            quantType,
  int64_t            softmaxPrecision,
  int64_t            winLeft,
  int64_t            winRight,
  int64_t            residualBlockMode,
  bool               isConsistentTopK,
  const aclTensor   *metadataOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionMetadata(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize

- **Parameters**

  See <a href="#baseline-description">Baseline Description</a> for the meaning of shape variables in the table.

  <table style="table-layout: fixed; width: 100%; word-break: break-word; overflow-wrap: anywhere;">
  <colgroup>
  <col style="width: 13%">
  <col style="width: 7%">
  <col style="width: 17%">
  <col style="width: 28%">
  <col style="width: 10%">
  <col style="width: 6%">
  <col style="width: 13%">
  <col style="width: 6%">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sparseBlockIdx</td>
      <td>Input</td>
      <td>Sparse block index array, specifying the KV block indices selected for each Q block.</td>
      <td>The value must be a valid KV block index (blocked by cu storage length); invalid positions can be filled with -1, and valid values must fall within the first sparseBlockCount positions. Detailed reference: <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#layout-correspondence-description">Layout Correspondence Description</a>.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks, maxSparseBlockCount]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>Input</td>
      <td>Number of KV blocks actually selected for each Q block.</td>
      <td>For a detailed reference, see <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#layout-correspondence-description">layout correspondence</a>.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>Input</td>
      <td>Describes the query sequence length corresponding to each batch, stored as prefix sums.</td>
      <td>        Optional input, used for variable-length sequence use cases.
        <ul>
          <li>Must be passed in when layoutQ is TND.</li>
          <li>When layoutQ is BNSD/BSND: if passed in, the operator processes the input based on the actual sequence lengths specified by this input; if nullptr is passed in, the input is processed based on S in the query shape (currently not supported).</li>
          <li>Elements are prefix sums: the 0th element is 0, the last element equals totalQTokens, and each element must be ≥ the previous element; differences between adjacent elements give the <strong>storage length</strong> qStorageLen_i of each batch, and Σ qStorageLen_i = totalQTokens.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>Input</td>
      <td>Describes the key/value sequence length for each batch, stored as prefix sums.</td>
      <td>        Optional input for variable-length sequence tasks.
        <ul>
          <li>Must be passed in when layoutKv is TND.</li>
          <li>Not passed in when layoutKv is PA_BBND/PA_BNBD (must be nullptr).</li>
          <li>When layoutKv is BNSD/BSND: if this input is passed in, the operator processes data based on the actual sequence length specified by this input; if nullptr is passed in, the operator processes data based on S in the shape of key/value (not currently supported).</li>
          <li>Elements are prefix sums: the 0th element is 0, the last element equals the sum of the KV storage lengths of all batches, and each element must be greater than or equal to the previous one.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>Input</td>
      <td>Actual sequence length of the query in each batch.</td>
      <td>        <ul>
          <li>If the actual length is not specified, nullptr can be passed in, indicating that the storage length is the same as that obtained by differencing cuSeqLengthsQOptional.</li>
          <li>When passed in, the shape is (B,), and each element must be ≥0 and ≤ the cu storage length of the corresponding batch. For the dual-length semantics when coexisting with cu, see <a href="#other-constraints">Other Constraints</a>.</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>Input</td>
      <td>Actual sequence length of kv in each batch.</td>
      <td>        <ul>
          <li>Must be passed in when layoutKv is PA_BBND. Shape is (B,), and each element must be ≥ 0.</li>
          <li>When layoutKv is TND: pass in nullptr if the actual length is not specified, indicating that the storage length is the same as that obtained by differencing cuSeqLengthsKvOptional; if passed in, the shape is (B,), and each element must be ≥ 0 and ≤ the cu storage length of the corresponding batch. For dual-length semantics when coexisting with cu, see <a href="#other-constraints">Other Constraints</a>.</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>maxQSeqLen</td>
      <td>Input</td>
      <td>Maximum Sequence Length of Query of the primary operator.</td>
      <td>Must be greater than 0 and consistent with the primary operator and length-type inputs. See Also <a href="#other-constraints">Other Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxKvSeqLen</td>
      <td>Input</td>
      <td>Maximum Sequence Length of Key/Value of the primary operator.</td>
      <td>Must be greater than 0 and consistent with the primary operator and length-type inputs. See Also <a href="#other-constraints">Other Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numQHeads</td>
      <td>Input</td>
      <td>Number of Query heads of the primary operator.</td>
      <td>Must be greater than 0 and consistent with the primary operator. For the Q/KV head count relationship, see <a href="#other-constraints">Other Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKvHeads</td>
      <td>Input</td>
      <td>Number of Key/Value heads of the primary operator.</td>
      <td>Must be greater than 0 and consistent with the primary operator. For the Q/KV head count relationship, see <a href="#other-constraints">Other Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>Input</td>
      <td>Feature dimension of each head of Query, Key, and Value of the Primary Operator.</td>
      <td>Only 128 is supported. The value must be the same as that of the Primary Operator.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>Input</td>
      <td>Represents the array of sparse block shapes.</td>
      <td>Sparse block shape [blockShapeX, blockShapeY]. Non-union: [1,Y]; union (maskType=3): [8,Y] (currently only 8). Must match the main operator. For product scope, see <a href="#other-constraints">Other Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>Input</td>
      <td>Represents the data layout format of the input query.</td>
      <td>Supports "TND", "BSND", and "BNSD". See <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#layout-correspondence-description">Layout Correspondence Description</a> and <a href="#other-constraints">Other Constraints</a> for product scope.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>Input</td>
      <td>Represents the data layout format of the input key and value.</td>
      <td>Supports "TND", "BSND", "BNSD", "PA_BBND", and "PA_BNBD". See <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#layout-correspondence-description">Layout Correspondence Description</a> and <a href="#other-constraints">Other Constraints</a> for product scope.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>Input</td>
      <td>Represents the data layout format of the input sparseBlockIdx and sparseBlockCount.</td>
      <td>Currently only "4" is supported. For a detailed reference, see the <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#layout-correspondence-description">Layout Correspondence Description</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>Input</td>
      <td>Specifies the mask type in the attention computation.</td>
      <td>        Value: 0-5. For details, see <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#mask-description">Mask Description</a>.<br>
        Supports 0 (non-causal) and 1 (causal); Ascend 950PR&amp;950DT and non-quantized Atlas A2/A3 also support 3 (union). For product scope, see <a href="#other-constraints">Other Constraints</a>.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>Input</td>
      <td>Represents the quantization method used.</td>
      <td>Value: 0-5. See <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#quantization-related-description">Quantization-Related Description</a> for the complete configuration:<br>
        Currently available: 0; 5 is optional on Ascend 950PR&950DT products. Values 1-4 fail validation.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>Input</td>
      <td>Precision level used for Softmax computation.</td>
      <td>        Controls the data type used for computation in the online softmax and rescale stages. Value: 0 or 1:
        <ul>
          <li>0: fp32 is used for both online softmax and rescale. Suitable for use cases that require high computation precision.</li>
          <li>1: Mixed precision; online softmax uses fp16/bf16 (the same as attentionOut) and rescale uses fp32. Numeric overflow may occur in the online softmax stage.</li>
        </ul>
        Chip constraints: in non-Union scenarios (maskType=0/1) of Ascend 950PR&amp;950DT, non-quant FLOAT16/BFLOAT16 supports 0 or 1 and FP8 supports only 1; in Union scenarios (maskType=3), both non-quant and FP8 support only 0; on Atlas A2/A3, FLOAT16 supports 0 or 1 and BFLOAT16 supports only 0.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>Input</td>
      <td>Number of tokens that the sliding window needs to include forward in the sliding-window attention use case.</td>
      <td>Used for sliding-window attention; must be -1 when disabled, and must be used together with maskType. See <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#mask-description">Mask Description</a>. Currently, only -1 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>Input</td>
      <td>Number of tokens that the sliding window needs to include backward in the sliding window attention use case.</td>
      <td>Used for sliding window attention; must be -1 when not enabled and used together with maskType. See <a href="../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#mask-description">Mask Description</a>. Currently, only -1 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualBlockMode</td>
      <td>Input</td>
      <td>Indicates the state of the trailing incomplete block after the KV sequence is sparsified in units of blockShapeY.</td>
      <td>Supports 0 or 1 (union supports only 0). For 0, whether the trailing incomplete block participates in the computation is determined by the value passed in through sparseBlockIdx. For 1, the trailing incomplete block must participate in the computation but is definitely excluded from sparseBlockIdx.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isConsistentTopK</td>
      <td>Input</td>
      <td>Whether the maximum number of KV blocks selected by each Q block is the same within the same batch and head when the preceding operator performs sparse block selection.</td>
      <td>Only 0 or 1 is supported. 0 indicates inconsistency, and 1 indicates consistency.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>Output</td>
      <td>Core splitting result generated by the AICPU operator.</td>
      <td>Mandatory. The caller must pre-allocate a Device-side INT32 tensor with shape (1024,). Its content is in an opaque format and must not be parsed, modified, or reused across different inputs and attributes.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied for on the Device.</td>
      <td>Currently returns 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 280px">
  <col style="width: 90px">
  <col style="width: 785px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>A required tensor is not passed in, or the tensor data type, shape, layout, blockShape, head count relationship, maskType, quantType, softmaxPrecision, window attributes, etc. do not meet the current constraints.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>Failed to create aclOpExecutor inside the API.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>workspaceSize or executor is null, or a null pointer exception occurs when the API internally executes Contiguous or registers AICPU tasks.</td>
    </tr>
  </tbody>
  </table>

## aclnnGenericBlockSparseAttentionMetadata

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device. Currently workspaceSize is 0, so nullptr can be passed in.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

### Baseline Description

<table style="undefined;table-layout: fixed; width: 1155px">
<colgroup>
  <col style="width: 260px">
  <col style="width: 895px">
</colgroup>
<thead>
  <tr>
    <th>Name</th>
    <th>Meaning</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>B</td>
    <td>Batch Size, the batch size of input samples.</td>
  </tr>
  <tr>
    <td>numQHeads</td>
    <td>Head Num of the query of the primary operator.</td>
  </tr>
  <tr>
    <td>numKvHeads</td>
    <td>Head Num of the key/value of the primary operator.</td>
  </tr>
  <tr>
    <td>headDim</td>
    <td>Head Dim of query, key, and value of the primary operator.</td>
  </tr>
  <tr>
    <td>qStorageLen_i</td>
    <td>Query storage length of the i-th batch, obtained by differencing adjacent elements of cuSeqLengthsQOptional.</td>
  </tr>
  <tr>
    <td>totalQBlocks</td>
    <td>Total number of Q blocks after partitioning by storage length, i.e.$\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$</td>
  </tr>
  <tr>
    <td>maxSparseBlockCount</td>
    <td>Last dimension of sparseBlockIdx. It must be no less than the maximum of all elements in sparseBlockCount.</td>
  </tr>
</tbody>
</table>

### Other Constraints

- This API must be used together with `aclnnGenericBlockSparseAttention`. Shared tensors and attributes must be exactly the same as those of the subsequently called primary operator, and `metadataOptional` must be regenerated before each call to the primary operator. For the complete constraints of the primary operator, see [aclnnGenericBlockSparseAttention](../../generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention_en.md#constraints).
- Layout combinations and product scope match the main operator: `layoutSparsePattern=4`, `winLeft=winRight=-1`; non-union uses `blockShapeX=1`, union uses `blockShapeX=8`. Non-quant Ascend 950PR&amp;950DT: `maskType` supports 0/1/3; non-union uses `blockShape=[1,Y]` (`1≤Y≤128`), union uses `blockShape=[8,Y]`; `residualBlockMode` supports 0 or 1 (union supports only 0). Atlas A2/A3: `maskType` supports 0/1/3; non-union uses `blockShape=[1,Y]` (Y in {1,2,4,8,16,32,64,128}), union uses `blockShape=[8,Y]`. Regular FP8: `headDim=128`, `layoutQ/layoutKv=TND/PA_BBND`, `maskType=1`, `residualBlockMode=0`, `blockShape=[1,Y]` (`1≤Y≤128`). Union (maskType=3) supports FP16/BF16 non-quant on Atlas A2/A3 (`blockShape=[8,Y]` with power-of-two Y, `residualBlockMode=0`, INT32 exactMask) and FP16/BF16/FP8 on Ascend 950 with `blockShape=[8,Y]` (`1≤Y≤128`), `residualBlockMode=0`; FP8 Union additionally requires `headDim=128`, and for paged KV `Y` must divide the PA page size 128. Union does not support `quantType=6`.
- The shape of `sparseBlockIdx` is `[numKvHeads, totalQBlocks, maxSparseBlockCount]`, and the shape of `sparseBlockCount` is `[numKvHeads, totalQBlocks]`. `maxSparseBlockCount` must be greater than 0 and not less than the maximum of all elements in `sparseBlockCount`. Capacity upper limit is INT32_MAX for non-quant and FP8 paths.
- For TND, `cuSeqLengthsQOptional` must be passed in; `sequsedQOptional` can be nullptr. When sequsedQ is passed in, each element must be within the range `[0, corresponding batch storage length]`. Sparse block partitioning is still calculated based on the storage length described by the cu prefix sum.
- When layoutKv is PA_BBND/PA_BNBD, `sequsedKvOptional` must be passed in and `cuSeqLengthsKvOptional` must not be passed in. When layoutKv is TND, `sequsedKvOptional` can be used together with `cuSeqLengthsKvOptional`, and the dual-length semantics are the same as those on the Q side.
- `numQHeads >= numKvHeads` and `numQHeads % numKvHeads == 0` must be satisfied, and `groupSize = numQHeads / numKvHeads` must not exceed 128.
- The output `metadataOptional` is fixed to INT32 with a shape of `(1024,)`, and must not be parsed, modified, or reused across different inputs and attributes.

## Example

The following sample code is for reference only. For details, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

aclnnStatus RunMetadata(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                        const aclTensor *cuSeqLengthsQOptional, const aclTensor *sequsedKvOptional,
                        aclTensor *metadataOptional, aclrtStream stream)
{
    const int64_t blockShapeData[] = {1, 128};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    if (blockShape == nullptr) {
        return ACL_ERROR_BAD_ALLOC;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, nullptr, nullptr, sequsedKvOptional, 16, 2048,
        32, 8, 128, blockShape, "TND", "PA_BBND", 4, 1, 0, 0, -1, -1, 0, 0, metadataOptional, &workspaceSize,
        &executor);
    aclDestroyIntArray(blockShape);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnGenericBlockSparseAttentionMetadata(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
