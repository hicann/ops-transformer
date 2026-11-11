# aclnnGenericBlockSparseAttentionGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:08:35.517Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: aclnnGenericBlockSparseAttentionGrad is the backward computation operator for generic block sparse attention. Based on the indexes defined by `sparseBlockIdx`/`sparseBlockCount` (the sparse block index table), it computes and propagates gradients only on the selected KV blocks and supports dynamic, variable-length block sparse patterns. Before calling this API, call `aclnnGenericBlockSparseAttentionGradMetadata` to generate the kernel-splitting `metadataOptional`.
- Formula:

$$
P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
$$

$$
dP = dO @ selectedV^{T}
$$

$$
dS = P \odot (dP - SoftmaxGrad(dO, O))
$$

$$
dQ = dS @ selectedK \cdot scale
$$

$$
dK = dS^{T} @ Q \cdot scale
$$

$$
dV = P^{T} @ dO
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnGenericBlockSparseAttentionGradGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnGenericBlockSparseAttentionGrad" API to perform the computation.

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *dout,
    const aclTensor *out,
    const aclTensor *lse,
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    double scaleValue,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    aclTensor *dQuery,
    aclTensor *dKey,
    aclTensor *dValue,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
```

## aclnnGenericBlockSparseAttentionGradGetWorkspaceSize

- **Parameters**

  <table style="undefined; table-layout: fixed; width: 1567px">
          <colgroup>
              <col style="width: 220px">
              <col style="width: 120px">
              <col style="width: 300px">
              <col style="width: 400px">
              <col style="width: 212px">
              <col style="width: 100px">
              <col style="width: 190px">
              <col style="width: 145px">
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
                  <td>query</td>
                  <td>Input</td>
                  <td>Input Q of the attention structure.</td>
                  <td>                      <ul>
                          <li>The data type must be consistent with key, value, dout, out, dQuery, dKey, and dValue.</li>
                          <li>Empty tensors are not supported.</li>
                      </ul>
                  </td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D), (T1,N1,D), (B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>key</td>
                  <td>Input</td>
                  <td>Input K of the attention structure.</td>
                  <td>                      <ul>
                          <li>Data type is consistent with query.</li>
                          <li>Empty tensors are not supported.</li>
                      </ul>
                  </td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D), (T2,N2,D), (B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>value</td>
                  <td>Input</td>
                  <td>Input V of the attention structure.</td>
                  <td>                      <ul>
                          <li>Shape is the same as key.</li>
                          <li>Empty tensors are not supported.</li>
                      </ul>
                  </td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D), (T2,N2,D), (B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>dout</td>
                  <td>Input</td>
                  <td>Gradient of the attention output matrix.</td>
                  <td>                      <ul>
                          <li>B, S1, and N1 are consistent with query; D is 128.</li>
                          <li>Empty Tensor Not Supported.</li>
                      </ul>
                  </td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D), (T1,N1,D), (B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>out</td>
                  <td>Input</td>
                  <td>Attention output matrix.</td>
                  <td>                      <ul>
                          <li>Shape is consistent with dout.</li>
                          <li>Empty Tensor Not Supported.</li>
                      </ul>
                  </td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D), (T1,N1,D), (B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>lse</td>
                  <td>Input</td>
                  <td>lse output of forward attention computation, with layout consistent with the forward output.</td>
                  <td>                      <ul>
                          <li>The data type is FLOAT32.</li>
                          <li>Empty tensors are not supported.</li>
                      </ul>
                  </td>
                  <td>FLOAT32</td>
                  <td>ND</td>
                  <td>TND: (T1,N1,1); BNSD: (B,N1,S1,1); BSND: (B,S1,N1,1) or (B,N1,S1,1). See Constraints for details.</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>sparseBlockIdx</td>
                  <td>Input</td>
                  <td>Array of sparse block indices, specifying the Q block/token index selected for each KV block.</td>
                  <td>                      <ul>
                          <li>The Q sparse pattern corresponding to each KVHead in the same group is consistent (isPackedGQA=1).</li>
                          <li>The maxS1 of the 4th dimension should be ≥ the maximum value of all elements in sparseBlockCount.</li>
                          <li>Empty tensors are not supported.</li>
                      </ul>
                  </td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>TND: (B,N2,ceilDiv(maxS2,blockShapeY),maxS1); BNSD/BSND: (B,N2,ceilDiv(S2,blockShapeY),S1)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>sparseBlockCount</td>
                  <td>Input</td>
                  <td>Specifies the number of Q actually selected for each KV block.</td>
                  <td>Empty tensors are not supported.</td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(B,N2,ceilDiv(maxS2,blockShapeY)) or (B,N2,ceilDiv(S2,blockShapeY))</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>metadataOptional</td>
                  <td>Optional Input</td>
                  <td>Kernel splitting information generated by aclnnGenericBlockSparseAttentionGradMetadata.</td>
                  <td>
                      <ul>
                          <li>Required when layoutSparsePattern=1; must be nullptr when layoutSparsePattern=0. See <a href="#Constraints">Constraints</a>.</li>
                          <li>When layoutSparsePattern=1, supported metaSize lengths differ by product; see the notes below the table.</li>
                      </ul>
                  </td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(x,)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>attenMaskOptional</td>
                  <td>Optional Input</td>
                  <td>atten_mask in the formula. Takes effect when overlaid with the sparse pattern.</td>
                  <td>Currently not supported. Pass nullptr.</td>
                  <td>BOOL</td>
                  <td>ND</td>
                  <td>-</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>cuSeqLengthsQOptional</td>
                  <td>Optional Input</td>
                  <td>Prefix sum of query sequence lengths corresponding to each batch.</td>
                  <td>Must be configured when layoutQ is "TND"; pass nullptr when layoutQ is "BNSD" or "BSND".</td>
                  <td>INT64</td>
                  <td>ND</td>
                  <td>(B+1,)</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>cuSeqLengthsKvOptional</td>
                  <td>Optional Input</td>
                  <td>Prefix sum of key/value sequence lengths corresponding to each batch.</td>
                  <td>Must be configured when layoutKv is "TND"; pass nullptr when layoutKv is "BNSD" or "BSND".</td>
                  <td>INT64</td>
                  <td>ND</td>
                  <td>(B+1,)</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>sequsedQOptional</td>
                  <td>Optional Input</td>
                  <td>Actual sequence length of the query in each batch. Takes effect only when layoutQ is "TND"; pass nullptr when layoutQ is "BNSD" or "BSND", and the actual length is obtained from the S dimension of query.</td>
                  <td>The length is B.</td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(B,)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>sequsedKvOptional</td>
                  <td>Optional Input</td>
                  <td>Actual sequence length of kv in each batch. Takes effect when layoutKv is "TND"; pass nullptr when layoutKv is "BNSD" or "BSND", and the actual length is taken from the S dimension of key/value.</td>
                  <td>Length: B.</td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(B,)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>blockShape</td>
                  <td>Input</td>
                  <td>Sparse block shape array.</td>
                  <td>                      <ul>
                          <li>Contains two elements [blockShapeX, blockShapeY].</li>
                          <li>blockShapeX currently only supports 1.</li>
                          <li>blockShapeY must be ≥ 128 and aligned to 64. Cube/Softmax splits each sparse block along S2 by baseN=128 (LSE Softmax, numerically equivalent to the whole block).</li>
                      </ul>
                  </td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>isPackedGQA</td>
                  <td>Input</td>
                  <td>Whether qHeads in the same group share the same sparse pattern.</td>
                  <td>Currently Only Supports 1. Not shared across batches.</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>layoutQ</td>
                  <td>Input</td>
                  <td>Data layout format of the input query.</td>
                  <td>Currently Supports "TND", "BNSD", "BSND".</td>
                  <td>STRING</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>layoutKv</td>
                  <td>Input</td>
                  <td>Data layout format of the input key and value.</td>
                  <td>Currently Supports "TND", "BNSD", "BSND". Must be consistent with layoutQ.</td>
                  <td>STRING</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>scaleValue</td>
                  <td>Input</td>
                  <td>scale in the formula, representing the scaling factor.</td>
                  <td>Must be a finite value (not NaN/Inf). Suggested value: reciprocal of the square root of d in the formula.</td>
                  <td>FLOAT</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>maskType</td>
                  <td>Input</td>
                  <td>Mask type in the attention computation.</td>
                  <td>For supported mask modes, see <a href="#constraints">Constraints</a>. Currently only supports 1.</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>softmaxPrecision</td>
                  <td>Input</td>
                  <td>Precision level for the Softmax computation.</td>
                  <td>                      <ul>
                          <li>Currently Only Supports 0.</li>
                          <li>0: Both online softmax and rescale use fp32.</li>
                      </ul>
                  </td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>winLeft</td>
                  <td>Input</td>
                  <td>Number of tokens the sliding window needs to include backward in the sliding window attention use case.</td>
                  <td>Must be -1 when disabled, used together with maskType and mask.</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>winRight</td>
                  <td>Input</td>
                  <td>Number of tokens that the sliding window needs to contain backward in sliding window attention use cases.</td>
                  <td>Must be -1 when disabled, used together with maskType and mask.</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>dQuery</td>
                  <td>Output</td>
                  <td>Gradient of query.</td>
                  <td>Data type and shape are consistent with query.</td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D), (T1,N1,D), (B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>dKey</td>
                  <td>Output</td>
                  <td>Gradient of key.</td>
                  <td>Data type and shape are consistent with key.</td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D), (T2,N2,D), (B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>dValue</td>
                  <td>Output</td>
                  <td>Gradient of value.</td>
                  <td>Data type and shape are consistent with those of value.</td>
                  <td>FLOAT16, BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D), (T2,N2,D), (B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>workspaceSize</td>
                  <td>Output</td>
                  <td>Returns the size of the workspace that the user needs to allocate on the Device.</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>executor</td>
                  <td>Output</td>
                  <td>Op executor, containing the operator computation flow.</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
          </tbody>
      </table>

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
    metadataOptional length ≥ 198.
  <!-- end id10 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: metadataOptional length ≥ 80+B×N1×J×4 (number of int32 elements).
  <!-- end id9 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1100px">
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 720px">
      </colgroup>
      <thead>
        <tr>
          <th>Return Value</th>
          <th>Error Code</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>A required parameter or output is a null pointer; metadataOptional is not passed in when layoutSparsePattern=1.</td>
        </tr>
        <tr>
          <td class="merged-cell" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="2">161002</td>
          <td>The data type, data format, or value of inputs, outputs, or attributes is out of the supported range; layoutQ is inconsistent with layoutKv.</td>
        </tr>
        <tr>
          <td>isPackedGQA!=1; winLeft/Right!=-1; blockShape does not meet the constraints; cuSeqLengthsQ/KvOptional is not provided when layout is "TND"; scaleValue is not finite in float32.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>An exception occurs when the API calls the npu runtime API for memory.</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnGenericBlockSparseAttentionGrad

- **Parameters**

  <table><thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by calling the first-phase API aclnnGenericBlockSparseAttentionGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGenericBlockSparseAttentionGrad defaults to a non-deterministic implementation. A deterministic implementation is not currently supported, and the deterministic computation configuration does not take effect.
- Call [aclnnGenericBlockSparseAttentionGradMetadata](../../generic_block_sparse_attention_grad_metadata/docs/aclnnGenericBlockSparseAttentionGradMetadata_en.md) to generate `metadataOptional` before calling this API.
- The data types of query, key, value, dout, out, dQuery, dKey, and dValue should be kept consistent; FLOAT16 and BFLOAT16 are supported.
- The data type of lse should be FLOAT32.
- The data types of sparseBlockIdx, sparseBlockCount, sequsedQOptional, and sequsedKvOptional should be INT32.
- The data types of cuSeqLengthsQOptional and cuSeqLengthsKvOptional should be INT64; the data type of metadataOptional should be INT32.
- layoutQ and layoutKv currently support TND, BNSD, and BSND, and must be kept consistent.
- cuSeqLengthsQOptional is required when layoutQ is TND; cuSeqLengthsKvOptional is required when layoutKv is TND.
- sequsedQOptional/sequsedKvOptional takes effect only when TND is used; for BNSD/BSND, nullptr must be passed, and the actual sequence length is obtained from the S dimension of Q/K.
- HeadDim is fixed to 128; the value range of N1/N2 is [1, 128], with N1 > N2 and N1 % N2 == 0.
- blockShape: blockShapeX supports only 1; blockShapeY must be ≥128 and a multiple of 64 (Cube splits S2 of each sparse block with baseN=128); isPackedGQA currently only supports 1; maskType currently only supports 1; softmaxPrecision currently only supports 0.
- winLeft and winRight must be -1 when disabled; attenMaskOptional should currently be set to nullptr.
- Softmax LSE: TND is `(T1,N1,1)`, BNSD is `(B,N1,S1,1)`; for BSND, Ascend 950 (A5) requires `(B,N1,S1,1)`, and A2/A3 use `(B,S1,N1,1)`.
- scaleValue must be finite in float32 (not NaN/Inf, and still finite after cast to float); this applies to both layoutSparsePattern=0 and layoutSparsePattern=1.
- The 4th dimension maxS1 of `sparseBlockIdx` should be ≥ the maximum value of all elements in `sparseBlockCount`.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // BNSD smoke case
    const int64_t B = 1;
    const int64_t N1 = 1;
    const int64_t N2 = 1;
    const int64_t S1 = 128;
    const int64_t S2 = 128;
    const int64_t D = 128;
    const int64_t blockX = 1;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t maskType = 1;
    const double scaleValue = 1.0 / std::sqrt(static_cast<double>(D));
    // metadata size = 80 + B * N1 * J * 4
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> qShape = {B, N1, S1, D};
    std::vector<int64_t> kvShape = {B, N2, S2, D};
    std::vector<int64_t> lseShape = {B, N1, S1};
    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x2E66); // ~0.1 fp16
    std::vector<uint16_t> kHost(GetShapeSize(kvShape), 0x2E66);
    std::vector<uint16_t> vHost(GetShapeSize(kvShape), 0x2E66);
    std::vector<uint16_t> doutHost(GetShapeSize(qShape), 0x211E); // ~0.01 fp16
    std::vector<uint16_t> outHost(GetShapeSize(qShape), 0x2E66);
    std::vector<float> lseHost(GetShapeSize(lseShape), 5.0f);
    std::vector<int32_t> idxHost(GetShapeSize(idxShape), -1);
    std::vector<int32_t> cntHost(GetShapeSize(cntShape), 0);
    std::vector<int32_t> metaHost(metaSize, 0);
    std::vector<uint16_t> dqHost(GetShapeSize(qShape), 0);
    std::vector<uint16_t> dkHost(GetShapeSize(kvShape), 0);
    std::vector<uint16_t> dvHost(GetShapeSize(kvShape), 0);

    // A single KV block selects all Q tokens
    for (int64_t q = 0; q < S1; ++q) {
        idxHost[q] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr, *lseAddr = nullptr;
    void *idxAddr = nullptr, *cntAddr = nullptr, *metaAddr = nullptr;
    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;

    aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
    aclTensor *dout = nullptr, *out = nullptr, *lse = nullptr;
    aclTensor *idx = nullptr, *cnt = nullptr, *metadata = nullptr;
    aclTensor *dq = nullptr, *dk = nullptr, *dv = nullptr;

    ret = CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHost, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHost, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &dout);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, qShape, &outAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lseHost, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHost, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dkHost, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dvHost, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t blockShapeData[] = {blockX, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char qLayout[] = "BNSD";
    char kvLayout[] = "BNSD";

    // 1) Metadata
    uint64_t metaWsSize = 0;
    aclOpExecutor *metaExecutor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, qLayout, kvLayout, maskType, 0,
        -1, -1, metadata, &metaWsSize, &metaExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *metaWs = nullptr;
    if (metaWsSize > 0) {
        ret = aclrtMalloc(&metaWs, metaWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(metaWs, metaWsSize, metaExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Metadata failed. ERROR: %d\n", ret); return ret);

    // 2) Grad
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
        q, k, v, dout, out, lse, idx, cnt, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, blockShape, 1,
        qLayout, kvLayout, scaleValue, maskType, 0, -1, -1, dq, dk, dv, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionGrad failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("GenericBlockSparseAttentionGrad finished successfully.\n");

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(dout);
    aclDestroyTensor(out);
    aclDestroyTensor(lse);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dv);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(vAddr);
    aclrtFree(doutAddr);
    aclrtFree(outAddr);
    aclrtFree(lseAddr);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    aclrtFree(dqAddr);
    aclrtFree(dkAddr);
    aclrtFree(dvAddr);
    if (metaWsSize > 0) {
        aclrtFree(metaWs);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
