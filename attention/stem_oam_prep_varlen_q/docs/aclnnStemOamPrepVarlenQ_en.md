# aclnnStemOamPrepVarlenQ

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:40:57.283Z -->

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

- Description: Performs the Q-side preprocessing computation for Stem OAM block-sparse attention. Converts a varlen Q tensor stored in token-concatenated layout ([total_tokens, H_q, D]) into flattened qFlat output grouped by stem block (128 tokens) for subsequent OAM score computation.
- Formula:

  Stage 1 Scale Fusion:
  $$q\_scale[b,h,pos] = qScale[cu\_seqlens\_q[b]+pos,\ h]$$ (pos is the token index within the batch sequence)

  Stage 2 Varlen Positioning:
  $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

  Stage 3 Weighted Group Sum (natural order, NO flip):
  $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times qScale[cu\_seqlens\_q[b]+position(qb,r,g),\ h]$$ (where position(qb,r,g) = qb*stemBlockSize + r*stemStride + g; positions beyond qSeqLens[b] are zero-filled by the kernel and contribute zero)

  Stage 4 Flatten:
  $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

  Stage 5 Cast output:
  $$qflat\_out = qflat.to(\text{BF16})$$

- Key feature: The stride dimension on the Q side is in natural order (no flip), which differs from the flip processing on the K side.

## Function Prototype

Each operator is split into a two-phase API. You must first call the "aclnnStemOamPrepVarlenQGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnStemOamPrepVarlenQ" API to perform computation.

```cpp
aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(
  const aclTensor   *q,
  const aclIntArray *qSeqLens,
  const aclIntArray *cuSeqLensQ,
  const aclTensor   *qScale,
  int64_t            stemBlockSize,
  int64_t            stemStride,
  aclTensor         *qFlat,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnStemOamPrepVarlenQ(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnStemOamPrepVarlenQGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>q (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the varlen Q tensor, where tokens of all batches are stored in concatenation. Corresponds to q in the formula.</td>
      <td><ul><li>Empty tensors are supported. For an empty tensor, the API returns directly and workspaceSize is 0.</li><li>The last dimension must be equal to 128.</li><li>Only FLOAT8_E4M3FN is supported for the data type.</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3 ([total_tokens, H_q, D]), D=128</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qSeqLens (aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the Q sequence length of each batch. Corresponds to q_seq_lens in the formula.</td>
      <td><ul><li>The length is equal to batch, and the batch value range is [0, 1024].</li><li>Each value is ≥ 0.</li><li>If the length is 0, it is processed as an empty tensor and directly returned.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>1 ([batch])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cuSeqLensQ (aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the cumulative sequence length offset of Q, used for varlen indexing. Corresponds to cu_seqlens_q in the formula.</td>
      <td><ul><li>The length is batch+1.</li><li>cuSeqLensQ[0] must be 0.</li><li>cuSeqLensQ[batch] must be equal to total_tokens.</li><li>The prefix-sum relation cuSeqLensQ[b+1] - cuSeqLensQ[b] equals qSeqLens[b] (equal adjacent values are allowed for zero-length sequences).</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>1 ([batch+1])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qScale (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the per-token scale factor of Q, corresponding to qscale in the formula.</td>
      <td><ul><li>Required. Must not be nullptr when q is FLOAT8_E4M3FN.</li><li>The shape is [total_tokens, H_q].</li><li>The data type must be FLOAT.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2 ([total_tokens, H_q])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stem block size, corresponding to B in the formula. Controls the number of tokens in each stem block and determines the grouping granularity of Q Processing.</td>
      <td><ul><li>Only 128 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemStride (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stem stride size, corresponding to S in the formula. Controls the number of tokens in a stride group within a stem block, determining the dimension granularity of qFlat.</td>
      <td><ul><li>Only 16 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qFlat (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the flattened Q output, used for OAM score computation. Corresponds to qflat_out in the formula.</td>
      <td><ul><li>The data type is fixed to BFLOAT16.</li><li>The shape is [batch, H_q, max_Qb, kflat_dim], where max_Qb=ceil(max(qSeqLens)/stemBlockSize) and kflat_dim=stemStride×D=16×128=2048.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4 ([batch, H_q, max_Qb, kflat_dim])</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the Device side-applied workspace.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
      <td>q, qSeqLens, cuSeqLensQ, qFlat, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of q is not FLOAT8_E4M3FN.</td>
    </tr>
    <tr>
      <td>qScale is nullptr when q is FP8.</td>
    </tr>
    <tr>
      <td>The data type of qScale is not FLOAT.</td>
    </tr>
    <tr>
      <td>The data type of qFlat is not BFLOAT16.</td>
    </tr>
    <tr>
      <td>The dimension of q is not 3 or the last dimension is not equal to 128; the dimension of qScale is not 2; the dimension of qFlat is not 4.</td>
    </tr>
    <tr>
      <td>The length of qSeqLens is greater than 1024; the length of cuSeqLensQ is not equal to batch+1, cuSeqLensQ[0] is not 0, cuSeqLensQ[b+1] - cuSeqLensQ[b] is not equal to qSeqLens[b], or cuSeqLensQ[batch] is not equal to total_tokens.</td>
    </tr>
    <tr>
      <td>stemBlockSize is not 128 or stemStride is not 16.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>Internal error: the Tiling computation fails.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_FIND_KERNEL_ERROR</td>
      <td>561003</td>
      <td>Internal error: the kernel binary fails to be found.</td>
    </tr>
  </tbody></table>

## aclnnStemOamPrepVarlenQ

- **Parameters**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>Input</td><td>Memory address of the workspace side-applied on the Device.</td></tr>
    <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnStemOamPrepVarlenQGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>Op executor, containing the operator computation flow.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Specifies the Stream where the task is executed.</td></tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnStemOamPrepVarlenQ defaults to a deterministic implementation.
- The last dimension D of q must be equal to 128.
- Only 128 is supported for stemBlockSize, and only 16 for stemStride. Derived values: R = stemBlockSize / stemStride = 8, kflat_dim = stemStride × D = 2048.
- qFlat output dimension kflat_dim = stemStride × D = 16 × 128 = 2048.
- The stride dimension on the Q side is in natural order (g ∈ [0, S-1]) and is not flipped (unlike the K-side processing).
- When a batch value in qSeqLens is 0, the qFlat output corresponding to that batch is filled with zeros.
- q supports only the FLOAT8_E4M3FN data type. qScale is mandatory and its data type is FLOAT. The qFlat output is fixed to BFLOAT16.
- Empty tensor input is supported: when q is empty or the length of qSeqLens is 0, the API returns directly and workspaceSize is 0.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<term>Ascend 950PR&950DT products</term>:

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_stem_oam_prep_varlen_q.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. Initialize Device and Stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Create input tensor
    int64_t totalTokens = 256;
    int64_t numHeadQ = 32;
    int64_t dimQk = 128;
    int64_t batch = 2;
    int64_t stemBlockSize = 128;
    int64_t stemStride = 16;
    int64_t kflatDim = stemStride * dimQk;

    std::vector<int64_t> qShape = {totalTokens, numHeadQ, dimQk};
    std::vector<int64_t> qScaleShape = {totalTokens, numHeadQ};
    std::vector<int64_t> qFlatShape = {batch, numHeadQ, 1, kflatDim};

    void *qDeviceAddr = nullptr;
    void *qScaleDeviceAddr = nullptr;
    void *qFlatDeviceAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *qScale = nullptr;
    aclTensor *qFlat = nullptr;

    std::vector<uint8_t> hostQ(GetShapeSize(qShape), 1);
    std::vector<float> hostQScale(GetShapeSize(qScaleShape), 1.0f);
    std::vector<uint16_t> hostQFlat(GetShapeSize(qFlatShape), 0);

    ret = CreateAclTensor(hostQ, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &q,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostQScale, qScaleShape, &qScaleDeviceAddr, aclDataType::ACL_FLOAT, &qScale,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostQFlat, qFlatShape, &qFlatDeviceAddr, aclDataType::ACL_BF16, &qFlat,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t qSeqLensData[] = {128, 128};
    int64_t cuSeqLensQData[] = {0, 128, 256};
    aclIntArray *qSeqLens = aclCreateIntArray(qSeqLensData, 2);
    aclIntArray *cuSeqLensQ = aclCreateIntArray(cuSeqLensQData, 3);

    // 3. Obtain the workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnStemOamPrepVarlenQGetWorkspaceSize(q, qSeqLens, cuSeqLensQ, qScale, stemBlockSize, stemStride,
                                                  qFlat, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepVarlenQGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 4. Allocate the workspace and perform the computation.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnStemOamPrepVarlenQ(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepVarlenQ failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Release resources.
    aclDestroyTensor(q);
    aclDestroyTensor(qScale);
    aclDestroyTensor(qFlat);
    aclDestroyIntArray(qSeqLens);
    aclDestroyIntArray(cuSeqLensQ);
    aclrtFree(qDeviceAddr);
    aclrtFree(qScaleDeviceAddr);
    aclrtFree(qFlatDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
