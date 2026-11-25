# aclnnLightningIndexerV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:08:04.569Z -->

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

- Description: LightningIndexerV2 obtains the top-k positions corresponding to each token based on a series of operations.

- Formula:

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

- The computation process is as follows:

  1. Multiply the input parameter `q` ($Q_{index}\in\R^{g\times d}$) corresponding to a token by the given context `k` ($K_{index}\in\R^{S_{k}\times d}$) to obtain the relevance.
  2. After filtering invalid negative relevance signals through the activation function $ReLU$, obtain the relevance score vector of the current token against all preceding tokens.
  3. After multiplying it with the weight coefficient `w` ($W$), select the top $Top-k$ index values along the g direction to obtain the output $sparseIndices$, and output the corresponding $sparseValues$ as the input of Attention.

## Parameters

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnLightningIndexerV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnLightningIndexerV2" API to perform the computation.

```Cpp
aclnnStatus aclnnLightningIndexerV2GetWorkspaceSize(
    const aclTensor     *q,
    const aclTensor     *k,
    const aclTensor     *w,
    const aclTensor     *cuSeqlensQOptional,
    const aclTensor     *cuSeqlensKOptional,
    const aclTensor     *sequsedQOptional,
    const aclTensor     *sequsedKOptional,
    const aclTensor     *cmpResidualKOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *outputIdxOffsetOptional,
    const aclTensor     *metadataOptional,
    int64_t              topk,
    int64_t              maxSeqlenQ,
    char                *layoutQOptional,
    char                *layoutKOptional,
    int64_t              maskMode,
    int64_t              cmpRatio,
    int64_t              returnValue,
    const aclTensor     *sparseIndices,
    const aclTensor     *sparseValues,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnLightningIndexerV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnLightningIndexerV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 240px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 233px">
  <col style="width: 119px">
  <col style="width: 215px">
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
    <td>Input Q in the formula.</td>
    <td>Empty tensor not supported.</td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
    <td><ul><li>When layoutQOptional is BSND, the shape is (B,S1,N1,D).</li><li>When layoutQOptional is TND, the shape is (T1,N1,D).</li><li>Empty tensor not supported.</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>k (aclTensor*)</td>
    <td>Input</td>
    <td>Input K in the formula.</td>
    <td><ul><li>Empty tensors are not supported.</li><li>block_num is the total number of blocks in PageAttention, and block_size is the number of tokens in a block.</li></ul></td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
    <td><ul><li>If layoutKOptional is PA_BBND, the shape is (block_num, block_size, N2, D).</li><li>If layoutKOptional is BSND, the shape is (B, S2, N2, D).</li><li>If layoutKOptional is TND, the shape is (T2, N2, D).</li></ul>
    </td>
    <td>✓</td>
    </tr>
    <tr>
    <td>w (aclTensor*)</td>
    <td>Input</td>
    <td>Input W in the formula.</td>
    <td>Empty tensors are not supported.</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td><ul><li>When layoutQOptional is BSND, the shape is (B,S1,N1).</li><li>When layoutQOptional is TND, the shape is (T1,N1).</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensQOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Cumulative sum of the number of valid q tokens in the current batch and preceding batches.</td>
    <td><ul><li>The shape is constrained to a one-dimensional tensor of length B+1.</li><li>Negative values cannot appear.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B+1,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensKOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Cumulative sum of the number of valid k tokens in the current batch and preceding batches.</td>
    <td><ul><li>The shape is a one-dimensional tensor constrained to length B+1.</li><li>Negative values cannot appear.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B+1,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sequsedQOptional (aclTensor*)</td>
    <td>Input</td>
    <td>The actual used length of q in different batches.</td>
    <td><ul><li>The shape is constrained to a one-dimensional tensor of length B.</li><li>Negative values cannot appear.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sequsedKOptional (aclTensor*)</td>
    <td>Input</td>
    <td>The actual used length of k in different batches.</td>
    <td><ul><li>Shape is constrained to length B, one-dimensional tensor.</li><li>Negative values cannot appear.</li><li>When layoutKOptional is PA_BBND, this parameter must be passed in.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cmpResidualKOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Indicates the remainder of dividing the number of tokens in k before compression by cmpRatio.</td>
    <td><ul><li>Empty tensors are not supported.</li><li>Must be used in the use case where maskMode is 3 and cmpRatio is not 1.</li><li>Each element value of this parameter must be less than the compression ratio cmpRatio.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>blockTableOptional(aclTensor*)</td>
    <td>Input</td>
    <td>Indicates the block mapping table used for KV storage in PageAttention.</td>
    <td><ul><li>Empty tensors are not supported.</li><li>In PageAttention use cases, block_table must be two-dimensional. The length of the first dimension must be equal to B, and the length of the second dimension cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum sequence length in each batch).</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>Supported shape (B,S2_max/block_size)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>outputIdxOffsetOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Indicates the offset to be added to the topK result output index.</td>
    <td><ul><li>The value must be greater than or equal to 0.</li><li>After the offset is added, the topK index cannot exceed the maximum value of int32.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td><ul><li>When layoutQOptional is "BSND", the shape is [B, S1, N2].</li><li>When layoutQOptional is "TND", the shape is [T1, N2].</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>metadataOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Split-core information passed in by the LightningIndexerV2Metadata operator, including the number of cores used, block size, and the start point of data processed by each core.</td>
    <td><ul><li>Empty tensors are not supported.</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>The shape is fixed to [1024].</td>
    <td>x</td>
    </tr>
    <tr>
    <td>topk (int64_t)</td>
    <td>Input</td>
    <td>Number of Key token indexes to be retained in the topK phase.</td>
    <td><ul><li>Currently supported [1, 8192].</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maxSeqlenQ (int64_t)</td>
    <td>Input</td>
    <td>Maximum sequence length of q.</td>
    <td><ul><li>Currently supported [-1] or greater than or equal to [0].</li><li>-1 indicates any possible length.</li><li>The recommended value is -1.</li></ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutQOptional (char*)</td>
    <td>Input</td>
    <td>Identifies the data layout format of the input q.</td>
    <td><ul><li>Currently supported: BSND and TND.</li><li>Recommended value is BSND.</li></ul></td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutKOptional (char*)</td>
    <td>Input</td>
    <td>Indicates the data layout format of input k.</td>
    <td><ul><li>Currently supported: PA_BBND, BSND, TND.</li><li>Recommended value is BSND.</li></ul></td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskMode (int64_t)</td>
    <td>Input</td>
    <td>Indicates the mask mode.</td>
    <td><ul><li>When mask_mode is 0, defaultMask mode is used.</li><li>When mask_mode is 3, the mask in rightDownCausal mode is used, corresponding to the lower triangle use case with the right vertex as the dividing point.</li><li>The recommended value is 0.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>cmpRatio (int64_t)</td>
    <td>Input</td>
    <td>Used for sparse computation. Indicates the compression ratio of k.</td>
    <td><ul><li>Supported [1, 128].</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>returnValue (int64_t)</td>
    <td>Input</td>
    <td>Indicates whether to return the Values corresponding to Indices.</td>
    <td><ul><li>0 indicates no return, and 1 indicates return.</li><li>Recommended value is 0.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseIndices (aclTensor*)</td>
    <td>Output</td>
    <td>In the Formula, the Indices output.</td>
    <td>Empty tensor not supported.</td>
    <td>INT32</td>
    <td>ND</td>
    <td><ul><li>Output shape is [B, S1, N2, topk] when layoutQOptional is "BSND".</li><li>Output shape is [T1, N2, topk] when layoutQOptional is "TND".</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>sparseValues (aclTensor*)</td>
    <td>Output</td>
    <td>In the Formula, the Values output corresponding to Indices.</td>
    <td>Outputs the corresponding value when returnValue is 1; outputs an empty tensor with shape (0,) when returnValue is 0.</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td><ul><li>When returnValue is 1 and layoutQOptional is "BSND", the output shape is [B, S1, N2, topk].</li><li>When returnValue is 1 and layoutQOptional is "TND", the output shape is [T1, N2, topk].</li></ul></td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the workspace size that needs to be applied on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor (aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor containing the operator computation process.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

  - Dimension meanings of parameters q, k, and w: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, N (Head Num) indicates the number of heads, D (Head Dim) indicates the size of the smallest unit of the hidden layer, and D = H/N. T indicates the sum of the sequence lengths of input samples in all batches.
  - S1 and S2 respectively represent the sequence lengths of the input samples of q and k, N1 and N2 respectively represent the numbers of heads corresponding to q and k, and k indicates the number of indexes finally selected. The values of D in parameters q and k are both 128. T1 and T2 respectively represent the sums of the sequence lengths of the input samples of q and k.

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
    <thead>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a passed-in parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types or data formats of q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset, metadata, sparseIndices, or sparseValues are not within the supported ranges.</td>
      </tr>
      </tbody>
    </table>

## aclnnLightningIndexerV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
    </colgroup>
    <thead>
      <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied by the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied by the Device side, obtained through the first-phase API aclnnLightningIndexerV2GetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. Contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLightningIndexerV2 defaults to a deterministic implementation.
- The N of q supports 1-64. The N of k supports 1.
- headdim supports 128.
- pa_kv_cache supports non-contiguous axis 0; pa_block_size supports 1-1024, with the block size aligned to 32 bytes.
- The data types of q and k must be the same.
- The invalid part of sparseIndices is filled with -1; the invalid part of sparseValues is filled with -inf.
- Each element in the passed cmpResidualKOptional must be smaller than the compression ratio cmpRatio.

<!-- npu="A3,910b" id7 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - The value range of topk currently supports only [1, 2048], as well as 3072, 4096, 5120, 6144, 7168, and 8192.
  - The sequsedQOptional, outputIdxOffsetOptional, and maxSeqlenQ functions are currently not supported. Passing in these parameters is not recommended.
  - When layoutKOptional is PA_BBND, sequsedKOptional must be passed in; when layoutKOptional is BSND or TND, sequsedKOptional is optional.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  - When the passed parameter layoutQOptional is BSND, passing in cuSeqlensQOptional is not supported; when layoutKOptional is BSND or PA_BBND, passing in cuSeqlensKOptional is not supported.
  - When the parameter outputIdxOffsetOptional is passed in, index offset values greater than or equal to 0 are supported. The following constraint must be met: after the passed index offset is added, the resulting sparseIndice value must not exceed the maximum value of INT32.
  - When the passed parameter layoutQOptional is TND, cuSeqlensQOptional must be passed in. If sequsedQOptional is also passed in, ensure that the query length of each batch passed in by sequsedQOptional does not exceed the q sequence length of each batch calculated based on cuSeqlensQOptional. When the q sequence length seqlen1 passed in by sequsedQOptional for a batch is less than the query length seqlen2 calculated based on cuSeqlensQOptional, the TND Padding feature is enabled, and the sparseIndices and sparseValues outputs of the queries in the part corresponding to the difference between seqlen2 and seqlen1 in the batch are all set to invalid values. In some long sequence use cases, if too much invalid data needs to be padded, aicore execution may time out due to hardware limitations. You can calculate the amount of data to be padded using (seqlen2 - seqlen1) * topk. It is recommended to keep this amount within 400 million.
  - When the passed cmpRatio > 1 and maskMode = 3, cmpResidualKOptional must be passed in. In other cases, do not pass it in.
  - When the passed parameter layoutKOptional is PA_BBND, sequsedKOptional must be passed in.
  - Parameter metadataOptional must be passed in.

<!-- end id8 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_v2.h"
#include "aclnnop/aclnn_lightning_indexer_v2_metadata.h"
#include "aclnn/opdev/platform.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
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
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct TensorResources {
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *cmpResidualKDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *sparseValuesDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *weightsTensor = nullptr;
    aclTensor *cmpResidualKTensor = nullptr;
    aclTensor *metadataTensor = nullptr;
    aclTensor *sparseIndicesTensor = nullptr;
    aclTensor *sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources &resources)
{
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N1 = 16;
    int64_t D = 128;
    int64_t N2 = 1;
    int64_t topk = 32;
    int64_t cmpRatio = 4;
    int64_t S2Orig = 130;
    int64_t S2Compressed = S2Orig / cmpRatio;
    int64_t resK = S2Orig % cmpRatio;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2Compressed, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> cmpResidualKShape = {B};
    std::vector<int64_t> metadataShape = {2048};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<float> weightsHostData(weightsShapeSize, 1.0f);
    std::vector<int32_t> cmpResidualKHostData(B, static_cast<int32_t>(resK));
    std::vector<int32_t> metadataHostData(metadataShapeSize, 0);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 0.0f);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_FLOAT16,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &resources.cmpResidualKDeviceAddr,
                          aclDataType::ACL_INT32, &resources.cmpResidualKTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(metadataHostData, metadataShape, &resources.metadataDeviceAddr, aclDataType::ACL_INT32,
                          &resources.metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    return ACL_SUCCESS;
}

int GenerateMetadata(TensorResources &resources, aclrtStream stream, int64_t B, int64_t S1, int64_t S2, int64_t N1,
                     int64_t N2, int64_t D, int64_t topk, int64_t maskMode, int64_t cmpRatio)
{
    constexpr const char layoutQ[] = "BSND";
    constexpr const char layoutK[] = "BSND";
    constexpr size_t layoutLen = sizeof(layoutQ);
    char layoutQCopy[layoutLen];
    char layoutKCopy[layoutLen];
    errno_t memcpyRet = memcpy_s(layoutQCopy, sizeof(layoutQCopy), layoutQ, layoutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("metadata memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKCopy, sizeof(layoutKCopy), layoutK, layoutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("metadata memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }

    aclOpExecutor *executor;
    uint64_t workspaceSize = 0;
    int ret = aclnnLightningIndexerV2MetadataGetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, resources.cmpResidualKTensor, N1, N2, D, topk, B, S1, S2, layoutQCopy,
        layoutKCopy, maskMode, cmpRatio, resources.metadataTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    void *metadataWsAddr = nullptr;
    if (workspaceSize > 0ULL) {
        ret = aclrtMalloc(&metadataWsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("metadata allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexerV2Metadata(metadataWsAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2Metadata failed. ERROR: %d\n", ret);
        if (metadataWsAddr) {
            (void)aclrtFree(metadataWsAddr);
        }
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("metadata synchronize stream failed. ERROR: %d\n", ret);
        if (metadataWsAddr) {
            (void)aclrtFree(metadataWsAddr);
        }
        return ret;
    }

    if (metadataWsAddr) {
        (void)aclrtFree(metadataWsAddr);
    }
    return ACL_SUCCESS;
}
int ExecuteLightningIndexerV2(TensorResources &resources, aclrtStream stream, void **workspaceAddr,
                              uint64_t *workspaceSize)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 3;
    int64_t cmpRatio = 4;
    int64_t returnValue = 0;
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor *executor;

    int ret = aclnnLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        resources.cmpResidualKTensor, nullptr, nullptr, resources.metadataTensor, topk, maxSeqlenQ, layoutQuery,
        layoutKey, maskMode, cmpRatio, returnValue, resources.sparseIndicesTensor, resources.sparseValuesTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("sparse_indices result[%ld] is: %d\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources &resources, void *workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
        aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
        aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.cmpResidualKTensor) {
        aclDestroyTensor(resources.cmpResidualKTensor);
    }
    if (resources.metadataTensor) {
        aclDestroyTensor(resources.metadataTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
        aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
        aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.cmpResidualKDeviceAddr) {
        aclrtFree(resources.cmpResidualKDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    if (stream) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N2 = 1;
    int64_t S2 = 32;
    int64_t N1 = 16;
    int64_t D = 128;
    int64_t topk = 32;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        LOG_PRINT("This example requires Ascend 950.\n");
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return 0;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    ret = GenerateMetadata(resources, stream, B, S1, S2, N1, N2, D, topk, 3, 4);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("GenerateMetadata failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }
    ret = ExecuteLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
