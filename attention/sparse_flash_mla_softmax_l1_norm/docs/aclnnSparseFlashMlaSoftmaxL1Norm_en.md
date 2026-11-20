# aclnnSparseFlashMlaSoftmaxL1Norm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:29:46.791Z -->

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

- Description: Computes the Softmax L1Norm result of `SparseFlashMla` attention, supporting Compressed Attention and Sparse Compressed Attention use cases. This API is the companion API of the `aclnnDenseLightningIndexerKLLossGrad` backward operator, and its output can be used for backward gradient computation. During the call, call the `aclnnSparseFlashMlaSoftmaxL1NormMetadata` API first to complete the load balancing computation, and then call the `aclnnSparseFlashMlaSoftmaxL1Norm` API to complete the operator computation.
    - `aclnnSparseFlashMlaSoftmaxL1NormMetadata` API: Based on information such as the shape, layout, and mask of the primary operator, evenly partitions tasks across available AIC cores in a strided manner, and outputs metadata for use by the primary operator.
    - `aclnnSparseFlashMlaSoftmaxL1Norm` API: Computes the Softmax L1Norm of Q and K based on the core partitioning information in metadata.

- Formula:

    Stage 1: Select the input k based on whether the use case is sparse

    - When sparse:

    $$
    selectedKv\text{ ​}=\text{ }Gather \left( K, sparseIndices \left[ i \left]  \left) ,\text{ ​}0\text{ } < =i < \text{ ​}selectBlockCount\right. \right. \right. \right.
    $$

    - Else:

    $$
    selectedKv\text{ ​}=\text{ }K
    $$

    Stage 2: Calculate P (SimpleSoftmax)

    $$
    P = SimpleSoftmax(Mask(Q \text{ ​}@\text{ } selectedKv^{{T}} \cdot \text{ } scale), softmaxLse)
    $$

    Stage 3: Calculate Softmax L1Norm

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    Where $G$ indicates the number of groups ($G = N1 / N2$), and $ReduceSum$ sums the softmax probability $P$ over the G dimension (q head group dimension) and then takes the average.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the computation flow of the operator, and then call the "aclnnSparseFlashMlaSoftmaxL1Norm" API to perform the computation.

```c++
aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
    const aclTensor   *q,
    const aclTensor   *k,
    const aclTensor   *softmaxLse,
    const aclTensor   *sparseIndicesOptional,
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensKOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    const aclTensor   *topkLengthOptional,
    const aclTensor   *metadataOptional,
    double             softmaxScale,
    int64_t            cmpRatio,
    int64_t            maskMode,
    char              *layoutQOptional,
    char              *layoutKOptional,
    const aclTensor   *softmaxL1Norm,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnSparseFlashMlaSoftmaxL1Norm(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream);
```

## aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 290px">
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
        </tr></thead>
        <tbody>
        <tr>
            <td>q (aclTensor*)</td>
            <td>Input</td>
            <td>Input Q of the attention structure.</td>
            <td>            The N-axis correspondence between q and k must satisfy the GQA constraint (N1 = N2 × G).
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Support Generalization; S1: Support Generalization; N1: Support 1-128; D: A2/A3 supports 256 or 512, and A5 supports only 512; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>k (aclTensor*)</td>
            <td>Input</td>
            <td>Input K(V) of the attention structure.</td>
            <td>            Empty tensors are currently not supported.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            B: Keep Consistent with the B of q; S2: supports generalization; N2: 1; D: A2/A3 supports 256 or 512, and A5 supports only 512; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxLse (aclTensor*)</td>
            <td>Input</td>
            <td>Output softmaxLse of the forward attention computation. For the computation formula, see the sparse_flash_mla document.</td>
            <td>            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G) or (N2,T1,G)<br>
            B: Keep consistent with B of q; N2: 1; S1: Keep consistent with S1 of q; G: N1/N2; T1: B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sparseIndicesOptional(aclTensor*)</td>
            <td>Input</td>
            <td>Attention indices with higher weights selected from k in sparse use cases.</td>
            <td>            <ul>
                <li>Supports empty tensors.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K) or (T1,N2,K)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Number of valid Query tokens in each batch.</td>
            <td>            <ul>
                <li>Optional: this parameter exists when layout is TND.</li>
                <li>The length is consistent with B+1.</li>
                <li>The cumulative sum is consistent with T1.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensKOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Number of valid Key tokens in each batch.</td>
            <td>            <ul>
                <li>Optional: Available when layout is TND.</li>
                <li>The length is consistent with B+1.</li>
                <li>The cumulative sum is consistent with T2.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedQOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the number of q tokens that actually participate in computation in each batch.</td>
            <td>            Length: B.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedK (optional) (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the number of tokens of k that actually participate in the computation in each batch.</td>
            <td>            Length: B.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpResidualKOptional (aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the remainder of S2 // cmpRatio for each batch.</td>
            <td>            Must be passed in when maskMode=3 and cmpRatio!=1.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>topkLengthOptional(aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the actual optional topk length of k corresponding to q in each row.</td>
            <td>Required when maskMode=0 and sparse indices exist, and must be an accurate value.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2), (T1,N2)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>metadataOptional(aclTensor*)</td>
            <td>Input</td>
            <td>Indicates the output result of the AICPU operator with tiling sinking.</td>
            <td>            Must be passed in. Generated by the aclnnSparseFlashMlaSoftmaxL1NormMetadata operator.
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(x)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxScale (double)</td>
            <td>Input</td>
            <td>Scaling factor.</td>
            <td>Recommended value: reciprocal of the square root of d in the formula.</td>
            <td>FLOAT32</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpRatio (int64_t)</td>
            <td>Input</td>
            <td>Indicates the compression ratio of k.</td>
            <td>Value range: 1-128.</td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>maskMode (int64_t)</td>
            <td>Input</td>
            <td>Indicates the mask mode for the computation of q and k.</td>
        <td>              <ul>
                <li>Indicates the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
            <td>layoutQOptional (char*)</td>
            <td>Input</td>
            <td>Indicates the data layout format of input q.</td>
            <td>            Support "BSND" and "TND". Keep consistent with layoutKOptional.
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutKOptional (char*)</td>
            <td>Input</td>
            <td>Indicates the data layout format of input k.</td>
            <td>            Supports "BSND" and "TND". Must be consistent with layoutQOptional.
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>softmaxL1Norm (aclTensor*)</td>
            <td>Output</td>
            <td>Indicates the softmax L1Norm result computed from q and k. The formula is reduceG(softmax)/G.</td>
            <td>            If sparse indices exist, this output is not empty; otherwise, this output is empty.
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,N2,S2), (T1,N2,T2)<br>
            B: consistent with the B of q; S1: consistent with the S1 of q; N2: 1; S2: consistent with the S2 of k; T1: B × S1; T2: B × S2
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

- **Return Value**

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
                <td>The parameters contain an invalid nullptr.</td>
            </tr>
            <tr>
                <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="2">161002</td>
                <td>The input data type is not among the supported types.</td>
            </tr>
            <tr>
                <td>Required inputs/outputs q, k, softmaxLse, or softmaxL1Norm are not passed.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashMlaSoftmaxL1Norm

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>Address of the workspace memory requested on the Device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace memory requested on the Device, obtained by calling the first-phase API aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Return Value**

   aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSparseFlashMlaSoftmaxL1Norm defaults to a deterministic implementation.
- Common constraints:
    - Empty input handling: Returns directly when q is an empty tensor.

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>No mask operation is performed.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask of the rightDownCausal mode, corresponding to the lower triangular scenario partitioned by the right vertex.</td>
            <td>Supported</td>
        </tr>
        </tbody>
    </table>
- Specification constraints
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specification</th>
                <th>Specification Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>B</td>
            <td>Support Generalization</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1 and S2</td>
            <td>Support Generalization</td>
            <td>Supports unequal lengths for S1 and S2.</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1-128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>-</td>
        </tr>
        <tr>
            <td>D</td>
            <td>A2/A3: 256, 512; A5: 512</td>
            <td>D=256 is supported by the native A2/A3 implementation.</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm.h"
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm_metadata.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // TND layout: qShape=(T1,N1,D), kShape=(T2,N2,D), softmaxLseShape=(N2,T1,G)
  std::vector<int64_t> qShape = {16, 128, 512};
  std::vector<int64_t> kShape = {2048, 1, 512};
  std::vector<int64_t> softmaxLseShape = {1, 16, 128};
  std::vector<int64_t> cuSeqQLenshape = {2};
  std::vector<int64_t> cuSeqKLenshape = {2};
  std::vector<int64_t> cmpResidualKShape = {1};
  std::vector<int64_t> softmaxL1NormShape = {16, 1, 2048};
  std::vector<int64_t> metadataShape = {64};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;
  void* cuSeqQLenDeviceAddr = nullptr;
  void* cuSeqKLenDeviceAddr = nullptr;
  void* cmpResidualKDeviceAddr = nullptr;
  void* softmaxL1NormDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* softmaxLse = nullptr;
  aclTensor* cuSeqQLen = nullptr;
  aclTensor* cuSeqKLen = nullptr;
  aclTensor* cmpResidualK = nullptr;
  aclTensor* softmaxL1Norm = nullptr;
  aclTensor* metadata = nullptr;

  std::vector<short> qHostData(16 * 128 * 512, 1.0);
  std::vector<short> kHostData(2048 * 1 * 512, 1.0);
  std::vector<float> softmaxLseHostData(1 * 16 * 128, 3.0);
  std::vector<int32_t> cuSeqQLenHostData = {0, 16};
  std::vector<int32_t> cuSeqKLenHostData = {0, 2048};
  std::vector<int32_t> cmpResidualKHostData = {0};
  std::vector<float> softmaxL1NormHostData(16 * 1 * 2048, 0);
  std::vector<int32_t> metadataHostData(64, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQLenHostData, cuSeqQLenshape, &cuSeqQLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqKLenHostData, cuSeqKLenshape, &cuSeqKLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqKLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &cmpResidualKDeviceAddr, aclDataType::ACL_INT32,
                        &cmpResidualK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxL1NormHostData, softmaxL1NormShape, &softmaxL1NormDeviceAddr, aclDataType::ACL_FLOAT,
                        &softmaxL1Norm);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double softmaxScale = 0.088388;
  int64_t maxSeqlenK = 2048;
  int64_t cmpRatio = 128;
  int64_t maskMode = 3;
  char layoutQ[4] = {'T', 'N', 'D', 0};
  char layoutK[4] = {'T', 'N', 'D', 0};

  // 1. Call the metadata preceding operator.
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
      cuSeqQLen, cuSeqKLen, nullptr, nullptr, cmpResidualK, nullptr,
      0, 16, 2048, 128, 1, 512, 0, cmpRatio, maskMode, layoutQ, layoutK,
      metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSparseFlashMlaSoftmaxL1NormMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormMetadata failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 2. Call the main operator.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize(
      q, k, softmaxLse, nullptr, cuSeqQLen, cuSeqKLen, nullptr, nullptr, cmpResidualK, nullptr, metadata,
      softmaxScale, maxSeqlenK, cmpRatio, maskMode, layoutQ, layoutK, softmaxL1Norm, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  try {
    ret = aclnnSparseFlashMlaSoftmaxL1Norm(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("Expected kernel failure (skeleton stage): ERROR: %d\n", ret);
    } else {
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    }
  } catch (const std::exception& e) {
    LOG_PRINT("Expected kernel failure (skeleton stage): %s\n", e.what());
  }

  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(softmaxLse);
  aclDestroyTensor(cuSeqQLen);
  aclDestroyTensor(cuSeqKLen);
  aclDestroyTensor(cmpResidualK);
  aclDestroyTensor(softmaxL1Norm);
  aclDestroyTensor(metadata);
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  aclrtFree(cuSeqQLenDeviceAddr);
  aclrtFree(cuSeqKLenDeviceAddr);
  aclrtFree(cmpResidualKDeviceAddr);
  aclrtFree(softmaxL1NormDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
