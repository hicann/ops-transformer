
# aclnnDenseLightningIndexerSoftmaxLse

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: The DenseLightningIndexerSoftmaxLse operator is a branch operator for the DenseLightningIndexerGradKlLoss operator to compute the Softmax input.

- Formulas:

  $$
  \text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
  $$

  $$
  \text{maxIndex}=\text{max}\left(res\right)
  $$

  $$
  \text{sumIndex}=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
  $$

  maxIndex and sumIndex are transferred to the DenseLightningIndexerGradKlLoss operator as inputs for calculating Softmax.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnDenseLightningIndexerSoftmaxLse` is called to perform computation.

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
    const aclTensor   *queryIndex,
    const aclTensor   *keyIndex,
    const aclTensor   *weight,
    const aclIntArray *actualSeqLengthsQueryOptional,
    const aclIntArray *actualSeqLengthsKeyOptional,
    char              *layoutOptional,
    int64_t            sparseMode,
    int64_t            preTokens,
    int64_t            nextTokens,
    const aclTensor   *softmaxMaxOut,
    const aclTensor   *softmaxSumOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLse(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnDenseLightningIndexerSoftmaxLse

- **Parameters**:
  
    <table style="undefined;table-layout: fixed; width: 1550px">
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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
     <tr>
      <td>queryIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Input queryIndex of the lightningIndexer structure.</td>
      <td><ul><li>B: Generalization is supported and is the same as that of B in the query. </li><li>S1: Generalization is supported, and the value cannot be the M axis of Matmul. </li><li>Nidx1: 64, 32, 16, 8. </li><li>D: 128. </li><li>T1: S1s of multiple batches are accumulated.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(B,S1,Nidx1,D), (T1,Nidx1,D)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>keyIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Input keyIndex of the lightningIndexer structure.</td>
      <td><ul><li>B: Generalization is supported and is the same as that of B in queryIndex. </li> <li>S2: Generalization is supported. </li><li>Nidx2: 1. </li><li>D: 128. </li><li>T2: S2s of multiple batches are accumulated.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(B,S2,Nidx2,D), (T2,Nidx2,D)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Weight</td>
      <td><ul><li>B: Generalization is supported and is the same as that of B in queryIndex. </li> <li>S1: Generalization is supported and is the same as that of S1 in queryIndex. </li><li>Nidx1: 64, 32, 16, 8. </li><li>T1: S1s of multiple batches are accumulated.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(B,S1,Nidx1), (T1,Nidx1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>actualSeqLengthsQueryOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid tokens in the query of each batch</td>
      <td><ul><li>Value dependency. </li><li>The length is the same as that of B. </li><li>In the TND format, the last element is the accumulated sum, which is the same as that of T1.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>-</td>
     </tr>
     <tr>
      <td>actualSeqLengthsKeyOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid tokens of the key in each batch</td>
      <td><ul><li>Value dependency. </li><li>The length is the same as that of B. </li><li>In the TND format, the last element is the accumulated sum, which is the same as that of T2.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>-</td>
     </tr>
      <tr>
      <td>layoutOptional (char*)</td>
      <td>Input</td>
      <td>Layout format</td>
      <td><ul><li>Only the BSND and TND formats are supported.</li></ul></td><td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>sparseMode (int64_t)</td>
      <td>Input</td>
      <td>Sparse mode</td>
      <td><ul><li>Sparse mode. For details about sparse modes, see <a href="#constraints">Constraints</a>. </li><li>Only mode 3 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
       <td>preTokens (int64_t)</td>
       <td>Input</td>
       <td>Used for sparse computation, indicating that the attention needs to be associated with the first several tokens.</td>
       <td><ul><li>The definition is the same as that of preTokens in the attention. This parameter takes effect when sparseMode is set to 0 or 4. Only 2^63-1 is supported.</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
     <tr>
       <td>nextTokens (int64_t)</td>
       <td>Input</td>
       <td>Used for sparse computation, indicating that the attention needs to be associated with the last several tokens.</td>
       <td><ul><li>The definition is the same as that of nextTokens in the attention. This parameter takes effect when sparseMode is set to 0 or 4. Only 2^63-1 is supported.</li></ul></td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
      <td>softmaxMaxOut (aclTensor*)</td>
      <td>Output</td>
      <td>Max value used for softmax computation</td>
      <td><ul><li>B: The generalization is supported and is the same as that of queryIndex. </li><li>Nidx2: The value is the same as that of Nidx2 in keyIndex. </li><li>S1: The generalization is supported and is the same as that of S1 in queryIndex. </li><li>T1: S1s of multiple batches are accumulated.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,Nidx2,S1), (Nidx2,T1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>softmaxSumOut (aclTensor*)</td>
      <td>Output</td>
      <td>Sum value used for softmax computation</td>
      <td><ul><li>B: The generalization is supported and is the same as that of query. </li><li>Nidx2: The value is the same as that of Nidx2 in keyIndex. </li><li>S1: The generalization is supported and is the same as that of S1 in queryIndex. </li><li>T1: S1s of multiple batches are accumulated.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,Nidx2,S1), (Nidx2,T1)</td>
      <td>×</td>
     </tr>
     <tr> 
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     </tbody>
    </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
     <th>Return</th>
     <th>Error Code</th>
     <th>Description</th>
    </thead>
    <tbody>
     <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The mandatory parameter or output is a null pointer.</td>
      </tr>
     <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and formats of the input variables such as queryIndex, keyIndex, and weights are not supported.</td>
     </tr>
     <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shapes of multiple input tensors do not match. For details, see <a href="#constraints">Parameters</a>.</td>
     </tr>
     </tbody>
    </table>

## aclnnDenseLightningIndexerSoftmaxLse

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
    </colgroup>
    <thead>
     <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
     <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
     </tr>
     <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize.</td>
     </tr>
     <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
     </tr>
     <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
     </tr>
    </tbody>
    </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The data types of the queryIndex and keyIndex parameters must be the same.

- If the weights parameter is not of type float32, the data types of the queryIndex, keyIndex, and weights parameters must be the same.

- Deterministic computation:
  The default deterministic implementation of aclnnDenseLightningIndexerSoftmaxLse is used.

- Common constraints
  - Processing when the input parameter is empty:
    - If queryIndex is an empty tensor, the function returns directly.
    - The scenarios where the input parameters are empty in the SFAG common restrictions are the same as those in the FAG.

  <table style="undefined;table-layout: fixed; width: 901px"><colgroup>
  <col style="width: 168px">
  <col style="width: 565px">
  <col style="width: 168px">
  </colgroup>
  <thead>
    <tr>
    <th>sparseMode</th>
    <th>Description</th>
    <th>Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>0</td>
    <td><code>defaultMask</code> mode. If <code>attenMask</code> is not passed, the mask operation is not performed, and <code>preTokens</code> and <code>nextTokens</code> are ignored. If <code>attenMask</code> is passed, a complete <code>attenMask</code> matrix needs to be passed, indicating that the portion between <code>preTokens</code> and <code>nextTokens</code> needs to be calculated.</td>
    <td>Not supported.</td>
    </tr>
    <tr>
    <td>1</td>
    <td><code>allMask</code> mode. A complete <code>attenMask</code> matrix must be passed.</td>
    <td>Not supported.</td>
    </tr>
    <tr>
    <td>2</td>
    <td><code>leftUpCausal</code> mode. An optimized <code>attenMask</code> matrix needs to be passed.</td>
    <td>Not supported.</td>
    </tr>
    <tr>
    <td>3</td>
    <td><code>rightDownCausal</code> mode. This corresponds to a lower-triangular matrix partitioned by the top-right vertex. An optimized <code>attenMask</code> matrix needs to be passed.</td>
    <td>Supported</td>
  </tr>
  <tr>
    <td>4</td>
    <td><code>band</code> mode. An optimized <code>attenMask</code> matrix needs to be passed.</td>
    <td>Not supported.</td>
  </tr>
  <tr>
    <td>5</td>
    <td>prefix</td>
    <td>Not supported.</td>
  </tr>
  <tr>
    <td>6</td>
    <td>global</td>
    <td>Not supported.</td>
  </tr>
  <tr>
    <td>7</td>
    <td>dilated</td>
    <td>Not supported.</td>
  </tr>
  <tr>
    <td>8</td>
    <td>block_local</td>
    <td>Not supported.</td>
  </tr>
  </tbody>
  </table>

- Constraints

  <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
  <col style="width: 125px">
  <col style="width: 182px">
  <col style="width: 602px">
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
    <td>1~256</td>
    <td>-</td>
  </tr>
  <tr>
    <td>S1, S2</td>
    <td>1~128K</td>
    <td>S1 and S2 can be of different lengths. When the layout is BSND, S1 <= S2. When the layout is TND, the value of actualSeqLengthsQuery is less than or equal to the value of actualSeqLengthsKey at the same index position, and S1 <= S2 at the same index position.</td>
  </tr>
  <tr>
    <td>Nidx1</td>
    <td>8, 16, 32, 64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Nidx2</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>D</td>
    <td>128</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layout</td>
    <td>BSND/TND</td>
    <td>-</td>
  </tr>
  </tbody>
  </table>

- Typ.

  <table style="undefined;table-layout: fixed; width: 903px"><colgroup>
  <col style="width: 164px">
  <col style="width: 739px">
  </colgroup>
  <thead>
  <tr>
    <th>Specification</th>
    <th>Typical Value</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>queryIndex</td>
    <td>N1 = 64/32;  D = 128 ; S1 = 64k/128k</td>
  </tr>
  <tr>
    <td>keyIndex</td>
    <td>D = 128.</td>
  </tr>
  </tbody>
  </table>

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // (Fixed writing) Initialize AscendCL.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int64_t s1 = 4096;
  int64_t s2 = 4096;
  int64_t n1Index = 8;
  int64_t n2Index = 1;
  int64_t dQueryIndex = 128;
  int64_t t1 = s1;
  int64_t t2 = s2;
  int64_t G = n1Index / n2Index;

  std::vector<int64_t> qIndexShape = {t1, n1Index, dQueryIndex};
  std::vector<int64_t> kIndexShape = {t2, n2Index, dQueryIndex};
  std::vector<int64_t> weightShape = {t1, n1Index};
  std::vector<int64_t> softmaxMaxIndexShape = {n2Index, t1};
  std::vector<int64_t> softmaxSumIndexShape = {n2Index, t1};

  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* softmaxMaxIndexDeviceAddr = nullptr;
  void* softmaxSumIndexDeviceAddr = nullptr;

  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* softmaxMaxIndex = nullptr;
  aclTensor* softmaxSumIndex = nullptr;

  std::vector<aclFloat16> qIndexHostData(t1 * n1Index * dQueryIndex, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> kIndexHostData(t2 * n2Index * dQueryIndex, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> weightHostData(t1 * n1Index, aclFloatToFloat16(0.005));

  std::vector<float> softmaxMaxIndexHostData(t1 * n2Index, 25.4483f);
  std::vector<float> softmaxSumIndexHostData(t1 * n2Index, 1.0f);

  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxIndexHostData, softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr,
      aclDataType::ACL_FLOAT, &softmaxMaxIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumIndexHostData, softmaxSumIndexShape, &softmaxSumIndexDeviceAddr,
      aclDataType::ACL_FLOAT, &softmaxSumIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t>  acSeqQLenOp = {t1};
  std::vector<int64_t>  acSeqKvLenOp = {t2};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  int64_t preTokens = 9223372036854775807;
  int64_t nextTokens = 9223372036854775807;
  int64_t sparseMode = 3;

  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize.
  ret = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
            qIndex, kIndex, weight, acSeqQLen, acSeqKvLen, layOut,
            sparseMode, preTokens, nextTokens, softmaxMaxIndex, softmaxSumIndex,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnDenseLightningIndexerSoftmaxLse.
  ret = aclnnDenseLightningIndexerSoftmaxLse(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLse failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr);
  PrintOutResult(softmaxSumIndexShape, &softmaxSumIndexDeviceAddr);
  
  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(weight);
  aclDestroyTensor(softmaxMaxIndex);
  aclDestroyTensor(softmaxSumIndex);
  
  // 7. Free device resources.
  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(softmaxMaxIndexDeviceAddr);
  aclrtFree(softmaxSumIndexDeviceAddr);
  
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return 0;
}

```
