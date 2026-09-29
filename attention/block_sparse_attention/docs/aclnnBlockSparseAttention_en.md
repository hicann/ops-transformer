# aclnnBlockSparseAttention

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Function**: BlockSparseAttention performs sparse attention computation and supports flexible block-level sparsity modes. It uses BlockSparseMask to specify the KV blocks selected for each Q block, achieving efficient sparse attention computation.

- **Formula**: Sparse block size of $blockShapeX \times blockShapeY$, with sparsity pattern defined by `selectIdx`

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  The input query, key, and value of BlockSparseAttention can be arranged in multiple dimensions, which can be specified by qInputLayout and kvInputLayout.

  - `B` (`Batch`): input batch size
  - T: total token length with combined B and S dimensions
  - `S` (`Seq-Length`): sequence length of input samples
  - `H` (`Head-Size`): hidden-layer size
  - `N` (`Head-Num`): number of heads
  - `D` (`Head-Dim`): minimum unit size of the hidden layer (`D` = `H`/`N`)

  The following layouts are supported:

  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBlockSparseAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnBlockSparseAttention` is called to perform computation.

```c++
aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  const aclTensor   *blockTableOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            innerPrecise,
  int64_t            blockSize,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            softmaxLseFlag,
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnBlockSparseAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream stream)
```

## aclnnBlockSparseAttentionGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px">
  <colgroup>
  <col style="width: 269px">
  <col style="width: 132px">
  <col style="width: 318px">
  <col style="width: 236px">
  <col style="width: 185px">
  <col style="width: 119px">
  <col style="width: 146px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>`query` in the formulas.</td>
      <td>Supported shapes:
        <ul><li>TND: [totalQTokens, headNum, headDim].</li>
        <li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li></ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>`key` in the formulas.</td>
      <td>Supported shapes:
        <ul>
          <li>TND: [totalKTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>`value` in the formula.</td>
      <td>
        Supported shapes:
        <ul>
          <li>TND: [totalVTokens, numKeyValueHeads, headDim].</li>
          <li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockSparseMaskOptional</td>
      <td>Input</td>
      <td>Actual sparse pattern.</td>
      <td>
        Optional (Mandatory in the current version)
        <ul>
          <li>The shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)].</li>
          <li>Indicates which blocks are involved in computation (value 1) and which blocks are not involved in computation (value 0) after blocks are divided.</li>
          <li>If nullptr is passed, block sparsity is disabled, that is, the attention scores between all tokens are computed.</li>
        </ul>
      </td>
      <td>INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>`atten_mask` in the formula.</td>
      <td>The effect of atten_mask is superimposed with that of the sparse pattern. This parameter is not supported currently. nullptr must be passed.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
    <tr>
      <td>blockShapeOptional</td>
      <td>Input</td>
      <td>aclIntArray on the host, which is the array of sparse block shapes.</td>
      <td>
        Used together with blockSparseMaskOptional:
        <ul>
          <li>When blockSparseMaskOptional is configured, if this input is configured, the operator obtains the sparse block size from it. If this input is not configured, the operator uses the default sparse block size [128, 128].</li>
          <li>When blockSparseMaskOptional is not configured, the operator ignores this input regardless of how it is configured.</li>
        </ul>
        When this input is configured, it must contain two elements [blockShapeX, blockShapeY].
        <ul>
          <li>blockShapeX: block size in the Q direction. The value must be greater than 0.</li>
          <li>blockShapeY: block size in the KV direction. The value must be greater than 0 and a multiple of 128.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
      <td>actualSeqLengthsOptional</td>
      <td>Input</td>
      <td>aclIntArray on the host, describing the length of the query sequence corresponding to each batch.</td>
      <td>
        Optional input, used in variable-length sequence scenarios:
        <ul>
          <li>When qInputLayout is set to TND, this input must be configured.</li>
          <li>When qInputLayout is set to BNSD, if this input is configured, the operator processes the input based on the actual sequence length specified by the input. If this input is not configured (that is, nullptr is passed), the operator processes the input based on the value of S in the query shape.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional</td>
      <td>Input</td>
      <td>aclIntArray on the host, which describes the length of the key-value sequence corresponding to each batch.</td>
      <td>
        Optional input, used in variable-length sequence scenarios:
        <ul>
          <li>This input is mandatory when kvInputLayout is set to TND.</li>
          <li>When kvInputLayout is set to BNSD, if this input is configured, the operator processes the input based on the actual sequence length specified by the input. If this input is not configured (that is, nullptr is passed), the operator processes the input based on S in the shape of the key-value pair.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Block table for PagedAttention.</td>
      <td>Currently, this parameter is not supported. nullptr must be passed.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qInputLayout</td>
      <td>Input</td>
      <td>String on the host, indicating the data layout format of the input query.</td>
      <td>Currently, only TND and BNSD are supported. qInputLayout and kvInputLayout must be the same.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout</td>
      <td>Input</td>
      <td>String on the host, indicating the data layout format of the input key and value.</td>
      <td>Currently, only TND and BNSD are supported. The value of qInputLayout must be the same as that of kvInputLayout.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>Input</td>
      <td>int64_t on the host, indicating the number of key/value heads.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>Input</td>
      <td>int64_t on the host, indicating the mask type in attention computation.</td>
      <td>
        Currently, only 0 is supported.
        <ul>
          <li>0: indicates that no mask is added.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>double on the host, indicating the scaling coefficient in the formula.</td>
      <td>Generally, this parameter is set to D<sup>–0.5</sup>.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Input</td>
      <td>int64_t on the host, indicating the precision level used for Softmax computation.</td>
      <td>
        It controls the data type used in the online softmax and rescale phases. Currently, only 0 or 1 can be passed.
        <ul>
          <li>0: The fp32 data type is used for both online softmax and rescale, which is suitable for scenarios that require high computation precision.</li>
          <li>1: This value can be configured only when the input query, key, and value are all of the fp16 data type. In this case, the fp16 data type is used for both online softmax and rescale, which provides better performance but lower precision. In addition, value overflow may occur during computation. You need to determine whether to use this value based on the value range.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>Input</td>
      <td>int64_t on the host, indicating the block size of PagedAttention.</td>
      <td>This parameter is used in the PagedAttention scenario. Currently, the PagedAttention function is not supported. Therefore, only 0 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>int64_t on the host, indicating the number of tokens that the sliding window needs to contain in the forward direction in the sliding window attention scenario.</td>
      <td>This parameter is used in the sliding window attention scenario. Currently, sliding window attention is not supported. Only 2147483647 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>int64_t on the host, indicating the number of tokens that the sliding window needs to contain in the backward direction in the sliding window attention scenario.</td>
      <td>This parameter is used in the sliding window attention scenario. Currently, sliding window attention is not supported. Only 2147483647 can be passed.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxLseFlag</td>
      <td>Input</td>
      <td>int64_t on the host, indicating whether to enable the softmaxLse output.</td>
      <td>
        Currently, only 0 or 1 can be passed.
        <ul>
          <li>0: The softmaxLse output is not generated.</li>
          <li>1: The softmaxLse output is generated, which may cause performance loss.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>`attentionOut` in the formula.</td>
      <td>The data type and shape must be the same as those of `query`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>Output</td>
      <td>Intermediate result of the log-sum-exp operation in Softmax.</td>
      <td>
        The supported shapes vary with the shape of query:
        <ul>
          <li>When query is TND: [totalQTokens, headNum, 1].</li>
          <li>When query is BNSD: [batch, headNum, maxQSeqLength, 1].</li>
        </ul>
      </td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3/4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input query, key, and value are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of `query`, `key`, or `value` is not supported.</td>
    </tr>
    <tr>
      <td>qInputLayout or kvInputLayout is invalid.</td>
    </tr>
    <tr>
      <td>blockShape is invalid (the number of elements is less than 2 or an element value is less than or equal to 0).</td>
    </tr>
    <tr>
      <td>innerPrecise is invalid. It must be 0, 1, or 4.</td>
    </tr>
  </tbody>
  </table>

## aclnnBlockSparseAttention

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnBlockSparseAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>AscendCL stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

`aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnBlockSparseAttention is used.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Currently, `qInputLayout` can only be TND or BNSD.
- Currently, `kvInputLayout` can only be TND or BNSD.
- The input layouts of the query, key, and value must be the same.
- The input query, key, and value must be of the same data type, which can be either FLOAT16 or BFLOAT16.
- The dimension D of the query, key, and value can only be set to 64 or 128.
- If blockShapeOptional is passed, it must contain at least two elements [blockShapeX, blockShapeY], and the values must be greater than 0. blockShapeY must be a multiple of 128.
- blockSparseMaskOptional must be passed, and its shape must be [batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKVS, blockShapeY)].
- Currently, only nullptr can be passed to attentionMaskOptional.
- actualSeqLengthsOptional is required when qInputLayout is set to "TND". actualSeqLengthsKvOptional is required when kvInputLayout is set to "TND".
- actualSeqLengthsOptional and actualSeqLengthsKvOptional must be both configured or both not configured. If only one of them is configured, the operator will intercept the configuration.
- Currently, only nullptr can be passed to blockTableOptional, indicating that the PagedAttention feature is disabled.
- `innerPrecise` must be 0 (for float32 Softmax) or 1 (for fp16 Softmax). If the `query` input is BFLOAT16, `innerPrecise` can only be set to 0.
- `qSeqlen` and `kvSeqlen` do not need to be exactly divided by `blockShape`. When they are not divisible, the actual number of blocks is determined by ceiling division.
- If headNum of input `query` is N1 and headNum of input `key` and `value` is N2, then `N1 >= N2 && N1% N2 == 0`.
- Currently, maskType can only be set to 0, indicating that no mask is added.
- Currently, blockSize can only be set to 0, indicating that the paged cache is not supported.
- Currently, preTokens and nextTokens can only be set to 2147483647, indicating that all tokens before and after the current token are involved in the attention operation. That is, sliding window attention is not supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention.h"

using namespace std;

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

int Init(int32_t deviceId, aclrtStream* stream) {
    // (Fixed writing) Initialize AscendCL.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // Check whether shape is valid.
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }
    
    auto size = GetShapeSize(shape) * sizeof(T);

    // Check whether the size of hostData matches GetShapeSize(shape).
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    // Call aclrtMalloc to allocate memory on the device.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    
    // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              return ret);
    
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
            return -1);
    return 0;
}

void FreeResource(aclTensor *query, aclTensor *key, aclTensor *value, aclTensor *blockSparseMask,
                  aclTensor *attentionOut, aclIntArray *actualSeqLengths, aclIntArray *actualSeqLengthsKv,
                  aclIntArray *blockShape, void *queryDeviceAddr, void *keyDeviceAddr, void *valueDeviceAddr,
                  void *blockSparseMaskDeviceAddr, void *attentionOutDeviceAddr, void *actualSeqLengthsDeviceAddr,
                  void *actualSeqLengthsKvDeviceAddr, void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    // Free resources.
    if (query) {
        aclDestroyTensor(query);
    }
    if (key) {
        aclDestroyTensor(key);
    }
    if (value != nullptr) {
        aclDestroyTensor(value);
    }
    if (blockSparseMask) {
        aclDestroyTensor(blockSparseMask);
    }
    if (attentionOut) {
        aclDestroyTensor(attentionOut);
    }
    if (actualSeqLengths) {
        aclDestroyIntArray(actualSeqLengths);
    }
    if (actualSeqLengthsKv) {
        aclDestroyIntArray(actualSeqLengthsKv);
    }
    if (blockShape) {
        aclDestroyIntArray(blockShape);
    }

    if (queryDeviceAddr) {
        aclrtFree(queryDeviceAddr);
    }
    if (keyDeviceAddr) {
        aclrtFree(keyDeviceAddr);
    }
    if (valueDeviceAddr) {
        aclrtFree(valueDeviceAddr);
    }
    if (blockSparseMaskDeviceAddr) {
        aclrtFree(blockSparseMaskDeviceAddr);
    }
    if (attentionOutDeviceAddr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (actualSeqLengthsDeviceAddr) {
        aclrtFree(actualSeqLengthsDeviceAddr);
    }
    if (actualSeqLengthsKvDeviceAddr) {
        aclrtFree(actualSeqLengthsKvDeviceAddr);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Set parameters.
    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;
    
    // Calculate the dimensions in TND format.
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX; // Number of query blocks along the X dimension.
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY; // Number of key-value blocks along the Y dimension.
    // totalQBlocks = qBlockNum * numHeads (each query block corresponds to one head)
    int32_t totalQBlocks = qBlockNum * numHeads;
    int32_t maxKvBlockNum = kvBlockNum;
    
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockSparseMaskTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
    aclIntArray *actualSeqLengthsKv = nullptr;
    aclIntArray *blockShape = nullptr;

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockSparseMaskDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *actualSeqLengthsDeviceAddr = nullptr;
    void *actualSeqLengthsKvDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    // 3. Create the query tensor (TND format: [totalQTokens, numHeads, headDim]).
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 4. Create a key-value tensor (TND format: [totalKvTokens, numKvHeads, headDim]).
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    std::vector<op::fp16_t> valueHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 5. Create the blockSparseMask tensor ([batch, numHeads, qBlockNum, kvBlockNum]).
    std::vector<int8_t> blockSparseMaskHostData(totalQBlocks * numHeads, 0);
    blockSparseMaskHostData[0] = static_cast<int8_t>(1);
    std::vector<int64_t> blockSparseMaskShape = {batch, numHeads, qBlockNum, kvBlockNum};
    ret = CreateAclTensor(blockSparseMaskHostData, blockSparseMaskShape, &blockSparseMaskDeviceAddr, aclDataType::ACL_INT8, &blockSparseMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create block sparse mask tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 6. Create an output tensor.
    std::vector<int64_t> attentionOutShape = {totalQTokens, numHeads, headDim};
    int64_t attentionOutElementCount = totalQTokens * numHeads * headDim;
    std::vector<op::fp16_t> attentionOutHostData(attentionOutElementCount, 0.0f);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create attentionOut tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 7. Create a blockShape array.
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 8. Create mandatory parameters actualSeqLengths and actualSeqLengthsKv.
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));
    
    size_t seqLengthsSize = batch * sizeof(int64_t);

    ret = aclrtMalloc(&actualSeqLengthsDeviceAddr, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = aclrtMalloc(&actualSeqLengthsKvDeviceAddr, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    ret = aclrtMemcpy(actualSeqLengthsDeviceAddr, seqLengthsSize, actualSeqLengthsHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDeviceAddr, seqLengthsSize, actualSeqLengthsKvHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // aclCreateIntArray expects a data pointer on the host instead of one on the device.
    actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr,
              LOG_PRINT("Failed to create actualSeqLengths arrays\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 9. Prepare string parameters. Ensure that the buffer sizes are sufficient and include the null terminator.
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);
    
    // 10. Calculate scaleValue.
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
    
    // 11. Call the first-phase API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnBlockSparseAttentionGetWorkspaceSize(
        queryTensor,           // query
        keyTensor,             // key
        valueTensor,           // value
        blockSparseMaskTensor, // blockSparseMask
        nullptr,               // attenMaskOptional
        blockShape,            // blockShape
        actualSeqLengths,      // actualSeqLengthsOptional
        actualSeqLengthsKv,    // actualSeqLengthsKvOptional
        nullptr,               // blockTableOptional
        qLayoutBuffer,         // qInputLayout
        kvLayoutBuffer,        // kvInputLayout
        numKvHeads,            // numKeyValueHeads
        0,                     // maskType
        scaleValue,            // scaleValue
        0,                     // innerPrecise (1=fp16 softmax)
        0,                     // blockSize
        2147483647,            // preTokens
        2147483647,            // nextTokens
        0,                     // softmaxLseFlag
        attentionOutTensor,          // attentionOut
        nullptr,               // softmaxLseOptional
        &workspaceSize,        // workspaceSize (out)
        &executor);            // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 12. Allocate memory for workspace.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
                      FreeResource(
                      queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                      actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                      blockSparseMaskDeviceAddr, attentionOutDeviceAddr, actualSeqLengthsDeviceAddr,
                      actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 12. Call the second-phase API.
    ret = aclnnBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttention failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 13. Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 14. Obtain the output value and copy the result from the device memory to the host.
    int64_t attentionOutSize = GetShapeSize(attentionOutShape);
    std::vector<op::fp16_t> resultData(attentionOutSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), attentionOutDeviceAddr,
                      attentionOutSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 15. Print partial results.
    uint64_t printNum = 10;
    LOG_PRINT("attentionOut results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }
    
    // 16. Release resources.
    FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                 actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                 blockSparseMaskDeviceAddr, attentionOutDeviceAddr, actualSeqLengthsDeviceAddr,
                 actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
