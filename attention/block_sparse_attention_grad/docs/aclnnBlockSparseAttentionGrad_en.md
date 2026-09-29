# aclnnBlockSparseAttentionGrad

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products</term>|      √     |
|<term>Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>                                        |    ×    |
|<term>Atlas inference products</term>                                                |    ×    |
|<term>Atlas training products</term>                                                |    ×    |

## Function

* Function: The aclnnBlockSparseAttention API performs backward sparse attention computation. It supports flexible block-level sparse modes and uses BlockSparseMask to specify the KV block selected by each Q block, implementing efficient sparse attention computation.
* The calculation formula is as follows:
Sparse block size: $blockShapeX×blockShapeY$. BlockSparseMask specifies the sparse mode.
  
  The known forward computation formula is as follows:
  
  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$
  
  For convenience, the formula can be represented using variables $S$ and $P$:
  
  $$
  S = Mask(scale⋅query⋅key_{sparse}^{T},atten\_mask)
  $$
  
  $$
  P = SoftMax(S)
  $$

  $$
  V = value_{sparse}
  $$

  $$
  Out = PV
  $$
  
  The backward computation formula is as follows:

  $$
  softmax\_grad = softmaxGrad(dOut, attentionOut)
  $$

  $$
  dP=dOut * V^T
  $$

  $$
  dS = P * (dP-softmax\_grad)
  $$

  $$
  dV=P^T * dOut
  $$

  $$
  dQ=(dS*K)*scale
  $$

  $$
  dK=(dS^T*Q)*scale
  $$

The data layout formats of the BlockSparseAttentionGrad inputs dout, query, key, value, and attentionOut can be interpreted from multiple dimensions. The formats can be passed through qInputLayout and kvInputLayout. To facilitate understanding of the supported layout formats (such as BNSD and TND), the meaning of each dimension represented by the abbreviations in the layout formats is described as follows:

* `B` (`Batch`): input batch size
* T: total token length with combined B and S dimensions
* `S` (`Seq-Length`): sequence length of input samples
* `H` (`Head-Size`): hidden-layer size
* `N` (`Head-Num`): number of heads
* `D` (`Head-Dim`): minimum unit size of the hidden layer (`D` = `H`/`N`)

The following layouts are supported:

* qInputLayout: "TND" "BNSD"
* kvInputLayout: "TND" "BNSD"

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBlockSparseAttentionGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnBlockSparseAttentionGrad` is called to perform computation.

```Cpp
aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
  const aclTensor   *dout,
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attentionOut,
  const aclTensor   *softmaxLse,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            preTokens,
  int64_t            nextTokens,
  aclTensor         *dq,
  aclTensor         *dk,
  aclTensor         *dv,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnBlockSparseAttentionGrad(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

### aclnnBlockSparseAttentionGradGetWorkspaceSize

* **Parameters**

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
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
    </tr>
</thead>
<tbody>
    <tr>
    <td>dout (aclTensor*) </td>
    <td>Input</td>
    <td>Reversely output the gradient, which represents the gradient information of the final output with respect to the current operator.</td>
    <td>Empty tensors are not supported.<br>The supported shape is as follows: <ul><li>TND: [totalQTokens, headNum, headDim]. </li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>query (aclTensor*) </td>
    <td>Input</td>
    <td>Query vector in attention calculation, that is, query in the formula.</td>
    <td>Empty tensors are not supported.<br>The supported shape is as follows: <ul><li>TND: [totalQTokens, headNum, headDim]. </li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key (aclTensor*) </td>
    <td>Input</td>
    <td>Key vector in attention calculation, that is, key in the formula.</td>
    <td>Empty tensors are not supported.<br>The supported shape is as follows: <ul><li>TND: [totalKTokens, numKeyValueHeads, headDim]. </li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value (aclTensor*) </td>
    <td>Input</td>
    <td>Value vector in attention calculation, that is, value in the formula.</td>
    <td>Empty tensors are not supported.<br>The supported shape is as follows: <ul><li>TND: [totalVTokens, numKeyValueHeads, headDim]. </li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attentionOut (aclTensor*) </td>
    <td>Input</td>
    <td>Output of the forward BlockSparseAttention computation, that is, attentionOut in the formula.</td>
    <td>Empty tensors are not supported.<br>The supported shape is as follows: <ul><li>TND: [totalQTokens, headNum, headDim]. </li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>softmaxLse (aclTensor*) </td>
    <td>Input</td>
    <td>Intermediate result of the log-sum-exp operation in Softmax. It is used to reversely calculate the logarithm and exponential gradient.</td>
    <td>Empty tensors are not supported.<br>The supported shape is as follows: <ul><li>TND: [totalQTokens, headNum, 1]. </li><li>BNSD: [batch, headNum, maxQSeqLength, 1].</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockSparseMaskOptional (aclTensor*) </td>
    <td>Input</td>
    <td>Block sparse mask, indicating the actual sparse pattern. It determines which blocks are actually involved in the attention computation.</td>
    <td>Empty tensors are not supported.<br>Optional (mandatory in the current version): <ul><li>shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]. </li><li>Indicates which blocks are involved in computation (value 1) and which blocks are not involved in computation (value 0) after blocks are divided. </li><li>If nullptr is passed, block sparsity computation is disabled, that is, the attention scores between all tokens are computed.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional (aclTensor*) </td>
    <td>Input</td>
    <td>Attention mask, that is, atten_mask in the formula. It is used to mask specific tokens that should not be involved in computation.</td>
    <td>Empty tensors are supported.<br>This parameter is not supported currently. nullptr should be passed.</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td rowspan="3">blockShapeOptional (aclIntArray*) </td>
    <td rowspan="3">Input</td>
    <td rowspan="3">Array of sparse block shapes. It specifies the two-dimensional size (number of rows and columns) of each sparse block.</td>
    <td> <ul><li>When blockSparseMaskOptional is configured: If this input is configured, the operator obtains the sparse block size from this input. If this input is not configured, the operator uses the default sparse block size [128, 128].</li></ul></td>
    <td rowspan="3">INT64</td>
    <td rowspan="3">-</td>
    <td rowspan="3">1</td>
    <td rowspan="3">-</td>
    </tr>
    <tr>
    <td><ul><li>If blockSparseMaskOptional is not configured, the operator ignores this parameter regardless of its setting.</li></ul></td>
    </tr>
    <tr>
    <td>The element requirements when this input is configured are as follows: <ul><li>The input must contain at least two elements [blockShapeX, blockShapeY]. </li><li>blockShapeX: block size in the Q direction. The value must be greater than 0. </li><li>blockShapeY: block size in the KV direction. The value must be greater than 0.</li></ul></td>
    </tr>
    <tr>
    <td rowspan="2">actualSeqLengthsOptional (aclIntArray*) </td>
    <td rowspan="2">Input</td>
    <td rowspan="2">Actual sequence length array of the query.<br>It is used to describe the number of valid query tokens in each batch in the variable-length sequence scenario (that is, the scenario with padding data).</td>
    <td>In the variable-length sequence scenario (when qInputLayout is set to TND), this input must be configured. Because the TND format is arranged in one dimension, the operator needs to use this array to accurately segment and define the actual boundaries of each sequence.</td>
    <td rowspan="2">INT64</td>
    <td rowspan="2">-</td>
    <td rowspan="2">1</td>
    <td rowspan="2">-</td>
    </tr>
    <tr>
    <td>Fixed-length/Variable-length scenario (when qInputLayout is set to "BNSD"): <ul><li>If this parameter is set, the operator processes data based on the specified valid length and ignores the padding data, improving performance. </li><li>If this parameter is not set (nullptr is passed), the operator uses the S dimension in the query shape as the valid length by default for full processing.</li></ul></td>
    </tr>
    <tr>
    <td rowspan="2">actualSeqLengthsKvOptional (aclIntArray*) </td>
    <td rowspan="2">Input</td>
    <td rowspan="2">Array of actual sequence lengths of keys and values.<br>It is used to describe the number of valid key/value tokens in each batch in the variable-length sequence scenario (that is, the scenario with padding data).</td>
    <td>Variable-length sequence scenario (when kvInputLayout is set to "TND"): This parameter must be configured. Because the TND format is one-dimensional and continuous, the operator needs to use this array to accurately segment and define the actual boundaries of each sequence.</td>
    <td rowspan="2">INT64</td>
    <td rowspan="2">-</td>
    <td rowspan="2">1</td>
    <td rowspan="2">-</td>
    </tr>
    <tr>
    <td>Fixed-length/Variable-length scenario (when kvInputLayout is set to "BNSD"): <ul><li>If this parameter is set, the operator processes data based on the specified valid length and ignores the padding data, improving performance. </li><li>If this parameter is not set (nullptr is passed), the operator uses the S dimension in the key/value shape as the valid length for full processing by default.</li></ul></td>
    </tr>
    <tr>
    <td>qInputLayout (char*) </td>
    <td>Input</td>
    <td>Data layout format of the query. It indicates the specific layout of the input tensor in the memory (for example, contiguous or combined axis layout).</td>
    <td>Currently, only "TND" and "BNSD" are supported. qInputLayout and kvInputLayout must be the same.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout (char*) </td>
    <td>Input</td>
    <td>Data layout format of the key and value. It indicates the specific layout of the input tensor in the memory.</td>
    <td>Currently, only "TND" and "BNSD" are supported. qInputLayout and kvInputLayout must be the same.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads (int64_t) </td>
    <td>Input</td>
    <td>Number of key/value attention heads. This parameter is used to support the head ratio mapping under the GQA (group query attention) mechanism.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType (int64_t) </td>
    <td>Input</td>
    <td>Mask type in attention calculation. Specifies the mask logic of a preset rule.</td>
    <td>Currently, only 0 can be transferred, indicating that no mask is added.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue (double) </td>
    <td>Input</td>
    <td>Scaling coefficient, that is, the scale in the formula. It is used for normalization of attention scores.</td>
    <td>Generally, this parameter is set to D<sup>–0.5</sup>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens (int64_t) </td>
    <td>Input</td>
    <td>Number of tokens included in the forward sliding window. It limits the number of historical tokens that can be used to calculate attention with the current token.</td>
    <td>This parameter is used in the sliding window attention scenario. Currently, sliding window attention is not supported. Only 2147483647 can be transferred.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens (int64_t) </td>
    <td>Input</td>
    <td>Number of tokens included in the backward sliding window. It limits the number of future tokens that can be used to calculate attention with the current token.</td>
    <td>This parameter is used in the sliding window attention scenario. Currently, sliding window attention is not supported. Only 2147483647 can be transferred.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dq (aclTensor*) </td>
    <td>Output</td>
    <td>Gradient output of query, that is, dq in the formula.</td>
    <td>Empty tensors are not supported.<br>The data type and shape are the same as those of the input query.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dk (aclTensor*) </td>
    <td>Output</td>
    <td>Gradient output of key, that is, dk in the formula.</td>
    <td>Empty tensors are not supported.<br>The data type and shape are the same as those of the input key.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dv (aclTensor*) </td>
    <td>Output</td>
    <td>Gradient output of value, that is, dv in the formula.</td>
    <td>Empty tensors are not supported.<br>The data type and shape are the same as those of the input value.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
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

* **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>The input pointers of dout, query, key, value, and attentionOut are null.</td>
    </tr>
    <tr>
      <td>When qInputLayout is set to TND, the input pointer of actualSeqLengthsOptional is null.</td>
    </tr>
    <tr>
      <td>When kvInputLayout is set to TND, the input pointer of actualSeqLengthsKvOptional is null.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of dout, query, key, and value are not supported.</td>
    </tr>
    <tr>
      <td>The input of qInputLayout or kvInputLayout is invalid, and the parameter validity check fails.</td>
    </tr>
  </tbody></table>

### aclnnBlockSparseAttentionGrad 

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnBlockSparseAttentionGradGetWorkspaceSize.</td>
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

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

* When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
* actualSeqLengthsOptional is mandatory when qInputLayout is set to TND. actualSeqLengthsKvOptional is mandatory when kvInputLayout is set to TND.
* The size of the head dimension in the shape of the query tensor is denoted as N1, and the size of the head dimension in the shape of the key and value tensors is denoted as N2. N1 must be greater than or equal to N2, and N1 % N2 must be equal to 0. (For example, in the BNSD layout, N1 corresponds to the second dimension of the query, and N2 corresponds to the second dimension of the key/value.)
* headdim=128.
* Currently, only BNSD and MHA (N1 == N2) are supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "../op_host/op_api/aclnn_block_sparse_attention_grad.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);
    
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
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. Initialize the device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Set core parameters (using the BNSD layout as an example).
    int32_t batch = 1;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t headDim = 128;
    int32_t blockShapeX = 64;
    int32_t blockShapeY = 64;

    // Calculate the number of blocks.
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilKv = (kvSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. Construct the tensor shape.
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kvShape = {batch, numKvHeads, kvSeqlen, headDim};
    std::vector<int64_t> lseShape = {batch, numHeads, qSeqlen}; // LSE Generally, there is no trailing dimension 1. This prevents GE squeezing.
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilKv};

    // 4. Allocate and initialize host data.
    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);

    // Initialize Q, K, and V to small numbers such as 0.1f.
    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kvSize, 0.1f);
    std::vector<op::fp16_t> vData(kvSize, 0.1f);
    
    // The initial value of the gradient can be a small positive number.
    std::vector<op::fp16_t> doutData(qSize, 0.01f);
    std::vector<op::fp16_t> outData(qSize, 0.1f);
    
    // LSE is set to a reasonable positive number, for example, 5.0f. In this way, exp(S - LSE) is a very safe negative exponent and will never overflow.
    std::vector<float> lseData(GetShapeSize(lseShape), 5.0f);
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 1);

    // Create all forward input/output aclTensors.
    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr;
    void *lseAddr = nullptr, *maskAddr = nullptr;
    
    aclTensor *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *doutTensor = nullptr, *outTensor = nullptr;
    aclTensor *lseTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(vData, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CreateAclTensor(doutData, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &doutTensor);
    CreateAclTensor(outData, qShape, &outAddr, aclDataType::ACL_FLOAT16, &outTensor);
    
    CreateAclTensor(lseData, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lseTensor); // uses FP32 strictly.
    CreateAclTensor(maskData, maskShape, &maskAddr, aclDataType::ACL_UINT8, &maskTensor); // uses UINT8 strictly.

    // 5. Create the backward output gradients (dq, dk, and dv).
    std::vector<op::fp16_t> dqData(qSize, 0.0f);
    std::vector<op::fp16_t> dkData(kvSize, 0.0f);
    std::vector<op::fp16_t> dvData(kvSize, 0.0f);
    
    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;

    CreateAclTensor(dqData, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dqTensor);
    CreateAclTensor(dkData, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dkTensor);
    CreateAclTensor(dvData, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dvTensor);

    // 6. Create the aclIntArray attribute parameter (BlockShape & ActualSeqLengths).
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());
    
    std::vector<int64_t> qSeqLenVec(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> kvSeqLenVec(batch, static_cast<int64_t>(kvSeqlen));
    aclIntArray *qSeqLenArr = aclCreateIntArray(qSeqLenVec.data(), batch);
    aclIntArray *kvSeqLenArr = aclCreateIntArray(kvSeqLenVec.data(), batch);

    // 7. Set scalar and string parameters.
    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    // Forcibly specify the maximum value of the sliding window.
    int64_t preTokens = 2147483647; 
    int64_t nextTokens = 2147483647;

    // 8. Call the first API in the sequence: GetWorkspaceSize.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnBlockSparseAttentionGradGetWorkspaceSize...\n");
    ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
        doutTensor, 
        qTensor, 
        kTensor, 
        vTensor, 
        outTensor, 
        lseTensor, 
        maskTensor,                 // blockSparseMaskOptional
        nullptr, // attenMaskOptional must be null.
        blockShapeArr, 
        qSeqLenArr, 
        kvSeqLenArr, 
        qLayoutBuffer, 
        kvLayoutBuffer, 
        static_cast<int64_t>(numKvHeads), 
        maskType, 
        scaleValue, 
        preTokens, 
        nextTokens, 
        dqTensor, 
        dkTensor, 
        dvTensor, 
        &workspaceSize, 
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 9. Allocate workspace.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 10. Call the second API to perform computation.
    LOG_PRINT("Calling aclnnBlockSparseAttentionGrad...\n");
    ret = aclnnBlockSparseAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 11. Synchronize the stream and wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 12. Copy the result to the host and print it.
    ret = aclrtMemcpy(dqData.data(), qSize * sizeof(op::fp16_t), dqAddr, qSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! Output results (first 10 elements of dQ):\n");
    for (uint64_t i = 0; i < 10 && i < dqData.size(); i++) {
        LOG_PRINT("  dQ index %lu: %f\n", i, static_cast<float>(dqData[i]));
    }

    // 13. Release all resources.
    LOG_PRINT("Cleaning up resources...\n");
    if (workspaceAddr) aclrtFree(workspaceAddr);
    
    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(doutAddr); aclrtFree(outAddr);
    aclrtFree(lseAddr); aclrtFree(maskAddr);
    aclrtFree(dqAddr); aclrtFree(dkAddr); aclrtFree(dvAddr);

    aclDestroyTensor(qTensor); aclDestroyTensor(kTensor); aclDestroyTensor(vTensor);
    aclDestroyTensor(doutTensor); aclDestroyTensor(outTensor);
    aclDestroyTensor(lseTensor); aclDestroyTensor(maskTensor);
    aclDestroyTensor(dqTensor); aclDestroyTensor(dkTensor); aclDestroyTensor(dvTensor);
    
    aclDestroyIntArray(blockShapeArr);
    aclDestroyIntArray(qSeqLenArr);
    aclDestroyIntArray(kvSeqLenArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BlockSparseAttentionGrad Test completed successfully!\n");
    return 0;
}
```
