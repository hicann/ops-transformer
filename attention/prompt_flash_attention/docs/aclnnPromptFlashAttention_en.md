# aclnnPromptFlashAttention

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference accelerator cards</term>|      √    |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: FlashAttention operator in the full inference scenario.

- Formula:

    Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to obtain three $n × d$ matrices.

    The computation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention."

    $$
     Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the `Softmax` function is used, instead of the `Score` function. The self-attention computation formula is as follows:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    The product of $Q$ and $K^T$ represents the attention to the input $x$. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. The result is then multiplied by $V$ to produce an $n × d$ matrix.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnPromptFlashAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnPromptFlashAttention` is called to perform computation.

```cpp
aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *pseShift,
    const aclTensor   *attenMask,
    const aclIntArray *actualSeqLengths,
    int64_t            numHeads,
    double             scaleValue,
    int64_t            preTokens,
    int64_t            nextTokens,
    char             *inputLayout,
    int64_t           numKeyValueHeads,
    const aclTensor  *attentionOut,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttention(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnPromptFlashAttentionGetWorkspaceSize

- **Parameters**
  
  <div style="overflow-x: auto;">
   <table style="undefined;table-layout: fixed; width: 1567px"><colgroup> 
   <col style="width: 170px"> 
   <col style="width: 120px"> 
   <col style="width: 300px"> 
   <col style="width: 330px"> 
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
    <th>Usage Notes</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (Shape)</th>
    <th>Non-contiguous Tensor</th>
  </tr></thead>
  <tbody>
  <tr>
    <td>query</td>
    <td>Input</td>
    <td>Input <code>Q</code> in the formula.</td>
    <td>The data type must be the same as that of <code>key</code> and <code>value</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3–4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>key</td>
    <td>Input</td>
    <td>Input <code>K</code> in the formula.</td>
    <td>The data type must be the same as that of <code>query</code> and <code>value</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3–4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>Input <code>V</code> in the formula.</td>
    <td>The data type must be the same as that of <code>query</code> and <code>key</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3–4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>pseShift</td>
    <td>Input</td>
    <td>Positional encoding.</td>
    <td>This parameter is reserved and not used currently. This parameter is forcibly set to <code>nullptr</code>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>attenMask</td>
    <td>Input</td>
    <td>Mask matrix.</td>
    <td><ul><li>If this parameter is not used, pass <code>nullptr</code>.</li></ul>
        <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>BOOL, INT8, UINT8</td>
    <td>ND</td>
    <td>2–4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>actualSeqLengths</td>
    <td>Input</td>
    <td>Valid sequence lengths of <code>query</code> in different batches.</td>
    <td><ul><li>If no sequence length is specified, pass <code>nullptr</code>.</li></ul>
        <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numHeads</td>
    <td>Input</td>
    <td>Number of heads in <code>query</code>.</td>
    <td>Restrictions: In the BNSD/NSD scenario, the value must be the same as the N-axis value of <code>query</code> in the shape.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>Input</td>
    <td>Reciprocal of the square root of <code>d</code> in the formula.</td>
    <td><ul><li>Its data type must be compatible with that of <code>query</code> according to the type deduction rules. </li></ul>
        <ul><li>If no specific value is required, <code>1.0</code> is recommended. </li></ul></td>
    <td>DOUBLE</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>preTokens</td>
    <td>Input</td>
    <td>Number of preceding tokens to associate in attention computation.</td>
    <td><ul><li>If no specific value is required, <code>2147483647</code> is recommended.</li></ul>
        <ul><li>Negative numbers are supported.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>nextTokens</td>
    <td>Input</td>
    <td>Number of succeeding tokens to associate in attention computation.</td>
    <td><ul><li>If no specific value is required, <code>0</code> is recommended.</li></ul>
        <ul><li>Negative numbers are supported.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>inputLayout</td>
    <td>Input</td>
    <td>Layout of the input <code>query</code>, <code>key</code>, and <code>value</code>.</td>
    <td><ul><li>If no specific value is required, <code>BSH</code> is recommended.</li></ul>
        <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>CHAR</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numKeyValueHeads</td>
    <td>Input</td>
    <td>Number of heads in <code>key</code> and <code>value</code>.</td>
    <td><ul><li>If no specific value is required, <code>0</code> is recommended, indicating that <code>key</code>/<code>value</code> and <code>query</code> have the same number of heads.</li></ul>
        <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>Output</td>
    <td>Output in the formula.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3–4</td>
    <td>-</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be allocated on the device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  </tbody></table>
  </div>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
    <tr>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type or data format of <code>query</code>, <code>key</code>, <code>value</code>, <code>pseShift</code>, <code>attenMask</code>, or <code>attentionOut</code> is not supported.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>An exception occurred when the NPU runtime API was called.</td>
    </tr>
    </tbody>
    </table>
    </div>

## aclnnPromptFlashAttention

- **Parameters**
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnPromptFlashAttentionV3GetWorkspaceSize</code>.</td>
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
    </div>
- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnPromptFlashAttention` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Processing logic for a null input parameter: The operator checks whether `query` is a null pointer. If so, an error is reported. If `query` is not an empty tensor but `key` and `value` are empty tensors (that is, `S2` is 0), `attentionOut` is filled with all zeros. If `attentionOut` is an empty tensor, the framework handles it. For other input parameters which support the passing of `nullptr` as described in the preceding parameter description, no processing is performed when they are null pointers.
- The data layout of `query`, `key`, and `value` can be interpreted from multiple dimensions. To be specific, `B` (`Batch`) indicates the size of an input sample batch, `S` (`Seq-Length`) indicates the length of the input sample sequence, `H` (`Head-Size`) indicates the size of the hidden layer, `N` (`Head-Num`) indicates the number of heads, and `D` (`Head-Dim`) indicates the minimum unit size of the hidden layer (`D` = `H`/`N`).

- Restrictions on `query`, `key`, and `value`:

  - Input shape restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>:

      - The B axis must be less than or equal to `65536` (64K). If the input type is INT8 and the D axis is not 32-byte aligned, or the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the B axis can be up to `128`.

      - The N axis must be less than or equal to `256`.

      - The S axis must be less than or equal to `20971520` (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator execution may time out (an AI Core error is reported, and `errorStr` is `timeout or trap error`). In this case, S axis splitting is recommended. Note: The computation load is affected by parameters such as `B`, `S`, `N`, and `D`. Larger values indicate larger computation loads. The following lists some typical scenarios with long sequences (that is, the product of `B`, `S`, `N`, and `D` is large).
        <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 200px">
        </colgroup><thead>
        <tr>
        <th><code>B</code></th>
        <th><code>Q_N</code></th>
        <th><code>Q_S</code></th>
        <th><code>D</code></th>
        <th><code>KV_N</code></th>
        <th><code>KV_S</code></th>
        </tr></thead>
        <tbody>
        <tr>
        <td>1</td>
        <td>20</td>
        <td>2097152</td>
        <td>256</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        <tr>
        <td>1</td>
        <td>2</td>
        <td>20971520</td>
        <td>256</td>
        <td>2</td>
        <td>20971520</td>
        </tr>
        <tr>
        <td>20</td>
        <td>1</td>
        <td>2097152</td>
        <td>256</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        <tr>
        <td>1</td>
        <td>10</td>
        <td>2097152</td>
        <td>512</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        </tbody>
        </table>
      - The D axis must be less than or equal to `512`. If `inputLayout` is `BSH` or `BSND`, `N × D` must be less than `65535`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: Constraints on the input of `query`, `key`, and `value` in the TND scenario:
        - `T` is less than or equal to `65536`.
        - `N` is `8`, `16`, `32`, `64`, or `128`, and `Q_N`, `K_N`, and `V_N` are equal.
        - `Q_D` and `K_D` are `192`, and `V_D` is `128` or `192`.
        - The data type is BFLOAT16.
        - The sparse mode can only be `0` without a mask or `3` with a mask.
        - When the sparse mode is `3`, `actualSeqLengths` must be less than `actualSeqLengthsKv` for each batch.
    - <term>Atlas inference accelerator cards</term>:
        - The B axis must be less than or equal to `128`.
        - The N axis must be less than or equal to `256`.
        - The S axis must be less than or equal to `65535` (64K). `atten_mask` cannot be configured if `Q_S` or `KV_S` is not a multiple of 128 or `Q_S` and `KV_S` have different values.
        - The D axis must be less than or equal to `512`.
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
    - <term>Atlas inference accelerator cards</term>: The data type can only be FLOAT16.
- Restrictions on `pseShift`:
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
    - <term>Atlas inference accelerator cards</term>: The value can only be `nullptr`.
- Restrictions on `attenMask`:
  - Input shape restrictions: Recommended shapes are `Q_S, KV_S`, `B, Q_S, KV_S`, `1, Q_S, KV_S`, `B, 1, Q_S, KV_S`, and `1, 1, Q_S, KV_S`. `Q_S` is `S` in the shape of `query`, and `KV_S` is `S` in the shape of `key` and `value`.
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be BOOL, INT8, or UINT8.
    - <term>Atlas inference accelerator cards</term>: The data type can only be BOOL.
  - Other restrictions: In the scenario where `KV_S` of `attenMask` is not 32-byte aligned, it is recommended that it be padded to 32 bytes to improve the performance, filling excess positions with ones.
- Restrictions on the input of `actualSeqLengths`:
  - Input value range restrictions: The valid sequence length of each batch in the input parameter should be less than or equal to the sequence length of the corresponding batch in `query`.
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be INT64.
    - <term>Atlas inference accelerator cards</term>: The data type can be INT64.
- Restrictions on the input of `preTokens`:
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be INT64.
    - <term>Atlas inference accelerator cards</term>: The value can only be `2147483647`.
- Restrictions on the input of `nextTokens`:
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be INT64.
    - <term>Atlas inference accelerator cards</term>: The value can only be `0` or `2147483647`.
- Restrictions on the input of `inputLayout`:
  - Input data type restrictions:
    - The input format can be `BSH`, `BSND`, `BNSD`, or `BNSD_BSND`. When the input format is `BNSD`, the output format is `BSND`. If no specific value is required, `BSH` is recommended.
- Restrictions on the input of `numKeyValueHeads`:
  - Input attribute restrictions: `numHeads` must be divisible by `numKeyValueHeads`, and in BSND, BNSD, BNSD_BSND scenarios, it must match the N-axis value of `key`/`value` in the shape. Otherwise an error is reported.
  - Input data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can only be INT64.
    - <term>Atlas inference accelerator cards</term>: The value can only be `0`.
- Restrictions on the input of `attentionOut`:
  - Shape restrictions: When `inputLayout` is set to `BNSD_BSND`, the shape of the input `query` is BNSD and the output shape is BSND. In other cases, the shape of this input parameter must be the same as that of the input parameter `query`.
  - Data type restrictions:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
    - <term>Atlas inference accelerator cards</term>: The data type can only be FLOAT16.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"

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
  // (Boilerplate) Initialize resources.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
  std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);

  // Create a query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an atten aclTensor.
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  int64_t numKeyValueHeads = numHeads;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  int64_t preTokens = 65535;
  int64_t nextTokens = 65535;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnPromptFlashAttentionGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, numHeads, scaleValue, 
                                                  preTokens, nextTokens, layerOut, numKeyValueHeads, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnPromptFlashAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttention failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<double> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release resources.
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
