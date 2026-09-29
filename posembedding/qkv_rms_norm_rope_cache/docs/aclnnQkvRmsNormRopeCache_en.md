# aclnnQkvRmsNormRopeCache

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT    |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- API function: Inputs the qkv fusion tensor, splits the q, k, and v tensors using SplitVD, performs the RmsNorm, ApplyRotaryPosEmb, Quant, and Scatter fusion operations, and outputs qOut, kCache, vCache, qBeforeQuant (optional), kBeforeQuant (optional), and vBeforeQuant (optional).
- The following table lists the scenarios supported by this API.

  |Scenario Type|Description|
  |:---|:---|
  |<ul><li>cacheMode is PA_NZ</li><li>q. </li><li>K and V support no quantization, symmetric quantization, and asymmetric quantization. </li><li>qBeforeQuant, kBeforeQuant, and vBeforeQuant are not output. </li></ul>|The qkv shape is [$B_{qkv}$ *$S_{qkv}$, $N_{qkv}$* $D_{qkv}$]. q, k, and v have the same D dimension. The mapping between the main calculation process and the output is as follows:<br><ul><li>SplitVD is performed on qkv to obtain q, k, and v.</li><li>RmsNorm and RoPE are performed on q to obtain qOut.</li><li>RmsNorm, RoPE, Quant (optional), and Scatter are performed on k to obtain kCache.</li><li>Quant (optional) and Scatter are performed on v to obtain vCache.</li></ul>|

- Formulas:

  (1) SplitVD:

  In the following formula, $N_q$, $N_k$, and $N_v$ indicate the number of attention heads of the q, k, and v components, respectively. The following conditions must be met:

  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] * D_{qkv}] \\
  k &= qkv[..., [N_q:-N_v] * D_{qkv}] \\
  v &= qkv[..., [-N_v:] * D_{qkv}]
  \end{aligned}
  $$

  (2) RmsNorm:

  Here, x and y indicate the input and output tensors of RmsNorm, respectively. The normalization is performed along the last dimension (feature dimension). This calculation rule is also applicable to the q and k components.

  $$
  squareX = x * x
  $$

  $$
  meanSquareX = squareX.mean(dim = -1, keepdim = True)
  $$

  $$
  rms = \sqrt{meanSquareX + epsilon}
  $$

  $$
  y = (x / rms) * gamma
  $$

  (3) RoPE (Half-and-Half):

  y here refers to the output result of the RmsNorm calculation.

  $$
  y1 = y[\ldots, :d/2]
  $$

  $$
  y2 = y[\ldots, d/2:]
  $$

  $$
  y\_RoPE = torch.cat((-y2, y1), dim = -1)
  $$
  
  $$
  y\_embed = (y * cos) + y\_RoPE * sin
  $$

  (4) Quant:

  Without quantization:

  $$
  kQuant = kRoPE \\
  vQuant = v
  $$

  Symmetric quantization:

  $$
  kQuant = kRoPE / kScale \\
  vQuant = v / vScale
  $$

  Asymmetric quantization:
  
  $$
  kQuant = kRoPE / kScale + kOffset \\
  vQuant = v / vScale + vOffset
  $$

## Prototype

Each operator is divided into [Two-Phase API](../../../docs/en/context/two_phase_api.md). You must call the aclnnQkvRmsNormRopeCacheGetWorkspaceSize API to obtain the input parameters, calculate the required workspace size based on the process, and then call the aclnnQkvRmsNormRopeCache API to perform computation.

```Cpp
aclnnStatus aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor   *qkv,
  const aclTensor   *qGamma,
  const aclTensor   *kGamma,
  const aclTensor   *cos,
  const aclTensor   *sin,
  const aclTensor   *index,
  aclTensor         *qOut,
  aclTensor         *kCache,
  aclTensor         *vCache,
  const aclTensor   *kScaleOptional,
  const aclTensor   *vScaleOptional,
  const aclTensor   *kOffsetOptional,
  const aclTensor   *vOffsetOptional,
  const aclIntArray *qkvSize,
  const aclIntArray *headNums,
  double             epsilon,
  char              *cacheModeOptional,
  const aclTensor   *qOutBeforeQuant,
  const aclTensor   *kOutBeforeQuant,
  const aclTensor   *vOutBeforeQuant,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnQkvRmsNormRopeCache(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnQkvRmsNormRopeCacheGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 392px">
  <col style="width: 228px">
  <col style="width: 101px">
  <col style="width: 100px">
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
      <td>qkv</td>
      <td>Input</td>
      <td>Inputs used to split the input data into q, k, and v, corresponding to qkv in the calculation formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>qkv</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qGamma</td>
      <td>Input</td>
      <td>Input data for calculating the rms_norm of q, corresponding to gamma in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv. </li><li>The shape is [D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kGamma</td>
      <td>Input</td>
      <td>Input data for calculating the rms_norm of k, corresponding to gamma in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv. </li><li>The shape is [D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for rope computation. It performs cosine transform on the input tensor, corresponding to cos in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, 1 * D_rope], D_rope = D<sub>qkv</sub>, and the number of bytes occupied by the D<sub>qkv</sub> * qkv data type must be exactly divided by 32.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Input data used for rope computation. It performs sine transform on the input tensor, corresponding to sin in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type and format are the same as those of the input cos.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Index position for writing data to the cache, corresponding to index in the calculation formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>]. The index value must be unique, and the value range is [–1, BlockNum x BlockSize].</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qOut</td>
      <td>Input and output</td>
      <td>Cache to be applied for. The input and output share the same address.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCache</td>
      <td>Input and output</td>
      <td>Cache requested for allocation. The input and output share the same address.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv (when k is not quantized), or INT8 (when k is quantized). </li><li>The shape is [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16] (when k is not quantized), or [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32] (when k is quantized).</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vCache</td>
      <td>Input and output</td>
      <td>Cache requested for allocation. The input and output share the same address.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of the input qkv (when v is not quantized), or INT8 (when v is quantized). </li><li>The shape is [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16] (when v is not quantized), or [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32] (when v is quantized).</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kScaleOptional</td>
      <td>Optional input</td>
      <td>Required when kCache is of the INT8 data type. It corresponds to kScale in the formula.</td>
      <td>The shape is [N<sub>k</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vScaleOptional</td>
      <td>Optional input</td>
      <td>Required when vCache is of the INT8 data type. It corresponds to vScale in the formula.</td>
      <td>The shape is [N<sub>v</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOffsetOptional</td>
      <td>Optional input</td>
      <td>Corresponds to kOffset in the formula.</td>
      <td><ul><li>This parameter is required when the kCache data type is INT8, the kScaleOptional input exists, and the quantization scenario is asymmetric quantization. </li><li>The shape is [N<sub>k</sub>, D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOffsetOptional</td>
      <td>Optional input</td>
      <td>Corresponds to vOffset in the calculation formula.</td>
      <td><ul><li>This parameter is required when the vCache data type is INT8, the vScaleOptional input is available, and the quantization mode is asymmetric quantization. </li><li>The shape is [N<sub>v</sub>, D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvSize</td>
      <td>Input</td>
      <td>Inputs the B, S, N, and D dimensions of the qkv matrix in the order of [B<sub>qkv</sub>, S<sub>qkv</sub>, N<sub>qkv</sub>, D<sub>qkv</sub>].</td>
      <td><ul><li>The value must be N<sub>qkv</sub> = N<sub>q</sub> + N<sub>k</sub> + N<sub>v</sub>. </li><li>The number of elements must be 4 and each element must be a valid positive integer.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNums</td>
      <td>Input</td>
      <td>The input is transferred in the order of [N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>]. It provides the specific size of the N dimension in the qkv component unit in the input parameter qkv matrix.</td>
      The value of <td>[N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>] must be the actual original value and cannot be reduced.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Used to prevent division by 0 during RMSNorm calculation. It corresponds to epsilon in the calculation formula.</td>
      <td>The recommended value is 1e-6</td>.
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>Input</td>
      <td>Format of the cache.</td>
      <td>The recommended value is "PA_NZ".</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qOutBeforeQuant</td>
      <td>Output</td>
      <td>Indicates the data to be written to qOut.</td>
      <td><ul><li>The data type and shape must be the same as those of the input parameter qkv. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOutBeforeQuant</td>
      <td>Output</td>
      <td>Indicates the data to be written to vCache, which is the intermediate computation result before quantization and Scatter.</td>
      <td><ul><li>The data type and shape must be the same as those of the input parameter qkv. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOutBeforeQuant</td>
      <td>Output</td>
      <td>Indicates the intermediate computation result that is about to be written to the vCache, before quantization and Scatter.</td>
      <td><ul><li>The data type and shape must be the same as those of the input parameter qkv. </li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
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
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="1">161001</td>
      <td>The input qkv, qGamma, kGamma, cos, or sin is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td>The input or output parameter dimension is not supported.</td>
    </tr>
    <tr>
      <td>The value of dim is not within the specified range.</td>
    </tr>
  </tbody></table>

## aclnnQkvRmsNormRopeCache

- **Parameters:**

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnQkvRmsNormRopeCacheGetWorkspaceSize.</td>
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

- Deterministic computation:
  - The default deterministic implementation of aclnnQkvRmsNormRopeCache is used.

- Input shape restrictions:
    * B<sub>qkv</sub> indicates the batch size of the input qkv, and S<sub>qkv</sub> indicates the sequence length of the input qkv. The size is determined by qkvSize.
    * N<sub>qkv</sub> indicates the number of input qkv heads. D<sub>qkv</sub> indicates the head dimension of the input qkv. Currently, only 128 is supported. D<sub>q</sub>, D<sub>k</sub>, and D<sub>k</sub> are the head dimensions of q, k, and v, respectively. It is required that D<sub>qkv</sub> = D<sub>q</sub> = D<sub>k</sub> = D<sub>v</sub> and D<sub>qkv</sub> be divisible by 32 (D<sub>qkv</sub>*qkv data type occupies bytes).
    * According to the rope rule, D<sub>k</sub> and D<sub>q</sub> must be even numbers. If cacheMode is set to PA_NZ, D<sub>k</sub> and D<sub>q</sub> must be 32-byte aligned, and BlockSize must be 32-byte aligned.
    * The alignment value in the preceding 32-byte alignment scenario is determined by the data type of the cache. Take BlockSize as an example. If the data type of the cache is int8, BlockSize must be a multiple of 32. If the data type of the cache is float16, BlockSize must be a multiple of 16. If the data types of kCache and vCache are different, BlockSize must be a multiple of both 32 and 16.
    * BlockNum indicates the number of memory blocks written to the cache. The size is determined by the user input scenario and must be BlockNum >= Ceil(S<sub>qkv</sub> / BlockSize) * B<sub>qkv</sub>.
    * The requireMemory parameter indicates the size of the space required for storing data. The following condition must be met: requireMemory >= (B<sub>qkv</sub> *S<sub>qkv</sub>* N<sub>qkv</sub> *D<sub>qkv</sub> + 2* D<sub>qkv</sub> + 2 *B<sub>qkv</sub>* S<sub>qkv</sub> *D<sub>qkv</sub> + B<sub>qkv</sub>* S<sub>qkv</sub> *N<sub>q</sub>* D<sub>qkv</sub> + BlockNum *BlockSize* N<sub>v</sub> *D<sub>qkv</sub> + BlockNum* BlockSize *N<sub>k</sub>* D<sub>qkv</sub>) *sizeof(FLOAT16) + B<sub>qkv</sub>* S<sub>qkv</sub> *sizeof(INT64) + (2* N<sub>k</sub> *D<sub>qkv</sub> + 2* N<sub>v</sub>) * sizeof(FLOAT). If the calculated value of requireMemory exceeds the total GM space of the current AI processor, this API cannot be used.
- Other restrictions:
    * For index, the value range of index must be [-1, BlockNum * BlockSize). The value cannot be repeated. When index is -1, the update is skipped.
    * kScaleOptional and vScaleOptional indicate the scaling factors for symmetric quantization. Therefore, if the parameters are passed, the values cannot be 0.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_qkv_rms_norm_rope_cache.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // (Fixed writing) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. Initialize the device or stream. For details, see the ACL API.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> qkvShape = {48, 2304};
    std::vector<int64_t> qGammaShape = {128};
    std::vector<int64_t> kGammaShape = {128};
    std::vector<int64_t> cosShape = {48, 128};
    std::vector<int64_t> sinShape = {48, 128};
    std::vector<int64_t> indexShape = {48};
    std::vector<int64_t> qOutShape = {48, 2048};
    std::vector<int64_t> kCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> vCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> kScaleShape = {1, 128};
    std::vector<int64_t> vScaleShape = {1, 128};
    std::vector<int64_t> qkv_size_list = {16, 3, 18, 128};
    std::vector<int64_t> head_nums_list = {16, 1, 1};

    std::vector<int16_t> qkvHostData(48 * 2304, 0);
    std::vector<int16_t> qGammaHostData(128, 0);
    std::vector<int16_t> kGammaHostData(128, 0);
    std::vector<int16_t> cosHostData(48 * 128, 0);
    std::vector<int16_t> sinHostData(48 * 128, 0);
    std::vector<int64_t> indexHostData(48, 0);
    std::vector<int16_t> qOutHostData(48 * 2048, 0);
    std::vector<int16_t> kCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<int16_t> vCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<int16_t> kScaleHostData(1 * 128, 0);
    std::vector<int16_t> vScaleHostData(1 * 128, 0);

    void* qkvDeviceAddr = nullptr;
    void* qGammaDeviceAddr = nullptr;
    void* kGammaDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* qOutDeviceAddr = nullptr;
    void* kCacheDeviceAddr = nullptr;
    void* vCacheDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* vScaleDeviceAddr = nullptr;

    aclTensor* qkv = nullptr;
    aclTensor* qGamma = nullptr;
    aclTensor* kGamma = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* index = nullptr;
    aclTensor* qOut = nullptr;
    aclTensor* kCache = nullptr;
    aclTensor* vCache = nullptr;
    aclTensor* kScale = nullptr;
    aclTensor* vScale = nullptr;

    aclIntArray *qkv_size = aclCreateIntArray(qkv_size_list.data(), qkv_size_list.size());
    aclIntArray *head_nums = aclCreateIntArray(head_nums_list.data(), head_nums_list.size());

    double epsilon = 1e-6;
    char* cacheMode = "PA_NZ";
    bool isOutputQkv = false;

    ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT16, &qkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qGammaHostData, qGammaShape, &qGammaDeviceAddr, aclDataType::ACL_FLOAT16, &qGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kGammaHostData, kGammaShape, &kGammaDeviceAddr, aclDataType::ACL_FLOAT16, &kGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qOutHostData, qOutShape, &qOutDeviceAddr, aclDataType::ACL_FLOAT16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHostData, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_INT8, &kCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vCacheHostData, vCacheShape, &vCacheDeviceAddr, aclDataType::ACL_INT8, &vCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kScaleHostData, kScaleShape, &kScaleDeviceAddr, aclDataType::ACL_FLOAT, &kScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vScaleHostData, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first part of the aclnnQkvRmsNormRopeCache API.
    ret = aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
        qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScale, vScale, nullptr, nullptr, qkv_size,
        head_nums, epsilon, cacheMode, nullptr, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second API of aclnnQkvRmsNormRopeCache.
    ret = aclnnQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(kCacheShape, &kCacheDeviceAddr);
    PrintOutResult(vCacheShape, &vCacheDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(qkv);
    aclDestroyTensor(qGamma);
    aclDestroyTensor(kGamma);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(index);
    aclDestroyTensor(qOut);
    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kScale);
    aclDestroyTensor(vScale);

    // 7. Release device resources.
    aclrtFree(qkvDeviceAddr);
    aclrtFree(qGammaDeviceAddr);
    aclrtFree(kGammaDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(qOutDeviceAddr);
    aclrtFree(kCacheDeviceAddr);
    aclrtFree(vCacheDeviceAddr);
    aclrtFree(kScaleDeviceAddr);
    aclrtFree(vScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
