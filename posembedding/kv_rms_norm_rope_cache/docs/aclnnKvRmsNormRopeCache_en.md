# aclnnKvRmsNormRopeCache

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/kv_rms_norm_rope_cache)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Splits the tail axis of the input tensor (kv) into the left half for rms_norm computation and the right half for RoPE computation, and then scatters the computation results to two caches.

- Formulas:
  
  (1) interleaveRope:

  $$
  x=kv[...,Dv:]
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y=x\_part1*cos+x\_part2*sin
  $$

  (2) rmsNorm:

  $$
  x=kv[...,:Dv]
  $$

  $$
  square\_x=x*x
  $$

  $$
  mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
  $$

  $$
  rms=torch.sqrt(mean\_square\_x+epsilon)
  $$

  $$
  y=(x/rms)*gamma
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnKvRmsNormRopeCacheGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnKvRmsNormRopeCache` is called to perform computation.

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor* kv, 
  const aclTensor* gamma, 
  const aclTensor* cos, 
  const aclTensor* sin, 
  const aclTensor* index, 
  aclTensor*       kCacheRef, 
  aclTensor*       ckvCacheRef, 
  const aclTensor* kRopeScaleOptional, 
  const aclTensor* ckvScaleOptional, 
  const aclTensor* kRopeOffsetOptional, 
  const aclTensor* cKvOffsetOptional, 
  double           epsilon, 
  char*            cacheModeOptional, 
  bool             isOutputKv, 
  aclTensor*       kRopeOut, 
  aclTensor*       cKvOut,
  uint64_t         workspaceSize, 
  aclOpExecutor*   executor)
```

```Cpp
aclnnStatus aclnnKvRmsNormRopeCache(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnKvRmsNormRopeCacheGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 403px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
  <col style="width: 146px">
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
      <td>kv</td>
      <td>Input</td>
      <td>Input data used to split the data required for rms_norm computation (Dv) and the data required for RoPE computation (Dk) in the formula.</td>
      <td><ul><li>The shape supports only four dimensions: [Bkv, N, Skv, D]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Input data used for rms_norm computation in the formula.</td>
      <td><ul><li>The data type is the same as that of the KV data, and the shape is one-dimensional [Dv,]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for RoPE calculation in the formula, which performs cosine transformation on the input tensor Dk.</td>
      <td><ul><li>The data type is the same as that of the KV pair. The shape is 4-dimensional, [Bkv, 1, Skv, Dk] or [Bkv, 1, 1, Dk]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Optional input</td>
      <td>Input data used for RoPE calculation in the formula, which performs sine transformation on the input tensor Dk.</td>
      <td><ul><li>The data type is the same as that of the KV pair. The shape is 4-dimensional, [Bkv, 1, Skv, Dk] or [Bkv, 1, 1, Dk], and is the same as that of cos. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Index position for writing to the cache. If the value of index is -1, the update is skipped.</td>
      <td><ul><li>When cacheModeOptional is set to Norm, the shape is two-dimensional [Bkv, Skv]. </li><li>When cacheModeOptional is set to PA_BNSD or PA_NZ, the shape is one-dimensional [Bkv * Skv]. </li><li>When cacheModeOptional is set to PA_BLK_BNSD or PA_BLK_NZ, the shape is one-dimensional [Bkv * ceil_div(Skv, BlockSize)]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCacheRef</td>
      <td>Input/Output</td>
      <td>The cache is allocated in advance. The input and output share the same address.</td>
      <td><ul><li>In non-quantization scenarios, the data type is the same as that of the input KV. </li><li>In quantization scenarios, the data type is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN. </li><li>When cacheModeOptional is set to PA (PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the shape is four-dimensional [BlockNum, BlockSize, N, Dk]. </li><li>When cacheModeOptional is set to Norm, the shape is four-dimensional [Bcache, N, Scache, Dk]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvCacheRef</td>
      <td>Input/Output</td>
      <td>The cache that is applied for in advance. The input and output share the same address.</td>
      <td><ul><li>In non-quantization scenarios, the data type is the same as that of the input KV. </li><li>In quantization scenarios, the data type is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN. </li><li>When cacheModeOptional is set to PA (PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the shape is 4-dimensional [BlockNum, BlockSize, N, Dv]. </li><li>When cacheModeOptional is set to Norm, the shape is 4-dimensional [Bcache, N, Scache, Dv]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeScaleOptional</td>
      <td>Input</td>
      <td>This input parameter is required when the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN.</td>
      <td><ul><li>The shape is 2-dimensional [N, Dk], 1-dimensional [Dk,], or 1-dimensional [1,]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvScaleOptional</td>
      <td>Input</td>
      <td>This input is required when the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN.</td>
      <td><ul><li>The shape is 2-dimensional [N, Dv], 1-dimensional [Dv,], or 1-dimensional [1,]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeOffsetOptional</td>
      <td>Input</td>
      <td>This input is required when the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN, the corresponding kRopeScaleOptional input exists, and the quantization scenario is asymmetric quantization.</td>
      <td><ul><li>The shape is 2-dimensional [N, Dk], 1-dimensional [Dk,], or 1-dimensional [1,]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOffsetOptional</td>
      <td>Input</td>
      <td>This parameter is required when the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN, the corresponding ckvScaleOptional input exists, and the quantization scenario is asymmetric quantization.</td>
      <td><ul><li>The shape can be 2-dimensional [N, Dv], 1-dimensional [Dv,], or 1-dimensional [1,]. </li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Attribute</td>
      <td>The rms_norm calculation prevents division by zero.</td>
      <td>1e-5 is recommended.</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>Attribute</td>
      <td>Cache format selection flag.</td>
      <td>The value can be Norm, PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ. Norm is recommended.</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isOutputKv</td>
      <td>Attribute</td>
      <td>Output control flag of kRopeOut and cKvOut.</td>
      <td>When isOutputKv is true, kRopeOut and cKvOut need to be output. It is recommended that this parameter be set to false.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeOut</td>
      <td>Output</td>
      <td>The value is determined by isOutputKv. When isOutputKv is true, the output is required. The data type is the same as that of the input kv.</td>
      <td>The shape is 4-dimensional [Bkv, N, Skv, Dk].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOut</td>
      <td>Output</td>
      <td>The value is determined by isOutputKv. When isOutputKv is true, the output is required. The data type is the same as that of the input kv.</td>
      <td>The shape is 4-dimensional [Bkv, N, Skv, Dv].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 325px">
  <col style="width: 126px">
  <col style="width: 809px">
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
      <td>The input and output tensors are null pointers.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>The shape of the kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional and cKvOffsetOptional parameters is invalid.</td>
    </tr>
    <tr>
      <td>The dtype of the kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional and cKvOffsetOptional parameters is invalid.</td>
    </tr>
    <tr>
      <td>In the PA scenario (cacheModeOptional is set to PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the value of the BlockSize dimension of the cache is invalid.</td>
    </tr>
    <tr>
      <td>In the NZ scenario (cacheModeOptional is set to PA_NZ or PA_BLK_NZ), the values of the Dk and Dv dimensions are invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnKvRmsNormRopeCache

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnKvRmsNormRopeCacheGetWorkspaceSize.</td>
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

- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

* The shape-related variables used in the parameter descriptions are defined as follows:
  * Bkv indicates the batch size of the input KV, and Skv indicates the sequence length of the input KV. The size is determined by the user input scenario and is not limited.
  * N indicates the number of heads of the input KV. This operator is closely related to the DeepSeekV3 network structure and supports only the scenario where N is 1.
  * D indicates the head dimension of the input KV. The data Dv required for rms_norm computation and the data Dk required for RoPE computation are obtained by splitting the input KV data D. Therefore, the size of Dk and Dv must meet the following requirement: Dk + Dv = D. In addition, Dk must meet the rope rules. According to the rope rules, Dk must be an even number. If cacheModeOptional is set to NZ (cacheModeOptional is PA_NZ or PA_BLK_NZ), Dk and Dv must be 32-byte aligned.
  * If cacheModeOptional is set to PA (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), BlockSize must be 32-byte aligned.
  * The alignment value in the preceding 32-byte alignment scenarios is determined by the data type of the cache. Take BlockSize as an example. If the data type of the cache is INT8, HIFLOAT8, FLOAT8E5M2 or FLOAT8E4M3FN, BlockSize%32 must be equal to 0. If the data type of the cache is float16 or bfloat16, BlockSize%16 must be equal to 0. If the dtype of kCacheRef is different from that of ckvCacheRef, BlockSize must meet both BlockSize%32 = 0 and BlockSize%16 = 0.
  * Bcache is the batch size of the input cache, and Scache is the sequence length of the input cache. The size is determined by the user input scenario and is not limited.
  * BlockNum is the number of memory blocks written to the cache. The size is determined by the user input scenario and is not limited.
* Restrictions on index:
  * When cacheModeOptional is set to Norm, the shape is two-dimensional [Bkv, Skv], and the value of index must be in the range of [–1, Scache). The value can be repeated under different Bkv.
  * When cacheModeOptional is set to PA_BNSD or PA_NZ, the shape is one-dimensional [Bkv *Skv], and the value of index must be in the range of [–1, BlockNum* BlockSize). The value cannot be repeated.
  * When cacheModeOptional is set to PA_BLK_BNSD or PA_BLK_NZ, the shape is one-dimensional [Bkv *ceil_div(Skv, BlockSize)], and the value of index must be in the range of [–1, BlockNum* BlockSize). The value/BlockSize cannot be repeated.
* Restrictions on quantization scenarios:
  * Supported quantization scenario 1: The data type of kCacheRef is FLOAT16 or BFLOAT16, and the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.
  * Supported quantization scenario 2: The data type of ckvCacheRef is FLOAT16 or BFLOAT16, and the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.
  * Supported quantization scenario 3: The data types of kCacheRef and ckvCacheRef are the same, which are INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache.h"

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
  std::vector<int8_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> kvShape = {181,1,1,576};
  std::vector<int64_t> gammaShape = {512,};
  std::vector<int64_t> cosShape = {181,1,1,64};
  std::vector<int64_t> sinShape = {181,1,1,64};
  std::vector<int64_t> indexShape = {181,1};
  std::vector<int64_t> kpeCacheShape = {181,1,1,64};
  std::vector<int64_t> ckvCacheShape = {181,1,1,512};
  std::vector<int64_t> kRopeShape = {181,1,1,64};
  std::vector<int64_t> cKvShape = {181,1,1,512};
  
  std::vector<int16_t> kvHostData(181*1*1*576,0);
  std::vector<int16_t> gammaHostData(512,0);
  std::vector<int16_t> cosHostData(181*1*1*64,0);
  std::vector<int16_t> sinHostData(181*1*1*64,0);
  std::vector<int64_t> indexHostData(181*1,0);
  std::vector<int16_t> kpeCacheHostData(181*1*1*64,0);
  std::vector<int16_t> ckvCacheHostData(181*1*1*512,0);
  std::vector<int16_t> kRopeHostData(181*1*1*64,0);
  std::vector<int16_t> cKvHostData(181*1*1*512,0);
  
  void* kvDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* kpeCacheDeviceAddr = nullptr;
  void* ckvCacheDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* cKvDeviceAddr = nullptr;
 
  aclTensor* kv = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* index = nullptr;
  aclTensor* kpeCache = nullptr;
  aclTensor* ckvCache = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* cKv = nullptr;
 

  double epsilon = 1e-5;
  char cacheMode[] = "Norm";
  bool isOutputKv = false;

  ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT16, &kv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kpeCacheHostData, kpeCacheShape, &kpeCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kpeCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(ckvCacheHostData, ckvCacheShape, &ckvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &ckvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cKvHostData, cKvShape, &cKvDeviceAddr, aclDataType::ACL_FLOAT16, &cKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnKvRmsNormRopeCache.
  ret = aclnnKvRmsNormRopeCacheGetWorkspaceSize(kv,gamma,cos,sin,index, 
                                                kpeCache,ckvCache,nullptr,nullptr,nullptr,nullptr,epsilon,cacheMode,isOutputKv,kRope,cKv,&workspaceSize,&executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnKvRmsNormRopeCache.
  ret = aclnnKvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(kpeCacheShape, &kpeCacheDeviceAddr);
  PrintOutResult(ckvCacheShape, &ckvCacheDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(kv);
  aclDestroyTensor(gamma);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(index);
  aclDestroyTensor(kpeCache);
  aclDestroyTensor(ckvCache);
  aclDestroyTensor(kRope);
  aclDestroyTensor(cKv);

  // 7. Release device resources.
  aclrtFree(kvDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(kpeCacheDeviceAddr);
  aclrtFree(ckvCacheDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(cKvDeviceAddr);


  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
