# aclnnDequantRopeQuantKvcache

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/dequant_rope_quant_kvcache)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Description

- Description: Performs an optional dequantization on the input tensor (x), then splits it along the last axis into q, k, and vOut according to the specified sizeSplits. It applies Rotary Positional Embedding (RoPE) to q and k to generate qOut and kOut. Finally, kOut and vOut are quantized and updated into kCacheRef and vCacheRef based on the provided indices.

- Formula:
  
  $$
  dequantX = Dequant(x,weightScaleOptional,activationScaleOptional,biasOptional)
  $$
  
  $$
  q,k,vOut = SplitTensor(dequantX,dim=-1,`sizeSplits`)
  $$
  
  $$
  qOut,kOut = ApplyRotaryPosEmb(q,k,cos,sin)
  $$
  
  $$
  quantK = Quant(kOut,scaleK,offsetKOptional)
  $$
  
  $$
  quantV = Quant(vOut,scaleV,offsetVOptional)
  $$
  
  If cacheModeOptional is contiguous:
  
  $$
  kCacheRef[i][indice[i]]=quantK[i]
  $$
  
  $$
  vCacheRef[i][indice[i]]=quantV[i]
  $$
  
  If cacheModeOptional is page:
  
  $$
  kCacheRefView=kCacheRef.view(-1,kCacheRef[-2],kCacheRef[-1])
  $$
  
  $$
  vCacheRefView=vCacheRef.view(-1,vCacheRef[-2],vCacheRef[-1])
  $$
  
  $$
  kCacheRefView[indices[i]]=quantK[i]
  $$
  
  $$
  vCacheRefView[indices[i]]=quantV[i]
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnDequantRopeQuantKvcacheGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnDequantRopeQuantKvcache` is called to perform computation.

```Cpp
aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
  const aclTensor   *x, 
  const aclTensor   *cos, 
  const aclTensor   *sin, 
  aclTensor         *kCacheRef, 
  aclTensor         *vCacheRef, 
  const aclTensor   *indices, 
  const aclTensor   *scaleK, 
  const aclTensor   *scaleV, 
  const aclTensor   *offsetKOptional, 
  const aclTensor   *offsetVOptional, 
  const aclTensor   *weightScaleOptional, 
  const aclTensor   *activationScaleOptional, 
  const aclTensor   *biasOptional, 
  const aclIntArray *sizeSplits, 
  char              *quantModeOptional, 
  char              *layoutOptional, 
  bool               kvOutput, 
  char              *cacheModeOptional, 
  const aclTensor   *qOut, 
  const aclTensor   *kOut, 
  const aclTensor   *vOut, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnDequantRopeQuantKvcache(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnDequantRopeQuantKvcacheGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
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
    </tr></thead>
    <tbody>
    <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input x used for splitting in the formula.</td>
        <td>The shape is [B, S, H] or [B, H], where H = (Nq + Nkv + Nkv) x D. The last axis of `x` is less than or equal to 4096 and is 64-pixel aligned.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>cos</td>
        <td>Input</td>
        <td>Input cos used for positional encoding in the formula.</td>
        <td>When x is 3-dimensional, the shape is [B, S, 1, D]. When x is 2-dimensional, the shape is [B, D].</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2, 4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sin</td>
        <td>Input</td>
        <td>Input sin used for positional encoding in the formula.</td>
        <td>When x is 3-dimensional, the shape is [B, S, 1, D]. When x is 2-dimensional, the shape is [B, D].</td>
        <td>Same as cos</td>
        <td>ND</td>
        <td>2, 4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>kCacheRef</td>
        <td>Input</td>
        <td>Input kCacheRef used for caching k in the formula.</td>
        <td>The shape is [C_1, C_2, Nkv, D].</td>
        <td>INT8</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>vCacheRef</td>
        <td>Input</td>
        <td>Input vCacheRef used to cache v in the formula.</td>
        <td>The shape is [C_1, C_2, Nkv, D].</td>
        <td>INT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Input indices that indicate the token position information of Kvcache in the formula.</td>
        <td>When cache_mode is set to page and x is 3-dimensional, the shape is [B*S]. Otherwise, the shape is [B].</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scaleK</td>
        <td>Input</td>
        <td>Input scaleK in the formula is used to quantize the scale factor of k.</td>
        <td>When cache_mode is set to page and x is 3-dimensional, the shape is [B*S]. Otherwise, the shape is [B].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scaleV</td>
        <td>Input</td>
        <td>scaleV in the formula is used to quantize the scale factor of v.</td>
        <td>The shape is [Nkv, D]</td>.
        <td>FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>offsetKOptional</td>
        <td>Input</td>
        <td>offsetKoptional in the formula is used to quantize the offset factor of k.</td>
        <td>The shape is [Nkv, D].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>offsetVOptional</td>
        <td>Input</td>
        <td>offsetVOptional in the formula is used to quantize the offset factor of v.</td>
        <td>The shape is [Nkv, D].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>weightScaleOptional</td>
        <td>Input</td>
        <td>weightScaleoptional in the formula is used to quantize the weight scale factor.</td>
        <td>The shape is [H].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>activationScaleOptional</td>
        <td>Input</td>
        <td>The input activationScaleOptional in the formula is used to dequantize the activation scale factor.</td>
        <td>If x is 3-dimensional, the shape is [B*S]. If x is 2-dimensional, the shape is [B].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>The input in the formula is used to dequantize the biasOptional.</td>
        <td>The shape is [H].</td>
        <td>FLOAT, FLOAT16, INT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sizeSplits</td>
        <td>Input</td>
        <td>Indicates the length of the input qkv split.</td>
        <td>The size is 3, and the value is [Nq*D, Nkv*D, Nkv*D].</td>
        <td>AclIntArray</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantModeOptional</td>
        <td>Input</td>
        <td>Indicates the supported quantization type.</td>
        <td>Currently, only static is passed.</td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutOptional</td>
        <td>Input</td>
        <td>Indicates the supported data format.</td>
        <td>Currently, only BSND is supported.</td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvOutput</td>
        <td>Input</td>
        <td>Indicates the supported data format.</td>
        <td>Currently, only BSND is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cacheModeOptional</td>
        <td>Input</td>
        <td>Indicates the update mode of kCacheRef.</td>
        <td>Currently, only "page" and "contiguous" are supported. The default value is "contiguous".</td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
    </tbody></table>

- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 119px">
  <col style="width: 762px">
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
  </tbody>
  </table>

## aclnnDequantRopeQuantKvcache

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnDequantRopeQuantKvcacheGetWorkspaceSize.</td>
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

1. Deterministic computing:
     - `aclnnDequantRopeQuantKvcache` defaults to a deterministic implementation.

2. When cacheModeOptional is set to contiguous, the 0th dimension of kCacheRef is greater than the 0th dimension of x, and the indices data value is greater than or equal to 0 and less than or equal to the first dimension of vCacheRef (s in the [b, s, n, d] format) minus the first dimension of x.
3. When cacheModeOptional is set to page, the indices data value is greater than or equal to 0, less than the product of the 0th and 1st dimensions of kCacheRef, and is unique.
4. If the input x is not of type INT32, the data types of x, cos, and sin are the same as those of the outputs qOut, kOut, and vOut. In this case, activationScaleOptional, weightScaleOptional, and biasOptional do not take effect.
5. If the input x is of type INT32, the data types of cos and sin are the same as those of the outputs qOut, kOut, and vOut. In this case, weightScaleOptional is required, and activationScaleOptional and biasOptional are optional. (The data type of biasOptional does not need to be the same as that of other inputs.)
6. The last axis of `x` is less than or equal to 4096 and is 64-pixel aligned.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_rope_quant_kvcache.h"

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
    LOG_PRINT("mean result[%ld] is: %d\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> inputShape = {320, 1, 1280};
  std::vector<int64_t> cosShape = {320, 1, 1, 128};
  std::vector<int64_t> sinShape = {320, 1, 1, 128};
  std::vector<int64_t> kcacheShape = {320, 1280, 1, 128};
  std::vector<int64_t> vcacheShape = {320, 1280, 1, 128};
  std::vector<int64_t> indicesShape = {320};
  std::vector<int64_t> kscaleShape = {128};
  std::vector<int64_t> vscaleShape = {128};
  std::vector<int64_t> koffsetShape = {128};
  std::vector<int64_t> voffsetShape = {128};

  std::vector<int64_t> weightShape = {1280};
  std::vector<int64_t> activationShape = {1280};
  std::vector<int64_t> biasShape = {8192};

  std::vector<int16_t> inputHostData(320*1280, 1);
  std::vector<int16_t> cosHostData(320*128, 1);
  std::vector<int16_t> sinHostData(320*128, 1);
  std::vector<int8_t> kcacheHostData(320*1280*128, 6);
  std::vector<int8_t> vcacheHostData(320*1280*128, 6);
  std::vector<int32_t> indicesHostData(320, 0);
  std::vector<int32_t> kscaleHostData(128, 2);
  std::vector<int32_t> vscaleHostData(128, 2);
  std::vector<int32_t> koffsetHostData(128, 2);
  std::vector<int32_t> voffsetHostData(128, 2);

  std::vector<int32_t> weightHostData(1280, 2);
  std::vector<int32_t> activationHostData(1280, 2);
  std::vector<int32_t> biasHostData(8192, 2);

  void* inputDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* kcacheDeviceAddr = nullptr;
  void* vcacheDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* kscaleDeviceAddr = nullptr;
  void* vscaleDeviceAddr = nullptr;
  void* koffsetDeviceAddr = nullptr;
  void* voffsetDeviceAddr = nullptr;

  void* weightDeviceAddr = nullptr;
  void* activationDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* kcache = nullptr;
  aclTensor* vcache = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* kscale = nullptr;
  aclTensor* vscale = nullptr;
  aclTensor* koffset = nullptr;
  aclTensor* voffset = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* activation = nullptr;
  aclTensor* bias = nullptr;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_INT32, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kcacheHostData, kcacheShape, &kcacheDeviceAddr, aclDataType::ACL_INT8, &kcache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vcacheHostData, vcacheShape, &vcacheDeviceAddr, aclDataType::ACL_INT8, &vcache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kscaleHostData, kscaleShape, &kscaleDeviceAddr, aclDataType::ACL_FLOAT, &kscale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vscaleHostData, vscaleShape, &vscaleDeviceAddr, aclDataType::ACL_FLOAT, &vscale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(koffsetHostData, koffsetShape, &koffsetDeviceAddr, aclDataType::ACL_FLOAT, &koffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(voffsetHostData, voffsetShape, &voffsetDeviceAddr, aclDataType::ACL_FLOAT, &voffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(activationHostData, activationShape, &activationDeviceAddr, aclDataType::ACL_FLOAT, &activation);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  std::vector<int64_t> qShape = {320,1,8,128};
  std::vector<int16_t> qHostData(320*8*128, 9);
  aclTensor* q = nullptr;
  void* qDeviceAddr = nullptr;
  std::vector<int64_t> kShape = {320,1,1,128};
  std::vector<int16_t> kHostData(320*128, 10);
  aclTensor* k = nullptr;
  void* kDeviceAddr = nullptr;
  std::vector<int64_t> vShape = {320,1,1, 128};
  std::vector<int16_t> vHostData(320*128, 10);
  aclTensor* v = nullptr;
  void* vDeviceAddr = nullptr;

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> splitData = {1024, 128, 128};
  aclIntArray *sizeSplits = aclCreateIntArray(splitData.data(), splitData.size());

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnDequantRopeQuantKvcache.
  ret = aclnnDequantRopeQuantKvcacheGetWorkspaceSize(input, cos, sin, kcache, vcache, indices, kscale, vscale, koffset,
                                                     voffset, weight, activation, bias,sizeSplits, "static", "BSND", true,
                                                     "contiguous", q, k, v, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  //Call the second-phase API of aclnnDequantRopeQuantKvcache.
  ret = aclnnDequantRopeQuantKvcache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcache failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Synchronously wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(kcacheShape, &kcacheDeviceAddr);
  PrintOutResult(vcacheShape, &vcacheDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(q);

  // 7. Release device resources.
  aclrtFree(inputDeviceAddr);
  aclrtFree(qDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
