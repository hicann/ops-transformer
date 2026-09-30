# aclnnDequantRopeQuantKvcache

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Description

- **API function**: Performs an optional dequantization on the input tensor (x), then splits it along the last axis into q, k, and vOut according to the specified sizeSplits. It applies Rotary Positional Embedding (RoPE) to q and k to generate qOut and kOut. Finally, kOut and vOut are quantized and updated into kCacheRef and vCacheRef based on the provided indices.

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

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. First, `aclnnDequantRopeQuantKvcacheGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnDequantRopeQuantKvcache` is called to perform computation.

* `aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, aclTensor *kCacheRef, aclTensor *vCacheRef, const aclTensor *indices, const aclTensor *scaleK, const aclTensor *scaleV, const aclTensor *offsetKOptional, const aclTensor *offsetVOptional, const aclTensor *weightScaleOptional, const aclTensor *activationScaleOptional, const aclTensor *biasOptional, const aclIntArray *sizeSplits, char *quantModeOptional, char *layoutOptional, bool kvOutput, char *cacheModeOptional, const aclTensor *qOut, const aclTensor *kOut, const aclTensor *vOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnDequantRopeQuantKvcache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnDequantRopeQuantKvcacheGetWorkspaceSize

- **Parameters**
  
  `x` (aclTensor\*, compute input): input x used for splitting in the formula, aclTensor on the device. The shape is [B, S, H] or [B, H]. H = (Nq + Nkv + Nkv) x D. The data type can be FLOAT16, INT32, or BFLOAT16. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape can only be 2D or 3D.
  * `cos` (aclTensor\*, compute input): input cos used for position encoding in the formula, aclTensor on the device. If x is 3D, the shape is [B, S, 1, D]. If x is 2D, the shape is [B, D]. The data type can be FLOAT16 or BFLOAT16, which should be the same as that of sin. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape can only be 2D or 4D.
  * `sin` (aclTensor\*, compute input): input sin used for position encoding in the formula, aclTensor on the device. If x is 3D, the shape is [B, S, 1, D]. If x is 2D, the shape is [B, D]. The data type can be FLOAT16 or BFLOAT16, which should be the same as that of cos. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape can only be 2D or 4D.
  * `kCacheRef` (aclTensor\*, compute input): input kCacheRef used for caching k in the formula, aclTensor on the device, shape [C_1, C_2, Nkv, D], and INT8 data type. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 4D.
  * `vCacheRef` (aclTensor\*, compute input): input vCacheRef used for caching v in the formula, aclTensor on the device. The shape is [C_1, C_2, Nkv, D], and the data type can be INT8. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 4D.
  * `indices` (aclTensor\*, compute input): input indices indicating the token location information of Kvcache in the formula, aclTensor on the device. When cache_mode is page and x is 3D, the shape is [B*S]. Otherwise, the shape is [B]. The data type can be INT32. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape can only be 1D or 2D.
  * `scaleK` (aclTensor\*, compute input): scaleK in the formula, used to quantize the scale factor of k. It is an aclTensor on the device, with shape [Nkv, D] and data type FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 2D.
  * `scaleV` (aclTensor\*, compute input): scaleV in the formula, used to quantize the scale factor of v. It is an aclTensor on the device, with shape [Nkv, D] and data type FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 2D.
  * `offsetKOptional` (aclTensor\*, compute input): offsetKoptional in the formula, used to quantize the offset factor of k. It is an aclTensor on the device, with shape [Nkv, D] and data type FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 2D.
  * `offsetVOptional` (aclTensor\*, compute input): offsetVoptional in the formula, used to quantize the offset factor. It is an aclTensor on the device, with shape [Nkv, D] and data type FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 2D.
  * `weightScaleOptional` (aclTensor\*, compute input): weightScaleoptional input in the formula, weight scale factor for dequantization. It is an aclTensor on the device, with shape [H] and data type FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 1D.
  * `activationScaleOptional` (aclTensor\*, compute input): activationScaleOptional input in the formula, activation scale factor for dequantization. It is an aclTensor on the device. If x is 3D, the shape is [B*S]. If x is 2D, the shape is [B]. The data type can be FLOAT. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 1D.
  * `biasOptional` (aclTensor\*, compute input): biasOptional input in the formula, bias for dequantization. It is an aclTensor on the device. The shape is [H], and the data type can be FLOAT, FLOAT16 (HALF), INT32, or BFLOAT16. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [data format](../../../docs/en/context/data_format.md) supports ND. The shape supports only 1D.
  * `sizeSplits` (aclIntArray\*, compute input): aclIntArray on the host. The data type is int array. The size is 3, and the value is [`Nq * D, Nkv * D, Nkv * D`]. It indicates the length of the input qkv to be split.
  * `quantModeOptional` (char\*, compute input): expression string on the host. It indicates the supported quantization type. Currently, only `static` is supported.
  * `layoutOptional` (char\*, compute input): expression string on the host. It indicates the supported data format. Currently, only `BSND` is supported.
  * `kvOutput` (bool, compute input): Boolean value of the expression on the host. It indicates whether to output `kOut` and `vOut`.
  * `cacheModeOptional` (char\*, compute input): expression string on the host. It indicates the update mode of `kCacheRef`. Currently, only `page` and `contiguous` are supported. The default value is `contiguous`.
  * `qOut` (aclTensor\*, compute output): output `qOut` in the formula, which indicates the processed `q`. It is an aclTensor on the device. If `x` is a 3D tensor, the shape is [B, S, Nq, D]. If `x` is a 2D tensor, the shape is [B, Nq, D]. The data type can be FLOAT16 or BFLOAT16, and must be the same as that of `sin`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `kOut` (aclTensor\*, compute output): output `kOut` in the formula, which indicates the processed `k`. It is an aclTensor on the device. When `kvOutput` is false, `kOut` is empty. Otherwise, when `x` is a 3D tensor, the shape is [B, S, Nkv, D]. When `x` is a 2D tensor, the shape is [B, Nkv, D]. Has the same data type as `sin`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `vOut` (aclTensor\*, compute output): output `vOut` in the formula, which indicates the processed `v`. It is an aclTensor on the device. When `kvOutput` is false, `vOut` is empty. Otherwise, when `x` is a 3D tensor, the shape is [B, S, Nkv, D]. When `x` is a 2D tensor, the shape is [B, Nkv, D]. Has the same data type as `sin`. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The [data format](../../../docs/en/context/data_format.md) can be ND.
  * `workspaceSize` (uint64_t\*, output): size of the workspace to be allocated on the device.
  * `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
- **Returns**
  
  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  ```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown.
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
  ```

## aclnnDequantRopeQuantKvcache

- **Parameters**
  
  * `workspace` (void\*, input): start address of the workspace to be allocated on the device.
  * `workspaceSize` (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnDequantRopeQuantKvcacheGetWorkspaceSize.
  * `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input parameter): stream for executing the task.
  
- **Returns**
  
  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computing:
     - `aclnnDequantRopeQuantKvcache` defaults to a deterministic implementation.

2. When `cacheModeOptional` is set to contiguous, the 0th dimension of `kCacheRef` is greater than that of x, and the value of indices is greater than or equal to 0 and less than or equal to the value of the 1st dimension of `vCacheRef` (s in the [b, s, n, d] format) minus the value of the 1st dimension of `x`. When `cacheModeOptional` is set to page, the value of inidces is greater than or equal to 0, less than the value of the 0th dimension multiplied by the value of the first dimension of `kCacheRef`, and is unique.
3. The last axis of `x` is less than or equal to 4096 and is 64-pixel aligned.
4. If `x` is not of type INT32, the data types of `x`, `cos`, and `sin` are the same as those of the outputs `qOut`, `kOut`, and `vOut`. In this case, `activationScaleOptional`, `weightScaleOptional`, and `biasOptional` do not take effect. If x is of type INT32, the data types of `cos` and `sin` are the same as those of the outputs `qOut`, `kOut`, and `vOut`. In this case, `weightScaleOptional` is mandatory, and `activationScaleOptional` and `biasOptional` are optional (`biasOptional` does not need to be the same as other input types).

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

  // 2. Construct the inputs and outputs based on the API definition.
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

  // 4. (Fixed writing) Wait until the task execution is complete.
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
