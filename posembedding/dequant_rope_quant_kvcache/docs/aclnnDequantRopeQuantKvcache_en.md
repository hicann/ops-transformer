# aclnnDequantRopeQuantKvcache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:46:55.471Z -->

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

- Description: performs dequantization (optional) on the input tensor (x), splits the last axis by `sizeSplits` (the split lengths) into q, k, and vOut, applies rotary position embedding to q and k to generate qOut and kOut, and then quantizes kOut and vOut and updates them to kCacheRef and vCacheRef based on `indices`.

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDequantRopeQuantKvcacheGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnDequantRopeQuantKvcache" API to perform the computation.

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

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 550px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <td>x</td>
        <td>Input</td>
        <td>Input x used in the formula for splitting.</td>
        <td>The shape is [B, S, H] or [B, H], where H=(Nq+Nkv+Nkv)*D. The last dimension of x is less than or equal to 4096 and is aligned to 64.</td>
        <td>FLOAT16, BFLOAT16, INT32</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>cos</td>
        <td>Input</td>
        <td>Input cos used in the formula for positional encoding.</td>
        <td>When x is 3-dimensional, the shape is [B, S, 1, D]; when x is 2-dimensional, the shape is [B, D].</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2, 4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sin</td>
        <td>Input</td>
        <td>Input sin used for position encoding in the formula.</td>
        <td>When x is 3-dimensional, the shape is [B, S, 1, D]; when x is 2-dimensional, the shape is [B, D].</td>
        <td>Keep consistent with cos</td>
        <td>ND</td>
        <td>2, 4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>kCacheRef</td>
        <td>Input</td>
        <td>Input kCacheRef used in the formula to cache k.</td>
        <td>The shape is [C_1, C_2, Nkv, D].</td>
        <td>INT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>vCacheRef</td>
        <td>Input</td>
        <td>vCacheRef, the input used for caching v in the formula.</td>
        <td>The shape is [C_1, C_2, Nkv, D].</td>
        <td>INT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Input indices in the formula, representing the token position information of Kvcache.</td>
        <td>When cache_mode is page and x is 3-dimensional, the shape is [B*S]; otherwise, the shape is [B].</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scaleK</td>
        <td>Input</td>
        <td>Input scaleK in the formula is the scale factor used for quantization of k.</td>
        <td>The number of elements is Nkv*D. The recommended shape is [Nkv, D], compatible with the flattened one-dimensional shape [Nkv*D].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>≥1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scaleV</td>
        <td>Input</td>
        <td>The input scaleV in the formula is the scale factor used for quantization of v.</td>
        <td>The number of elements is Nkv*D. The recommended shape is [Nkv, D], compatible with the flattened one-dimensional shape [Nkv*D].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>≥1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>offsetKOptional</td>
        <td>Input</td>
        <td>In formula, the input offsetKOptional is the offset factor used for quantization of k.</td>
        <td>The number of elements is Nkv*D, the recommended shape is [Nkv, D], compatible with one-dimensional flattened shape [Nkv*D].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>≥1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>offsetVOptional</td>
        <td>Input</td>
        <td>The input offsetVOptional in the formula is the offset factor used for quantization of v.</td>
        <td>The number of elements is Nkv*D. The recommended shape is [Nkv, D], compatible with one-dimensional flattened shape [Nkv*D].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>≥1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>weightScaleOptional</td>
        <td>Input</td>
        <td>Input weightScaleOptional in the formula, the weight scale factor used for dequantization.</td>
        <td>The shape is [H].</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>activationScaleOptional</td>
        <td>Input</td>
        <td>Input activationScaleOptional in the formula, the activation scale factor used for dequantization.</td>
        <td>Shape is [B*S] when x is 3-dimensional, and [B] when x is 2-dimensional.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Bias biasOptional in the formula, used for dequantization.</td>
        <td>Shape is [H].</td>
        <td>FLOAT, FLOAT16, INT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sizeSplits</td>
        <td>Input</td>
        <td>Specifies the lengths for splitting the input qkv.</td>
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
        <td>Currently, only "static" is supported.</td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutOptional</td>
        <td>Input</td>
        <td>Indicates the supported data format.</td>
        <td>Currently, only "BSND" is supported.</td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvOutput</td>
        <td>Input</td>
        <td>Boolean value on the Host side. Indicates whether to output kOut and vOut.</td>
        <td>If true, kOut and vOut with valid shapes are output; if false, the shapes of kOut and vOut are empty.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cacheModeOptional</td>
        <td>Input</td>
        <td>Indicates the update mode of kCacheRef.</td>
        <td>Currently, only "page" and "contiguous" are supported. The default is "contiguous".</td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>qOut</td>
        <td>Output</td>
        <td>q after rotary position embedding in the formula.</td>
        <td>Shape is [B, S, Nq, D] when x is 3-dimensional, and [B, Nq, D] when x is 2-dimensional. The data type must keep consistent with cos and sin.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>kOut</td>
        <td>Output</td>
        <td>k in the formula after rotary position embedding.</td>
        <td>When kvOutput is true, the shape is [B, S, Nkv, D] if x is 3-dimensional, and [B, Nkv, D] if x is 2-dimensional; when kvOutput is false, the shape is empty. The data type must be kept consistent with cos and sin.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>vOut</td>
        <td>Output</td>
        <td>The v obtained by splitting in the formula.</td>
        <td>When kvOutput is true, the shape is [B, S, Nkv, D] if x has 3 dimensions, and [B, Nkv, D] if x has 2 dimensions; when kvOutput is false, the shape is empty. The data type keeps consistent with cos and sin.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 119px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input or output tensor is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of the input and output are not in the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnDequantRopeQuantKvcache API

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnDequantRopeQuantKvcacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computation:
     - aclnnDequantRopeQuantKvcache defaults to a deterministic implementation.

2. When cacheModeOptional is contiguous: dimension 0 of kCacheRef is greater than or equal to dimension 0 of x. If x is three-dimensional, the data values of indices are greater than or equal to 0 and less than or equal to dimension 1 of kCacheRef minus dimension 1 of x; if x is two-dimensional, the data values of indices are greater than or equal to 0 and less than or equal to dimension 1 of kCacheRef minus 1.
3. When cacheModeOptional is page: the data values of indices are greater than or equal to 0, less than dimension 0 * dimension 1 of kCacheRef, and are not duplicated.
4. When the input x is not INT32, the data types of x, cos, sin, and the outputs qOut, kOut, and vOut must be consistent. In this case, activationScaleOptional, weightScaleOptional, and biasOptional do not take effect.
5. When the input x is INT32, the data types of cos, sin, and the outputs qOut, kOut, and vOut must be consistent. In this case, weightScaleOptional is mandatory, while activationScaleOptional and biasOptional are optional (biasOptional does not need to match the data type of other inputs).
6. The last dimension of x must be less than or equal to 4096 and aligned to 64.
7. Empty tensors with B=0 or S=0 are supported. If x is two-dimensional and B=0, or x is three-dimensional and B=0 or S=0, qOut, kOut, and vOut contain zero elements, while kCacheRef and vCacheRef remain unchanged. The last dimension of x, Nkv, D, scaleK, and scaleV must still satisfy the constraints above; D=0, Nkv=0, and empty mandatory scales are not supported.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate: resource initialization
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data on the host to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Boilerplate: initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs according to the API definition.
  std::vector<int64_t> inputShape = {320, 1, 1280};
  std::vector<int64_t> cosShape = {320, 1, 1, 128};
  std::vector<int64_t> sinShape = {320, 1, 1, 128};
  std::vector<int64_t> kcacheShape = {320, 1280, 1, 128};
  std::vector<int64_t> vcacheShape = {320, 1280, 1, 128};
  std::vector<int64_t> indicesShape = {320};
  std::vector<int64_t> kscaleShape = {1, 128};
  std::vector<int64_t> vscaleShape = {1, 128};
  std::vector<int64_t> koffsetShape = {1, 128};
  std::vector<int64_t> voffsetShape = {1, 128};

  std::vector<int64_t> weightShape = {1280};
  std::vector<int64_t> activationShape = {320};
  std::vector<int64_t> biasShape = {1280};

  std::vector<int32_t> inputHostData(320*1280, 1);
  std::vector<int16_t> cosHostData(320*128, 1);
  std::vector<int16_t> sinHostData(320*128, 1);
  std::vector<int8_t> kcacheHostData(320*1280*128, 6);
  std::vector<int8_t> vcacheHostData(320*1280*128, 6);
  std::vector<int32_t> indicesHostData(320, 0);
  std::vector<float> kscaleHostData(128, 2);
  std::vector<float> vscaleHostData(128, 2);
  std::vector<float> koffsetHostData(128, 2);
  std::vector<float> voffsetHostData(128, 2);

  std::vector<float> weightHostData(1280, 2);
  std::vector<float> activationHostData(320, 2);
  std::vector<float> biasHostData(1280, 2);

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
  char quantMode[] = "static";
  char layout[] = "BSND";
  char cacheMode[] = "contiguous";

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnDequantRopeQuantKvcache.
  ret = aclnnDequantRopeQuantKvcacheGetWorkspaceSize(input, cos, sin, kcache, vcache, indices, kscale, vscale, koffset,
                                                     voffset, weight, activation, bias, sizeSplits, quantMode, layout,
                                                     true, cacheMode, q, k, v, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnDequantRopeQuantKvcache.
  ret = aclnnDequantRopeQuantKvcache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantRopeQuantKvcache failed. ERROR: %d\n", ret); return ret);

  // 4. Boilerplate: wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result on the device memory to the host side. Modify the code based on the specific API definition.
  PrintOutResult(kcacheShape, &kcacheDeviceAddr);
  PrintOutResult(vcacheShape, &vcacheDeviceAddr);

  // 6. Release aclTensor and aclIntArray. Modify the code based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(kcache);
  aclDestroyTensor(vcache);
  aclDestroyTensor(indices);
  aclDestroyTensor(kscale);
  aclDestroyTensor(vscale);
  aclDestroyTensor(koffset);
  aclDestroyTensor(voffset);
  aclDestroyTensor(weight);
  aclDestroyTensor(activation);
  aclDestroyTensor(bias);
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyIntArray(sizeSplits);

  // 7. Release device resources.
  aclrtFree(inputDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(kcacheDeviceAddr);
  aclrtFree(vcacheDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(kscaleDeviceAddr);
  aclrtFree(vscaleDeviceAddr);
  aclrtFree(koffsetDeviceAddr);
  aclrtFree(voffsetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(activationDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
