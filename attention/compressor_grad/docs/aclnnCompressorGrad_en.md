# aclnnCompressorGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:34:58.388Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: CompressorGrad is the backward operator of Compressor. It computes the gradients of the input $X$, weights $W^{KV}$/$W^{Gate}$, and positional encoding $Ape$. When gradEnabled is true, the forward pass exports the intermediate results softmax\_score (grouped softmax result) and kv (Hadamard product of the softmax result and kv\_state), which are inputs of this operator. The main computation procedure is as follows:
    1. Compute the Hadamard product backward block by block: multiply the upstream gradient $dC$ element-wise with softmax\_score and kv to obtain $dK$ and $dS^\prime$;
    2. Softmax backward: compute the softmax backward of $dS^\prime$ along the compression axis to obtain $dZ$;
    3. APE gradient computation: accumulate $dZ$ by token position to obtain $dApe$;
    4. matmul backward: perform the matmul backward with $dK$, $dZ$, and the weights to obtain $dX$, $dW^{KV}$, and $dW^{Gate}$.

- Formula:

    1. Compute the Hadamard product backward. $N$ is the total number of compressed blocks, $i$ is the compressed block index, $dC_i$ is the upstream gradient of the $i$-th block, and $S_i$ and $K_i$ are softmax\_score and the $i$-th kv block, respectively:

        $$
        dK_i = dC_i \odot S_i,~ i=1,\cdots,N
        $$

        $$
        dS^\prime_i = dC_i \odot K_i,~ i=1,\cdots,N
        $$

    2. Compute the softmax backward (sum along the compression axis). $k$ is the row index within a block:

        $$
        dZ_i = S_i \odot \left(dS^\prime_i - \sum_{k=1}^{coff \cdot cmp\_ratio} \left(S_i \odot dS^\prime_i\right)_{k,:}\right),~ i=1,\cdots,N
        $$

    3. Compute the APE gradient. $pos$ is the global position of the token corresponding to each row of $dZ$:

        $$
        dApe = ScatterAdd\left(dZ,~ pos \% cmp\_ratio\right)
        $$

    4. Compute the matmul backward. $dNewKv$ and $dNewScore$ are obtained by mapping $dK$ and $dZ$ back to the global token rows by compressed block (when coff=2, the prev/cur halves correspond to the token rows of the previous block and the current block, respectively, matching $W^{aKV}$/$W^{bKV}$ in the forward pass):

        $$
        dX = dNewKv @ W^{KV} + dNewScore @ W^{Gate}
        $$

        $$
        dW^{KV} = dNewKv^T @ X,~ dW^{Gate} = dNewScore^T @ X
        $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnCompressorGradGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the process, and then call "aclnnCompressorGrad" to perform the computation.

```cpp
aclnnStatus aclnnCompressorGradGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    const aclTensor *dCmpKv,
    const aclTensor *softmaxScore,
    const aclTensor *kv,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          cmpRatio,
    int64_t          coff,
    const aclTensor *dXOut,
    const aclTensor *dWkvOut,
    const aclTensor *dWgateOut,
    const aclTensor *dApeOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

``` cpp
aclnnStatus aclnnCompressorGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCompressorGradGetWorkspaceSize

- **Parameters**

    | Name | Input/Output | Description | Instructions | Data Type | Data Format | Dimensions (shape) | Non-contiguous Tensor |
    |----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|-|-|-|
    | x | Input | $X$ in the formula, original data of the forward input. | Empty tensors are not supported. | FLOAT16, BFLOAT16 | ND | BS merged: [T,H], BS non-merged: [B,S,H]|×|
    | wkv | Input | $W^{KV}$ in the formula, kv compression weight of the forward pass. |Empty tensors are not supported.| FLOAT16, BFLOAT16 | ND |[coff* D,H]|×|
    | wgate | Input | $W^{Gate}$ in the formula, gate compression weight of the forward pass. |Empty tensors are not supported.| FLOAT16, BFLOAT16 | ND |[coff* D,H]|×|
    | dCmpKv | Input | $dC$ in the formula, upstream gradient of the forward output cmp\_kv. | Empty tensors are not supported. | FLOAT16, BFLOAT16 | ND |BS merged: [min(T,T//cmp_ratio+B),D], BS non-merged: [B,ceil(S/cmp_ratio),D]|×|
    | softmaxScore | Input | $S$ in the formula, grouped softmax intermediate result exported in the forward pass when gradEnabled is true. | Empty tensors are not supported. | FLOAT32 | ND |BS merged: [min(T,T//cmp_ratio+B), coff*cmp_ratio, D], BS non-merged: [B,ceil(S/cmp_ratio),coff*cmp_ratio,D]|×|
    | kv | Input | $K$ in the formula, intermediate result of the Hadamard product between the softmax result exported in the forward pass when gradEnabled is true and kv\_state. | Empty tensors are not supported. | FLOAT32 | ND |Same as softmaxScore|×|
    | cuSeqlensOptional | Optional input | Number of valid tokens in each Batch. | Required when the shape of x is [T,H]; the input shape is [B+1,]. Must be empty when the shape of x is [B,S,H]. Empty tensors are not supported.| INT32 | ND |[B+1,] when the shape of x is [T,H]|×|
    | sequsedOptional | Optional input | Number of tokens actually participating in compression in each Batch. | When set to None, the value is the same as the Sequence Length of each Batch; seqused[n] must not exceed the corresponding Sequence Length and must be greater than or equal to 0.| INT32 | ND |[B,]|×|
    | startPosOptional | Optional input | Start position of the computation. | When set to None, the computation starts from 0.| INT32 | ND |[B,]|×|
    | cmpRatio | Input | Data compression ratio used for sparse computation, consistent with the forward pass. |An integer in the range [2, 128].| INT32 | - |-|-|
    | coff | Optional input | Whether to perform overlap data rearrangement, consistent with the forward pass. |A value in the range [1, 2]. If coff=1, overlap data rearrangement is not required. If coff=2, overlap data rearrangement is required.| INT32 | - |-|-|
    | dXOut | Output | $dX$ in the formula, gradient of input x. | Empty tensors are not supported.| FLOAT16, BFLOAT16 | ND |Same as x: BS merged: [T,H], BS non-merged: [B,S,H]|×|
    | dWkvOut | Output | $dW^{KV}$ in the formula, gradient of weight wkv. | Empty tensors are not supported.| FLOAT16, BFLOAT16 | ND |[coff* D,H]|×|
    | dWgateOut | Output | $dW^{Gate}$ in the formula, gradient of weight wgate. | Empty tensors are not supported.| FLOAT16, BFLOAT16 | ND |[coff* D,H]|×|
    | dApeOut | Output | $dApe$ in the formula, gradient of the APE positional encoding. | Empty tensors are not supported.| FLOAT32 | ND |[cmp_ratio,coff* D]|×|

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    When the first-phase API performs parameter validation and tiling, the following errors may be returned:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
      <thead>
        <tr>
          <th>Return Value</th>
          <th>Error Code</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>A null pointer exists in a required parameter (such as an input/output parameter on which the core of the API depends).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>The dtype (data type) of the input parameter is not supported by the API.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>An exception occurs when the API calls the NPU Runtime API for memory operations (for example, the Runtime service is not started, or memory application fails).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>An exception occurs during tiling: the rank, shape (dimension/size), or attribute value of an input parameter, or the shape consistency between inputs, is not supported by the API.</td>
        </tr>
      </tbody>
    </table>

    Validation layering: failures in the aclnn parameter validation layer (required null pointers, dtype, and so on) return 161001/161002; failures in the tiling-stage validation (rank, shape dimensions, shape consistency between inputs, attribute values, and so on) return 561002. The returned code depends on the validation stage actually reached.

## aclnnCompressorGrad

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>Address of the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device. Obtained by the first-phase API aclnnCompressorGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnCompressorGrad defaults to a deterministic implementation.
- Meanings of dimensions of x: B (Batch Size) denotes the batch size of input samples, S (Sequence Length) denotes the sequence length of input samples, H (Head Size) denotes the size of the hidden layer, D (Head Dim) denotes the size of the minimum unit of the hidden layer, and T denotes the sum of sequence lengths of input samples across all batches.
- Input shape constraints:
    - wkv supports input shape [coff* D,H].
    - wgate supports input shape [coff* D,H].
    - softmaxScore supports input shape: [min(T,T//cmp_ratio+B), coff*cmp_ratio, D] when BS axes are merged; [B,ceil(S/cmp_ratio),coff*cmp_ratio,D] when BS axes are not merged.
    - kv supports input shape: same as softmaxScore.
    - dCmpKv supports input shape: [min(T,T//cmp_ratio+B),D] when BS axes are merged; [B,ceil(S/cmp_ratio),D] when BS axes are not merged.
    - dX supports output shape: same as x, [T,H] when BS axes are merged and [B,S,H] when BS axes are not merged.
    - dWkv and dWgate support output shape [coff* D,H].
    - dApe supports output shape [cmp_ratio,coff* D].
    - startPos supports input shape [B,].
    - If the dimensions of x use BS axis merging, that is, the input shape of x is [T,H].
        - The cuSeqlens input shape must be [B+1,]. Each element in this parameter indicates the total number of tokens in the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element, and the first element must be 0.
        - seqused, supporting the input shape [B,]. The number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n], and must not be less than 0.
    - If the dimensions of x do not use the BS merged axis, the input shape of x is [B,S,H]
        - cuSeqlens: this parameter must be empty.
        - seqused, supporting the input shape [B,]. The number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= S, and must not be less than 0.
- Input value range restrictions:
  - This API supports generalization of B and S, with the following use case restrictions:
      - **Empty tensors with B, S, or T of 0 are not supported**: Different from the forward Compressor, no input/output of CompressorGrad supports empty tensors. shapeSize must be greater than 0.
      - In some long-sequence use cases, excessive computation may cause an error indicating that the NPU memory is exceeded. Note: The computation is affected by the input shape of x. The larger the value, the greater the computation.
- Input attribute restrictions:
  - D of 128/512 is supported.
  - Supports H in the range 1K-10K, aligned to 512.
  - Supports coff of 1/2.
  - Supports cmp\_ratio in the range 2-128.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_compressor_grad.cpp
 * \brief aclnn call example of the CompressorGrad operator (A5 / ascend950)
 *        Use case: C4A (D=512, coff=2, cmp_ratio=4, BSH layout, BF16)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_compressor_grad.h"
#include "opdev/bfloat16.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

using op::bfloat16;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void PrintBf16Result(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<bfloat16> resultData(size, bfloat16(0.0f));
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) { // 10: max print
        LOG_PRINT("result[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
}

void PrintF32Result(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float_t> resultData(size, 0.0f);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) { // 10: max print
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

} // namespace

int main()
{
    // 1. Initialize the device/stream.
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Use case parameters: C4A (D=512, coff=2, cmp_ratio=4, BSH layout, BF16)
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t coff = 2;
    int64_t cmpRatio = 4;

    int64_t coffD = coff * headDim;
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;

    // 2. Construct input and output shapes.
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {coffD, hiddenSize};
    std::vector<int64_t> wgateShape = {coffD, hiddenSize};
    std::vector<int64_t> dCmpKvShape = {B, Sr, headDim};
    std::vector<int64_t> softmaxScoreShape = {B, Sr, coff * cmpRatio, headDim};
    std::vector<int64_t> kvShape = {B, Sr, coff * cmpRatio, headDim};
    std::vector<int64_t> startPosShape = {B};
    std::vector<int64_t> dXShape = {B, S, hiddenSize};
    std::vector<int64_t> dWkvShape = {coffD, hiddenSize};
    std::vector<int64_t> dWgateShape = {coffD, hiddenSize};
    std::vector<int64_t> dApeShape = {cmpRatio, coffD};

    // 3. Construct host data.
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t dCmpKvSize = GetShapeSize(dCmpKvShape);
    int64_t softmaxScoreSize = GetShapeSize(softmaxScoreShape);
    int64_t kvSize = GetShapeSize(kvShape);
    int64_t dXSize = GetShapeSize(dXShape);
    int64_t dWkvSize = GetShapeSize(dWkvShape);
    int64_t dWgateSize = GetShapeSize(dWgateShape);
    int64_t dApeSize = GetShapeSize(dApeShape);

    std::vector<bfloat16> xHostData(xSize, bfloat16(0.1f));
    std::vector<bfloat16> wkvHostData(wkvSize, bfloat16(0.1f));
    std::vector<bfloat16> wgateHostData(wgateSize, bfloat16(0.1f));
    std::vector<bfloat16> dCmpKvHostData(dCmpKvSize, bfloat16(0.1f));
    std::vector<float_t> softmaxScoreHostData(softmaxScoreSize, 0.1f);
    std::vector<float_t> kvHostData(kvSize, 0.1f);
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<bfloat16> dXHostData(dXSize, bfloat16(0.0f));
    std::vector<bfloat16> dWkvHostData(dWkvSize, bfloat16(0.0f));
    std::vector<bfloat16> dWgateHostData(dWgateSize, bfloat16(0.0f));
    std::vector<float_t> dApeHostData(dApeSize, 0.0f);

    // 4. Create an aclTensor.
    void *xDeviceAddr = nullptr;
    void *wkvDeviceAddr = nullptr;
    void *wgateDeviceAddr = nullptr;
    void *dCmpKvDeviceAddr = nullptr;
    void *softmaxScoreDeviceAddr = nullptr;
    void *kvDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *dXDeviceAddr = nullptr;
    void *dWkvDeviceAddr = nullptr;
    void *dWgateDeviceAddr = nullptr;
    void *dApeDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *wkv = nullptr;
    aclTensor *wgate = nullptr;
    aclTensor *dCmpKv = nullptr;
    aclTensor *softmaxScore = nullptr;
    aclTensor *kv = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *dX = nullptr;
    aclTensor *dWkv = nullptr;
    aclTensor *dWgate = nullptr;
    aclTensor *dApe = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_BF16, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_BF16, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dCmpKvHostData, dCmpKvShape, &dCmpKvDeviceAddr, aclDataType::ACL_BF16, &dCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxScoreHostData, softmaxScoreShape, &softmaxScoreDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxScore);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT, &kv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dXHostData, dXShape, &dXDeviceAddr, aclDataType::ACL_BF16, &dX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dWkvHostData, dWkvShape, &dWkvDeviceAddr, aclDataType::ACL_BF16, &dWkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dWgateHostData, dWgateShape, &dWgateDeviceAddr, aclDataType::ACL_BF16, &dWgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dApeHostData, dApeShape, &dApeDeviceAddr, aclDataType::ACL_FLOAT, &dApe);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. Call aclnnCompressorGradGetWorkspaceSize.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnCompressorGradGetWorkspaceSize(x, wkv, wgate, dCmpKv, softmaxScore, kv, nullptr, nullptr, startPos,
                                              cmpRatio, coff, dX, dWkv, dWgate, dApe, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. Allocate workspace.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. Call aclnnCompressorGrad.
    ret = aclnnCompressorGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorGrad failed. ERROR: %d\n", ret); return ret);

    // 8. Wait for synchronization.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. Obtain the output.
    LOG_PRINT("CompressorGrad execution succeeded.\n");
    PrintBf16Result(dXShape, &dXDeviceAddr);
    PrintBf16Result(dWkvShape, &dWkvDeviceAddr);
    PrintBf16Result(dWgateShape, &dWgateDeviceAddr);
    PrintF32Result(dApeShape, &dApeDeviceAddr);

    // 10. Release resources.
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(dCmpKv);
    aclDestroyTensor(softmaxScore);
    aclDestroyTensor(kv);
    aclDestroyTensor(startPos);
    aclDestroyTensor(dX);
    aclDestroyTensor(dWkv);
    aclDestroyTensor(dWgate);
    aclDestroyTensor(dApe);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(dCmpKvDeviceAddr);
    aclrtFree(softmaxScoreDeviceAddr);
    aclrtFree(kvDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(dXDeviceAddr);
    aclrtFree(dWkvDeviceAddr);
    aclrtFree(dWgateDeviceAddr);
    aclrtFree(dApeDeviceAddr);
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
