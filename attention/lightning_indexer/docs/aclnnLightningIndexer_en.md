# aclnnLightningIndexer

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Obtains the Top-$k$ positions for each token through a series of operations.

- Formulas:

  $$
  Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
  $$

  For an index query $Q_{index} \in \mathbb{R}^{g \times d}$ corresponding to a token, given the context index key $K_{index} \in \mathbb{R}^{S_{k} \times d}$ and $W \in \mathbb{R}^{g \times 1}$. $g$ indicates the group size in Grouped-Query Attention (GQA). $d$ indicates the dimension of each head. $S_k$ indicates the context length.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLightningIndexerGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLightningIndexer` is called to perform computation.

```Cpp
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    char            *layoutQueryOptional,
    char            *layoutKeyOptional,
    int64_t          sparseCount,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    bool             returnValues,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnLightningIndexer(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnLightningIndexerGetWorkspaceSize

- **Parameters:**

> [!NOTE]
>
> - Dimension definitions for the `query`, `key`, and `weights` parameters:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br>`D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
> - `S1` indicates the `S` dimension in the shape of `query`.<br>`S2` indicates the `S` dimension in the shape of `key`.<br>`T1` indicates the `T` dimension in the shape of `query`.<br>`T2` indicates the `T` dimension in the shape of `key`.<br>`N1` indicates the `N` dimension in the shape of `query`.<br>`N2` indicates the `N` dimension in the shape of `key`.

  <table style="undefined;table-layout: fixed; width: 1580px"><colgroup>
  <col style="width: 231px">
  <col style="width: 120px">
  <col style="width: 242px">
  <col style="width: 332px">
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 228px">
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
      <td>query</td>
      <td>Input</td>
      <td>Input <code>Q</code> in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_query is BSND, the shape is (B, S1, N1, D).</li>
                <li>When layout_query is TND, the shape is (T1, N1, D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Input <code>K</code> in the formula.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>block_num indicates the total number of blocks, and block_size indicates the number of tokens in a block.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_key is PA_BSND, the shape is (block_num, block_size, N2, D).</li>
                <li>When layout_kv is BSND, the shape is (B, S2, N2, D).</li>
                <li>When layout_kv is TND, the shape is (T2, N2, D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Input W in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_query is BSND, the shape is (B, S1, N1).</li>
                <li>When layout_query is TND, the shape is (T1, N1).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional</td>
      <td>Input</td>
      <td>Number of valid tokens in each batch for query.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>If seqlen is not specified, you can pass None, indicating that the value is the same as the length of S in the shape of the query.</li>
                <li>The number of valid tokens in each batch in this input parameter cannot be greater than the dimension S in the query and must be greater than or equal to 0. A one-dimensional tensor with the length of B is supported.</li>
                <li>When layout_query is set to TND, this input parameter must be passed, and the number of elements in this input parameter is used as the value of B. The value of each element in this input parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of the next element must be greater than or equal to that of the previous element.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKeyOptional</td>
      <td>Input</td>
      <td>Number of valid tokens in each batch for key.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>If seqlen is not specified, you can pass None, indicating that the value is the same as the length of S in the shape of the key.</li>
                <li>The number of valid tokens in each batch in this parameter cannot be greater than the dimension S in the key/value and must be greater than or equal to 0. A one-dimensional tensor with the length of B is supported.</li>
                <li>When layout_key is set to TND or PA_BSND, this input parameter is mandatory. When layout_key is set to TND, the value of each element in this parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of the next element must be greater than or equal to that of the previous element.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Block mapping table used for KV storage in PageAttention.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>In the PageAttention scenario, block\_table must be two-dimensional. The length of the first dimension must be equal to B, and the length of the second dimension cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq indicates the maximum number of blocks corresponding to actual\_seq\_lengths\_key in each batch).</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>The shape supports (B,S2/block_size).</td>
      <td>x</td>
    </tr>
    <tr>
      <td>layoutQueryOptional</td>
      <td>Input</td>
      <td>Format of the input query data.</td>
      <td>
          <ul>
                <li>If not specified, the default value `BSND` is passed.</li>
                <li>Currently, BSND and TND are supported.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>Input</td>
      <td>Layout of the input key.</td>
      <td>
          <ul>
                <li>If not specified, the default value `BSND` is passed.</li>
                <li>Currently, PA_BSND, BSND, and TND are supported.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseCount</td>
      <td>Input</td>
      <td>Number of blocks to be retained in the topK phase.</td>
      <td>Supports [1, 2048], 3072, 4096, 5120, 6144, 7168, and 8192.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td>
          <ul>
                <li>Value `0` of sparse_mode indicates the defaultMask mode.</li>
                <li>Value `3` of sparse_mode enables rightDownCausal` mode mask, corresponding to lower triangular scenarios where the dividing line extends from the right vertex.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>Number of preceding tokens to associate in attention computation for sparse computation.</td>
      <td>Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>Used for sparse computation, indicating that the attention needs to be associated with the last several tokens.</td>
      <td>Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValues</td>
      <td>Input</td>
      <td>Indicates whether to output sparseValuesOut.</td>
      <td>
          <ul>
                <li>True indicates that the output is supported, but this parameter is not supported in graph mode. False indicates that the output is not supported. The default value is False</li>.
                <li>This parameter is supported only in training mode when layout_key is not PA_BSND.</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut</td>
      <td>Output</td>
      <td>Indices output in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>-</td>
      <td>
          <ul>
                <li>When layout_query is BSND, the output shape is [B, S1, N2, sparseCount].</li>
                <li>When layout_query is TND, the output shape is [T1, N2, sparseCount].</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>Output</td>
      <td>Value of the indices output in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>The shape is the same as that of sparseIndicesOut.</td>
      <td>x</td>
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
          <th>Return</th>
          <th>Error Code</th>
          <th>Description</th>
      </thead>
      <tbody>
          <tr>
              <td>ACLNN_ERR_PARAM_NULLPTR</td>
              <td>161001</td>
              <td>The required input, output, or attribute is passed as a null pointer.</td>
          </tr>
          <tr>
              <td>ACLNN_ERR_PARAM_INVALID</td>
              <td>161002</td>
              <td>query, key, weights, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, returnValues, sparseIndicesOut. The data type and format of sparseValuesOut are not supported.</td>
          </tr>
      </tbody>
  </table>

## aclnnLightningIndexer

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnLightningIndexerGetWorkspaceSize.</td>
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

- The value of N in the query parameter can be less than or equal to 64, and the value of N in the key parameter can be 1.
- headdim supports 128.
- The value of block_size is a multiple of 16, and the maximum value is 1024.
- The data types of `query` and `key` must be identical.
- When the data type of `weights` is not `float32`, the data types of `query`, `key`, and `weights` must be identical.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
 * \file test_incre_flash_attention_v4.cpp
 * \brief
 */
//testci
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 128};
    std::vector<int64_t> keyShape = {1, 2, 1, 128};
    std::vector<int64_t> weightsShape = {1, 2, 1};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2048};
    std::vector<int64_t> sparseValuesShape = {1, 2, 1, 2048};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> weightsHostData(weightsShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                         aclDataType::ACL_FLOAT16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteLightningIndexer(TensorResources& resources, aclrtStream stream,
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t sparseCount = 2048;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    bool returnValue = true;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKey[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnLightningIndexerGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr,
                                                    layoutQuery, layoutKey, sparseCount, sparseMode, preTokens, nextTokens,returnValue,
                                                    resources.sparseIndicesTensor, resources.sparseValuesTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintValueOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

int PrintIndicesOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
      aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
      aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
      aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
      aclrtFree(resources.sparseValuesDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2048};
    std::vector<int64_t> sparseValuesShape = {1, 2, 1, 2048};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintIndicesOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);
    PrintValueOutResult(sparseValuesShape, &resources.sparseValuesDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
