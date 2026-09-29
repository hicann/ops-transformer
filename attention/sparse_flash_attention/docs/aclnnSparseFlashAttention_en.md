# aclnnSparseFlashAttention

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention)

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 inference products</term>  | √  |
|<term>Atlas A3 inference products</term>  | √  |

## Function

- Provides highly efficient attention computations for long-sequence inference scenarios. `sparse_flash_attention` (SFA) reduces computational cost by computing only the critical portions of attention. However, it introduces a large amount of discrete memory access. This increases data movement overhead and impacts overall performance.

- Formulas:

  $$
  \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
  $$

  $\tilde{K}$ and $\tilde{V}$ represent key and value tensors with higher importance obtained through a selection algorithm such as `lightning_indexer`. They typically feature sparse or block-sparse characteristics. $d_k$ represents the per-head dimension of $Q$ and $\tilde{K}$.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSparseFlashAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSparseFlashAttention` is called to perform computation.

```Cpp
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value, 
    const aclTensor     *sparseIndices,
    const aclTensor     *blockTable,
    const aclTensor     *actualSeqLengthsQuery,
    const aclTensor     *actualSeqLengthsKv,
    const aclTensor     *queryRope,
    const aclTensor     *keyRope,
    double              scaleValue,
    int64_t             sparseBlockSize,
    char                *layoutQuery,
    char                *layoutKv,
    int64_t             sparseMode,
    int64_t             preTokens,
    int64_t             nextTokens,
    int64_t             attentionMode,
    bool                returnSoftmaxLse,
    const aclTensor     *attentionOutOut,
    const aclTensor     *softmaxMaxOut,
    const aclTensor     *softmaxSumOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnSparseFlashAttention(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    const aclrtStream stream)
```

## aclnnSparseFlashAttentionGetWorkspaceSize

- **Parameters:**

> [!NOTE]  
>
>- Dimension definitions for the `query`, `key`, and `value` parameters:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br>`D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
>- `Q_S` or `S1` indicates the S dimension in the shape of `query`.<br>`KV_S` or `S2` indicates the S dimension in the shape of `key`.<br>`Q_N` or `N1` indicates `num_query_heads`.<br>`KV_N` or `N2` indicates `num_key_value_heads`.<br>`T1` indicates the T dimension in the shape of `query`.<br>`T2` indicates the accumulated sum of the input sample sequence lengths in the shape of `key`.

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 500px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 400px">
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
      <td>query (aclTensor) </td>
      <td>Input</td>
      <td>Query input of the attention structure.</td>
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
      <td>key (aclTensor)</td>
      <td>Input</td>
      <td>Key input of the attention structure</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>Total number of blocks when block_num is PageAttention.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_kv is PA_BSND, the shape is (block_num, block_size, KV_N, D).</li>
                <li>When layout_kv is BSND, the shape is (B, S2, KV_N, D).</li>
                <li>When layout_kv is TND, the shape is (T2, KV_N, D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>value (aclTensor)</td>
      <td>Input</td>
      <td>Value input of the attention structure.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>The shape is the same as that of the key.</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseIndices (aclTensor)</td>
      <td>Input</td>
      <td>Index of the KV cache for discrete selection.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>sparse_size indicates the number of blocks selected in discrete mode. Ensure that the valid values in each row are in the first half and the invalid values are in the second half. In addition, sparse_size must be greater than 0.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_query is BSND, the shape is (B, Q_S, KV_N, sparse_size).</li>
                <li>When layout_query is TND, the shape is (Q_T, KV_N, sparse_size).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTable (aclTensor)</td>
      <td>Input</td>
      <td>Block mapping table used by the KV cache in PageAttention.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>The length of the second dimension is greater than or equal to the number of blocks corresponding to the maximum S2 value in all batches, that is, the rounded-up value of S2_max/block_size.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>The supported shape is (B, S2/block_size).</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery (aclTensor)</td>
      <td>Input</td>
      <td>Number of valid tokens in the query of different batches.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>If seqlen is not specified, you can pass None, which indicates that the value is the same as the length of S in the shape of the query.</li>
                <li>The number of valid tokens in each batch in this input parameter cannot be greater than the dimension S in the query and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`.</li>
                <li>When layout_query is TND, this parameter must be passed, and the number of elements in this parameter is used as the value of B. The value of each element in this parameter indicates the total number of tokens in the current batch and all previous batches.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKv (aclTensor)</td>
      <td>Input</td>
      <td>Number of valid tokens in the key and value of different batches.</td>
      <td>
          <ul>
                <li>Empty tensors are not supported.</li>
                <li>If seqlen is not specified, you can pass None, which indicates that the value is the same as the length of S in the shape of the key.</li>
                <li>The number of valid tokens in each batch in this parameter cannot be greater than the dimension S in the key or value and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`.</li>
                <li>When layout_kv is TND or PA_BSND, this input parameter must be passed.</li>
                <li>If layout_kv is TND, the value of each element in this parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of the next element must be greater than or equal to that of the previous element.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryRope (aclTensor)</td>
      <td>Input</td>
      <td>Rope information of the query in the MLA structure.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>If layout_query is TND, the shape is (B, S1, N1, Dr).</li>
                <li>If layout_query is BSND, the shape is (T1, N1, Dr).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyRope (aclTensor)</td>
      <td>Input</td>
      <td>Rope information of the key in the MLA structure.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>If layout_kv is TND, the shape is (B, S1, N1, Dr).</li>
                <li>If layout_kv is BSND, the shape is (T1, N1, Dr).</li>
                <li>When layout_kv is set to PA_BSND, the shape is (block_num, block_size, N2, Dr).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue (double)</td>
      <td>Input</td>
      <td>Indicates the scaling coefficient.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the block size in the sparse phase.</td>
      <td>
          <ul>
                <li>When sparse_block_size is set to 1, token-wise sparsification is performed. Each token is regarded as an independent unit. When calculating the importance score, the independent association degree between each query token and each key-value token is evaluated.</li>
                <li>When sparse_block_size is set to a value greater than 1 and less than or equal to 128, block-wise sparsification is performed. The token sequence is divided into blocks of a fixed size, and the importance of each block is evaluated. Tokens in the same block share the same sparsification decision.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQuery (char)</td>
      <td>Input</td>
      <td>Indicates the data format of the input query.</td>
      <td>
          <ul>
                <li>If the user does not specify the value, the default value "BSND" can be transferred.</li>
                <li>Both BSND and TND can be transferred.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv (char)</td>
      <td>Input</td>
      <td>Indicates the data layout format of the input key.</td>
      <td>
          <ul>
                <li>If the user does not specify the value, the default value "BSND" can be transferred.</li>
                <li>TND, BSND, and PA_BSND can be transferred. PA_BSND is used when PageAttention is enabled.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode (int64_t)</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td>
          <ul>
                <li>When sparse_mode is set to 0, all computations are performed.</li>
                <li>When sparse_mode is set to 3, it indicates the mask of the rightDownCausal mode, which corresponds to the lower triangle scenario where the dividing line is from the lower right vertex to the upper left vertex.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens (int64_t)</td>
      <td>Input</td>
      <td>Number of preceding tokens to associate in attention computation for sparse computation.</td>
      <td>Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens (int64_t)</td>
      <td>Input</td>
      <td>Used for sparse computation, indicating that the attention needs to be associated with the last several tokens.</td>
      <td>Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionMode (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>Only 2 is supported, indicating the MLA-absorb mode.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse (bool)</td>
      <td>Input</td>
      <td>Whether to return softmax_max and softmax_sum.</td>
      <td>
          <ul>
                <li>True indicates that the values are returned. However, this parameter is not supported in graph mode. False indicates that the values are not returned. The default value is False.</li>
                <li>This parameter is supported only in training mode when layout_kv is not PA_BSND.</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut (aclTensor)</td>
      <td>Output</td>
      <td>Output in the formula.</td>
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
      <td>softmaxMaxOut (aclTensor)</td>
      <td>Output</td>
      <td>The Attention algorithm calculates the max value of the result of multiplying the query by the key to obtain softmax_max.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_query is BSND, the shape is (B, N2, S1, N1/N2).</li>
                <li>When layout_query is TND, the shape is (N2, T1, N1/N2).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
   <tr>
      <td>softmaxSumOut (aclTensor)</td>
      <td>Output</td>
      <td>Attention algorithm: The result of query multiplied by key is subtracted from softmax_max, and then exp is calculated. Finally, sum is calculated to obtain softmax_sum.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>When layout_query is BSND, the shape is (B, N2, S1, N1/N2).</li>
                <li>When layout_query is TND, the shape is (N2, T1, N1/N2).</li>
          </ul>
      </td>
      <td>x</td>
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
      <td>executor (aclOpExecutor)</td>
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
                <td>query, key, value, sparseIndices, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, keyRope, scaleValue, sparseBlockSize, layoutQuery, layoutKv, sparseMode, attentionMode, returnSoftmaxLse, attentionOut, softmaxMaxOut. The data type and format of softmaxSumOut are not supported.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttention

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
      <td>Workspace size allocated on the device, which is obtained by the first API aclnnSparseFlashAttentionGetWorkspaceSize.</td>
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

- This API can be used in inference scenarios.
- This API supports graph mode.
- N1 supports 1, 2, 4, 8, 16, 32, 64, and 128.
- block_size indicates the number of tokens in a block. The value of block_size must be a multiple of 16 and the maximum value is 1024.
- The value of D in the query parameter is the same as that of D in the key and value parameters, which is 512. The value of Dr in the query_rope parameter is the same as that of Dr in the key_rope parameter, which is 64.
- The data types of the query, key, and value parameters must be the same.
- sparse_block_size must be exactly divisible by block_size.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_incre_flash_attention_v4.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention.h"

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
    void* valueDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;
    void* keyRopeDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;
    aclTensor* softmaxMaxTensor = nullptr;
    aclTensor* softmaxSumTensor = nullptr;
    aclTensor* queryRopeTensor = nullptr;
    aclTensor* keyRopeTensor = nullptr; 
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 512};
    std::vector<int64_t> keyShape = {1, 2, 1, 512};
    std::vector<int64_t> valueShape = {1, 2, 1, 512};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 1, 2};
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
    std::vector<int64_t> softmaxMaxShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxSumShape = {1, 2, 1, 16};
    std::vector<int64_t> queryRopeShape = {1, 2, 1, 64};
    std::vector<int64_t> keyRopeShape = {1, 2, 1, 64};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t sparseIndicesShapeSize =  GetShapeSize(sparseIndicesShape);
    int64_t attentionOutShapeSize = GetShapeSize(attentionOutShape);
    int64_t softmaxMaxShapeSize = GetShapeSize(softmaxMaxShape);
    int64_t softmaxSumShapeSize = GetShapeSize(softmaxSumShape);
    int64_t queryRopeShapeSize = GetShapeSize(queryRopeShape);
    int64_t keyRopeShapeSize = GetShapeSize(keyRopeShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> attentionOutHostData(attentionOutShapeSize, 1);
    std::vector<float> softmaxMaxHostData(softmaxMaxShapeSize, 1);
    std::vector<float> softmaxSumHostData(softmaxSumShapeSize, 1);
    std::vector<float> queryRopeHostData(queryRopeShapeSize, 1);
    std::vector<float> keyRopeHostData(keyRopeShapeSize, 1);

    // Create query aclTensor.
    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, 
                             aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create sparseIndices aclTensor.
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr, 
                         aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create queryRope aclTensor.
    ret = CreateAclTensor(queryRopeHostData, queryRopeShape, &resources.queryRopeDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.queryRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create keyRope aclTensor.
    ret = CreateAclTensor(keyRopeHostData, keyRopeShape, &resources.keyRopeDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create attention_out aclTensor.
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &resources.attentionOutDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create softmax_max aclTensor.
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &resources.softmaxMaxDeviceAddr, 
                         aclDataType::ACL_FLOAT, &resources.softmaxMaxTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create softmax_sum aclTensor.
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &resources.softmaxSumDeviceAddr, 
                         aclDataType::ACL_FLOAT, &resources.softmaxSumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteSparseFlashAttention(TensorResources& resources, aclrtStream stream, 
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t d = 2;
    double scaleValue = 1 / sqrt(d);
    int64_t sparseBlockSize = 64;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKv[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKv, sizeof(layoutKv), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutKv failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int64_t attentionMode = 2;
    bool returnSoftmaxLse = false;
    aclOpExecutor* executor;

    int ret = aclnnSparseFlashAttentionGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor, resources.sparseIndicesTensor, nullptr, nullptr, nullptr, resources.queryRopeTensor, resources.keyRopeTensor,
                                                    scaleValue, sparseBlockSize, layoutQuery, layoutKv, sparseMode, preTokens,
                                                    nextTokens, attentionMode, returnSoftmaxLse, resources.attentionOutTensor, resources.softmaxMaxTensor, resources.softmaxSumTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseFlashAttention(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
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

void CleanupResources(TensorResources& resources, void* workspaceAddr, 
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.attentionOutTensor) {
      aclDestroyTensor(resources.attentionOutTensor);
    }
    if (resources.softmaxMaxTensor) {
      aclDestroyTensor(resources.softmaxMaxTensor);
    }
    if (resources.softmaxSumTensor) {
      aclDestroyTensor(resources.softmaxSumTensor);
    }
    if (resources.queryRopeTensor) {
      aclDestroyTensor(resources.queryRopeTensor);
    }
    if (resources.keyRopeTensor) {
      aclDestroyTensor(resources.keyRopeTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.attentionOutDeviceAddr) {
      aclrtFree(resources.attentionOutDeviceAddr);
    }
    if (resources.softmaxMaxDeviceAddr) {
      aclrtFree(resources.softmaxMaxDeviceAddr);
    }
    if (resources.softmaxSumDeviceAddr) {
      aclrtFree(resources.softmaxSumDeviceAddr);
    }
    if (resources.queryRopeDeviceAddr) {
      aclrtFree(resources.queryRopeDeviceAddr);
    }
    
    if (resources.keyRopeDeviceAddr) {
      aclrtFree(resources.keyRopeDeviceAddr);
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
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxMaxShape = {1, 2, 1, 16};
    std::vector<int64_t> softmaxSumShape = {1, 2, 1, 16}; 
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
    ret = ExecuteSparseFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
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
    printf("-----------attentionOut output-----------\n");
    PrintOutResult(attentionOutShape, &resources.attentionOutDeviceAddr);
    printf("-----------softmaxMax output-----------\n");
    PrintOutResult(softmaxMaxShape, &resources.softmaxMaxDeviceAddr);
    printf("-----------softmaxSum output-----------\n");
    PrintOutResult(softmaxSumShape, &resources.softmaxSumDeviceAddr);
    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
