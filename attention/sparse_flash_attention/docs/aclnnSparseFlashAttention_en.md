# aclnnSparseFlashAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:09:04.934Z -->

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

- Description: sparse_flash_attention (SFA) is an efficient attention computation module designed for large sequence length inference scenarios. This module significantly reduces the computation workload by "computing only the critical parts." However, it introduces a large amount of discrete memory access, which increases data transfer time and thereby affects overall performance.

- Formula:

$$
\text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
$$

Where $\tilde{K},\tilde{V}$ are the Key and Value of higher importance obtained based on a certain selection algorithm (such as `lightning_indexer`), generally having sparse or block-sparse characteristics, and $d_k$ is the dimension of each head of $Q,\tilde{K}$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSparseFlashAttentionGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation flow of the operator, and then call "aclnnSparseFlashAttention" to perform the computation.

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
    const aclTensor     *attentionOut,
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
  >- Dimensions of query, key, and value: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the hidden layer size, N (Head Num) indicates the number of heads, D (Head Dim) indicates the size of the smallest unit of the hidden layer, where D = H/N, and T indicates the sum of the sequence lengths of input samples across all batches.
  >- S1 indicates S in the query shape, S2 indicates S in the key shape, N1 indicates the number of query heads, N2 indicates the number of key heads, T1 indicates T in the query shape, and T2 indicates the sum of the sequence lengths of input samples in the key shape.

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
      <td>query (aclTensor)</td>
      <td>Input</td>
      <td>Query input of the attention structure.</td>
      <td>Does not support null tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
     <td>          <ul>
                <li>If layoutQuery is BSND, the shape is (B,S1,N1,D).</li>
                <li>If layoutQuery is TND, the shape is (T1,N1,D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key (aclTensor)</td>
      <td>Input</td>
      <td>attention structural Key input.</td>
      <td>          <ul>
                <li>Does not support null tensor.</li>
                <li>blockNum is the total number of blocks when PageAttention is used.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>When layoutKv is PA_BSND, the shape is (blockNum, blockSize, N2, D).</li>
                <li>When layoutKv is BSND, the shape is (B, S2, N2, D).</li>
                <li>When layoutKv is TND, the shape is (T2, N2, D).</li>
          </ul>
      </td>
      <td>✓</td>
    </tr>
    <tr>
      <td>value (aclTensor)</td>
      <td>Input</td>
      <td>attention structural Value input.</td>
      <td>Supports passing a null pointer. When a null pointer is passed, key is used as value. Does not support null tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>The shape is the same as that of key.</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>sparseIndices (aclTensor)</td>
      <td>Input</td>
      <td>Index for discretely selecting kvCache.</td>
      <td>          <ul>
                <li>Does not support null tensor.</li>
                <li>sparseSize indicates the number of blocks selected in one discrete selection. Valid values in each row must be in the first half, and invalid values in the second half. sparseSize must be greater than 0.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layoutQuery is BSND, the shape is (B, S1, N2, sparseSize).</li>
                <li>If layoutQuery is TND, the shape is (T1, N2, sparseSize).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTable (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the block mapping table used for kvCache storage in PageAttention.</td>
      <td>          <ul>
                <li>Does not support null tensor.</li>
                <li>The length of the second dimension must be no less than the number of blocks corresponding to the maximum S2 across all batches, that is, S2_max / blockSize rounded up.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>Supported shape: (B, S2/blockSize).</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens in query for different batches.</td>
      <td>          <ul>
                <li>Does not support null tensors.</li>
                <li>If seqlen is not specified, pass in None to indicate that the value is the same as the S dimension of the query shape.</li>
                <li>In this parameter, the number of valid tokens of each batch does not exceed the size of the S dimension in query and is not less than 0. A one-dimensional tensor of length B is supported.</li>
                <li>When layoutQuery is TND, this parameter is mandatory, and the number of elements in this parameter is used as the value of B. In this parameter, the value of each element indicates the total number of tokens in the current batch and all preceding batches.</li>
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
      <td>Indicates the number of valid tokens of key and value in different batches.</td>
      <td>          <ul>
                <li>Does not support null tensor.</li>
                <li>If seqlen is not specified, pass None, indicating that the length is the same as the S dimension of the key shape.</li>
                <li>The number of valid tokens of each batch in this parameter does not exceed the size of dimension S in key/value and is not less than 0. A one-dimensional tensor of length B is supported.</li>
                <li>This parameter is mandatory when layoutKv is TND or PA_BSND.</li>
                <li>When layoutKv is TND, the value of each element in this parameter indicates the total number of tokens of the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element.</li>
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
      <td>Indicates the rope information of the query in the MLA structure.</td>
      <td>          <ul>
                <li>Optional. Must be paired with keyRope: both are null pointers (rope not used), or both are non-null tensors (rope used, Dr is 64).</li>
                <li>Does not support null tensor.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layoutQuery is TND, the shape is (B,S1,N1,Dr).</li>
                <li>If layoutQuery is BSND, the shape is (T1,N1,Dr).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyRope (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the rope information of the key in the MLA structure.</td>
      <td>          <ul>
                <li>Optional. Must be paired with queryRope as null pointers (rope not used), or non-null tensors (rope used, Dr is 64).</li>
                <li>Does not support null tensors.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layoutKv is TND, the shape is (B,S1,N1,Dr).</li>
                <li>If layoutKv is BSND, the shape is (T1,N1,Dr).</li>
                <li>If layoutKv is PA_BSND, the shape is (blockNum,blockSize,N2,Dr).</li>
          </ul>
      </td>
      <td>✓</td>
    </tr>
    <tr>
      <td>scaleValue (double)</td>
      <td>Input</td>
      <td>Represents the scaling factor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the block size of the sparse stage.</td>
      <td>          <ul>
                <li>If sparseBlockSize is 1, the Token-wise sparsification use case applies. Each token is treated as an independent unit. When computing importance scores, the independent correlation between each query token and each key-value token is evaluated.</li>
                <li>If sparseBlockSize is greater than 1 and less than or equal to 128, the Block-wise sparsification use case applies. The token sequence is divided into fixed-size contiguous blocks, importance is evaluated on a per-block basis, and tokens within a block share the same sparsification decision.</li>
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
      <td>Identifies the data layout format of the input query.</td>
      <td>          <ul>
                <li>If not specified by the user, the default value "BSND" can be passed.</li>
                <li>BSND and TND are supported.</li>
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
      <td>Identifies the data layout format of the input key.</td>
      <td>          <ul>
                <li>If not specified by the user, the default value "BSND" can be passed.</li>
                <li>TND, BSND, and PA_BSND are supported. PA_BSND is used when PageAttention is enabled.</li>
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
      <td>Indicates the sparse mode.</td>
      <td>          <ul>
                <li>If sparseMode is 0, all computations are performed.</li>
                <li>If sparseMode is 3, the mask is in rightDownCausal mode, corresponding to the lower triangle with the dividing line extending from the bottom-right vertex to the top-left.</li>
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
      <td>Used for sparse computation. Indicates that attention needs to be computed in association with the preceding tokens.</td>
      <td>Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens (int64_t)</td>
      <td>Input</td>
      <td>Used for sparse computation. Indicates that attention computation needs to be associated with the next several tokens.</td>
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
      <td>Indicates whether to return softmaxMax and softmaxSum.</td>
      <td>          <ul>
                <li>True indicates return, and False indicates no return; the default value is False.</li>
                <li>This parameter is supported only in the training use case when layoutKv is not PA_BSND.</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut(aclTensor)</td>
      <td>Output</td>
      <td>Output in the formula.</td>
      <td>Does not support null tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>When layoutQuery is BSND, the shape is (B,S1,N1,D).</li>
                <li>When layoutQuery is TND, the shape is (T1,N1,D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>softmaxMaxOut(aclTensor)</td>
      <td>Output</td>
      <td>softmaxMax is obtained by taking the max of the result of multiplying query by key in the Attention algorithm.</td>
      <td>Does not support null tensor.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>          <ul>
                <li>When layoutQuery is BSND, the shape is (B,N2,S1,N1/N2).</li>
                <li>When layoutQuery is TND, the shape is (N2,T1,N1/N2).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
   <tr>
      <td>softmaxSumOut (aclTensor)</td>
      <td>Output</td>
      <td>In the Attention algorithm, softmaxSum is obtained by subtracting softmaxMax from the result of query multiplied by key, applying exp, and then computing the sum.</td>
      <td>Does not support null tensors.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layoutQuery is BSND, the shape is (B,N2,S1,N1/N2).</li>
                <li>If layoutQuery is TND, the shape is (N2,T1,N1/N2).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return value</th>
            <th>Error code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>If a mandatory input, output, or mandatory attribute is a null pointer, 161001 is returned. When value is a null pointer, key is used as value, and this error code is not returned due to a null value.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and data formats of query, key, value, sparseIndices, blockTable, actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, keyRope, scaleValue, sparseBlockSize, layoutQuery, layoutKv, sparseMode, attentionMode, returnSoftmaxLse, attentionOut, softmaxMaxOut, and softmaxSumOut are out of the supported range.</td>
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
      <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnSparseFlashAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSparseFlashAttention defaults to a deterministic implementation.
- This API supports use in inference scenarios.
- N2 supports only 1.
- blockSize indicates the number of tokens in a block. The value of blockSize is a multiple of 16, and the maximum supported value is 1024.
- The D of query is equal to the D of key and value, which is 512.
- queryRope and keyRope must be both null pointers or both non-null tensors: if both are null pointers, it indicates that rope is not used and the attention computation is performed only on the 512-dimensional nope part; if both are non-null tensors, Dr must be 64. The API does not support null tensors.
- The data types of the query, key, and value parameters must be consistent.
- <term>Ascend 950PR&950DT products</term>:
  - N1 supports 1-128.
  - Only sparseBlockSize 1 is supported.
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - N1 supports 1/2/4/8/16/32/64/128.
  - sparseBlockSize supports [1,128] and must be a power of 2. In the PageAttention use case, sparseBlockSize must be divisible by blockSize.
  - key, value, and key_rope do not support non-contiguous tensors.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    std::vector<int64_t> softmaxMaxShape = {1, 1, 2, 1};
    std::vector<int64_t> softmaxSumShape = {1, 1, 2, 1};
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

    std::vector<aclFloat16> queryHostData(queryShapeSize, 1);
    std::vector<aclFloat16> keyHostData(keyShapeSize, 1);
    std::vector<aclFloat16> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<aclFloat16> attentionOutHostData(attentionOutShapeSize, 1);
    std::vector<float> softmaxMaxHostData(softmaxMaxShapeSize, 1);
    std::vector<float> softmaxSumHostData(softmaxSumShapeSize, 1);
    std::vector<aclFloat16> queryRopeHostData(queryRopeShapeSize, 1);
    std::vector<aclFloat16> keyRopeHostData(keyRopeShapeSize, 1);

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
    int64_t sparseBlockSize = 1;
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
    std::vector<int64_t> attentionOutShape = {1, 2, 1, 512};
    std::vector<int64_t> softmaxMaxShape = {1, 1, 2, 1};
    std::vector<int64_t> softmaxSumShape = {1, 1, 2, 1};
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
    printf("-----------attentionOut output -----------\n");
    PrintOutResult(attentionOutShape, &resources.attentionOutDeviceAddr);
    printf("-----------softmaxMax output -----------\n");
    PrintOutResult(softmaxMaxShape, &resources.softmaxMaxDeviceAddr);
    printf("-----------softmaxSum output -----------\n");
    PrintOutResult(softmaxSumShape, &resources.softmaxSumDeviceAddr);
    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
