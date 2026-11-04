# aclnnSparseFlashAttentionV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:19:35.711Z -->

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

- Description: sparse_flash_attention (SFA) is an efficient attention computation module for inference use cases with long sequences. The module greatly reduces the computation by "computing only the key parts", but introduces a large amount of discrete memory access, increasing the data transfer time and affecting the overall performance. The V2 version adds the sinks parameter.

- Formula:

$$
\text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
$$

Where $\tilde{K},\tilde{V}$ are the Key and Value of higher importance obtained based on a certain selection algorithm (such as `lightning_indexer`), generally having sparse or block-sparse characteristics, and $d_k$ is the dimension of each head of $Q,\tilde{K}$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSparseFlashAttentionV2GetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnSparseFlashAttentionV2" to execute the computation.

```Cpp
aclnnStatus aclnnSparseFlashAttentionV2GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *blockTableOptional,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthsKvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    const aclTensor     *sinksOptional,
    double              scaleValue,
    int64_t             sparseBlockSizeOptional,
    char                *layoutQueryOptional,
    char                *layoutKvOptional,
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
aclnnStatus aclnnSparseFlashAttentionV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnSparseFlashAttentionV2GetWorkspaceSize

- **Parameters:**

  > [!NOTE]
  >
  >- Dimensions of the query, key, and value parameters: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, N (Head Num) indicates the number of heads, D (Head Dim) indicates the size of the smallest unit of the hidden layer, and satisfies D=H/N. T indicates the accumulated sum of sequence lengths of input samples across all batches.
  >- Q\_S and S1 indicate the S in the query shape, KV\_S and S2 indicate the S in the key shape, Q\_N and N1 indicate num\_query\_heads, KV\_N and N2 indicate num\_key\_value\_heads, T1 indicates the T in the query shape, and T2 indicates the accumulated sum of sequence lengths of input samples in the key shape.

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
      <td>Does not support null tensors.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
     <td>          <ul>
                <li>If layout_query is BSND, the shape is (B,S1,N1,D).</li>
                <li>If layout_query is TND, the shape is (T1,N1,D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key (aclTensor)</td>
      <td>Input</td>
      <td>Key input of the attention structure.</td>
      <td>          <ul>
                <li>Does not support null tensor.</li>
                <li>When PageAttention is used, block_num indicates the total number of blocks.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
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
      <td>Supports passing a null pointer. When a null pointer is passed, key is used as value. Does not support null tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>The shape is the same as that of key.</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseIndices (aclTensor)</td>
      <td>Input</td>
      <td>Indices for discretely selecting kvCache.</td>
      <td>          <ul>
                <li>Does not support null tensor.</li>
                <li>sparse_size indicates the number of blocks for one discrete selection. Valid values in each row must be in the first half, and invalid values in the second half. sparse_size must be greater than 0.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layout_query is BSND, the shape is (B, Q_S, KV_N, sparse_size).</li>
                <li>If layout_query is TND, the shape is (Q_T, KV_N, sparse_size).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the block mapping table used for kvCache storage in PageAttention.</td>
      <td>          <ul>
                <li>Does not support null tensors.</li>
                <li>The second dimension length is not less than the number of blocks corresponding to the maximum S2 across all batches, that is, S2_max / block_size rounded up.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>The supported shape is (B,S2/block_size).</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of query in each batch.</td>
      <td>          <ul>
                <li>Does not support null tensors.</li>
                <li>If seqlen is not specified, None can be passed in, indicating that the value is the same as the S length of the query shape.</li>
                <li>The number of valid tokens of each batch in this parameter does not exceed the size of dimension S in query and is not less than 0. Supports a one-dimensional tensor of length B.</li>
                <li>If layout_query is TND, this parameter must be passed in, and the number of elements in this parameter is used as the value of B. The value of each element in this parameter indicates the total number of tokens in the current batch and all preceding batches.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of key and value in different batches.</td>
      <td>          <ul>
                <li>Does not support null tensors.</li>
                <li>If seqlen is not specified, None can be passed in, indicating that the S length is the same as that of the key shape.</li>
                <li>In this parameter, the number of valid tokens in each batch does not exceed the size of dimension S in key/value and is not less than 0. Supports a 1D tensor of length B.</li>
                <li>When layout_kv is TND or PA_BSND, this parameter must be passed in.</li>
                <li>When layout_kv is TND, the value of each element in this parameter indicates the total number of tokens in the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of an element must be greater than or equal to that of the preceding element.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryRopeOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the rope information of the query in the MLA structure.</td>
      <td>          <ul>
                <li>Optional. Must be paired with keyRopeOptional as a null pointer (rope not used), or paired with a non-null tensor (rope used, Dr is 64).</li>
                <li>Does not support null tensors.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layout_query is TND, the shape is (B,S1,N1,Dr).</li>
                <li>If layout_query is BSND, the shape is (T1,N1,Dr).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyRopeOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the rope information of the key in the MLA structure.</td>
      <td>          <ul>
                <li>Optional. Must be paired with queryRopeOptional as a null pointer (rope not used), or as a non-null tensor (rope used, Dr is 64).</li>
                <li>Does not support empty tensors.</li>
          </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layout_kv is TND, the shape is (B,S1,N1,Dr).</li>
                <li>If layout_kv is BSND, the shape is (T1,N1,Dr).</li>
                <li>If layout_kv is PA_BSND, the shape is (block_num,block_size,N2,Dr).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sinksOptional (aclTensor)</td>
      <td>Input</td>
      <td>Learnable sinks information in the attention structure.</td>
      <td>Does not support null tensor.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>          <ul>
                <li>Supported shape: (N1).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue (double)</td>
      <td>Input</td>
      <td>Indicates the scaling factor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSizeOptional (int64_t)</td>
      <td>Input</td>
      <td>Indicates the block size in the sparse stage.</td>
      <td>          <ul>
                <li>If sparse_block_size is 1, the use case is token-wise sparsification. Each token is treated as an independent unit. When computing importance scores, the independent correlation between each query token and each key-value token is evaluated.</li>
                <li>If sparse_block_size is greater than 1 and less than or equal to 128, the use case is block-wise sparsification. The token sequence is divided into contiguous blocks of a fixed size, and importance is evaluated at the block level. Tokens in the same block share the same sparsification decision.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional (char)</td>
      <td>Input</td>
      <td>Indicates the data layout format of the input query.</td>
      <td>          <ul>
                <li>If not specifically specified by the user, pass the default value "BSND".</li>
                <li>Supports BSND and TND.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char)</td>
      <td>Input</td>
      <td>Indicates the data layout format of the input key.</td>
      <td>          <ul>
                <li>If not specified by the user, pass the default value "BSND".</li>
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
                <li>When sparse_mode is 0, all elements are computed.</li>
                <li>When sparse_mode is 3, the mask is in rightDownCausal mode, corresponding to the lower-triangular task where the dividing line starts from the bottom-right vertex and extends to the upper-left.</li>
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
      <td>Used for sparse computation. Indicates that attention needs to be associated with the preceding tokens.</td>
      <td>Only the default value 2^63-1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens (int64_t)</td>
      <td>Input</td>
      <td>Used for sparse computation. Indicates that attention is computed with association to the subsequent tokens.</td>
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
      <td>returnSoftmaxLse(bool)</td>
      <td>Input</td>
      <td>Indicates whether to return softmax_max and softmax_sum.</td>
      <td>          <ul>
                <li>True indicates returning, False indicates not returning; the default value is False.</li>
                <li>This parameter is supported only in the training use case where layout_kv is not PA_BSND.</li>
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
      <td>Does not support null tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layout_query is BSND, the shape is (B,S1,N1,D).</li>
                <li>If layout_query is TND, the shape is (T1,N1,D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>softmaxMaxOut (aclTensor)</td>
      <td>Output</td>
      <td>The Attention algorithm takes the max of the result of query multiplied by key to obtain softmax_max.</td>
      <td>Does not support null tensor.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layout_query is BSND, the shape is (B,N2,S1,N1/N2).</li>
                <li>If layout_query is TND, the shape is (N2,T1,N1/N2).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
   <tr>
      <td>softmaxSumOut (aclTensor)</td>
      <td>Output</td>
      <td>The Attention algorithm subtracts softmax_max from the result of query multiplied by key, applies exp, and then computes the sum to obtain softmax_sum.</td>
      <td>Does not support empty tensors.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>          <ul>
                <li>When layout_query is BSND, the shape is (B,N2,S1,N1/N2).</li>
                <li>When layout_query is TND, the shape is (N2,T1,N1/N2).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor)</td>
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
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>If a passed-in parameter is a required input, output, or attribute and is a null pointer, 161001 is returned. When value is a null pointer, key is used as value, and this error code is not returned due to a null value.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and data formats of query, key, value, sparseIndices, blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, queryRopeOptional, keyRopeOptional, sinksOptional, scaleValue, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, sparseMode, attentionMode, returnSoftmaxLse, attentionOut, softmaxMaxOut, and softmaxSumOut are not within the supported range.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttentionV2

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnSparseFlashAttentionV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSparseFlashAttentionV2 defaults to a deterministic implementation.
- This API supports use in inference scenarios.
- N1 support:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - N1 supports a range of 1-128.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - N1 supports 1/2/4/8/16/32/64/128.
  <!-- end id8 -->

- N2 supports only 1.
- block_size is the number of tokens in a block. The value of block_size must be a multiple of 16, with a maximum of 1024.
- The D value in the query parameter is equal to the D values of key and value, which is 512.
- queryRopeOptional and keyRopeOptional must be null pointers in pairs or both passed in as non-null tensors: when both are null pointers, it indicates that rope is not used, and attention is computed only on the 512-dimensional nope part; when both are non-null tensors, Dr is equal and is 64. Does not support null tensors.
- The data types of the query, key, and value parameters must be consistent.
- sparse_block_size must be divisible by block_size.

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Only sparse_block_size 1 is supported.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - Supports [1,128], and the value must be a power of 2. In the PageAttention use case, sparse_block_size must be divisible by block_size
  <!-- end id10 -->

- The sinks parameter is supported only on Ascend 950PR&950DT products.

## Example

The following is sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/*!
* \file test_aclnn_sparse_flash_attention_v2.cpp
 * \brief aclnnSparseFlashAttentionV2 API call example
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_sparse_flash_attention_v2.h"
#include "aclnn/opdev/platform.h"

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
    std::vector<short> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
    *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
                           CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %e\n", i, resultData[i]);
        }
    }

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
// Boilerplate, AscendCL initialization
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
aclDataType dataType, aclTensor** tensor) {
                    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // Call aclrtMalloc to allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

// Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
        }

// Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
    shape.data(), shape.size(), *deviceAddr);
                              return 0;
    }

int main() {
if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
    return 0;
        }
    // 1. (Boilerplate) Initialize device/context/stream. For details, see the list of public AscendCL APIs
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

// 2. Construct the input and output. Customize the construction based on the API
    std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
    std::vector<int64_t> kShape = {2048, 1, 512};              // T2, N2, D
    std::vector<int64_t> vShape = {2048, 1, 512};              // T2, N2, D
    std::vector<int64_t> sparseIndicesShape = {1, 1, 2048};    // T1, N2, K
    std::vector<int64_t> outShape = {1, 16, 512};             // T1, N1, D
    std::vector<int64_t> softmaxMaxShape = {1, 1, 16};        // N2, T1, G
    std::vector<int64_t> softmaxSumShape = {1, 1, 16};        // N2, T1, G
    std::vector<int64_t> actSeqQLenshape = {1};               // B
    std::vector<int64_t> actSeqKvLenshape = {1};           // B
    std::vector<int64_t> qRopeShape = {1, 16, 64};            // T1, N1, Drope
    std::vector<int64_t> kRopeShape = {2048, 1, 64};          // T2, N2, Drope
    std::vector<int64_t> sinksShape = {16};                   // N1

void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* actSeqQLenDeviceAddr = nullptr;
    void* actSeqKvLenDeviceAddr = nullptr;
    void* qRopeDeviceAddr = nullptr;
    void* kRopeDeviceAddr = nullptr;
    void* sinksDeviceAddr = nullptr;

aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* sparseIndices = nullptr;
    aclTensor* out = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* actSeqQLen = nullptr;
    aclTensor* actSeqKvLen = nullptr;
    aclTensor* qRope = nullptr;
    aclTensor* kRope = nullptr;
    aclTensor* sinks = nullptr;

std::vector<op::fp16_t> qHostData(1 * 16 * 512, 1.0);
    std::vector<op::fp16_t> kHostData(2048 * 1 * 512, 1.0);
    std::vector<op::fp16_t> vHostData(2048 * 1 * 512, 1.0);
    std::vector<int32_t> sparseIndicesHostData(2048);
    std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
    std::vector<op::fp16_t> outHostData(1 * 16 * 512, 1.0);
    std::vector<float> softmaxMaxHostData(16, 3.0);
    std::vector<float> softmaxSumHostData(16, 3.0);
    std::vector<int32_t> actSeqQLenHostData(1, 1);
    std::vector<int32_t> actSeqKvLenHostData(1, 2048);
    std::vector<op::fp16_t> qRopeHostData(1 * 16 * 64, 1.0);
    std::vector<op::fp16_t> kRopeHostData(2048 * 1 * 64, 1.0);
    std::vector<float> sinksHostData(16, 0.0);

ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape,
    &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
                          CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape,
    &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
                          CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape,
    &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
                          CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape,
    &actSeqQLenDeviceAddr, aclDataType::ACL_INT32, &actSeqQLen);
                          CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape,
    &actSeqKvLenDeviceAddr, aclDataType::ACL_INT32, &actSeqKvLen);
                          CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

double scaleValue = 0.0416666666666667;
    int64_t sparseBlockSize = 1;
    int64_t sparseMode = 0;
    int64_t attentionMode = 2;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    // bool deterministic = false;
    char layoutQuery[5] = {'T', 'N', 'D', 0};
    char layoutKey[5] = {'T', 'N', 'D', 0};

// 3. Call the CANN operator library API. Replace it with the specific API name
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool returnSoftmaxLse = false;

// Call the first-phase API of aclnnSparseFlashAttentionV2
    ret = aclnnSparseFlashAttentionV2GetWorkspaceSize(q, k, v, sparseIndices, nullptr, actSeqQLen, actSeqKvLen,
    qRope, kRope, sinks, scaleValue, sparseBlockSize, layoutQuery,
                                                      layoutKey, sparseMode, preTokens, nextTokens, attentionMode,
                                                      returnSoftmaxLse, out, softmaxMax, softmaxSum, &workspaceSize,
                                                      &executor);
                                                      CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("aclnnSparseFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

// Allocate device memory based on the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }

// Call the second-phase API of aclnnSparseFlashAttentionV2
    ret = aclnnSparseFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);

// 4. (Boilerplate) Wait synchronously for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// 5. Release the aclTensor and aclScalar. Modify it based on the specific API definition
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(sparseIndices);
    aclDestroyTensor(out);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(actSeqQLen);
    aclDestroyTensor(actSeqKvLen);
    aclDestroyTensor(qRope);
    aclDestroyTensor(kRope);
    aclDestroyTensor(sinks);

// 6. Release device resources
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(sparseIndicesDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(actSeqQLenDeviceAddr);
    aclrtFree(actSeqKvLenDeviceAddr);
    aclrtFree(qRopeDeviceAddr);
    aclrtFree(kRopeDeviceAddr);
    aclrtFree(sinksDeviceAddr);
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
