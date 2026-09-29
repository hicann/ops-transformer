# aclnnSparseFlashAttentionGrad

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Function**: Rearranges data of the selectedBlockSize size from the key and value based on topkIndices, and then performs the backward output of attention calculation in the training scenario.

- **Calculation formula**: Rearranges selectedBlockCount pieces of data of the selectedBlockSize size from the keyIn and value based on the input topkIndice. The formula is as follows:

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

  Phase 1: Calculate $dP$ and $dV$ according to the matrix multiplication derivative rules.

</div>

  $$
   dP\mathop{{}}\nolimits_{{t,:}}=dO\mathop{{}}\nolimits_{{t,:}}\text{@}V\mathop{{}}\nolimits^{{T}}
  $$

  $$
   dV \left[ u \left] =P\mathop{{}}\nolimits_{{T}}^{{t,:}}\text{@}dO\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

<div style="padding-left:40px;">

   Phase 2: Calculate $dS$.

</div>

  $$
   d\mathop{{S}}\nolimits_{{t,:}}= \left[ P\mathop{{}}\nolimits_{{t,:}}@ \left( dP\mathop{{}}\nolimits_{{t,:}}-FlashSoftmaxGrad \left( dO,O \left)  \left)  \right] \right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

   Phase 3: Calculate $dQ$ and $dK$.

</div>

  $$
   d\mathop{{Q}}\nolimits_{{t,:}}=d\mathop{{S}}\nolimits_{{t,:}}@K \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}/\sqrt{{d\mathop{{}}\nolimits_{{k,:}}}}\right. \right.
  $$

  $$
   dK \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}=dS\mathop{{}}\nolimits_{{t,:t}}\mathop{{}}\nolimits^{{T}}\text{@}Q/\sqrt{{d\mathop{{}}\nolimits_{{t,:}}}}\right. \right. 
  $$

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSparseFlashAttentionGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnSparseFlashAttentionGrad` is called to perform computation.

```c++
aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor     *query, 
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *dOut,
    const aclTensor     *out,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthskvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    double               scaleValue,
    int64_t              sparseBlockSize,
    char                *layoutOptional,
    int64_t              sparseMode,
    int64_t              preTokens,
    int64_t              nextTokens,
    bool                 deterministic,
    const aclTensor     *dQueryOut,
    const aclTensor     *dKeyOut,
    const aclTensor     *dValueOut,
    const aclTensor     *dQueryRopeOutOptional,
    const aclTensor     *dKeyRopeOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnSparseFlashAttentionGrad(
    void             *workspace, 
    uint64_t          workspaceSize, 
    aclOpExecutor    *executor, 
    aclrtStream stream)
```

## aclnnSparseFlashAttentionGradGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 290px">
            <col style="width: 145px">
            </colgroup>
        <thead>
        <tr>
            <th>Name</th>
            <th>Input/Output</th>
            <th>Description</th>
            <th>Precaution</th>
            <th>Data Type</th>
            <th>Data Format</th>
            <th>Dimension (Shape)</th>
            <th>Non-contiguous Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>Input</td>
            <td>Input Q of the attention structure.</td>
            <td>
            The shape dimensions of query, key, value, sparseIndices, dOut, out, softmaxMax, and softmaxSum are the same.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)<br>
            B: Generalization is supported. S1: Generalization is supported. N1: 128, 64, 32, 16, 8, 4, 2, and 1 are supported. D: 512. T1: B x S1
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>key</td>
            <td>Input</td>
            <td>Input K of the attention structure.</td>
            <td>-</td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)<br>
            N2: 1; T2: B × S2
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>value</td>
            <td>Input</td>
            <td>Input V of the attention structure.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>Input</td>
            <td>Attention index with a higher weight selected in the sparse scenario.</td>
            <td>
            -
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,S1,N2,K), (T1,N2,K)<br>
            K: 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dOut</td>
            <td>Input</td>
            <td>Gradient of the attention output matrix.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>out</td>
            <td>Input</td>
            <td>Attention output matrix.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>Input</td>
            <td>Intermediate output of the forward attention calculation.</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)<br>
            G: N1/N2
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>Input</td>
            <td>Intermediate output of the forward attention calculation.</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)
            </td>
            <td>x</td>
        </tr>
  <tr>
            <td>actualSeqLengthsQueryOptional</td>
            <td>Input</td>
            <td>Number of valid tokens in the query in each batch.</td>
            <td>
            <ul>
                <li>This variable is available when the layout is TND.</li>
                <li>The length is the same as that of B.</li>
                <li>The accumulated sum is the same as that of T1.</li>
                <li>The value must be a non-negative number. If a negative value is passed, an alarm may be triggered.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>actualSeqLengthskvOptional</td>
            <td>Input</td>
            <td>Number of valid tokens in the key and value in each batch.</td>
            <td>
            <ul>
                <li>This variable is available when the layout is TND.</li>
                <li>The length is the same as that of B.</li>
                <li>The accumulated sum is the same as that of T2.</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>queryRopeOptional</td>
            <td>Input</td>
            <td>MLA rope part: output of the position encoding of the query.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)<br>
            Dr: 64
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>keyRopeOptional</td>
            <td>Input</td>
            <td>MLA rope part: output of the position encoding of the key.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr), (T2,N2,Dr)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>Input</td>
            <td>Scale factor.</td>
            <td>
            Recommended value: the reciprocal of the square root of d in the formula.
            </td>
            <td>FLOAT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseBlockSize</td>
            <td>Input</td>
            <td>Size of the selected block.</td>
            <td>
            A2/A3 supports 1, 8, 16, 32, and 64.<br>
            The 950 supports 1.
            </td>
            <td>INT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>Input</td>
            <td>Layout format.</td>
            <td>
            BSND and TND are supported.
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>Input</td>
            <td>Sparse mode.</td>
        <td>
            <ul>
                <li>Sparse mode. For details about the sparse modes, see <a href="#constraints">Restrictions and Limitations</a>.</li>
                <li>Only modes 0 and 3 are supported.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>x</td>
        </tr>
        <tr>
        <td>preTokens</td>
            <td>Input</td>
            <td>Start position of the sliding window for the S matrix in the Attention operator.</td>
            <td>
            <ul>
                <li>pre_tokens is valid only when sparseMode is set to 4.</li>
                <li>The value can only be 2147483647</li>.
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
        <td>nextTokens</td>
            <td>Input</td>
            <td>End position of the sliding window for the S matrix in the Attention operator.</td>
            <td>
            <ul>
                <li>When sparseMode is set to 4, next_tokens takes effect.</li>
                <li>The value can only be 2147483647</li>.
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
        <td>deterministic</td>
            <td>Input</td>
            <td>**Deterministic Computation**</td>
            <td>
            The value must be the same as that of the network-wide deterministic parameter use_deterministic_algorithms.
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dQuery</td>
            <td>Output</td>
            <td>Gradient of the query.</td>
            <td>
            The value must be the same as the shape dimension of the input query.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKey</td>
            <td>Output</td>
            <td>Indicates the gradient of the key.</td>
            <td>
            -
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)
            </td>
            <td>x</td>
        </tr>
        <tr>  
            <td>dValue</td>
            <td>Output</td>
            <td>Indicates the gradient of the value.</td>
            <td>
            The shape is the same as that of the input value.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)</td>
            <td>x</td>
        </tr>
          <tr>
            <td>dQueryRopeOptional</td>
            <td>Output</td>
            <td>Indicates the gradient of queryRope.</td>
            <td>
            <ul>
                <li>This variable is output only when the input queryRope exists.</li>
                <li>The shape is the same as that of the input query.</li>
            </ul>
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)
            </td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKeyRopeOptional</td>
            <td>Output</td>
            <td>Indicates the gradient of keyRope.</td>
            <td>
            This variable is output only when the input keyRope exists.
            </td>
            <td>BFLOAT16, FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr), (T2,N2,Dr)
            </td>
            <td>x</td>
        </tr>
        </tbody>
    </table>

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

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
                <td>The mandatory parameter or output is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data type and format of the input variable, such as query, key, value, and sparseIndices, are not supported.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception occurred when the NPU Runtime API was called.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_TILING_ERROR</td>
                <td>561002</td>
                <td>The value of the input parameter (such as layout or sparseMode) is out of the supported range, or the shape dimension of the input tensor does not meet the requirements.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttentionGrad

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>Address of the workspace to be allocated on the device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, which is obtained by the first API aclnnSparseFlashAttentionGradGetWorkspaceSize.</td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - By default, aclnnSparseFlashAttentionGrad is implemented in non-deterministic mode. Deterministic computing can be enabled by using aclrtCtxSetSysParamOpt.
- Common Constraints
    - Handling of scenarios where the input parameter is empty:
        - If `query` is an empty tensor, the result is returned directly.
    - Currently, only the scenario where the value and key are the same is supported.

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>Description</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>The mask operation is not performed.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask. The complete attenmask matrix must be transferred.</td>
            <td>Not supported. </td>
        </tr>
        <tr>
            <td>2</td>
            <td>Mask in leftUpCausal mode. The optimized attenmask matrix needs to be transferred.</td>
            <td>Not supported. </td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask in rightDownCausal mode, which corresponds to the lower triangular scenario where the right vertex is used for division.</td>
            <td>Supported</td>
  </tr>
        <tr>
            <td>4</td>
            <td>Mask in band mode. The optimized attenmask matrix needs to be transferred.</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>Not supported.</td>
        </tr>
        </tbody>
    </table>
- Specification Restrictions
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specifications</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>deterministic</td>
            <td>bool</td>
            <td>
            A2/A3 supports deterministic computing.<br>
            The 950 does not support deterministic computing.
            </td>
        </tr>
        <tr>
            <td>B</td>
            <td>1~256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1, S2</td>
            <td>1~128K</td>
            <td>S1 and S2 support unequal lengths.</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1, 2, 4, 8, 16, 32, 64, 128</td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA is MQA, and Nidx2 is 1.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Drope</td>
            <td>64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>K</td>
            <td>1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192</td>
            <td>A2/A3: It is not recommended that the value of K x sparseBlockSize exceed 100 KB. Otherwise, oom</td> may occur due to hardware restrictions of the internal algorithm.
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

## Examples

The following example is for reference only (using <term>Atlas A2 training products/Atlas A2 inference products</term> as examples). For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention_grad.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
  // Set the deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
  std::vector<int64_t> kShape = {2048, 1, 512};              // T2, N2, D
  std::vector<int64_t> vShape = {2048, 1, 512};              // T2, N2, D
  std::vector<int64_t> sparseIndicesShape = {1, 1, 2048};    // T1, N2, K
  std::vector<int64_t> outShape = {1, 16, 512};             // T1, N1, D
  std::vector<int64_t> dOutShape = {1, 16, 512};            // T1, N1, D
  std::vector<int64_t> softmaxMaxShape = {1, 1, 16};        // N2, T1, G
  std::vector<int64_t> softmaxSumShape = {1, 1, 16};        // N2, T1, G
  std::vector<int64_t> actSeqQLenshape = {1};               // B
  std::vector<int64_t> actSeqKvLenshape = {1};              // B
  std::vector<int64_t> qRopeShape = {1, 16, 64};            // T1, N1, Drope
  std::vector<int64_t> kRopeShape = {2048, 1, 64};          // T2, N2, Drope

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* sparseIndicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* dOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* actSeqQLenDeviceAddr = nullptr;
  void* actSeqKvLenDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dqRopeDeviceAddr = nullptr;
  void* dkRopeDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* sparseIndices = nullptr;
  aclTensor* out = nullptr;
  aclTensor* dOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* actSeqQLen = nullptr;
  aclTensor* actSeqKvLen = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;

  std::vector<short> qHostData(1 * 16 * 512, 1.0);
  std::vector<short> kHostData(2048 * 1 * 512, 1.0);
  std::vector<short> vHostData(2048 * 1 * 512, 1.0);
  std::vector<int32_t> sparseIndicesHostData(2048);
  std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
  std::vector<short> outHostData(1 * 16 * 512, 1.0);
  std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
  std::vector<float> softmaxMaxHostData(16, 3.0);
  std::vector<float> softmaxSumHostData(16, 3.0);
  std::vector<int32_t> actSeqQLenHostData(1, 1);
  std::vector<int32_t> actSeqKvLenHostData(1, 2048);
  std::vector<short> qRopeHostData(1 * 16 * 64, 1.0);
  std::vector<short> kRopeHostData(2048 * 1 * 64, 1.0);
  std::vector<short> dqHostData(1 * 16 * 512, 0);
  std::vector<short> dkHostData(2048 * 1 * 512, 0);
  std::vector<short> dvHostData(2048 * 1 * 512, 0);
  std::vector<short> dqRopeHostData(1 * 16 * 64, 0);
  std::vector<short> dkRopeHostData(2048 * 1 * 64, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOutHostData, dOutShape, &dOutDeviceAddr, aclDataType::ACL_FLOAT16, &dOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape, &actSeqQLenDeviceAddr, aclDataType::ACL_INT32, &actSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape, &actSeqKvLenDeviceAddr, aclDataType::ACL_INT32, &actSeqKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, vShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqRopeHostData, qRopeShape, &dqRopeDeviceAddr, aclDataType::ACL_FLOAT16, &dqRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkRopeHostData, kRopeShape, &dkRopeDeviceAddr, aclDataType::ACL_FLOAT16, &dkRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  double scaleValue = 0.088388;
  int64_t sparseBlockSize = 1;
  int64_t sparseMode = 0;
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  bool deterministic = false;
  char layout[5] = {'T', 'N', 'D', 0};
  
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // Call the first-phase API of aclnnSparseFlashAttentionGrad.
  ret = aclnnSparseFlashAttentionGradGetWorkspaceSize(q, k, v, sparseIndices, dOut, out, softmaxMax, softmaxSum, actSeqQLen, actSeqKvLen,
            qRope, kRope, scaleValue, sparseBlockSize, layout, sparseMode, preTokens, nextTokens, deterministic, dq, dk, dv, dqRope, dkRope, 
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnSparseFlashAttentionGrad.
  ret = aclnnSparseFlashAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionGrad failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Fixed writing) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(qShape, &dqDeviceAddr);
  PrintOutResult(kShape, &dkDeviceAddr);
  PrintOutResult(vShape, &dvDeviceAddr);
  PrintOutResult(qRopeShape, &dqRopeDeviceAddr);
  PrintOutResult(kRopeShape, &dkRopeDeviceAddr);
  
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(sparseIndices);
  aclDestroyTensor(out);
  aclDestroyTensor(dOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(actSeqQLen);
  aclDestroyTensor(actSeqKvLen);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyTensor(dqRope);
  aclDestroyTensor(dkRope);
  
  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sparseIndicesDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(dOutDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(dqRopeDeviceAddr);
  aclrtFree(dkRopeDeviceAddr);
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
