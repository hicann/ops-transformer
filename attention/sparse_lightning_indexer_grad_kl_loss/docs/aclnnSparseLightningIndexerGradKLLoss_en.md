# aclnnSparseLightningIndexerGradKLLoss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:37:59.639Z -->

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

- Description: The SparselightningIndexerGradKlLoss operator is the backward operator of LightningIndexer, with an additional fused loss computation function. The LightningIndexer operator selects the TopK highest intrinsic connections between QueryToken and KeyToken and stores them in SparseIndices, thereby reducing the computation workload of Attention in long-sequence scenarios and accelerating the inference and training performance of long-sequence networks.

- Formula:
   The formula for computing the value used to obtain Top-k can be expressed as:

   $$
   I_{t,:}=W_{t,:}@ReLU(q_{t,:}@(K_{:t,:})^T)
   $$

   Where $W$ is the weights corresponding to the $t$-th token, $q$ is the matrix formed by concatenating the $G$ query heads corresponding to the $t$-th token, and $K$ is a matrix with $t$ rows.

   LightningIndexer is trained separately, and its corresponding loss function is:

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
   $$

   where $p$ is the target distribution, obtained by summing the main attention score across all heads and then applying L1 normalization along the context dimension. $D_{KL}$ is the KL divergence, expressed as:

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the loss can be obtained by derivation:

   $$
   dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
   $$

   The gradients of the weights, query, and key matrices can be computed using the chain rule:

   $$
   dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left)  \left) \mathop{{}}\nolimits^{{T}}\right. \right. \right. \right.
   $$

   $$
   d\mathop{{q}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@K\mathop{{}}\nolimits_{{:t,:}}
   $$

   $$
   dK\mathop{{}}\nolimits_{{:t,:}}= \left( dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{{T}}@q\mathop{{}}\nolimits_{{:t,:}}\right. \right.
   $$

   Where S is the result of softmax on the QK matrix.

<!-- - **Note**:
   <blockquote>The query, key, and value data layout formats support interpretation in multiple dimensions. B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the minimum unit size of the hidden layer, satisfying D=H/N. T indicates the accumulated sum of sequence lengths of all Batch input samples.
   </blockquote> -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process. Then call the "aclnnSparseLightningIndexerGradKLLoss" API to execute the computation.

```c++
aclnnStatus aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *queryIndex,
    const aclTensor     *keyIndex,
    const aclTensor     *weights,
    const aclTensor     *sparseIndices,
    const aclTensor     *softmaxMax,
    const aclTensor     *softmaxSum,
    const aclTensor     *queryRope,
    const aclTensor     *keyRope,
    const aclIntArray   *actualSeqLengthsQuery,
    const aclIntArray   *actualSeqLengthsKey,
    double               scaleValue,
    char                *layout,
    int64_t              sparseMode,
    int64_t              preTokens,
    int64_t              nextTokens,
    bool                 deterministic,
    const aclTensor     *dQueryIndex,
    const aclTensor     *dKeyIndex,
    const aclTensor     *dWeights,
    const aclTensor     *loss,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnSparseLightningIndexerGradKLLoss(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">
            <col style="width: 400px">
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
            <th>Shape</th>
            <th>Non-contiguous Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>Input</td>
            <td>Input Q of the attention structure.</td>
            <td>            <ul>
                <li>The data type must be the same as that of key/queryIndex/keyIndex.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>FLOAT16, BFLOAT16 </td>
            <td>ND</td>
            <td>(B,S1,N1,DQuery), (T1,N1,DQuery)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>key</td>
            <td>Input</td>
            <td>Input K of the attention structure.</td>
            <td>            <ul>
                <li>The data type must be the same as that of query/queryIndex/keyIndex.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>FLOAT16, BFLOAT16 </td>
            <td>ND</td>
            <td>(B,S2,N2,DQuery), (T2,N2,DQuery)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>queryIndex</td>
            <td>Input</td>
            <td>Input queryIndex of the lightingIndexer structure.</td>
            <td>            <ul>
                <li>The data type must be the same as that of query/key/keyIndex.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,Nidx1,DQueryIndex), (T1,Nidx1,DQueryIndex)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>keyIndex</td>
            <td>Input</td>
            <td>Input keyIndex of the lightingIndexer structure.</td>
            <td>            <ul>
                <li>The data type is the same as query/key/queryIndex.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,Nidx2,DQueryIndex), (T2,Nidx2,DQueryIndex)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>weights</td>
            <td>Input</td>
            <td>Weights.</td>
            <td>Empty tensors are not supported.</td>
            <td>FLOAT16, BFLOAT16, FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx1), (T1,Nidx1)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>Input</td>
            <td>topk_index, used to select the keys and values corresponding to each query.</td>
            <td>Empty tensors are not supported.</td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx2,K), (T1,Nidx2,K)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>Input</td>
            <td>Intermediate output of forward attention computation.</td>
            <td>Empty tensors are not supported.</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>Input</td>
            <td>Intermediate output of the forward attention computation.</td>
            <td>Empty tensors are not supported.</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>queryRope</td>
            <td>Input</td>
            <td>MLA rope part: output of the Query position encoding.</td>
            <td>            <ul>
                <li>The layout dimensions are consistent with those of query.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,DRope), (T1,N1,DRope)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>Input</td>
            <td>MLA rope part: Output of Key position encoding.</td>
            <td>            <ul>
                <li>Consistent with the layout dimensions of key.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,DRope), (T2,N2,DRope)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>actualSeqLengthsQuery</td>
            <td>Input</td>
            <td>Number of valid Query tokens in each batch.</td>
            <td>            <ul>
                <li>Value dependency.</li>
                <li>The length is the same as B.</li>
                <li>The accumulated sum is the same as T1.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>actualSeqLengthsKey</td>
            <td>Input</td>
            <td>Number of valid Key tokens in each batch.</td>
            <td>            <ul>
                <li>Value dependency.</li>
                <li>Length consistent with B.</li>
                <li>Accumulated sum consistent with T2.</li>
                <li>Empty tensors are not supported.</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>Input</td>
            <td>Scale factor.</td>
            <td>Recommended value: the reciprocal of the square root of d in the formula.</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>Input</td>
            <td>layout format.</td>
            <td>            Only BSND and TND formats are supported.
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>Input</td>
            <td>Sparse mode.</td>
        <td>              <ul>
                <li>Indicates the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li>
                <li>Only mode 3 is supported.</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
            <td>preTokens</td>
            <td>Input</td>
            <td>Number of tokens retained by the left-side mask.</td>
            <td>            -
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>nextTokens</td>
            <td>Input</td>
            <td>Number of tokens retained by the right-side mask.</td>
            <td>-</td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
        <td>deterministic</td>
            <td>Input</td>
            <td>Deterministic computation.</td>
            <td>            The network-wide deterministic configuration takes precedence. This parameter has no effect.
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dQueryIndex</td>
            <td>Output</td>
            <td>Gradient of QueryIndex.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,Nidx1,DQueryIndex), (T1,Nidx1,DQueryIndex)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKeyIndex</td>
            <td>Output</td>
            <td>Gradient of KeyIndex.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,Nidx2,DQueryIndex), (T2,Nidx2,DQueryIndex)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dWeights</td>
            <td>Output</td>
            <td>Gradient of Weights.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16, FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx1), (T1,Nidx1)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>loss</td>
            <td>Output</td>
            <td>Value of the loss function.</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>Output</td>
            <td>Returns the size of the workspace to be requested on the Device.</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>Output</td>
            <td>Returns the op executor, which contains the operator computation process.</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - T1 must be greater than or equal to the accumulated sum of actualSeqLengthsQuery, and T2 must be greater than or equal to the accumulated sum of actualSeqLengthsKey.

<!-- end id7 -->
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
                <td>A required parameter or output is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and data formats of input variables such as query, key, queryIndex, keyIndex, weights, sparseIndices, and softmaxMax are out of the supported range.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception occurs when the API calls the npu runtime API for memory operations.</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseLightningIndexerGradKLLoss

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>Workspace memory address allocated on the device.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnSparseLightningIndexerGradKLLoss defaults to a non-deterministic implementation. To enable the deterministic computation, use aclrtSetSysParamOpt.
- General Constraints
    - The data types of query, key, queryIndex, and keyIndex must be the same.
    - If weights is not float32, the data types of query, key, queryIndex, keyIndex, and weights must be the same.
    - Handling of empty input parameters:
        - query is an empty tensor: return directly.
        - The scenario where the input is empty in the common constraints is consistent with FAG.
    - sparseMode descriptions:
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
            <td>defaultMask mode. If attenmask is not passed in, no mask operation is performed and preTokens and nextTokens are ignored. If attenmask is passed in, the complete attenmask matrix must be passed in, indicating that the portion between preTokens and nextTokens needs to be calculated.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask. The complete attenmask matrix must be passed in.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Mask of leftUpCausal mode. Pass in the optimized attenmask matrix.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask of rightDownCausal mode, corresponding to the lower-triangular use case with the right vertex as the boundary. Pass in the optimized attenmask matrix.</td>
            <td>Supported</td>
  </tr>
        <tr>
            <td>4</td>
            <td>Mask of band mode. Pass in the optimized attenmask matrix.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>Not supported</td>
        </tr>
        </tbody>
    </table>

- Specification constraints
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specification</th>
                <th>Specification Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>B</td>
            <td>Supports 1-256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1, S2</td>
            <td>S1 supports 1-32K, S2 supports ~~1-1M~~</td>
            <td>S1 and S2 support unequal lengths; S1 must be less than or equal to S2.</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>            1~128
            </td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>Nidx1</td>
            <td>            8, 16, 32, 64
            </td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA is MQA, and Nidx2=1.</td>
        </tr>
        <tr>
            <td>Nidx2</td>
            <td>1</td>
            <td>SparseFA is MQA, and N2=1.</td>
        </tr>
        <tr>
            <td>DQuery</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>DQueryIndex</td>
            <td>128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>DRope</td>
            <td>64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>K</td>
            <td>1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>
    <!-- npu="950" id8 -->
    <term>Ascend 950PR&950DT products</term>: N1 supports only 32, 48, 64, and 128, and Nidx1 additionally supports 24. Only the (48,24) combination is allowed between the two; other pairings are prohibited. B, S1, and S2 support generalization.
    <!-- end id8 -->

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_grad_kl_loss.h"

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
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

void PrintOutResultFp16(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // Boilerplate: AscendCL initialization
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
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (boilerplate) Initialize device/context/stream. For details, see the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 3;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> qShape = {1,64,512};
  std::vector<int64_t> kShape = {1,1,512};
  std::vector<int64_t> qRopeShape = {1,64,64};
  std::vector<int64_t> kRopeShape = {1,1,64};
  std::vector<int64_t> qIndexShape = {1,32,128};
  std::vector<int64_t> kIndexShape = {1,1,128};
  std::vector<int64_t> weightShape = {1,32};
  std::vector<int64_t> sparseIndicesShape = {1, 1,2048};
  std::vector<int64_t> softmaxMaxShape = {1, 1, 64};
  std::vector<int64_t> softmaxSumShape = {1, 1, 64};

  std::vector<int64_t> dQIndexShape = {1,32,128};
  std::vector<int64_t> dKIndexShape = {1,1,128};
  std::vector<int64_t> dWeightShape = {1,32};
  std::vector<int64_t> lossShape = {1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* sparseIndicesDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  void* dQIndexDeviceAddr = nullptr;
  void* dKIndexDeviceAddr = nullptr;
  void* dWeightDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* sparseIndices = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;

  aclTensor* dQIndex = nullptr;
  aclTensor* dKIndex = nullptr;
  aclTensor* dWeight = nullptr;
  aclTensor* loss = nullptr;

  std::vector<float> qHostData(1*64*512, 1);
  std::vector<float> kHostData(1*1*512, 1);
  std::vector<float> qRopeHostData(1*64*64, 1);
  std::vector<float> kRopeHostData(1*1*64, 1);
  std::vector<float> qIndexHostData(1*32*128, 1);
  std::vector<float> kIndexHostData(1*1*128, 1);
  std::vector<float> weightHostData(1*32, 1);
  std::vector<int32_t> sparseIndicesHostData(2048, 1);
  std::vector<float> softmaxMaxHostData(1*64, 1);
  std::vector<float> softmaxSumHostData(1*64, 1);

  std::vector<float> dQIndexHostData(1*32*128, 1);
  std::vector<float> dKIndexHostData(1*1*128, 1);
  std::vector<float> dWeightHostData(1*32, 1);
  std::vector<float> lossHostData(1, 1);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(dQIndexHostData, dQIndexShape, &dQIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dQIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dKIndexHostData, dKIndexShape, &dKIndexDeviceAddr, aclDataType::ACL_FLOAT16, &dKIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dWeightHostData, dWeightShape, &dWeightDeviceAddr, aclDataType::ACL_FLOAT16, &dWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t>  acSeqQLenOp = {1};
  std::vector<int64_t>  acSeqKvLenOp = {1};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  double scaleValue = 0.044194173824159216;
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  int64_t sparseMode = 3;
  bool deterministic = false;

  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize.
  ret = aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(
            q, k, qIndex, kIndex, weight, sparseIndices, softmaxMax, softmaxSum, qRope, kRope, acSeqQLen, acSeqKvLen,
            scaleValue, layOut, sparseMode, preTokens, nextTokens, deterministic, dQIndex, dKIndex, dWeight, loss,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API aclnnSparseLightningIndexerGradKLLoss.
  ret = aclnnSparseLightningIndexerGradKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseLightningIndexerGradKLLoss failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result on the device memory to the host, and modify the code based on the specific API definition.
  PrintOutResultFp16(dQIndexShape, &dQIndexDeviceAddr);
  PrintOutResultFp16(dKIndexShape, &dKIndexDeviceAddr);
  PrintOutResultFp16(dWeightShape, &dWeightDeviceAddr);
  PrintOutResult(lossShape, &lossDeviceAddr);

  // 6. Release aclTensor and aclScalar, and modify the code based on the specific API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(qRope);
  aclDestroyTensor(kRope);
  aclDestroyTensor(weight);
  aclDestroyTensor(sparseIndices);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  aclDestroyTensor(dQIndex);
  aclDestroyTensor(dKIndex);
  aclDestroyTensor(dWeight);
  aclDestroyTensor(loss);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(sparseIndicesDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);

  aclrtFree(dQIndexDeviceAddr);
  aclrtFree(dKIndexDeviceAddr);
  aclrtFree(dWeightDeviceAddr);
  aclrtFree(lossDeviceAddr);
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
