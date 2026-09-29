# aclnnSparseLightningIndexerGradKLLoss

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|     √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Function: The SparselightningIndexerGradKlLoss operator is the backward operator of LightningIndexer and integrates the loss calculation function. The LightningIndexer operator selects the top K tokens with the highest intrinsic relationship between the query token and key token, and stores them in SparseIndices. This reduces the amount of attention calculation in long sequence scenarios and accelerates the inference and training performance of the network with long sequences.

- Formula:
   The formula for calculating the top-k value can be expressed as follows:

   $$
   I_{t,:}=W_{t,:}@ReLU(q_{t,:}@(K_{:t,:})^T)
   $$

   In the formula, $W$ is the weights corresponding to the $t$th token, $q$ is the matrix obtained by combining $G$ query heads corresponding to the $t$th token, and $K$ is the $K$ matrix of the $t$th row.

   LightningIndexer is trained separately, and the corresponding loss function is as follows:

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
   $$

   In the formula, $p$ is the target distribution, which is obtained by summing up all the heads of the main attention score and then performing L1 regularization on the sum result in the context direction. $D_{KL}$ is the KL divergence, and its expression is as follows:
   
   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the loss can be obtained by derivation as follows:
   
   $$
   dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right. 
   $$

   The chain rule can be used to calculate the gradients of the weights, query, and key matrices.
   
   $$
   dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left)  \left) \mathop{{}}\nolimits^{{T}}\right. \right. \right. \right. 
   $$

   $$
   d\mathop{{q}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@K\mathop{{}}\nolimits_{{:t,:}}
   $$

   $$
   dK\mathop{{}}\nolimits_{{:t,:}}= \left( dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{{T}}@q\mathop{{}}\nolimits_{{:t,:}}\right. \right. 
   $$

   S is the softmax result of the QK matrix.

**Description**
   The data format of `query`, `key`, and `value` can be interpreted from multiple dimensions. To be specific, **B (Batch)** indicates the size of an input sample batch, **S (Seq-Length)** indicates the length of the input sample sequence, **H (Head-Size)** indicates the size of the hidden layer, **N (Head-Num)** indicates the number of heads, and **D (Head-Dim)** indicates the minimum unit size of the hidden layer (**D** = **H**/**N**). **T** indicates the total length of all input sample sequences.
    -->

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnSparseLightningIndexerGradKLLoss` is called to perform computation.

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
    int64_t              pre_tokens,
    int64_t              next_tokens,
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
    aclrtStream stream)
```

## aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize

- **Parameters**:
  
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
            <td>The data type must be the same as that of key/queryIndex/keyIndex.</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D), (T1,N1,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>key</td>
            <td>Input</td>
            <td>Input K of the attention structure.</td>
            <td>The data type must be the same as that of query/queryIndex/keyIndex.</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D), (T2,N2,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>queryIndex</td>
            <td>Input</td>
            <td>Input queryIndex of the lightingIndexer structure.</td>
            <td>The data type must be the same as that of query/key/keyIndex.</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,Nidx1,D), (T1,Nidx1,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>keyIndex</td>
            <td>Input</td>
            <td>Input keyIndex of the lightingIndexer structure.</td>
            <td>The data type must be the same as that of query/key/queryIndex.</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,Nidx2,D), (T2,Nidx2,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>weights</td>
            <td>Input</td>
            <td>Weight.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16, FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx1), (T1,Nidx1)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>Input</td>
            <td>topk_index, which is used to select the key and value corresponding to each query.</td>
            <td>-</td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx2,K), (T1,Nidx2,K)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxMax</td>
            <td>Input</td>
            <td>Intermediate output of the forward attention calculation.</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>softmaxSum</td>
            <td>Input</td>
            <td>Intermediate output of the forward attention calculation.</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G), (N2,T1,G)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>queryRope</td>
            <td>Input</td>
            <td>MLA rope part: output of the position encoding of the query.</td>
            <td>
            The dimension is consistent with that of the query layout.
            </td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,Dr), (T1,N1,Dr)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>keyRope</td>
            <td>Input</td>
            <td>MLA rope part: output of the position encoding of the key.</td>
            <td>
            The dimension is consistent with that of the key layout.
            </td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,Dr), (T2,N2,Dr)</td>
            <td>x</td>
        </tr>    
        <tr>
            <td>actualSeqLengthsQuery</td>
            <td>Input</td>
            <td>Number of valid tokens in the query in each batch.</td>
            <td>
            <ul>
                <li>Value dependency.</li>
                <li>The length is the same as that of B.</li>
                <li>The accumulated sum is the same as that of T1.</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>actualSeqLengthsKey</td>
            <td>Input</td>
            <td>Number of valid tokens of the key in each batch.</td>
            <td>
            <ul>
                <li>Value dependency.</li>
                <li>The length is the same as that of B.</li>
                <li>The accumulated sum is the same as that of T2.</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>ND</td>
            <td>(B,)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>scaleValue</td>
            <td>Input</td>
            <td>Scale factor.</td>
            <td>Reciprocal of the square root of <code>d</code> in the formula.
            </td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>Input</td>
            <td>Layout format.</td>
            <td>
            Only the BSND and TND formats are supported.
            </td>
            <td>STRING</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>sparseMode</td>
            <td>Input</td>
            <td>Sparse mode.</td>
        <td>
            <ul>
                <li>Sparse mode. For details about sparse modes, see <a href="#Constraints">Constraints</a>.</li>
                <li>Only mode 3 is supported.</li>
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
            <td>Deterministic computation</td>
            <td>
            The deterministic configuration of the entire network is preferred. This parameter does not take effect.
            </td>
            <td>BOOL</td>
            <td>-</td>
            <td>-</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dQueryIndex</td>
            <td>Output</td>
            <td>Gradient of QueryIndex.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S1,Nidx1,D), (T1,Nidx1,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dKeyIndex</td>
            <td>Output</td>
            <td>Gradient of KeyIndex.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16</td>
            <td>ND</td>
            <td>(B,S2,Nidx2,D), (T2,Nidx2,D)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>dWeights</td>
            <td>Output</td>
            <td>Gradient of the weights.</td>
            <td>-</td>
            <td>FLOAT16, BFLOAT16, FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,Nidx1), (T1,Nidx1)</td>
            <td>x</td>
        </tr>
        <tr>
            <td>loss</td>
            <td>Output</td>
            <td>Loss function value.</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(1,)</td>
            <td>x</td>
        </tr>
        </tbody>
    </table>

- **Returns:**

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
                <td>The data types and formats of the input variables such as query, key, queryIndex, keyIndex, weights, sparseIndices, and softmaxMax are not supported.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>An exception occurred when the NPU Runtime API was called.</td>
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
        <td>Size of the workspace allocated on the device, which is obtained by calling the first API aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize.</td>
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

- Deterministic computation:
  - By default, aclnnSparseLightningIndexerGradKLLoss is implemented in non-deterministic mode. Deterministic computing cannot be enabled using aclrtCtxSetSysParamOpt.
- Common Constraints
    - Handling of the scenario where the input parameter is empty:
        - If `query` is an empty tensor, the result is returned directly.
        - The scenario where the input parameter is empty in the general restrictions is the same as that in the FAG.
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
            <td>In defaultMask mode, if attenmask is not passed, the mask operation is not performed, and preTokens and nextTokens are ignored. If attenmask is passed, the complete attenmask matrix must be passed, indicating that the part between preTokens and nextTokens needs to be calculated.</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask. The complete attenmask matrix must be passed.</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Mask in leftUpCausal mode. The optimized attenmask matrix needs to be passed.</td>
            <td>Not supported.</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask in rightDownCausal mode, corresponding to the lower triangle scenario where the right vertex is used as the division point. The optimized attenmask matrix needs to be passed.</td>
            <td>Supported</td>
  </tr>
        <tr>
            <td>4</td>
            <td>Mask in band mode. The optimized attenmask matrix needs to be passed.</td>
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
            <td>Not supported. </td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>Not supported. </td>
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
                <th>Specification</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>deterministic</td>
            <td>bool</td>
            <td>
            A2/A3 supports deterministic computation.<br>
            The 950 does not support deterministic computation.
            </td>
        </tr>
        <tr>
            <td>B</td>
            <td>A2/A3 supports 1 to 256.<br>
                The 950 supports 1~128</td>.
            <td>-</td>
        </tr>
        <tr>
            <td>S1, S2</td>
            <td>S1 supports 1 to 8K, and S2 supports 1~128K</td>.
            <td>S1 and S2 support unequal lengths.</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>32, 64, 128</td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>Nidx1</td>
            <td>
            A2/A3 supports 8, 16, 32, and 64.<br>
            The 950 supports 32 and 64.
            </td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA is MQA, and Nidx2 is 1.</td>
        </tr>
        <tr>
            <td>Nidx2</td>
            <td>1</td>
            <td>SparseFA is MQA, and N2 is 1.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>The D value of query is different from that of query_index.</td>
        </tr>
        <tr>
            <td>Drope</td>
            <td>64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>K</td>
            <td>1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>
- Typ.
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification</th>
                <th>Typical Value</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>N1=128/64; D =512</td>
        </tr>
        <tr>
            <td>queryIndex</td>
            <td>
            A2/A3 supports N1 = 64/32/16/8; D = 128; S1 = 64k/128k.<br>
            The 950 supports N1 = 64/32; D = 128; S1 = 64k/128k.
            </td>
        </tr>
        <tr>
            <td>keyIndex</td>
            <td>D = 128</td>
        </tr>
        <tr>
            <td>topk</td>
            <td>topk = 1024/2048/3072/4096/5120/6144/7168/8192</td>
        </tr>
        <tr>
            <td>qRope</td>
            <td>d= 64</td>
        </tr>
        </tbody>
    </table>

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  int32_t deviceId = 3;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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
  std::vector<int64_t> dWeightShape = {1,1,32};
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
  std::vector<float> sparseIndicesHostData(2048, 1);
  std::vector<float> softmaxMaxHostData(1*64, 1);
  std::vector<float> softmaxSumHostData(1*64, 1);

  std::vector<float> dQIndexHostData(1*32*128, 1);
  std::vector<float> dKIndexHostData(1*1*128, 1);
  std::vector<float> dWeightHostData(1*1*32, 1);
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

    // 3. Call the CANN operator library API. Change the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize.
  ret = aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(
            q, k, qIndex, kIndex, weight, sparseIndices, softmaxMax, softmaxSum, qRope, kRope, acSeqQLen, acSeqKvLen,
            scaleValue, layOut, sparseMode, preTokens, nextTokens, deterministic, dQIndex, dKIndex, dWeight, loss,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnSparseLightningIndexerGradKLLoss.
  ret = aclnnSparseLightningIndexerGradKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseLightningIndexerGradKLLoss failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Fixed writing) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(dQIndexShape, &dQIndexDeviceAddr);
  PrintOutResult(dKIndexShape, &dKIndexDeviceAddr);
  PrintOutResult(dWeightShape, &dWeightDeviceAddr);
  PrintOutResult(lossShape, &lossDeviceAddr);
  
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
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
