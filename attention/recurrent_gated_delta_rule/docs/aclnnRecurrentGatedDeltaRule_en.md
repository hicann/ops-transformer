# aclnnRecurrentGatedDeltaRule

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |    ×     |
|  <term>Atlas training products</term>   |     ×    |

## Description

- **API function**: Calculates the variable step-size Recurrent Gated Delta Rule (RGDR).

- Formula:

  RGDR is an operator designed for recurrent neural networks (RNNs) and serves as a key component in linear attention mechanisms.
  At each time step $t$, the network processes the current inputs query $q_t$, key $k_t$, and value $v_t$ along with the previous hidden state $S_{t-1}$ to produce the attention output $o_t$ and new hidden state $S_t$.
  The gating mechanism controls what proportion of new information enters the hidden state and what proportion of existing information is discarded.

  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$

  $$
  o := \frac{S_t q_t}{\sqrt{d_k}}
  $$

  In the formula, $S_{t-1},S_t \in R^{d_v \times d_k}$, $q_t, k_t \in R^{d_k}$, $v_t \in R^{d_v}$, $\alpha_t \in R$, $\beta_t \in R$, $o \in R^{d_v}$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRecurrentGatedDeltaRule` is called to perform computation.

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    aclTensor       *stateRef,
    const aclTensor *actualSeqLengths,
    const aclTensor *ssmStateIndices,
    const aclTensor *g,
    const aclTensor *gk,
    const aclTensor *numAcceptedTokens,
    float           scaleValue,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnRecurrentGatedDeltaRule(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnRecurrentGatedDeltaRuleGetWorkspaceSize

- **Parameters**

  <table style="undefined; table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 165px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>q in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>k in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>v in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>β in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>stateRef</td>
      <td>Input and output</td>
      <td>State matrix, that is, S in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(BlockNum, Nv, Dv, Dk)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>actualSeqLengths</td>
      <td>Input</td>
      <td>Valid sequence lengths of different batches.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ssmStateIndices</td>
      <td>Input</td>
      <td>Mapping index from the input sequence to the state matrix.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>state[ssmStateIndices[i]] indicates the state matrix of the i-th token.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(T,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>g</td>
      <td>Input</td>
      <td>Decay coefficient, α in the formula, which is equal to e^g.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>If nullptr is passed, it represents an all-zero tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gk</td>
      <td>Input</td>
      <td>Reserved and not supported by the current version.</td>
      <td><ul><li>Pass nullptr.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>(T, Nv, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numAcceptedTokens</td>
      <td>Input</td>
      <td>Number of accepted tokens per sequence.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scaling factor for query, corresponding to 1/sqrt(d_k) in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>o in the formula.</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>√</td>
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
  </tbody>
  </table>
  
  $B$ indicates the batch size. Let $L_i$ be the length of the *i*th sequence, then $T=\sum_i^B L_i$ represents the cumulative sequence length. $N_k$ indicates the number of key heads, $N_v$ indicates the number of value heads, $D_k$ indicates the dimension of the key vector, and $D_v$ indicates the dimension of the value vector.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, numAcceptedTokens, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input tensor is not supported.</td>
    </tr>
    <tr>
      <td>The data format of the input tensor is not supported.</td>
    </tr>
    <tr>
      <td>The shape of the input tensor is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnRecurrentGatedDeltaRule

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace (void*)</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnRecurrentGatedDeltaRuleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>workspace (void*)</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>workspace (void*)</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnRecurrentGatedDeltaRule` defaults to a deterministic implementation.
- The input shape size must meet the following constraints: $L_i \le 8$, $N_k \le 256$, $N_v \le 256$, $D_k \le 256$, $D_v \le 256$.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_gated_delta_rule.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, int16_tToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // (Fixed writing) Initialize resources.
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data from the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *gamaDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *stateRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *ssmStaIdDeviceAddr = nullptr;
    void *numAccTokDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;

    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *gama = nullptr;
    aclTensor *gamak = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *stateRef = nullptr;
    aclTensor *actSeqLen = nullptr;
    aclTensor *ssmStaId = nullptr;
    aclTensor *numAccTok = nullptr;
    aclTensor *attnOut = nullptr;

    // Custom inputs and attributes.
    int32_t batchSize = 2;
    int32_t mtp = 2;
    int32_t headKNum = 4;
    int32_t headVNum = 8;
    int32_t dimV = 32;
    int32_t dimK = 32;


    std::vector<int64_t> stateShape = {batchSize * mtp, headVNum, dimV, dimK};
    std::vector<int64_t> qkShape = {batchSize * mtp, headKNum, dimK};
    std::vector<int64_t> vShape = {batchSize * mtp, headVNum, dimV};
    std::vector<int64_t> gamaShape = {batchSize * mtp, headVNum};
    std::vector<int64_t> actSeqLenShape = {batchSize};
    std::vector<int64_t> ssmStaIdShape = {batchSize * mtp};
    std::vector<int16_t> stateRefHostData(GetShapeSize(stateShape));
    std::vector<int16_t> queryHostData(GetShapeSize(qkShape));
    std::vector<int16_t> keyHostData(GetShapeSize(qkShape));
    std::vector<int16_t> valueHostData(GetShapeSize(vShape));
    std::vector<float> gamaHostData(GetShapeSize(gamaShape));
    std::vector<int16_t> betaHostData(GetShapeSize(gamaShape));
    std::vector<int32_t> actSeqLenHostData(batchSize, mtp);
    std::vector<int32_t> ssmStaIdHostData(batchSize * mtp);
    std::vector<int32_t> numAccTokHostData(batchSize, 1);
    for (int i = 0; i < stateRefHostData.size(); i++) {
        stateRefHostData[i] = 1;
    }
    for (int i = 0; i < queryHostData.size(); i++) {
        queryHostData[i] = 1;
    }
    for (int i = 0; i < keyHostData.size(); i++) {
        keyHostData[i] = 1;
    }
    for (int i = 0; i < valueHostData.size(); i++) {
        valueHostData[i] = z;
    }
    for (int i = 0; i < betaHostData.size(); i++) {
        betaHostData[i] = z;
    }
    for (int i = 0; i < ssmStaIdHostData.size(); i++) {
        ssmStaIdHostData[i] = i;
    }

    std::vector<int16_t> attnOutHostData(valueHostData);

    ret = CreateAclTensor(stateRefHostData, stateShape, &stateRefDeviceAddr, aclDataType::ACL_BF16, &stateRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, qkShape, &queryDeviceAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkShape, &keyDeviceAddr, aclDataType::ACL_BF16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, vShape, &valueDeviceAddr, aclDataType::ACL_BF16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gamaHostData, gamaShape, &gamaDeviceAddr, aclDataType::ACL_FLOAT, &gama);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gamaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqLenHostData, actSeqLenShape, &actSeqLenDeviceAddr, aclDataType::ACL_INT32, &actSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ssmStaIdHostData, ssmStaIdShape, &ssmStaIdDeviceAddr, aclDataType::ACL_INT32, &ssmStaId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numAccTokHostData, actSeqLenShape, &numAccTokDeviceAddr, aclDataType::ACL_INT32, &numAccTok);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, vShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    float scale = 1.0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnRecurrentGatedDeltaRuleGetWorkspaceSize.
    ret = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(query, key, value, beta, stateRef, actSeqLen, ssmStaId, gama,
                                                       gamak, numAccTok, scale, attnOut, &workspaceSize,
                                                       &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnRecurrentGatedDeltaRule.
    ret = aclnnRecurrentGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRule failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    PrintOutResult(stateShape, &stateRefDeviceAddr);
    PrintOutResult(vShape, &attnOutDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(gama);
    aclDestroyTensor(beta);
    aclDestroyTensor(stateRef);
    aclDestroyTensor(actSeqLen);
    aclDestroyTensor(ssmStaId);
    aclDestroyTensor(numAccTok);
    aclDestroyTensor(attnOut);

    // 7. Release device resources.
    aclrtFree(query);
    aclrtFree(key);
    aclrtFree(value);
    aclrtFree(gama);
    aclrtFree(beta);
    aclrtFree(stateRef);
    aclrtFree(actSeqLen);
    aclrtFree(ssmStaId);
    aclrtFree(numAccTok);
    aclrtFree(attnOut);
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
