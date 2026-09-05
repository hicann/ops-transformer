# aclnnBlockAttentionResidualsGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/block_attention_residuals_grad)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：`BlockAttentionResidualsGrad` 是正向算子 `BlockAttentionResiduals`（注意力残差）的反向传播算子，该接口根据前向保存的invNorm、probs以及正向参数projWeight、normWeight，结合gradHiddenStates，计算partialBlock、blockRes、projWeight和normWeight的梯度。validBlockNum是预留属性，当前版本不参与计算，取值不会影响输出。

- 计算公式：

  $$
  g_{t,i} = \sum_{h=0}^{H-1} grad\_output_{t,h} \cdot v_{t,i,h}
  $$

  $$
  grad\_score_{t,i} = probs_{t,i} \cdot \left(g_{t,i} - \sum_{j=0}^{N} probs_{t,j} \cdot g_{t,j}\right)
  $$

  $$
  grad\_k_{t,i,h} = grad\_score_{t,i} \cdot score\_weight_{h}
  $$

  $$
  grad\_score\_weight_{h} = \sum_{t=0}^{T-1} \sum_{i=0}^{N} grad\_score_{t,i} \cdot k_{t,i,h}
  $$

  $$
  grad\_inv\_rms_{t,i} = \sum_{h=0}^{H-1} grad\_k_{t,i,h} \cdot v_{t,i,h}
  $$

  $$
  grad\_v_{t,i,h} = grad\_output_{t,h} \cdot probs_{t,i} + grad\_k_{t,i,h} \cdot inv\_rms_{t,i} - \frac{grad\_inv\_rms_{t,i} \cdot inv\_rms_{t,i}^{3}}{H} \cdot v_{t,i,h}
  $$

  其中：

  $$
  grad\_block\_res_{t,i,h} = grad\_v_{t,i,h}, \quad i < N
  $$

  $$
  grad\_partial\_block_{t,h} = grad\_v_{t,N,h}
  $$

  $$
  grad\_norm\_weight_{h} = grad\_score\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  grad\_proj\_weight_{0,h} = grad\_score\_weight_{h} \cdot norm\_weight_{h}
  $$

  当前版本直接使用前向保存的probs进行反向计算，不根据validBlockNum重新构造掩码。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttentionResidualsGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用“aclnnBlockAttentionResidualsGrad”接口执行计算。

```c++
aclnnStatus aclnnBlockAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *partialBlock,
    const aclTensor *blockRes,
    const aclTensor *projWeight,
    const aclTensor *normWeight,
    const aclTensor *gradHiddenStates,
    const aclTensor *invNorm,
    const aclTensor *probs,
    int64_t          validBlockNum,
    const aclTensor *gradPartialBlock,
    const aclTensor *gradBlockRes,
    const aclTensor *gradProjWeight,
    const aclTensor *gradNormWeight,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor);
```
```c++
aclnnStatus aclnnBlockAttentionResidualsGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnBlockAttentionResidualsGradGetWorkspaceSize

- **参数说明**
  <table style="table-layout: fixed; width: 1500px">
          <colgroup>
              <col style="width: 220px">
              <col style="width: 120px">
              <col style="width: 300px">
              <col style="width: 350px">
              <col style="width: 210px">
              <col style="width: 100px">
              <col style="width: 100px">
              <col style="width: 100px">
          </colgroup>
          <thead>
          <tr>
              <th>参数名</th>
              <th>输入/输出</th>
              <th>描述</th>
              <th>使用说明</th>
              <th>数据类型</th>
              <th>数据格式</th>
              <th>维度(shape)</th>
              <th>非连续Tensor</th>
          </tr></thead>
          <tbody>
          <tr>
              <td>partialBlock（aclTensor*）</td>
              <td>输入</td>
              <td>前向输入前缀和，对应公式中第N+1个value。</td>
              <td><li>不支持空Tensor。</li><li>数据类型与其余输入保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(T,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>blockRes（aclTensor*）</td>
              <td>输入</td>
              <td>前向输入分块残差，对应公式中前N个value。</td>
              <td><li>不支持空Tensor。</li><li>数据类型与partialBlock保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(T,N,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>projWeight（aclTensor*）</td>
              <td>输入</td>
              <td>前向线性投影权重，与normWeight共同构成score_weight。</td>
              <td><li>不支持空Tensor。</li><li>数据类型与partialBlock保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(1,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>normWeight（aclTensor*）</td>
              <td>输入</td>
              <td>前向归一化权重，与projWeight共同构成score_weight。</td>
              <td><li>不支持空Tensor。</li><li>数据类型与partialBlock保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>validBlockNum（int64_t）</td>
              <td>属性</td>
              <td>预留属性，当前版本不参与计算。</td>
              <td><li>建议传入默认值0。</li><li>其他取值当前也不会改变输出。</li></td>
              <td>INT64</td>
              <td>-</td>
              <td>标量</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradHiddenStates（aclTensor*）</td>
              <td>输入</td>
              <td>前向输出out的梯度。</td>
              <td><li>不支持空Tensor。</li><li>数据类型与partialBlock保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(T,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>invNorm（aclTensor*）</td>
              <td>输入</td>
              <td>前向保存的逐行归一化系数。</td>
              <td><li>不支持空Tensor。</li><li>仅支持FLOAT32。</li></td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>(T,N+1)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>probs（aclTensor*）</td>
              <td>输入</td>
              <td>前向softmax输出概率。</td>
              <td><li>不支持空Tensor。</li><li>仅支持FLOAT32。</li></td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>(T,N+1)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>gradPartialBlock（aclTensor*）</td>
              <td>输出</td>
              <td>partialBlock的梯度。</td>
              <td><li>不支持空Tensor。</li><li>数据类型和shape与partialBlock保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(T,H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradBlockRes（aclTensor*）</td>
              <td>输出</td>
              <td>blockRes的梯度。</td>
              <td><li>不支持空Tensor。</li><li>数据类型和shape与blockRes保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(T,N,H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradProjWeight（aclTensor*）</td>
              <td>输出</td>
              <td>projWeight的梯度。</td>
              <td><li>不支持空Tensor。</li><li>数据类型和shape与projWeight保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(1,H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradNormWeight（aclTensor*）</td>
              <td>输出</td>
              <td>normWeight的梯度。</td>
              <td><li>不支持空Tensor。</li><li>数据类型和shape与normWeight保持一致。</li></td>
              <td>FLOAT16、BFLOAT16、FLOAT32</td>
              <td>ND</td>
              <td>(H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>workspaceSize（uint64_t*）</td>
              <td>输出</td>
              <td>返回需要在Device侧申请的workspace大小</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <tr>
              <td>executor（aclOpExecutor**）</td>
              <td>输出</td>
              <td>返回op执行器，包含算子计算流程。</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          </tbody>
      </table>

- **返回值**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
      </colgroup>
          <thead>
              <th>返回值</th>
              <th>错误码</th>
              <th>描述</th>
          </thead>
          <tbody>
              <tr>
                  <td>ACLNN_ERR_PARAM_NULLPTR</td>
                  <td>161001</td>
                  <td>partialBlock、blockRes、projWeight、normWeight、gradHiddenStates、invNorm、probs及输出张量存在空指针。</td>
              </tr>
              <tr>
                  <td>ACLNN_ERR_PARAM_INVALID</td>
                  <td>161002</td>
                  <td>输入张量的数据类型、数据格式或shape不在支持的范围内。</td>
              </tr>
              <tr>
                  <td>ACLNN_ERR_RUNTIME_ERROR</td>
                  <td>361001</td>
                  <td>API内存调用npu runtime的接口异常。</td>
              </tr>
          </tbody>
      </table>

## aclnnBlockAttentionResidualsGrad

- **参数说明**
  <table style="table-layout: fixed; width: 1000px"><colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 700px">
      </colgroup>
      <thead>
          <tr>
          <th>参数名</th>
          <th>输入/输出</th>
          <th>描述</th>
          </tr></thead>
      <tbody>
          <tr>
          <td>workspace</td>
          <td>输入</td>
          <td>在Device侧申请的workspace内存地址。</td>
          </tr>
          <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttentionResidualsGradGetWorkspaceSize获取。</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>输入</td>
          <td>op执行器，包含了算子计算流程。</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>输入</td>
          <td>指定执行任务的Stream。</td>
          </tr>
      </tbody>
      </table>

- **返回值**
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnBlockAttentionResidualsGrad默认确定性实现。
- partialBlock、blockRes、projWeight、normWeight、gradHiddenStates及其对应输出的数据类型保持一致，支持FLOAT16、BFLOAT16、FLOAT32。
- validBlockNum为INT64预留属性，当前版本不参与shape推导、tiling或Kernel计算，建议传入0。
- partialBlock、blockRes、projWeight、normWeight、gradHiddenStates、invNorm和probs均只支持ND格式。
- shape需满足：partialBlock为(T,H)，blockRes为(T,N,H)，projWeight为(1,H)，normWeight为(H)，gradHiddenStates为(T,H)，invNorm为(T,N+1)，probs为(T,N+1)，其中B、N、H均为正整数，且各张量中的B、H以及invNorm和probs的N+1保持一致。
- 输入张量支持非连续Tensor，接口内部统一转为Contiguous后计算。
- probs应为前向softmax输出，invNorm应为前向逐行归一化系数，且invNorm和probs仅支持FLOAT32。当前实现完全以保存的probs和invNorm为准。
- validBlockNum取不同值时，当前版本的计算结果保持不变。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t T  = 2;
    const int64_t N  = 4;
    const int64_t H  = 64;
    const int64_t N1 = N + 1;

    std::vector<int64_t> partialBlockShape      = {T, H};
    std::vector<int64_t> blockResShape  = {T, N, H};
    std::vector<int64_t> projWeightShape     = {1, H};
    std::vector<int64_t> normWeightShape     = {H};
    std::vector<int64_t> gradHiddenStatesShape     = {T, H};
    std::vector<int64_t> invNormShape         = {T, N1};
    std::vector<int64_t> probsShape          = {T, N1};

    std::vector<uint16_t> partialBlockData     (GetShapeSize(partialBlockShape),     0x3C00);
    std::vector<uint16_t> blockResData (GetShapeSize(blockResShape),  0x3C00);
    std::vector<uint16_t> projWeightData    (GetShapeSize(projWeightShape),     0x3C00);
    std::vector<uint16_t> normWeightData    (GetShapeSize(normWeightShape),     0x3C00);
    std::vector<uint16_t> gradHiddenStatesData    (GetShapeSize(gradHiddenStatesShape),     0x3C00);
    std::vector<float> invNormData           (GetShapeSize(invNormShape),         1.0f);
    std::vector<float> probsData            (GetShapeSize(probsShape),          1.0f);
    int64_t validBlockNum = 0; // 预留属性，当前版本不参与计算

    aclTensor *partialBlock = nullptr, *blockRes = nullptr, *projWeight = nullptr, *normWeight = nullptr;
    aclTensor *gradHiddenStates = nullptr, *invNorm = nullptr, *probs = nullptr;
    void *d_partialBlock = nullptr, *d_blockRes = nullptr, *d_projWeight = nullptr, *d_normWeight = nullptr;
    void *d_gradHiddenStates = nullptr, *d_invNorm = nullptr, *d_probs = nullptr;

    ret = CreateAclTensor(partialBlockData, partialBlockShape, &d_partialBlock, aclDataType::ACL_FLOAT16, &partialBlock); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockResData, blockResShape, &d_blockRes, aclDataType::ACL_FLOAT16, &blockRes); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projWeightData, projWeightShape, &d_projWeight, aclDataType::ACL_FLOAT16, &projWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightData, normWeightShape, &d_normWeight, aclDataType::ACL_FLOAT16, &normWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHiddenStatesData, gradHiddenStatesShape, &d_gradHiddenStates, aclDataType::ACL_FLOAT16, &gradHiddenStates); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invNormData, invNormShape, &d_invNorm, aclDataType::ACL_FLOAT, &invNorm); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsData, probsShape, &d_probs, aclDataType::ACL_FLOAT, &probs); CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *gradPartialBlock = nullptr, *gradBlockRes = nullptr, *gradProjWeight = nullptr, *gradNormWeight = nullptr;
    void *d_gradPartialBlock = nullptr, *d_gradBlockRes = nullptr, *d_gradProjWeight = nullptr, *d_gradNormWeight = nullptr;

    std::vector<uint16_t> gradPartialBlockData    (GetShapeSize(partialBlockShape),     0);
    std::vector<uint16_t> gradBlockResData(GetShapeSize(blockResShape), 0);
    std::vector<uint16_t> gradProjWeightData   (GetShapeSize(projWeightShape),    0);
    std::vector<uint16_t> gradNormWeightData   (GetShapeSize(normWeightShape),    0);

    ret = CreateAclTensor(gradPartialBlockData, partialBlockShape, &d_gradPartialBlock, aclDataType::ACL_FLOAT16, &gradPartialBlock); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBlockResData, blockResShape, &d_gradBlockRes, aclDataType::ACL_FLOAT16, &gradBlockRes); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradProjWeightData, projWeightShape, &d_gradProjWeight, aclDataType::ACL_FLOAT16, &gradProjWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormWeightData, normWeightShape, &d_gradNormWeight, aclDataType::ACL_FLOAT16, &gradNormWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnBlockAttentionResidualsGradGetWorkspaceSize(
        partialBlock, blockRes, projWeight, normWeight,
        gradHiddenStates, invNorm, probs, validBlockNum,
        gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnBlockAttentionResidualsGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("BlockAttentionResidualsGrad example ran successfully!\n");

    aclDestroyTensor(partialBlock); aclDestroyTensor(blockRes); aclDestroyTensor(projWeight);
    aclDestroyTensor(normWeight); aclDestroyTensor(gradHiddenStates); aclDestroyTensor(invNorm);
    aclDestroyTensor(probs);
    aclrtFree(d_partialBlock); aclrtFree(d_blockRes); aclrtFree(d_projWeight);
    aclrtFree(d_normWeight); aclrtFree(d_gradHiddenStates); aclrtFree(d_invNorm);
    aclrtFree(d_probs);

    aclDestroyTensor(gradPartialBlock); aclDestroyTensor(gradBlockRes); aclDestroyTensor(gradProjWeight);
    aclDestroyTensor(gradNormWeight);
    aclrtFree(d_gradPartialBlock); aclrtFree(d_gradBlockRes); aclrtFree(d_gradProjWeight);
    aclrtFree(d_gradNormWeight);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
