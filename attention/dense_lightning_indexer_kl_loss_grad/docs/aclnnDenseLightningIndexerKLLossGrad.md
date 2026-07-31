# aclnnDenseLightningIndexerKLLossGrad

## 产品支持情况

<!-- npu="950" id1 -->
<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->

<!-- end id1 -->
<!-- npu="A3" id2 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->

<!-- end id2 -->
<!-- npu="910b" id3 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->

<!-- end id3 -->
<!-- npu="310b" id4 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->

<!-- end id4 -->
<!-- npu="310p" id5 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->

<!-- end id5 -->
<!-- npu="910" id6 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

<!-- end id6 -->

## 功能说明

- 接口功能：DenseLightningIndexerKLLossGrad算子是LightningIndexer KL Loss的反向计算算子（Dense模式）。该接口接收Lightning Indexer分支的q、k、w，以及由主Attention分支预先计算得到的attnSoftmaxL1Norm和softmaxLse，计算并输出dq、dk、dw和softmaxOut。本接口对全部key做dense注意力计算，不再依赖sparseIndices做top-k筛选；也不需要在kernel内部重算主Attention的`Q @ K -> softmax -> reduce/l1 norm`，不再输出loss；主Attention的目标分布由attnSoftmaxL1Norm输入提供，softmaxOut可用于后续loss计算。

- 计算公式：
    Indexer logits可表示为：

   $$
   S_{t,:}=q_{t,:}@K^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   其中，$q$和$K$分别对应本接口的q和k，$W$对应本接口的w。Indexer分支的softmax输出为：

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   本接口将$y$写出到softmaxOut。目标分布$p$由attnSoftmaxL1Norm输入提供，等价于旧版kernel内部由main attention score经head求和和L1归一化得到的结果。若后续继续计算KL Loss，其形式与旧版保持一致：

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   通过求导可得Loss的梯度表达式：

   $$
   dI_{t,:}=\operatorname{Softmax}(I_{t,:})-p_{t,:}
   $$

   利用链式法则可以进行w、q和k矩阵的梯度计算：

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K
   $$

   $$
   dK=\sum_t\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK直接累积到对应key位置上，无需scatter-add。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnDenseLightningIndexerKLLossGrad”接口执行计算。

```c++
aclnnStatus aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize(
    const aclTensor     *q,
    const aclTensor     *k,
    const aclTensor     *w,
    const aclTensor     *attnSoftmaxL1Norm,
    const aclTensor     *softmaxLse,
    const aclTensor     *cuSeqLensQOptional,
    const aclTensor     *cuSeqLensKOptional,
    const aclTensor     *seqUsedQOptional,
    const aclTensor     *seqUsedKOptional,
    const aclTensor     *cmpResidualKOptional,
    const aclTensor     *metadataOptional,
    char                *layoutQ,
    char                *layoutK,
    int64_t              maskMode,
    int64_t              cmpRatio,
    const aclTensor     *dq,
    const aclTensor     *dk,
    const aclTensor     *dw,
    const aclTensor     *softmaxOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnDenseLightningIndexerKLLossGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream stream)
```

## aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize

- **参数说明:**

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
                <th>参数名</th>
                <th>输入/输出</th>
                <th>描述</th>
                <th>使用说明</th>
                <th>数据类型</th>
                <th>数据格式</th>
                <th>维度(shape)</th>
                <th>非连续Tensor</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>q</td>
                <td>输入</td>
                <td>Lightning Indexer分支的query输入。</td>
                <td>
                    <ul>
                        <li>数据类型与k、dq、dk保持一致。</li>
                        <li>不支持空Tensor。</li>
                    </ul>
                </td>
                <td>FLOAT16、BFLOAT16</td>
                <td>ND</td>
                <td>(B,S1,N1,D)、(T1,N1,D)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>k</td>
                <td>输入</td>
                <td>Lightning Indexer分支的key输入。</td>
                <td>
                    <ul>
                        <li>数据类型与q、dq、dk保持一致。</li>
                        <li>当前仅支持N2=1。</li>
                        <li>不支持空Tensor。</li>
                    </ul>
                </td>
                <td>FLOAT16、BFLOAT16</td>
                <td>ND</td>
                <td>(B,S2,N2,D)、(T2,N2,D)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>w</td>
                <td>输入</td>
                <td>Indexer logits的head权重。</td>
                <td>不支持空Tensor。</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N1)、(T1,N1)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>attnSoftmaxL1Norm</td>
                <td>输入</td>
                <td>主Attention分支预先计算出的目标分布。</td>
                <td>
                    <ul>
                        <li>用于替代旧kernel内部的主Attention重算结果。</li>
                        <li>维度为(B,S1,N2,S2)或(T1,N2,maxSeqLenK)，覆盖全部key。</li>
                        <li>不支持空Tensor。</li>
                    </ul>
                </td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N2,S2)、(T1,N2,maxSeqLenK)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>softmaxLse</td>
                <td>输入</td>
                <td>softmax log-sum-exp值，由DenseLightningIndexerSoftmaxLse预先计算。</td>
                <td>
                    <ul>
                        <li>用于数值稳定计算。</li>
                        <li>维度为(B,N2,S1)或(N2,T1)。</li>
                        <li>不支持空Tensor。</li>
                    </ul>
                </td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,N2,S1)、(N2,T1)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>cuSeqLensQOptional</td>
                <td>可选输入</td>
                <td>TND layout下q的累积序列长度。</td>
                <td>当layoutQ为TND时需要传入，长度为B+1，首元素为0。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B+1,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>cuSeqLensKOptional</td>
                <td>可选输入</td>
                <td>TND layout下k的累积序列长度。</td>
                <td>当layoutK为TND时需要传入，长度为B+1，首元素为0。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B+1,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>seqUsedQOptional</td>
                <td>可选输入</td>
                <td>表示每个batch实际使用的q长度。</td>
                <td>长度为B。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>seqUsedKOptional</td>
                <td>可选输入</td>
                <td>表示每个batch实际使用的k长度。</td>
                <td>长度为B。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>cmpResidualKOptional</td>
                <td>可选输入</td>
                <td>maskMode=3且cmpRatio!=1时需要传入。</td>
                <td>长度为B。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>metadataOptional</td>
                <td>可选输入</td>
                <td>由aclnnDenseLightningIndexerKLLossGradMetadata生成的分核信息。</td>
                <td>传入后kernel将使用metadata中的分核切分，必须传入。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(x,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>layoutQ</td>
                <td>输入</td>
                <td>q、w、attnSoftmaxL1Norm、softmaxLse、dq、dw、softmaxOut的layout。</td>
                <td>仅支持BSND和TND。</td>
                <td>STRING</td>
                <td>-</td>
                <td>-</td>
                <td>x</td>
            </tr>
            <tr>
                <td>layoutK</td>
                <td>输入</td>
                <td>k和dk的layout。</td>
                <td>仅支持BSND和TND。</td>
                <td>STRING</td>
                <td>-</td>
                <td>-</td>
                <td>x</td>
            </tr>
            <tr>
                <td>maskMode</td>
                <td>输入</td>
                <td>mask模式。</td>
                <td>支持的mask模式详见<a href="#约束说明">约束说明</a>。</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>x</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>输入</td>
                <td>压缩比。</td>
                <td>取值范围为1~128。</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>x</td>
            </tr>
            <tr>
                <td>dq</td>
                <td>输出</td>
                <td>q的梯度。</td>
                <td>数据类型和shape与q一致。</td>
                <td>FLOAT16、BFLOAT16</td>
                <td>ND</td>
                <td>(B,S1,N1,D)、(T1,N1,D)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>dk</td>
                <td>输出</td>
                <td>k的梯度。</td>
                <td>数据类型和shape与k一致。</td>
                <td>FLOAT16、BFLOAT16</td>
                <td>ND</td>
                <td>(B,S2,N2,D)、(T2,N2,D)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>dw</td>
                <td>输出</td>
                <td>w的梯度。</td>
                <td>数据类型和shape与w一致。</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N1)、(T1,N1)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>softmaxOut</td>
                <td>输出</td>
                <td>Indexer分支softmax输出。</td>
                <td>shape与attnSoftmaxL1Norm一致。</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N2,S2)、(T1,N2,S2)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>workspaceSize</td>
                <td>输出</td>
                <td>返回用户需要在Device侧申请的workspace大小。</td>
                <td>-</td>
                <td>UINT64</td>
                <td>-</td>
                <td>-</td>
                <td>x</td>
            </tr>
            <tr>
                <td>executor</td>
                <td>输出</td>
                <td>op执行器，包含算子计算流程。</td>
                <td>-</td>
                <td>aclOpExecutor*</td>
                <td>-</td>
                <td>-</td>
                <td>x</td>
            </tr>
        </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px">
        <colgroup>
            <col style="width: 319px">
            <col style="width: 144px">
            <col style="width: 671px">
        </colgroup>
        <thead>
            <tr>
                <th>返回值</th>
                <th>错误码</th>
                <th>描述</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>必选参数或输出为空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入、输出、属性的数据类型、数据格式或取值不在支持范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnDenseLightningIndexerKLLossGrad

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px">
        <colgroup>
            <col style="width: 144px">
            <col style="width: 125px">
            <col style="width: 700px">
        </colgroup>
        <thead>
            <tr>
                <th>参数名</th>
                <th>输入/输出</th>
                <th>描述</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>workspace</td>
                <td>输入</td>
                <td>在Device侧申请的workspace内存地址。</td>
            </tr>
            <tr>
                <td>workspaceSize</td>
                <td>输入</td>
                <td>在Device侧申请的workspace大小，由第一段接口aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize获取。</td>
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

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：aclnnDenseLightningIndexerKLLossGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

    <!-- end id7 -->

- 公共约束：
  - 参数q、k、dq、dk的数据类型应保持一致，支持FLOAT16和BFLOAT16。
  - 参数w、dw、attnSoftmaxL1Norm、softmaxLse、softmaxOut的数据类型应为FLOAT32。
  - 参数cuSeqLensQOptional、cuSeqLensKOptional、seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional、metadataOptional的数据类型应为INT32。
  - layoutQ和layoutK当前支持BSND和TND。
  - 当layoutQ为TND时，需要传入cuSeqLensQOptional；当layoutK为TND时，需要传入cuSeqLensKOptional。
  - attnSoftmaxL1Norm的维度为(B,S1,N2,S2)或(T1,N2,maxSeqLenK)，覆盖全部key的目标分布。
  - softmaxLse由DenseLightningIndexerSoftmaxLse算子预先计算提供，维度为(B,N2,S1)或(N2,T1)，用于数值稳定计算。
  - metadata输入必须传入，用于计算分核信息。

    <table style="undefined;table-layout: fixed; width: 942px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 740px">
            <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>defaultMask模式，不做causal mask。</td>
                <td>支持</td>
            </tr>
            <tr>
                <td>1</td>
                <td>allMask，必须传入完整的attenmask矩阵。</td>
                <td>不支持</td>
            </tr>
            <tr>
                <td>2</td>
                <td>leftUpCausal模式的mask。</td>
                <td>不支持</td>
            </tr>
            <tr>
                <td>3</td>
                <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
                <td>支持</td>
            </tr>
            <tr>
                <td>4</td>
                <td>band模式的mask。</td>
                <td>不支持</td>
            </tr>
            <tr>
                <td>5</td>
                <td>prefix模式。</td>
                <td>不支持</td>
            </tr>
            <tr>
                <td>6</td>
                <td>global模式。</td>
                <td>不支持</td>
            </tr>
            <tr>
                <td>7</td>
                <td>dilated模式。</td>
                <td>不支持</td>
            </tr>
            <tr>
                <td>8</td>
                <td>block_local模式。</td>
                <td>不支持</td>
            </tr>
        </tbody>
    </table>

- 规格约束：
    <table style="undefined;table-layout: fixed; width: 942px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 300px">
            <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>B</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>S1、S2</td>
                <td>-</td>
                <td>S1、S2支持不等长</td>
            </tr>
            <tr>
                <td>N1</td>
                <td>1~128</td>
                <td>MQA场景</td>
            </tr>
            <tr>
                <td>N2</td>
                <td>1</td>
                <td>当前仅支持N2=1。</td>
            </tr>
            <tr>
                <td>D</td>
                <td>128</td>
                <td>-</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>1~128</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad_metadata.h"

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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < std::min<int64_t>(size, 5); i++) {
    LOG_PRINT("softmaxOut[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
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

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  const int64_t batchSize = 1;
  const int64_t t1 = 16;
  const int64_t t2 = 4;
  const int64_t numHeadsQ = 8;
  const int64_t numHeadsK = 1;
  const int64_t headDim = 128;
  const int64_t maskMode = 3;
  const int64_t cmpRatio = 4;

  std::vector<int64_t> qShape = {t1, numHeadsQ, headDim};
  std::vector<int64_t> kShape = {t2, numHeadsK, headDim};
  std::vector<int64_t> wShape = {t1, numHeadsQ};
  std::vector<int64_t> l1NormShape = {t1, numHeadsK, t2};
  std::vector<int64_t> softmaxLseShape = {numHeadsK, t1};
  std::vector<int64_t> softmaxShape = {t1, numHeadsK, t2};
  std::vector<int64_t> cuSeqLensShape = {batchSize + 1};
  std::vector<int64_t> metadataShape = {64};

  std::vector<uint16_t> qHostData(GetShapeSize(qShape), 0x3C00);
  std::vector<uint16_t> kHostData(GetShapeSize(kShape), 0x3C00);
  std::vector<float> wHostData(GetShapeSize(wShape), 1.0f);
  std::vector<float> attnSoftmaxL1NormHostData(GetShapeSize(l1NormShape), 0.0f);
  for (int64_t t = 0; t < t1; ++t) {
    auto validK = std::min<int64_t>(std::max<int64_t>(1, (t + 1) / cmpRatio), t2);
    for (int64_t kIdx = 0; kIdx < validK; ++kIdx) {
      attnSoftmaxL1NormHostData[t * numHeadsK * t2 + kIdx] = 1.0f / static_cast<float>(validK);
    }
  }
  std::vector<float> softmaxLseHostData(GetShapeSize(softmaxLseShape), 0.0f);
  std::vector<int32_t> cuSeqLensQHostData = {0, static_cast<int32_t>(t1)};
  std::vector<int32_t> cuSeqLensKHostData = {0, static_cast<int32_t>(t2)};
  std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);
  std::vector<uint16_t> dqHostData(GetShapeSize(qShape), 0);
  std::vector<uint16_t> dkHostData(GetShapeSize(kShape), 0);
  std::vector<float> dwHostData(GetShapeSize(wShape), 0.0f);
  std::vector<float> softmaxOutHostData(GetShapeSize(softmaxShape), 0.0f);

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* wDeviceAddr = nullptr;
  void* attnSoftmaxL1NormDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensKDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dwDeviceAddr = nullptr;
  void* softmaxOutDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* w = nullptr;
  aclTensor* attnSoftmaxL1Norm = nullptr;
  aclTensor* softmaxLse = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensK = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dw = nullptr;
  aclTensor* softmaxOut = nullptr;

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT, &w);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnSoftmaxL1NormHostData, l1NormShape, &attnSoftmaxL1NormDeviceAddr,
                        aclDataType::ACL_FLOAT, &attnSoftmaxL1Norm);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr,
                        aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqLensKHostData, cuSeqLensShape, &cuSeqLensKDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dwHostData, wShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxOutHostData, softmaxShape, &softmaxOutDeviceAddr, aclDataType::ACL_FLOAT, &softmaxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[4] = {'T', 'N', 'D', 0};
  char layoutK[4] = {'T', 'N', 'D', 0};

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, batchSize, t1, t2, numHeadsQ, numHeadsK, headDim,
      layoutQ, layoutK, maskMode, cmpRatio, metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnDenseLightningIndexerKLLossGradMetadata(metadataWorkspaceAddr, metadataWorkspaceSize,
                                                      metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGradMetadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize(
      q, k, w, attnSoftmaxL1Norm, softmaxLse, cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, metadata,
      layoutQ, layoutK, maskMode, cmpRatio, dq, dk, dw, softmaxOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnDenseLightningIndexerKLLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGrad failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  PrintOutResult(softmaxShape, &softmaxOutDeviceAddr);

  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(w);
  aclDestroyTensor(attnSoftmaxL1Norm);
  aclDestroyTensor(softmaxLse);
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensK);
  aclDestroyTensor(metadata);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dw);
  aclDestroyTensor(softmaxOut);

  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(wDeviceAddr);
  aclrtFree(attnSoftmaxL1NormDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  aclrtFree(cuSeqLensQDeviceAddr);
  aclrtFree(cuSeqLensKDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dwDeviceAddr);
  aclrtFree(softmaxOutDeviceAddr);
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
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
