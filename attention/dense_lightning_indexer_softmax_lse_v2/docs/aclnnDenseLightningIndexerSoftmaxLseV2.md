# aclnnDenseLightningIndexerSoftmaxLseV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：DenseLightningIndexerSoftmaxLseV2 算子是 DenseLightningIndexerGradKlLoss 算子计算 Softmax 输入的分支算子。相比 aclnnDenseLightningIndexerSoftmaxLse，新增了压缩注意力（Compressed Attention）支持，并支持通过 metadata 前置算子进行分核负载均衡。

  主要计算过程为：
  1. 对 query_index 和 key_index 计算 attention score。
  2. 根据 weight 对 attention score 进行加权求和。
  3. 对加权结果执行 Softmax 计算，输出 log-sum-exp（LSE）结果。

- 计算公式：

  $$
  \text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
  $$

  $$
  \text{lse}=\text{ReduceMax}\left(\text{res}\right)+\text{log}\left(\text{ReduceSum}\left(\text{exp}\left(\text{res}-\text{ReduceMax}\left(\text{res}\right)\right)\right)\right)
  $$

  lse 作为输出传递给算子 DenseLightningIndexerGradKlLoss 作为输入计算 Softmax 使用。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize"接口获取入参并根据计算流程计算所需 workspace 大小，再调用"aclnnDenseLightningIndexerSoftmaxLseV2"接口执行计算。

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(
    const aclTensor  *queryIndex,
    const aclTensor  *keyIndex,
    const aclTensor  *weight,
    const aclTensor  *cuSeqLensQOptional,
    const aclTensor  *cuSeqLensKOptional,
    const aclTensor  *seqUsedQOptional,
    const aclTensor  *seqUsedKOptional,
    const aclTensor  *cmpResidualKOptional,
    const aclTensor  *metadataOptional,
    const char       *layoutQ,
    const char       *layoutK,
    int64_t           maskMode,
    int64_t           cmpRatio,
    const aclTensor  *softmaxLseOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor);
```

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize

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
    </tr></thead>
    <tbody>
     <tr>
      <td>queryIndex（aclTensor*）</td>
      <td>输入</td>
      <td>lightningIndexer 结构的输入 queryIndex。</td>
      <td><ul><li>B：支持泛化且与 keyIndex 的 B 保持一致。</li><li>N1：1~128，且必须能被 N2 整除。</li><li>D：128。</li><li>T1：多个 Batch 的 S1 累加。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(B,S1,N1,D);(T1,N1,D)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>keyIndex（aclTensor*）</td>
      <td>输入</td>
      <td>lightningIndexer 结构的输入 keyIndex。</td>
      <td><ul><li>B：支持泛化且与 queryIndex 的 B 保持一致。</li><li>S2：支持泛化。</li><li>N2：1。</li><li>D：128，且与 queryIndex 的 D 保持一致。</li><li>T2：多个 Batch 的 S2 累加。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(B,S2,N2,D);(T2,N2,D)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>权重张量。</td>
      <td><ul><li>B：支持泛化且与 queryIndex 的 B 保持一致。</li><li>S1：支持泛化且与 queryIndex 的 S1 保持一致。</li><li>N1：与 queryIndex 的 N1 保持一致。</li><li>T1：多个 Batch 的 S1 累加。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,S1,N1);(T1,N1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>cuSeqLensQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>当前 Batch 及前序 Batch 中 q 的有效 token 数的累加和。</td>
      <td><ul><li>TND 场景下必传，第一个值固定为 0。</li><li>支持空 Tensor。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape 为 (B+1,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>cuSeqLensKOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>当前 Batch 及前序 Batch 中 k 的有效 token 数的累加和。</td>
      <td><ul><li>TND 场景下必传，第一个值固定为 0。</li><li>支持空 Tensor。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape 为 (B+1,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>seqUsedQOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>不同 Batch 中 q 的实际使用长度。</td>
      <td><ul><li>支持空 Tensor。当该入参存在时，各 batch 按 seqused_q 的实际值参与运算。</li><li>seqused_q 的值不大于各 Batch 的实际 seqlen_q。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape 为 (B,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>seqUsedKOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>不同 Batch 中 k 的实际使用长度。</td>
      <td><ul><li>支持空 Tensor。当该入参存在时，各 batch 按 seqused_k 的实际值参与运算。</li><li>seqused_k 的值不大于各 Batch 的实际 seqlen_k。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape 为 (B,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>cmpResidualKOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示 k 的 sequence length 与 cmpRatio 相关的残差。</td>
      <td><ul><li>支持空 Tensor。</li><li>当 maskMode=3 且 cmpRatio>1 时必须传入。</li><li>cmp_residual_k 的每个值必须小于 cmpRatio。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape 为 (B,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>metadataOptional（aclTensor*）</td>
      <td>输入</td>
      <td>前置 AICPU 算子输出的分核负载均衡信息。</td>
      <td><ul><li>必须传入。由 aclnnDenseLightningIndexerSoftmaxLseV2Metadata 算子输出，减少 tiling 阶段对 host array 的访问。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape 为 (64,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>layoutQ（char*）</td>
      <td>输入</td>
      <td>表示 query 侧的排列格式。</td>
      <td>支持 BSND、TND，传空指针时为 BSND。layoutQ 与 layoutK 必须一致。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>layoutK（char*）</td>
      <td>输入</td>
      <td>表示 key 侧的排列格式。</td>
      <td>支持 BSND、TND，传空指针时为 BSND。layoutK 与 layoutQ 必须一致。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>maskMode（int64_t）</td>
      <td>输入</td>
      <td>表示 mask 的模式。</td>
      <td><ul><li>0：No mask。</li><li>3：rightDownCausal 模式的 mask，对应以右顶点为划分的下三角场景。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>表示 key 的压缩倍数。</td>
      <td>取值范围 [1, 128]，表示无压缩。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>softmaxLseOut（aclTensor*）</td>
      <td>输出</td>
      <td>softmax 计算使用的 LSE（log-sum-exp）值。</td>
      <td><ul><li>B：支持泛化与 queryIndex 的 B 保持一致。</li><li>N2：key 的多头数，当前固定为 1。</li><li>S1：支持泛化，且与 queryIndex 的 S1 保持一致。</li><li>T1：多个 Batch 的 S1 累加。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,N2,S1);(N2,T1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     </tbody>
    </table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
                <td>参数中存在非法的nullptr。</td>
            </tr>
            <tr>
                <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="2">161002</td>
                <td>输入的数据类型不满足支持类型。</td>
            </tr>
            <tr>
                <td>queryIndex、keyIndex、weight、softmaxLseOut必选输入/输出未传。</td>
            </tr>
        </tbody>
    </table>

## aclnnDenseLightningIndexerSoftmaxLseV2

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
     </tr>
     <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize 获取。</td>
     </tr>
     <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
     </tr>
     <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream 流。</td>
     </tr>
    </tbody>
    </table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - 确定性说明：aclnnDenseLightningIndexerSoftmaxLseV2默认确定性实现。

  - 公共约束
    - 入参为空的场景处理：
        - queryIndex 为空 Tensor：直接返回。

    <table style="undefined;table-layout: fixed; width: 901px"><colgroup>
    <col style="width: 168px">
    <col style="width: 565px">
    <col style="width: 168px">
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
      <td>No mask，不做 mask 操作。</td>
      <td>支持</td>
     </tr>
     <tr>
      <td>3</td>
      <td>rightDownCausal 模式的 mask，对应以右顶点为划分的下三角场景。</td>
      <td>支持</td>
     </tr>
    </tbody>
    </table>

  - 规格约束

    <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
    <col style="width: 125px">
    <col style="width: 182px">
    <col style="width: 602px">
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
      <td>1~256</td>
      <td>-</td>
    </tr>
    <tr>
      <td>S1、S2</td>
      <td>0~128K</td>
      <td>S1、S2 支持不等长。</td>
    </tr>
    <tr>
      <td>N1</td>
      <td>1~128</td>
      <td>必须能被 N2 整除。</td>
    </tr>
    <tr>
      <td>N2</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>D</td>
      <td>128</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>BSND/TND</td>
      <td>layoutQ 与 layoutK 必须一致。</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>1~128</td>
      <td>1 表示无压缩。</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>0、3</td>
      <td>0=No mask，3=rightDownCausal。</td>
    </tr>
    </tbody>
    </table>

  - 特殊约束
    - BSND 场景下，必须传入 batch_size 和 max_seqlen_q 相关信息（通过 tensor shape 体现）。
    - TND 场景下，必须传入 cuSeqLensQ 和 cuSeqLensK。
    - 当 maskMode=3 且 cmpRatio>1 时，必须传入 cmpResidualK，且 cmpResidualK 的每个值必须小于 cmpRatio。
    - seqUsedQ 的每个值不大于各 Batch 的实际 seqlen_q。
    - seqUsedK 的每个值不大于各 Batch 的实际 seqlen_k。
    - metadataOptional 为前置算子 aclnnDenseLightningIndexerSoftmaxLseV2Metadata 的输出，必须传入以执行相关负载均衡策略。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.h"

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
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // BSND layout: q[B,S1,N1,D] k[B,S2,N2,D] w[B,S1,N1] lse[B,N2,S1]
  int64_t B = 2;
  int64_t S1 = 128;
  int64_t S2 = 256;
  int64_t N1 = 8;
  int64_t N2 = 1;
  int64_t D = 128;

  std::vector<int64_t> qIndexShape = {B, S1, N1, D};
  std::vector<int64_t> kIndexShape = {B, S2, N2, D};
  std::vector<int64_t> weightShape = {B, S1, N1};
  std::vector<int64_t> softmaxLseShape = {B, N2, S1};
  std::vector<int64_t> metadataShape = {64};

  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;

  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* softmaxLse = nullptr;
  aclTensor* metadata = nullptr;

  std::vector<aclFloat16> qIndexHostData(B * S1 * N1 * D, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> kIndexHostData(B * S2 * 1 * D, aclFloatToFloat16(0.1));
  std::vector<float> weightHostData(B * S1 * N1, 0.005f);
  std::vector<float> softmaxLseHostData(B * N2 * S1, 0.0f);
  std::vector<int32_t> metadataHostData(64, 0);

  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr,
      aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // BSND layout, maskMode=0 (defaultMask), cmpRatio=1
  constexpr const char layoutQStr[] = "BSND";
  constexpr const char layoutKStr[] = "BSND";
  constexpr size_t layoutQLen = sizeof(layoutQStr);
  constexpr size_t layoutKLen = sizeof(layoutKStr);
  char layoutQ[layoutQLen];
  char layoutK[layoutKLen];
  errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
  if (memcpyRet != 0) {
    LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
    return -1;
  }
  memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
  if (memcpyRet != 0) {
    LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
    return -1;
  }
  int64_t maskMode = 0;
  int64_t cmpRatio = 1;

  // 1. 调用 metadata 前置算子，生成分核负载均衡信息
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
      nullptr, nullptr, nullptr, nullptr, nullptr, B, S1, S2, N1, N2, D, layoutQ, layoutK, maskMode, cmpRatio,
      metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnDenseLightningIndexerSoftmaxLseV2Metadata(metadataWorkspaceAddr, metadataWorkspaceSize,
                                                        metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2Metadata failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 2. 调用主算子
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize(
            qIndex, kIndex, weight,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            metadata,
            layoutQ, layoutK, maskMode, cmpRatio,
            softmaxLse,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnDenseLightningIndexerSoftmaxLseV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseV2 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(softmaxLseShape, &softmaxLseDeviceAddr);
  LOG_PRINT("pass\n");

  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(weight);
  aclDestroyTensor(softmaxLse);
  aclDestroyTensor(metadata);

  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  aclrtFree(metadataDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
