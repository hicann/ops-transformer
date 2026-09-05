# aclnnGenericBlockSparseAttentionGradMetadata

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

- 接口功能：aclnnGenericBlockSparseAttentionGradMetadata根据rsvdBlockIdx、rsvdBlockCount、seqlen等信息进行稀疏attention的分核与负载均衡，为aclnnGenericBlockSparseAttentionGrad的前置AICPU算子。按B → N2 → J → G顺序展开`(b, n2, j, g)`任务列表，并在AIC核间做`[baseM, baseN] = [128, 128]`基本块粒度的贪心负载均衡，输出metadata供主Grad算子消费。
- 该算子不建议单独使用，建议与aclnnGenericBlockSparseAttentionGrad配合使用，形成完整工作流。

$$
\text{metaSize} = 80 + B \times N1 \times J \times 4
$$

其中`J = ceilDiv(maxKvSeqlen, blockShapeY)`。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGenericBlockSparseAttentionGradMetadata”接口执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *rsvdBlockIdx,
    const aclTensor *rsvdBlockCount,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen,
    int64_t maxKvSeqlen,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headDim,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize

- **参数说明**

  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
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
        <td>rsvdBlockIdx</td>
        <td>输入</td>
        <td>稀疏块索引数组，指定每个KV块选择的Q块/token索引。</td>
        <td>同group每个KVHead对应的Q稀疏pattern一致（isPackedGQA=1）。不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>TND：[B, N2, ceilDiv(maxS2, blockShapeY), maxS1]；BNSD/BSND：[B, N2, ceilDiv(S2, blockShapeY), S1]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>rsvdBlockCount</td>
        <td>输入</td>
        <td>指定每个KV块实际选择的Q数量。</td>
        <td>不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B, N2, ceilDiv(maxS2, blockShapeY)]或[B, N2, ceilDiv(S2, blockShapeY)]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cuSeqLengthsQOptional</td>
        <td>可选输入</td>
        <td>每个Batch对应的query序列长度前缀和。</td>
        <td>layoutQ为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>[B+1]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cuSeqLengthsKvOptional</td>
        <td>可选输入</td>
        <td>每个Batch对应的key/value序列长度前缀和。</td>
        <td>layoutKv为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>[B+1]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sequsedQOptional</td>
        <td>可选输入</td>
        <td>各batch中query的实际序列长度。</td>
        <td>长度为B。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sequsedKvOptional</td>
        <td>可选输入</td>
        <td>各batch中kv的实际序列长度。</td>
        <td>长度为B。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxQSeqlen</td>
        <td>输入</td>
        <td>所有batch中qSeqlen的最大值。</td>
        <td>须≥0。BNSD/BSND场景须与query的S维一致。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maxKvSeqlen</td>
        <td>输入</td>
        <td>所有batch中kvSeqlen的最大值。</td>
        <td>须≥0。用于计算J = ceilDiv(maxKvSeqlen, blockShapeY)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numQHeads</td>
        <td>输入</td>
        <td>query的head数（N1）。</td>
        <td>须落在[1, 128]，且能被numKvHeads整除。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKvHeads</td>
        <td>输入</td>
        <td>key/value的head数（N2）。</td>
        <td>须落在[1, 128]，且与rsvdBlockIdx的N2维一致。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headDim</td>
        <td>输入</td>
        <td>query/key/value的embed维度。</td>
        <td>当前固定128。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockShape</td>
        <td>输入</td>
        <td>稀疏块形状数组。</td>
        <td>含两个元素[blockShapeX, blockShapeY]。blockShapeX当前仅支持1；blockShapeY当前仅支持128。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isPackedGQA</td>
        <td>输入</td>
        <td>同一group内的qHead是否共享同样的稀疏pattern。</td>
        <td>当前仅支持1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>layoutQ</td>
        <td>输入</td>
        <td>输入query的数据排布格式。</td>
        <td>当前支持"TND"、"BNSD"、"BSND"。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>layoutKv</td>
        <td>输入</td>
        <td>输入key、value的数据排布格式。</td>
        <td>当前支持"TND"、"BNSD"、"BSND"，须与layoutQ一致。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maskType</td>
        <td>输入</td>
        <td>attention计算中的掩码类型。</td>
        <td>当前仅支持1（RIGHT_DOWN_CAUSAL）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxPrecision</td>
        <td>输入</td>
        <td>Softmax计算采取的精度级别。</td>
        <td>仅支持0或1，当前实现传0。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>winLeft</td>
        <td>输入</td>
        <td>滑窗向前包含token数。</td>
        <td>不使能时必须为-1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>winRight</td>
        <td>输入</td>
        <td>滑窗向后包含token数。</td>
        <td>不使能时必须为-1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>metadata</td>
        <td>输出</td>
        <td>稀疏attention的分核信息。</td>
        <td>不支持空Tensor。metaSize ≥ 80 + B×N1×J×4。</td>
        <td>INT64</td>
        <td>ND</td>
        <td>[metaSize]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1100px">
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 720px">
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
          <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
          <td>561101</td>
          <td>创建aclOpExecutor失败。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_NULLPTR</td>
          <td>561103</td>
          <td>workspaceSize或executor为空指针。</td>
        </tr>
        <tr>
          <td class="merged-cell" rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="3">161002</td>
          <td>rsvdBlockIdx/rsvdBlockCount/metadata为空或shape/dtype非法。</td>
        </tr>
        <tr>
          <td>layout为TND时未提供cuSeqLengths；headDim、blockShape、isPackedGQA、window参数不在支持范围。</td>
        </tr>
        <tr>
          <td>rsvdBlockIdx的J维与ceilDiv(maxKvSeqlen, blockShapeY)不一致；metadata长度不足；B×N1×J超过任务数上限（1M）。</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnGenericBlockSparseAttentionGradMetadata

- **参数说明**

  <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - 本算子输出为确定性结果。
- 须与[aclnnGenericBlockSparseAttentionGrad](../../generic_block_sparse_attention_grad/docs/aclnnGenericBlockSparseAttentionGrad.md)配合使用；主算子调用前必须先成功执行本算子。
- HeadDim固定为128；numQHeads/numKvHeads须落在[1, 128]，且numQHeads % numKvHeads == 0。
- blockShape当前仅支持[1, 128]；isPackedGQA当前仅支持1；maskType当前仅支持1。
- layoutQ与layoutKv须相同，取值"TND"/"BNSD"/"BSND"；TND布局下cuSeqLengthsQOptional/cuSeqLengthsKvOptional必选。
- winLeft和winRight不使能时必须为-1。
- rsvdBlockIdx最后一维maxS1须≥maxQSeqlen；J = ceilDiv(maxKvSeqlen, blockShapeY)须与rsvdBlockIdx第3维一致。
- metadata长度须满足shape[0] ≥ 80 + B × numQHeads × J × 4；任务数上界B × numQHeads × J ≤ 1048576。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (auto v : shape) {
        n *= v;
    }
    return n;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t B = 1, N1 = 1, N2 = 1, S1 = 128, S2 = 128, D = 128;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<int32_t> idxHost(static_cast<size_t>(GetShapeSize(idxShape)), -1);
    std::vector<int32_t> cntHost(static_cast<size_t>(GetShapeSize(cntShape)), 0);
    std::vector<int64_t> metaHost(static_cast<size_t>(metaSize), 0);
    for (int64_t q = 0; q < S1; ++q) {
        idxHost[static_cast<size_t>(q)] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *idxAddr = nullptr, *cntAddr = nullptr, *metaAddr = nullptr;
    aclTensor *idx = nullptr, *cnt = nullptr, *metadata = nullptr;
    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT64, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);

    const int64_t blockShapeData[] = {1, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char layout[] = "BNSD";

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, layout, layout, 1, 0, -1, -1,
        metadata, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);

    void *ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(ws, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);

    LOG_PRINT("GenericBlockSparseAttentionGradMetadata finished successfully.\n");

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    if (wsSize > 0) {
        aclrtFree(ws);
    }
    Finalize(deviceId, stream);
    return 0;
}
```
