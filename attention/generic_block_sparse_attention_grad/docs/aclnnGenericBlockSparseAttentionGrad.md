# aclnnGenericBlockSparseAttentionGrad

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

- 接口功能：aclnnGenericBlockSparseAttentionGrad是通用块稀疏注意力的反向计算算子。依据`rsvdBlockIdx`/`rsvdBlockCount`（稀疏块索引表）定义的索引，仅在被选中的KV块上计算和传播梯度，支持动态、可变长的分块稀疏模式。调用前须先通过`aclnnGenericBlockSparseAttentionGradMetadata`生成分核`metadata`。
- 计算公式：

$$
P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
$$

$$
dP = dO @ selectedV^{T}
$$

$$
dS = P \odot (dP - SoftmaxGrad(dO, O))
$$

$$
dQ = dS @ selectedK \cdot scale
$$

$$
dK = dS^{T} @ Q \cdot scale
$$

$$
dV = P^{T} @ dO
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGenericBlockSparseAttentionGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGenericBlockSparseAttentionGrad”接口执行计算。

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *dout,
    const aclTensor *out,
    const aclTensor *lse,
    const aclTensor *rsvdBlockIdx,
    const aclTensor *rsvdBlockCount,
    const aclTensor *metadata,
    const aclTensor *attenMaskOptional,
    const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGqa,
    char *qInputLayout,
    char *kvInputLayout,
    double scaleValue,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t windowSizeLeft,
    int64_t windowSizeRight,
    aclTensor *dQuery,
    aclTensor *dKey,
    aclTensor *dValue,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
```

## aclnnGenericBlockSparseAttentionGradGetWorkspaceSize

- **参数说明**

  <table style="undefined; table-layout: fixed; width: 1567px">
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
                  <td>query</td>
                  <td>输入</td>
                  <td>attention结构的输入Q。</td>
                  <td>
                      <ul>
                          <li>数据类型与key、value、dout、out、dQuery、dKey、dValue保持一致。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>key</td>
                  <td>输入</td>
                  <td>attention结构的输入K。</td>
                  <td>
                      <ul>
                          <li>数据类型与query保持一致。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D)、(T2,N2,D)、(B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>value</td>
                  <td>输入</td>
                  <td>attention结构的输入V。</td>
                  <td>
                      <ul>
                          <li>Shape与key相同。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D)、(T2,N2,D)、(B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>dout</td>
                  <td>输入</td>
                  <td>注意力输出矩阵的梯度。</td>
                  <td>
                      <ul>
                          <li>B、S1、N1与query保持一致；D为128。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>out</td>
                  <td>输入</td>
                  <td>注意力输出矩阵。</td>
                  <td>
                      <ul>
                          <li>Shape与dout保持一致。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>lse</td>
                  <td>输入</td>
                  <td>注意力正向计算的输出lse，layout与正向输出保持一致。</td>
                  <td>
                      <ul>
                          <li>数据类型为FLOAT32。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>FLOAT32</td>
                  <td>ND</td>
                  <td>TND：(T1,N1,1)；BNSD：(B,N1,S1,1)；BSND：(B,S1,N1,1)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>rsvdBlockIdx</td>
                  <td>输入</td>
                  <td>稀疏块索引数组，指定每个KV块选择的Q块/token索引。</td>
                  <td>
                      <ul>
                          <li>同group每个KVHead对应的Q稀疏pattern一致（isPackedGqa=1）。</li>
                          <li>第4维maxS1应≥rsvdBlockCount中所有元素的最大值。</li>
                          <li>不支持空Tensor。</li>
                      </ul>
                  </td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>TND：(B,N2,ceilDiv(maxS2,blockShapeY),maxS1)；BNSD/BSND：(B,N2,ceilDiv(S2,blockShapeY),S1)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>rsvdBlockCount</td>
                  <td>输入</td>
                  <td>指定每个KV块实际选择的Q数量。</td>
                  <td>不支持空Tensor。</td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(B,N2,ceilDiv(maxS2,blockShapeY))或(B,N2,ceilDiv(S2,blockShapeY))</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>metadata</td>
                  <td>输入</td>
                  <td>由aclnnGenericBlockSparseAttentionGradMetadata生成的分核信息。</td>
                  <td>
                      <ul>
                          <li>必须传入。</li>
                          <li>长度≥80+B×N1×J×4（int64元素个数）。</li>
                      </ul>
                  </td>
                  <td>INT64</td>
                  <td>ND</td>
                  <td>(x,)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>attenMaskOptional</td>
                  <td>可选输入</td>
                  <td>公式中的atten_mask，与稀疏pattern叠加产生作用。</td>
                  <td>当前暂不支持，应传nullptr。</td>
                  <td>BOOL</td>
                  <td>ND</td>
                  <td>-</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>cuSeqLengthsOptional</td>
                  <td>可选输入</td>
                  <td>每个Batch对应的query序列长度前缀和。</td>
                  <td>qInputLayout为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。</td>
                  <td>INT64</td>
                  <td>ND</td>
                  <td>(B+1,)</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>cuSeqLengthsKvOptional</td>
                  <td>可选输入</td>
                  <td>每个Batch对应的key/value序列长度前缀和。</td>
                  <td>kvInputLayout为"TND"时必须配置；为"BNSD"或"BSND"时传nullptr。</td>
                  <td>INT64</td>
                  <td>ND</td>
                  <td>(B+1,)</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>sequsedQOptional</td>
                  <td>可选输入</td>
                  <td>各batch中query的实际序列长度。</td>
                  <td>长度为B。</td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(B,)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>sequsedKvOptional</td>
                  <td>可选输入</td>
                  <td>各batch中kv的实际序列长度。</td>
                  <td>长度为B。</td>
                  <td>INT32</td>
                  <td>ND</td>
                  <td>(B,)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>blockShape</td>
                  <td>输入</td>
                  <td>稀疏块形状数组。</td>
                  <td>
                      <ul>
                          <li>含两个元素[blockShapeX, blockShapeY]。</li>
                          <li>blockShapeX当前仅支持1。</li>
                          <li>blockShapeY须≥128且按64对齐；当前实现仅支持128。</li>
                      </ul>
                  </td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>isPackedGqa</td>
                  <td>输入</td>
                  <td>同一group内的qHead是否共享同样的稀疏pattern。</td>
                  <td>当前仅支持1。不同batch之间不共享。</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>qInputLayout</td>
                  <td>输入</td>
                  <td>输入query的数据排布格式。</td>
                  <td>当前支持"TND"、"BNSD"、"BSND"。</td>
                  <td>STRING</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>kvInputLayout</td>
                  <td>输入</td>
                  <td>输入key、value的数据排布格式。</td>
                  <td>当前支持"TND"、"BNSD"、"BSND"，须与qInputLayout一致。</td>
                  <td>STRING</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>scaleValue</td>
                  <td>输入</td>
                  <td>公式中的scale，代表缩放系数。</td>
                  <td>建议值：公式中d开根号的倒数。</td>
                  <td>FLOAT</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>maskType</td>
                  <td>输入</td>
                  <td>attention计算中的掩码类型。</td>
                  <td>支持的mask模式详见<a href="#约束说明">约束说明</a>。当前仅支持1。</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>softmaxPrecision</td>
                  <td>输入</td>
                  <td>Softmax计算采取的精度级别。</td>
                  <td>
                      <ul>
                          <li>仅支持0或1。</li>
                          <li>0：online softmax和rescale均使用fp32。</li>
                          <li>1：online softmax使用fp16/bf16，rescale使用fp32，可能发生数值溢出。</li>
                          <li>当前实现传0。</li>
                      </ul>
                  </td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>windowSizeLeft</td>
                  <td>输入</td>
                  <td>滑窗attention场景下，滑窗需要向前包含多少个token。</td>
                  <td>不使能时必须为-1，需要与maskType、mask配合使用。</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>windowSizeRight</td>
                  <td>输入</td>
                  <td>滑窗attention场景下，滑窗需要向后包含多少个token。</td>
                  <td>不使能时必须为-1，需要与maskType、mask配合使用。</td>
                  <td>INT64</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>dQuery</td>
                  <td>输出</td>
                  <td>query的梯度。</td>
                  <td>数据类型和shape与query一致。</td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S1,N1,D)、(T1,N1,D)、(B,N1,S1,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>dKey</td>
                  <td>输出</td>
                  <td>key的梯度。</td>
                  <td>数据类型和shape与key一致。</td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D)、(T2,N2,D)、(B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>dValue</td>
                  <td>输出</td>
                  <td>value的梯度。</td>
                  <td>数据类型和shape与value一致。</td>
                  <td>FLOAT16、BFLOAT16</td>
                  <td>ND</td>
                  <td>(B,S2,N2,D)、(T2,N2,D)、(B,N2,S2,D)</td>
                  <td>√</td>
              </tr>
              <tr>
                  <td>workspaceSize</td>
                  <td>输出</td>
                  <td>返回用户需要在Device侧申请的workspace大小。</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
                  <td>-</td>
              </tr>
              <tr>
                  <td>executor</td>
                  <td>输出</td>
                  <td>op执行器，包含算子计算流程。</td>
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
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>必选参数或输出为空指针；qInputLayout为"TND"时未提供cuSeqLengthsOptional；kvInputLayout为"TND"时未提供cuSeqLengthsKvOptional。</td>
        </tr>
        <tr>
          <td class="merged-cell" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="2">161002</td>
          <td>输入、输出、属性的数据类型、数据格式或取值不在支持范围内；qInputLayout与kvInputLayout不一致。</td>
        </tr>
        <tr>
          <td>isPackedGqa!=1；windowSizeLeft/Right!=-1；blockShape不满足约束。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>API内存调用npu runtime的接口异常。</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnGenericBlockSparseAttentionGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGenericBlockSparseAttentionGradGetWorkspaceSize获取。</td>
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
  - aclnnGenericBlockSparseAttentionGrad默认为非确定性实现，暂不支持确定性实现，确定性计算配置后不会生效。
- 须先调用[aclnnGenericBlockSparseAttentionGradMetadata](../../generic_block_sparse_attention_grad_metadata/docs/aclnnGenericBlockSparseAttentionGradMetadata.md)生成`metadata`，再调用本接口。
- 参数query、key、value、dout、out、dQuery、dKey、dValue的数据类型应保持一致，支持FLOAT16和BFLOAT16。
- 参数lse的数据类型应为FLOAT32。
- 参数rsvdBlockIdx、rsvdBlockCount、sequsedQOptional、sequsedKvOptional的数据类型应为INT32。
- 参数cuSeqLengthsOptional、cuSeqLengthsKvOptional、metadata的数据类型应为INT64。
- qInputLayout和kvInputLayout当前支持TND、BNSD、BSND，且必须保持一致。
- 当qInputLayout为TND时，需要传入cuSeqLengthsOptional；当kvInputLayout为TND时，需要传入cuSeqLengthsKvOptional。
- HeadDim固定为128；N1/N2取值范围[1, 128]，且N1 % N2 == 0。
- blockShape当前仅支持[1, 128]；isPackedGqa当前仅支持1；maskType当前仅支持1。
- windowSizeLeft和windowSizeRight不使能时必须为-1；attenMaskOptional当前应传nullptr。
- Softmax LSE的head/seq轴语义须与query布局一致。
- `rsvdBlockIdx`第4维maxS1应≥`rsvdBlockCount`中所有元素的最大值。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad.h"
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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
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

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // BNSD smoke case
    const int64_t B = 1;
    const int64_t N1 = 1;
    const int64_t N2 = 1;
    const int64_t S1 = 128;
    const int64_t S2 = 128;
    const int64_t D = 128;
    const int64_t blockX = 1;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t maskType = 1;
    const double scaleValue = 1.0 / std::sqrt(static_cast<double>(D));
    // metadata size = 80 + B * N1 * J * 4
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> qShape = {B, N1, S1, D};
    std::vector<int64_t> kvShape = {B, N2, S2, D};
    std::vector<int64_t> lseShape = {B, N1, S1};
    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0x2E66); // ~0.1 fp16
    std::vector<uint16_t> kHost(GetShapeSize(kvShape), 0x2E66);
    std::vector<uint16_t> vHost(GetShapeSize(kvShape), 0x2E66);
    std::vector<uint16_t> doutHost(GetShapeSize(qShape), 0x211E); // ~0.01 fp16
    std::vector<uint16_t> outHost(GetShapeSize(qShape), 0x2E66);
    std::vector<float> lseHost(GetShapeSize(lseShape), 5.0f);
    std::vector<int32_t> idxHost(GetShapeSize(idxShape), -1);
    std::vector<int32_t> cntHost(GetShapeSize(cntShape), 0);
    std::vector<int64_t> metaHost(metaSize, 0);
    std::vector<uint16_t> dqHost(GetShapeSize(qShape), 0);
    std::vector<uint16_t> dkHost(GetShapeSize(kvShape), 0);
    std::vector<uint16_t> dvHost(GetShapeSize(kvShape), 0);

    // 单个 KV 块选中全部 Q token
    for (int64_t q = 0; q < S1; ++q) {
        idxHost[q] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr, *lseAddr = nullptr;
    void *idxAddr = nullptr, *cntAddr = nullptr, *metaAddr = nullptr;
    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;

    aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
    aclTensor *dout = nullptr, *out = nullptr, *lse = nullptr;
    aclTensor *idx = nullptr, *cnt = nullptr, *metadata = nullptr;
    aclTensor *dq = nullptr, *dk = nullptr, *dv = nullptr;

    ret = CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHost, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHost, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &dout);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, qShape, &outAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lseHost, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT64, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dqHost, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dkHost, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dvHost, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t blockShapeData[] = {blockX, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char qLayout[] = "BNSD";
    char kvLayout[] = "BNSD";

    // 1) Metadata
    uint64_t metaWsSize = 0;
    aclOpExecutor *metaExecutor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, qLayout, kvLayout, maskType, 0,
        -1, -1, metadata, &metaWsSize, &metaExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *metaWs = nullptr;
    if (metaWsSize > 0) {
        ret = aclrtMalloc(&metaWs, metaWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(metaWs, metaWsSize, metaExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Metadata failed. ERROR: %d\n", ret); return ret);

    // 2) Grad
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
        q, k, v, dout, out, lse, idx, cnt, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, blockShape, 1,
        qLayout, kvLayout, scaleValue, maskType, 0, -1, -1, dq, dk, dv, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionGrad failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("GenericBlockSparseAttentionGrad finished successfully.\n");

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(dout);
    aclDestroyTensor(out);
    aclDestroyTensor(lse);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dv);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(vAddr);
    aclrtFree(doutAddr);
    aclrtFree(outAddr);
    aclrtFree(lseAddr);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    aclrtFree(dqAddr);
    aclrtFree(dkAddr);
    aclrtFree(dvAddr);
    if (metaWsSize > 0) {
        aclrtFree(metaWs);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
