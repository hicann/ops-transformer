# aclnnSparseLightningIndexerKLLossGrad

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
- <term>Atlas 200I/500 A2 推理产品</term>：x
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：x
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：x
<!-- end id6 -->

## 功能说明

- 接口功能：SparseLightningIndexerKLLossGrad算子是LightningIndexer KL Loss的反向计算算子。该接口接收Lightning Indexer分支的q、k、w、sparseIndices，以及由主Attention分支预先计算得到的attnSoftmaxL1Norm，计算并输出dq、dk、dw和softmaxOut。与SparseLightningIndexerGradKLLoss相比，本接口不再在kernel内部重算主Attention的`Q @ K -> softmax -> reduce/l1 norm`，也不再输出loss；主Attention的目标分布由attnSoftmaxL1Norm输入提供，softmaxOut可用于后续loss计算。

- 计算公式：
   用于取Top-k的value的Indexer logits可表示为：

   $$
   S_{t,:}=q_{t,:}@K_{\operatorname{topk}(t),:}^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   其中，$q$和$K$分别对应本接口的q和k，$W$对应本接口的w，$\operatorname{topk}(t)$由sparseIndices给出。Indexer分支的softmax输出为：

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
   dq_{t,:}=dS_{t,:}@K_{\operatorname{topk}(t),:}
   $$

   $$
   dK_{\operatorname{topk}(t),:}=\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK写回时会按照sparseIndices指向的key位置做scatter-add，无效top-k位置不参与计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnSparseLightningIndexerKLLossGrad”接口执行计算。

```c++
aclnnStatus aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize(
    const aclTensor     *q,
    const aclTensor     *k,
    const aclTensor     *w,
    const aclTensor     *sparseIndices,
    const aclTensor     *attnSoftmaxL1Norm,
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
aclnnStatus aclnnSparseLightningIndexerKLLossGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream stream)
```

## aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize

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
                <td>x</td>
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
                <td>x</td>
            </tr>
            <tr>
                <td>w</td>
                <td>输入</td>
                <td>Indexer logits的head权重。</td>
                <td>不支持空Tensor。</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N1)、(T1,N1)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>sparseIndices</td>
                <td>输入</td>
                <td>每个query对应的top-k key下标。</td>
                <td>
                    <ul>
                        <li>有效部分填充key下标。</li>
                        <li>无效部分填充-1。</li>
                        <li>不支持空Tensor。</li>
                    </ul>
                </td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,S1,N2,K)、(T1,N2,K)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>attnSoftmaxL1Norm</td>
                <td>输入</td>
                <td>主Attention分支预先计算出的目标分布。</td>
                <td>
                    <ul>
                        <li>用于替代旧kernel内部的主Attention重算结果。</li>
                        <li>维度需与sparseIndices一致。</li>
                        <li>不支持空Tensor。</li>
                    </ul>
                </td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N2,K)、(T1,N2,K)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>cuSeqLensQOptional</td>
                <td>可选输入</td>
                <td>TND layout下q的累积序列长度。</td>
                <td>当layoutQ为TND时需要传入，长度为B+1，首元素为0。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B+1,)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>cuSeqLensKOptional</td>
                <td>可选输入</td>
                <td>TND layout下k的累积序列长度。</td>
                <td>当layoutK为TND时需要传入，长度为B+1，首元素为0。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B+1,)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>seqUsedQOptional</td>
                <td>可选输入</td>
                <td>表示每个batch实际使用的q长度。</td>
                <td>长度为B。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>seqUsedKOptional</td>
                <td>可选输入</td>
                <td>表示每个batch实际使用的k长度。</td>
                <td>长度为B。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>cmpResidualKOptional</td>
                <td>可选输入</td>
                <td>maskMode=3且cmpRatio!=1时需要传入。</td>
                <td>长度为B。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>metadataOptional</td>
                <td>可选输入</td>
                <td>由aclnnSparseLightningIndexerKLLossGradMetadata生成的分核信息。</td>
                <td>传入后kernel将使用metadata中的分核切分。</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(x,)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>layoutQ</td>
                <td>输入</td>
                <td>q、w、sparseIndices、attnSoftmaxL1Norm、dq、dw、softmaxOut的layout。</td>
                <td>仅支持BSND和TND。</td>
                <td>STRING</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>layoutK</td>
                <td>输入</td>
                <td>k和dk的layout。</td>
                <td>仅支持BSND和TND。</td>
                <td>STRING</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>maskMode</td>
                <td>输入</td>
                <td>mask模式。</td>
                <td>支持的mask模式详见<a href="#约束说明">约束说明</a>。</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>输入</td>
                <td>压缩比。</td>
                <td>取值范围为1~128。</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
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
                <td>(B,S1,N2,K)、(T1,N2,K)</td>
                <td>x</td>
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

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持seqUsedQOptional、seqUsedKOptional字段。

<!-- end id7 -->
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

## aclnnSparseLightningIndexerKLLossGrad

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
                <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize获取。</td>
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

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：aclnnSparseLightningIndexerKLLossGrad默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：aclnnSparseLightningIndexerKLLossGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

    <!-- end id9 -->

- 公共约束：
  - 参数q、k、dq、dk的数据类型应保持一致，支持FLOAT16和BFLOAT16。
  - 参数w、dw、attnSoftmaxL1Norm、softmaxOut的数据类型应为FLOAT32。
  - 参数sparseIndices、cuSeqLensQOptional、cuSeqLensKOptional、seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional、metadataOptional的数据类型应为INT32。
  - layoutQ和layoutK当前支持BSND和TND。
  - 当layoutQ为TND时，需要传入cuSeqLensQOptional；当layoutK为TND时，需要传入cuSeqLensKOptional。
  - sparseIndices中有效位置必须位于当前batch的key序列范围内；无效位置使用-1填充。
  - attnSoftmaxL1Norm的无效top-k位置建议置零，并与sparseIndices的有效位置保持一致。

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
                <td>N2</td>
                <td>1</td>
                <td>当前仅支持N2=1。</td>
            </tr>
            <tr>
                <td>D</td>
                <td>128</td>
                <td>q和k最后一维需保持一致。</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>1~128</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

  - 参数B的支持情况:

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：B支持1~256。
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：B>0。
    <!-- end id11 -->

  - 参数S1、S2的支持情况:

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：S1支持1~8K，S2支持1~512K。
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：S1>0，S2>0。
    <!-- end id13 -->

  - 参数N1的支持情况:

    <!-- npu="A3,910b" id14 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：N1支持8、16、32、64。
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：N1支持1~128。
    <!-- end id15 -->

  - 参数K的支持情况:

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：K支持512、1024、2048、4096、8192。
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：K支持1~2048。

    <!-- end id17 -->

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"

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
  const int64_t topk = 512;
  const int64_t maskMode = 3;
  const int64_t cmpRatio = 1;

  std::vector<int64_t> qShape = {t1, numHeadsQ, headDim};
  std::vector<int64_t> kShape = {t2, numHeadsK, headDim};
  std::vector<int64_t> wShape = {t1, numHeadsQ};
  std::vector<int64_t> sparseShape = {t1, numHeadsK, topk};
  std::vector<int64_t> softmaxShape = {t1, numHeadsK, topk};
  std::vector<int64_t> cuSeqLensShape = {batchSize + 1};
  std::vector<int64_t> metadataShape = {64};

  std::vector<uint16_t> qHostData(GetShapeSize(qShape), 0x3C00);
  std::vector<uint16_t> kHostData(GetShapeSize(kShape), 0x3C00);
  std::vector<float> wHostData(GetShapeSize(wShape), 1.0f);
  std::vector<int32_t> sparseHostData(GetShapeSize(sparseShape), -1);
  std::vector<float> attnSoftmaxL1NormHostData(GetShapeSize(softmaxShape), 0.0f);
  for (int64_t t = 0; t < t1; ++t) {
    auto validK = std::min<int64_t>(std::max<int64_t>(1, (t + 1) / cmpRatio), t2);
    for (int64_t kIdx = 0; kIdx < validK; ++kIdx) {
      sparseHostData[t * numHeadsK * topk + kIdx] = static_cast<int32_t>(kIdx);
      attnSoftmaxL1NormHostData[t * numHeadsK * topk + kIdx] = 1.0f / static_cast<float>(validK);
    }
  }
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
  void* sparseDeviceAddr = nullptr;
  void* attnSoftmaxL1NormDeviceAddr = nullptr;
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
  aclTensor* sparseIndices = nullptr;
  aclTensor* attnSoftmaxL1Norm = nullptr;
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
  ret = CreateAclTensor(sparseHostData, sparseShape, &sparseDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnSoftmaxL1NormHostData, softmaxShape, &attnSoftmaxL1NormDeviceAddr,
                        aclDataType::ACL_FLOAT, &attnSoftmaxL1Norm);
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
  ret = aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, batchSize, t1, t2, numHeadsQ, numHeadsK, headDim, topk,
      layoutQ, layoutK, maskMode, cmpRatio, metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSparseLightningIndexerKLLossGradMetadata(metadataWorkspaceAddr, metadataWorkspaceSize,
                                                     metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseLightningIndexerKLLossGradMetadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize(
      q, k, w, sparseIndices, attnSoftmaxL1Norm, cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, metadata,
      layoutQ, layoutK, maskMode, cmpRatio, dq, dk, dw, softmaxOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnSparseLightningIndexerKLLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSparseLightningIndexerKLLossGrad failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  PrintOutResult(softmaxShape, &softmaxOutDeviceAddr);

  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(w);
  aclDestroyTensor(sparseIndices);
  aclDestroyTensor(attnSoftmaxL1Norm);
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
  aclrtFree(sparseDeviceAddr);
  aclrtFree(attnSoftmaxL1NormDeviceAddr);
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
