# aclnnSparseFlashAttentionV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention)

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

- 接口功能：sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。V2版本新增sinks参数。

- 计算公式：

$$
\text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
$$

其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSparseFlashAttentionV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseFlashAttentionV2”接口执行计算。

```Cpp
aclnnStatus aclnnSparseFlashAttentionV2GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *sparseIndices,
    const aclTensor     *blockTableOptional,
    const aclTensor     *actualSeqLengthsQueryOptional,
    const aclTensor     *actualSeqLengthsKvOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    const aclTensor     *sinksOptional,
    double              scaleValue,
    int64_t             sparseBlockSizeOptional,
    char                *layoutQueryOptional,
    char                *layoutKvOptional,
    int64_t             sparseMode,
    int64_t             preTokens,
    int64_t             nextTokens,
    int64_t             attentionMode,
    bool                returnSoftmaxLse,
    const aclTensor     *attentionOutOut,
    const aclTensor     *softmaxMaxOut,
    const aclTensor     *softmaxSumOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnSparseFlashAttentionV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnSparseFlashAttentionV2GetWorkspaceSize

- **参数说明：**

  > [!NOTE]
  >
  >- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
  >- Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N和N1表示num\_query\_heads，KV\_N和N2表示num\_key\_value\_heads，T1表示query shape中的T，T2表示key shape中的输入样本序列长度的累加和。

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 500px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 400px">
  <col style="width: 146px">
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
      <td>query（aclTensor）</td>
      <td>输入</td>
      <td>attention结构的Query输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
     <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key（aclTensor）</td>
      <td>输入</td>
      <td>attention结构的Key输入</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为(block_num, block_size, KV_N, D)。</li>
                <li>layout_kv为BSND时，shape为(B, S2, KV_N, D)。</li>
                <li>layout_kv为TND时，shape为(T2, KV_N, D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>value（aclTensor）</td>
      <td>输入</td>
      <td>attention结构的Value输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>shape与key的shape一致。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseIndices（aclTensor）</td>
      <td>输入</td>
      <td>离散取kvCache的索引。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse_size大于0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B, Q_S, KV_N, sparse_size)。</li>
                <li>layout_query为TND时，shape为(Q_T, KV_N, sparse_size)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示PageAttention中kvCache存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>第二维长度不小于所有batch中最大的S2对应的block数量，即S2_max / block_size向上取整。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape支持(B,S2/block_size)。</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示不同Batch中query的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和query的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0。支持长度为B的一维tensor。</li>
                <li>layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该参数中每个元素的值表示当前batch与之前所有batch的token数总和。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示不同Batch中key和value的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li>该参数中每个Batch的有效token数不超过key/value中的维度S大小且不小于0。支持长度为B的一维tensor。</li>
                <li>当layout_kv为TND或PA_BSND时，该入参必须传入。</li>
                <li>layout_kv为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryRopeOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示MLA结构中的query的rope信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为TND时，shape为(B,S1,N1,Dr)。</li>
                <li>layout_query为BSND时，shape为(T1,N1,Dr)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyRopeOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示MLA结构中的key的rope信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为TND时，shape为(B,S1,N1,Dr)。</li>
                <li>layout_kv为BSND时，shape为(T1,N1,Dr)。</li>
                <li>layout_kv为PA_BSND时，shape为(block_num,block_size,N2,Dr)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sinksOptional（aclTensor）</td>
      <td>输入</td>
      <td>attention结构中的可学习的sinks信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>shape支持(N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue（double）</td>
      <td>输入</td>
      <td>代表缩放系数。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSizeOptional（int64_t）</td>
      <td>输入</td>
      <td>代表sparse阶段的block大小。</td>
      <td>
          <ul>
                <li>sparse_block_size为1时，为Token-wise稀疏化场景，将每个token视为独立单元，在计算重要性分数时，评估每个查询token与每个键值token之间的独立关联程度。</li>
                <li>sparse_block_size为大于1小于等于128时，为Block-wise稀疏化场景，将token序列划分为固定大小的连续块，以块为单位进行重要性评估，块内token共享相同的稀疏化决策。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional（char）</td>
      <td>输入</td>
      <td>标识输入query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>支持传入BSND和TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional（char）</td>
      <td>输入</td>
      <td>标识输入key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>支持传入TND、BSND和PA_BSND，其中PA_BSND在开启PageAttention时使用。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>sparse_mode为0时，代表全部计算。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionMode（int64_t）</td>
      <td>输入</td>
      <td>-</td>
      <td>仅支持传入2，表示MLA-absorb模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse（bool）</td>
      <td>输入</td>
      <td>用于表示是否返回softmax_max和softmax_sum。</td>
      <td>
          <ul>
                <li>True表示返回，False表示不返回；默认值为False。</li>
                <li>该参数仅在训练且layout_kv不为PA_BSND场景支持。</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut（aclTensor）</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>softmaxMaxOut（aclTensor）</td>
      <td>输出</td>
      <td>Attention算法对query乘key的结果，取max得到softmax_max。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,N2,S1,N1/N2)。</li>
                <li>layout_query为TND时shape为(N2,T1,N1/N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
   <tr>
      <td>softmaxSumOut（aclTensor）</td>
      <td>输出</td>
      <td>Attention算法query乘key的结果减去softmax_max,再取exp，接着求sum，得到softmax_sum。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,N2,S1,N1/N2)。</li>
                <li>layout_query为TND时shape为(N2,T1,N1/N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor）</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
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
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、value、sparseIndices、blockTableOptional、actualSeqLengthsQueryOptional、actualSeqLengthsKvOptional、queryRopeOptional、keyRopeOptional、sinksOptional、scaleValue、sparseBlockSizeOptional、layoutQueryOptional、layoutKvOptional、sparseMode、attentionMode、returnSoftmaxLse、attentionOut、softmaxMaxOut、softmaxSumOut的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashAttentionV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashAttentionV2GetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnSparseFlashAttentionV2默认确定性实现。
- 该接口支持推理场景下使用。
- N1支持情况：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - N1支持1~128。
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - N1支持1/2/4/8/16/32/64/128。
  <!-- end id8 -->

- N2仅支持1。
- block_size为一个block的token数，block_size取值为16的倍数，且最大支持1024。
- 参数query中的D和key、value的D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- 支持sparse_block_size整除block_size。

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 只支持sparse_block_size为1。
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 支持[1,128]，且要求是2的幂次方，在PageAttention场景下要求sparse_block_size整除block_size
  <!-- end id10 -->

- 参数sinks仅支持Ascend 950PR/Ascend 950DT。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
/*!
 * \file test_aclnn_sparse_flash_attention_v2.cpp
 * \brief aclnnSparseFlashAttentionV2接口调用示例
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_sparse_flash_attention_v2.h"

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
    // 固定写法，AscendCL初始化
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
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
    std::vector<int64_t> kShape = {2048, 1, 512};              // T2, N2, D
    std::vector<int64_t> vShape = {2048, 1, 512};              // T2, N2, D
    std::vector<int64_t> sparseIndicesShape = {1, 1, 2048};    // T1, N2, K
    std::vector<int64_t> outShape = {1, 16, 512};             // T1, N1, D
    std::vector<int64_t> softmaxMaxShape = {1, 1, 16};        // N2, T1, G
    std::vector<int64_t> softmaxSumShape = {1, 1, 16};        // N2, T1, G
    std::vector<int64_t> actSeqQLenshape = {1};               // B
    std::vector<int64_t> actSeqKvLenshape = {1};           // B
    std::vector<int64_t> qRopeShape = {1, 16, 64};            // T1, N1, Drope
    std::vector<int64_t> kRopeShape = {2048, 1, 64};          // T2, N2, Drope
    std::vector<int64_t> sinksShape = {16};                   // N1

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* actSeqQLenDeviceAddr = nullptr;
    void* actSeqKvLenDeviceAddr = nullptr;
    void* qRopeDeviceAddr = nullptr;
    void* kRopeDeviceAddr = nullptr;
    void* sinksDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* sparseIndices = nullptr;
    aclTensor* out = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* actSeqQLen = nullptr;
    aclTensor* actSeqKvLen = nullptr;
    aclTensor* qRope = nullptr;
    aclTensor* kRope = nullptr;
    aclTensor* sinks = nullptr;

    std::vector<op::fp16_t> qHostData(1 * 16 * 512, 1.0);
    std::vector<op::fp16_t> kHostData(2048 * 1 * 512, 1.0);
    std::vector<op::fp16_t> vHostData(2048 * 1 * 512, 1.0);
    std::vector<int32_t> sparseIndicesHostData(2048);
    std::iota(sparseIndicesHostData.begin(), sparseIndicesHostData.end(), 0);
    std::vector<op::fp16_t> outHostData(1 * 16 * 512, 1.0);
    std::vector<float> softmaxMaxHostData(16, 3.0);
    std::vector<float> softmaxSumHostData(16, 3.0);
    std::vector<int32_t> actSeqQLenHostData(1, 1);
    std::vector<int32_t> actSeqKvLenHostData(1, 2048);
    std::vector<op::fp16_t> qRopeHostData(1 * 16 * 64, 1.0);
    std::vector<op::fp16_t> kRopeHostData(2048 * 1 * 64, 1.0);
    std::vector<float> sinksHostData(16, 0.0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape,
                          &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape,
                          &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape,
                          &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqQLenHostData, actSeqQLenshape,
                          &actSeqQLenDeviceAddr, aclDataType::ACL_INT32, &actSeqQLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqKvLenHostData, actSeqKvLenshape,
                          &actSeqKvLenDeviceAddr, aclDataType::ACL_INT32, &actSeqKvLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_FLOAT16, &qRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double scaleValue = 0.0416666666666667;
    int64_t sparseBlockSize = 1;
    int64_t sparseMode = 0;
    int64_t attentionMode = 2;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    // bool deterministic = false;
    char layoutQuery[5] = {'T', 'N', 'D', 0};
    char layoutKey[5] = {'T', 'N', 'D', 0};

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool returnSoftmaxLse = false;

    // 调用aclnnSparseFlashAttentionV2第一段接口
    ret = aclnnSparseFlashAttentionV2GetWorkspaceSize(q, k, v, sparseIndices, nullptr, actSeqQLen, actSeqKvLen,
                                                      qRope, kRope, sinks, scaleValue, sparseBlockSize, layoutQuery,
                                                      layoutKey, sparseMode, preTokens, nextTokens, attentionMode,
                                                      returnSoftmaxLse, out, softmaxMax, softmaxSum, &workspaceSize,
                                                      &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnSparseFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnSparseFlashAttentionV2第二段接口
    ret = aclnnSparseFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparseFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(sparseIndices);
    aclDestroyTensor(out);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(actSeqQLen);
    aclDestroyTensor(actSeqKvLen);
    aclDestroyTensor(qRope);
    aclDestroyTensor(kRope);
    aclDestroyTensor(sinks);

    // 6. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(sparseIndicesDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(actSeqQLenDeviceAddr);
    aclrtFree(actSeqKvLenDeviceAddr);
    aclrtFree(qRopeDeviceAddr);
    aclrtFree(kRopeDeviceAddr);
    aclrtFree(sinksDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
