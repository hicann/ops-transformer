# BlockSparseAttentionGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="310b" id6 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id7 -->
<!-- npu="910" id8 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id8 -->

## 功能说明

* ​算子功能​：aclnnBlockSparseAttention稀疏注意力反向计算，支持灵活的块级稀疏模式，通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。
* ​计算公式​：

  稀疏块大小：$blockShapeX×blockShapeY$，BlockSparseMask指定稀疏模式。

  已知正向计算公式为：

  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S = Mask(scale⋅query⋅key_{sparse}^{T},atten\_mask)
  $$

  $$
  P = SoftMax(S)
  $$

  $$
  V = value_{sparse}
  $$

  $$
  Out = PV
  $$

  则反向计算公式为：

  $$
  softmax\_grad = softmaxGrad(dOut, attentionOut)
  $$

  $$
  dP=dOut * V^T
  $$

  $$
  dS = P * (dP-softmax\_grad)
  $$

  $$
  dV=P^T * dOut
  $$

  $$
  dQ=(dS*K)*scale
  $$

  $$
  dK=(dS^T*Q)*scale
  $$

BlockSparseAttentionGrad输入dout、query、key、value, attentionOut的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。为了方便理解后续支持的具体排布格式（如 BNSD、TND 等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

* B：表示输入样本批量大小（Batch）
* T：B和S合轴紧密排列的长度（Total tokens）
* S：表示输入样本序列长度（Seq-Length）
* H：表示隐藏层的大小（Head-Size）
* N：表示多头数（Head-Num）
* D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

当前支持的布局：

* qInputLayout: "TND" "BNSD" "BSND"
* kvInputLayout: "TND" "BNSD" "BSND"

- <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>:
不支持"BSND"。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnBlockSparseAttentionGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBlockSparseAttentionGrad"接口执行计算。

```Cpp
aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
  const aclTensor   *dout,
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attentionOut,
  const aclTensor   *softmaxLse,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            preTokens,
  int64_t            nextTokens,
  aclTensor         *dq,
  aclTensor         *dk,
  aclTensor         *dv,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnBlockSparseAttentionGrad(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnBlockSparseAttentionGradGetWorkspaceSize

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
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
    <td>dout（aclTensor*）</td>
    <td>输入</td>
    <td>反向输出梯度，代表最终输出对当前算子的梯度信息。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>query（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的查询向量，即公式中的query。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的键向量，即公式中的key。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的值向量，即公式中的value。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attentionOut（aclTensor*）</td>
    <td>输入</td>
    <td>正向 BlockSparseAttention 计算的输出结果，即公式中的attentionOut。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>softmaxLse（aclTensor*）</td>
    <td>输入</td>
    <td>Softmax计算的log-sum-exp中间结果。用于反向计算梯度的对数和指数逆推。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, 1]。</li><li>BNSD: [batch, headNum, maxQSeqLength, 1]。</li><li>BNS1: [batch, maxQSeqLength, headNum, 1]。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockSparseMaskOptional（aclTensor*）</td>
    <td>输入</td>
    <td>块状稀疏掩码，表示实际的稀疏pattern。决定哪些block实际参与注意力计算。</td>
    <td>不支持空Tensor。<br>可选输入（当前版本为必选）：<ul><li>shape为[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]。</li><li>表示按block划分后哪些block需要参与计算（为1），哪些block不参与计算（为0）。</li><li>如传入nullptr，则视为不开启块稀疏计算，即所有token之间的注意力分数都会被计算。</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional（aclTensor*）</td>
    <td>输入</td>
    <td>注意力掩码，即公式中的atten_mask。用于屏蔽不应参与计算的特定token。</td>
    <td>支持空Tensor。<br>当前不支持，应传入nullptr。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td rowspan="3">blockShapeOptional（aclIntArray*）</td>
    <td rowspan="3">输入</td>
    <td rowspan="3">稀疏块形状数组。指定每个稀疏块的二维尺寸（行数和列数）。</td>
    <td> <ul><li>当配置了blockSparseMaskOptional时：如配置此输入，算子会从中获取稀疏块尺寸；如不配置此输入，算子将默认稀疏块尺寸为[128,128]。</li></ul></td>
    <td rowspan="3">INT64</td>
    <td rowspan="3">-</td>
    <td rowspan="3">1</td>
    <td rowspan="3">-</td>
    </tr>
    <tr>
    <td><ul><li>当未配置blockSparseMaskOptional时：无论此项如何配置，算子均将忽略。</li></ul></td>
    </tr>
    <tr>
    <td>当配置此输入时的元素要求：<ul><li>必须包含至少两个元素 [blockShapeX, blockShapeY]。</li><li>blockShapeX: Q方向块大小，值必须大于0。</li><li>blockShapeY: KV方向块大小，值必须大于0。</li></ul></td>
    </tr>
    <tr>
    <td>actualSeqLengthsOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>query的实际序列长度数组。<br>用于描述变长序列场景下（即含有 Padding 填充数据的场景），每个 Batch 中实际有效的 query token 数量。</td>
    <td> 变长序列场景（当 qInputLayout 为 "TND" 时）：该项输入必须配置。因为 TND 格式为一维连续排布，算子需要依赖该数组来准确切分界定各个序列的真实边界。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKvOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>key/value的实际序列长度数组。<br>用于描述变长序列场景下（即含有 Padding 填充数据的场景），每个 Batch 中实际有效的 key/value token 数量。</td>
    <td> 变长序列场景（当 kvInputLayout 为 "TND" 时）：该项输入必须配置。因为 TND 格式为一维连续排布，算子需要依赖该数组来准确切分界定各个序列的真实边界。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout（char*）</td>
    <td>输入</td>
    <td>query的数据排布格式。指示输入张量在内存中的具体排布（如连续或合轴排列）。</td>
    <td>当前仅支持"TND"、"BNSD"、"BSND"，qInputLayout与kvInputLayout需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout（char*）</td>
    <td>输入</td>
    <td>key和value的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>当前仅支持"TND"、"BNSD"、"BSND"，qInputLayout与kvInputLayout需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads（int64_t）</td>
    <td>输入</td>
    <td>key/value的注意力头数。用于支持GQA（分组查询注意力）机制下的头数比例映射。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType（int64_t）</td>
    <td>输入</td>
    <td>注意力计算中的掩码类型。指定采用何种预设规则的掩码逻辑。</td>
    <td>当前只支持传 0：代表不加mask场景。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue（double）</td>
    <td>输入</td>
    <td>缩放系数，即公式中的scale。用于注意力分数的归一化处理。</td>
    <td>一般设置为D^-0.5。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens（int64_t）</td>
    <td>输入</td>
    <td>滑窗向前包含的token数量。限制当前token只能与前方的多少个历史token计算注意力。</td>
    <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens（int64_t）</td>
    <td>输入</td>
    <td>滑窗向后包含的token数量。限制当前token只能与后方的多少个未来token计算注意力。</td>
    <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dq（aclTensor*）</td>
    <td>输出</td>
    <td>query的梯度输出结果，即公式中的dq。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入query保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dk（aclTensor*）</td>
    <td>输出</td>
    <td>key的梯度输出结果，即公式中的dk。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入key保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dv（aclTensor*）</td>
    <td>输出</td>
    <td>value的梯度输出结果，即公式中的dv。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入value保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
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
    <td>executor（aclOpExecutor**）</td>
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

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>输入dout，query，key，value，attentionOut传入的是空指针。</td>
    </tr>
    <tr>
      <td>qInputLayout为"TND"时，actualSeqLengthsOptional传入的是空指针。</td>
    </tr>
    <tr>
      <td>kvInputLayout为"TND"时，actualSeqLengthsKvOptional传入的是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>dout，query，key，value数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>qInputLayout或kvInputLayout输入不合法，参数有效性校验失败。</td>
    </tr>
  </tbody></table>

## aclnnBlockSparseAttentionGrad

* **参数说明：**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockSparseAttentionGradGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的AscendCL stream流。</td>
    </tr>
  </tbody>
  </table>

* **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
* actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
* softmaxLse的layout需要与query的layout保持一致。如果query的layout为"BSND"时，softmaxLse的layout应传入"BNS1"。
* HeadDim必须等于128。
* 根据算子支持的输入 Layout，query 张量 Shape 中对应的 head 维度大小记为 N1，key 和 value 张量 Shape 中对应的 head 维度大小记为 N2。必须满足N1 % N2 == 0。
  - <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>：当前只支持MHA，即N1等于N2。

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：支持MQA、MHA、GQA场景。
  <!-- end id9 -->

* actualSeqLengthsOptional与actualSeqLengthsKvOptional相关约束：

  - <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>：当 qInputLayout或kvInputLayout 为 "BNSD" 时，如配置该项，算子会按指定的有效长度处理，忽略 Padding 部分的数据，提升性能；如不配置（传nullptr），算子将默认把 query shape 中的 S 维度作为有效长度进行全量处理。

  <!-- npu="950" id10 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：当qInputLayout或kvInputLayout 为非"TND"时，会忽略这两个入参。

  <!-- end id10 -->

* 不支持确定性计算场景。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention_grad.h"

using namespace std;

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

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置核心参数(以BNSD Layout为例)
    int32_t batch = 1;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t headDim = 128;
    int32_t blockShapeX = 64;
    int32_t blockShapeY = 64;

    // 块数量计算
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilKv = (kvSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. 构建张量Shape
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kvShape = {batch, numKvHeads, kvSeqlen, headDim};
    std::vector<int64_t> lseShape = {batch, numHeads, qSeqlen}; // LSE通常没有尾部1维，防止GE squeeze
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilKv};

    // 4. 分配并初始化Host数据
    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);

    // 将Q, K, V初始化为0.1f等较小的数
    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kvSize, 0.1f);
    std::vector<op::fp16_t> vData(kvSize, 0.1f);

    // 梯度初始值可以给一个小正数
    std::vector<op::fp16_t> doutData(qSize, 0.01f);
    std::vector<op::fp16_t> outData(qSize, 0.1f);

    // LSE给一个合理的正数，比如5.0f，这样exp(S - LSE)就是一个非常安全的负指数，绝对不会溢出
    std::vector<float> lseData(GetShapeSize(lseShape), 5.0f);
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 1);

    // 创建所有的前向输入/输出aclTensor
    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr;
    void *lseAddr = nullptr, *maskAddr = nullptr;

    aclTensor *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *doutTensor = nullptr, *outTensor = nullptr;
    aclTensor *lseTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(vData, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CreateAclTensor(doutData, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &doutTensor);
    CreateAclTensor(outData, qShape, &outAddr, aclDataType::ACL_FLOAT16, &outTensor);

    CreateAclTensor(lseData, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lseTensor);     // 严格使用FP32
    CreateAclTensor(maskData, maskShape, &maskAddr, aclDataType::ACL_UINT8, &maskTensor); // 严格使用UINT8

    // 5. 创建反向输出梯度(dq, dk, dv)
    std::vector<op::fp16_t> dqData(qSize, 0.0f);
    std::vector<op::fp16_t> dkData(kvSize, 0.0f);
    std::vector<op::fp16_t> dvData(kvSize, 0.0f);

    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;

    CreateAclTensor(dqData, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dqTensor);
    CreateAclTensor(dkData, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dkTensor);
    CreateAclTensor(dvData, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dvTensor);

    // 6. 创建aclIntArray属性参数(BlockShape & ActualSeqLengths)
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    std::vector<int64_t> qSeqLenVec(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> kvSeqLenVec(batch, static_cast<int64_t>(kvSeqlen));
    aclIntArray *qSeqLenArr = aclCreateIntArray(qSeqLenVec.data(), batch);
    aclIntArray *kvSeqLenArr = aclCreateIntArray(kvSeqLenVec.data(), batch);

    // 7. 标量与字符串参数配置
    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    // 强制规定滑动窗口极大值
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;

    // 8. 调用第一段接口: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnBlockSparseAttentionGradGetWorkspaceSize...\n");
    ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
        doutTensor,
        qTensor,
        kTensor,
        vTensor,
        outTensor,
        lseTensor,
        maskTensor,                 // blockSparseMaskOptional
        nullptr,                    // attenMaskOptional必须为空
        blockShapeArr,
        qSeqLenArr,
        kvSeqLenArr,
        qLayoutBuffer,
        kvLayoutBuffer,
        static_cast<int64_t>(numKvHeads),
        maskType,
        scaleValue,
        preTokens,
        nextTokens,
        dqTensor,
        dkTensor,
        dvTensor,
        &workspaceSize,
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 9. 分配workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 10. 调用第二段接口: 执行计算
    LOG_PRINT("Calling aclnnBlockSparseAttentionGrad...\n");
    ret = aclnnBlockSparseAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 11. 同步Stream，等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 12. 将结果拷贝回Host侧打印
    ret = aclrtMemcpy(dqData.data(), qSize * sizeof(op::fp16_t), dqAddr, qSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! Output results (first 10 elements of dQ):\n");
    for (uint64_t i = 0; i < 10 && i < dqData.size(); i++) {
        LOG_PRINT("  dQ index %lu: %f\n", i, static_cast<float>(dqData[i]));
    }

    // 13. 释放所有资源
    LOG_PRINT("Cleaning up resources...\n");
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(doutAddr); aclrtFree(outAddr);
    aclrtFree(lseAddr); aclrtFree(maskAddr);
    aclrtFree(dqAddr); aclrtFree(dkAddr); aclrtFree(dvAddr);

    aclDestroyTensor(qTensor); aclDestroyTensor(kTensor); aclDestroyTensor(vTensor);
    aclDestroyTensor(doutTensor); aclDestroyTensor(outTensor);
    aclDestroyTensor(lseTensor); aclDestroyTensor(maskTensor);
    aclDestroyTensor(dqTensor); aclDestroyTensor(dkTensor); aclDestroyTensor(dvTensor);

    aclDestroyIntArray(blockShapeArr);
    aclDestroyIntArray(qSeqLenArr);
    aclDestroyIntArray(kvSeqLenArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BlockSparseAttentionGrad Test completed successfully!\n");
    return 0;
}
```
