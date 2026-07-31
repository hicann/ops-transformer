# aclnnBSASelectBlockMask

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
- <term>Atlas A3训练系列产品</term>：支持
- <term>Atlas A3推理系列产品</term>：支持
- <term>Atlas A2训练系列产品</term>：支持
- <term>Atlas A2推理系列产品</term>：支持
- <term>Atlas 200I/500 A2推理产品</term>：不支持

<!-- npu="310p" id2 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id3 -->

## 功能说明

- 接口功能​：aclnnBSASelectBlockMask是BSA（BlockSparseAttention）的前置算子，负责根据Query和Key的内容动态生成blockSparseMask，使BSA的调用链从"手动提供掩码"变为"根据Q/K内容自适应选择稀疏模式"。

- 计算公式​：

  设blockShape = [blockShapeX, blockShapeY]，Sq是query最大序列长度，Skv是key最大序列长度, 则压缩后块数：

  $$
  Xblocks = \lceil Sq / blockShapeX \rceil,\quad Yblocks = \lceil Skv / blockShapeY \rceil
  $$

  **Step1：均值池化压缩 (Mean Pooling Compression)**

  当actualBlockLenQuery / actualBlockLenKey为null时（完整压缩）：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{blockShapeX} \sum_{i=0}^{blockShapeX-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{blockShapeY} \sum_{j=0}^{blockShapeY-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  当actualBlockLenQuery / actualBlockLenKey非null时（部分压缩），仅对每个block内前actualBlockLen个token取均值：

  $$
  q\_compressed[b, n, x, d] = \frac{1}{actualBlockLenQ[b,x]} \sum_{i=0}^{actualBlockLenQ[b,x]-1} query[b, n, x \cdot blockShapeX + i, d]
  $$

  $$
  k\_compressed[b, n, y, d] = \frac{1}{actualBlockLenK[b,y]} \sum_{j=0}^{actualBlockLenK[b,y]-1} key[b, n, y \cdot blockShapeY + j, d]
  $$

  **Step2a：QK Matmul**

  $$
  score[b, n, x, y] = scale \cdot \sum_{d=0}^{D-1} q\_compressed[b, n, x, d] \cdot k\_compressed[b, n, y, d]
  $$

  **Step2b：Softmax**

  $$
  attn\_score[b, n, x, y] = softmax(score[b, n, x, :]) = \frac{\exp(score[b, n, x, y] - m_{final})}{l_{final}}
  $$

  **Step3：TopK选择生成索引**

  $$
  topk\_value = \text{round}(sparsity \times Xblocks \times Yblocks)
  $$

  $$
  \mathcal{indices}= \text{TopK}\left(attn\_score[b, n, x, y],\; topK\_value\right)
  $$

  其中indices为attn\_score[b, n, x, y] 中topk\_value个最大值对应的索引集合。

  **Step4：生成BlockSparseMask**
  $$
  blockSparseMaskOut[b, n, x, y] =
  \begin{cases}
  1 & (b, n, x, y) \in \mathcal{indices} \\
  0 & (b, n, x, y) \notin \mathcal{indices}
  \end{cases}
  $$

- 数据排布格式：

  BSASelectBlockMask输入query、key的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。为了方便理解后续支持的具体排布格式（如BNSD、TND等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

  - B：表示输入样本批量大小（Batch）
  - T：B和S合轴紧密排列的长度（Total tokens）
  - S：表示输入样本序列长度（Seq-Length）
  - H：表示隐藏层的大小（Head-Size）
  - N：表示多头数（Head-Num）
  - D：表示隐藏层最小的单元尺寸，需满足D = H / N（Head-Dim）

- 当前支持的布局：

  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnBSASelectBlockMaskGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBSASelectBlockMask"接口执行计算。

```Cpp
aclnnStatus aclnnBSASelectBlockMaskGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclIntArray *blockShape,
  const aclIntArray *postBlockShape,
  const aclIntArray *actualSeqLengths,
  const aclIntArray *actualSeqLengthsKV,
  const aclIntArray *actualBlockLenQuery,
  const aclIntArray *actualBlockLenKey,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  double             scaleValue,
  double             sparsity,
  aclTensor         *blockSparseMaskOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnBSASelectBlockMask(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnBSASelectBlockMaskGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 140px">
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
    <td>query（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的query矩阵，即公式中的query。</td>
    <td><ul><li>不支持空Tensor。</li><li>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li></ul></li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的key矩阵，即公式中的key。</td>
    <td><ul><li>不支持空Tensor。</li><li>支持的shape为：<ul><li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKSeqLength, headDim]。</li></ul></li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
        <tr>
    <td>blockShape（aclIntArray*）</td>
    <td>输入</td>
    <td>稀疏块形状数组。指定每个稀疏块的二维尺寸（行数和列数）。</td>
    <td><ul><li>当配置此输入时的元素要求：<ul><li>必须包含至少两个元素 [blockShapeX, blockShapeY]。</li><li>blockShapeX: Q方向块大小，必须为64的倍数且大于0。</li><li>blockShapeY: KV方向块大小，必须为64的倍数且大于0。</li></ul></li><li>如不配置（传nullptr），算子将默认blockShapeX = 128，blockShapeY = 128。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>postBlockShape（aclIntArray*）</td>
    <td>输入</td>
    <td>预留参数，用于Softmax后二次压缩。</td>
    <td><ul>当前不支持，必须传入nullptr。</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualSeqLengths（aclIntArray*）</td>
    <td>输入</td>
    <td>每个batch的query的实际序列长度。<br>用于描述变长序列场景下（即含有Padding填充数据的场景），每个Batch中实际有效的query token数量。</td>
    <td><ul><li>变长序列场景（当qInputLayout为 "TND" 时）：该项输入必须配置。</li><li>定长/变长场景（当qInputLayout为 "BNSD" 时）：<ul><li>如配置该项，算子会按指定的有效长度处理，忽略Padding部分的数据，提升性能；</li><li>如不配置（传nullptr），算子将默认把query shape中的S维度作为有效长度进行全量处理。</li></ul></li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualSeqLengthsKV（aclIntArray*）</td>
    <td>输入</td>
    <td>key的实际序列长度数组。<br>用于描述变长序列场景下（即含有Padding填充数据的场景），每个Batch中实际有效的key token数量。</td>
    <td><ul><li>变长序列场景（当kvInputLayout为 "TND" 时）：该项输入必须配置。</li><li>>定长/变长场景（当kvInputLayout为 "BNSD" 时）：<ul><li>如配置该项，算子会按指定的有效长度处理，忽略Padding部分的数据，提升性能；</li><li>如不配置（传nullptr），算子将默认把key shape中的S维度作为有效长度进行全量处理。</li></ul></li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualBlockLenQuery（aclIntArray*）</td>
    <td>输入</td>
    <td>每个query block内实际压缩的有效seq长度。<br>用于部分压缩场景（如末尾不完整块或仅压缩有效token）。</td>
    <td><ul><li>可选输入：<ul><li>BNSD场景：shape为 [B, Xblocks]。</li><li>TND场景：shape为 [TotalBlockNum_Q]（各batch的实际有效Xblocks堆叠，vaildXblocks = ceil（actualSeqQ / blockShapeX））。</li><li>每个元素取值范围 [0, blockShapeX]。</li><li>当actualBlockLen = 0时：对应block的q_cmp填0向量，不会被topK选中。</li><li>当actualBlockLen > 0时：仅对前actualBlockLen个token取均值。</li></ul></li>
    <li>如不配置（传nullptr）：对query进行完整压缩（使用完整blockShapeX长度）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
        <tr>
    <td>actualBlockLenKey（aclIntArray*）</td>
    <td>输入</td>
    <td>每个key block内实际压缩的有效seq长度。<br>用于部分压缩场景（如末尾不完整块或仅压缩有效token）。</td>
    <td><ul><li>可选输入：<ul><li>BNSD场景：shape为 [B, Yblocks]。</li><li>TND场景：shape为 [TotalBlockNum_K]（各batch的实际有效Yblocks堆叠，vaildYblocks = ceil（actualSeqK / blockShapeY））。</li><li>每个元素取值范围 [0, blockShapeY]。</li><li>当actualBlockLen = 0时：对应block的k_cmp填0向量， 不会被topK选中。</li><li>当actualBlockLen > 0时：仅对前actualBlockLen个token取均值。</li></ul></li>
    <li>如不配置（传nullptr）：对key进行完整压缩（使用完整blockShapeY长度）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout（char*）</td>
    <td>输入</td>
    <td>query的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td><ul>当前仅支持 "TND"、"BNSD"，qInputLayout与kvInputLayout需要保持一致。</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout（char*）</td>
    <td>输入</td>
    <td>key的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td><ul>当前仅支持 "TND"、"BNSD"，qInputLayout与kvInputLayout需要保持一致。</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads（int64_t）</td>
    <td>输入</td>
    <td>key的注意力头数。</td>
    <td><ul>当前仅支持MHA，numKeyValueHeads必须等于numHeads。</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue（double）</td>
    <td>输入</td>
    <td>缩放系数，即公式中的scale。用于注意力分数的归一化处理。</td>
    <td><ul>一般设置为1 / sqrt(D)。</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparsity（double）</td>
    <td>输入</td>
    <td>稀疏度保留比例。指定公式中attn_score中需要保留的块位置占全部块位置的比例。</td>
    <td><ul>取值范围 (0.0, 1.0)。</ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>blockSparseMaskOut（aclTensor*）</td>
    <td>输出</td>
    <td>块状稀疏掩码输出，表示根据Q/K内容自适应生成的稀疏pattern。可直接作为BSA算子的blockSparseMask输入。</td>
    <td> <ul><li>不支持空Tensor。</li><li>shape为 [B, N, Xblocks, Yblocks]：<ul><li>Xblocks = ceilDiv(maxQSeqlen, blockShapeX)。</li><li>Yblocks = ceilDiv(maxKSeqlen, blockShapeY)。</li><li>值为1表示该block参与注意力计算，值为0表示不参与。</li></ul></li></ul></td>
    <td>INT8</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
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
      <td>输入query传入的是空指针。</td>
    </tr>
    <tr>
      <td>输入key传入的是空指针。</td>
    </tr>
    <tr>
      <td>blockSparseMaskOut传入的是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、key或者blockSparseMaskOut数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>query和key数据类型不一致。</td>
    </tr>
  </tbody></table>

## aclnnBSASelectBlockMask

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBSASelectBlockMaskGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 该接口若与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- actualSeqLengths在qInputLayout为 "TND" 时必选；actualSeqLengthsKV在kvInputLayout为 "TND" 时必选。
- 根据算子支持的输入Layout，query张量Shape中对应的head维度大小记为N1，key张量Shape中对应的head维度大小记为N2。必须满足N1 = N2（仅支持MHA）。
- headDim = 128。
- blockShapeX和blockShapeY必须为64的倍数。
- query和key压缩后，query和key对应的Xblocks和Yblocks需满足Xblocks - Yblocks > 1。
- query和key的数据类型必须一致，仅支持FLOAT16和BFLOAT16。
- blockSparseMaskOut数据类型为INT8（二值：0或1）。
- postBlockShape当前不支持，必须传入nullptr。
- actualBlockLenQuery / actualBlockLenKey若非null，每个元素取值范围 [0, blockShapeX] / [0, blockShapeY]；为null时完整压缩。
- 不涉及确定性计算。

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
#include "aclnnop/aclnn_bsa_select_block_mask.h"

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
     // 固定写法，资源初始化
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
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

   // 2.构造输入与输出， (以BNSD Layout为例)
    int32_t batch = 1;
    int32_t numHeads = 4;
    int32_t numKHeads = 4;      // 仅支持MHA: N == KN
    int32_t qSeqlen = 256 * 1024; // 256K
    int32_t kSeqlen = 256 * 1024; // 256K
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;

    // 块数量计算
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilK = (kSeqlen + blockShapeY - 1) / blockShapeY;

    // 构建张量Shape
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kShape = {batch, numKHeads, kSeqlen, headDim};
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilK};

    // 分配并初始化Host数据
    int64_t qSize = GetShapeSize(qShape);
    int64_t kSize = GetShapeSize(kShape);

    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kSize, 0.1f);

    // mask输出初始化为0
    std::vector<int8_t> maskOutData(GetShapeSize(maskShape), 0);

    // 创建所有aclTensor
    void *qAddr = nullptr, *kAddr = nullptr, *maskAddr = nullptr;
    aclTensor *qTensor = nullptr, *kTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(maskOutData, maskShape, &maskAddr, aclDataType::ACL_INT8, &maskTensor);

    // 创建aclIntArray属性参数
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    // 标量与字符串参数配置
    char queryLayoutBuffer[16] = "BNSD";
    char keyLayoutBuffer[16] = "BNSD";
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    double sparsity = 0.5;

     // 3.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnBSASelectBlockMask第一段接口
    LOG_PRINT("Calling aclnnBSASelectBlockMaskGetWorkspaceSize...\n");
    ret = aclnnBSASelectBlockMaskGetWorkspaceSize(
        qTensor,
        kTensor,
        blockShapeArr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        queryLayoutBuffer,
        keyLayoutBuffer,
        static_cast<int64_t>(numKHeads),
        scaleValue,
        sparsity,
        maskTensor,
        &workspaceSize,
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnBSASelectBlockMask第二段接口
    LOG_PRINT("Calling aclnnBSASelectBlockMask...\n");
    ret = aclnnBSASelectBlockMask(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMask failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧
    int64_t maskSize = GetShapeSize(maskShape);
    ret = aclrtMemcpy(maskOutData.data(), maskSize * sizeof(int8_t), maskAddr, maskSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! BlockSparseMask output (first 20 elements):\n");
    int64_t displayCount = (maskSize < 20) ? maskSize : 20;
    for (int64_t i = 0; i < displayCount; i++) {
        LOG_PRINT("  mask index %ld: %u\n", i, static_cast<unsigned int>(maskOutData[i]));
    }

    // 6.释放aclTensor和aclIntArray
    LOG_PRINT("Cleaning up resources...\n");
    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(maskTensor);

    aclDestroyIntArray(blockShapeArr);

    // 7.释放device资源
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(maskAddr);
    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BSASelectBlockMask Test completed successfully!\n");
    return 0;
}

```
