# aclnnBlockAttnResUpdateV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  完成Attention Residuals历史残差注意力计算的阶段二：将`delta`原地累加到`partialBlockRef`，计算更新后残差的RMS score，与历史在线Softmax状态合并得到当前层输出，并支持对该输出融合执行均方根归一化（Root Mean Square Normalization，RMSNorm）。历史状态可取自[aclnnBlockAttnResPrepare](../../block_attn_res_prepare/docs/aclnnBlockAttnResPrepare.md)的某个目标slot。

- 版本演进：

  本接口在[aclnnBlockAttnResUpdate](aclnnBlockAttnResUpdate.md)的基础上，新增可选输入`outputNormWeightOptional`和独立属性`outputNormEps`，用于控制输出RMSNorm。

- 计算公式：

  对每个token $t$，先更新局部残差并原地写回：

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  通过RMSNorm和Query点积得到当前残差的score：

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  将当前残差与历史在线Softmax状态合并：

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H^{(0)}_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  `outputNormWeightOptional`为`nullptr`时，输出 $H=H^{(0)}$；否则对BF16混合结果执行输出RMSNorm：

  $$
  H_t=\operatorname{BF16}\left(
  \operatorname{RMSNorm}\left(\operatorname{FP32}(H^{(0)}_t),\operatorname{FP32}(W),\epsilon_{\mathrm{out}}\right)
  \right)
  $$

  其中：

  - $T$ 和 $D$ 分别表示token数和隐藏维度，$t\in[0,T)$；$P_t$、$p_t$、$\Delta_t$、$O_t$、$H_t$ 和 $Q$ 均为 $D$ 维向量。
  - $P$ 和 $P^{\mathrm{out}}$ 分别表示`partialBlockRef`调用前后的值，$\Delta$ 表示`delta`，$Q$ 表示`pseudoQuery`；$p$ 为FP32残差累加结果。
  - $O$、$M$、$L$ 分别表示历史状态`numerator`、`logitMax`、`expSum`，$H$ 表示输出`h`。$W$ 表示`outputNormWeightOptional`。
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$，$\epsilon_{\mathrm{out}}=\operatorname{FP32}(\texttt{outputNormEps})$，两者分别用于打分和输出归一化。
  - RMSNorm沿 $D$ 维计算，打分时的缩放参数为全1向量；打分所需的缩放参数可预先与原始Query逐元素相乘，合并到 $Q$ 中。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttnResUpdateV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBlockAttnResUpdateV2”接口执行计算。调用时须显式传入`eps`和`outputNormEps`。

```cpp
aclnnStatus aclnnBlockAttnResUpdateV2GetWorkspaceSize(
  aclTensor        *partialBlockRef,
  const aclTensor  *delta,
  const aclTensor  *pseudoQuery,
  const aclTensor  *numerator,
  const aclTensor  *logitMax,
  const aclTensor  *expSum,
  const aclTensor  *outputNormWeightOptional,
  double            eps,
  double            outputNormEps,
  aclTensor        *h,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResUpdateV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBlockAttnResUpdateV2GetWorkspaceSize

- **参数说明**

  所有Tensor参数的原始格式和存储格式均仅支持ND，且必须为连续Tensor。

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 100px">
      <col style="width: 230px">
      <col style="width: 330px">
      <col style="width: 160px">
      <col style="width: 90px">
      <col style="width: 150px">
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
        <td style="white-space: nowrap;">partialBlockRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>已累计的局部残差，执行后原地保存残差累加结果，对应公式中的P和p。</td>
        <td>
          <ul>
            <li>T &gt;= 0，0 &lt;= D &lt;= 8192。</li>
            <li>支持空Tensor，仍须满足各输入输出之间的shape关系。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">delta（const aclTensor*）</td>
        <td>输入</td>
        <td>本次需要累加到局部残差的增量，对应公式中的Δ。</td>
        <td>
          <ul>
            <li>shape必须与partialBlockRef相同。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">pseudoQuery（const aclTensor*）</td>
        <td>输入</td>
        <td>用于计算当前残差归一化score的有效Query，对应公式中的Q。</td>
        <td>
          <ul>
            <li>长度必须等于partialBlockRef的D。</li>
            <li>D为0时支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">numerator（const aclTensor*）</td>
        <td>输入</td>
        <td>历史在线Softmax加权分子，对应公式中的O；本接口只读。</td>
        <td>
          <ul>
            <li>shape必须与partialBlockRef相同。</li>
            <li>支持空Tensor；非空历史状态的数值要求见<a href="#约束说明">约束说明</a>。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">logitMax（const aclTensor*）</td>
        <td>输入</td>
        <td>每个token的历史最大logit，对应公式中的M；本接口只读。</td>
        <td>
          <ul>
            <li>长度必须等于partialBlockRef的T。</li>
            <li>T为0时支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">expSum（const aclTensor*）</td>
        <td>输入</td>
        <td>每个token的历史softmax指数和，对应公式中的L；本接口只读。</td>
        <td>
          <ul>
            <li>长度必须等于partialBlockRef的T。</li>
            <li>T为0时支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>outputNormWeightOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>下一子层RMSNorm的gamma，对应公式中的W。</td>
        <td>
          <ul>
            <li>传入nullptr时禁用输出RMSNorm。</li>
            <li>非空指针时启用输出RMSNorm，长度必须等于D。</li>
            <li>D为0时可传shape为[0]的空Tensor；D大于0时不能用空Tensor代替nullptr。</li>
          </ul>
        </td>
        <td>BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">eps（double）</td>
        <td>输入</td>
        <td>当前残差RMS score计算的数值稳定项，对应公式中的ε。</td>
        <td>必须为有限正数。非空计算时，转换为FLOAT32后的值也必须有限且大于0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">outputNormEps（double）</td>
        <td>输入</td>
        <td>输出RMSNorm的独立数值稳定项，对应公式中的ε_out。</td>
        <td>outputNormWeightOptional非空时，转换为FLOAT32前的值必须在<code>[FLT_MIN, FLT_MAX]</code>范围内；outputNormWeightOptional为nullptr时忽略本参数，不校验其取值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">h（aclTensor*）</td>
        <td>输出</td>
        <td>当前层输出，对应公式中的H；启用输出RMSNorm时为归一化后的结果。</td>
        <td>
          <ul>
            <li>shape必须与partialBlockRef相同。</li>
            <li>T为0或D为0时返回对应shape的空Tensor。</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小，单位为字节。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">executor（aclOpExecutor**）</td>
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

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码/状态码</th>
        <th>码值</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap;">ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>partialBlockRef、delta、pseudoQuery、numerator、logitMax、expSum、h、workspaceSize或executor为空指针。outputNormWeightOptional允许为空指针。</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>
          <ul>
            <li>必选Tensor或非空outputNormWeightOptional的数据类型、原始格式或存储格式不符合参数说明。</li>
            <li>Tensor维度数或各参数之间的shape关系不符合参数说明，包括非空outputNormWeightOptional必须为[D]。</li>
            <li>T小于0、D不在[0, 8192]内，或非空场景T * D超出int64_t可表示范围。</li>
            <li>必选Tensor或非空outputNormWeightOptional为非连续Tensor。</li>
            <li>eps不是有限正数。</li>
            <li>outputNormWeightOptional非空时，outputNormEps转换为FLOAT32前的值不在<code>[FLT_MIN, FLT_MAX]</code>范围内。</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>当前运行环境不在接口支持的产品范围内。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建算子执行器失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>创建Tensor视图、算子计算节点或输出拷贝节点失败。</td>
      </tr>
    </tbody>
  </table>

## aclnnBlockAttnResUpdateV2

- **参数说明**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
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
        <td style="white-space: nowrap;">workspace（void*）</td>
        <td>输入</td>
        <td>Device侧workspace内存地址。workspaceSize为0时可传入<code>nullptr</code>。</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">workspaceSize（uint64_t）</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttnResUpdateV2GetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">executor（aclOpExecutor*）</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">stream（aclrtStream）</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
    </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- **组合约束**：`partialBlockRef`、`delta`、`numerator`、`h`的shape均为`[T, D]`；`pseudoQuery`和非空`outputNormWeightOptional`为`[D]`；`logitMax`、`expSum`为`[T]`。
- **空Tensor**：`T == 0`或`D == 0`时，完成第一段接口的参数校验后返回对应shape的空输出，不启动计算Kernel。必选Tensor不能用`nullptr`代替；传入权重时仍需满足`[D]`、dtype、ND格式和连续性约束，`outputNormEps`仍会校验。
- **非空计算范围**：`T > 0`，`1 <= D <= 8192`，`T * D`不能超出`int64_t`可表示范围；按核切分后的T大小须能由`uint32_t`表示。完整D维数据及可选权重必须满足统一缓冲区（Unified Buffer，UB）预算，否则Tiling失败。
- **历史状态**：非空计算时，调用者须保证每个token的历史集合非空，`numerator`和`logitMax`中的元素有限，`expSum`中的元素有限且严格大于0，并且三者来自同一份历史在线Softmax状态。本接口不更新这三个输入，也不读取其元素值进行Host校验；不支持以`logitMax=-inf`、`expSum=0`和零`numerator`表示的空历史状态。
- **输出归一化epsilon**：启用输出RMSNorm时，在转换为FLOAT32前校验`outputNormEps`位于`[FLT_MIN, FLT_MAX]`范围内；`FLT_MIN = 2^-126 ≈ 1.1754943508222875e-38`，`FLT_MAX ≈ 3.4028234663852886e38`。默认Sqrt采用FTZ模式，不能使用次正规epsilon保证全零混合结果的分母非零。
- **精度与数值**：`eps`和启用时的`outputNormEps`转换为FLOAT32后参与计算，二者用途独立。输出RMSNorm使用已经舍入为BFLOAT16的混合结果，不省略该舍入过程。输入元素中的零值、极值、NaN或Inf不做额外处理，平方、乘法或累加溢出时可能产生非有限结果。
- **确定性计算**：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnBlockAttnResUpdateV2默认确定性实现。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

完整示例源码见[test_aclnn_block_attn_res_update_v2.cpp](../examples/arch35/test_aclnn_block_attn_res_update_v2.cpp)。示例使用FLOAT32类型的`outputNormWeightOptional`，默认启用输出RMSNorm，`eps`设为`1e-6`，`outputNormEps`设为`1e-5`。运行可执行文件时添加`--no-output-norm`，则通过传入`nullptr`演示禁用输出RMSNorm。

在仓库根目录编译并运行示例：

```bash
bash build.sh --run_example block_attn_res_update eager --soc=ascend950 \
    --example_name=test_aclnn_block_attn_res_update_v2
```

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_update_v2.h"

#define CHECK_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            returnExpr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
            returnExpr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr size_t FIRST_ELEMENT_INDEX = 0UL;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int RunBlockAttnResUpdateV2(int32_t deviceId, aclrtStream& stream, bool applyOutputNorm)
{
    // 1. 初始化Device和Stream。
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入、输出及RMSNorm权重数据。
    constexpr int64_t tokenNum = 2;
    constexpr int64_t hiddenSize = 64;
    const std::vector<int64_t> matrixShape = {tokenNum, hiddenSize};
    const std::vector<int64_t> queryShape = {hiddenSize};
    const std::vector<int64_t> statsShape = {tokenNum};

    std::vector<float> partialBlockRefHostData(GetShapeSize(matrixShape), 0.25F);
    std::vector<uint16_t> deltaHostData(GetShapeSize(matrixShape), FloatToBfloat16(0.125F));
    std::vector<float> pseudoQueryHostData(GetShapeSize(queryShape), 1.0F / static_cast<float>(hiddenSize));
    std::vector<float> numeratorHostData(GetShapeSize(matrixShape), 0.5F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 1.0F);
    std::vector<uint16_t> hHostData(GetShapeSize(matrixShape), 0U);
    std::vector<float> outputNormWeightHostData(hiddenSize);
    for (int64_t d = 0; d < hiddenSize; ++d) {
        outputNormWeightHostData[d] = 0.5F + static_cast<float>(d % 8) * 0.125F;
    }

    void* partialBlockRefDeviceAddr = nullptr;
    void* deltaDeviceAddr = nullptr;
    void* pseudoQueryDeviceAddr = nullptr;
    void* numeratorDeviceAddr = nullptr;
    void* logitMaxDeviceAddr = nullptr;
    void* expSumDeviceAddr = nullptr;
    void* hDeviceAddr = nullptr;

    aclTensor* partialBlockRef = nullptr;
    aclTensor* delta = nullptr;
    aclTensor* pseudoQuery = nullptr;
    aclTensor* numerator = nullptr;
    aclTensor* logitMax = nullptr;
    aclTensor* expSum = nullptr;
    aclTensor* h = nullptr;

    // 3. 创建连续ND格式的输入、输出Tensor，使用RAII管理资源。
    ret = CreateAclTensor<float>(partialBlockRefHostData, matrixShape, &partialBlockRefDeviceAddr,
                                 aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &partialBlockRef);
    std::unique_ptr<void, aclError (*)(void*)> partialBlockRefDeviceAddrPtr(partialBlockRefDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> partialBlockRefTensorPtr(partialBlockRef,
                                                                                           aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(deltaHostData, matrixShape, &deltaDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &delta);
    std::unique_ptr<void, aclError (*)(void*)> deltaDeviceAddrPtr(deltaDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> deltaTensorPtr(delta, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(pseudoQueryHostData, queryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &pseudoQuery);
    std::unique_ptr<void, aclError (*)(void*)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(numeratorHostData, matrixShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &numerator);
    std::unique_ptr<void, aclError (*)(void*)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> numeratorTensorPtr(numerator, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &logitMax);
    std::unique_ptr<void, aclError (*)(void*)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &expSum);
    std::unique_ptr<void, aclError (*)(void*)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> expSumTensorPtr(expSum, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(hHostData, matrixShape, &hDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &h);
    std::unique_ptr<void, aclError (*)(void*)> hDeviceAddrPtr(hDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> hTensorPtr(h, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. 两个epsilon独立设置；禁用输出RMSNorm时权重保持nullptr。
    constexpr double eps = 1.0e-6;
    constexpr double outputNormEps = 1.0e-5;
    void* outputNormWeightDeviceAddr = nullptr;
    aclTensor* outputNormWeightOptional = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> outputNormWeightDevicePtr(nullptr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputNormWeightTensorPtr(nullptr, aclDestroyTensor);
    if (applyOutputNorm) {
        ret = CreateAclTensor<float>(outputNormWeightHostData, queryShape, &outputNormWeightDeviceAddr,
                                     aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputNormWeightOptional);
        outputNormWeightDevicePtr.reset(outputNormWeightDeviceAddr);
        outputNormWeightTensorPtr.reset(outputNormWeightOptional);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // 5. 调用第一段接口，获取workspace大小和执行器。
    ret = aclnnBlockAttnResUpdateV2GetWorkspaceSize(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum,
                                                    outputNormWeightOptional, eps, outputNormEps, h, &workspaceSize,
                                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. 根据第一段接口的返回值申请workspace。
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 7. 调用第二段接口并等待任务执行完成。
    ret = aclnnBlockAttnResUpdateV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdateV2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 8. 将原地残差更新结果和BF16输出拷贝回Host侧。
    const size_t partialBlockRefSize = partialBlockRefHostData.size() * sizeof(float);
    ret = aclrtMemcpy(partialBlockRefHostData.data(), partialBlockRefSize, partialBlockRefDeviceAddr,
                      partialBlockRefSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy partialBlockRef to host failed. ERROR: %d\n", ret); return ret);

    const size_t hSize = hHostData.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(hHostData.data(), hSize, hDeviceAddr, hSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("partialBlockRef[0] after in-place update: %.6f\n", partialBlockRefHostData[FIRST_ELEMENT_INDEX]);
    LOG_PRINT("h[0]: %.6f\n", Bfloat16ToFloat(hHostData[FIRST_ELEMENT_INDEX]));
    LOG_PRINT("Output RMSNorm: %s\n", applyOutputNorm ? "enabled" : "disabled");
    LOG_PRINT("block_attn_res_update_v2 example execute success.\n");
    return ACL_SUCCESS;
}

int main(int argc, char** argv)
{
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--no-output-norm") != 0)) {
        LOG_PRINT("Usage: %s [--no-output-norm]\n", argv[0]);
        return 1;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunBlockAttnResUpdateV2(deviceId, stream, argc == 1);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBlockAttnResUpdateV2 failed. ERROR: %d\n", ret); return ret);

    // Tensor和Device内存已在Run函数退出时释放，再释放Stream和Device资源。
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
