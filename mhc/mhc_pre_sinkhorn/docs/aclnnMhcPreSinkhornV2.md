# aclnnMhcPreSinkhornV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- 接口功能：基于一系列计算得到MHC架构中hidden层的$\mathbf{H'_{res}}$和$\mathbf{H_{post}}$投影矩阵以及Attention或MLP层的输入矩阵$h_{in}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。包括sigmoid计算之后的$\mathbf{H^{pre}_l}$矩阵、$\mathbf{\vec{x^{'}_{l}}}$与$\mathbf{\varphi}$矩阵乘的结果，输入x的RmsNorm结果$\mathbf{\vec{x^{'}_{l}}}$、迭代过程中的中间归一化结果$normOut$和求和结果$sumOut$。其中，该接口在Cube中默认使用HF32计算模式（与premix能力合入前一致）。

  <!-- npu="A3,910b" id12 -->
  <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：可通过可选属性`enableHf32`控制，传入false时切换为FP32计算模式（默认true）。
  <!-- end id12 -->

- 计算公式

  $$
  \begin{aligned}
  \vec{x^{'}_{l}} &= \frac{1}{\sqrt{\frac{1}{d} \sum_{\dim=-2,\text{keepdim}=\text{True}} x_i^2 + \epsilon}}\\
  H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
  H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
  H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
  H^{pre}_l &= \sigma (H^{pre}_{l})\\
  H^{post}_l &= 2\sigma (H^{post}_{l})\\
  h_{in} &=\vec{x_{l}}H^{pre}_l
  \end{aligned}
  $$

  当传入可选输入premix时，$h_{in}$的加权系数使用$premix$替换$\mathbf{H^{pre}_l}$（$premix$通常填上一层输出的hPre；其余公式不变：

  $$
  h_{in} =\vec{x_{l}}{premix}_{l}
  $$

  - 将$\mathbf{H^{res}_l}$作为输入，Sinkhorn变换共执行$numIters$次迭代，迭代过程中生成中间归一化结果${normOut}[k]$和求和结果${sumOut}[k]$，最终输出最后一次迭代的$normOut$作为变换结果。

    第一次迭代（初始化）：

    $$
    \begin{aligned}
        {normOut}[0] &= \text{softmax}({H^{res}_l},  \dim=-1) + \epsilon, \\
        {sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} {normOut}[0] + \epsilon, \\
        {normOut}[1] &= \frac{{normOut}[0]}{{sumOut}[1]}, \\
    \end{aligned}
    $$

    第$i$次迭代（$i = 1, 2, \dots, ({numIters}-1)$）：

    $$
    \begin{aligned}
        {sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} {normOut}[2i-1] + \epsilon, \\
        {normOut}[2i] &= \frac{{normOut}[2i-1]}{{sumOut}[2i]}, \\
        {sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} {normOut}[2i] + \epsilon, \\
        {normOut}[2i+1] &= \frac{{normOut}[2i]}{{sumOut}[2i+1]}, \\
    \end{aligned}
    $$

  - 最终输出

    $$
    {normOut}[2 \times {numIters} - 1]
    $$

    $$
    {sumOut}[2 \times {numIters} - 1]
    $$

  - 符号说明

    | 符号                                       | 含义                                              |
    | ------------------------------------------ | ------------------------------------------------- |
    | $x$                               | 输入张量（mHC层的$\mathbf{H}'_{\text{res}}$矩阵） |
    | $\epsilon$                                 | 防除零参数（对应入参`eps`）                       |
    | $\text{softmax}(\cdot, \dim=-1)$           | 在最后一维执行softmax归一化                       |
    | $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | 在指定维度$d$上求和并保持维度                     |
    | ${normOut}[k]$                      | 第$k$步归一化中间结果                             |
    | ${sumOut}[k]$                       | 第$k$步求和中间结果                               |
    | ${numIters}$                        | 迭代次数（入参）                                  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnMhcPreSinkhornV2GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMhcPreSinkhornV2`接口执行计算。

```Cpp
aclnnStatus aclnnMhcPreSinkhornV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    const aclTensor *premix,
    int64_t          hcMult,
    int64_t          numIters,
    double           hcEps,
    double           normEps,
    bool             needBackward,
    bool             enableHf32,
    aclTensor       *hin,
    aclTensor       *hPost,
    aclTensor       *hRes,
    aclTensor       *hPre,
    aclTensor       *hcBeforeNorm,
    aclTensor       *invRms,
    aclTensor       *sumOut,
    aclTensor       *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMhcPreSinkhornV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMhcPreSinkhornV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1435px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 250px">
    <col style="width: 100px">
    <col style="width: 120px">
    <col style="width: 200px">
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
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>待计算数据，表示网络中mHC层的输入数据。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持(bs, seq_len, n, c)或者(t, n, c)。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>phi（const aclTensor*）</td>
      <td>输入</td>
      <td>mHC的参数矩阵。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持(n * n + 2 * n, n * c)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha（const aclTensor*）</td>
      <td>输入</td>
      <td>mHC的缩放参数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持(3)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias（const aclTensor*）</td>
      <td>输入</td>
      <td>mHC的bias参数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持(n * n + 2 * n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>premix（const aclTensor*）</td>
      <td>输入</td>
      <td>可选输入，premix加权系数（通常填上一层输出的hPre）。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持(bs, seq_len, n)或者(t, n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hcMult（int64_t）</td>
      <td>输入</td>
      <td>残差流数量，HC维度大小。</td>
      <td>当前仅支持4。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numIters（int64_t）</td>
      <td>输入</td>
      <td>表示sinkhorn算法迭代次数。</td>
      <td>当前仅支持20。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hcEps（double）</td>
      <td>输入</td>
      <td>H<sub>pre</sub>的sigmoid后的eps参数。</td>
      <td>建议值：1e-6。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normEps（double）</td>
      <td>输入</td>
      <td>RmsNorm的防除零参数。</td>
      <td>建议值：1e-6。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needBackward（bool）</td>
      <td>输入</td>
      <td>是否需要输出额外属性。</td>
      <td>建议值为true。</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>enableHf32（bool）</td>
      <td>输入</td>
      <td>可选属性（仅V2接口），表示Cube是否启用HF32计算模式。</td>
      <td>可为true（默认，与V1接口一致）。传入false切换为FP32。</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hin（aclTensor*）</td>
      <td>输出</td>
      <td>输出的hin作为Atten/MLP层的输入。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape支持(bs, seq_len, c)、(t, c)。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>hPost（aclTensor*）</td>
      <td>输出</td>
      <td>输出的mHC的hPost变换矩阵。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape支持(bs, seq_len, n)、(t, n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>hRes（aclTensor*）</td>
      <td>输出</td>
      <td>输出的mHC的hRes变换矩阵。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape支持(bs, seq_len, n * n)、(t, n * n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>hPre（aclTensor*）</td>
      <td>输出</td>
      <td>需要反向时输出，做完sigmoid计算之后的hPre矩阵。</td>
      <td><ul><li>不支持空Tensor。</li><li>当needBackward为false时，此输出无效；传入premix时即使needBackward为false也输出有效值。</li><li>shape支持(bs, seq_len, n)、(t, n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>hcBeforeNorm（aclTensor*）</td>
      <td>输出</td>
      <td>需要反向时输出，x与phi矩阵乘的结果。</td>
      <td><ul><li>不支持空Tensor。</li><li>当needBackward为false时，此输出无效。</li><li>shape支持(bs, seq_len, n*n + 2*n)、(t, n*n + 2*n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>invRms（aclTensor*）</td>
      <td>输出</td>
      <td>需要反向时输出，RmsNorm计算得到的1/r。</td>
      <td><ul><li>不支持空Tensor。</li><li>当needBackward为false时，此输出无效。</li><li>shape支持(bs, seq_len, 1)、(t, 1)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sumOut（aclTensor*）</td>
      <td>输出</td>
      <td>需要反向时输出，每一次迭代的colSum/rowSum结果。</td>
      <td><ul><li>不支持空Tensor。</li><li>当needBackward为false时，此输出无效。</li><li>shape支持(sk_iter_count * 2, bs, seq_len, n)、(sk_iter_count * 2, t, n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>normOut（aclTensor*）</td>
      <td>输出</td>
      <td>需要反向时输出，每一次colSum/rowSum迭代后的comb结果。</td>
      <td><ul><li>不支持空Tensor。</li><li>当needBackward为false时，此输出无效。</li><li>shape支持(sk_iter_count * 2, bs, seq_len, n, n)、(sk_iter_count * 2, t, n, n)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>4-5</td>
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

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT系列产品</term>：参数`enableHf32`在当前产品使用场景下无效。
  <!-- end id11 -->

  > 说明：
  >
  > 4维输入为BSND格式，3维输入为TND格式。其中bs表示Batch大小，seq_len表示单Batch序列长度，t表示所有Batch序列长度的累加和，n表示残差流数量，c表示尾轴隐藏维度大小。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
    <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>输入变量的数据类型或shape维度不在支持的范围内。</td>
  </tr>
  <tr>
    <td>numIters不为20。</td>
  </tr>
  <tr>
    <td>hcMult的值非4。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>调用NPU Runtime接口申请内存/创建Tensor失败。</td>
  </tr>
  </tbody>
  </table>

## aclnnMhcPreSinkhornV2

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreSinkhornV2GetWorkspaceSize获取。</td>
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

- 确定性计算：

  - aclnnMhcPreSinkhornV2默认确定性实现。

- Batch一致性说明：
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 默认非Batch一致性实现，支持通过aclrtSetSysParamOpt开启Batch一致性。开启Batch一致性后性能可能会有一定程度的劣化。
  <!-- end id8 -->

- 规格约束

  | 规格项        | 规格 | 规格说明                              |
  | :------------ | :--- | :------------------------------------ |
  | numIters      | 20   | 目前只支持20。                        |
  | hcMult        | 4    | 目前只支持4。                         |

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 输入x仅支持4维BSND格式，shape为(bs, seq_len, n, c)。
    - 尾轴c需为128的倍数且取值范围为[1, 100000]。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 输入x支持4维BSND格式和3维TND格式，shape分别为(bs, seq_len, n, c)和(t, n, c)。
    - TND格式下，t表示所有Batch序列长度的累加和；对应输出shape按参数说明中的t轴展开。
    - 尾轴c需为128的倍数且取值范围为[1, 100000]。
    - 输入phi的数据范围限定在$\pm \frac{1}{\sqrt{nc}}$范围内，eps限定在1e-6，此范围内具有较好的数值稳定性。
  <!-- end id10 -->

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

void PrintTensorDataFloat(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

void PrintTensorDataBfloat16(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_bf16(size);

    aclError ret = aclrtMemcpy(host_bf16.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements, BF16 hex): ");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), size); ++i) {
        LOG_PRINT("0x%04x ", host_bf16[i]);
    }
    LOG_PRINT("\n");
}

int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

int CreateAclTensorFloat32(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                           aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                            aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        // fp32 -> bf16 标准舍入: 取高16位(符号1+指数8+尾数7), 低位四舍五入;
        // 不可用 x*32767 截断近似, 会改变数值幅值
        uint32_t bits = 0;
        std::memcpy(&bits, &host_data[i], sizeof(bits));
        uint32_t lsb = (bits >> 16) & 1U;
        bits = ((bits + (0x7FFFU + lsb)) >> 16) & 0xFFFFU;
        host_data_bf16[i] = static_cast<uint16_t>(bits);
    }

    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_bf16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct Tensors {
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void* premix_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor* premix = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
};

int CreateInputTensors(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& phi_shape,
                       const std::vector<int64_t>& alpha_shape, const std::vector<int64_t>& bias_shape,
                       const std::vector<int64_t>& premix_shape, Tensors& tensors)
{
    // 温和伪随机数据: 全 1.0 会使 x*phi 矩阵乘溢出 sigmoid 饱和, sinkhorn 输出 NaN。
    // 注意用有符号 int64_t 循环变量: size_t 下 (mod - 9) 会无符号下溢成极大值
    std::vector<float> x_host_data(GetShapeSize(x_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(x_host_data.size()); ++i) {
        x_host_data[i] = 0.5f * static_cast<float>((i * 37 % 19) - 9) / 9.0f;
    }
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 0.0f);
    float phi_scale = 16.0f * static_cast<float>(phi_shape[1]); // 16 * (N*C)
    for (int64_t i = 0; i < static_cast<int64_t>(phi_host_data.size()); ++i) {
        phi_host_data[i] = static_cast<float>((i * 91 % 17) - 8) / phi_scale;
    }
    std::vector<float> alpha_host_data = {0.3f, 0.3f, 0.3f};
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(bias_host_data.size()); ++i) {
        bias_host_data[i] = 0.05f * static_cast<float>((i * 13 % 7) - 3);
    }
    std::vector<float> premix_host_data(GetShapeSize(premix_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(premix_host_data.size()); ++i) {
        premix_host_data[i] = 0.5f + static_cast<float>(i * 11 % 10) / 20.0f; // [0.5, 0.95]
    }

    int ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(premix_host_data, premix_shape, tensors.premix_addr, tensors.premix);
    CHECK_RET(ret == 0, LOG_PRINT("Create premix_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t>& hin_shape, const std::vector<int64_t>& h_post_shape,
                        const std::vector<int64_t>& h_res_shape, const std::vector<int64_t>& h_pre_shape,
                        const std::vector<int64_t>& hc_before_norm_shape, const std::vector<int64_t>& inv_rms_shape,
                        const std::vector<int64_t>& sum_out_shape, const std::vector<int64_t>& norm_out_shape,
                        Tensors& tensors)
{
    int ret = CreateAclTensorBfloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(hc_before_norm_shape, tensors.hc_before_norm_addr, tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors& tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.premix);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
}

void FreeDeviceMemory(Tensors& tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.premix_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;

    int64_t bs = 2, seq_len = 128, n = 4, c = 4096;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    float hc_eps = 1e-6f, norm_eps = 1e-6f;
    bool need_backward = true;
    bool enable_hf32 = true;

    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};
    std::vector<int64_t> premix_shape = {bs, seq_len, n};

    std::vector<int64_t> hin_shape = {bs, seq_len, c};
    std::vector<int64_t> h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> h_res_shape = {bs, seq_len, n * n};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    ret = CreateInputTensors(x_shape, phi_shape, alpha_shape, bias_shape, premix_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, h_pre_shape, hc_before_norm_shape, inv_rms_shape,
                              sum_out_shape, norm_out_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // V2 接口: 5 张量输入(含可选 premix) + 6 attr(含 enable_hf32), 8 输出。
    // premix 非空时 hin 使用 premix 加权(替换内部 sinkhorn 权重 h_pre);
    // premix 为 nullptr 时与 V1 行为一致(走 nullptr 即可, 本例展示 premix 用法)。
    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornV2GetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias, tensors.premix, hc_mult, num_iters, hc_eps, norm_eps,
        need_backward, enable_hf32, tensors.hin, tensors.h_post, tensors.h_res, tensors.h_pre, tensors.hc_before_norm,
        tensors.inv_rms, tensors.sum_out, tensors.norm_out, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreSinkhornV2GetWorkspaceSize failed, error: %d\n", aclnn_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        aclError acl_ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed, error: %d\n", acl_ret);
                  FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhornV2(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornV2 failed, error: %d\n", aclnn_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    aclError acl_ret = aclrtSynchronizeStream(stream);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", acl_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    PrintTensorDataBfloat16(hin_shape, tensors.hin_addr);
    PrintTensorDataFloat(h_post_shape, tensors.h_post_addr);
    PrintTensorDataFloat(h_res_shape, tensors.h_res_addr);
    PrintTensorDataFloat(h_pre_shape, tensors.h_pre_addr);

    if (workspace_size > 0) {
        (void)aclrtFree(workspace_addr);
    }
    FreeDeviceMemory(tensors);
    DestroyTensors(tensors);

    acl_ret = aclrtDestroyStream(stream);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("destroy stream failed, error: %d\n", acl_ret); return -1);
    acl_ret = aclrtDestroyContext(context);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("destroy context failed, error: %d\n", acl_ret); return -1);
    (void)aclFinalize();

    return 0;
}
```
