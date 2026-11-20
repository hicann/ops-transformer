# aclnnEngramGate

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/950DT系列产品</term> | √ |
| <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term> | × |
| <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 接口功能：执行Engram残差门控前向计算，根据残差特征、Key和融合权重计算门控值，并将共享Value写入残差流。

- 计算公式：参见[EngramGate](../README.md#功能说明)。`dotOutOptional`保存未归一化加权点积`raw_dot`。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnEngramGateGetWorkspaceSize`获取workspace大小和executor，再调用`aclnnEngramGate`执行计算。

```cpp
aclnnStatus aclnnEngramGateGetWorkspaceSize(
    const aclTensor *hiddenStates,
    const aclTensor *kv,
    const aclTensor *weightFused,
    double eps,
    double clampValue,
    bool saveForBackward,
    const aclTensor *outputOut,
    const aclTensor *dotOutOptional,
    const aclTensor *gateScoreOutOptional,
    const aclTensor *rstdXOutOptional,
    const aclTensor *rstdKOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnEngramGate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnEngramGateGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed;width: 1500px">
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
      <td>hiddenStates</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示残差特征。</td>
      <td>T大于等于0，H固定为4，D为64的正整数倍；T为0时支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3（T，H，D）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kv</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示Key和共享Value的融合输入。</td>
      <td>每个Token依次存放H组Key和1组共享Value，数据类型必须与hiddenStates一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2（T，（H+1）×D）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightFused</td>
      <td>输入</td>
      <td>Device侧的aclTensor，表示已融合权重。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2（H，D）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>可选输入</td>
      <td>RMS计算中的防除零参数。</td>
      <td>必须为有限正数（不接受0），默认值为1e-6。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampValue</td>
      <td>可选输入</td>
      <td>带符号开方前的幅值下限。</td>
      <td>必须为有限正数，默认值为1e-6。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>saveForBackward</td>
      <td>可选输入</td>
      <td>是否输出反向计算所需的4个中间结果。</td>
      <td>默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOut</td>
      <td>输出</td>
      <td>Device侧的aclTensor，表示门控残差输出。</td>
      <td>shape和数据类型必须与hiddenStates一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3（T，H，D）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dotOptional（C接口名dotOutOptional）</td>
      <td>可选输出</td>
      <td>Device侧的aclTensor，保存未归一化加权点积raw_dot。</td>
      <td>saveForBackward为true时必须提供；为false时可传nullptr或shape为[0]的空Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2（T，H）或1（0）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gateScoreOptional（C接口名gateScoreOutOptional）</td>
      <td>可选输出</td>
      <td>Device侧的aclTensor，保存门控值。</td>
      <td>传入规则与dotOutOptional一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2（T，H）或1（0）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rstdXOptional（C接口名rstdXOutOptional）</td>
      <td>可选输出</td>
      <td>Device侧的aclTensor，保存hiddenStates沿D轴计算得到的RMS倒数。</td>
      <td>传入规则与dotOutOptional一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2（T，H）或1（0）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rstdKOptional（C接口名rstdKOutOptional）</td>
      <td>可选输出</td>
      <td>Device侧的aclTensor，保存Key沿D轴计算得到的RMS倒数。</td>
      <td>传入规则与dotOutOptional一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2（T，H）或1（0）</td>
      <td>-</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
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

  <table>
  <thead>
    <tr><th>返回值</th><th>错误码</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>ACLNN_ERR_PARAM_NULLPTR</td><td>161001</td><td>必选输入、必选输出、workspaceSize或executor为空指针。</td></tr>
    <tr><td>ACLNN_ERR_INNER_TILING_ERROR</td><td>561002</td><td>Tensor的shape或数据类型不满足约束，属性不是有限正数，或saveForBackward为true时未提供完整的中间结果输出。</td></tr>
  </tbody>
  </table>

## aclnnEngramGate

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  | :--- | :--- | :--- |
  | workspace | 输入 | 在Device侧申请的workspace内存地址。workspaceSize为0时可传nullptr。 |
  | workspaceSize | 输入 | workspace大小，由第一段接口aclnnEngramGateGetWorkspaceSize获取。 |
  | executor | 输入 | op执行器，由第一段接口aclnnEngramGateGetWorkspaceSize获取。 |
  | stream | 输入 | 指定执行任务的Stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- hiddenStates必须为3维Tensor，kv和weightFused必须为2维Tensor。
- T大于等于0，H固定为4，D必须为64的正整数倍。T为0时支持空Tensor；H不等于4或D不满足约束时返回错误。
- hiddenStates、kv和outputOut的数据类型必须一致，只支持FLOAT16和BFLOAT16；weightFused和4个中间结果输出的数据类型必须为FLOAT32。
- 3个输入均支持非连续Tensor，框架执行前会自动转为连续Tensor。
- 不支持输入与输出使用同一块Device内存，不支持原地更新。
- 确定性计算：aclnnEngramGate默认确定性实现。
- NaN和Inf按照浮点计算规则传播；输入乘法或归约发生溢出时，不保证输出为有限值。

## 调用示例

示例代码见[test_aclnn_engram_gate.cpp](../examples/test_aclnn_engram_gate.cpp)，编译与运行过程见[EngramGate调用示例](../examples/README.md)。
