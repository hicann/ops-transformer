# EngramGate

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

- 算子功能：EngramGate用于Engram网络中的残差门控。算子根据当前残差特征、Key和融合权重计算门控值，再将各Head共享的Value按门控值写入残差流；可选输出反向计算所需的中间结果。

- 计算公式：

  `hidden_states`的shape为`[T, H, D]`。对于第`t`个Token、第`h`个Head：

  - $x=hidden\_states[t,h,:]$。
  - $k=kv[t,hD:(h+1)D]$。
  - $v=kv[t,HD:(H+1)D]$，同一Token的全部Head共享该Value。
  - $w=weight\_fused[h,:]$。

  对应输出按下式计算：

  $$
  rstd_x=\frac{1}{\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}x_d^2+eps}},
  \qquad
  rstd_k=\frac{1}{\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}k_d^2+eps}}
  $$

  $$
  raw\_dot=\sum_{d=0}^{D-1}x_dw_dk_d,
  \qquad
  normalized\_dot=\frac{raw\_dot\cdot rstd_x\cdot rstd_k}{\sqrt{D}}
  $$

  $$
  signed\_root=\operatorname{copysign}
  \left(\sqrt{\max\left(\left|normalized\_dot\right|,clamp\_value\right)},normalized\_dot\right)
  $$

  $$
  gate\_score=\frac{1}{1+\exp(-signed\_root)},
  \qquad
  output[t,h,d]=\operatorname{cast}_{hidden\_states.dtype}
  \left(x_d+gate\_score\cdot v_d\right)
  $$

  `dot`输出保存公式中的`raw_dot`，不保存`normalized_dot`。

## 参数说明

本文使用以下维度符号：

- T：Token数量。
- H：Head数量。
- D：每个Head的特征维度。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| hidden_states | 输入 | 残差特征，shape为`[T, H, D]`。 | FLOAT16、BFLOAT16 | ND |
| kv | 输入 | Key和共享Value的融合输入，shape为`[T, (H+1)×D]`。每个Token依次存放H组Key和1组共享Value。 | 数据类型与`hidden_states`一致 | ND |
| weight_fused | 输入 | 已融合权重，shape为`[H, D]`。 | FLOAT32 | ND |
| eps | 可选属性 | RMS计算中的防除零参数，必须为有限正数，默认值为`1e-6`。 | FLOAT | - |
| clamp_value | 可选属性 | 带符号开方前的幅值下限，必须为有限正数，默认值为`1e-6`。 | FLOAT | - |
| save_for_backward | 可选属性 | 是否输出反向计算所需的4个中间结果，默认值为`true`。 | BOOL | - |
| output | 输出 | 门控残差输出，shape和数据类型与`hidden_states`一致。 | FLOAT16、BFLOAT16 | ND |
| dot | 可选输出 | 未归一化加权点积`raw_dot`。`save_for_backward=true`时shape为`[T, H]`，否则为空Tensor。 | FLOAT32 | ND |
| gate_score | 可选输出 | 公式中的门控值，shape规则与`dot`一致。 | FLOAT32 | ND |
| rstd_x | 可选输出 | `hidden_states`沿D轴计算得到的RMS倒数，shape规则与`dot`一致。 | FLOAT32 | ND |
| rstd_k | 可选输出 | Key沿D轴计算得到的RMS倒数，shape规则与`dot`一致。 | FLOAT32 | ND |

## 约束说明

- `hidden_states`必须为3维Tensor，`kv`和`weight_fused`必须为2维Tensor。
- T大于等于0，H固定为4，D必须为64的正整数倍。T为0时支持空Tensor；H不等于4或D不满足约束时返回错误。
- `hidden_states`、`kv`和`output`的数据类型必须一致，只支持FLOAT16和BFLOAT16；`weight_fused`及4个中间结果输出的数据类型必须为FLOAT32。
- aclnn接口支持非连续输入，框架执行前会将输入转为连续Tensor；PyTorch接口要求3个输入均为连续Tensor。
- 不支持输入与输出使用同一块Device内存，不支持原地更新。
- 算子内部使用FLOAT32进行乘法、归约和门控计算，`output`转换为输入数据类型。
- 确定性计算：aclnnEngramGate默认确定性实现，不使用跨核归约或原子累加。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_engram_gate.cpp](./examples/test_aclnn_engram_gate.cpp) | 通过[aclnnEngramGate](./docs/aclnnEngramGate.md)调用算子。 |
| GE图模式 | [test_geir_engram_gate.cpp](./examples/test_geir_engram_gate.cpp) | 通过EngramGate GE IR原型构图并执行。 |

样例的编译与运行方式见[EngramGate调用示例](./examples/README.md)。
