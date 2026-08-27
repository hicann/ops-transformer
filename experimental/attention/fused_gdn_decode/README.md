# FusedGdnDecode

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：融合门控Delta网络（Gated Delta Network，GDN）单token解码中的QKV拆分、Q/K归一化、门控计算、循环状态更新和输出投影。
- 计算公式：

  $$
  q_h = scale \times \frac{q_h}{\sqrt{\sum_i q_{h,i}^2+\epsilon}}, \quad
  k_h = \frac{k_h}{\sqrt{\sum_i k_{h,i}^2+\epsilon}}
  $$

  $$
  g_j = -\exp(A_j)\times softplus(a_j+dtBias_j), \quad
  \beta_j = sigmoid(b_j)
  $$

  $$
  S_j = \exp(g_j)S_j+\beta_j(v_j-\exp(g_j)S_jk_h)k_h^T, \quad
  out_j = S_jq_h
  $$

  其中，$h=\lfloor j/(H_v/H)\rfloor$，$S_j\in R^{V\times K}$。`stateRef`原地保存更新后的状态。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| mixedQkv | 输入 | 按Q、K、V顺序拼接的输入，shape为`(B, 2*H*K+Hv*V)`。 | BFLOAT16、FLOAT16 | ND |
| a | 输入 | 门控输入a，shape为`(B, Hv)`。 | 与mixedQkv一致 | ND |
| b | 输入 | 门控输入b，shape为`(B, Hv)`。 | 与mixedQkv一致 | ND |
| aLog | 输入 | 门控参数A，shape为`(Hv,)`。 | FLOAT32 | ND |
| dtBias | 输入 | softplus偏置，shape为`(Hv,)`。 | 与mixedQkv一致 | ND |
| stateRef | 输入&输出 | 循环状态，shape为`(BlockNum, Hv, V, K)`，原地更新。 | FLOAT32或与mixedQkv一致 | ND |
| ssmStateIndices | 输入 | batch到stateRef槽位的映射，shape为`(B,)`。 | INT32 | ND |
| scale | 属性 | Q归一化后的缩放系数，默认值为1.0。 | FLOAT | - |
| softplusThreshold | 属性 | softplus阈值，默认值为20.0。 | FLOAT | - |
| out | 输出 | GDN输出，shape为`(B, 1, Hv, V)`。 | 与mixedQkv一致 | ND |

## 约束说明

- `64 <= K <= 2032`，`Hv % H == 0`，所有维度均为正数。
- stateRef为FLOAT32时，需满足`ceil(K/16)*16-K <= 8`。
- `stateRef`必须为连续Tensor；其他Tensor支持非连续输入。
- `ssmStateIndices[i] <= 0`表示无效槽位，对应输出为0且`stateRef`不更新。
- `ssmStateIndices[i] > 0`时，用户需保证`ssmStateIndices[i] < BlockNum`，且同一批次内的正索引互不重复。
- aclnnFusedGdnDecode默认确定性实现。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_fused_gdn_decode.cpp](./examples/test_aclnn_fused_gdn_decode.cpp) | 通过[aclnnFusedGdnDecode](./docs/aclnnFusedGdnDecode.md)调用FusedGdnDecode算子。 |
| torch接口 | [torch_ops_extension](./torch_ops_extension/README.md) | 安装torch_ops_extension后，通过`torch.ops.custom.npu_fused_gdn_decode`调用FusedGdnDecode算子。 |
