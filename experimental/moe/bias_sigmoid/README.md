# 算子名称：BiasSigmoid

## 产品支持情况

| 产品                                                       | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                         | 是       |

## 功能说明

- 算子功能：对输入张量进行Sigmoid激活计算，并叠加偏置（Bias）。该算子同时输出两个结果：一个是纯粹的Sigmoid激活结果，另一个是Sigmoid叠加Bias后的结果。

- 该算子对输入张量 $expertX$ 执行以下操作：

1. **Sigmoid激活计算**：
   $$\text{sigmoidOut} = \sigma(expertX)$$

2. **叠加Bias计算**：
   $$\text{biasOut} = \text{sigmoidOut} + \text{bias}$$
    其中：
    - $expertX$ 是输入张量
    - $\sigma$ 表示Sigmoid激活函数
    - $\text{bias}$ 是偏置项
    - 输出包含两个张量：纯粹的Sigmoid结果和叠加Bias后的结果

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>blockDim</td>
      <td>输入</td>
      <td>AI CORE的数量，比如：Ascend910B是40。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
	<tr>
      <td>expertX</td>
      <td>输入</td>
      <td>输入张量（Router logits），shape为(rows, expert_num)</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>bias</td>
      <td>输入</td>
      <td>偏置张量，shape为(expert_num,)，将被广播加到每一行</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>sigmoidOut</td>
      <td>输出</td>
      <td>Sigmoid激活后的输出张量，shape同输入X</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>biasOut</td>
      <td>输出</td>
      <td>Sigmoid叠加Bias后的输出张量，shape同输入X</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

```
torch.ops.npu_ops_transformer_ext.bias_sigmoid(blockDim, expertX, bias, sigmoidOut, biasOut)
```
