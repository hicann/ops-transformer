# 算子名称：FinalRouting

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品                                         | 是       |

## 功能说明

- 算子功能：用于MoE模型的最终路由阶段，将各个Expert计算的结果按照评分加权合并，得到每个Token的最终输出，完成MoE的Combine阶段。

- 计算公式：
对于每个Token `t`：

$$
out[t] = \sum_{e} \big(in[token\_table[t, e]] \cdot score\_table[t, e]\big)
$$

其中，仅当 `token_table[t, e] >= 0` 时参与计算。

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
      <td>in</td>
      <td>输入</td>
      <td>expert输出张量， shape为(expert_num*token_num, hidden_size)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>token_table</td>
      <td>输入</td>
      <td>token到expert的映射表， shape为(token_num, expert_num)</td>
      <td>int32_t</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>score_table</td>
      <td>输入</td>
      <td>每个token在每个expert的评分，shape为(token_num, expert_num)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>out</td>
      <td>输出</td>
      <td>加权合并后的输出张量，shape为(token_num, hidden_size)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- token_table中小于0的值表示该expert对此token无效

## 调用说明

```
torch.ops.npu_ops_transformer_ext.final_routing(block_dim, input, token_table,score_table, output)
```
