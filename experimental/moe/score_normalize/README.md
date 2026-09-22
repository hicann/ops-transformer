# ScoreNormalize

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                                         | 是       |

## 功能说明

- 算子功能：MoE路由专家得分进行归一化。
- 计算公式：
  $$
  x = \frac{x}{\sum_{i=1}^{k} x_i} \times 2.5
  $$
- 这里$x$是一个二维张量，表示路由专家的分数，shape为\[$rows$, $k$\]，每行表示一个token选择的topk个专家的分数。计算公式对每个token选择的所有专家分数进行归一化，除以所有分数的总和再乘2.5。

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
      <td>x</td>
      <td>输入/输出</td>
      <td>公式中的输入/输出张量x，shape为(rows, k)</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rows</td>
      <td>输入</td>
      <td>输入/输出x的第一维</td>
      <td>INT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>k</td>
      <td>输入</td>
      <td>输入/输出x的第二维</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- x仅支持BFLOAT16类型。
- k <= 16。

## 调用说明

```
torch.ops.npu_ops_transformer_ext.score_normalize(x, rows, k)
```
