# MaskedCausalConv1d

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对hidden层的token之间进行带mask的因果一维分组卷积操作。

- 计算公式：

  假设输入x和输出y的shape是[S, B, H]，卷积权重weight的shape是[W, H]，i和j分别表示S和B轴的索引，那么输出将被表示为：

  $$
  y[i,j] = mask[j,i] * \sum_{k=0}^{W-1} x[i-k,j] * weight[W-1-k]
  $$

  其中，无效位置的padding为0填充；当前W仅支持3；H轴为elementwise操作，上述公式不体现。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>输入序列，shape为[S, B, H]，对应公式中x。不支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>因果1维分组卷积核，shape为[W, H]，W固定为3，对应公式中weight。不支持空Tensor。</td>
        <td>数据类型与x一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mask</td>
        <td>可选输入</td>
        <td>布尔掩码，shape为[B, S]，对应公式中mask。默认值是None，为None时等价于全True。不支持空Tensor。</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>输出结果，shape与x一致。不支持空Tensor。</td>
        <td>数据类型与x一致</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入值域限制：
  - B * S：取值范围为1~512K。
  - H（hiddenSize）：取值范围384~24576（64的整数倍）。
  - W：当前只支持3。
- 算子入参与中间计算结果，在对应运行数据类型（float16/bfloat16）下，数值均不会超出该类型值域范围。
- 算子输入不支持有±inf和nan的情况。

## 调用说明

  | 调用方式  | 样例代码                                                     | 说明                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn接口 | [test_aclnn_masked_causal_conv1d](./examples/test_aclnn_masked_causal_conv1d.cpp) | 通过[aclnnMaskedCausalConv1d](./docs/aclnnMaskedCausalConv1d.md)调用MaskedCausalConv1d算子 |
