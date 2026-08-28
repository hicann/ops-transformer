# RecurrentKda

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：完成KDA（Kimi Delta Attention）的递归前向计算，面向decode和MTP（Multi-Token Prediction）短序列场景。

- 计算公式：

  对于每个时间步 $t$，令 $\bar q_t$ 为乘过缩放系数的query，$\bar k_t$ 为key，$d_t$ 为gate对应的衰减系数，$\bar\beta_t$ 为beta的有效值，则递归状态 $S_t$ 和输出 $o_t$ 的计算公式如下：

  $$
  \bar S_t = S_{t-1}\odot d_t
  $$

  $$
  \Delta_t = \bar\beta_t(v_t-\bar S_t\bar k_t)
  $$

  $$
  S_t = \bar S_t+\Delta_t\bar k_t^T
  $$

  $$
  o_t = S_t\bar q_t
  $$

  其中，$S_{t-1},S_t\in R^{V_{dim}\times K_{dim}}$，$q_t,k_t\in R^{K_{dim}}$，$v_t,o_t\in R^{V_{dim}}$。$d_t$ 沿状态矩阵的K维广播。算子支持在kernel内对query和key进行L2归一化、转换raw gate以及对beta执行sigmoid计算。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
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
    <td>query</td>
    <td>输入</td>
    <td>公式中的q。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的k。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的v。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gate</td>
    <td>输入</td>
    <td>衰减系数对应的step log gate或raw gate。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>Delta更新系数。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initialStateRef</td>
    <td>输入&输出</td>
    <td>状态矩阵，公式中的输入S。</td>
    <td>BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cuSeqlensOptional</td>
    <td>可选输入</td>
    <td>变长序列累计偏移。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ssmStateIndicesOptional</td>
    <td>可选输入</td>
    <td>状态池槽位索引。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>aLogOptional</td>
    <td>可选输入</td>
    <td>kernel内转换raw gate时使用的A_log参数。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtBiasOptional</td>
    <td>可选输入</td>
    <td>kernel内转换raw gate时使用的偏置。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>numAcceptedTokensOptional</td>
    <td>可选输入</td>
    <td>投机解码中每条序列已接受的token数。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attnOut</td>
    <td>输出</td>
    <td>公式中的o。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>finalState</td>
    <td>输出</td>
    <td>最终状态矩阵。</td>
    <td>BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入tensor的shape大小、可选参数依赖和属性取值需满足一定约束，具体见[aclnnRecurrentKda](./docs/aclnnRecurrentKda.md)。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [aclnnRecurrentKda](./docs/aclnnRecurrentKda.md) | 通过[aclnnRecurrentKda](./docs/aclnnRecurrentKda.md)调用RecurrentKda算子 |
