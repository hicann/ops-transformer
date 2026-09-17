# FusedGdnGating

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      √     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：FusedGdnGating是GDN（Gated Delta Network）模型中的融合门控算子，用于在一次kernel调用中完成softplus门控计算和sigmoid门控计算，减少kernel launch开销。

- 计算公式：

  门控输出$g$的计算：

  $$
  g = -e^{a\_log} \cdot \text{softplus}(a + dt\_bias)
  $$

  其中$\text{softplus}(x) = \frac{1}{\beta}\ln(1 + e^{\beta x})$，当$\beta \cdot (a + dt\_bias) > \text{threshold}$时，采用线性近似$\text{softplus}(x) \approx x$以避免数值溢出。

  门控输出$\text{beta\_output}$的计算：

  $$
  \text{beta\_output} = \text{sigmoid}(b) = \frac{1}{1 + e^{-b}}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px">
  <colgroup>
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>aLog</td>
      <td>输入</td>
      <td>公式中的$a\_log$，log域参数，用于指数衰减。</td>
      <td>数据类型须与dtBias一致。</td>
      <td>FLOAT32、BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>a</td>
      <td>输入</td>
      <td>公式中的$a$，门控输入数据。</td>
      <td>支持BFLOAT16或FLOAT16。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>公式中的$b$，sigmoid门控输入数据。</td>
      <td>数据类型和shape与a一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtBias</td>
      <td>输入</td>
      <td>公式中的$dt\_bias$，偏置参数。</td>
      <td>数据类型须与aLog一致，shape与aLog一致。</td>
      <td>FLOAT32、BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td>公式中的$\beta$，softplus缩放系数。</td>
      <td>默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>属性</td>
      <td>softplus阈值，超过该值时采用线性近似。</td>
      <td>默认值为20.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>g</td>
      <td>输出</td>
      <td>公式中的$g$，门控输出。</td>
      <td>数据类型固定为FLOAT32。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>betaOutput</td>
      <td>输出</td>
      <td>公式中的$\text{beta\_output}$，sigmoid输出。</td>
      <td>数据类型与a一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入aLog和dtBias的数据类型必须一致。
- 输入a和b的数据类型和shape必须一致。
- 输入a的shape为[batch, num_heads]，aLog和dtBias的shape为[num_heads]，且num_heads与a的第二维一致。
- 输出g的shape为[1, batch, num_heads]，数据类型固定为FLOAT32。
- 输出betaOutput的shape为[1, batch, num_heads]，数据类型与a一致。
- beta取值不能为0，否则会触发除零保护，inv_beta将被设为无穷大。
- <term>Atlas A2训练系列产品</term>：支持如下6种数据类型组合：

  <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th>序号</th>
      <th>aLog / dtBias</th>
      <th>a</th>
      <th>b</th>
      <th>g</th>
      <th>betaOutput</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>FLOAT32</td><td>BFLOAT16</td><td>BFLOAT16</td><td>FLOAT32</td><td>BFLOAT16</td></tr>
    <tr><td>2</td><td>FLOAT32</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT32</td><td>FLOAT16</td></tr>
    <tr><td>3</td><td>BFLOAT16</td><td>BFLOAT16</td><td>BFLOAT16</td><td>FLOAT32</td><td>BFLOAT16</td></tr>
    <tr><td>4</td><td>BFLOAT16</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT32</td><td>FLOAT16</td></tr>
    <tr><td>5</td><td>FLOAT16</td><td>BFLOAT16</td><td>BFLOAT16</td><td>FLOAT32</td><td>BFLOAT16</td></tr>
    <tr><td>6</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT32</td><td>FLOAT16</td></tr>
  </tbody>
  </table>

- <term>Atlas 推理系列产品</term>和<term>Atlas 200I/500 A2 推理产品</term>：仅支持全FLOAT16数据类型组合（即aLog、a、b、dtBias均为FLOAT16，g为FLOAT32，betaOutput为FLOAT16）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn调用 | [test_aclnn_fused_gdn_gating](./examples/test_aclnn_fused_gdn_gating.cpp) | 通过aclnn接口方式调用FusedGdnGating算子。 |
| PyTorch调用 | [torch_ops_extension](./torch_ops_extension/README.md) | 通过PyTorch扩展注册为`torch.ops.custom.npu_fused_gdn_gating`调用，构建与安装见说明文档。 |
| pytest测试 | [tests/pytest](./tests/pytest/README.md) | CPU golden与NPU结果精度对比验证，执行`bash test_run.sh single`。 |

## 参考资源
