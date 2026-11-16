# EngramFusedWeight

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                   |    √    |
| <term>Atlas A3系列产品</term>                             |    ×    |
| <term>Atlas A2系列产品</term>                             |    ×    |
| <term>Atlas 200I/500 A2推理产品</term>                    |    ×    |
| <term>Atlas推理系列产品</term>                            |    ×    |
| <term>Atlas训练系列产品</term>                            |    ×    |

## 功能说明

- **算子功能**：Engram 门控的权重融合算子。将两个BF16/FP16权重逐元素相乘并升精度输出 FP32：`weight_fused = fp32(weight_hidden) * fp32(weight_embed)`。

- **计算公式**：对展平后的每个元素 $i$（$i = 0, 1, \dots, hcMult \times hiddenSize - 1$）：

  $$
  y_i = \mathrm{float}(x1_i) \times \mathrm{float}(x2_i)
  $$

  kernel 内先做 BF16/FP16 → FP32 升精度再相乘，保证 FP32 精度。

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>weight_hidden</td>
        <td>输入</td>
        <td>Engram 门控的第一个权重（q_weight），shape为(hcMult, hiddenSize)。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight_embed</td>
        <td>输入</td>
        <td>Engram 门控的第二个权重（k_weight），shape和数据类型与weight_hidden相同。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight_fused</td>
        <td>输出</td>
        <td>融合权重，shape与输入相同。</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- <term>Ascend 950PR&950DT系列产品</term>：
  - 输入仅支持2维格式，shape为(hcMult, hiddenSize)，两个输入shape和数据类型必须相同。
  - 总元素数（hcMult × hiddenSize）须满足256对齐。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_engram_fused_weight](examples/test_aclnn_engram_fused_weight.cpp) | 通过[aclnnEngramFusedWeight](docs/aclnnEngramFusedWeight.md)接口方式调用EngramFusedWeight算子。 |
| PyTorch API | - | 通过[torchapi_engram_fused_weight](docs/torchapi_engram_fused_weight.md)接口方式调用EngramFusedWeight算子。 |
