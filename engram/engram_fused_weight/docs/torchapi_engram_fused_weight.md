# engram_fused_weight

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  Engram 门控的权重融合接口。将两个BF16/FP16权重逐元素相乘并升精度输出 FP32：`weight_fused = fp32(weight_hidden) * fp32(weight_embed)`。

- **计算公式**：对展平后的每个元素 $i$（$i = 0, 1, \dots, hcMult \times hiddenSize - 1$）：

  $$
  y_i = \mathrm{float}(x1_i) \times \mathrm{float}(x2_i)
  $$

  接口内先做 BF16/FP16 → FP32 升精度再相乘，保证 FP32 精度。

## 函数原型

```python
cann_ops_transformer.engram_fused_weight(weight_hidden, weight_embed) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 480px">
<col style="width: 140px">
<col style="width: 240px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>weight_hidden</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>Engram 门控的第一个权重（q_weight）。对应公式中$x1_i$。</td>
        <td>bfloat16、float16</td>
        <td>(hcMult, hiddenSize)</td>
    </tr>
    <tr>
        <td>weight_embed</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>Engram 门控的第二个权重（k_weight），shape和数据类型需与weight_hidden相同。对应公式中$x2_i$。</td>
        <td>bfloat16、float16</td>
        <td>(hcMult, hiddenSize)</td>
    </tr>
    </tbody>
</table>

## 输出说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 140px">
<col style="width: 100px">
<col style="width: 100px">
<col style="width: 480px">
<col style="width: 140px">
<col style="width: 240px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>weight_fused</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>融合权重，为新分配张量，shape需与输入相同。对应公式中$y_i$。</td>
        <td>float32</td>
        <td>(hcMult, hiddenSize)</td>
    </tr>
    </tbody>
</table>

> 说明：hcMult表示残差流数量（HC维度大小），hiddenSize表示隐藏层维度大小。

## 约束说明

- 该接口支持推理场景下使用（权重静态，通常离线预计算一次）。
- 该接口支持单算子模式和图模式调用。
- 参数约束：两个输入的shape和数据类型必须相同，输入输出支持空Tensor。
- 规格约束：
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
    - 输入仅支持2维格式，shape为(hcMult, hiddenSize)。
    - 总元素数（hcMult × hiddenSize）须满足256对齐。
  <!-- end id7 -->

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import engram_fused_weight

  torch_npu.npu.set_device(0)

  hcMult = 4
  hiddenSize = 5120

  weight_hidden = torch.randn(hcMult, hiddenSize, dtype=torch.bfloat16, device="npu")
  weight_embed = torch.randn(hcMult, hiddenSize, dtype=torch.bfloat16, device="npu")

  weight_fused = engram_fused_weight(weight_hidden, weight_embed)

  print(f"Output shape: {weight_fused.shape}")
  ```

- 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import engram_fused_weight

  torch_npu.npu.set_device(0)

  hcMult = 4
  hiddenSize = 5120

  class EngramFusedWeightModel(torch.nn.Module):
      def forward(self, weight_hidden, weight_embed):
          return engram_fused_weight(weight_hidden, weight_embed)

  model = EngramFusedWeightModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  weight_hidden = torch.randn(hcMult, hiddenSize, dtype=torch.bfloat16, device="npu")
  weight_embed = torch.randn(hcMult, hiddenSize, dtype=torch.bfloat16, device="npu")

  weight_fused = model(weight_hidden, weight_embed)
  ```
