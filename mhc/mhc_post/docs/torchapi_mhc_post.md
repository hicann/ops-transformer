# mhc_post

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id5 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id5 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

    实现MHC Post组件的前向计算，用于Transformer模型中多层残差连接的后处理阶段。该算子将残差矩阵变换与输出状态投影融合为单次计算，避免多次独立算子调用带来的额外开销。

- 计算公式：

  - MHC Post算子的核心计算公式为：

    $$x_{l+1} = (H_{l}^{res})^{T} \cdot x_{l} + h_{l}^{out} \cdot H_{t}^{post}$$

    其中：

    - $(H_{l}^{res})^{T} \cdot x_{l}$ 表示对输入$x_{l}$进行残差矩阵的转置矩阵乘法。对于输出中的第$i$行（对应第$i$个head），计算过程为：

      $$x_{l+1}[i] = \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j]$$

      即将$H_{l}^{res}$矩阵按转置方式与$x_{l}$做矩阵乘法，$H_{l}^{res}[j, i]$为标量，对$x_{l}$的第$j$行做标量乘法后累加到第$i$行输出。

    - $h_{l}^{out} \cdot H_{t}^{post}$ 表示输出状态$h_{l}^{out}$与后处理权重$H_{t}^{post}$的逐元素乘法与广播。对于第$i$个head：

      $$x_{l+1}[i] += H_{t}^{post}[i] \cdot h_{l}^{out}$$

      即$H_{t}^{post}[i]$为标量，对$h_{l}^{out}$整行做标量乘法后加到第$i$行输出。

  - 综合完整计算过程为：

    $$x_{l+1}[i, :] = H_{t}^{post}[i] \cdot h_{l}^{out}[:] + \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j, :]$$

    其中，$x_{l}$表示参数`x`，$H_{l}^{res}$表示参数`h_res`，$h_{l}^{out}$表示参数`h_out`，$H_{t}^{post}$表示参数`h_post`，$x_{l+1}$表示输出`y`。

  - 当`h_res`传入None时，跳过Res Mapping，计算公式退化为直接残差连接：

    $$x_{l+1} = x_{l} + h_{l}^{out} \cdot H_{t}^{post}$$

  >**说明：**
  >
  > 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch序列长度的累加和（$T = B \times S$），n表示头数（head数量），D表示每个头的隐藏维度大小（headdim）。

## 函数原型

```python
cann_ops_transformer.mhc_post(x, h_res, h_out, h_post) -> Tensor
```

其中`h_res`为可选输入，可传入None（仅Ascend 950PR/Ascend 950DT的单算子模式支持传入None，图模式不支持）。

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
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
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>当前层的输入token特征，对应公式中的x<sub>l</sub>。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_res</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>残差连接矩阵，对应公式中的H<sub>l</sub><sup>res</sup>。传入None时跳过Res Mapping，计算公式退化为直接残差连接，仅Ascend 950PR/Ascend 950DT支持传入None，其他产品形态传入None会报错。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n, n)</li>
                <li>(T, n, n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_out</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>上一层的输出状态，对应公式中的H<sub>l</sub><sup>out</sup>。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, D)</li>
                <li>(T, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_post</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>后处理权重矩阵，对应公式中的H<sub>t</sub><sup>post</sup>。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n)</li>
                <li>(T, n)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
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
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MHC Post计算输出，对应公式中的x<sub>l+1</sub>，数据类型与输入`x`保持一致，shape与输入`x`保持一致。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 数据类型约束：
  - `x`和`h_out`的数据类型必须相同。
  - 输出`y`的数据类型与`x`保持一致。

- `h_res`可选约束：
  - `h_res`传入None时跳过Res Mapping，仅Ascend 950PR/Ascend 950DT支持；Atlas A2/A3系列产品`h_res`为必传参数，传入None会报错。
  - `h_res`传入None仅支持单算子模式调用；图模式（torch.compile）下`h_res`必须传入，传入None会在GE编译阶段报错。

- 维度约束：
  - `h_res`的维度需与`x`维度格式匹配：4维时为(B, S, n, n)，3维时为(T, n, n)。

- Shape一致性约束：
  - 4维（BSND）格式下：
    - `h_res`的(B, S)维度需与`x`的(B, S)维度一致，`h_res`的后两维为(n, n)，其中n与`x`的第3维一致。
    - `h_out`的(B, S)维度需与`x`的(B, S)维度一致，`h_out`的D维度需与`x`的D维度一致。
    - `h_post`的(B, S)维度需与`x`的(B, S)维度一致，`h_post`的n维度需与`x`的n维度一致。
  - 3维（TND）格式下：
    - `h_res`的T维度需与`x`的T维度一致，后两维为(n, n)。
    - `h_out`的T维度需与`x`的T维度一致，D维度需与`x`的D维度一致。
    - `h_post`的T维度需与`x`的T维度一致，n维度需与`x`的n维度一致。

- 所有输入Tensor的shape各维度值必须为正数（大于0）。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_post

  B = 2
  S = 8
  n = 4
  D = 128

  x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
  h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
  h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
  h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

  y = mhc_post(x, h_res, h_out, h_post)
  print(f"output shape: {y.shape}")

  # h_res缺省时（仅Ascend 950PR/Ascend 950DT支持，且仅支持单算子模式）
  y = mhc_post(x, None, h_out, h_post)
  print(f"output shape: {y.shape}")
  ```

- 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import mhc_post

  torch_npu.npu.set_device(0)

  B = 2
  S = 8
  n = 4
  D = 128

  class MhcPostModel(torch.nn.Module):
      def forward(self, x, h_res, h_out, h_post):
          return mhc_post(x, h_res, h_out, h_post)

  model = MhcPostModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device="npu")
  h_res = torch.randn(B, S, n, n, dtype=torch.float32, device="npu")
  h_out = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
  h_post = torch.randn(B, S, n, dtype=torch.float32, device="npu")

  y = model(x, h_res, h_out, h_post)
  ```
