# inplace_partial_rotary_mul

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
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

    执行单路旋转位置编码的Inplace计算，直接修改输入张量`x`，不产生新的输出张量。该接口支持通过`partial_slice`参数指定输入张量最后一维上的局部范围，仅对该范围内的数据执行旋转位置编码，其余位置保持原值。

- **计算公式**：

    interleave模式（`rotary_mode`为`"interleave"`）下，设`partial_slice=[start, end]`，被旋转的局部张量为：

    $$x_{slice} = x[..., start:end]$$

    计算过程如下：

    $$x_1 = x_{slice}[..., ::2]$$

    $$x_2 = x_{slice}[..., 1::2]$$

    $$x_{rotate} = \text{cat}(-x_2, x_1)$$

    $$x_{out} = x_{slice} \cdot \cos + x_{rotate} \cdot \sin$$

    最终将$x_{out}$原地写回`x[..., start:end]`。其中，$x$表示参数`x`，$\cos$表示参数`r1`，$\sin$表示参数`r2`。当`start`与`end`相等时，不执行旋转位置编码，`x`保持不变。反向算子`inplace_partial_rotary_mul_backward`同样支持空Tensor和切片长度为零的场景（执行no-op，详见[约束说明](#约束说明)）。

- **说明**：

  - 输入`x`采用BSND维度格式，其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，N（Head-Num）表示多头数，D（Head-Dim）表示每个头的隐藏维度大小。
  - `partial_slice`作用于输入`x`的最后一维D维，取值范围为左闭右开区间`[start, end)`。Python接口中`partial_slice`默认值为`None`，内部按`[0, 0]`处理。

## 函数原型

```python
cann_ops_transformer.inplace_partial_rotary_mul(x, r1, r2, *, rotary_mode="interleave", partial_slice=None) -> None
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | 必选 | 待执行旋转位置编码的张量，对应公式中的x。Inplace模式下，计算结果直接写回该Tensor。 | bfloat16、float16、float32 | (B, S, N, D) |
| r1 | Tensor | 必选 | 位置编码张量，对应公式中的cos分量。 | bfloat16、float16、float32 | 4维，需与x满足广播关系 |
| r2 | Tensor | 必选 | 位置编码张量，对应公式中的sin分量，`r2`和`r1`的数据类型必须一致。 | bfloat16、float16、float32 | 4维，需与x满足广播关系 |
| rotary_mode | str | 可选 | 旋转模式。当前仅支持"interleave"，默认值为"interleave"。 | - | - |
| partial_slice | List[int] | 可选 | 部分旋转的切片范围[start, end)，作用于x的最后一维D维。默认值为None，接口内部按[0, 0]处理。 | - | - |

## 返回值说明

该接口无返回值（`None`）。计算结果直接inplace写回输入张量 `x`，`x` 在计算后shape和数据类型保持不变，`partial_slice` 指定范围以外的数据保持原值。

> **自动微分说明**：当 `x.requires_grad` 为True时，对loss执行 `.backward()` 即自动触发 `inplace_partial_rotary_mul_backward`，无需手动调用反向算子。`r1`（cos）、`r2`（sin）的梯度**不计算**，始终为None。反向算子 `inplace_partial_rotary_mul_backward` 支持空Tensor和切片长度为零的场景（执行no-op），自动微分可正常使用。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 不支持非连续Tensor。
- `x`最后一维D大小不超过1024，且D必须为2的倍数。
- `partial_slice`必须包含两个整数，满足`start >= 0`、`end >= 0`、`end <= D`、`end - start >= 0`。
- `partial_slice`切片长度（即`end - start`）必须为2的倍数。当`end`和`start`相等时（切片长度为零），正向和反向计算均执行no-op，直接返回。
- `r1`、`r2`最后一维大小必须相同，且必须等于`partial_slice`的切片长度（即`end - start`）。
- `r1`、`r2`的shape必须与`x[..., start:end]`满足广播关系，且存在如下约束：
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：`r1`、`r2`的shape当前只支持BSND、B1ND、B11D、111D排布。
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`r1`、`r2`的shape当前只支持BS1D、B11D排布。
  <!-- end id8 -->
- `x`的各维度值必须大于0；当`partial_slice`不是空切片时，`r1`、`r2`参与计算的维度值必须大于0。
- **自动微分约束**：仅计算 `x` 的梯度；`r1`、`r2` 的梯度不计算，始终为None。因算子为输入输出同地址操作，`x` 不能是 `requires_grad=True` 的叶子张量。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    inplace_partial_rotary_mul(
        x,
        r1,
        r2,
        rotary_mode="interleave",
        partial_slice=[slice_start, slice_end],
    )
    ```

- 训练模式调用（自动微分）

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    B, S, N, D = 2, 32, 8, 128
    slice_start, slice_end = 0, 64

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16, requires_grad=True)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    y = x * 1.0
    y.retain_grad()

    # 正向：自动追踪计算图（y被inplace修改，无需接收返回值）
    inplace_partial_rotary_mul(
        y, r1, r2,
        rotary_mode="interleave",
        partial_slice=[slice_start, slice_end],
    )

    # 继续前向计算
    loss = y.sum()
    loss.backward()  # 自动调用inplace_partial_rotary_mul_backward

    print(y.grad.shape)
    print(x.grad.shape)
    # r1.grad, r2.grad始终为None（cos/sin不计算梯度）
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    class InplacePartialRotaryMulModel(torch.nn.Module):
        def forward(self, x, r1, r2):
            inplace_partial_rotary_mul(
                x,
                r1,
                r2,
                rotary_mode="interleave",
                partial_slice=[slice_start, slice_end],
            )
            return x

    model = InplacePartialRotaryMulModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    output = model(x, r1, r2)
    ```
