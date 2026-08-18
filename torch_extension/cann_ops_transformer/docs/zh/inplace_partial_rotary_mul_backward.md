# inplace_partial_rotary_mul_backward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

    执行局部旋转位置编码`inplace_partial_rotary_mul`的反向计算。该接口对输入`grad_output`的D维度上切片`[start, end)`区域执行旋转位置编码梯度计算，计算结果inplace写回`grad_output`。切片之外的元素保持原值不变。

- **计算公式**：

    interleave模式（`rotary_mode`为`"interleave"`）下，设`partial_slice=[start, end]`，被计算的局部张量为：

    $$dy' = grad\_output[..., start:end]$$

    $$cos' = r1[..., start:end]$$

    $$sin' = r2[..., start:end]$$

    令：

    $$dy1', dy2' = dy'[..., ::2], dy'[..., 1::2]$$

    $$cos1', cos2' = cos'[..., ::2], cos'[..., 1::2]$$

    $$sin1', sin2' = sin'[..., ::2], sin'[..., 1::2]$$

    则梯度计算为：

    $$dx' = stack((cos1' \cdot dy1' + sin2' \cdot dy2', \ cos2' \cdot dy2' - sin1' \cdot dy1'), \ dim=-1).reshape(dy'.shape)$$

    $dx'$的结果inplace写回`grad_output`的`[start, end)`区间。

- **说明**:

  - 该算子当前仅实现了`rotary_mode="interleave"`模式。half（0）、quarter（2）、interleave-half（3）模式暂未支持。
  - 输入`grad_output`采用BSND维度格式，其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，N（Head-Num）表示多头数，D（Head-Dim）表示每个头的隐藏维度大小。
  - `partial_slice`作用于输入`grad_output`的最后一维D维，取值范围为左闭右开区间`[start, end)`。

## 函数原型

```python
cann_ops_transformer.inplace_partial_rotary_mul_backward(grad_output, r1, r2, *, rotary_mode="interleave", partial_slice=None) -> None
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| grad_output | Tensor | 必选 | 正向输出y的梯度，对应正向算子`inplace_partial_rotary_mul`的`x`的梯度。Inplace模式下，计算结果直接写回该Tensor，成为正向输入`x`的梯度`dx`。 | bfloat16、float16、float32 | (B, S, N, D) |
| r1 | Tensor | 必选 | 位置编码张量，对应正向计算中的cos分量。 | bfloat16、float16、float32 | 4维，需与grad_output满足广播关系 |
| r2 | Tensor | 必选 | 位置编码张量，对应正向计算中的sin分量。`r2`和`r1`的数据类型必须一致。 | bfloat16、float16、float32 | 4维，需与grad_output满足广播关系 |
| rotary_mode | str | 可选 | 旋转模式。当前仅支持`"interleave"`，默认值为`"interleave"`。 | - | - |
| partial_slice | List[int] | 可选 | 部分旋转的切片范围`[start, end)`，作用于`grad_output`的最后一维D维。默认值为`None`，接口内部按`[0, 0]`处理。 | - | - |

## 返回值说明

该接口无返回值。计算结果直接写回输入张量`grad_output`，`grad_output`在计算后shape和数据类型保持不变，`partial_slice`指定范围以外的数据保持原值，指定范围内的数据被替换为正向输入`x`的梯度`dx`。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- 该算子仅支持连续Tensor，不支持非连续Tensor。
- **该算子当前版本仅支持interleave模式（`rotary_mode="interleave"`）**。half（0）、quarter（2）、interleave-half（3）模式暂未实现。
- **空输入支持**：当切片长度为0（`partial_slice`的start等于end，如`[0, 0]`），或`grad_output`、`r1`、`r2`中存在空Tensor（某维大小为0）时，算子执行no-op操作，不做旋转位置编码梯度计算，直接返回。
- `grad_output`最后一维D大小不超过1024。
- `partial_slice`必须包含两个整数，满足`start >= 0`、`end >= 0`、`end <= D`、`end - start >= 0`。
- 当切片长度（即`end - start`）大于0时，切片长度必须为2的倍数。
- 当切片长度（即`end - start`）大于0时，`r1`、`r2`最后一维大小必须相同，且必须等于切片长度。
- `r1`、`r2`的shape必须与`grad_output[..., start:end]`满足广播关系，且`r1`、`r2`shape当前只支持BSND、B1ND、B11D、111D排布。
- 当切片长度（即`end - start`）大于0时，`grad_output`、`r1`、`r2` 的各维度值必须大于0。

## 确定性计算

默认支持确定性计算。

## 配套接口

该算子为[inplace_partial_rotary_mul](./inplace_partial_rotary_mul.md)的反向算子，两者参数`r1`、`r2`、`rotary_mode`、`partial_slice`需保持一致。

> **说明**：当正向 `inplace_partial_rotary_mul` 中 `x.requires_grad` 为True时，`loss.backward()` 会自动触发本算子，无需手动调用。仅在需要显式控制梯度的场景下保留手动调用路径。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul
    from cann_ops_transformer.ops import inplace_partial_rotary_mul_backward

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    # 前向：x原地修改
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

    # 反向：grad_output原地修改为dx
    grad_output = torch.randn_like(x)
    inplace_partial_rotary_mul_backward(
        grad_output,
        r1,
        r2,
        rotary_mode="interleave",
        partial_slice=[slice_start, slice_end],
    )
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import inplace_partial_rotary_mul_backward

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    class InplacePartialRotaryMulBackwardModel(torch.nn.Module):
        def forward(self, grad_output, r1, r2):
            inplace_partial_rotary_mul_backward(
                grad_output,
                r1,
                r2,
                rotary_mode="interleave",
                partial_slice=[slice_start, slice_end],
            )
            return grad_output

    model = InplacePartialRotaryMulBackwardModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    grad_output = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    output = model(grad_output, r1, r2)
    ```
