# grouped_matmul_activation_quant

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

  `cann_ops_transformer.grouped_matmul_activation_quant`融合GroupedMatmul、激活函数和量化计算，当前用于WeightNz路径下的`gelu_tanh`激活和MXFP8量化场景，底层调用`aclnnGroupedMatmulActivationQuantWeightNz`接口完成计算。

- **计算公式**：

  具体计算流程如下：
  1. 按`group_list`将`x`在M轴方向划分为多个group。
  2. 每个group分别与对应的`weight`做矩阵乘计算，并结合`x_scale`和`weight_scale`完成MXFP8反量化。
  3. 对矩阵乘结果执行`gelu_tanh`激活。
  4. 对激活结果执行MX动态量化，输出`y`和`y_scale`。

## 函数原型

```python
cann_ops_transformer.grouped_matmul_activation_quant(
    x,
    group_list,
    weight,
    weight_scale,
    activation_type,
    *,
    bias=None,
    x_scale=None,
    group_list_type=0,
    tuning_config=None,
    quant_mode=None,
    y_dtype=None,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | 左矩阵，表示GroupedMatmul的输入激活。 | `torch.float8_e4m3fn`、`torch.float8_e5m2` | `(M, K)` |
| `group_list` | Tensor | 必选 | 分组信息。`group_list_type=0`时表示每个group在M轴上的累计结束位置；`group_list_type=1`时表示每个group的M轴长度。 | `torch.int64` | `(E,)` |
| `weight` | List[Tensor] | 必选 | 右矩阵TensorList，当前MX FP8场景tensorList长度仅支持1。调用者必须传入FRACTAL_NZ格式的`weight`，torch接口按3维逻辑shape解析。 | `torch.float8_e4m3fn` | 非转置：`(E, K, N)`<br>转置：`(E, N, K)` |
| `weight_scale` | List[Tensor] | 必选 | `weight`的MX量化scale，当前MX FP8场景tensorList长度仅支持1。 | 通过`weight_scale_dtype`按`torch_npu.float8_e8m0fnu`解析 | 非转置：`(E, ceil(K / 64), N, 2)`<br>转置：`(E, N, ceil(K / 64), 2)` |
| `activation_type` | str | 必选 | 激活函数类型，当前仅支持`"gelu_tanh"`。 | string | - |
| `bias` | List[Tensor] | 可选 | bias TensorList，默认值为`None`。当前MX FP8场景必须为空，支持`None`、空TensorList或单个空Tensor。 | `torch.float32` | - |
| `x_scale` | Tensor | 可选 | `x`的MX量化scale。当前MX FP8场景必须传入有效Tensor。 | 通过`x_scale_dtype`按`torch_npu.float8_e8m0fnu`解析 | `(M, ceil(K / 64), 2)` |
| `group_list_type` | int | 可选 | `group_list`语义类型，默认值为0，支持0或1。 | int | - |
| `tuning_config` | List[int] | 可选 | 预留调优参数，默认值为`None`。 | int | - |
| `quant_mode` | str | 可选 | 量化模式，默认值为`None`。torch层不解析该参数，直接透传到aclnn层；显式传值时当前仅支持`"mx"`。 | string | - |
| `y_dtype` | torch.dtype | 可选 | 输出`y`的数据类型，默认值为`None`；为`None`时推导为与`x`相同的数据类型。 | `torch.float8_e4m3fn`、`torch.float8_e5m2` | - |
| `round_mode` | str | 可选 | 舍入模式，默认值为`"rint"`，当前仅支持`"rint"`。 | string | - |
| `scale_alg` | int | 可选 | MX量化scale算法，默认值为0；0表示OCP实现，1表示cuBLAS实现。 | int | - |
| `dst_type_max` | float | 可选 | 表示maxType的取值，对应公式中的Amax(DType)，默认值为0.0。当前MX FP8场景仅支持0.0，表示Amax(DType)为量化结果数据类型的最大值。 | float | - |
| `x_dtype` | int | 可选 | `x`的dtype wrapper覆盖值，默认值为`None`。 | - | - |
| `weight_dtype` | int | 可选 | `weight`的dtype wrapper覆盖值，默认值为`None`。 | - | - |
| `weight_scale_dtype` | int | 可选 | `weight_scale`的dtype wrapper覆盖值，默认值为`None`。当前MX FP8场景需要传入`torch_npu.float8_e8m0fnu`。 | `torch_npu.float8_e8m0fnu` | - |
| `x_scale_dtype` | int | 可选 | `x_scale`的dtype wrapper覆盖值，默认值为`None`。当前MX FP8场景需要传入`torch_npu.float8_e8m0fnu`。 | `torch_npu.float8_e8m0fnu` | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | 必选 | 激活后量化输出。 | 由`y_dtype`指定；`y_dtype=None`时与`x`相同 | `(M, N)` |
| `y_scale` | Tensor | 必选 | 输出`y`对应的MX量化scale。 | `torch_npu.float8_e8m0fnu` | `(M, ceil(N / 64), 2)` |

## 约束说明

- 适用场景：该接口支持训练、推理场景下使用。
- 调用方式：该接口支持单算子模式调用。
- `N`必须为64的整数倍，`E`取值范围为`[1, 1024]`。
- 支持M为0或N为0的空Tensor场景；该场景下允许K为0。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer

  E = 1
  M = 64
  K = 128
  N = 128

  x = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  weight = torch.randint(-8, 8, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn)
  weight_scale = torch.randint(-8, 8, (E, math.ceil(K / 64), N, 2), dtype=torch.int8).npu()
  x_scale = torch.randint(-8, 8, (M, math.ceil(K / 64), 2), dtype=torch.int8).npu()
  group_list = torch.tensor([M], dtype=torch.int64).npu()

  y, y_scale = cann_ops_transformer.grouped_matmul_activation_quant(
      x,
      group_list,
      [weight],
      [weight_scale],
      "gelu_tanh",
      bias=None,
      x_scale=x_scale,
      quant_mode="mx",
      y_dtype=None,
      weight_scale_dtype=torch_npu.float8_e8m0fnu,
      x_scale_dtype=torch_npu.float8_e8m0fnu,
  )
  ```
