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

  `cann_ops_transformer.grouped_matmul_activation_quant`融合GroupedMatmul、激活函数和量化计算，当前用于WeightNz路径下的`gelu_tanh`激活以及MXFP8、MXFP4输入场景，底层调用`aclnnGroupedMatmulActivationQuantWeightNz`接口完成计算。

- **计算公式**：

  具体计算流程如下：
  1. 按`group_list`将`x`在M轴方向划分为多个group。
  2. 每个group分别与对应的`weight`做矩阵乘计算，并结合`x_scale`和`weight_scale`完成MX反量化。
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
| `x` | Tensor | 必选 | MXFP4必须使用`torch.uint8`存放两个一组的打包数据，并且必须通过`x_dtype`指定逻辑类型。 | MXFP8：`torch.float8_e4m3fn`、`torch.float8_e5m2`；MXFP4：`torch.uint8` | MXFP8：`(M, K)`；MXFP4物理shape：`(M, K / 2)` |
| `group_list` | Tensor | 必选 | 分组信息。`group_list_type=0`时表示每个group在M轴上的累计结束位置，最后一个值不大于M轴大小；`group_list_type=1`时表示每个group的M轴长度，数值总和不大于M轴大小。 | `torch.int64` | `(E,)` |
| `weight` | List[Tensor] | 必选 | 右矩阵TensorList，长度仅支持1。调用者必须传入FRACTAL_NZ格式的`weight`；MXFP4必须使用`torch.uint8`存放打包数据，并且必须通过`weight_dtype`指定逻辑类型。转置场景先对源Weight执行`npu_format_cast(..., 29)`，再对末两维执行transpose并传入该view。 | MXFP8：`torch.float8_e4m3fn`；MXFP4：`torch.uint8` | 非转置逻辑shape：`(E, K, N)`；MXFP4物理shape：`(E, K, N / 2)`<br>转置源逻辑shape：`(E, N, K)`；MXFP4源物理shape：`(E, N, K / 2)` |
| `weight_scale` | List[Tensor] | 必选 | `weight`的MX量化scale，tensorList长度仅支持1。转置场景由源scale对中间两维执行transpose后传入。 | 通过`weight_scale_dtype`按`torch_npu.float8_e8m0fnu`解析 | 传入算子的shape：`(E, ceil(K / 64), N, 2)`<br>转置源shape：`(E, N, ceil(K / 64), 2)` |
| `activation_type` | str | 必选 | 激活函数类型，当前仅支持`"gelu_tanh"`。 | string | - |
| `bias` | List[Tensor] | 可选 | bias TensorList，默认值为`None`。当前MX场景必须为空，支持`None`、空TensorList或单个空Tensor。 | `torch.float32` | - |
| `x_scale` | Tensor | 可选 | `x`的MX量化scale。当前MX场景必须传入有效Tensor。 | 通过`x_scale_dtype`按`torch_npu.float8_e8m0fnu`解析 | `(M, ceil(K / 64), 2)` |
| `group_list_type` | int | 可选 | `group_list`语义类型，默认值为0，支持0或1。 | int | - |
| `tuning_config` | List[int] | 可选 | 预留调优参数，默认值为`None`。 | int | - |
| `quant_mode` | str | 可选 | 量化模式，默认值为`None`。torch层不解析该参数，直接透传到aclnn层；显式传值时当前仅支持`"mx"`。 | string | - |
| `y_dtype` | torch.dtype | 可选 | 输出`y`的数据类型，默认值为`None`，此时推导为与`x`相同的FP8或FP4逻辑类型。 | `torch.float8_e4m3fn`、`torch.float8_e5m2`、两种FLOAT4 wrapper dtype | - |
| `round_mode` | str | 可选 | 舍入模式，默认值为`"rint"`，当前仅支持`"rint"`。 | string | - |
| `scale_alg` | int | 可选 | MX量化scale算法，默认值为0。支持0、1、2：0表示OCP实现；1表示cuBLAS实现，仅支持FP8输出；2表示FLOAT4动态范围实现，仅支持FLOAT4_E2M1输出。FLOAT4_E1M2输出仅支持0。 | int | - |
| `dst_type_max` | float | 可选 | 表示maxType的取值，对应公式中的Amax(DType)，默认值为0.0。`scale_alg=2`时支持0.0或`[6.0, 12.0]`；0.0表示使用FLOAT4_E2M1的最大值6.0。其他`scale_alg`仅支持0.0。 | float | - |
| `x_dtype` | int | MXFP4必选 | `x`的逻辑dtype wrapper。MXFP4必须传`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`。 | int | - |
| `weight_dtype` | int | MXFP4必选 | `weight`的逻辑dtype wrapper。MXFP4必须传`torch_npu.float4_e2m1fn_x2`或`torch_npu.float4_e1m2fn_x2`。 | int | - |
| `weight_scale_dtype` | int | 可选 | `weight_scale`的dtype wrapper覆盖值，默认值为`None`。当前MX场景需要传入`torch_npu.float8_e8m0fnu`。 | `torch_npu.float8_e8m0fnu` | - |
| `x_scale_dtype` | int | 可选 | `x_scale`的dtype wrapper覆盖值，默认值为`None`。当前MX场景需要传入`torch_npu.float8_e8m0fnu`。 | `torch_npu.float8_e8m0fnu` | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | 必选 | 激活后量化输出。FP4以`torch.uint8`承载打包数据。 | 由`y_dtype`指定；`y_dtype=None`时逻辑类型跟随`x` | FP8：`(M, N)`；FP4物理shape：`(M, N / 2)` |
| `y_scale` | Tensor | 必选 | 输出`y`对应的MX量化scale。 | `torch_npu.float8_e8m0fnu` | `(M, ceil(N / 64), 2)` |

## 约束说明

- 适用场景：该接口支持训练、推理场景下使用。
- 调用方式：该接口支持单算子模式调用。
- `N`必须为64的整数倍，`E`取值范围为`[1, 1024]`。
- MXFP4场景下`x`与`weight`必须同时为FLOAT4，二者可分别选择E2M1或E1M2；`K`必须为偶数且不能为2。
- MXFP4 Torch输入必须使用`torch.uint8`打包载体，并显式传入`x_dtype`和`weight_dtype`。
- WeightNZ转置场景必须先对源Weight执行`torch_npu.npu_format_cast(weight_source, 29)`，再执行`transpose(-1, -2)`；`weight_scale`必须对源shape的中间两维同步执行transpose。
- `scale_alg=1`仅支持FP8输出；`scale_alg=2`仅支持FLOAT4_E2M1输出，且`dst_type_max`只能取0.0或`[6.0, 12.0]`。FLOAT4_E1M2输出仅支持`scale_alg=0`。
- 支持M为0或N为0的空Tensor场景；该场景下允许K为0。

## 确定性计算

默认支持确定性计算。

## 调用说明

- MXFP8单算子模式调用：

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

- MXFP4打包`uint8`输入、WeightNZ转置、`scale_alg=2`单算子模式调用：

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer

  # K为偶数且不等于2；N为64的倍数。
  E = 3
  M = 52
  K = 82
  N = 128
  group_list = torch.tensor([1, 31, 20], dtype=torch.int64).npu()

  # MXFP4的两个逻辑元素打包到一个uint8中，因此x的物理末维为K / 2。
  x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8).npu()
  x_scale = torch.randint(120, 127, (M, math.ceil(K / 64), 2), dtype=torch.int8).npu()

  # 转置WeightNZ：源物理shape为(E, N, K / 2)。先转NZ，再转置末两维。
  weight_source = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8).npu()
  weight = torch_npu.npu_format_cast(weight_source, 29).transpose(-1, -2)
  weight_scale_source = torch.randint(
      120, 127, (E, N, math.ceil(K / 64), 2), dtype=torch.int8
  ).npu()
  weight_scale = weight_scale_source.transpose(-3, -2)

  y, y_scale = cann_ops_transformer.grouped_matmul_activation_quant(
      x,
      group_list,
      [weight],
      [weight_scale],
      "gelu_tanh",
      bias=None,
      x_scale=x_scale,
      group_list_type=1,
      quant_mode="mx",
      y_dtype=torch_npu.float4_e2m1fn_x2,
      round_mode="rint",
      scale_alg=2,
      dst_type_max=9.0,
      x_dtype=torch_npu.float4_e2m1fn_x2,
      weight_dtype=torch_npu.float4_e2m1fn_x2,
      weight_scale_dtype=torch_npu.float8_e8m0fnu,
      x_scale_dtype=torch_npu.float8_e8m0fnu,
  )
  torch.npu.synchronize()
  # y为打包uint8，物理shape为(M, N / 2)；y_scale的shape为(M, ceil(N / 64), 2)。
  print(y.shape, y.dtype, y_scale.shape, y_scale.dtype)
  ```
