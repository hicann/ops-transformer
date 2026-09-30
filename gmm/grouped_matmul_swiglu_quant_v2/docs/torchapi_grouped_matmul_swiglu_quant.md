# grouped_matmul_swiglu_quant

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

  `cann_ops_transformer.grouped_matmul_swiglu_quant`融合GroupedMatmul、dequant、SwiGLU和quant。当前仅支持Ascend 950PR/Ascend 950DT上的单Tensor、MXFP8、weight FRACTAL_NZ场景，且`swiglu_mode`必须显式设置为2。原V2场景请使用`torch_npu.npu_grouped_matmul_swiglu_quant_v2`。

- **计算公式**：

  对`group_list`指定的每个expert，先执行GroupedMatmul并完成MXFP8反量化：

  $$C_i[m,n]=\sum_{k=0}^{K-1}\left(X_i[m,k]\cdot s_i^X[m,\lfloor k/32\rfloor]\right)\left(W_i[k,n]\cdot s_i^W[\lfloor k/32\rfloor,n]\right)$$

  其中$W_i$为逻辑上的`[K,N]`权重，$s_i^X$和$s_i^W$为解码后的逻辑scale，每32个K元素共用一个scale。MX缩放发生在K方向求和之前，不能将随K变化的scale移到matmul结果之后相乘。

  再将$C_i$沿最后一维前后均分为$x_{glu}$和$x_{linear}$，按`swiglu_mode=2`计算：

  $$x_{glu}=min(x_{glu}, clampLimit)$$

  $$x_{linear}=clip(x_{linear}, -clampLimit, clampLimit)$$

  $$S_i=x_{glu}\cdot sigmoid(gluAlpha\cdot x_{glu})\cdot(x_{linear}+gluBias)$$

  最后将$S_i$按最近偶数舍入转为BF16，沿输出N轴每32个元素构成一组并执行MXFP8动态量化，输出`y`和`y_scale`；每64个元素存储两个E8M0 scale。设组内BF16值为$V_j$，最大绝对值为$a=\max_j|V_j|$。输出类型固定为`torch.float8_e4m3fn`，其最大有限值为448。对处于正常可表示范围的非零有限$a$，`scale_alg`决定共享scale的计算方式：

  - `scale_alg=0`（OCP）：使用最大值的指数，不向上进位。

    $$e_{\mathrm{OCP}}=\lfloor\log_2 a\rfloor-8,\qquad y\_scale=2^{e_{\mathrm{OCP}}}$$

  - `scale_alg=1`（cuBLAS）：先除以目标类型最大有限值，再对缩放比的指数向上取整。

    $$r=\frac{a}{448},\qquad e_{\mathrm{cuBLAS}}=\lceil\log_2 r\rceil,\qquad y\_scale=2^{e_{\mathrm{cuBLAS}}}$$

  两种算法均使用各组的共享scale量化每个元素：

  $$y_j=\operatorname{cast}_{\mathrm{E4M3FN},\mathrm{rint}}\!\left(V_j\times\operatorname{BF16}\!\left(\frac{1}{y\_scale}\right)\right)$$

  scale以`torch_npu.float8_e8m0fnu`存储；低于E8M0编码下界时截断到编码0。全零组使用编码0（解码值为$2^{-127}$），含Inf/NaN的组使用编码255。实际计算使用BF16倒数乘法，公式中的倒数不能替换为无限精度除法。`dst_type_max=0.0`表示使用E4M3FN默认最大值448，并非以0为除数。

## 函数原型

```python
cann_ops_transformer.grouped_matmul_swiglu_quant(
    x,
    weight,
    weight_scale,
    x_scale,
    group_list,
    *,
    smooth_scale=None,
    weight_assist_matrix=None,
    bias=None,
    dequant_mode=2,
    dequant_dtype=6,
    quant_mode=2,
    quant_dtype=torch.float8_e4m3fn,
    group_list_type=0,
    tuning_config=None,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
    x_scale_dtype=torch_npu.float8_e8m0fnu,
    swiglu_mode=None,
    clamp_limit=None,
    glu_alpha=None,
    glu_bias=None,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | 左矩阵，仅支持非转置输入。 | `torch.float8_e4m3fn` | `(M, K)` |
| `weight` | List[Tensor] | 必选 | 右矩阵TensorList，长度必须为1，且唯一Tensor必须为FRACTAL_NZ格式。非转置和转置的API入口view均为`(E, K, N)`；转置场景由`(E, N, K)` source执行末两轴转置得到。仅支持约束说明中的NZ/ZN stride和storage shape。 | `torch.float8_e4m3fn` | API入口view：`(E, K, N)` |
| `weight_scale` | List[Tensor] | 必选 | `weight`的MX量化scale，TensorList长度必须为1。Tensor使用`torch.int8`或`torch.uint8`承载，通过`weight_scale_dtype`按E8M0解析；转置属性需与`weight`一致。转置场景由`(E, N, ceil(K / 64), 2)` source交换中间两轴得到入口view。 | `torch.int8`或`torch.uint8`（逻辑类型为`torch_npu.float8_e8m0fnu`） | API入口view：`(E, ceil(K / 64), N, 2)` |
| `x_scale` | Tensor | 必选 | `x`的MX量化scale。Tensor使用`torch.int8`或`torch.uint8`承载，通过`x_scale_dtype`按E8M0解析。 | `torch.int8`或`torch.uint8`（逻辑类型为`torch_npu.float8_e8m0fnu`） | `(M, ceil(K / 64), 2)` |
| `group_list` | Tensor | 必选 | 分组信息，长度为expert数E，其语义由`group_list_type`决定。 | `torch.int64` | `(E,)` |
| `smooth_scale` | Tensor | 可选 | 平滑量化因子，默认值为`None`。当前场景不支持，必须传`None`。 | - | - |
| `weight_assist_matrix` | List[Tensor] | 可选 | 权重辅助矩阵，默认值为`None`。当前场景不支持，仅支持`None`或空List。 | - | - |
| `bias` | List[Tensor] | 可选 | 矩阵乘偏置列表，默认值为`None`。接口与GroupedMatmul保持一致；当前V3场景不支持有效bias，仅支持`None`或空List。 | - | - |
| `dequant_mode` | int | 可选 | 反量化模式，默认值为2，当前仅支持2，表示MX反量化。 | int | - |
| `dequant_dtype` | torch.dtype/int | 可选 | GroupedMatmul中间结果类型，默认值为`6`（表示`torch.float32`）；也支持显式传入`torch.float32`或`0`。 | `6` | - |
| `quant_mode` | int | 可选 | 输出量化模式，默认值为2，当前仅支持2，表示MX量化。 | int | - |
| `quant_dtype` | torch.dtype/int | 可选 | 输出`y`的数据类型，默认值为`torch.float8_e4m3fn`，当前仅支持`torch.float8_e4m3fn`。 | `torch.float8_e4m3fn` | - |
| `group_list_type` | int | 可选 | `group_list`的解释方式，默认值为0。0表示cumsum，1表示count。 | int | - |
| `tuning_config` | List[int] | 可选 | 调优参数，默认值为`None`。当前场景不支持，仅支持`None`或空List。 | int | - |
| `x_dtype` | torch.dtype/int | 可选 | `x`的逻辑数据类型，默认值为`None`。MXFP8场景无需传入；显式传入时仅支持`torch.float8_e4m3fn`。 | `torch.float8_e4m3fn` | - |
| `weight_dtype` | torch.dtype/int | 可选 | `weight`的逻辑数据类型，默认值为`None`。MXFP8场景无需传入；显式传入时仅支持`torch.float8_e4m3fn`。 | `torch.float8_e4m3fn` | - |
| `weight_scale_dtype` | torch.dtype/int | 可选 | `weight_scale`的逻辑数据类型，默认值为`torch_npu.float8_e8m0fnu`，当前仅支持该类型。 | `torch_npu.float8_e8m0fnu` | - |
| `x_scale_dtype` | torch.dtype/int | 可选 | `x_scale`的逻辑数据类型，默认值为`torch_npu.float8_e8m0fnu`，当前仅支持该类型。 | `torch_npu.float8_e8m0fnu` | - |
| `swiglu_mode` | int | 可选 | SwiGLU计算模式，默认值为`None`。当前Torch Extension要求显式传入2：沿最后一维前后分半，前半为激活分支，后半为线性分支。 | int | - |
| `clamp_limit` | float | 可选 | SwiGLU裁剪上限，默认值为`None`，此时按7.0处理。必须为有限正数；激活分支上限为`clamp_limit`，线性分支范围为`[-clamp_limit, clamp_limit]`。 | float | - |
| `glu_alpha` | float | 可选 | Sigmoid输入的缩放系数，默认值为`None`，此时按1.702处理。必须为有限且可由`torch.float32`表示的值。 | float | - |
| `glu_bias` | float | 可选 | 线性分支的偏置，默认值为`None`，此时按1.0处理。必须为有限且可由`torch.float32`表示的值。 | float | - |
| `round_mode` | str | 可选 | MX量化舍入模式，默认值为`"rint"`，当前仅支持`"rint"`。 | string | - |
| `scale_alg` | int | 可选 | MX量化scale算法，默认值为0。0表示OCP实现，1表示cuBLAS实现；当前MXFP8输出仅支持0或1。 | int | - |
| `dst_type_max` | float | 可选 | 表示目标数据类型的最大值，默认值为0.0。当前MXFP8输出仅支持0.0。 | float | - |

## 返回值说明

| 返回值名 | 返回值类型 | 必选/可选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | 必选 | SwiGLU计算后的MXFP8量化结果。 | `torch.float8_e4m3fn` | `(M, N / 2)` |
| `y_scale` | Tensor | 必选 | `y`对应的MX量化scale。 | `torch_npu.float8_e8m0fnu` | `(M, ceil((N / 2) / 64), 2)` |

## 约束说明

- 适用场景：支持训练和推理场景。
- 调用方式：支持单算子模式调用。
- 仅支持MXFP8 weight FRACTAL_NZ场景，`x`和`weight`仅支持`torch.float8_e4m3fn`；`weight_scale`和`x_scale`使用`torch.int8`或`torch.uint8`承载`torch_npu.float8_e8m0fnu`数据。
- `weight`和`weight_scale`的List长度必须均为1；唯一Tensor的第一维包含全部E个expert，不要将各expert拆成多个Tensor。
- `E`取值范围为[1, 1024]。`M`和`K`必须大于0。`N`必须大于0且为64的整数倍。
- 非转置和转置`weight`的API入口view shape均为`(E, K, N)`。非转置使用连续stride `(K * N, N, 1)`，storage shape为`(E, ceil(N / 32), ceil(K / 16), 16, 32)`；转置场景先构造`(E, N, K)` source并转换为FRACTAL_NZ，再交换末两轴形成stride `(K * N, 1, K)`的入口view，storage shape为`(E, ceil(K / 32), ceil(N / 16), 16, 32)`。
- 非转置和转置`weight_scale`的API入口view shape均为`(E, ceil(K / 64), N, 2)`。非转置source/storage与入口view相同，stride为`(ceil(K / 64) * N * 2, N * 2, 2, 1)`；转置场景先构造`(E, N, ceil(K / 64), 2)` source/storage，再交换中间两轴形成stride `(N * ceil(K / 64) * 2, 2, ceil(K / 64) * 2, 1)` 的入口view。
- 转置场景中，`weight`和`weight_scale`的转置属性必须一致；即使`K=N`，也通过Tensor的view stride识别。`weight`不支持规定NZ/ZN布局之外的任意非连续view。
- `group_list_type=0`时，`group_list`表示每个group在M轴上的累计结束位置，数值必须非负、单调不递减且不大于M；`group_list_type=1`时，`group_list`表示每个group的M轴长度，数值必须非负且总和不大于M。`group_list`未指定的输出区域不会被更新。
- `dequant_mode=2`、`dequant_dtype=torch.float32`、`quant_mode=2`、`quant_dtype=torch.float8_e4m3fn`、`swiglu_mode=2`。
- `quant_dtype`显式传入`None`时使用默认的`torch.float8_e4m3fn`输出类型；不支持`torch.int8`、`torch.float8_e5m2`及其对应的数值枚举，数值`1`也不作为FP8输出的别名。
- `weight_assist_matrix`、`bias`、`smooth_scale`和`tuning_config`当前不支持有效数据。
- `round_mode`仅支持`"rint"`；底层ACLNN接口传入`nullptr`或空字符串时等效于`"rint"`。

## 确定性计算

默认支持确定性计算。

## 调用说明

- MXFP8 weight FRACTAL_NZ单算子模式调用：

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer

  E = 1
  M = 64
  K = 128
  N = 128

  x = torch.randint(-4, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  weight = torch.randint(-4, 5, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn)
  weight_scale = torch.randint(
      120, 127, (E, math.ceil(K / 64), N, 2), dtype=torch.int8
  ).npu()
  x_scale = torch.randint(
      120, 127, (M, math.ceil(K / 64), 2), dtype=torch.int8
  ).npu()
  group_list = torch.tensor([M], dtype=torch.int64).npu()

  y, y_scale = cann_ops_transformer.grouped_matmul_swiglu_quant(
      x,
      [weight],
      [weight_scale],
      x_scale,
      group_list,
      swiglu_mode=2,
      weight_scale_dtype=torch_npu.float8_e8m0fnu,
      x_scale_dtype=torch_npu.float8_e8m0fnu,
      scale_alg=0,
  )
  torch.npu.synchronize()
  print(y.shape, y.dtype, y_scale.shape, y_scale.dtype)
  ```

- weight 转置（ZN）输入时，必须先以 `(E, N, K)` 创建源 Tensor，再执行
  `npu_format_cast(..., 29).transpose(-1, -2)`。其中 format cast 前的源 Tensor 决定 ZN
  物理 storage，末两维转置则创建 GMM 入口统一使用的 `(E, K, N)` 转置 view；两者缺一不可。
  `weightScale` 同样先创建 `(E, N, ceil(K / 64), 2)`，再执行
  `transpose(-3, -2)`。不要先以 `(E, K, N)` 创建并 format cast 后再转置，否则底层 NZ storage
  仍是非转置布局，无法满足转置输入的 storage shape 约束。例如：

  ```python
  weight_source = torch.randint(
      -4, 5, (E, N, K), dtype=torch.int8
  ).to(torch.float8_e4m3fn).npu()
  weight = torch_npu.npu_format_cast(
      weight_source, 29, customize_dtype=torch.float8_e4m3fn
  ).transpose(-1, -2)
  weight_scale_source = torch.randint(
      120, 127, (E, N, math.ceil(K / 64), 2), dtype=torch.int8
  ).npu()
  weight_scale = weight_scale_source.transpose(-3, -2)
  ```
