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

  `cann_ops_transformer.grouped_matmul_swiglu_quant`融合GroupedMatmul、dequant、SwiGLU和quant。支持Ascend 950PR&950DT系列产品上的单Tensor、MXFP8或MX A8W4、weight FRACTAL_NZ场景，且`swiglu_mode`必须显式设置为2。

- **计算公式**：

  对`group_list`指定的每个expert，先执行GroupedMatmul并完成MX分块反量化：

  $$C_i[m,n]=\sum_{k=0}^{K-1}\left(X_i[m,k]\cdot s_i^X[m,\lfloor k/32\rfloor]\right)\left(W_i[k,n]\cdot s_i^W[\lfloor k/32\rfloor,n]\right)$$

  其中$W_i$为逻辑上的`[K,N]`权重，$s_i^X$和$s_i^W$为解码后的逻辑scale，每32个K元素共用一个scale。MX缩放发生在K方向求和之前，不能将随K变化的scale移到matmul结果之后相乘。

  再将$C_i$沿最后一维前后均分为$x_{glu}$和$x_{linear}$，按`swiglu_mode=2`计算：

  $$x_{glu}=min(x_{glu}, clampLimit)$$

  $$x_{linear}=clip(x_{linear}, -clampLimit, clampLimit)$$

  $$S_i=x_{glu}\cdot sigmoid(gluAlpha\cdot x_{glu})\cdot(x_{linear}+gluBias)$$

  MX A8W4可选传入FP32平滑因子`(E,N/2)`，此时令$T_i[m,j]=S_i[m,j]\times smooth\_scale[i,j]$；未传入时令$T_i=S_i$。MX A8W8分支不支持smooth。

  最后将$T_i$按最近偶数舍入转为BF16，沿输出N轴每32个元素构成一组并执行MXFP8动态量化，输出`y`和`y_scale`；每64个元素存储两个E8M0 scale。设组内BF16值为$V_j$，最大绝对值为$a=\max_j|V_j|$。输出类型固定为`torch.float8_e4m3fn`，其最大有限值为448。对处于正常可表示范围的非零有限$a$，`scale_alg`决定共享scale的计算方式：

  - `scale_alg=0`（OCP）：使用最大值的指数，不向上进位。

    $$e_{\mathrm{OCP}}=\lfloor\log_2 a\rfloor-8,\qquad y\_scale=2^{e_{\mathrm{OCP}}}$$

  - `scale_alg=1`（cuBLAS）：先除以目标类型最大有限值，再对缩放比的指数向上取整。

    $$r=\frac{a}{448},\qquad e_{\mathrm{cuBLAS}}=\lceil\log_2 r\rceil,\qquad y\_scale=2^{e_{\mathrm{cuBLAS}}}$$

  两种算法均使用各组的共享scale量化每个元素：

  $$y_j=\operatorname{cast}_{\mathrm{E4M3FN},\mathrm{rint}}\!\left(V_j\times\operatorname{BF16}\!\left(\frac{1}{y\_scale}\right)\right)$$

  scale以`torch_npu.float8_e8m0fnu`存储。`dst_type_max=0.0`表示使用E4M3FN默认最大值448，并非以0为除数。

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

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度 |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | 必选 | 左矩阵，仅支持非转置输入。 | `torch.float8_e4m3fn` | `(M, K)` |
| `weight` | List[Tensor] | 必选 | 右矩阵TensorList，长度必须为1，且唯一Tensor必须为FRACTAL_NZ格式，其第一维包含全部E个expert。MX A8W8支持非转置和转置布局；MX A8W4仅支持转置WeightNZ及配套转置weight_scale view，不支持其他任意非连续view。 | `torch.float8_e4m3fn`，或`torch.float4_e2m1fn_x2`（MX A8W4） | 3维，具体shape见下表。 |
| `weight_scale` | List[Tensor] | 必选 | `weight`的MX量化scale，TensorList长度必须为1。Tensor使用`torch.int8`或`torch.uint8`承载，通过`weight_scale_dtype`按E8M0解析；转置属性需与`weight`一致。 | `torch.int8`或`torch.uint8`（逻辑类型为`torch_npu.float8_e8m0fnu`） | 4维，具体shape见下表。 |
| `x_scale` | Tensor | 必选 | `x`的MX量化scale。Tensor使用`torch.int8`或`torch.uint8`承载，通过`x_scale_dtype`按E8M0解析。 | `torch.int8`或`torch.uint8`（逻辑类型为`torch_npu.float8_e8m0fnu`） | `(M, ceil(K / 64), 2)` |
| `group_list` | Tensor | 必选 | 分组信息，长度为expert数E，其语义由`group_list_type`决定。 | `torch.int64` | `(E,)` |
| `smooth_scale` | Tensor | 可选 | 平滑量化因子，默认值为`None`。仅MX A8W4支持，必须是与x同NPU的ND Tensor，按expert逐元素乘；不支持`(E,)`广播。旧MXFP8仍须为`None`。 | `torch.float32` | `(E,N/2)` |
| `weight_assist_matrix` | List[Tensor] | 可选 | 权重辅助矩阵，默认值为`None`。当前场景不支持，仅支持`None`或空List。 | - | - |
| `bias` | List[Tensor] | 可选 | 矩阵乘偏置列表，默认值为`None`。接口与GroupedMatmul保持一致；当前场景不支持有效bias，仅支持`None`或空List。 | - | - |
| `dequant_mode` | int | 可选 | 反量化模式，默认值为2，当前仅支持2，表示MX反量化。 | - | - |
| `dequant_dtype` | torch.dtype/int | 可选 | GroupedMatmul中间结果类型，默认值为`6`（表示`torch.float32`）。 | - | - |
| `quant_mode` | int | 可选 | 输出量化模式，默认值为2，当前仅支持2，表示MX量化。 | - | - |
| `quant_dtype` | torch.dtype/int | 可选 | 输出`y`的数据类型，默认值为`torch.float8_e4m3fn`，显式传入`None`时使用该默认类型。当前仅支持`torch.float8_e4m3fn`，不支持`torch.int8`、`torch.float8_e5m2`及其对应的数值枚举。 | - | - |
| `group_list_type` | int | 可选 | `group_list`的解释方式，默认值为0，0表示cumsum，1表示count。 | - | - |
| `tuning_config` | List[int] | 可选 | 调优参数，默认值为`None`。当前场景不支持，仅支持`None`或空List。 | - | - |
| `x_dtype` | torch.dtype/int | 可选 | `x`的逻辑数据类型，默认值为`None`。MXFP8场景无需传入；显式传入时仅支持`torch.float8_e4m3fn`。 | - | - |
| `weight_dtype` | torch.dtype/int | 可选 | `weight`的逻辑数据类型，默认值为`None`。MXFP8场景无需传入；MX A8W4可用原生`torch.float4_e2m1fn_x2`，或int8/uint8 packed存储并显式指定`torch_npu.float4_e2m1fn_x2`。每字节承载两个FP4，桥接层不做数值转换或重排。 | - | - |
| `weight_scale_dtype` | torch.dtype/int | 可选 | `weight_scale`的逻辑数据类型，默认值为`torch_npu.float8_e8m0fnu`，当前仅支持该类型。 | - | - |
| `x_scale_dtype` | torch.dtype/int | 可选 | `x_scale`的逻辑数据类型，默认值为`torch_npu.float8_e8m0fnu`，当前仅支持该类型。 | - | - |
| `swiglu_mode` | int | 可选 | SwiGLU计算模式，默认值为`None`。当前Torch Extension要求显式传入2：沿最后一维前后分半，前半为激活分支，后半为线性分支。 | - | - |
| `clamp_limit` | float | 可选 | SwiGLU裁剪上限，默认值为`None`，此时按7.0处理。必须为有限、可由`torch.float32`表示且转换后仍大于0的值；激活分支上限为`clamp_limit`，线性分支范围为`[-clamp_limit, clamp_limit]`。 | - | - |
| `glu_alpha` | float | 可选 | Sigmoid输入的缩放系数，默认值为`None`，此时按1.702处理。必须为有限且可由`torch.float32`表示的值。 | - | - |
| `glu_bias` | float | 可选 | 线性分支的偏置，默认值为`None`，此时按1.0处理。必须为有限且可由`torch.float32`表示的值。 | - | - |
| `round_mode` | str | 可选 | MX量化舍入模式，默认值为`"rint"`，支持`"rint"`或空字符串（按`"rint"`处理）；不接受内嵌NUL字符或`None`。 | - | - |
| `scale_alg` | int | 可选 | MX量化scale算法，默认值为0，0表示OCP实现，1表示cuBLAS实现；当前MXFP8输出仅支持0或1。 | - | - |
| `dst_type_max` | float | 可选 | 表示目标数据类型的最大值，默认值为0.0。当前MXFP8输出仅支持0.0。 | - | - |

张量的shape和布局要求如下：

| 参数名 | shape |
| --- | --- |
| `x` | `(M, K)` |
| `x_scale` | `(M, ceil(K / 64), 2)` |
| `group_list` | `(E,)` |
| `y` | `(M, N / 2)` |
| `y_scale` | `(M, ceil((N / 2) / 64), 2)` |

### MXFP8 weight布局

`source shape`指构造转置view之前的源Tensor形状，`view shape`指实际传入的Tensor形状，`view stride`指各维度索引增加1时对应的元素偏移量，`storage shape`指底层存储形状。表中的view stride用于区分转置与非转置布局。

| 参数名 | 场景 | source shape | view shape | view stride | storage shape |
| --- | --- | --- | --- | --- | --- |
| `weight` | 非转置 | `(E, K, N)` | `(E, K, N)` | `(K * N, N, 1)` | `(E, ceil(N / 32), ceil(K / 16), 16, 32)` |
| `weight` | 转置 | `(E, N, K)` | `(E, K, N)` | `(K * N, 1, K)` | `(E, ceil(K / 32), ceil(N / 16), 16, 32)` |
| `weight_scale` | 非转置 | `(E, ceil(K / 64), N, 2)` | `(E, ceil(K / 64), N, 2)` | `(ceil(K / 64) * N * 2, N * 2, 2, 1)` | `(E, ceil(K / 64), N, 2)` |
| `weight_scale` | 转置 | `(E, N, ceil(K / 64), 2)` | `(E, ceil(K / 64), N, 2)` | `(N * ceil(K / 64) * 2, 2, ceil(K / 64) * 2, 1)` | `(E, N, ceil(K / 64), 2)` |

转置场景先对形状为`(E, N, K)`的源`weight`执行`npu_format_cast(..., 29)`，再执行`transpose(-1, -2)`；形状为`(E, N, ceil(K / 64), 2)`的源`weight_scale`执行`transpose(-3, -2)`。因此，传入的转置和非转置Tensor具有相同的view shape，但view stride和storage布局不同。

### MX A8W4 weight布局

`x`为E4M3FN，`weight`为E2M1，输出为E4M3FN/E8M0。K和完整N均须为128的倍数。

MX A8W4仅支持转置weight及配套的转置weight_scale。下表中weight的source shape、view shape和view stride按Torch packed元素计数（每字节两个FP4），storage shape按ACL逻辑FP4元素计数。

| 参数名 | 场景 | source shape | view shape | view stride | storage shape |
| --- | --- | --- | --- | --- | --- |
| `weight` | 转置 | `(E, N, K / 2)` | `(E, K / 2, N)` | `(N * K / 2, 1, K / 2)` | `(E, K / 64, N / 16, 16, 64)` |
| `weight_scale` | 转置 | `(E, N, K / 64, 2)` | `(E, K / 64, N, 2)` | `(N * K / 64 * 2, 2, K / 64 * 2, 1)` | `(E, N, K / 64, 2)` |

转置场景先对形状为`(E, N, K / 2)`的packed源`weight`执行`npu_format_cast(..., 29)`，再执行`transpose(-1, -2)`；形状为`(E, N, K / 64, 2)`的源`weight_scale`执行`transpose(-3, -2)`。weight的ACL逻辑view shape为`(E, K, N)`。

以下示例使用INT8承载packed FP4权重，并通过`weight_dtype=torch_npu.float4_e2m1fn_x2`指定逻辑类型。

```python
import math
import torch
import torch_npu
import cann_ops_transformer

E, M, K, N = 4, 17, 128, 384
x = torch.randint(-4, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
x_scale = torch.full((M, math.ceil(K / 64), 2), 127, dtype=torch.uint8).npu()
group_list = torch.tensor([0, 8, 0, 9], dtype=torch.int64).npu()
smooth_scale = torch.ones((E, N // 2), dtype=torch.float32).npu()

for transpose_weight in (True,):
    packed_shape = (E, N, K // 2) if transpose_weight else (E, K, N // 2)
    source = torch.randint(-128, 128, packed_shape, dtype=torch.int8).npu()
    weight = torch_npu.npu_format_cast(source, 29)
    scale_shape = (E, N, math.ceil(K / 64), 2) if transpose_weight else (E, math.ceil(K / 64), N, 2)
    weight_scale = torch.full(scale_shape, 127, dtype=torch.uint8).npu()
    if transpose_weight:
        weight = weight.transpose(-1, -2)
        weight_scale = weight_scale.transpose(-3, -2)
    y, y_scale = cann_ops_transformer.grouped_matmul_swiglu_quant(
        x, [weight], [weight_scale], x_scale, group_list,
        swiglu_mode=2, group_list_type=1, smooth_scale=smooth_scale,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
    )
    torch.npu.synchronize()
    print(y.shape, y_scale.shape)
```

不需要smooth时传`None`。

## 返回值说明

| 返回值名 | 返回值类型 | 必选/可选 | 描述 | 数据类型 | 维度 |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | 必选 | SwiGLU计算后的MXFP8量化结果。 | `torch.float8_e4m3fn` | 2 |
| `y_scale` | Tensor | 必选 | `y`对应的MX量化scale。 | `torch_npu.float8_e8m0fnu` | 3 |

## 约束说明

- 适用场景：支持训练和推理场景。
- 调用方式：支持单算子模式调用。
- `group_list_type=0`时，`group_list`表示每个group在M轴上的累计结束位置，数值必须非负、单调不递减且不大于M；`group_list_type=1`时，`group_list`表示每个group的M轴长度，数值必须非负且总和不大于M。`group_list`未指定的`y`区域不参与计算。对应的`y_scale`如果为奇数场景均会填充0。

各维度的取值约束如下，与ACLNN接口一致：

| 维度 | 含义 | 取值约束 |
| --- | --- | --- |
| `E` | expert数量。 | 整数，取值范围为`[1, 1024]`。 |
| `M` | `x`的行数，即总token数。 | MXFP8为非负整数，支持`M=0`的空Tensor；MX A8W4为正整数。 |
| `K` | `x`与逻辑weight的矩阵乘归约维度，即输入特征数。 | MXFP8为正整数，无额外对齐要求；MX A8W4须为128的倍数。 |
| `N` | GroupedMatmul结果的列数，即SwiGLU拆分前的宽度。 | MXFP8为64的正整数倍；MX A8W4为128的正整数倍。SwiGLU输出宽度为`N / 2`。 |

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

- weight 转置输入时，先以`(E, N, K)`创建源Tensor，再执行
  `npu_format_cast(..., 29).transpose(-1, -2)`；`weight_scale`先以
  `(E, N, ceil(K / 64), 2)`创建源Tensor，再执行`transpose(-3, -2)`。例如：

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
