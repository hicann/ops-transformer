# grouped_matmul_activation

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

融合分组矩阵乘（Grouped Matrix Multiplication, GMM）和Swish门控线性单元（Swish-Gated Linear Unit, SwiGLU）激活，用于混合专家网络的专家前向计算，直接输出BF16结果。

对`group_list`指定的每个专家执行矩阵乘，`x_scale`和`weight_scale`的每个float8_e8m0缩放因子作用于K轴上连续的32个元素。矩阵乘结果`C`的view shape为`(M,N)`。

`swiglu_mode=2`表示沿N轴前后分半：前半作为激活分支，先做上限裁剪再激活；后半作为线性分支，对称裁剪后加偏置。计算公式为：

$$
u = \min(C[..., :N/2], clamp\_limit)
$$

$$
v = \operatorname{clip}(C[..., N/2:],-clamp\_limit,clamp\_limit)+glu\_bias
$$

$$
y = \operatorname{Cast}_{BF16}\left(\frac{u}{1+\exp(-glu\_alpha\times u)}\times v\right)
$$

`M`为输入token总数，`K`为输入特征维度，`N`为矩阵乘输出维度，`E`为专家个数。矩阵乘和SwiGLU采用FP32中间结果，最终转换为BF16，不产生输出scale。

## 函数原型

```python
cann_ops_transformer.grouped_matmul_activation(
    x,
    x_scale,
    group_list,
    weight,
    weight_scale,
    *,
    bias=None,
    activation_type="swiglu",
    group_list_type=0,
    tuning_config=None,
    swiglu_mode=2,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
    x_dtype=None,
    weight_dtype=None,
    x_scale_dtype=None,
    weight_scale_dtype=None,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | 激活矩阵，view shape为`(M,K)`，ND格式，不支持空Tensor。 | torch_npu.float4_e2m1fn_x2（可用torch.int8、torch.uint8承载） | `(M,K/2)` |
| x_scale | Tensor | 必选 | 激活的缩放因子，ND格式，不支持空Tensor。 | torch_npu.float8_e8m0fnu（可用torch.int8、torch.uint8承载） | `(M,K/64,2)` |
| group_list | Tensor | 必选 | 专家分组信息，ND格式，可位于CPU或与x相同的NPU，不支持空Tensor。 | torch.int64 | `(E)` |
| weight | List[Tensor] | 必选 | 权重列表，长度为1，Tensor为FRACTAL_NZ格式，包含全部专家，不支持空Tensor。 | torch_npu.float4_e2m1fn_x2（可用torch.int8、torch.uint8承载） | 非转置`(E,K,N/2)`；转置视图`(E,K/2,N)` |
| weight_scale | List[Tensor] | 必选 | 权重缩放因子列表，长度为1，ND格式，转置时须同步调整stride，不支持空Tensor。 | torch_npu.float8_e8m0fnu（可用torch.int8、torch.uint8承载） | `(E,K/64,N,2)` |
| bias | List[Tensor] | 可选 | 预留的矩阵乘偏置，仅支持None或空列表，默认值为None。 | - | - |
| activation_type | str | 可选 | 激活类型，仅支持`"swiglu"`，默认值为`"swiglu"`。 | - | - |
| group_list_type | int | 可选 | 0表示累计结束位置，1表示各组token数，默认值为0。 | - | - |
| tuning_config | List[int] | 可选 | 预留的调优配置，仅支持None或空列表，默认值为None。 | - | - |
| swiglu_mode | int | 可选 | 仅支持2，表示前后分半、先裁剪再激活并应用alpha/bias的SwiGLU，默认值为2。 | - | - |
| clamp_limit | float | 可选 | 裁剪上限，须为不超过FLOAT32最大有限值的正有限数，转换为FLOAT32后须大于0，默认值为7.0。 | - | - |
| glu_alpha | float | 可选 | sigmoid输入缩放系数，须为有限数且绝对值不超过FLOAT32最大有限值，默认值为1.702。 | - | - |
| glu_bias | float | 可选 | 线性分支偏置，须为有限数且绝对值不超过FLOAT32最大有限值，默认值为1.0。 | - | - |
| x_dtype | dtype或int | 可选 | x的实际数据类型，支持`torch_npu.float4_e2m1fn_x2`或整数`296`；默认None沿用float4_e2m1解释。 | - | - |
| weight_dtype | dtype或int | 可选 | weight的实际数据类型，支持`torch_npu.float4_e2m1fn_x2`或整数`296`；默认None沿用float4_e2m1解释。 | - | - |
| x_scale_dtype | dtype或int | 可选 | x_scale的实际数据类型，支持`torch_npu.float8_e8m0fnu`或整数`293`；默认None沿用float8_e8m0解释。 | - | - |
| weight_scale_dtype | dtype或int | 可选 | weight_scale的实际数据类型，支持`torch_npu.float8_e8m0fnu`或整数`293`；默认None沿用float8_e8m0解释。 | - | - |

### Tensor数据类型、shape与格式

下表同时列出view shape和storage shape；每个字节承载两个float4_e2m1元素或一个float8_e8m0元素。

| 参数名 | 逻辑数据类型 | view shape | storage shape | 数据格式 |
| --- | --- | --- | --- | --- |
| x | `torch_npu.float4_e2m1fn_x2` | `(M,K)` | `(M,K/2)` | ND |
| x_scale | `torch_npu.float8_e8m0fnu` | `(M,K/64,2)` | `(M,K/64,2)` | ND |
| weight | `torch_npu.float4_e2m1fn_x2` | `(E,K,N)` | 非转置：`(E,K,N/2)`；转置视图：`(E,K/2,N)` | FRACTAL_NZ |
| weight_scale | `torch_npu.float8_e8m0fnu` | `(E,K/64,N,2)` | `(E,K/64,N,2)` | ND |
| group_list | `torch.int64` | `(E,)` | `(E,)` | ND |
| y | `torch.bfloat16` | `(M,N/2)` | `(M,N/2)` | ND |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | SwiGLU计算结果，ND格式。 | torch.bfloat16 | `(M,N/2)` |

## 约束说明

- 当前仅支持单算子模式。
- M须大于0，K、N须为正数且满足128对齐。
- x和weight的逻辑K须一致。
- 专家数E的取值范围为`[1,1024]`。`group_list_type=0`时，分组列表须非负、单调非递减，最后一个值不大于M；`group_list_type=1`时，各组token数须非负且总和不超过M。允许专家的token数为0。
- weight和weightScale的构造方式如下：

| 场景 | weight构造方式 | weight_scale构造方式 |
| --- | --- | --- |
| 非转置 | 先按`(E,K,N/2)`构造打包的权重Tensor，再转换为FRACTAL_NZ。 | 按`(E,K/64,N,2)`构造。 |
| 转置 | 先按`(E,N,K/2)`构造打包的权重Tensor并转换为FRACTAL_NZ，再用`transpose(1,2)`得到传入视图。 | 先按`(E,N,K/64,2)`构造，再用`permute(0,2,1,3)`得到匹配视图。 |

- 不支持weight和weightScale转置状态不一致的输入。

## 确定性计算

默认支持确定性计算。

## 调用示例

以下为非转置权重的单算子调用示例，输入为随机float4_e2m1编码。

```python
import torch
import torch_npu
import cann_ops_transformer

M, K, N, E = 32, 128, 128, 1
torch.manual_seed(0)

# 每个字节承载两个FP4元素，float8_e8m0编码127表示scale=1。
x = torch.randint(-128, 128, (M, K // 2), dtype=torch.int8, device="npu")
x_scale = torch.full((M, K // 64, 2), 127, dtype=torch.uint8, device="npu")
weight = torch.randint(-128, 128, (E, K, N // 2), dtype=torch.int8, device="npu")
weight = torch_npu.npu_format_cast(weight, 29)
weight_scale = torch.full((E, K // 64, N, 2), 127, dtype=torch.uint8, device="npu")
group_list = torch.tensor([M], dtype=torch.int64, device="npu")

# 前后分半、裁剪和SwiGLU融合计算，直接返回一个BF16输出。
y = cann_ops_transformer.grouped_matmul_activation(
    x,
    x_scale,
    group_list,
    [weight],
    [weight_scale],
    activation_type="swiglu",
    group_list_type=0,
    swiglu_mode=2,
    clamp_limit=7.0,
    glu_alpha=1.702,
    glu_bias=1.0,
    x_dtype=torch_npu.float4_e2m1fn_x2,
    weight_dtype=torch_npu.float4_e2m1fn_x2,
    x_scale_dtype=torch_npu.float8_e8m0fnu,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
)

assert y.dtype == torch.bfloat16
assert tuple(y.shape) == (M, N // 2)
```
