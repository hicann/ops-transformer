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

对`group_list`指定的每个专家执行矩阵乘，`x_scale`和`weight_scale`的每个E8M0缩放因子作用于K轴上连续的32个元素。矩阵乘结果`C`的逻辑shape为`(M,N)`。

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
) -> Tensor
```

## 参数说明

表中shape为实际传入的PyTorch Tensor shape。`x`和`weight`使用INT8或UINT8作为存储载体，每个字节承载两个FP4 E2M1元素；scale使用一个字节承载一个E8M0编码。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | 激活矩阵，逻辑shape为`(M,K)`，ND格式，不支持空Tensor。 | torch.int8、torch.uint8 | `(M,K/2)` |
| x_scale | Tensor | 必选 | 激活的缩放因子，ND格式，不支持空Tensor。 | torch.int8、torch.uint8 | `(M,K/64,2)` |
| group_list | Tensor | 必选 | 专家分组信息，ND格式，可位于CPU或与x相同的NPU，不支持空Tensor。 | torch.int64 | `(E)` |
| weight | List[Tensor] | 必选 | 权重列表，长度为1，Tensor为FRACTAL_NZ格式，包含全部专家，不支持空Tensor。 | torch.int8、torch.uint8 | 非转置`(E,K,N/2)`；转置视图`(E,K/2,N)` |
| weight_scale | List[Tensor] | 必选 | 权重缩放因子列表，长度为1，ND格式，转置时须同步调整stride，不支持空Tensor。 | torch.int8、torch.uint8 | `(E,K/64,N,2)` |
| bias | List[Tensor]或None | 可选 | 预留的矩阵乘偏置，仅支持None或空列表，默认值为None。 | - | - |
| activation_type | str | 可选 | 激活类型，仅支持`"swiglu"`，默认值为`"swiglu"`。 | - | - |
| group_list_type | int | 可选 | 0表示累计结束位置，1表示各组token数，默认值为0。 | - | - |
| tuning_config | List[int]或None | 可选 | 预留的调优配置，仅支持None或空列表，默认值为None。 | - | - |
| swiglu_mode | int | 可选 | 仅支持2，表示前后分半、先裁剪再激活并应用alpha/bias的SwiGLU，默认值为2。 | - | - |
| clamp_limit | float | 可选 | 裁剪上限，须为不超过FLOAT32最大有限值的正有限数，转换为FLOAT32后须大于0，默认值为7.0。 | - | - |
| glu_alpha | float | 可选 | sigmoid输入缩放系数，须为有限数且绝对值不超过FLOAT32最大有限值，默认值为1.702。 | - | - |
| glu_bias | float | 可选 | 线性分支偏置，须为有限数且绝对值不超过FLOAT32最大有限值，默认值为1.0。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | SwiGLU计算结果，ND格式。 | torch.bfloat16 | `(M,N/2)` |

## 约束说明

- 接口用于专家前向计算，当前验证范围为单算子模式，TorchAir图模式尚未验证；不提供自动求导接口。
- M须大于0，K、N须为正数且满足128对齐，专家数E的取值范围为`[1,1024]`。x和weight的逻辑K须一致。
- 非转置时，先按`(E,K,N/2)`构造打包的权重Tensor，再转换为FRACTAL_NZ。
- 转置时，先按`(E,N,K/2)`构造打包的权重Tensor并转换为FRACTAL_NZ，再用`transpose(1,2)`得到传入视图；weight_scale先按`(E,N,K/64,2)`构造，再用`permute(0,2,1,3)`得到匹配视图。接口自动识别转置状态，不支持权重和scale转置状态不一致的输入。
- `group_list_type=0`时，分组列表须非负、单调非递减，最后一个值不大于M；`group_list_type=1`时，各组token数须非负且总和不超过M。允许专家的token数为0。
- CPU分组列表须完整覆盖M行：累计值的末项或各组token数之和须等于M。
- 除group_list可位于CPU外，x、x_scale、weight和weight_scale须位于同一NPU。

## 确定性计算

默认支持确定性计算。

## 调用示例

以下为非转置权重的单算子调用示例，输入为随机FP4 E2M1编码。

```python
import torch
import torch_npu
import cann_ops_transformer

M, K, N, E = 32, 128, 128, 1
torch.manual_seed(0)

# 每个字节承载两个FP4元素，E8M0编码127表示scale=1。
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
)

assert y.dtype == torch.bfloat16
assert tuple(y.shape) == (M, N // 2)
```
