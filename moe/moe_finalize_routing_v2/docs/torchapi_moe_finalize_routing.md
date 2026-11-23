# moe_finalize_routing

## 产品支持情况

- <term>Ascend 950PR&950DT系列产品</term>：支持
- <term>Atlas A3系列产品</term>：支持
- <term>Atlas A2系列产品</term>：支持
- <term>Atlas 200I/500 A2推理产品</term>：不支持
- <term>Atlas推理系列产品</term>：支持
- <term>Atlas训练系列产品</term>：不支持

## 功能说明

`moe_finalize_routing`是基于`torch_npu`的`cann_ops_transformer`扩展接口。该接口用于MoE（Mixture of Experts）计算的最后阶段，将各专家FFN的输出结果按路由权重加权合并，还原为原始token序列。

输入为 `FLOAT4_E2M1` 时，融合 E2M1/E8M0 反量化与路由合并，固定输出 BF16，无需指定输出类型参数。

计算公式：

输入为 FLOAT16、BFLOAT16、FLOAT32 时：

$$
\begin{aligned} &\mathrm{expertId} = \mathrm{expertIdx}[i,k] \\ &\text{if } \mathrm{expertId} \in \mathrm{zero\_expert\_range}:\\&\quad \text{skip}; \\&\text{elif } \mathrm{expertId} \in \mathrm{copy\_expert\_range}: \\ &\quad x = x[i] \\ &\quad \mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=1}^{K}\mathrm{scales}_{i,k} \cdot \left(x + \mathrm{bias}_{\mathrm{expertId},j}\right) \\ &\text{elif } \mathrm{expertId} \in \mathrm{constant\_expert\_range}: \\&\quad x = \alpha_1 \cdot x[i] + \alpha_2 \cdot v \\ &\quad \mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=1}^{K}\mathrm{scales}_{i,k} \cdot \left(x + \mathrm{bias}_{\mathrm{expertId},j}\right) \\ &\text{else}: \\ &\quad \mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=0}^{K}\mathrm{scales}_{i,k} \cdot\big(\mathrm{expandedX}_{\mathrm{expandedRowIdx}_{i\cdot K+k},j} + \mathrm{bias}_{\mathrm{expertId}_{i\cdot K+k},j}\big) \end{aligned}
$$

## 函数原型

```python
cann_ops_transformer.ops.moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    x=None,
    alpha1=None,
    alpha2=None,
    v=None,
    drop_pad_mode=0,
    zero_expert_range=None,
    copy_expert_range=None,
    constant_expert_range=None,
    k=1,
    *,
    expanded_x_scale=None,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| --- | --- | --- | --- | --- | --- |
| `expanded_x` | Tensor | 必选 | MoE FFN 输出。shape 为 `(NUM_ROWS*K,H)` 或 `(E,C,H)`；输入为 FLOAT4_E2M1 时，drop_pad_mode 为 0/2 时为 `(R,H/2)`，drop_pad_mode 为 1/3 时为 `(E,C,H/2)` 且 R=E*C。 | FLOAT16、BFLOAT16、FLOAT32、FLOAT4_E2M1、UINT8（E2M1 编码） | 默认：`(NUM_ROWS*K,H)` / `(E,C,H)`；FLOAT4_E2M1：`(R,H/2)` / `(E,C,H/2)` |
| `expanded_row_idx` | Tensor | 必选 | token索引映射，即公式中的expandedRowIdx，输入不为 FLOAT4_E2M1 时，drop_pad_mode为0或2时，值域为`[0, NUM_ROWS * K - 1]`；为1或3时，值域为`[-1, E * C - 1]`，其中-1表示该token被丢弃。 | INT32 | `(NUM_ROWS * K,)` |
| `x1` | Tensor | 可选 | 第一个共享专家输出，即公式中的x1，shape与输出一致。 | 与 expanded_x 一致；输入为 FLOAT4_E2M1 时为 BFLOAT16 | `(NUM_ROWS, H)` |
| `x2` | Tensor | 可选 | 第二个共享专家输出，即公式中的x2，shape与输出一致。x1未传入时，x2也不能传入。 | 与 expanded_x 一致；输入为 FLOAT4_E2M1 时为 BFLOAT16 | `(NUM_ROWS, H)` |
| `bias` | Tensor | 可选 | 偏置量。bias存在时，expert_idx必须同时存在。 | 与 expanded_x 一致；输入为 FLOAT4_E2M1 时为 BFLOAT16 | `(E, H)` |
| `scales` | Tensor | 可选 | 路由权重。当 `expanded_x` 为 MXFP4 时，仅支持 FLOAT32。 | FLOAT16、BFLOAT16、FLOAT32 | `(NUM_ROWS,K)` |
| `expert_idx` | Tensor | 可选 | 专家索引，即公式中的expertIdx，bias存在时必须同时传入。值域为`[0, E-1]`。 | INT32 | `(NUM_ROWS, K)` |
| `x` | Tensor | 可选 | copy expert和constant expert场景的输入。MXFP4 输入时必须为 `None`。 | 非量化输入时与 `expanded_x` 一致 | `(NUM_ROWS, H)` |
| `alpha1` | Tensor | 可选 | constant expert的第一缩放系数。MXFP4 输入时必须为 `None`。 | 非量化输入时与 `expanded_x` 一致 | `(constant_expert_range_num, H)` |
| `alpha2` | Tensor | 可选 | constant expert的第二缩放系数。MXFP4 输入时必须为 `None`。 | 非量化输入时与 `expanded_x` 一致 | `(constant_expert_range_num, H)` |
| `v` | Tensor | 可选 | constant expert的偏移向量。MXFP4 输入时必须为 `None`。 | 非量化输入时与 `expanded_x` 一致 | `(constant_expert_range_num, H)` |
| `drop_pad_mode` | int | 可选 | 丢弃/填充模式及expandedRowIdx的排列方式。`0`：drop less，按列排列；`1`：drop pad，按列排列；`2`：drop less，按行排列；`3`：drop pad，按行排列。默认值为0。 | int64 | - |
| `zero_expert_range` | list[int] | 可选 | zero expert的范围`[start, end)`，左闭右开。该范围内的专家输出被跳过。MXFP4 输入时必须为 `None` 或空列表。 | list[int64] | 非量化输入传入区间时长度为 2 |
| `copy_expert_range` | list[int] | 可选 | copy expert的范围`[start, end)`，左闭右开。该范围内使用`x[i]`替代expandedX。MXFP4 输入时必须为 `None` 或空列表。 | list[int64] | 非量化输入传入区间时长度为 2 |
| `constant_expert_range` | list[int] | 可选 | constant expert的范围`[start, end)`，左闭右开。该范围内使用`alpha1 * x[i] + alpha2 * v`替代expandedX。MXFP4 输入时必须为 `None` 或空列表。 | list[int64] | 非量化输入传入区间时长度为 2 |
| `k` | int | 可选 | Top-K。expanded_x 不为 FLOAT4_E2M1 且 scales 非空时显式传入的 k 应与 scales 第二维一致；输入为 FLOAT4_E2M1 时直接使用属性 k，NUM_ROWS=expanded_row_idx.numel()/K，不从 scale 形状推导 K。默认值为 1。 | int64 | - |
| `expanded_x_scale` | Tensor | 输入为 FLOAT4_E2M1 时必选 | 块反量化系数，每 32 个元素共用一个，物理行与 expanded_x 对应。非 MXFP4 输入会忽略此参数。 | FLOAT8_E8M0 或 UINT8（E8M0 编码） | `(R,ceil(H/64),2)` |

## 返回值说明

返回 Tensor，shape 为 `(NUM_ROWS,H)`。数据类型与 expanded_x 一致；输入为 FLOAT4_E2M1 时固定为 `torch.bfloat16`，NUM_ROWS 由索引长度除以 k 得到。直接执行和 Meta/FakeTensor 推导使用相同的输出形状与类型规则。输入为 FLOAT4_E2M1 时，TorchAir 转换器使用 E2M1/E8M0 类型声明，图执行支持原生量化类型或 uint8 编码输入。

## 约束说明

- `expanded_x`必须是2D或3D Tensor。
- `expanded_row_idx`必须是1D Tensor，若显式传入`k`，则元素个数必须能被`k`整除。
- `drop_pad_mode`取值范围为`[0, 3]`。
- `x1`未传入时，`x2`也不能传入。
- `bias`存在时，`expert_idx`必须同时存在。
- `expanded_x` 不为 FLOAT4_E2M1 且 `scales`不为空时，若显式传入`k`，则`k`必须与`scales`的第二维一致。
- `scales`为空时，则必须传入`k`。
- `zero_expert_range`、`copy_expert_range`、`constant_expert_range`三个范围不能重叠。
- **自动反向（autograd）**：输入为 FLOAT16、BFLOAT16、FLOAT32 时，当`expanded_x`、`scales`等可微输入的`requires_grad`为True时支持自动反向，反向算子为[moe\_finalize\_routing\_grad](../../moe_finalize_routing_v2_grad/docs/torchapi_moe_finalize_routing_grad.md)（封装`aclnnMoeFinalizeRoutingV2Grad`）。自动反向约束如下：
  - 仅在正向退化为aclnnMoeFinalizeRoutingV2场景时支持，即不传入特殊专家相关输入`x`、`alpha1`、`alpha2`、`v`，且`zero_expert_range`、`copy_expert_range`、`constant_expert_range`均为None或无效（如`[-1, -1]`）。当使用了上述特殊专家相关特性时，调用自动反向会抛出`NotImplementedError`。
  - `drop_pad_mode`仅支持0或1（列排列模式），不支持2或3（行排列模式）。
  - 仅计算`grad_expanded_x`和`grad_scales`；`x1`、`x2`、`bias`的梯度不会被计算（返回`None`）。如需对`x1`/`x2`求梯度，建议使用外部残差加法替代，而非将其作为正向输入。
  - 正向`expanded_row_idx`采用`(K, R)`布局，反向算子采用`(R, K)`布局，自动反向下框架会自动转置。
  - **产品支持差异**：`aclnnMoeFinalizeRoutingV2Grad`在<term>Atlas推理系列产品</term>上不支持，尽管正向`moe_finalize_routing`在该产品上支持。因此在<term>Atlas推理系列产品</term>上，`moe_finalize_routing`不支持自动反向。

- 普通浮点输入调用 `aclnnMoeFinalizeRoutingV4`，FLOAT4_E2M1 输入调用 `aclnnMoeFinalizeRoutingV5`。
- 输入为 `FLOAT4_E2M1` 时，expanded_x 使用 `torch.float4_e2m1fn_x2` 或存储 E2M1 编码的 `torch.uint8`，每个 Torch 元素存两个 E2M1 值，最后一维为打包宽度 H/2；H 为逻辑输出宽度，必须为正偶数。expanded_x_scale 使用 `torch.float8_e8m0fnu` 或存储 E8M0 编码的 `torch.uint8`。uint8 输入沿用 FLOAT4_E2M1 的形状、输出及可选输入约束；已有量化张量可使用 `.view(torch.uint8)` 保留编码，不能使用 `.to(torch.uint8)` 进行数值转换。所有张量使用 ND；expanded_x 和 expanded_x_scale 必须连续。
- 输入为 `FLOAT4_E2M1` 时，R 为实际物理行数，裁剪后可小于 NUM_ROWS*K；每字节低 4 bit 对应偶数列。expanded_x_scale 为 `(R,ceil(H/64),2)`，每个系数对应 32 个元素，行尾无对应数据的预留 scale 不参与计算。该布局与 moe_init_routing_v3 的 MXFP4 输出一致，该算子当前的 FLOAT4_E2M1 路径不提供 Drop/Pad 输入。
- 输入为 `FLOAT4_E2M1` 时，k 必须大于 0，索引长度必须能被 k 整除。drop_pad_mode 为 0/1 时按 `expanded_row_idx[i+t*NUM_ROWS]` 取索引，为 2/3 时按 `expanded_row_idx[i*K+t]` 取索引；数据和 scale 使用同一个物理行 r。超出 `[0,R)` 的路由整路跳过，包括 bias。
- 输入为 `FLOAT4_E2M1` 时，x1/x2 为 `(NUM_ROWS,H)`，bias 为 `(E_bias,H)`，均使用 BFLOAT16。bias 依赖 INT32 的 expert_idx `(NUM_ROWS,K)`，有效专家索引在 `[0,E_bias)`。
- MXFP4 输入（FLOAT4_E2M1 或其 UINT8 编码）当前不支持零计算专家、拷贝专家和常量专家。`x`、`alpha1`、`alpha2`、`v` 必须为 `None`，三个专家区间 `zero_expert_range`、`copy_expert_range`、`constant_expert_range` 必须为 `None` 或空列表。
- 输入为 `FLOAT4_E2M1` 时，expanded_x_scale 表示反量化系数。传入 scales 时在加 bias 后进行路由加权；已融合的 bias 不应再次传入。反量化数据、可选 bias 及 x1/x2 使用 FP32 累加，最后一次转换为 BF16。不提供 out_dtype 参数，不支持自动反向，包括辅助输入的梯度计算。

底层约束见 [ACLNN V5 文档](aclnnMoeFinalizeRoutingV5.md)。

## 调用示例

```python
import torch
import torch_npu
from cann_ops_transformer import moe_finalize_routing

NUM_ROWS = 4
K = 2
H = 8
E = 8
CE_NUM = 2  # constant_expert_range_num = end - start

expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu")
row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu")
x1 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
x2 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
bias = torch.randn(E, H, dtype=torch.float32, device="npu")
expert_idx = torch.tensor(
    [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32, device="npu"
)
x = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
alpha1 = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")
alpha2 = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")
v = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")

out = moe_finalize_routing(
    expanded_x,
    row_idx,
    x1=x1,
    x2=x2,
    bias=bias,
    scales=scales,
    expert_idx=expert_idx,
    x=x,
    alpha1=alpha1,
    alpha2=alpha2,
    v=v,
    drop_pad_mode=0,
    zero_expert_range=[0, 2],
    copy_expert_range=[2, 4],
    constant_expert_range=[6, 8],
    k=K,
)
print(out.shape)  # torch.Size([4, 8])
```

- 自动反向调用（训练场景）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_finalize_routing

  torch_npu.npu.set_device(0)

  NUM_ROWS = 4
  K = 2
  H = 8

  # expanded_x.requires_grad=True 时自动启用autograd
  expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu",
                           requires_grad=True)
  row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
  scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu",
                       requires_grad=True)

  # 正向：不传入特殊专家相关输入（x/alpha1/alpha2/v），不设置zero_expert_range
  y = moe_finalize_routing(expanded_x, row_idx, scales=scales, k=K)

  # 反向自动触发 moe_finalize_routing_grad
  loss = y.sum()
  loss.backward()
  print(expanded_x.grad.shape)  # torch.Size([8, 8])
  print(scales.grad.shape)      # torch.Size([4, 2])
  ```

## 确定性计算

默认支持确定性计算。
