# moe_finalize_routing

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

`moe_finalize_routing`是基于`torch_npu`的`cann_ops_transformer`扩展接口。该接口用于MoE（Mixture of Experts）计算的最后阶段，将各专家FFN的输出结果按路由权重加权合并，还原为原始token序列。

计算公式：

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
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `expanded_x` | Tensor | 必选 | MoE FFN的输出，即公式中的expandedX。drop less场景shape为`(NUM_ROWS * K, H)`，drop pad场景shape为`(E, C, H)`。 | FLOAT16、BFLOAT16、FLOAT32 | `(NUM_ROWS * K, H)`或`(E, C, H)` |
| `expanded_row_idx` | Tensor | 必选 | token索引映射，即公式中的expandedRowIdx。drop_pad_mode为0或2时，值域为`[0, NUM_ROWS * K - 1]`；为1或3时，值域为`[-1, E * C - 1]`，其中-1表示该token被丢弃。 | INT32 | `(NUM_ROWS * K,)` |
| `x1` | Tensor | 可选 | 第一个共享专家输出，即公式中的x1。shape与输出一致。 | 与expanded_x一致 | `(NUM_ROWS, H)` |
| `x2` | Tensor | 可选 | 第二个共享专家输出，即公式中的x2。shape与输出一致。x1未传入时，x2也不能传入。 | 与expanded_x一致 | `(NUM_ROWS, H)` |
| `bias` | Tensor | 可选 | 偏置量。bias存在时，expert_idx必须同时存在。 | 与expanded_x一致 | `(E, H)` |
| `scales` | Tensor | 可选 | 路由权重系数。 | FLOAT16、BFLOAT16、FLOAT32 | `(NUM_ROWS, K)` |
| `expert_idx` | Tensor | 可选 | 专家索引，即公式中的expertIdx。bias存在时必须同时传入。值域为`[0, E-1]`。 | INT32 | `(NUM_ROWS, K)` |
| `x` | Tensor | 可选 | copy expert和constant expert场景的输入。 | 与expanded_x一致 | `(NUM_ROWS, H)` |
| `alpha1` | Tensor | 可选 | constant expert的第一缩放系数。 | 与expanded_x一致 | `(constant_expert_range_num, H)` |
| `alpha2` | Tensor | 可选 | constant expert的第二缩放系数。 | 与expanded_x一致 | `(constant_expert_range_num, H)` |
| `v` | Tensor | 可选 | constant expert的偏移向量。 | 与expanded_x一致 | `(constant_expert_range_num, H)` |
| `drop_pad_mode` | int | 可选 | 丢弃/填充模式及expandedRowIdx的排列方式。`0`：drop less，按列排列；`1`：drop pad，按列排列；`2`：drop less，按行排列；`3`：drop pad，按行排列。默认值为0。 | int64 | - |
| `zero_expert_range` | list[int] | 可选 | zero expert的范围`[start, end)`，左闭右开。该范围内的专家输出被跳过。 | list[int64] | 长度为2 |
| `copy_expert_range` | list[int] | 可选 | copy expert的范围`[start, end)`，左闭右开。该范围内使用`x[i]`替代expandedX。 | list[int64] | 长度为2 |
| `constant_expert_range` | list[int] | 可选 | constant expert的范围`[start, end)`，左闭右开。该范围内使用`alpha1 * x[i] + alpha2 * v`替代expandedX。 | list[int64] | 长度为2 |
| `k` | int | 可选 | 每个token选出的top-K专家个数。scales不为空时，若显式传入k则k必须与scales的第二维一致。默认值为1。 | int64 | - |

## 返回值说明

返回一个Tensor，shape为`(NUM_ROWS, H)`，数据类型与`expanded_x`一致。

## 约束说明

- `expanded_x`必须是2D或3D Tensor。
- `expanded_row_idx`必须是1D Tensor，若显式传入`k`，则元素个数必须能被`k`整除。
- `drop_pad_mode`取值范围为`[0, 3]`。
- `x1`未传入时，`x2`也不能传入。
- `bias`存在时，`expert_idx`必须同时存在。
- `scales`不为空时，若显式传入`k`，则`k`必须与`scales`的第二维一致。
- `scales`为空时，则必须传入`k`。
- `zero_expert_range`、`copy_expert_range`、`constant_expert_range`三个范围不能重叠。
- **自动反向（autograd）**：当`expanded_x`、`scales`等可微输入的`requires_grad`为True时支持自动反向，反向算子为[moe\_finalize\_routing\_grad](./moe_finalize_routing_grad.md)（封装`aclnnMoeFinalizeRoutingV2Grad`）。自动反向约束如下：
  - 仅在正向退化为aclnnMoeFinalizeRoutingV2场景时支持，即不传入aclnnMoeFinalizeRoutingV4特有输入`x`、`alpha1`、`alpha2`、`v`，且`zero_expert_range`、`copy_expert_range`、`constant_expert_range`均为None或无效（如`[-1, -1]`）。当使用了上述aclnnMoeFinalizeRoutingV4特有特性时，调用自动反向会抛出`NotImplementedError`。
  - `drop_pad_mode`仅支持0或1（列排列模式），不支持2或3（行排列模式）。
  - 仅计算`grad_expanded_x`和`grad_scales`；`x1`、`x2`、`bias`的梯度不会被计算（返回`None`）。如需对`x1`/`x2`求梯度，建议使用外部残差加法替代将其作为正向输入。
  - 正向`expanded_row_idx`采用`(K, R)`布局，反向算子采用`(R, K)`布局，自动反向下框架会自动转置。
  - **产品支持差异**：`aclnnMoeFinalizeRoutingV2Grad`在<term>Atlas 推理系列产品</term>上不支持，尽管正向`moe_finalize_routing`在该产品上支持。因此在<term>Atlas 推理系列产品</term>上，`moe_finalize_routing`不支持自动反向。

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

  # 正向：不传入aclnnMoeFinalizeRoutingV4特有输入（x/alpha1/alpha2/v），不设置zero_expert_range
  y = moe_finalize_routing(expanded_x, row_idx, scales=scales, k=K)

  # 反向自动触发 moe_finalize_routing_grad
  loss = y.sum()
  loss.backward()
  print(expanded_x.grad.shape)  # torch.Size([8, 8])
  print(scales.grad.shape)      # torch.Size([4, 2])
  ```

## 确定性计算

默认支持确定性计算。
