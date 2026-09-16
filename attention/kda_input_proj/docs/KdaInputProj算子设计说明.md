# KdaInputProj 算子设计说明

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

KdaInputProj是推理场景下Recurrent KDA的前处理算子，负责qkv，beta，gate和g的投影操作。整体计算流程如下所示：
![KdaInputProj Flow](figures/kda_input_proj_flow.png)

### Stage 1

在Stage 1，KdaInputProj算子分别利用AIC和AIV并行执行 `Matmul (beta/gate/g)` 和 `DynamicMxQuant` 两个计算任务。其中，`Matmul (beta/gate/g)` 所有的 BlockMmad 任务会被所有的AIC单元并行处理，以减少计算轮次，提高硬件利用率。计算公式如下：

$$
\begin{bmatrix}
\mathbf{beta}\\
\mathbf{gate}\\
\mathbf{g}\\
\end{bmatrix} = \mathbf{X}
\begin{bmatrix}
W_{beta}\\
W_{gate}\\
W_{g}
\end{bmatrix}
$$

### Stage 2

在Stage 2，KdaInputProj算子分别利用AIC和AIV并行执行 `QuantMatmul (qkv)` 和 `Sigmoid` 两个计算任务。其中 `QuantMatmul` 的激活为Stage 1中的 `DynamicMxQuant` 任务的输出，`qkv_weight` 为离线量化好的权重。


## 接口说明

该算子通过PyTorch扩展注册为`torch.ops.custom.npu_kda_input_proj`。

### PyTorch接口原型

```python
torch.ops.custom.npu_kda_input_proj(
    x: Tensor,
    weight_qkv: Tensor,
    weight_beta: Tensor,
    weight_gate: Tensor,
    weight_g: Tensor,
    weight_qkv_scale: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## 参数说明

维度符号说明：

- $T$：输入token个数。

| 参数名 | 输入 / 输出 | 数据类型 | 数据格式 | 维度（shape） | 描述 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `x` | 输入 | BFLOAT16 | ND | `[T, 7168]` | 隐藏层输入。 |
| `weight_qkv` | 输入 | FLOAT8_E4M3 | ND | `[4608, 7168]` | qkv投影的权重。 |
| `weight_beta` | 输入 | BFLOAT16 | ND | `[12, 7168]` | beta投影的权重。 |
| `weight_gate` | 输入 | BFLOAT16 | ND | `[1536, 7168]` | gate投影的权重。 |
| `weight_g` | 输入 | BFLOAT16 | ND | `[1536, 7168]` | g投影的权重。 |
| `weight_qkv_scale` | 输入 | FLOAT8_E8M0 | ND | `[4608, 112, 2]` | qkv投影权重的缩放系数。 |

## 约束说明

### 单参数约束

暂无。

### 存在性约束

暂无。

### 一致性约束

暂无。

### 特性交叉约束

暂无。

## Ascend 950PR/Ascend 950DT调用示例

```python
import torch
import torch_npu
import custom_ops

# 安装 torch_ops_extension 后调用（见 torch_ops_extension/build_and_install.sh）
out = torch.ops.custom.npu_kda_input_proj(
    x, weight_qkv, weight_beta, weight_gate, weight_g, weight_qkv_scale
)
```

## 相关接口

PyTorch 扩展编译与安装说明见[build_and_install.sh](../torch_ops_extension/build_and_install.sh)。精度与 pytest 用例在后续 PR 合入后提供。
