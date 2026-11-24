# dual_level_grouped_matmul

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

当前支持根据 `group_list` 沿输入矩阵的行维度分组。每组输入与对应专家的 FRACTAL_NZ 权重相乘，跨 K 段累加后加入可选偏置。

先用两侧 level1 缩放因子还原输入和权重，再计算每个 level0 K 段的部分和。

$$
P_{h,i,j} = \sum_{k \in G_h}
  \left(X_{i,k} \cdot S^{X,1}_{i,k}\right)
  \left(W_{e,k,j} \cdot S^{W,1}_{e,k,j}\right)
$$

将各段部分和乘以相应的 level0 缩放因子，以 FP32 累加后加入偏置，再转换为输出类型。

$$
Y_{i,j} = \operatorname{Cast}_{\mathrm{RINT}}\left(
  \sum_h P_{h,i,j} \cdot S^{X,0}_{i,h} \cdot S^{W,0}_{e,h,j} + B_{e,j}
\right)
$$

$i$ 和 $j$ 分别索引输入行和输出列，$e$ 是该行对应的专家。$G_h$ 是第 $h$ 个 level0 分组的 K 范围，由 `level0_group_size` 决定。$S^{X,1}_{i,k}$ 和 $S^{W,1}_{e,k,j}$ 分别取两侧覆盖位置 $k$ 的 level1 缩放因子，分组大小由 `level1_group_size` 决定。输出公式中的 $S^{X,0}$ 和 $S^{W,0}$ 按 K 段索引。省略偏置时，$B_{e,j}$ 取 0。RINT 表示舍入到最近可表示值，等距时取偶数。

## 函数原型

```python
cann_ops_transformer.dual_level_grouped_matmul(
    x, weight, x_level0_scale, x_level1_scale,
    weight_level0_scale, weight_level1_scale,
    level0_group_size=512, level1_group_size=32, *,
    group_list=None, bias=None, offset=None, antiquant_scale=None, antiquant_offset=None,
    activation_input=None, activation_quant_scale=None,
    activation_quant_offset=None, split_item=3, group_type=0,
    group_list_type=0, act_type=0, tuning_config=None,
    output_dtype=torch.bfloat16, x_dtype=None, weight_dtype=None,
    scale_dtype=None, per_token_scale_dtype=None,
) -> List[Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | List[Tensor] | 必选 | 左矩阵，逻辑形状 (M,K)，M 为输入行数，K 为归约长度。 | FP4 E2M1 | (M,K/2) |
| weight | List[Tensor] | 必选 | 专家权重，E 为专家数，N 为输出列数。 | FP4 E2M1 | 非转置 (E,K,N/2)，转置 view (E,K/2,N)。 |
| x_level0_scale | List[Tensor] | 必选 | 输入矩阵的 level0 缩放因子，格式为 ND。 | `torch.float32` | (M,ceil(K/level0_group_size)) |
| x_level1_scale | List[Tensor] | 必选 | 输入矩阵的 level1 缩放因子，格式为 ND。 | E8M0 | (M,ceil(K/(2×level1_group_size)),2) |
| weight_level0_scale | List[Tensor] | 必选 | 各专家权重的 level0 缩放因子，格式为 ND。 | `torch.float32` | (E,ceil(K/level0_group_size),N) |
| weight_level1_scale | List[Tensor] | 必选 | 各专家权重的 level1 缩放因子，采用与权重转置状态匹配的 ND 视图。 | E8M0 | (E,ceil(K/(2×level1_group_size)),N,2) |
| level0_group_size | int | 可选 | 每组 FP32 level0 scale 覆盖的 K 元素数，默认值为 512。 | - | - |
| level1_group_size | int | 可选 | 每组 E8M0 level1 scale 覆盖的 K 元素数，默认值为 32。 | - | - |
| group_list | Tensor | 必选 | 仅通过关键字传入，用于指定专家分组。 | `torch.int64` | type=0/1 为 (E,)，type=2 为 (E,2)。 |
| bias | List[Tensor] | 可选 | 专家偏置，默认值为 `None`。 | `torch.float32` | (E,N) |
| group_list_type | int | 可选 | 0 为累计结束位置，1 为各组行数，2 为稀疏分组列表。默认值为 0，传入 `None` 时同样按 0 处理。 | - | - |
| split_item | int | 可选 | 输出配置，支持 2 或 3，默认值为 3。本接口的输入和输出 TensorList 长度均为 1，两种取值等价。 | - | - |
| group_type | int | 可选 | 仅支持 0，默认 0。 | - | - |
| act_type | int | 可选 | 默认值为 0，表示无激活。传入 `None` 时同样按 0 处理。 | - | - |
| x_dtype、weight_dtype | int | 可选 | GMM 兼容参数。调用时使用默认值 `None`，接口按 FP4 E2M1 解释矩阵中的打包字节。 | - | - |
| scale_dtype、per_token_scale_dtype | int | 可选 | GMM 兼容参数。调用时使用默认值 `None`，接口按 E8M0 解释权重和左矩阵的 level1 scale 字节。 | - | - |
| offset、antiquant_scale、antiquant_offset | List[Tensor] | 可选 | 预留参数。仅支持默认值 `None`。 | - | - |
| activation_input、activation_quant_scale、activation_quant_offset | List[Tensor] | 可选 | 预留参数。仅支持默认值 `None`。 | - | - |
| tuning_config | List[int] | 可选 | 默认值为 `None`。显式配置时支持 `[0]`。 | - | - |
| output_dtype | `torch.dtype` | 可选 | 默认 `torch.bfloat16`，传入 `None` 时也使用默认类型。 | `torch.float16`、`torch.bfloat16` | - |

## 返回值说明

返回长度为 1 的 TensorList，唯一元素为 (M,N) 的 ND 张量，位于与 x 相同的 NPU。输出类型由 `output_dtype` 决定，为 `torch.float16` 或 `torch.bfloat16`。

## 约束说明

- 支持 Torch 单算子调用，不支持图模式。
- 接口中的 TensorList 在 PyTorch 中以 `List[Tensor]` 表示。输入 TensorList 以及提供的 `bias` 均须包含一个张量，所有输入须位于同一 NPU。
- `weight` 须为 FRACTAL_NZ，其余输入须为 ND。除规定的转置 `weight_level1_scale` 视图外，ND 输入须连续。各输入的存储偏移须为 0。
- 专家数须大于 0，归约长度须为偶数。非转置权重的输出列数还须为偶数。非空 Tensor 场景下，归约长度须大于 2，输出列数须大于 1。维度上限和地址范围见 [ACLNN 约束说明](aclnnDualLevelGroupedMatmulWeightNz.md#约束说明)。
- 支持 M 为 0 或 N 为 0 的空 Tensor 场景，此时允许 K 为 0。返回的 TensorList 包含一个形状为 `(M,N)` 的空 Tensor。
- `group_list` 须完整覆盖 M 行，允许专家的行数为 0。
  - `group_list_type=0` 表示累计结束位置，各项须非负且非递减，最后一项须等于 M。
  - `group_list_type=1` 表示各组行数，各项须非负，总和须等于 M。
  - `group_list_type=2` 表示稀疏分组，形状为 `(E,2)`，每行为 `[expert_id, row_count]`。专家编号须在 `[0,E)` 内，各组行数须非负且总和等于 M。非零行组在前，零行组在后，两类分组内部均按专家编号递增排列。输入和输出按列表顺序分段，权重、权重缩放因子和偏置按专家编号选取。
  - 分组值无效时，输出无效。
- 当前版本仅支持两级分组大小的默认值。level1 scale 末维的两个元素分别对应连续的两个 level1 分组。

- MXFP4 E2M1 二级量化场景支持权重转置和非转置，偏置可省略。
  - 非空参数支持的数据类型组合见下表。

    | 场景 | x、weight | x_level0_scale、weight_level0_scale | x_level1_scale、weight_level1_scale | bias | group_list | 输出 |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | MXFP4 E2M1 二级量化，FP16 输出 | FP4 E2M1 | `torch.float32` | E8M0 | `torch.float32` | `torch.int64` | `torch.float16` |
    | MXFP4 E2M1 二级量化，BF16 输出 | FP4 E2M1 | `torch.float32` | E8M0 | `torch.float32` | `torch.int64` | `torch.bfloat16` |

  - FP4 E2M1 的 `x` 和 `weight` 打包成 `torch.int8` 或 `torch.uint8` 张量。每个字节存储两个 FP4 E2M1 元素，低半字节在前。
  - E8M0 的 `x_level1_scale` 和 `weight_level1_scale` 以 `torch.int8` 或 `torch.uint8` 张量传入，每个字节存储一个 E8M0 编码。
  - 转置 weight 先以 (E,N,K/2) 的字节张量进行 NZ 格式转换，再 transpose(1,2)，得到 (E,K/2,N)。转置 weight_level1_scale 由连续的 (E,N,ceil(K/(2×level1_group_size)),2) 张量 transpose(1,2) 得到。
  - 调用前须完成矩阵和 level1 scale 的量化与打包，并按参数表构造对应的 PyTorch Tensor。

## 确定性计算

默认支持确定性计算。

## 调用示例

调用示例代码如下，仅供参考，PyTorch 扩展的安装及执行过程请参考[运行 PyTorch 示例](../examples/README.md#运行-pytorch-示例)。

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  以下示例将打包后的 FP4 权重转换为 FRACTAL_NZ，显式传入两级 group size，调用 Torch 单算子接口执行分组矩阵乘并校验 BF16 输出。示例省略 `split_item`，使用默认值 3。完整源码见[仓内示例](../examples/test_torch_dual_level_grouped_matmul.py)。

  ```python
  # Copyright (c) 2026 Huawei Technologies Co., Ltd.
  # Licensed under CANN Open Software License Agreement Version 2.0.
  """Run a dual-level MXFP4 grouped matmul and check both experts' outputs."""

  import importlib
  import os

  import torch
  import torch_npu


  def main():
      ops = importlib.import_module(
          os.environ.get("CANN_TRANSFORMER_PACKAGE", "cann_ops_transformer")
      )
      m, k, n, experts = 32, 640, 128, 2
      level0_group_size, level1_group_size = 512, 32
      fp4_elements_per_byte = 2
      mx_scale_pair = 2
      level1_pairs = k // (mx_scale_pair * level1_group_size)
      level0_groups = (k + level0_group_size - 1) // level0_group_size
      group_rows = m // experts
      packed_fp4_ones = 0x22
      e8m0_one, e8m0_two = 127, 128
      weight_level1_scale = 2
      tail_level0_scale = 2
      bias_step = 16
      format_nz, format_nd = 29, 2
      device = "npu:0"

      x = torch.full(
          (m, k // fp4_elements_per_byte),
          packed_fp4_ones,
          dtype=torch.int8,
          device=device,
      )
      weight = torch_npu.npu_format_cast(
          torch.full(
              (experts, k, n // fp4_elements_per_byte),
              packed_fp4_ones,
              dtype=torch.int8,
              device=device,
          ),
          format_nz,
      )
      x_level0 = torch.ones((m, level0_groups), dtype=torch.float32, device=device)
      x_level0[:, 1] = tail_level0_scale
      weight_level0 = torch.ones(
          (experts, level0_groups, n), dtype=torch.float32, device=device
      )
      weight_level0[:, 1, :] = tail_level0_scale
      x_level1 = torch.full(
          (m, level1_pairs, mx_scale_pair), e8m0_one, dtype=torch.uint8, device=device
      )
      weight_level1 = torch_npu.npu_format_cast(
          torch.full(
              (experts, level1_pairs, n, mx_scale_pair),
              e8m0_two,
              dtype=torch.uint8,
              device=device,
          ),
          format_nd,
      )
      group_list = torch.tensor([group_rows] * experts, dtype=torch.int64, device=device)
      bias = torch.full((experts, n), bias_step, dtype=torch.float32, device=device)
      bias[1, :] = bias_step + bias_step

      (output,) = ops.dual_level_grouped_matmul(
          [x],
          [weight],
          [x_level0],
          [x_level1],
          [weight_level0],
          [weight_level1],
          level0_group_size=level0_group_size,
          level1_group_size=level1_group_size,
          group_list=group_list,
          group_list_type=1,
          bias=[bias],
          output_dtype=torch.bfloat16,
      )

      matmul = weight_level1_scale * (
          level0_group_size
          + (k - level0_group_size) * tail_level0_scale * tail_level0_scale
      )
      expected = torch.arange(1, experts + 1, dtype=torch.float32) * bias_step + matmul
      expected = (
          expected.repeat_interleave(group_rows)
          .unsqueeze(1)
          .expand(m, n)
          .to(torch.bfloat16)
      )
      torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
      print(f"DualLevelGroupedMatmul passed, BF16 shape {tuple(output.shape)}")


  if __name__ == "__main__":
      main()
  ```
<!-- end id12 -->
