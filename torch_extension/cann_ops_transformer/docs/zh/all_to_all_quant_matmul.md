# all_to_all_quant_matmul

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

  `cann_ops_transformer.all_to_all_quant_matmul`完成All-to-All通信与MX量化矩阵乘法的融合计算。在多卡场景下，通过URMA all-to-all在各卡间重新分布左矩阵的行与列，再与右矩阵做MX FP8/FP4量化矩阵乘法，输出BF16或FP16结果。底层调用`aclnnAlltoAllMatmulV2`接口完成计算。
- **计算公式**：

  具体计算流程如下：

  1. 各卡持有左矩阵本地分片`A_local[BS, H]`，通过URMA all-to-all通信重分布得到`A_permuted[BS/rankSize, H*rankSize]`。
  2. `A_permuted`结合`x1_scale`、`x2_scale`完成MX反量化，与右矩阵`X2`做矩阵乘法。
  3. 对矩阵乘法结果累加`bias`（可选），得到输出`Y`。

  其中 rankSize 为 NPU 卡数，即 world_size。

## 函数原型

```python
cann_ops_transformer.all_to_all_quant_matmul(
    x1,
    x2,
    group,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    x1_quant_mode=0,
    x2_quant_mode=0,
    group_sizes=None,
    x1_dtype=-1,
    x2_dtype=-1,
    x1_scale_dtype=-1,
    x2_scale_dtype=-1,
    y_dtype=6,
    comm_mode=None,
    precision_mode=0,
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                                                           | 数据类型                                       | 维度(shape)                       |
| ------------------ | ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | --------------------------------- |
| `x1`             | Tensor       | 必选      | 左矩阵，对应各卡本地的A_local。MXFP4场景以uint8打包存储（2个fp4占1字节），实际传入的K轴大小为`H/2`，需配合`x1_dtype=296`。 | `torch.float8_e4m3fn`、`torch.float8_e5m2` | `(BS, H)`                       |
| `x2`             | Tensor       | 必选      | 右矩阵，对应公式中X2。MXFP4场景以uint8打包存储，实际传入的K轴大小为`H*rankSize/2`，需配合`x2_dtype=296`。                  | `torch.float8_e4m3fn`、`torch.float8_e5m2` | `(H*rankSize, N)`               |
| `group`          | ProcessGroup | 必选      | torch.distributed通信域，用于All-to-All通信。                                                                                  | -                                              | -                                 |
| `bias`           | Tensor       | 可选      | 偏置项，矩阵乘后累加，默认值为`None`。                                                                                       | `torch.float32`                              | `(N,)`                          |
| `x1_scale`       | Tensor       | 可选      | `x1`的MX量化scale，默认值为`None`。                                                                                        | `torch.float8_e8m0fnu`                       | `(BS, ceil(H / 64), 2)`         |
| `x2_scale`       | Tensor       | 可选      | `x2`的MX量化scale，默认值为`None`。                                                                                        | `torch.float8_e8m0fnu`                       | `(N, ceil(H*rankSize / 64), 2)` |
| `x1_quant_mode`  | int          | 可选      | `x1`量化模式，默认值为0；MX量化传6。                                                                                         | int                                            | -                                 |
| `x2_quant_mode`  | int          | 可选      | `x2`量化模式，默认值为0；MX量化传6。                                                                                         | int                                            | -                                 |
| `group_sizes`    | List[int]    | 可选      | MX量化group大小，默认值为`None`；MX场景传`[1, 1, 32]`。                                                                    | int                                            | -                                 |
| `x1_dtype`       | int          | 可选      | `x1`的dtype wrapper覆盖值，默认值为-1（使用tensor的dtype）；MXFP4场景传296。                                                 | int                                            | -                                 |
| `x2_dtype`       | int          | 可选      | `x2`的dtype wrapper覆盖值，默认值为-1（使用tensor的dtype）；MXFP4场景传296。                                                 | int                                            | -                                 |
| `x1_scale_dtype` | int          | 可选      | `x1_scale`的dtype wrapper覆盖值，默认值为-1；MX场景传293（e8m0）。                                                           | int                                            | -                                 |
| `x2_scale_dtype` | int          | 可选      | `x2_scale`的dtype wrapper覆盖值，默认值为-1；MX场景传293（e8m0）。                                                           | int                                            | -                                 |
| `y_dtype`        | int          | 可选      | 输出`y`的数据类型，默认值为fp32；未显式传15（BF16）或5（FP16）时会被算子校验拒绝。                                           | int                                            | -                                 |
| `comm_mode`      | str          | 可选      | 通信模式，默认值为`None`；当前仅支持`"urma"`。                                                                             | string                                         | -                                 |
| `precision_mode` | int          | 可选      | 精度模式，默认值为0。0/1/2性能由低到高，精度由高到低。                                                                         | int                                            | -                                 |

## 返回值说明

| 参数名          | 参数类型 | 描述                                                         | 数据类型                        | 维度(shape)          |
| --------------- | -------- | ------------------------------------------------------------ | ------------------------------- | -------------------- |
| `y`           | Tensor   | 矩阵乘输出。                                                 | 由`y_dtype`指定（BF16或FP16） | `(BS/rankSize, N)` |
| `all2all_out` | Tensor   | all-to-all通信中间输出，预留参数，暂不支持（返回`None`）。 | -                               | `None`             |

## 约束说明

- 适用场景：该接口支持训练、推理场景下使用。
- 调用方式：该接口支持单算子模式调用，需要在多卡通信域下使用。
- 仅支持Ascend 950系列产品。
- `world_size`需为2、4、8、16之一。
- `BS`需能被`world_size`整除。
- `H`需能被64整除。
- `x1`、`x2`需为2D Tensor。
- `x1`与`x2`需同为FP8或同为FP4。
- 通信buffer大小约束：本算子使用HCCL内置通信buffer完成All-to-All通信，该buffer大小由环境变量`HCCL_BUFFSIZE`（单位MB，不配置时默认200MB）控制。每个rank的commBuffer需容纳所有rank写来的数据，要求满足：`HCCL_BUFFSIZE` >= `world_size * (BS / world_size) * (H * sizeof(x1_dtype) + ceil(H / 64) * 2) + 2MB`（字节），其中`sizeof(x1_dtype)`在FP8场景为1字节、FP4场景为0.5字节，`ceil(H / 64) * 2`为x1_scale的通信量（每64个元素1字节scale、末维2字节）。若不满足，算子执行时会在tiling阶段报错，此时可通过`export HCCL_BUFFSIZE=<建议值>`调大buffer后重试。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用（MX FP8 场景）：

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer
  import torch.distributed as dist

  # 初始化多卡通信域
  dist.init_process_group(backend="hccl")
  group = dist.new_group()
  world_size = dist.get_world_size()

  BS = 128
  H = 128
  N = 64

  x1 = torch.randn(BS, H).to(torch.float8_e4m3fn).npu()
  x2 = torch.randn(H * world_size, N).to(torch.float8_e4m3fn).npu()
  x1_scale = torch.randint(0, 256, (BS, math.ceil(H / 64), 2), dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
  x2_scale = torch.randint(0, 256, (N, math.ceil(H * world_size / 64), 2), dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)

  y, all2all_out = cann_ops_transformer.all_to_all_quant_matmul(
      x1,
      x2,
      group,
      x1_scale=x1_scale,
      x2_scale=x2_scale,
      y_dtype=15,              # BF16
      x1_quant_mode=6,         # MX
      x2_quant_mode=6,         # MX
      group_sizes=[1, 1, 32],
      x1_scale_dtype=293,      # float8_e8m0fnu
      x2_scale_dtype=293,      # float8_e8m0fnu
      comm_mode="urma",
      precision_mode=0,
  )
  ```
