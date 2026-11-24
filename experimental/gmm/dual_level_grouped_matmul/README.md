# DualLevelGroupedMatmul

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：沿左矩阵的 M 轴分组，按专家执行 MXFP4 E2M1 双级量化矩阵乘。各专家共用 K、N，权重采用 FRACTAL_NZ 格式，输出为 FP16 或 BF16。
- 计算流程：按 `level1_group_size` 应用 E8M0 level1 scale，按 `level0_group_size` 划分 K 段。每段的 FP32 部分和乘以该段两侧的 FP32 level0 scale 后跨段累加，最后加入可选 bias 并转换为输出类型。

## 参数说明

下表按算子原型的注册顺序列出参数。M 为总行数，E 为专家数，K 为归约维度，N 为输出列数。权重和权重 level1 scale 的转置形状为传入算子原型的形状。ACLNN 和 PyTorch 调用方的视图、步长及命名见各自的接口文档。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :---: |
| x | 输入 | 左矩阵，形状为 `(M,K)`。 | FLOAT4_E2M1 | ND |
| x_level0_scale | 输入 | 左矩阵的 level0 scale，形状为 `(M,ceil(K/level0_group_size))`。 | FLOAT | ND |
| x_level1_scale | 输入 | 左矩阵的 level1 scale，形状为 `(M,ceil(K/(2×level1_group_size)),2)`。 | FLOAT8_E8M0 | ND |
| weight | 输入 | 全部专家的权重。非转置时为 `(E,K,N)`，转置时为 `(E,N,K)`。 | FLOAT4_E2M1 | FRACTAL_NZ |
| weight_level0_scale | 输入 | 权重的 level0 scale，形状固定为 `(E,ceil(K/level0_group_size),N)`。 | FLOAT | ND |
| weight_level1_scale | 输入 | 权重的 level1 scale。非转置时为 `(E,ceil(K/(2×level1_group_size)),N,2)`，转置时为 `(E,N,ceil(K/(2×level1_group_size)),2)`。 | FLOAT8_E8M0 | ND |
| grouplist | 输入 | 专家分组信息。类型 0/1 的形状为 `(E,)`，类型 2 为 `(E,2)`。 | INT64 | ND |
| bias | 可选输入 | 专家偏置，形状为 `(E,N)`，缺省时按 0 处理。 | FLOAT | ND |
| y | 输出 | 矩阵乘结果，形状为 `(M,N)`。 | FLOAT16、BF16 | ND |
| transpose_weight | 可选属性 | 权重是否转置，默认值为 `false`。 | BOOL | - |
| group_list_type | 可选属性 | 0 表示累计结束位置，1 表示各组行数，2 表示稀疏分组列表，默认值为 0。 | INT | - |
| tuning_config | 可选属性 | 调优配置，仅支持 `[0]`，默认值为 `[0]`。 | LIST_INT | - |
| dtype | 必选属性 | 输出类型对应的 GE 枚举，支持 FLOAT16 或 BF16。 | INT | - |
| level0_group_size | 可选属性 | 一个 FP32 scale 覆盖的 K 元素数，仅支持 512，默认值为 512。 | INT | - |
| level1_group_size | 可选属性 | 一个 E8M0 scale 覆盖的 K 元素数，仅支持 32，默认值为 32。 | INT | - |

## 约束说明

- 当前仅支持沿输入矩阵的行维度分组。输入和权重的秩分别固定为 2 和 3，可在秩已知时推导含未知维度的输出形状，实际 tiling 使用具体维度。M、E 须大于 0，K 须为大于 2 的偶数，N 须大于 1。非转置权重的 N 也须为偶数。
- 每个张量包含全部专家所需的数据。ACLNN 和 PyTorch 接口中的矩阵、scale、可选 bias 和输出均通过长度为 1 的列表表示，分组信息通过单个张量传入。
- 分组列表须完整覆盖 M 行。类型 2 的每行为 `[专家编号, 行数]`。非零行组在前，零行组在后，两类分组内部均按专家编号递增排列。输入和输出按列表顺序分段，权重、权重 scale 和 bias 按专家编号索引。完整约束见 [ACLNN 约束说明](docs/aclnnDualLevelGroupedMatmulWeightNz.md#约束说明)。
- 支持 Torch 单算子调用，不支持图模式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| ACLNN API | [C++ 示例](examples/arch35/test_aclnn_dual_level_grouped_matmul_weight_nz.cpp) | 通过 [aclnnDualLevelGroupedMatmulWeightNz](docs/aclnnDualLevelGroupedMatmulWeightNz.md) 两段式接口调用。 |
| PyTorch API | [Python 示例](examples/test_torch_dual_level_grouped_matmul.py) | 通过 [dual_level_grouped_matmul](docs/torchapi_dual_level_grouped_matmul.md) 接口调用。 |

构建算子包及运行示例见 [示例说明](examples/README.md)。UT 和精度验证见 [测试说明](tests/README.md)。
