# KvCacheTurboQuant

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR/950DT 系列产品 | × |
| Atlas A3 系列产品 | × |
| Atlas A2 系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`KvCacheTurboQuant` 面向标准 MHA/GQA KV cache，对 BF16 KV 向量执行旋转、低比特主量化以及
QJL 残差符号编码。算子不支持 MLA latent cache。

每个输入向量先按 L2 范数归一化，再通过旋转矩阵映射并使用 2/3/4-bit codebook 编码；量化残差
归一化后通过 QJL 矩阵投影为符号位。输出同时包含原始范数和残差尺度。

## 参数说明

| 参数名 | 输入/输出/属性 | Shape | 数据类型 | 格式 |
|---|---|---|---|---|
| kv_vectors | 输入 | `[T, H, D]` | BFLOAT16 | ND |
| rotation_matrix | 输入 | `[D, D]` | FLOAT32、BFLOAT16 | ND |
| qjl_matrix | 输入 | `[J, D]` | FLOAT32、BFLOAT16 | ND |
| mse_bits | 属性 | 2、3 或 4 | INT64 | - |
| output_dtype | 可选属性 | 0 表示 BF16，1 表示 FP16 | INT64 | - |
| quant_idx | 输出 | `[T, H, ceil(D * mse_bits / 8)]` | UINT8 | ND |
| quant_qjl | 输出 | `[T, H, ceil(J / 8)]` | UINT8 | ND |
| quant_norm | 输出 | `[T, H]` | BFLOAT16、FLOAT16 | ND |
| quant_gamma | 输出 | `[T, H]` | BFLOAT16、FLOAT16 | ND |

## 约束说明

- 当前仅注册 `ascend910b`，对应 Atlas A2 系列产品。
- 输入和输出均为连续 ND Tensor。
- `D` 和 `J` 支持 64、128、256；BF16 矩阵路径仅支持 `D=J=128`。
- `H` 取值范围为 `[4, 32]`。
- `mse_bits` 取 2、3 或 4。
- `output_dtype=0` 时 metadata 输出为 BF16；`output_dtype=1` 时为 FP16。

## 调用说明

ACLNN 接口和两阶段调用方法参见
[`docs/aclnnKvCacheTurboQuant.md`](docs/aclnnKvCacheTurboQuant.md)。可运行样例参见
[`examples/test_aclnn_kv_cache_turbo_quant.cpp`](examples/test_aclnn_kv_cache_turbo_quant.cpp)。

## 编译与测试

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate cann91
source /path/to/cann/set_env.sh

bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=kv_cache_turbo_quant

bash build.sh --ophost_test --experimental --soc=ascend910b \
    --ops=kv_cache_turbo_quant

# 安装上一步生成的 custom 包后执行 ACLNN 示例
bash build.sh --run_example kv_cache_turbo_quant eager cust \
    --experimental --soc=ascend910b
```

正式精度和性能自验应使用任务交付件中的固定 case、Golden 和 baseline；性能数据必须在空闲独占
设备上采集。

## 目录说明

```text
├── docs/                 # ACLNN 接口文档
├── examples/             # ACLNN 调用样例
├── op_host/              # OpDef、InferShape/InferDataType 和 Tiling
├── op_kernel/            # Ascend C mixed AIC/AIV kernel
└── tests/ut/op_host/     # Host 单元测试
```

## 贡献说明

贡献信息以最终 PR 记录为准。
