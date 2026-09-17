# chunk_gated_delta_rule_compute_wy torch_ops_extension

## 功能说明

该目录提供ChunkGatedDeltaRuleComputeWy算子的PyTorch扩展接口。安装后可通过如下接口调用：

```python
q_kernel, k_kernel, w_kernel, u_kernel, g_kernel = \
    torch.ops.custom.npu_chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, 64)
```

入参与出参：

| 名称 | shape | dtype |
| :--- | :--- | :--- |
| q, k | `[B, T, Hk, K]` | FLOAT16 |
| v | `[B, T, Hv, V]` | FLOAT16 |
| g | `[B, T, Hv]` | FLOAT32（须非正） |
| beta | `[B, T, Hv]` | FLOAT16 |
| chunk_size | 标量，当前仅支持64 | INT |
| q_kernel, k_kernel | `[B, Hk, T, K]` | FLOAT16 |
| w_kernel | `[B, Hv, T, K]` | FLOAT16 |
| u_kernel | `[B, Hv, T, V]` | FLOAT16 |
| g_kernel | `[B, Hv, T]` | FLOAT32 |

`chunk_size`为位置参数，调用形式与vllm-ascend侧的
`torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, 64)`一致，便于直接对拍。

约束：`T % 64 == 0`，`Hv % Hk == 0`，`K`/`V`为16的倍数且`<= 128`，`B <= 32`，`Hv <= 64`。

## 编译安装

编译安装前需先安装ChunkGatedDeltaRuleComputeWy自定义算子包，并配置自定义算子包环境变量。

```bash
bash build.sh --pkg --experimental --soc=ascend310p --ops=chunk_gated_delta_rule_compute_wy -j16
# 安装算子包并 source 环境变量后：
cd experimental/attention/chunk_gated_delta_rule_compute_wy/torch_ops_extension
bash build_and_install.sh
```

安装完成后：

```python
import torch
import torch_npu
import custom_ops  # 注册 torch.ops.custom.*
```
