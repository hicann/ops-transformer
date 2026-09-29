# engram_gate_dsl

`engram_gate` 是 Engram 残差门算子：融合双路 RMS 归一化、加权点积、
signed-sqrt sigmoid 门控与残差更新，为单融合 kernel（纯单机向量计算，无通信）。

Kernel 与 NPU host 由 net wheel（`cannbot-arena-net-ops`）的
`ops/engram_gate.py` 提供，经 `@register("engram_gate")` 注册构建。本目录
`torch_extension` 只做 `torch.library.custom_op` 算子边界注册，再
`from ops.engram_gate import engram_gate`。调用前必须安装包含
`ops.engram_gate` 的 net wheel。

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列产品 | × |
| Atlas A2 训练/推理系列产品 | × |

> 注：本次验证在 Ascend 950DT 上完成；Ascend 950PR 同族（arch35）未上板实测。

## 接口

```python
from engram_gate import engram_gate

y = engram_gate(
    x,
    key,
    value,
    weight,
    image_mask=None,
    eps=1e-6,
    clamp_value=1e-6,
)
```

| 参数 | dtype | shape |
| :--- | :--- | :--- |
| `x`, `key` | BF16 | `[..., hc_mult, dim]`，ndim ≥ 2；前导维折叠为 token，2-D 表示 `hc_mult=1` |
| `value` | BF16 | `[..., dim]`（与 `x` 前导维一致，跨 hc 共享） |
| `weight` | FP32 | `[hc_mult, dim]` |
| `image_mask` | BOOL，可选 | `[...]`（与 `x` 前导维一致），`True` 的 token 原样返回 `x` |
| `y` | BF16 | 与 `x` 同 shape |

`dim` 支持任意正整数：不足 64 的尾部按掩码处理；超过单行驻留 UB 预算
（`ceil(dim/64)*64 > 8448`）时自动切换为列分块流式路径，无 dim 上限。模型默认
配置为 `dim=5120, hc_mult=4`。

计算公式：

```text
rstd = rsqrt(mean(x²) + eps) * rsqrt(mean(key²) + eps)
dot  = sum(x * weight * key) * rstd / sqrt(dim)
gate = sigmoid(copysign(sqrt(max(abs(dot), clamp_value)), dot))
y    = bf16(x + gate * value)      # image_mask=True 的 token 直接返回 x
```

## 依赖

- Python ≥ 3.10，`torch`、`torch_npu`、`cannbotdsl`
- `cannbot-arena-net-ops` wheel（提供 `ops.engram_gate`；导出 T 集合内的形状走
  包内 AOT 产物，其余形状与 mask 路径由包内源码 JIT）

## 使用示例

```python
import torch
import torch_npu
from engram_gate import engram_gate

x = torch.randn(72, 4, 5120, dtype=torch.bfloat16, device="npu")
key = torch.randn(72, 4, 5120, dtype=torch.bfloat16, device="npu")
value = torch.randn(72, 5120, dtype=torch.bfloat16, device="npu")
weight = torch.randn(4, 5120, dtype=torch.float32, device="npu")
y = engram_gate(x, key, value, weight)          # [72, 4, 5120] BF16
```

预编译（可选，消除首次调用编译开销）：

```python
import cannbotdsl
from engram_gate import EngramGate

EngramGate(eps=1e-6, clamp_value=1e-6).run.compile(
    cannbotdsl.TensorSpec((72, 4, 5120), cannbotdsl.dtypes.bfloat16),
    cannbotdsl.TensorSpec((72, 4, 5120), cannbotdsl.dtypes.bfloat16),
    cannbotdsl.TensorSpec((72, 5120), cannbotdsl.dtypes.bfloat16),
    cannbotdsl.TensorSpec((4, 5120), cannbotdsl.dtypes.float32),
    cannbotdsl.TensorSpec((72, 4, 5120), cannbotdsl.dtypes.bfloat16),
)
```

## torch_extension（图模式接入）

[torch_extension](torch_extension) 将算子注册进 `cann_ops_transformer` 库
（`torch.ops.cann_ops_transformer.ds41.engram_gate`，含 Meta/FakeTensor 实现），
供图模式使用。经仓库 torch_extension 构建流程安装
cann_ops_transformer 后使用：

```bash
bash build.sh --torch_extension
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

```python
from cann_ops_transformer.ops.engram.engram_gate_dsl import engram_gate_torch

out = engram_gate_torch(x, key, value, weight)          # 与 engram_gate 语义一致
```

## 运行测试

```bash
bash experimental/engram/engram_gate_dsl/tests/test.sh
# 或
pytest -q experimental/engram/engram_gate_dsl/tests/
```

覆盖：默认形状编译、模型形状精度（含 image_mask 位级一致）、任意 dim（尾部掩码）、
任意 ndim（前导维折叠）、超大 dim（列分块路径，最大 100K）、ACL Graph 图捕获与
重放——含 `torch.npu.NPUGraph` 原生捕获（重放对新输入数据/新 mask 值结果正确）
与 torchair `npugraph_ex`（经 torch_extension 边界 dynamo 全图捕获重放，依赖
torchair，缺失时自动跳过）。
