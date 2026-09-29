# IndexerPrologueQw

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

`indexer_prologue_qw` 是融合 attention 前处理算子。一次调用同时完成：

- **Q 路**：MXFP8 GEMM（`qr @ wqb^T`）→ reshape 成 `(T, N, D)` → 对尾部 `Dr` 维做 inplace RoPE → 沿 `D` 做 MXFP4 量化（数据 E2M1，scale E8M0）。
- **W 路**：BF16 GEMM（`x @ ww^T`）再乘标量 `softmax_scale`，输出 FP32。

当前实现只覆盖冻结几何 `dim=5120, q_lora=1280, N=32, D=128, Dr=64`；可变轴是 token 数 `T`。默认验证用例是 `T=72`。

`wqb` 和 `ww` 必须已经是 FRACTAL_NZ（加载时转一次，见 `ops.indexer_prologue_qw.to_nz`）。`x` / `qr` / descale / RoPE 仍是 ND。

Kernel 与 NPU host 在 `cannbot-arena-ds41` 的 `net/ops/indexer_prologue_qw.py`，通过 `@register("indexer_prologue_qw")` 打进 net wheel。本目录 `torch_extension` 只做 schema 注册，再 `from ops.indexer_prologue_qw import indexer_prologue_qw`。

## 参数说明

```python
indexer_prologue_qw(
    x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos,
    *,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
) -> tuple[Tensor, Tensor, Tensor]
```

### 输入和属性

| 参数名 | 可选/必选 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 必选 | W 路激活 | `bfloat16` | `(T, dim)` |
| `qr` | 必选 | Q 路激活，MXFP8 E4M3 的 uint8 字节视图 | `uint8` | `(T, q_lora)` |
| `wqb` | 必选 | Q 路权重，**FRACTAL_NZ** | `uint8` | `(N*D, q_lora)` |
| `ww` | 必选 | W 路权重，**FRACTAL_NZ** | `bfloat16` | `(N, dim)` |
| `descale_qr` | 必选 | `qr` 的 paired E8M0 scale | `uint8` | `(T, ceil(q_lora/64), 2)` |
| `descale_wqb` | 必选 | `wqb` 的 paired E8M0 scale | `uint8` | `(N*D, ceil(q_lora/64), 2)` |
| `rope_sin` | 必选 | RoPE sine | `float32` | `(T, Dr)` |
| `rope_cos` | 必选 | RoPE cosine | `float32` | `(T, Dr)` |
| `softmax_scale` | 必选 | W 路 GEMM 之后的标量 | `float` | - |

上表的 dtype 是硬约束，不是建议：kernel 的 DMA 按声明的元素宽度发 burst，`rope_sin` / `rope_cos` 传 bf16 会让 MTE2 按 fp32 的长度去读只有一半字节的 buffer。host 侧现在会先校验 dtype 再下发（只看 metadata，不读 tensor 的值），不匹配直接 `ValueError`。

`softmax_scale` 是 kernel 标量，不是 device tensor，所以调用它不会触发 host→device 搬运。

`w` 的 K 轴归约在核间用 GM atomic add 完成，**不是逐位可复现的**：partial 落盘顺序不固定，fp32 加法不满足结合律，同一组输入重复跑低位会变（T=144 实测相对误差在 8.7e-3 ~ 1.5e-2 漂）。`q` / `descale_q` 走整数 epilogue，逐位稳定。`w` 不需要调用者预置 0，kernel 自己清零。

### 输出

| 参数名 | 描述 | 数据类型 | 形状 |
| :--- | :--- | :--- | :--- |
| `q` | 量化后的 Q | `uint8` | `(T, N, D/2)` |
| `descale_q` | Q 的 E8M0 scale | `uint8` | `(T, N, ceil(D/64), 2)` |
| `w` | W 路结果，已乘 `softmax_scale` | `float32` | `(T, N)` |

## 调用说明

先安装 arena 的 `cannbot-arena-net-ops` wheel，再按
[torch_extension/README.md](../../../torch_extension/README.md) 整包编译并安装
`cann_ops_transformer`。不要用 `--ops=` 打 `cann_ops_transformer_custom`：
整包产物才是 `cann_ops_transformer-1.0.0-*.whl`。

```bash
# 1) net 算子 wheel（arena 仓）
cd /path/to/cannbot-arena-ds41/net/native_package
./run-build.sh --operator indexer_prologue_qw
python3 -m pip install output/operators/indexer_prologue_qw/wheels/*.whl --force-reinstall --no-deps

# 2) ops-transformer 整包（本仓，见 torch_extension/README.md）
cd /path/to/ops-transformer_dsv41
python3 -m pip install -r torch_extension/requirements.txt
bash build.sh --torch_extension
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

```python
import torch
import torch_npu
import cann_ops_transformer.ops  # 触发 ds41 注册；不要只 import cann_ops_transformer
from ops.indexer_prologue_qw import to_nz
from cann_ops_transformer.ops.ds41 import indexer_prologue_qw
# 等价：from cann_ops_transformer.ops import indexer_prologue_qw

q, descale_q, w = indexer_prologue_qw(
    x, qr, to_nz(wqb), to_nz(ww), descale_qr, descale_wqb, rope_sin, rope_cos,
    softmax_scale=1.0 / (128 ** 0.5),
)
```

| 调用方式 | 入口 |
| :--- | :--- |
| PyTorch API | `cann_ops_transformer.ops.ds41.indexer_prologue_qw` |
| torch.ops | `torch.ops.cann_ops_transformer.ds41.indexer_prologue_qw` |
| DSL 直调 | `ops.indexer_prologue_qw.indexer_prologue_qw` |

## 测试说明

从 ops-transformer 根目录执行。真机用例走已安装的整包 wheel，不再按源码路径加载 DSL。

```bash
# CPU golden，不需要 NPU
python3 -m pytest -q experimental/attention/indexer_prologue_qw_dsl/tests/test_golden.py

# 真机 T=72（需要已安装 net ops + cann_ops_transformer）
python3 -m pytest -q experimental/attention/indexer_prologue_qw_dsl/tests/test_npu_precision.py

# 默认：T=72 NPU precision
bash experimental/attention/indexer_prologue_qw_dsl/tests/test.sh
```

`q` 与 `descale_q` 必须逐位对齐；`w` 使用 `rtol=atol=2e-2`。
