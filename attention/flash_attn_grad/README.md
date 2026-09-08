# FlashAttnGrad

## 功能说明

Flash Attention 反向梯度计算算子。根据前向注意力计算的中间结果（softmax_lse、attn_out）和上游梯度（do），计算 Q/K/V 的梯度（dq/dk/dv）。

其中 $P = Softmax(scale \cdot QK^T - softmax\_lse)$，反向计算公式为：

$$
dV=P^TdO
$$

$$
dQ=scale \cdot (dS \cdot K)
$$

$$
dK=scale \cdot (dS^T \cdot Q)
$$

## 输入输出

### 输入

| 名称 | 类型 | 必选 | 说明 |
|------|------|------|------|
| q | BF16/FP16 | 是 | Query tensor |
| k | BF16/FP16 | 是 | Key tensor |
| v | BF16/FP16 | 是 | Value tensor |
| do | BF16/FP16 | 是 | 上游梯度 |
| attn_out | BF16/FP16 | 是 | 前向注意力输出 |
| softmax_lse | FP32 | 是 | 前向 softmax LSE |
| cu_seqlens_q | INT32 | 否 | TND layout 累积序列长度 |
| cu_seqlens_kv | INT32 | 否 | TND layout 累积序列长度 |
| seqused_q | INT32 | 否 | 实际使用的 Q 序列长度 |
| seqused_kv | INT32 | 否 | 实际使用的 KV 序列长度 |
| sinks | FP32 | 否 | Sink tensor |
| attn_mask | INT8 | 否 | Attention mask (mask_mode=3/4) |
| metadata | INT32 | 否 | FAG metadata tensor |

### 输出

| 名称 | 类型 | 说明 |
|------|------|------|
| dq | BF16/FP16 | Query 梯度 |
| dk | BF16/FP16 | Key 梯度 |
| dv | BF16/FP16 | Value 梯度 |

### 属性

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| softmax_scale | Float | 0.0 | softmax 缩放因子，0.0 表示 1/sqrt(d) |
| mask_mode | Int | 0 | 0:全计算, 3:causal, 4:window |
| win_left | Int | -1 | window mask 左窗口 |
| win_right | Int | -1 | window mask 右窗口 |
| max_seqlen_q | Int | -1 | Q 最大序列长度 |
| max_seqlen_kv | Int | -1 | KV 最大序列长度 |
| layout_q | String | "BSND" | Q 布局 |
| layout_kv | String | "BSND" | KV 布局 |
| layout_out | String | "BSND" | 输出布局 |

## Quick Start

### 1. custom 包编译与安装

完整脚本（整合编译与安装步骤；`CANN_DIR` 为 CANN 安装根目录，按实际调整）：
```bash
# 前置：加载 CANN 环境
source ${CANN_DIR}/cann/set_env.sh

# 清理历史构建产物（避免残留影响增量编译）
rm -rf ./build ./build_out
rm -rf ${CANN_DIR}/vendors

# 编译（flash_attn_grad 与 flash_attn_metadata 需同时编译，metadata 生成反向分核信息）
bash build.sh --pkg --soc=ascend950 --ops=flash_attn_grad,flash_attn_metadata

# 安装
cd build_out
./cann-ops-transformer-*.run --install-path=${CANN_DIR}
```

编译产物：`build_out/cann-ops-transformer-custom_linux-x86_64.run`

**可选参数说明**：
- **`-j`（限制并行线程数）**：默认按机器核数并行。当机器内存不足、或 cgroup 实际限制核数小于 `/proc/cpuinfo` 报告值导致编译 OOM 或失败时，需显式指定较小值：

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn_grad,flash_attn_metadata -j16
  ```

### 2. torch 扩展包构建与安装

使用 torch 接口前必做。在仓根目录构建 torch 扩展 whl 并安装：

```bash
# 前置：加载 CANN 环境
source ${CANN_DIR}/cann/set_env.sh

# 清理 torch 扩展缓存（~ 为当前用户 home，需与安装/运行 torch 的用户一致，避免加载过期编译产物）
rm -rf ~/.cache/torch_extensions/*

# 构建 torch 扩展 whl（全量包，包名 cann_ops_transformer 保持不变，whl 输出到 build_out/）
bash build.sh --torch_extension --soc=ascend950

# 安装
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

**安装后验证**：
```bash
python3 -c "from cann_ops_transformer.ops import flash_attn_grad; print('ok')"
```

### 3. 接口调用

- **torch 接口**（`flash_attn_grad` 的函数原型、参数说明、返回值说明）：[flash_attn_grad.md](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn_grad.md)

调用分两步：先用 `flash_attn_metadata` 生成反向分核 metadata（必须设置 `is_grad_enabled=True`，并保证与主算子的 shape、layout、mask 和序列长度参数一致），再调用 `flash_attn_grad` 主算子。完整调用示例（含代码）见接口文档的[调用示例](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn_grad.md#调用示例)章节。

导入路径与安装包名一致（按步骤 2 构建的全量包）：

```python
from cann_ops_transformer.ops import flash_attn_grad
```

## 算子目录结构

| 文件 | 说明 |
|------|------|
| `op_kernel/flash_attn_grad.py` | pypto-pro kernel 实现（BN2GS1S2 模板） |
| `op_host/flash_attn_grad_tiling.cpp` | tiling 实现（tilingkey 编码、workspace 布局） |
| `op_host/flash_attn_grad_def.cpp` | 算子定义（op_proto/输入输出/属性） |
| `op_host/flash_attn_grad_infershape.cpp` | shape/dtype 推导 |
| `op_host/config/ascend950/flash_attn_grad_binary.json` | 二进制算子配置 |
| `torch_extension/flash_attn_grad.py` | PyTorch 算子 schema 与 python 绑定 |
| `torch_extension/csrc/flash_attn_grad.cpp` | PyTorch C++ 扩展（aclnn 调用层） |

## aclnn 接口说明

本算子 aclnn 接口**不对外开放**：`aclnnInnerFlashAttnGradGetWorkspaceSize` / `aclnnInnerFlashAttnGrad` 以 inner 符号编入自定义包 `libcust_opapi.so`，仅供 `cann_ops_transformer` torch 扩展内部调用，不在 `op_api/include/aclnnop` 安装头文件中导出，也不提供 aclnn 调用示例。对外统一使用 torch 接口 `cann_ops_transformer.flash_attn_grad`。
