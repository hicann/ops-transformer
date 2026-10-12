# FlashAttnGrad

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | ------ |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：根据前向注意力计算的中间结果（softmax_lse、attn_out）和上游梯度（do），计算 Q/K/V 的梯度（dq/dk/dv）；提供 `sinks` 时同时计算 sink 的梯度 `dsink`。
- 计算公式：

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

当提供 `sinks` 时，前向 softmax 分母额外含 sink 项（`lse` 已包含该贡献），
其梯度按 batch / S1 / S2 归约：

$$
dSink=-\sum_{b,s1,s2} P \cdot dP \cdot \exp(sink-softmax\_lse)
$$

其中 `dP=dO \cdot V^T`。kernel 按 task（s2 块）在块内把 `P \cdot dP` 逐行乘
`\exp(sink-softmax\_lse)` 后归约，再归约成 `dsink`：每个 AIV 子核把本 task
的部分和以 fp32 `atomicAdd` 累加到 workspace 中自己独占的槽位行（一行一个
AIV，只由本 AIV 写，无跨核竞争；进主循环前该行已由 pre 阶段清零）；`sync_all`
之后由 AIV0 按固定行序（aiv 0..aivCount-1）归约成 `dsink`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|-----|-----------|----|---------|------|
| q | 输入 | Query，公式中的 Q | BF16、FP16 | ND |
| k | 输入 | Key，公式中的 K | BF16、FP16 | ND |
| v | 输入 | Value，公式中的 V | BF16、FP16 | ND |
| do | 输入 | 上游梯度，公式中的 dO | BF16、FP16 | ND |
| attn_out | 输入 | 前向注意力输出 | BF16、FP16 | ND |
| softmax_lse | 输入 | 前向 softmax LSE | FP32 | ND |
| cu_seqlens_q | 可选输入 | TND layout 累积序列长度 | INT32 | ND |
| cu_seqlens_kv | 可选输入 | TND layout 累积序列长度 | INT32 | ND |
| seqused_q | 可选输入 | 实际使用的 Q 序列长度 | INT32 | ND |
| seqused_kv | 可选输入 | 实际使用的 KV 序列长度 | INT32 | ND |
| sinks | 可选输入 | Sink tensor | FP32 | ND |
| attn_mask | 可选输入 | Attention mask，mask_mode=3/4 时使用 | INT8 | ND |
| metadata | 可选输入 | FAG metadata tensor | INT32 | ND |
| dq | 输出 | Query 梯度 | BF16、FP16 | ND |
| dk | 输出 | Key 梯度 | BF16、FP16 | ND |
| dv | 输出 | Value 梯度 | BF16、FP16 | ND |
| dsink | 输出 | Sink 梯度，shape 为 (N1,)。torch 接口与 dq、dk、dv 一起返回 | FP32 | ND |
| softmax_scale | 属性 | softmax 缩放因子，默认 0.0，表示 1/sqrt(d) | FLOAT | - |
| mask_mode | 属性 | 默认 0。0 全计算，3 causal，4 window | INT | - |
| win_left | 属性 | window mask 左窗口，默认 -1 | INT | - |
| win_right | 属性 | window mask 右窗口，默认 -1 | INT | - |
| max_seqlen_q | 属性 | Q 最大序列长度，默认 -1 | INT | - |
| max_seqlen_kv | 属性 | KV 最大序列长度，默认 -1 | INT | - |
| layout_q | 属性 | Q 布局，默认 BSND | STRING | - |
| layout_kv | 属性 | KV 布局，默认 BSND | STRING | - |
| layout_out | 属性 | 输出布局，默认 BSND | STRING | - |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| PyTorch API | [调用示例](docs/torchapi_flash_attn_grad.md#调用示例) | 通过 [flash_attn_grad](docs/torchapi_flash_attn_grad.md) 调用。先用 flash_attn_metadata 生成 metadata（is_grad_enabled=True），再调用本算子。编译安装见下文 Quick Start。 |

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

- **torch 接口**（`flash_attn_grad` 的函数原型、参数说明、返回值说明）：[torchapi_flash_attn_grad.md](../../attention/flash_attn_grad/docs/torchapi_flash_attn_grad.md)

调用分两步：先用 `flash_attn_metadata` 生成反向分核 metadata（必须设置 `is_grad_enabled=True`，并保证与主算子的 shape、layout、mask 和序列长度参数一致），再调用 `flash_attn_grad` 主算子。完整调用示例（含代码）见接口文档的[调用示例](../../attention/flash_attn_grad/docs/torchapi_flash_attn_grad.md#调用示例)章节。

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
| `torch_extension/csrc/flash_attn_grad.cpp` | PyTorch C++ 扩展 |
