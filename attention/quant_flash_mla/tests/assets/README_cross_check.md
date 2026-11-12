# QuantFlashMla：三方 cross_check 比对

FP8 全量化下，CPU golden 与 kernel 的 P 量化路径不同（kernel 按分核出的
s2 分段独立取 running max 并量化 P）。直接用"NPU 对 golden"的两方判据，会把
golden 自身的量化噪声当成误差，产生假失败。

三方比对以一份高精度 CPU golden 为共同基准，判据是"NPU 相对基准的误差 /
XPU 参考相对基准的误差"，即 NPU 的误差不超过 XPU 参考噪声的若干倍。

| 角色 | 来源 | TTK 参数位 |
| --- | --- | --- |
| target | 被测 quant_flash_mla 算子输出 | `output` |
| bench | `--golden-mode promote` 的 CPU golden（fp32 输出） | `golden`（共同基准） |
| third_party | XPU（原生精度、不升档的**分段量化** golden） | `third_party` |

## 调用链

```text
customize_inputs → inputs.generate_qfmla_inputs()   固定种子生成算子输入
        │
        ├─ 算子执行                     → outputs = [atten(bf16), lse(fp32)?]
        │
golden  → impl/golden.cpu_qfmla()     → CPU 参考，按 TTK golden_mode 决定是否升档
        │
xpu     → spec._xpu_reference()       → 远端 xpu_server 调用分段量化 golden
        │                             （固定 256 列分块，原生精度不升档）
        │
compare → spec.compare()
             ├─ --compare cross_check：返回 None，由框架原生 CrossCheckComparison 三方比对
             └─ 默认两方：impl/compare.compare() 按 CSV 精度阈值逐输出判定
```

`spec.third_party = {"torch": _xpu_reference}` 声明 XPU 参考实现（函数体内
懒加载 `impl/golden.py`，避免主进程提前拉起 torch_npu）；远端执行时 spec 文件
与 `impl/` 目录由 TTK 同步到 xpu_server。

third_party 必须是**分段量化**参考（`fp8_fullquant_mla_sectioned_golden`，固定
256 列分块 running max + P 量化），不能复用全局精确 golden：cross_check 的分母
是 `|b-g|`，全局参考与升档 golden 同源时该值趋零，比值恒爆。只有带同等 P 量化
噪声的分段参考才能提供有效的噪声地板。

## 生成与回放

### 0. 约定

| 名称 | 取值 |
|------|------|
| ASSETS | `ops-transformer/attention/quant_flash_mla/tests/assets` |
| RUN | `qfmla_run/`（工作目录，保存生成和回放产物） |
| XPU 端口 | `127.0.0.1:9100` |

`--plugin` 传的是 **ASSETS 目录**（内含 `spec.py`），不是 .py 文件。

### 1. 环境准备（每个新终端都要做）

在两个终端中，先激活已安装 TTK 和 PyTorch 扩展的 Python 环境，并进入 `ops-transformer` 的上一级目录。将 `CANN_INSTALL_DIR` 设为本机 CANN 安装目录，将 `QFMLA_OPP_DIR` 设为已安装的 QFMLA 自定义算子包的 vendor 目录，然后执行：

```bash
source "${CANN_INSTALL_DIR}/set_env.sh"
export ASCEND_CUSTOM_OPP_PATH="${QFMLA_OPP_DIR}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

ASSETS="$PWD/ops-transformer/attention/quant_flash_mla/tests/assets"
RUN="$PWD/qfmla_run"
mkdir -p "$RUN"
```

### 2. 终端 A：启动 XPU 三方服务（常驻，不要退出）

```bash
python3 -m ttk.remote.server.xpu_server --port 9100
```

校验：

```bash
ss -lntp | grep 9100      # 应看到 LISTEN 127.0.0.1:9100
```

说明：xpu 不是另一块加速卡，而是本机 CPU 上跑分段量化参考（`spec._xpu_reference` →
`cpu_qfmla(_force_global_reference=True, _sectioned_reference=True)`），TTK 通过该端口取数。

### 3. 配置文件

`$RUN/ttk_remote_demo.yaml`：

```yaml
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9100
```

不要只传 ttk 包内的 `default.yaml`：其中的 `endpoints` 是注释状态，只传它不会连三方。
`--config` 会与 `default.yaml` 做深合并，所以这份最小 yaml 等价于「默认配置 + 三方端点」。

### 4. 生成用例 CSV

```bash
cd "$ASSETS/../pytest"
python3 pytest_paramset_to_csv.py --paramset qfmla_paramset_func_rdv \
  --output "$RUN/qfmla_func_rdv_dyn.csv"
```

注意两个前提（转换器已保证，改脚本时别破坏）：

- metadata 槽位写 `(-1, -1)`：动态槽位，由 `npu_preprocess` 在主算子调用前生成并回填。
  写死 `(2, N)` 会被当成普通输入随机填充，kernel 读到非法调度表。
- TND 用例的 `cu_seqlens_q` 必须是 `seqused_q` 的 packed 前缀和（`cu[-1] == sum(seqused_q)`），
  按 padded 步长生成会让 golden 直接拒算。
- 稠密用例（BSND/BNSD）反过来：q 行数必须是 `B * max_seqlen_q`（padded），cu 也用 padded 步长。
  若这里按 packed 行数生成，q 与 cu 末端对不上，golden 会报 `Golden generation failure`。

### 5. 终端 B：生成阶段（input + 升档 golden + XPU 一次产出）

```bash
cd "$RUN"
ttk e2e -i qfmla_func_rdv_dyn.csv -o fp8_xpu_gen.csv \
  --plugin "$ASSETS" \
  --config "$RUN/ttk_remote_demo.yaml" \
  --no-prof --golden-mode Promote --dump in,golden,xpu \
  --manual-data-dirs "$RUN/manual_fp8"
```

想拆开跑也行：

- 只要 CPU 升档真值：`--golden-mode Promote --dump in,golden`
- 只要 XPU 分段参考（原生精度，不升档）：`--dump in,xpu`
- 只跑某个用例：加 `--ti=0`（0-based 用例索引，不是设备号）

成功后每个 case 目录下的文件：

| 文件 | 含义 |
|------|------|
| `input_0..8_*.bin` | 9 个静态输入 |
| `input_9_none.bin` | 动态 metadata（0 字节，由 preprocess 现场生成） |
| `golden_0_float32_*.bin` | 升档 CPU golden（FP32，三方真值） |
| `golden_1_*.bin` | LSE golden，LSE 关闭时为 0 字节 |
| `xpu_torch_0_bfloat16_*.bin` | XPU 分段参考主输出（按 bf16 落盘） |
| `xpu_torch_1_*.bin` | XPU 分段参考 LSE |

### 6. 终端 B：回放阶段（跑 NPU 并做三方比对）

```bash
cd "$RUN"
ttk e2e -i qfmla_func_rdv_dyn.csv -o fp8_cross.csv \
  --plugin "$ASSETS" \
  --manual-data-dirs "$RUN/manual_fp8" \
  --compare cross_check
```

注意：**回放不能带 `--no-prof`**，与 `--manual-data-dirs` 冲突，会报
`--no-prof manual-data output requires --dump in`。

### 7. 看结果

- `precision_status`：整体判定 PASS / FAIL
- `precision_metrics`：每个输出的 `standard` / `level` / `config` / `result` / `pass`
- `xpu_metrics`：三方数据来源（应为 `manual_xpu_binaries` / PASS）

判定档位为 **L2**（在 `spec.py` 的 `tolerance` 里显式声明；TTK 代码默认是 L1）：

```
mare_ratio <= 2.0, mere_ratio <= 1.2, rmse_ratio <= 1.2
```

## 判定方法

走框架 `CrossCheckComparison`，逐输出通道处理（atten bf16 + lse fp32）：

1. 三方统一升精度，形状必须一致；
2. 大值域 `|bench| >= small_value`：
   - `mare = max|target-bench| / max|third-bench|`
   - `mere = mean|target-bench| / mean|third-bench|`
   - `rmse = sqrt(mean((target-bench)^2)) / sqrt(mean((third-bench)^2))`
3. 小值域 `|bench| < small_value`：统计 `|x-bench| > small_value_atol` 的点数，
   `count_target / count_third <= small_value_ratio`；
4. NaN/Inf 走框架内置的三方一致性判定；
5. 除法分母保护、分子分母下限由 TTK 内部实现。

## 阈值

标准 `cross_check`、等级 **L2**、`mare/mere/rmse = 2.0/1.2/1.2`（在
`spec.py` 的 `tolerance` 中以 `"level": "L2"` 声明）：

| 输出 dtype | small_value | small_value_atol | small_value_ratio |
| --- | --- | --- | --- |
| float16 | `2^-10` | `2^-16` | 2 |
| bfloat16 | `2^-10` | `2^-16` | 2 |
| float32（lse） | `2^-14` | `2^-30` | 2 |

默认两方比较仍走 `impl/compare.py` 的 CSV 阈值，不受 `level` 影响。
