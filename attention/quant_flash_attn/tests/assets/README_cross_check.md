# QuantFlashAttn：三方 cross_check 比对

量化 attention 的 kernel 会在每个分核/分段内独立做 online softmax 并对 P 做 FP8
量化。CPU golden 如果按全局 running max 计算，其 P 量化噪声与 NPU 不同源，
两方判据会把这种固有噪声当成误差，产生假失败。

三方比对用同一份高精度 CPU golden 做共同基准，判据是「NPU 相对基准的误差」
与「XPU 分段参考相对基准的误差」的比值，即 NPU 的误差不得超过竞品噪声上界的
若干倍。

| 角色 | 来源 | TTK 参数位 |
| --- | --- | --- |
| target | 被测 `quant_flash_attn` 算子输出 | `output` |
| bench | `--golden-mode Promote` 的 CPU golden（FP32 输出） | `golden`（共同基准） |
| third_party | XPU 分段量化参考（原生精度、不升档，主输出 cast bf16） | `third_party` |

## 调用链

```text
customize_inputs → impl/inputs.py                    固定种子生成算子输入
        │
        ├─ 算子执行                     → outputs = [atten(bf16), lse(fp32)?]
        │
golden  → spec._golden_dispatch()     → impl/golden.cpu_qfa_*()
        │                             （golden_mode=Promote 时角色切到 bench）
        │
xpu     → spec._xpu_reference()       → impl/golden.qfa_sectioned_reference()
        │                             （固定分段量化，原生精度不升档）
        │
compare → spec.compare（_CompareHook）
             ├─ --compare cross_check：解析为 None，由框架 CrossCheckComparison 三方比对
             └─ 默认两方：impl/compare.compare() 按 CSV 精度阈值逐输出判定
```

`spec.third_party = {"torch": _xpu_reference}` 声明 XPU 参考实现。函数体内懒加载
`impl/golden.py`，避免主进程提前拉起 torch_npu；远端执行时 spec 文件与 `impl/`
目录由 TTK 同步到 xpu_server。

third_party 必须是**分段量化**参考，不能复用全局精确 golden：cross_check 的分母
是 `|third_party - bench|`，两者同源时该值趋零，比值恒爆。只有带同等 P 量化噪声
的分段参考才能提供有效的噪声地板。

## 参考实现映射

`impl/golden.py` 通过 `_resolve_reference_mode()` 选择角色：

- 显式 `_reference`（third_party 入口注入 `sectioned`）；
- 否则 `--golden-mode Promote` 或 `--compare cross_check` → `bench`；
- 其余情况 → `golden`（两方语义）。

| quant_mode | 模式 | golden（两方） | bench（三方真值） | sectioned（XPU third_party） |
| --- | --- | --- | --- | --- |
| 1 / 2 | MXFP8 | `cpu_mxfp8_golden` | `cpu_mxfp8_bench` | `cpu_mxfp8_sectioned_golden` |
| 0 | HIF8 | `cpu_hif8_golden`（FD 场景走 `cpu_hif8_scheduled_golden`） | `cpu_hif8_bench(sectioned=False)` | `cpu_hif8_sectioned_golden` |
| 6 | GQA FP8 全量化 | `cpu_fp8_fullquant_golden` | `cpu_fp8_fullquant_bench(sectioned=False)` | `cpu_fp8_fullquant_bench(sectioned=True)` |
| 10 / 12 | HIFP4 / INT8 | 支持两方 | 不支持三方 | 不支持三方（`NotImplementedError`） |

## 生成与回放

### 0. 约定

| 名称 | 取值 |
| --- | --- |
| ASSETS | `ops-transformer/attention/quant_flash_attn/tests/assets` |
| QFA_PYTEST | `ops-transformer/attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test` |
| RUN | `qfa_run/`（工作目录，保存生成和回放产物） |
| XPU 端口 | `127.0.0.1:9100` |

`--plugin` 传的是 **ASSETS 目录**（内含 `spec.py`），不是 .py 文件。

### 1. 环境准备（每个新终端都要做）

激活已安装 TTK 和 PyTorch 扩展的 Python 环境，并进入 `ops-transformer` 的上一级目录。
将 `CANN_INSTALL_DIR` 设为本机 CANN 安装目录，将 `QFA_OPP_DIR` 设为已安装的
QuantFlashAttn 自定义算子包的 vendor 目录：

```bash
source "${CANN_INSTALL_DIR}/set_env.sh"
export ASCEND_CUSTOM_OPP_PATH="${QFA_OPP_DIR}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

ASSETS="$PWD/ops-transformer/attention/quant_flash_attn/tests/assets"
QFA_PYTEST="$PWD/ops-transformer/attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test"
RUN="$PWD/qfa_run"
mkdir -p "$RUN"
```

如果环境里 `ninja` / `c++` 不在 PATH，需要先补齐（CPU 参考的 JIT 扩展编译依赖）：

```bash
export PATH="<conda-env>/bin:$PATH"
```

### 2. 终端 A：启动 XPU 三方服务（常驻，不要退出）

```bash
python3 -m ttk.remote.server.xpu_server --port 9100
```

校验：

```bash
ss -lntp | grep 9100      # 应看到 LISTEN 127.0.0.1:9100
```

xpu 不是另一块加速卡，而是本机 CPU 上跑分段量化参考
（`spec._xpu_reference` → `qfa_sectioned_reference`），TTK 通过该端口取数。

### 3. 配置文件

`$RUN/ttk_remote_demo.yaml`：

```yaml
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9100
```

`--config` 会与 TTK 包内 `default.yaml` 深合并，这份最小 yaml 等价于
「默认配置 + 三方端点」。不要只传 `default.yaml`：其中 `endpoints` 是注释状态，
不会连三方。

### 4. 生成用例 CSV

```bash
cd "$QFA_PYTEST"
python3 pytest_paramset_to_csv.py \
  --paramset quant_flash_attn_paramset_debug \
  --output "$RUN/qfa_mxfp8.csv"
```

转换器保证两个前提（改脚本时别破坏）：

- metadata 槽位写 `(-1, -1)`：动态槽位，由 `impl/npu_preprocess.py` 在主算子调用前
  生成并回填；
- TND 用例的 `cu_seqlens_q` 是 `seqused_q` 的 packed 前缀和（`cu[-1] == sum(seqused_q)`），
  `batch_size` 属性留 `None`（`npu_preprocess` 对 varlen TND/NTD 强制 `batch_size=None`，
  否则 metadata 调用会报错）。

已有库存 CSV（含 redline / profiling 转换用例）也可以直接作为 `-i` 输入。

### 5. 终端 B：生成阶段（input + 升档 golden + XPU 一次产出）

```bash
cd "$RUN"
ttk e2e -i qfa_mxfp8.csv -o mxfp8_gen.csv \
  --plugin "$ASSETS" \
  --config "$RUN/ttk_remote_demo.yaml" \
  --no-prof --golden-mode Promote --dump in,golden,xpu \
  --manual-data-dirs "$RUN/manual_mxfp8"
```

想拆开跑也行：

- 只要 CPU 升档真值：`--golden-mode Promote --dump in,golden`
- 只要 XPU 分段参考（原生精度，不升档）：`--dump in,xpu`
- 只跑某个用例：加 `--ti=0`（0-based 用例索引，不是设备号）

成功后每个 case 目录下的文件：

| 文件 | 含义 |
| --- | --- |
| `input_0..17_*.bin` | 算子 18 个输入槽；`*_none.bin` 为可选/动态槽（metadata、attn_mask、v_tail 等），未使用时为 0 字节 |
| `golden_0_float32_*.bin` | 升档 CPU golden 主输出（FP32，三方真值） |
| `golden_1_float32_*.bin` | LSE golden；LSE 关闭时该文件不存在（golden 只有 1 个输出） |
| `xpu_torch_0_bfloat16_*.bin` | XPU 分段参考主输出（bf16） |
| `xpu_torch_1_float32_shape_0.bin` | LSE 关闭时 third_party 的空占位（0 字节）；开启时是 fp32 LSE |

### 6. 终端 B：回放阶段（跑 NPU 并做三方比对）

```bash
cd "$RUN"
ttk e2e -i qfa_mxfp8.csv -o mxfp8_cross.csv \
  --plugin "$ASSETS" \
  --manual-data-dirs "$RUN/manual_mxfp8" \
  --compare cross_check
```

注意：**回放不能带 `--no-prof`**，与 `--manual-data-dirs` 冲突，会报
`--no-prof manual-data output requires --dump in`。

### 7. 看结果

- `precision_status`：整体判定 PASS / FAIL
- `precision_metrics`：每个输出的 `standard` / `level` / `config` / `result` / `pass`
- `xpu_metrics`：三方数据来源（应为 `manual_binary` / `manual_xpu_binaries` / PASS）
- `*_compare_detail.csv`：逐输出明细（`testcase_name, run_mode, output_idx, output_dtype,
  standard, level, pass, mare, mere, rmse, small_err_cnt_target, small_err_cnt_third, ...`）

## 判定方法

走框架 `CrossCheckComparison`，逐输出通道处理（atten bf16 + lse fp32）：

1. 三方统一升精度，形状必须一致；
2. 大值域 `|bench| >= small_value`：
   - `mare = max|target-bench| / max|third-bench|`
   - `mere = mean|target-bench| / mean|third-bench|`
   - `rmse = sqrt(mean((target-bench)^2)) / sqrt(mean((third-bench)^2))`
3. 小值域 `|bench| < small_value`：统计 `|x-bench| > small_value_atol` 的点数，
   `count_target / max(count_third, 1) <= 2`；
4. NaN/Inf 走框架内置的三方一致性判定；
5. 除法分母保护、分子分母下限由 TTK 内部实现。

## 阈值

标准 `cross_check`、等级 **L2**（在 `spec.py` 的 `tolerance` 中以 `"level": "L2"`
显式声明；TTK 未声明时默认 L1）：

```
mare_ratio <= 2.0, mere_ratio <= 1.2, rmse_ratio <= 1.2
```

small 域阈值按 **输出 dtype** 查 TTK 默认表（`ttk/core_modules/common/comparison/resolve.py`）：

| 输出 dtype | small_value | small_value_atol | small 域判定 |
| --- | --- | --- | --- |
| float16 | `2^-11` | `2^-16` | `err_target / max(err_third, 1) <= 2` |
| bfloat16 | `2^-8` | `2^-16` | 同上 |
| float32（LSE） | `2^-14` | `2^-30` | 同上 |

`spec.py` 中的 `tolerance` 还保留两方比对的 `rtol/ptol/atol`；默认两方比较仍走
`impl/compare.py`，不受 `level` 影响。

## 注意事项

- **生成必须带 `--golden-mode Promote`**：既用于选中 bench 角色，也把 golden 输出
  升到 FP32。Promote 会把 e8m0 输入抬成 float32，`impl/golden.py` 的 `_to_torch_e8m0`
  按 `round(log2(v)) + 127` 还原 e8m0 编码（NaN 保留 0xFF）；改生成流程时不要绕过它。
- **回放不要带 `--no-prof`**：见第 6 节。
- **TND 的 `batch_size`**：varlen TND/NTD 必须为 `None`，由 `npu_preprocess.py` 统一处理；
  不要在 CSV 里给 TND 写死 batch_size。
- **LSE 关闭时**：算子只返回 1 个输出；third_party 入口会补空 tensor 槽位，保证
  `--dump xpu` 与 manual 回放的槽位数一致。
- **dtype 对齐**：bench/golden 保持 FP32；third_party 主输出 cast bf16 后再比对；
  LSE 保持 FP32。不要给 third_party 做升档，否则噪声地板消失。
- **quant_mode 10/12**（HIFP4 / INT8）目前不支持三方，`qfa_sectioned_reference` 会直接
  抛 `NotImplementedError`。
