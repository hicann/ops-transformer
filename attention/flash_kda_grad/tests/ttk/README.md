# flash_kda_grad × TTK

本目录通过 TTK（ops-test-kit）的 E2E 通路调用 FlashKDA 反向算子，提供接口适配、
CPU 参考实现、验收配置、CSV 资产和运行脚本。单元测试、metadata/调度测试及 CPU
参考实现自检统一放在 [`../ut/test_flash_kda_grad_ut.py`](../ut/test_flash_kda_grad_ut.py)。
在仓库根目录运行：

```bash
python -m pytest -m 'not whitebox' attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py
python -m pytest attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py
```

本算子的 `tests/pytest.ini` 仅注册 marker，不增加收集钩子或默认过滤。直接使用 pytest 原生 marker 分组；`-m 'not whitebox'` 选择常规用例，`-m whitebox` 只选择白盒用例，不加 `-m`
则收集全部用例。需要组合条件时写在同一个 `-m` 表达式中。

需要只运行 CPU 测试时添加 `-m 'not npu and not ascendc_toolchain'`；仅检查 TTK
参考实现可添加 `-k test_ttk_reference`。仍需安装该测试模块依赖的 PyTorch 和
CANNBotDSL；本文件同时包含 NPU 与工具链测试，按原有 marker 筛选。

CPU whitebox 的前端与 lowering 校验共用一次编译，保留 8 个代表用例，覆盖
保存缓存/重算两种模式下的 masked、full、mixed、staged 路径，以及 beta 输入/
梯度的三种 dtype 配对和两种 gate dtype；不再逐一编译它们的全部交叉组合。
全部 72 个导出变体的参数及 ABI 仍由轻量测试枚举检查；完整 AOT 编译检查仍由
带 `ascendc_toolchain` 标记的测试承担。

原 `attention/flash_kda_grad/tests/test_flash_kda_grad.py` 的参考公式、缓存适配和精度检查
已迁入上述 UT 模块，原文件已删除。独立 CLI 入口也随之迁移，例如：

```bash
python attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py --profile --help
python attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py --compare-bin --help
```

TTK 的 `api.py`、`spec.py` 仍使用本目录 `flash_kda_grad_ttk/golden.py`，
不导入 UT 模块。测试日志、profile 结果和 bin 对比报告应写到代码仓库之外。

## 1. 文件与入口

```text
attention/flash_kda_grad/tests/ttk/
├── flash_kda_grad_ttk/
│   ├── api.py                      # npu(...)：NPU 算子 / CPU 参考分派
│   ├── golden.py                   # 前向缓存、逐 token golden、分阶段反向 peer
│   ├── spec.py                     # golden / peer 与 TTK 默认 cross_check L2
│   └── __init__.py                 # 导出 npu
├── cases/
│   ├── flash_kda_grad_stc.csv      # 130 条 STC，含 30 条新增重算专项
│   └── flash_kda_grad_exception.csv # 异常用例，单独维护
└── run_one.sh                      # 唯一启动入口，按需生成临时 peer 配置
```

被测入口为 `flash_kda_grad_ttk.npu`，接受 PyTorch 张量，因此使用 E2E 通路。
NPU 分支只加载相邻 `op_kernel` 中的算子，导入失败直接报错，避免误测其他已安装版本。
CPU 分支调用参考实现。

## 2. 输入、输出与缓存

[`api.py`](flash_kda_grad_ttk/api.py) 的输入顺序如下，BSND 布局交换 S/N 轴，TND 使用 `[T,N,D]`：

| 顺序 | 张量 | shape（BNSD） | dtype |
|---|---|---|---|
| 1–2 | `q`, `k` | `[B,Nq,S,128]` | BF16 |
| 3 | `v` | `[B,Nv,S,128]` | BF16 |
| 4 | `g` | `[B,Nv,S,128]` | BF16 / FP32 |
| 5 | `beta` | `[B,Nv,S]` | BF16 / FP32 |
| 6 | `A_log` | `[Nv]` | FP32 |
| 7 | `dt_bias` | `[Nv,128]` | FP32 |
| 8 | `initial_state` | `[B,Nv,128,128]` | FP32 |
| 9 | `do` | `[B,Nv,S,128]` | BF16 |
| 10 | `dht` | `[B,Nv,128,128]` | FP32 |

属性包括 `scale`、`gate_mode`（bounded/softplus）、`lower_bound`、
`allow_neg_eigval`、`layout`（BNSD/BSND/TND）、`cache_source`（ref/fwd）及三个
`use_*_in_kernel` 开关。

输出顺序固定为 `dq, dk, dv, dh0, dg, dbeta, dA, dBias`。
其中 `dh0/dA/dBias` 为 FP32，其余梯度遵循对应输入的 dtype。

默认 `cache_source=ref`：适配器在 CPU 上构造前向缓存，上传到 NPU，再生成 AICPU
metadata 并调用反向算子。peer 使用相同的缓存构造规则与 BF16 舍入点。
`cache_source=fwd` 则调用真实前向算子生成缓存，用于检查前向与反向的完整链路；
此时需设置 `KDA_DSL_ROOT` 指向含兼容 `samples/flash_kda_fwd` 的 CANNBotDSL 仓库，
结果也包含前向实现的影响。默认 130 条 STC 均使用 `cache_source=ref`，不依赖该外部前向。

通过 `disable_recompute` 属性逐用例选择模式：`False` 启用重算，`True` 消费已保存
的缓存。显式属性优先于环境变量；仅省略或传 `None` 时回退到导入适配器时读取的
`KDA_GRAD_RECOMPUTE`，未设置该环境变量时默认使用保存缓存。

STC 共 130 条，统一放在 `flash_kda_grad_stc.csv`，编号连续为 001–130，
沿用已有 TTK 格式；异常用例仍单独放在 `flash_kda_grad_exception.csv`。

- 原有 96 条的参数、数据范围和精度设置不变。
- `kdag_stc_022/065/066/067` 以边界用例替换原先 CPU autograd 开销过大的配置，
  保留编号，参数见下表。
- 新增 `kdag_stc_101` 至 `kdag_stc_130`，全部
  按重算模式设计。覆盖三种布局各 10 条、BF16/FP32 gate 和 beta 组合，以及
  63/64/65、127/128/129、1025 长度、空序列、变长、GVA、N=128、初始状态、
  可选 bias 和全部八种融合开关组合。按最新统一重算实现更新 `remark`，记录输入
  dtype、整块/尾块数量及尾块须走 staged 的约束；这些字段用于说明覆盖范围，
  不执行运行时调度断言。无尾块时仍由 metadata 选择调度，不固定设备核数。

| 编号 | 布局 | 长度及物理存储 | Nq / Nv | 模式 | 有效 token-head |
|---|---|---|---|---|---|
| 022 | TND | B=31，`[0,1,63,64,65]` 重复六组，再加 129；总 T=1287 | 1 / 2 | 保存缓存 | 2574 |
| 065 | BNSD | B=2，物理 S=64，有效长度 `[17,0]` | 127 / 127 | 保存缓存 | 2159 |
| 066 | BSND | B=1，物理 S=33，有效长度 `[31]` | 1 / 128 | 重算 | 3968 |
| 067 | TND | B=3，长度 `[511,512,513]`；总 T=1536 | 1 / 1 | 重算 | 1536 |

四条替换用例的输入已改变，必须重新生成 golden，不能复用这些编号对应的旧缓存。

新增和替换用例保持 `D=128`、`Chunk=64`、`1 <= Nq <= Nv <= 128`，且满足
`B*S <= 1M`（TND 按总 token 数计算）。为控制逐 token autograd 成本，每条新增
重算专项的 `有效 token 数 * Nv` 不超过 2860，四条替换用例不超过 3968；
不替代独立的大规模性能或容量测试。

重算用例通过 `attributes.disable_recompute=False` 直接控制 NPU 重算；
当前 `api.py:npu()` 已接入并透传该参数，无须另外设置环境变量。
CPU golden 和 peer 仍使用原有实现及 cross_check L2，不能用放宽门限来使新路径通过。

激活环境后，从任意目录执行；默认读取全部 130 条 STC：

```bash
bash "$KDA_ASSETS/run_one.sh"
bash "$KDA_ASSETS/run_one.sh" -t kdag_stc_109
```

每条用例按 CSV 的 `disable_recompute` 属性选择保存缓存或重算，显式属性不会被
全局环境变量覆盖。不同模式的结果和 manual-data 缓存应可区分，避免混淆。
仅运行某一条时在命令末尾添加 `-t kdag_stc_022` 等用例名；不加筛选则运行全部 130 条。

### 重算异常契约

`flash_kda_grad_exception.csv` 单独维护 60 条异常目录，原 001–050 不变；
051–060 新增重算模式的初始状态类型、末维、序列数、头数、device 错误，以及
缺失 Aqk、Akk、q_rtsd、beta 激活缓存和错误 k_rstd dtype。全部显式设置
`disable_recompute=False`，预期在重算缓存分配和主计算前抛出 `ValueError`。
`initial_state=None` 是合法零初态；可重算的七项缓存可省略，均不作为异常。

普通 TTK E2E 不解释 `remark.mutation`，且 Aqk/Akk 等缓存由适配器内部构造。
异常目录的实际变异和预期异常断言由现有 CSV 契约测试执行，不能把普通 E2E 的
运行失败直接当成异常拦截成功。从算子仓库根目录运行 CPU 契约测试：

```bash
python -m pytest -q -s -m 'not npu' \
  attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py \
  -k 'exception_csv_host_contract or exception_csv_special_contract'
```

在已选择目标 NPU 的进程中，改为筛选 `exception_csv_npu_contract` 可验证真实 NPU
Tensor 的 Host 拦截；测试设有分配/计算哨兵，非法输入进入重算分配或计算准备即失败。
这部分验证参数契约，不修改 golden、数值门限、SDK 或 TTK 框架。

## 3. Golden、peer 与判据

当前 [`spec.py`](flash_kda_grad_ttk/spec.py) 使用以下实现：

| 角色 | 实现 | 计算方式 |
|---|---|---|
| golden | `golden.token_golden_backward` | FP32 逐 token 递推并由 autograd 求导，不模拟 BF16 中间存储；返回时由 spec 转换输出 dtype |
| peer | `golden.bf16_peer` | BF16 前向缓存及矩阵乘操作数，按 chunk 公式显式反传 |
| 被测 | `flash_kda_grad_ttk.npu` | NPU 上调用实际反向算子，CPU 上调用分阶段参考实现 |

`golden.py` 中也保留了 `emulated_autograd_backward` 等参考接口，但它们不是
当前 spec 的 golden 入口。CPU 被测分支与 peer 共用分阶段反向实现，因此 CPU 运行
可用于检查接入链路，不能据此得出 NPU 精度结论。

所有输出统一采用 TTK 框架内置的 `cross_check L2`，仅声明标准名和级别：

```python
tolerance = {
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}
```

门限和小值域参数均由 TTK 的 L2 预设提供，项目不覆盖这些参数。
`run_one.sh` 的默认运行、`l2` 和 `replay` 均使用该配置，不覆盖精度门限。

`cross_check` 比较被测实现和 peer 各自相对 golden 的误差，需关注
`mare_ratio`、`mere_ratio`、`rmse_ratio`。L2 使用比值门限 2.0 / 1.2 / 1.2。
运行时通过 `--plugin` 加载 `spec.py`，不传 `--compare` 覆盖其判据。

## 4. 环境准备

使用当前机器已经激活的 Conda/CANN 环境，Python 中需安装 PyTorch、torch_npu、
CANNBotDSL 0.8 及 TTK 依赖。脚本不切换 Conda、不 source 固定的 CANN 路径。

```bash
export KDA_ROOT=/path/to/ops-transformer/attention/flash_kda_grad
export KDA_ASSETS="$KDA_ROOT/tests/ttk"
# 使用 TTK 源码时设置；当前 Python 已安装 ttk 时可以省略。
export TTK_REPO=/path/to/ops-test-kit
export KDA_DEVICES=0
export KDA_MANUAL_DATA=/path/to/your/input_and_golden
export KDA_OUT=/path/to/your/results.csv
```

`KDA_ROOT` / `KDA_ASSETS` 默认从脚本位置推导；Python 默认取当前环境的 `python3`，
可通过 `KDA_PY` 或兼容变量 `TTK_PY` 覆盖。未设置 `TTK_REPO` 时先查找仓库同级的
`ops-test-kit`，否则使用环境中已安装的 TTK。worker 和 CPU peer 均继承资产目录的
`PYTHONPATH`。结果和 manual data 建议放在自建目录，避免写入代码目录。

## 5. 唯一运行入口

```bash
bash "$KDA_ASSETS/run_one.sh" --help
bash "$KDA_ASSETS/run_one.sh" validate       # CSV / 接口校验，不执行算子
bash "$KDA_ASSETS/run_one.sh" list           # 列出用例
bash "$KDA_ASSETS/run_one.sh" -t kdag_stc_109 # 默认 cross_check L2
bash "$KDA_ASSETS/run_one.sh" exec -t kdag_stc_109   # 执行但不比较 golden
bash "$KDA_ASSETS/run_one.sh" dump -t kdag_stc_109   # 在新目录生成输入/golden
bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109 # 读取已有输入/golden
KDA_PROF=true bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109
bash "$KDA_ASSETS/run_one.sh" xpu            # 单独启动 CPU peer
```

`run` 与 `l2` 是默认模式的别名。`-t` 等额外参数透传给 TTK；`KDA_CASE` 也可筛选
单个用例。`dump` 只准备数据，不能作为 NPU 精度通过的依据；已有缓存使用 `replay`。
四条替换用例必须使用与当前 CSV 一致的新数据，不能套用旧定义的 golden。

| 变量 | 默认值及用途 |
|---|---|
| `KDA_CASES` | `cases/flash_kda_grad_stc.csv` |
| `KDA_OUT` | 调用目录的 `kda_result.csv` |
| `KDA_MANUAL_DATA` | 调用目录的 `kda_manual_data`，用于 dump/replay |
| `KDA_CASE` | 用例筛选，等价于 `-t` |
| `KDA_DEVICES` | `0`，传给 TTK 客户端的 `--device-whitelist` |
| `KDA_SEED` | `20261027` |
| `KDA_PC` / `KDA_RUN` | 每卡进程数 / 执行次数，均为 `1` |
| `KDA_PROF` | `false`；设为 `true` 开启性能采集 |
| `KDA_XPU_PORT` | `9090`，本机 CPU peer 端口 |
| `KDA_XPU_LOG` | `$TMPDIR/kda_xpu_<端口>.log`，未设 TMPDIR 时使用 `/tmp` |

默认运行、`l2`、`replay` 会检查 peer 端口，未启动时自动启动 CPU 服务。服务会保留，
脚本不会重启或终止已有服务；切换代码目录时，使用独立的 `KDA_XPU_PORT`，避免复用
另一份代码的 peer。客户端临时 YAML 在脚本结束时清理，不再维护固定配置文件。

需要一个连续批跑日志时，可以在调用处保存完整输出：

```bash
set -o pipefail
bash "$KDA_ASSETS/run_one.sh" replay 2>&1 | tee /path/to/your/batch.log
```

远端 peer 可通过 TTK 原生命令和自建配置使用，例如已有 `peer.yaml` 时：

```bash
python3 -m ttk e2e -i "$KDA_ASSETS/cases/flash_kda_grad_stc.csv" \
  --plugin "$KDA_ASSETS/flash_kda_grad_ttk/spec.py" --config /path/to/peer.yaml \
  --device-whitelist "$KDA_DEVICES" --seed 20261027 --pc 1 \
  --proc-no-reuse --proc-timeout 3000 --warmup=false --run 1 \
  --task-prof false -o "$KDA_OUT"
```

## 6. 结果解读与排障

结果 CSV 使用 `-o` 或 `KDA_OUT` 指定。终端和精度明细中的八个输出位置依次为
`dq, dk, dv, dh0, dg, dbeta, dA, dBias`。常用字段：

[`spec.py`](flash_kda_grad_ttk/spec.py) 的 `pre_compare` 钩子会在终端和 `ttk.log`
打印八个梯度的数值表：首尾各 10 个元素、绝对差异最大的 10 个元素，以及最大有限
绝对误差和非有限值数量。`Index` 是展平后的元素下标，
`AbsDiff = abs(NpuOut - GoldenOut)`，`RateDiff = AbsDiff / abs(GoldenOut)`；golden
为零时，两者相等显示 0，否则显示 inf（NaN 保持显示 NaN）。空输出或形状不一致会
单独标注。表格仅供定位问题，不给逐元素误差另设 PASS/FAIL；最终结论仍由原生
`cross_check L2` 三方判据给出，golden、三方参考和阈值均不改变。
原生比较明细仍在 `ttk-compare.log`。设置 `KDA_COMPARE_DETAILS=0` 可关闭额外打印和扫描。

改 `api.py` 的签名时记得：**不要加 `from __future__ import annotations`**——TTK 的
`_annotation_to_type` 只认真实类型对象，字符串注解会全部退化成按参数名猜类型
（`beta` 会被猜成 `float`，`gate_mode` 会被猜成 `Tensor`）。

### TND 与有效序列长度

TTK adapter、golden 和 CPU peer 接受相同的可选 Tensor 参数，位于 `dht` 后：
`cu_seqlens=None, seqused=None`。原有十个 Tensor 的调用保持兼容。

- `layout="TND"`：传连续 int32 `[B+1]` 的 `cu_seqlens`，首项 0、末项 T，允许相邻项相等（空序列）；不传 `seqused`。
- `layout="BNSD"` / `"BSND"`：可传连续 int32 `[B]` 的 `seqused`，每项位于 `[0,S]`；省略表示全部使用 S 个 token。不传 `cu_seqlens`。
- NPU 路径的长度 Tensor 必须与 q 同设备。golden/peer 使用同一组长度数值；padding 区输入梯度为零。
- `initial_state` / `dht` 的 batch 维表示逻辑序列数 B，TND 的 B 不能由 q 的 token 维推断。

NPU 入口保持请求的实际 layout；参考缓存按有效长度生成，并映射到当前算子的 padded chunk 轴或 TND 预留 chunk 槽位。`cache_source="fwd"` 也将长度传给前向 metadata。
用 CSV 运行变长用例时，长度 Tensor 需要通过 TTK 自定义输入或已有输入文件提供精确值，不能将随机 int32 数据当作合法的前缀和。
