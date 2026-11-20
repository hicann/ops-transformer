# quant_mla_prolog TTK

assets 是该算子的 TTK 端到端测试资源目录，包含用例、接口适配、输入准备、CPU Golden、精度比较和图执行支持。

使用 [ops-test-kit](https://gitcode.com/cann/ops-test-kit) 的 E2E 模式测试本仓库的 `quant_mla_prolog` 公开算子。支持 MXFP8 和 HiFloat8 权重量化，以及 `qkv_quant_mode` 为 0、1、2、3 的配置。
TTK 输出槽为 `q`、`descale_q`、`qr`、`descale_qr` 和 `kv_cache`，其中 `kv_cache` 由算子原位更新。

支持芯片：昇腾 950，构建目标为 `ascend950`。

## 被测源码定位

`quant_mla_ttk_ops.py` 根据自身所在目录定位并加载
`attention/quant_mla_prolog/op_dsl/quant_mla_prolog.py`，调用公开接口 `quant_mla_prolog`。
模块加载时注册 Torch 算子接口，同一进程复用已加载的模块。

当前测试入口直接加载仓库内的实现模块；运行时可根据环境配置匹配已安装的 Native 产物。
该入口不经过已安装 wheel 的 `torch_extension` 接口。

设置下文的 `ASSETS` 和 `PYTHONPATH` 后，可检查解析到的源码路径：

```bash
python3 -c "import quant_mla_ttk_ops as adapter; print(adapter._resolve_sample_file())"
```

## 目录结构

下图仅列出说明文档和实现文件，用例由同目录的 CSV 文件提供。

```text
attention/quant_mla_prolog/tests/assets/
├── quant_mla_ttk_ops.py  # 被测接口适配与源码加载
├── spec.py  # TTK 插件及钩子注册
├── impl/
│   ├── audit.py  # 计时前的契约检查与输入恢复
│   ├── audit_reference.py  # 契约检查参考实现
│   ├── compare.py  # 精度比较、接受判定与结果展示
│   ├── golden.py  # CPU Golden 的输入与输出适配
│   ├── golden_reference.py  # 普通与高精度参考实现的访问入口
│   ├── graph.py  # ACLGraph 执行入口
│   ├── inputs.py  # 输入生成后的数据准备
│   ├── npu_preprocess.py  # 计时与图捕获前的设备预处理
│   ├── pytest_reference.py  # 独立数值参考与接受规则
│   └── transport.py  # 数据载体与存储布局传输
└── README.md
```

## 环境准备

准备与设备及 CANN 配套的 Python、PyTorch、torch_npu 和 cannbotdsl 环境，
并确保 PyTorch 支持用例所需的浮点数据类型。使用虚拟环境时，先激活自行创建的环境。

以下路径均需替换为本机实际路径：

```bash
export CANN_ROOT="/path/to/cann"
export DSL_REPO="/path/to/ops-transformer"
export TTK_REPO="/path/to/ops-test-kit"
source "$CANN_ROOT/set_env.sh"

export ASSETS="$DSL_REPO/attention/quant_mla_prolog/tests/assets"
export PYTHONPATH="$ASSETS:$TTK_REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$TTK_REPO"
python3 -m pip install -r requirements.txt
python3 -m ttk info
```

用例使用 E8M0 和 HiFloat8 语义数据类型，需要安装提供相应 NumPy 数据类型的 `en-dtypes`：

```bash
python3 -m pip install "en-dtypes>=0.0.4"
```

可设置 `CANNBOTDSL_CACHE_DIR` 复用编译缓存：

```bash
export CANNBOTDSL_CACHE_DIR="$HOME/.cache/cannbotdsl_compile"
```

未命中已安装的 Native 产物或编译缓存时，运行可能包含编译耗时。

## 运行

以下命令在 ops-test-kit 仓库根目录执行，并沿用环境准备中的变量。
默认启用 CPU Golden 和插件精度比较；通常无需额外指定 `--compare`。

### 校验用例

仅检查 CSV 和接口声明，不执行设备计算：

```bash
python3 -m ttk e2e -i "$ASSETS/quant_mla_prolog.csv" \
    --plugin "$ASSETS" --validate
```

### eager模式

```bash
python3 -m ttk e2e -i "$ASSETS/quant_mla_prolog.csv" \
    --plugin "$ASSETS" --devices 0 --run 1 --warmup false \
    -o quant_mla_prolog_result.csv
```

### ACLGraph 模式

追加 `--aclgraph`，通过 `torch.compile` 的 `npugraph_ex` 后端执行
`impl/graph.py::QuantMlaPrologAclGraph`。默认同时执行 eager 和图分支，
两者分别与 CPU Golden 比较。

```bash
python3 -m ttk e2e -i "$ASSETS/quant_mla_prolog.csv" \
    --plugin "$ASSETS" --devices 0 --aclgraph --run 1 --warmup false \
    -o quant_mla_prolog_aclgraph_result.csv
```

仅执行图分支时，在上述命令中追加 `--eager=false`。

### 关闭 Golden

`--golden-mode Disable` 跳过 Golden 生成和精度比较，可用于观察设备执行情况。
成功执行不代表精度通过，精度判定显示为 `SKIP`。

```bash
python3 -m ttk e2e -i "$ASSETS/quant_mla_prolog.csv" \
    --plugin "$ASSETS" --devices 0 --golden-mode Disable --run 1 --warmup false \
    -o quant_mla_prolog_execution_result.csv
```

该选项也可与 `--aclgraph` 组合使用。

### 选卡

使用 `--devices` 指定 TTK 设备白名单，例如 `--devices 1` 使用设备 1。
`--device` 表示使用的设备数量，不能用来指定设备编号。

其他常用参数：

- `-t <用例名>`：筛选用例。
- `--seed 42`：固定随机输入种子。
- `--dump-on-fail`：失败时保存输入、输出和 Golden。

## 复用已保存的输入与 Golden

先生成并保存数据，再使用相同的 CSV 从文件加载，避免重复生成输入和计算 CPU Golden。
以下 `MANUAL_DATA` 应指向当前用例集专用的数据目录。

```bash
export MANUAL_DATA="$PWD/manual_data/quant_mla_prolog"

# 准备数据，不执行设备计算
python3 -m ttk e2e -i "$ASSETS/quant_mla_prolog.csv" \
    --plugin "$ASSETS" --no-prof --manual-data-dirs "$MANUAL_DATA" \
    --dump in,golden --dump-format pt

# 加载数据并执行设备计算
python3 -m ttk e2e -i "$ASSETS/quant_mla_prolog.csv" \
    --plugin "$ASSETS" --manual-data-dirs "$MANUAL_DATA" --devices 0 \
    -o quant_mla_prolog_manual_result.csv
```

加载数据时保持默认 `--golden-mode Enable`，不与 `--validate` 或 `--cpu` 同用。
数据准备只指定一个输出目录；加载时按 CSV 的 shape 和 dtype 校验文件。
加载已有数据时不再执行输入定制钩子，随机种子和造数参数不会改变已保存的输入。
用例输入定义或参考计算发生变化后，应使用新的数据目录重新生成。

## 精度判据

精度比较由 `impl/compare.py` 调用 `impl/pytest_reference.py` 中的
`compare_for_acceptance`，统一组织双方比较、三方比较及接受判定。
`descale_qr` 按权重量化模式使用 E8M0 原始码或 FP32 scale 判据。

日志中的“数值判定结果”记录名称、通过状态、是否参与最终判定及接受路线。
诊断项不通过不一定否决输出，参与最终判定的接受项必须通过。

| 输出 | 判据 |
|---|---|
| Q | qkv 模式 0、1 使用 BF16 混合误差判据；模式 2 使用 E4M3FN 码距；模式 3 使用 HiFloat8 量化前参考 |
| descale_q | FP32 scale 混合误差判据；不启用时验证空输出 |
| QR | 按权重量化模式使用 E4M3FN 或 HiFloat8 判据；关闭时验证空输出 |
| descale_qr | MXFP8 使用 E8M0 原始码，HiFloat8 使用 FP32 scale 判据 |
| kv_cache | 按有效索引比较更新区域；模式 1 分别检查 KV、KR 和 descale；未更新逻辑区域逐字节核对 |

形状和 dtype 必须满足输出契约。缺少比较上下文、比较异常或未验证的结果不能作为精度通过。
TTK 展示的匹配率与最终接受状态分别记录；双方匹配率不等于三方接受结果。
`--compare close` 不会替换插件内定义的接受规则。
