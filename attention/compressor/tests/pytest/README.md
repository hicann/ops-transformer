# Compressor算子测试框架

## 功能说明

基于pytest测试框架，实现Compressor算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **双模式执行隔离**：支持直接pytest多进程执行和shell层进程隔离两种批量模式
- **性能采集**：支持挂载msprof采集算子性能数据并汇总输出

## 当前实现范围

### 参数限制

- 支持D为128/512。
- 支持H为1K~10K，512对齐。
- 支持cmp_ratio为2、4、8、16、32、64、128。
- 当cmp_ratio为4/128时，支持如下三种情况：
    - C4A: D=512, coff=2, cmp_ratio=4;
    - C4Li: D=128, coff=2, cmp_ratio=4;
    - C128A: D=512, coff=1, cmp_ratio=128。

### 环境配置

#### 前置要求

1、 确认TorchNPU为最新版本  

2、 确认已构建并安装compressor算子包与torch extension（二进制内不含compressor时）：

``` bash
# ① 在ops-transformer仓根目录构建compressor算子run包（--vendor_name可自定，避免覆盖已有vendors包）
# ${CANN_PATH}: CANNToolkit安装根路径(如 /usr/local/Ascend/ascend-toolkit)
# ${SOC}:       目标芯片(如 ascend910b)
source ${CANN_PATH}/set_env.sh
bash build.sh --pkg --soc=${SOC} --ops=compressor --vendor_name=cmpbit -j16
# 产物: build_out/custom_opp_*.run，安装到CANN vendors目录（交互确认即可）
bash build_out/custom_opp_*.run --install-path=${CANN_PATH}/latest

# ② 构建并安装torch extension wheel（整包或单算子）
bash build.sh --torch_extension --ops=compressor --vendor_name=cmpbit
pip3 install --force-reinstall --no-deps build_out/cann_ops_transformer_cmpbit-*.whl
```

3、 执行测试前需source CANN环境变量与算子vendor包环境（`${CANN_PATH}`为CANNToolkit安装根路径）：

``` bash
source ${CANN_PATH}/set_env.sh
# 还需source提供Compressor算子的vendor包环境脚本:
source ${CANN_PATH}/latest/vendors/<vendor>_transformer/bin/set_env.bash
```

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                                  # 执行脚本，支持single/bitlevel/batch三种命令（bitlevel只支持A2/A3）
- batch_isolated_run.sh                        # 批量隔离执行脚本（shell层进程隔离+msprof性能采集）
- compressor_golden.py                         # cpu侧算子golden实现以及精度对比
- collect_perf_data.py                         # msprof性能数据收集与汇总
- pytest.ini                                   # 创建测试标记

单用例测试：

- test_compressor_single.py                    # 测试单用例运行主程序
- compressor_operator_single.py                # CPU侧算子逻辑实现获取golden与npu算子直调
- test_compressor_paramset.py                  # 单用例入参配置

位级一致性测试：

- test_compressor_bitlevel.py                  # 位级一致性测试主程序（pair/position两类用例）
- compressor_bitlevel_golden.py                # 确定性数据构造 + NPU直调 + 位级对比支撑库
- test_compressor_bitlevel_paramset.py         # 位级一致性用例参数配置

批量测试：

- test_compressor_batch.py                     # 用例批量测试主程序并生成excel文件保存结果（支持隔离模式）
- ./batch/compressor_pt_loadprocess.py         # 读取pt文件并调用算子获取npu输出
- ./batch/compressor_pt_save.py                # 读取excel表格批量生成用例pt文件
- ./batch/replace_path.py                      # test_compressor_batch.py占位符替换

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_compressor_paramset.py的ENABLED_PARAMS参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### 位级一致性调测

1、手动配置test_compressor_bitlevel_paramset.py的BITLEVEL_TEST_PARAMS参数

2、执行指令：

``` bash
bash test_run.sh bitlevel
# 或直接
python3 -m pytest -rA -s test_compressor_bitlevel.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

3、环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CMP_TEST_DEVICE_ID` | `0` | 指定运行的NPU设备号 |
| `CMP_BITLEVEL_STRICT` | `1` | `1`：位级不一致即用例失败；`0`：降级为告警+阈值校验（用于区分位级尾差与功能错误） |

4、用例说明（详见test_compressor_bitlevel_paramset.py头部注释）：

| 用例 | 目的 |
|---|---|
| B=32/33、S=4 | tokenSize=128/132，FULL_LOAD → PERF 模板切换，跨模板位级一致性 |
| B=96/97、S=4 | 24 AIC核、coff=2、D=512下PERF动态D分块变化（dBaseSize 32↔64，hIdxStart改变） |
| B=31/32 | tokenSize=124/128，同为FULL_LOAD，同模板对照组 |
| 同B，目标样本换到首、中、尾位置 | batch位置一致性 |

注：`PerfDsplitCliff`组中的`dyn`用例会按当前设备AIC核数自动推导动态D分块悬崖边界对
（公式：`B_dyn = 128 * (AIC // (D/64)) / S`，24核设备上即为B=96/97，910B3=20核设备上为B=64/65），
与固定用例重复时自动去重。每个用例同时执行CPU golden精度校验与NPU结果位级对比。

5、基线结果：

**（问题记录，L0 行为）**：测量时算子尚未包含 batch 一致性修改（deterministic level =0），记录本修改所解决的问题现象。
环境：Atlas 910B3、20 AIC核、CANN 9.1.0-beta.1。

| 用例 | 位级结果 | 说明 |
|---|---|---|
| `FullLoadCtrl-pair_31_32` | 一致 | 同为FULL_LOAD模板，对照组通过 |
| `FullLoadToPerf-pair_32_33` | **不一致** | cmp_kv 7/16384元素差1个BF16 ULP；state更新约0.001%元素差FP32尾数（阈值内100%） |
| `PerfDsplitCliffDyn-pair_64_65` | **不一致** | 悬崖机制hIdxStart改变：cmp_kv 13/32768元素差1个BF16 ULP；state更新约1.3%元素差FP32尾数（阈值内100%） |
| `PerfDsplitCliff-pair_96_97` | 一致 | 20核下两侧均为dBaseSize=64，未跨悬崖 |
| `PositionConsistency-pos_B32/33/96/97` | 一致 | 目标样本首/中/尾位置结果位级一致 |

结论（修复前）：**跨模板（FULL_LOAD vs PERF）与PERF动态D分块悬崖（dBaseSize 32↔64导致hIdxStart改变）均存在位级不一致**，
差异均为1个BF16 ULP / FP32累加尾差级别（CPU golden精度校验全部Pass，无功能错误）；
batch位置一致性满足。

**修复后（level=3）实测**：`bash test_run.sh bitlevel` （deterministic level =3）默认配置（`CMP_BITLEVEL_STRICT=1`）。
环境：Atlas 910B3、20 AIC核、CANN 9.2.0，7/7 用例通过。

| 用例 | 位级结果 |
|---|---|
| `FullLoadCtrl-pair_31_32` | 一致 |
| `FullLoadToPerf-pair_32_33` | 一致 |
| `PerfDsplitCliff-pair_64_65` | 一致 |
| `PositionConsistency-pos_B32/33/96/97` | 一致 |

结论：**level=3 下跨模板与PERF动态D分块悬崖两侧均位级一致**；batch位置一致性保持。
另在泛化 batch 切分套件（13 组生产形态、388 个切分策略）上 level=3 全部位级一致。
使用`CMP_BITLEVEL_STRICT=0`可将位级失败降级为告警以区分尾差与功能错误（默认为1，即严格位级断言）。

#### 用例的批量生成与测试

##### 方式A：test_run.sh 批量执行

1、excel路径下存放用例excel表格

2、test_run.sh中设置读取的用例excel表格路径（PATH1）和pt文件存放路径（PATH2）

3、执行指令：

``` bash
bash test_run.sh batch
```

##### 方式B：手工分步执行

1、生成pt文件：

``` bash
python3 batch/compressor_pt_save.py "excel/*.xlsx" pt_path
```

2、替换测试脚本路径：

``` bash
python3 batch/replace_path.py test_compressor_batch.py pt_path
```

3、执行测试：

``` bash
python3 -m pytest -rA -s test_compressor_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4、恢复测试脚本：

``` bash
cp test_compressor_batch.py.bak test_compressor_batch.py
```

##### 方式C：批量隔离执行（推荐用于性能采集）

对每条用例单独拉起一个pytest进程，实现进程间完全隔离，避免单条用例崩溃影响其他用例。
用例获取逻辑：优先从 `./excel` 目录读取Excel文件的 `Testcase_Name` 列映射为 `.pt` 文件路径；若无Excel文件或缺少 `Testcase_Name` 列，则自动回退为扫描用例目录下所有 `.pt` 文件。

``` bash
bash batch_isolated_run.sh ./pt_path 0         # 不采集性能
bash batch_isolated_run.sh ./pt_path 1         # 采集性能（挂载msprof）
```

## Excel 用例表格式

`excel/test_cases.xlsx` 的 Sheet1 需包含以下列：

| 列名 | 类型 | 示例 | 说明 |
|---|---|---|---|
| Testcase_Name | str | `Prefill0` | 用例名，也作为生成的.pt文件名 |
| batch_size | int | `1` | |
| hidden_size | int | `4096` | |
| Seq_len | int | `8192` | |
| head_dim | int | `128` / `512` | |
| block_size | int | `128` | |
| cmp_ratio | int | `4` / `128` | |
| coff | int | `1` / `2` | |
| start_p | int | `0` / `8192` | |
| cache_mode | int | `1` / `2` | 1:连续buffer, 2:循环buffer |
| layout_x | str | `TH` / `BSH` | |
| data_type | str | `BF16` / `FP16` | |
| cu_seqlens | None/str | `None` 或 `"[0,8192]"` | TH布局必填 |
| seqused | None/str | `None` 或 `"[8192]"` | |
| start_pos | None/str | `None` 或 `"[0]"` | |
| x_datarange | str | `"[-10,10]"` | |
| wkv_datarange | str | `"[-10,10]"` | |
| wgate_datarange | str | `"[-10,10]"` | |
| ape_datarange | str | `"[-10,10]"` | |
| kv_state_datarange | str | `"[-10,10]"` | |
| score_state_datarange | str | `"[-10,10]"` | |

**注意事项**：

- `layout_x=TH` 时，`cu_seqlens` 为必填项（长度=batch_size+1的列表）

## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅msprof模式） |
| `batch_summary.log` | 批量隔离执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof性能原始数据目录 |
